import os
import gc
import json
import shutil
import argparse
import hashlib

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from galore_torch import GaLoreAdamW8bit
from safetensors.torch import load_file
from accelerate import dispatch_model


def setup_distributed(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training environment"""
    dist.destroy_process_group()


def parse_config(config_file: str) -> dict:
    """Parse the JSON config file and return the configuration dictionary."""
    with open(config_file, 'r', encoding='utf-8') as f:
        config_str = f.read().replace("\\", "/")
        return json.loads(config_str)


def save_training_state(checkpoint_dir: str, step: int, epoch: int,
                       optimizer_state: dict, scheduler_state: dict, scaler_state: dict) -> None:
    """Save training progress and states."""
    serializable_optimizer_state = {}
    for key, value in optimizer_state.items():
        if key == 'state':
            serializable_optimizer_state[key] = {}
            for param_id, param_state in value.items():
                serializable_optimizer_state[key][param_id] = {}
                for state_key, state_value in param_state.items():
                    if (hasattr(state_value, '__class__') and 
                        state_value.__class__.__name__ == 'GaLoreProjector'):
                        continue
                    if torch.is_tensor(state_value):
                        serializable_optimizer_state[key][param_id][state_key] = state_value.cpu().tolist()
                    else:
                        serializable_optimizer_state[key][param_id][state_key] = state_value
        else:
            serializable_optimizer_state[key] = value

    serializable_scheduler_state = {}
    for key, value in scheduler_state.items():
        if torch.is_tensor(value):
            serializable_scheduler_state[key] = value.cpu().tolist()
        else:
            serializable_scheduler_state[key] = value

    state = {
        'step': step,
        'epoch': epoch,
        'optimizer_state': serializable_optimizer_state,
        'scheduler_state': serializable_scheduler_state,
        'scaler_state': scaler_state
    }
    
    with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w', encoding='utf-8') as f:
        json.dump(state, f)


def load_training_state(checkpoint_dir: str, optimizer, scheduler, scaler) -> dict:
    """Load training progress and states."""
    state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if not os.path.exists(state_path):
        return None
        
    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)
        
    optimizer_state = state['optimizer_state']
    for param_id, param_state in optimizer_state['state'].items():
        for state_key, state_value in param_state.items():
            if isinstance(state_value, list):
                param_state[state_key] = torch.tensor(state_value)
                
    scheduler_state = state['scheduler_state']
    for key, value in scheduler_state.items():
        if isinstance(value, list):
            scheduler_state[key] = torch.tensor(value)
            
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    scaler.load_state_dict(state['scaler_state'])
    
    return state


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_optimized(model_path, bnb_config, rank, config):
    """Load model with optimized settings"""
    attn_implementation = "flash_attention_2" if config.get("use_flash_attention_2", False) else "eager"
    safetensors_path = os.path.join(model_path, "model.safetensors")
    use_safetensors = os.path.exists(safetensors_path)
    
    if use_safetensors:
        print(f"[Rank {rank}] Using safetensors for parallel weight initialization")
        model_config = AutoConfig.from_pretrained(model_path)
        
        if bnb_config is not None:
            model = AutoModelForCausalLM.from_config(
                model_config,
                quantization_config=bnb_config,
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_config(
                model_config,
                torch_dtype=torch.float16
            )
        
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        
        weight_map = load_file(safetensors_path, load_tensors_in_parallel=True)
        for weight_name, weight in weight_map.items():
            param_names = weight_name.split('.')
            curr_param = model
            for name in param_names:
                if name.isdigit():
                    curr_param = curr_param[int(name)]
                else:
                    try:
                        curr_param = getattr(curr_param, name)
                    except AttributeError:
                        break
            else:
                with torch.no_grad():
                    if curr_param.shape == weight.shape:
                        curr_param.copy_(weight)
        
        model = model.to(f"cuda:{rank}")
    else:
        print(f"[Rank {rank}] Using optimized standard loading")
        with autocast(enabled=True, dtype=torch.float16):
            if bnb_config is not None:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map={"": rank},
                    offload_folder="offload_folder",
                    offload_state_dict=True,
                    attn_implementation=attn_implementation
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map={"": rank},
                    offload_folder="offload_folder",
                    offload_state_dict=True,
                    attn_implementation=attn_implementation
                )
    
    return model


def preprocess_function(example: dict, tokenizer, max_seq_length: int, prompt_template: str = None) -> dict:
    """Preprocess function to format and tokenize examples."""
    if prompt_template is None:
        prompt_template = (
            "<|im_start|>system\n\n{instruction}<|im_end|>\n"
            "<|im_start|>user\n\n{input}<|im_end|>\n"
            "<|im_start|>assistant\n\n{output}<|im_end|>"
        )

    formatted_text = prompt_template.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", "")
    )

    tokenized = tokenizer(
        formatted_text,
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors=None
    )

    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized


def train(rank, world_size, config):
    """Main training function for each process."""
    setup_distributed(rank, world_size)
    
    # Ensure directories exist
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["cache_dir"], exist_ok=True)
    os.makedirs("offload_folder", exist_ok=True)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    tokenizer.pad_token = tokenizer.eos_token

    clear_gpu_memory()

    # QLoRA setup
    if config.get("use_qlora", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        if config.get("use_bitsandbytes", False):
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None

    # Load model
    model = load_model_optimized(config["model_path"], bnb_config, rank, config)

    # QLoRA Integration
    if config.get("use_qlora", False):
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    # Disable cache to fix gradient checkpointing error
    if hasattr(model, "config"):
        model.config.use_cache = False
    elif hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = False

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Dataset preparation
    num_proc = config.get("num_workers_dataset", 12)
    batch_size_processing = config.get("batch_size_processing", 32)
    
    dataset = load_dataset(
        "json",
        data_files=config["dataset_path"],
        split="train",
        cache_dir=config["cache_dir"],
        num_proc=num_proc
    )
    
    dataset_hash = hashlib.md5(config["dataset_path"].encode()).hexdigest()
    cache_file_name = os.path.join(config["cache_dir"], f"processed_dataset_{dataset_hash}.arrow")
    
    tokenized_dataset = dataset.map(
        lambda ex: preprocess_function(ex, tokenizer, config["max_seq_length"], config.get("prompt_template")),
        batched=True,
        batch_size=batch_size_processing,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        cache_file_name=cache_file_name,
        load_from_cache_file=True
    )

    # Setup distributed sampler and dataloader
    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config["batch_size"] // world_size,  # Adjust batch size per GPU
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=data_collator,
        persistent_workers=True if config["num_workers"] > 0 else False
    )

    # Optimizer setup
    target_modules_list = ["attn", "mlp"]
    galore_params = []
    model_to_iterate = model.module

    for module_name, module in model_to_iterate.named_modules():
        if isinstance(module, nn.Linear) and any(key in module_name for key in target_modules_list):
            module.weight.data = module.weight.data.to(torch.float16)
            galore_params.append(module.weight)

    id_galore_params = [id(p) for p in galore_params]
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
    for param in regular_params:
        if param.requires_grad:
            param.data = param.data.to(torch.float16)

    param_groups = [
        {'params': regular_params},
        {
            'params': galore_params,
            'rank': config["rank"],
            'update_proj_gap': config["update_proj_gap"],
            'scale': config["scale"],
            'proj_type': config["proj_type"]
        }
    ]

    # Initialize optimizer, scheduler, and gradient scaler
    optimizer = GaLoreAdamW8bit(param_groups, lr=config["learning_rate"])
    total_training_steps = len(train_dataloader)
    first_cycle_steps = int(total_training_steps * config["first_cycle_fraction"])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=first_cycle_steps,
        T_mult=config["t_mult"],
        eta_min=config["eta_min"]
    )
    scaler = GradScaler()

    start_step = 0
    start_epoch = 0
    model.train()
    total_steps = len(train_dataloader)
    prev_avg_loss = 0.0
    accumulation_steps = config["accumulation_steps"]

    for epoch in range(start_epoch, config["num_epochs"]):
        sampler.set_epoch(epoch)  # Set epoch for distributed sampler
        running_loss = 0.0
        optimizer.zero_grad()
        
        if rank == 0:  # Only create progress bar on main process
            progress_bar = tqdm(enumerate(train_dataloader), total=total_steps, initial=start_step)
        else:
            progress_bar = enumerate(train_dataloader)
            
        for step, batch in progress_bar:
            if step < start_step:
                continue

            with autocast(enabled=True):
                inputs = {
                    k: v.to(rank, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                if 'attention_mask' not in inputs:
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                outputs = model(**inputs)
                loss = outputs.loss / accumulation_steps

            # Scale loss and backward pass
            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (step + 1) % accumulation_steps == 0:
                # Unscale gradients for gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_grad_norm"])
                
                # Optimizer step with scaling
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Gather loss from all processes
                loss_tensor = torch.tensor(running_loss, device=f'cuda:{rank}')
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                running_loss = loss_tensor.item() / world_size

                if rank == 0:  # Only update progress bar on main process
                    current_lr = scheduler.get_last_lr()[0]
                    current_loss = running_loss
                    avg_loss = current_loss if step == 0 else (current_loss * 0.1 + prev_avg_loss * 0.9)
                    prev_avg_loss = avg_loss
                    progress_bar.set_postfix({
                        'epoch': epoch + 1,
                        'loss': f'{current_loss:.4f}',
                        'avg_loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'step': f'{step}/{total_steps}'
                    })
                running_loss = 0.0

            # Save checkpoints only on rank 0
            if rank == 0 and step > 0 and step % config["save_interval"] == 0:
                checkpoint_path = os.path.join(config['checkpoint_dir'], f"checkpoint-{step}")
                model_to_save = model.module
                model_to_save.save_pretrained(checkpoint_path)
                save_training_state(
                    checkpoint_path,
                    step,
                    epoch,
                    optimizer.state_dict(),
                    scheduler.state_dict(),
                    scaler.state_dict()
                )
                # Cleanup old checkpoints
                checkpoints = sorted([
                    d for d in os.listdir(config["checkpoint_dir"])
                    if d.startswith('checkpoint-')
                ])
                while len(checkpoints) > config["keep_last_checkpoints"]:
                    oldest_checkpoint = checkpoints.pop(0)
                    shutil.rmtree(os.path.join(config["checkpoint_dir"], oldest_checkpoint))

            if step % 100 == 0:
                clear_gpu_memory()

        if rank == 0:
            progress_bar.close()

    # Save final model only on rank 0
    if rank == 0:
        model_to_save = model.module
        model_to_save.save_pretrained(config["final_output_path"])
        tokenizer.save_pretrained(config["final_output_path"])
        save_training_state(
            config["final_output_path"],
            total_training_steps,
            epoch,
            optimizer.state_dict(),
            scheduler.state_dict(),
            scaler.state_dict()
        )

    cleanup_distributed()


def main():
    """Main function to setup distributed training"""
    parser = argparse.ArgumentParser(
        description="Train a model with DDP and GaLore optimization."
    )
    parser.add_argument('--config_file', type=str, required=True,
                      help='Path to the JSON configuration file.')
    args = parser.parse_args()

    # Load configuration
    config = parse_config(args.config_file)
    
    # Get world size from config
    world_size = config.get("num_gpus", 1)
    
    # Launch distributed processes
    torch.multiprocessing.spawn(
        train,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
