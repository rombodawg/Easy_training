import os
import gc
import json
import shutil
import argparse
import hashlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR
from torch.cuda.amp import autocast, GradScaler

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
from accelerate import infer_auto_device_map, dispatch_model, Accelerator


def parse_config(config_file: str) -> dict:
    """
    Parse the JSON config file and return the configuration dictionary.
    Automatically replaces backslashes with forward slashes for path consistency.
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config_str = f.read().replace("\\", "/")
        return json.loads(config_str)


def save_training_state(checkpoint_dir: str, step: int, epoch: int,
                          optimizer_state: dict, scheduler_state: dict, accelerator: Accelerator) -> None:
    """
    Save training progress, optimizer state, and scheduler state to a JSON file.
    """
    # Gather the training state from all processes
    state = {
        'step': accelerator.gather(torch.tensor(step)).cpu().tolist(),
        'epoch': accelerator.gather(torch.tensor(epoch)).cpu().tolist(),
        'optimizer_state': accelerator.gather(optimizer_state),  # Use passed optimizer_state
        'scheduler_state': accelerator.gather(scheduler_state)   # Use passed scheduler_state
    }

    # Only save on the main process
    if accelerator.is_main_process:
        serializable_optimizer_state = {}
        for key, value in state['optimizer_state'][0].items():  # Take state from the first gathered optimizer state
            if key == 'state':
                serializable_optimizer_state[key] = {}
                for param_id, param_state in value.items():
                    serializable_optimizer_state[key][param_id] = {}
                    for state_key, state_value in param_state.items():
                        # Skip GaLoreProjector references
                        if (
                            hasattr(state_value, '__class__') and
                            state_value.__class__.__name__ == 'GaLoreProjector'
                        ):
                            continue
                        if torch.is_tensor(state_value):
                            serializable_optimizer_state[key][param_id][state_key] = state_value.cpu().tolist()
                        else:
                            serializable_optimizer_state[key][param_id][state_key] = state_value
            else:
                serializable_optimizer_state[key] = value

        serializable_scheduler_state = {}
        for key, value in state['scheduler_state'][0].items():  # Take state from the first gathered scheduler state
            if torch.is_tensor(value):
                serializable_scheduler_state[key] = value.cpu().tolist()
            else:
                serializable_scheduler_state[key] = value

        final_state = {
            'step': state['step'][0],  # Take step from the first gathered value
            'epoch': state['epoch'][0],  # Take epoch from the first gathered value
            'optimizer_state': serializable_optimizer_state,
            'scheduler_state': serializable_scheduler_state
        }

        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w', encoding='utf-8') as f:
            json.dump(final_state, f)


def load_training_state(checkpoint_dir: str, optimizer, scheduler, accelerator: Accelerator) -> dict:
    """
    Load training progress, optimizer state, and scheduler state from a JSON file.
    Reconstructs the state for both optimizer and scheduler.
    """
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

    return state


def clear_gpu_memory() -> None:
    """
    Clear GPU memory and run garbage collection to prevent out-of-memory errors.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_optimized(model_path: str,
                         bnb_config: BitsAndBytesConfig,
                         device_map,
                         config: dict):
    """
    Load model with optimized settings.
    """
    attn_implementation = (
        "flash_attention_2"
        if config.get("use_flash_attention_2", False)
        else "eager"
    )

    print("Using standard from_pretrained to load weights (safetensors if available).")

    if bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            offload_folder="offload_folder",
            offload_state_dict=True,
            attn_implementation=attn_implementation
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            offload_folder="offload_folder",
            offload_state_dict=True,
            attn_implementation=attn_implementation
        )

    return model


def preprocess_function(example: dict, tokenizer, max_seq_length: int,
                        prompt_template: str = None) -> dict:
    """
    Format and tokenize each example from the dataset.
    """
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


def add_special_tokens(tokenizer, special_tokens_str: str):
    """
    Add special tokens to the tokenizer.

    Args:
        tokenizer: The tokenizer to modify
        special_tokens_str: Comma-separated string of special tokens to add

    Returns:
        Modified tokenizer with special tokens added
    """
    if not special_tokens_str:
        return tokenizer

    # Parse special tokens from string
    special_tokens = [token.strip() for token in special_tokens_str.split(',')]

    # Add special tokens to tokenizer
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Added {num_added} special tokens to the tokenizer: {special_tokens}")

    # Verify special tokens were added
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"Token '{token}' has ID: {token_id}")

    return tokenizer


def main() -> None:
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(
        description="Train a model with multi-GPU support and 4-bit QLoRA."
    )
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the JSON configuration file.')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to the checkpoint directory to resume training from.')
    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Load config and create directories
    config = parse_config(args.config_file)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["cache_dir"], exist_ok=True)
    os.makedirs("offload_folder", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens if specified in config
    if "added_tokens" in config and config["added_tokens"]:
        tokenizer = add_special_tokens(tokenizer, config["added_tokens"])

    num_gpus = config.get("num_gpus", 1)
    if num_gpus > 8:
        num_gpus = 8

    if torch.cuda.is_available():
        device_map = {"": accelerator.local_process_index}  # Use accelerator's local process index
    else:
        device_map = "cpu"

    clear_gpu_memory()

    # BitsAndBytes configuration
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
    model = load_model_optimized(
        config["model_path"],
        bnb_config,
        device_map,
        config
    )

    # Resize token embeddings to account for any new special tokens
    if "added_tokens" in config and config["added_tokens"]:
        print(f"Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # If using QLoRA, prepare for kbit training and enable LoRA
    if config.get("use_qlora", False):
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    prompt_template = config.get("prompt_template", None)

    # Prepare dataset
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
    # Add special tokens to the cache file name to ensure proper regeneration when tokens change
    special_tokens_hash = ""
    if "added_tokens" in config and config["added_tokens"]:
        special_tokens_hash = hashlib.md5(config["added_tokens"].encode()).hexdigest()[:8]

    cache_file_name = os.path.join(
        config["cache_dir"], f"processed_dataset_{dataset_hash}_{special_tokens_hash}.arrow"
    )

    tokenized_dataset = dataset.map(
        lambda ex: preprocess_function(
            ex, tokenizer, config["max_seq_length"], prompt_template
        ),
        batched=True,
        batch_size=batch_size_processing,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        cache_file_name=cache_file_name,
        load_from_cache_file=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=data_collator,
        persistent_workers=True if config["num_workers"] > 0 else False
    )

    # Separate parameters for GaLore
    target_modules_list = ["attn", "mlp"]
    galore_params = []
    model_to_iterate = model.module if hasattr(model, "module") else model

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

    optimizer = GaLoreAdamW8bit(
        param_groups, lr=config["learning_rate"]
    )

    total_training_steps = len(train_dataloader)
    first_cycle_steps = int(total_training_steps * config["first_cycle_fraction"])

    # Learning rate warmup (if specified in config)
    warmup_steps = config.get("warmup_steps", 0)
    if warmup_steps > 0:
        # Use a very small start_factor (e.g., 1e-8) instead of 0.0
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=first_cycle_steps,
            T_mult=config["t_mult"],
            eta_min=config["eta_min"]
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=first_cycle_steps,
            T_mult=config["t_mult"],
            eta_min=config["eta_min"]
        )

    # Set starting step/epoch based on whether we're resuming from a checkpoint.
    start_step = 0
    start_epoch = 0
    if args.resume_checkpoint is not None:
        state = load_training_state(args.resume_checkpoint, optimizer, scheduler, accelerator)
        if state is not None:
            start_step = state['step']
            start_epoch = state['epoch']
            print(f"Resuming training from epoch {start_epoch}, step {start_step}")

    # Prepare model, optimizer, dataloader, and scheduler for distributed training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    model.train()
    total_steps = len(train_dataloader)
    prev_avg_loss = 0.0
    accumulation_steps = config["accumulation_steps"]

    scaler = GradScaler()

    for epoch in range(start_epoch, config["num_epochs"]):
        running_loss = 0.0
        optimizer.zero_grad()
        progress_bar = tqdm(enumerate(train_dataloader), total=total_steps, initial=start_step, disable=not accelerator.is_main_process)  # Disable tqdm on non-main processes

        for step, batch in progress_bar:
            if step < start_step:
                continue

            inputs = {
                k: v.view(-1, v.size(-1)) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

            with autocast():
                outputs = model(**inputs)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (step + 1) % accumulation_steps == 0:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None and p.grad.dtype == torch.float16:
                            p.grad.data = p.grad.data.float()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config["max_grad_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                current_lr = scheduler.get_last_lr()[0]
                current_loss = running_loss
                avg_loss = current_loss if step == 0 else (current_loss * 0.1 + prev_avg_loss * 0.9)
                prev_avg_loss = avg_loss

                # Compute epoch progress as: current epoch index + (current step/total steps)
                epoch_progress = epoch + (step / total_steps)
                if accelerator.is_main_process:  # Only print progress bar on main process
                    progress_bar.set_postfix({
                        'epoch': f'{epoch_progress:.2f}',
                        'loss': f'{current_loss:.5f}',
                        'avg_loss': f'{avg_loss:.5f}',
                        'lr': f'{current_lr:.2e}',
                        'step': f'{step}/{total_steps}'
                    })
                running_loss = 0.0

            if step > 0 and step % config["save_interval"] == 0:
                checkpoint_path = os.path.join(
                    config['checkpoint_dir'], f"checkpoint-{step}"
                )
                # Save model, tokenizer and training state using accelerator.save and accelerator.wait_for_everyone
                accelerator.wait_for_everyone()  # Ensure all processes are at the same point before saving
                if accelerator.is_main_process:  # Only save on main process
                    model_to_save = accelerator.unwrap_model(model)  # Unwrap the distributed model
                    model_to_save.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    save_training_state(
                        checkpoint_path,
                        step,
                        epoch,
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                        accelerator
                    )

                    # Remove older checkpoints if necessary
                    checkpoints = sorted([
                        d for d in os.listdir(config["checkpoint_dir"])
                        if d.startswith('checkpoint-')
                    ])
                    while len(checkpoints) > config["keep_last_checkpoints"]:
                        oldest_checkpoint = checkpoints.pop(0)
                        shutil.rmtree(os.path.join(config["checkpoint_dir"], oldest_checkpoint))
                accelerator.wait_for_everyone()  # Wait for main process to finish saving

            if step % 100 == 0:
                clear_gpu_memory()

        progress_bar.close()

    accelerator.wait_for_everyone()  # Ensure all processes are finished before final save
    if accelerator.is_main_process:  # Only save on main process
        model_to_save = accelerator.unwrap_model(model)
        model_to_save.save_pretrained(config["final_output_path"])
        tokenizer.save_pretrained(config["final_output_path"])
        save_training_state(
            config["final_output_path"],
            total_training_steps,
            epoch,
            optimizer.state_dict(),
            scheduler.state_dict(),
            accelerator
        )
    accelerator.end_training()  # Finalize accelerator processes


if __name__ == "__main__":
    main()
