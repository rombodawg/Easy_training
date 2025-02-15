import os
import gc
import json
import shutil
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from galore_torch import GaLoreAdamW8bit


def parse_config(config_file: str) -> dict:
    """
    Parse the JSON config file and return the configuration dictionary.
    Automatically replaces backslashes with forward slashes.
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config_str = f.read().replace("\\", "/")
        return json.loads(config_str)


def save_training_state(checkpoint_dir: str, step: int, epoch: int,
                          optimizer_state: dict, scheduler_state: dict) -> None:
    """
    Save training progress, optimizer state, and scheduler state to a JSON file.
    """
    state = {
        'step': step,
        'epoch': epoch,
        'optimizer_state': optimizer_state,
        'scheduler_state': scheduler_state
    }
    with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w', encoding='utf-8') as f:
        json.dump(state, f)


def load_training_state(checkpoint_dir: str) -> dict:
    """
    Load training progress, optimizer state, and scheduler state from a JSON file.
    """
    state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if os.path.exists(state_path):
        with open(state_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def preprocess_function(example: dict, tokenizer, max_seq_length: int, prompt_template: str = None) -> dict:
    """
    Preprocess function to format and tokenize each example from the dataset.
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


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(
        description="Train a model with multi-GPU support and 4-bit QLoRA."
    )
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the JSON configuration file.')
    args = parser.parse_args()

    # Load configuration
    config = parse_config(args.config_file)

    # Ensure directories exist
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["cache_dir"], exist_ok=True)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    tokenizer.pad_token = tokenizer.eos_token

    # Determine device mapping based on CUDA availability and number of GPUs
    num_gpus = config.get("num_gpus", 1)
    if torch.cuda.is_available():
        if num_gpus > 1:
            # Use Hugging Face device mapping to split model layers across GPUs.
            device_map = "auto"
        else:
            device_map = {"": 0}
    else:
        device_map = "cpu"

    # QLoRA integration: load in 4-bit mode if requested.
    if config.get("use_qlora", False):
        # Build BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        # Fallback: if not using QLoRA and bitsandbytes, load normally
        if config.get("use_bitsandbytes", False):
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None

    # Load model with quantization config if available
    if bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map
        )

    # ---------------------- QLoRA Integration ----------------------
    if config.get("use_qlora", False):
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        # Prepare the model for k-bit training (freezing most weights)
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    # ---------------------------------------------------------------

    # Note: When using device_map="auto" the model is already distributed across GPUs.
    # Do not wrap the model with DataParallel in that case.
    if num_gpus > 1 and device_map != "auto":
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))

    # Setup prompt template from config (if provided)
    prompt_template = config.get("prompt_template", None)

    # Load and preprocess dataset
    dataset = load_dataset(
        "json",
        data_files=config["dataset_path"],
        split="train",
        cache_dir=config["cache_dir"]
    )
    tokenized_dataset = dataset.map(
        lambda ex: preprocess_function(ex, tokenizer, config["max_seq_length"], prompt_template),
        batched=True,
        batch_size=4,
        num_proc=config["num_workers"],
        remove_columns=dataset.column_names,
        cache_file_name=os.path.join(config["cache_dir"], "processed_dataset.arrow")
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=data_collator
    )

    # Separate parameters for GaLore optimization (targeting "attn" and "mlp" modules)
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

    # Instantiate optimizer and scheduler
    optimizer = GaLoreAdamW8bit(param_groups, lr=config["learning_rate"])
    total_training_steps = len(train_dataloader)
    first_cycle_steps = int(total_training_steps * config["first_cycle_fraction"])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=first_cycle_steps,
        T_mult=config["t_mult"],
        eta_min=config["eta_min"]
    )

    start_step = 0
    start_epoch = 0
    model.train()
    total_steps = len(train_dataloader)
    prev_avg_loss = 0.0
    accumulation_steps = config["accumulation_steps"]

    for epoch in range(start_epoch, config["num_epochs"]):
        running_loss = 0.0
        optimizer.zero_grad()
        progress_bar = tqdm(enumerate(train_dataloader), total=total_steps, initial=start_step)
        for step, batch in progress_bar:
            if step < start_step:
                continue

            # For inputs, we move tensors to the primary device.
            # With device_map="auto", the model is spread out already; we choose cuda:0 as primary.
            primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inputs = {
                k: v.view(-1, v.size(-1)).to(primary_device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            outputs = model(**inputs)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
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

            if step > 0 and step % config["save_interval"] == 0:
                checkpoint_path = os.path.join(config['checkpoint_dir'], f"checkpoint-{step}")
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(checkpoint_path)
                save_training_state(
                    checkpoint_path,
                    step,
                    epoch,
                    optimizer.state_dict(),
                    scheduler.state_dict()
                )
                checkpoints = sorted([
                    d for d in os.listdir(config["checkpoint_dir"])
                    if d.startswith('checkpoint-')
                ])
                while len(checkpoints) > config["keep_last_checkpoints"]:
                    oldest_checkpoint = checkpoints.pop(0)
                    shutil.rmtree(os.path.join(config["checkpoint_dir"], oldest_checkpoint))

            if step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        progress_bar.close()

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(config["final_output_path"])
    tokenizer.save_pretrained(config["final_output_path"])
    save_training_state(
        config["final_output_path"],
        total_steps,
        epoch,
        optimizer.state_dict(),
        scheduler.state_dict()
    )


if __name__ == "__main__":
    main()
