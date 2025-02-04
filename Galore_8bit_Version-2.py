import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from tqdm import tqdm
from galore_torch import GaLoreAdamW8bit
import torch.amp as amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import gc
import os
import json
import shutil
import argparse

def save_training_state(checkpoint_dir, step, epoch, optimizer_state, scheduler_state):
    """Save training progress, optimizer and scheduler state"""
    state = {
        'step': step,
        'epoch': epoch,
        'optimizer_state': optimizer_state,
        'scheduler_state': scheduler_state
    }
    with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
        json.dump(state, f)

def load_training_state(checkpoint_dir):
    """Load training progress, optimizer and scheduler state"""
    state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return None

def main():
    # Set the number of epochs here
    num_epochs = 1 # Change this value to your desired number of epochs

    # Training configuration
    save_interval = 100000000000000000 # Change this number to save checkpoint at specified step during training
    checkpoint_dir = "C:/Path/to/AI/Model/Checkpoint"
    keep_last_checkpoints = 3

    # Initialize starting point
    start_step = 0
    start_epoch = 0

    # Load the tokenizer and model
    model_path = "C:/Path/to/Input/AI/Model"
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Modified 8-bit configuration
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0}
    )

    # Dataset preparation
    cache_dir = "C:/Path/to/Cache/Location"
    os.makedirs(cache_dir, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files="C:/Path/to/Training/Dataset.json",
        split="train",
        cache_dir=cache_dir
    )

    def preprocess(example):
        formatted_text = (
            f"<|im_start|>system\n\n"
            f"{example['instruction']}<|im_end|>\n"
            f"<|im_start|>user\n\n"
            f"{example['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n\n"
            f"{example['output']}<|im_end|>"
        )
            
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=2048,
            padding='max_length',
            return_tensors=None
        )
        
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=100,
        num_proc=12,
        remove_columns=dataset.column_names,
        cache_file_name=os.path.join(cache_dir, "processed_dataset.arrow")
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    batch_size = 4000
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
        collate_fn=data_collator
    )

    # Optimizer setup
    accumulation_steps = 20
    galore_params = []
    target_modules_list = ["attn", "mlp"]

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(target_key in module_name for target_key in target_modules_list):
            continue
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
            'rank': 64,
            'update_proj_gap': 200,
            'scale': 0.25,
            'proj_type': 'std'
        }
    ]

    optimizer = GaLoreAdamW8bit(param_groups, lr=3e-4)
    
    # Calculate scheduler parameters
    total_training_steps = len(train_dataloader)
    first_cycle_steps = int(total_training_steps * 0.1)  # First cycle is 10% of total steps

    # Initialize the cosine scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=first_cycle_steps,  # Length of first cycle
        T_mult=2,  # Each cycle is 1.5x longer than the last
        eta_min=1e-6  # Minimum learning rate
    )

    # Training loop
    model.train()
    total_steps = len(train_dataloader)
    prev_avg_loss = 0  # Initialize for moving average calculation
    
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(enumerate(train_dataloader), total=total_steps, initial=start_step)
        
        for step, batch in progress_bar:
            if step < start_step:
                continue
                
            # Process batch
            inputs = {
                k: v.view(-1, v.size(-1)).cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # Forward and backward passes
            outputs = model(**inputs)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            running_loss += loss.item()
            
            if (step + 1) % accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()
                
                # Calculate metrics for progress bar
                current_lr = scheduler.get_last_lr()[0]  # Get current learning rate
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
            
            # Save checkpoint
            if step > 0 and step % save_interval == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint-{step}"
                model.save_pretrained(checkpoint_path)
                
                save_training_state(
                    checkpoint_path,
                    step,
                    epoch,
                    optimizer.state_dict(),
                    scheduler.state_dict()
                )
                
                # Remove old checkpoints
                checkpoints = sorted([d for d in os.listdir(checkpoint_dir) 
                                   if d.startswith('checkpoint-')])
                while len(checkpoints) > keep_last_checkpoints:
                    oldest_checkpoint = checkpoints.pop(0)
                    shutil.rmtree(os.path.join(checkpoint_dir, oldest_checkpoint))
            
            # Memory cleanup
            if step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        progress_bar.close()

    # Save final model
    final_path = "C:/Path/to/AI/Model/Final/Output"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    save_training_state(
        final_path,
        total_steps,
        epoch,
        optimizer.state_dict(),
        scheduler.state_dict()
    )

if __name__ == "__main__":
    main()
