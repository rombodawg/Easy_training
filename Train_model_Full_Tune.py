import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import load_dataset
from tqdm import tqdm
from torch.optim import AdamW

# Custom callback for visual progress bar
class ProgressBarCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.pbar = tqdm(total=total_steps, desc="Training Progress", unit="step")

    def on_step_end(self, args, state, control, **kwargs):
        self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.close()

if __name__ == "__main__":
    # Load the tokenizer and model
    model_path = "C:/Base/Model/Path/FolderName"  # Path to the model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)

    # Load the dataset
    dataset_path = "C:/Dataset/path/dataset.json"  # Path to the dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Preprocess the dataset
    def preprocess(example):
        # Match this to your instruction template and dataset format
        formatted_text = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n\n" \
                         f"<|im_start|>user\n{example['input']}<|im_end|>\n\n" \
                         f"<|im_start|>assistant\n{example['output']}<|endoftext|>"
        tokenized = tokenizer(formatted_text, truncation=True, max_length=8192, padding="max_length")
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }

    tokenized_dataset = dataset.map(preprocess, batched=False, remove_columns=dataset.column_names)

    # Set up DataCollator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Set up training arguments
    # Make sure "output_dir" matches "model.save_pretrained" and "tokenizer.save_pretrained" at end of script
    training_args = TrainingArguments(
        output_dir="C:/Final/Output/Model/Path/FolderName",  # Path to save the trained model
        # How many times are you loading the model onto the gpu for training (More = faster training but more vram use)
        per_device_train_batch_size=1,
        # How many times will the model be trained on the dataset ⬇
        num_train_epochs=1,  # Updated to 5 epochs
        learning_rate=2e-4,
        logging_dir="./logs",
        # How many steps until you see progress of training ⬇
        logging_steps=10,
        # Max checkpoints (delete if you don't care about storage space) ⬇
        save_total_limit=2,
        # How many checkpoints are you saving ⬇
        save_steps=50,  # Updated to match the first script
        evaluation_strategy="no",
        # What format is your base model (bf16, fp32, fp16) ⬇
        bf16=True,
        push_to_hub=False,
        report_to=["tensorboard"],
        gradient_accumulation_steps=1,
        # "dataloader_num_workers" should match your cpu threads ⬇
        dataloader_num_workers=2,
        disable_tqdm=True 
    )

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    # Calculate total steps for the progress bar
    total_steps = len(tokenized_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[ProgressBarCallback(total_steps)],
        optimizers=(optimizer, None)  # Pass the optimizer explicitly
    )

    # Train the model
    trainer.train()

    # Save the model
    # Make sure this matches the "output_dir" in training_args
    trainer.save_model("C:/Final/Output/Model/Path/FolderName")  # Path to save the trained model
    tokenizer.save_pretrained("C:/Final/Output/Model/Path/FolderName")  # Path to save the tokenizer
