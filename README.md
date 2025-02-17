A single runnable file for LLM training. Everything you need can be executed from the 1 file. Pick your method of training from the file name. Fill in all the places in the file with what you need to train, and run "Python *File_Name.py*" from a command prompt open in the same directory as the file.

As long as you have some sort of graphics card and train a model that fits in your VRAM, the training should work well.

It's as simple as that.


For better Lora training. Use my method bellow

# Continuous Fine-tuning Without Loss Using Lora and Mergekit

https://docs.google.com/document/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?usp=sharing

# INSTRUCTIONS:

Download the folder "Galore+Qlora_With_Multi_GPU_Support"

Run
```
pip install -r requirements.txt
```
edit the config file, and run this command to execute it
```
python QaloreTraining.py --config_file config.txt
```

# Explanation of Config Flags:

- **model_path**:  
  Folder (or Hugging Face ID) of the pre-trained model you start with.  
  Use: `"model_path": "./models/your_model"`

- **checkpoint_dir**:  
  Folder where training checkpoints (temporary model snapshots) are saved.  
  Use: `"checkpoint_dir": "./models/checkpoints"`

- **cache_dir**:  
  Directory for storing cached data (helps load data faster).  
  Use: `"cache_dir": "./models/cache"`

- **dataset_path**:  
  Path to the JSON file that contains your training examples.  
  Use: `"dataset_path": "./models/dataset.json"`

- **final_output_path**:  
  Folder where the final fine-tuned model is saved after training.  
  Use: `"final_output_path": "./models/final_model"`

- **num_epochs**:  
  Number of full passes through your entire dataset during training.  
  Use: `"num_epochs": 1`

- **save_interval**:  
  Number of training steps between each checkpoint save.  
  Use: `"save_interval": 900`

- **keep_last_checkpoints**:  
  Maximum number of recent checkpoints to keep; older ones are removed.  
  Use: `"keep_last_checkpoints": 3`

- **batch_size**:  
  Number of examples processed in one forward/backward pass.  
  Use: `"batch_size": 3`

- **accumulation_steps**:  
  Number of batches to accumulate gradients before updating weights (simulates a larger batch).  
  Use: `"accumulation_steps": 20`

- **num_workers**:  
  Number of CPU threads used for loading the data.  
  Use: `"num_workers": 12`

- **max_grad_norm**:  
  Maximum allowed value for gradients (to prevent exploding gradients).  
  Use: `"max_grad_norm": 1.0`

- **learning_rate**:  
  Speed at which the model learns; a smaller value means slower updates.  
  Use: `"learning_rate": 0.0003`

- **eta_min**:  
  The lowest learning rate the scheduler will use during training.  
  Use: `"eta_min": 1e-6`

- **first_cycle_fraction**:  
  Fraction of the total steps used for the first learning rate cycle (scheduler setting).  
  Use: `"first_cycle_fraction": 0.1`

- **t_mult**:  
  Multiplier to increase the cycle length after each restart in the learning rate scheduler.  
  Use: `"t_mult": 2`

- **rank**:  
  GaLore optimizer parameter defining the size of low-rank updates.  
  Use: `"rank": 64`

- **update_proj_gap**:  
  How often (in steps) the GaLore projection is updated.  
  Use: `"update_proj_gap": 200`

- **scale**:  
  A factor used by GaLore to adjust the strength of updates.  
  Use: `"scale": 0.25`

- **proj_type**:  
  Type of projection used in GaLore (often "std" for standard).  
  Use: `"proj_type": "std"`

- **max_seq_length**:  
  Maximum number of tokens per training example (longer sequences use more memory).  
  Use: `"max_seq_length": 8192`

- **use_qlora**:  
  If `true`, uses LoRA adapters (only a small set of parameters are trained).  
  If `false`, performs full fine-tuning (updates all model weights).  
  Use: `"use_qlora": true`

- **load_in_4bit**:  
  If `true`, loads the model with 4-bit quantization to save memory.  
  Use: `"load_in_4bit": true`

- **bnb_4bit_compute_dtype**:  
  Data type for 4-bit mode computations, balancing speed and precision.  
  Use: `"bnb_4bit_compute_dtype": "bfloat16"`

- **bnb_4bit_quant_type**:  
  Quantization method for 4-bit mode (affects model precision).  
  Use: `"bnb_4bit_quant_type": "nf4"`

- **bnb_4bit_use_double_quant**:  
  If `true`, uses double quantization for improved precision in 4-bit mode.  
  Use: `"bnb_4bit_use_double_quant": true`

- **lora_r**:  
  LoRA adapter rank; a lower number means fewer trainable parameters.  
  Use: `"lora_r": 32`

- **lora_alpha**:  
  Scaling factor for the LoRA adapter weights.  
  Use: `"lora_alpha": 32`

- **lora_dropout**:  
  Dropout rate in LoRA layers to help prevent overfitting.  
  Use: `"lora_dropout": 0.1`

- **num_gpus**:  
  Number of GPUs used during training for faster computation.  
  Use: `"num_gpus": 2`

- **prompt_template**:  
  Template for formatting each training example (instruction, input, output).  
  Use:  
  ```json
  "prompt_template": "<|im_start|>system\n\n{instruction}<|im_end|>\n<|im_start|>user\n\n{input}<|im_end|>\n<|im_start|>assistant\n\n{output}<|im_end|>"
  ```
