SUPPORT ME ON PATREON

- https://www.patreon.com/c/Rombodawg

_____________________________________________________________________________________________
# A single runnable file for LLM training. Everything you need can be executed from the 1 file. 

Note: This code supports full-tuning if "use_qlora:" flag is set to "false"
_______________________________________________________________________________________________

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
accelerate launch QaloreTraining.py --config_file config.txt
```
_______________________________________________________________________________________
Note: the (--resume_checkpoint) flag exists in the code and is technicallly functional but is mostly broken and I dont recommend using it
_______________________________________________________________________________________
### Explanation of Config Flags

- **model_path**:  
  Folder (or Hugging Face model ID) for the pre-trained model you start with.  
  *Example:* `"model_path": "./model"`

- **checkpoint_dir**:  
  Directory where training checkpoints (temporary model snapshots) are saved.  
  *Example:* `"checkpoint_dir": "./checkpoint"`

- **cache_dir**:  
  Directory for storing cached data (speeds up dataset loading).  
  *Example:* `"cache_dir": "./cache"`

- **dataset_path**:  
  Path to the JSON file containing your training examples.  
  *Example:* `"dataset_path": "./dataset.json"`

- **final_output_path**:  
  Directory where the final fine-tuned model and tokenizer will be saved.  
  *Example:* `"final_output_path": "./output"`

- **num_epochs**:  
  Number of full passes through your dataset during training.  
  *Example:* `"num_epochs": 1`

- **save_interval**:  
  Number of training steps between each checkpoint save.  
  *Example:* `"save_interval": 100000000000000000`

- **keep_last_checkpoints**:  
  Maximum number of most recent checkpoints to retain (older ones are removed).  
  *Example:* `"keep_last_checkpoints": 3`

- **batch_size**:  
  Number of examples processed per forward/backward pass.  
  *Example:* `"batch_size": 4`

- **accumulation_steps**:  
  Number of batches to accumulate before performing an optimizer update (simulates a larger batch size).  
  *Example:* `"accumulation_steps": 20`

- **num_workers**:  
  Number of CPU threads to use for loading data in the DataLoader.  
  *Example:* `"num_workers": 3`

- **max_grad_norm**:  
  Maximum gradient norm for gradient clipping (prevents exploding gradients).  
  *Example:* `"max_grad_norm": 1.0`

- **learning_rate**:  
  The learning rate for the optimizer; determines how fast the model learns.  
  *Example:* `"learning_rate": 0.0003`

- **eta_min**:  
  The minimum learning rate for the cosine annealing scheduler.  
  *Example:* `"eta_min": 1e-6`

- **first_cycle_fraction**:  
  Fraction of total steps used for the first cycle in the cosine annealing learning rate scheduler.  
  *Example:* `"first_cycle_fraction": 0.1`

- **t_mult**:  
  Multiplier to increase the cycle length after each restart in the learning rate scheduler.  
  *Example:* `"t_mult": 2`

- **rank**:  
  GaLore optimizer parameter defining the size of low-rank updates.  
  *Example:* `"rank": 64`

- **update_proj_gap**:  
  Frequency (in steps) at which the GaLore projection is updated.  
  *Example:* `"update_proj_gap": 200`

- **scale**:  
  Scaling factor used by GaLore to adjust update strength.  
  *Example:* `"scale": 0.25`

- **proj_type**:  
  Type of projection used in GaLore (commonly `"std"` for standard).  
  *Example:* `"proj_type": "std"`

- **max_seq_length**:  
  Maximum number of tokens per training example (longer sequences consume more memory).  
  *Example:* `"max_seq_length": 2048`

- **use_qlora**:  
  If set to `true`, uses LoRA adapters (only a subset of model parameters is trained). If set to `false` this will essentailly be full-tuning.
  *Example:* `"use_qlora": true`

- **load_in_4bit**:  
  If `true`, loads the model in 4-bit quantization mode to reduce memory usage.  
  *Example:* `"load_in_4bit": true`

- **bnb_4bit_quant_type**:  
  Specifies the quantization method for 4-bit mode (affects precision and speed).  
  *Example:* `"bnb_4bit_quant_type": "nf4"`

- **bnb_4bit_use_double_quant**:  
  If `true`, applies double quantization for improved precision in 4-bit mode.  
  *Example:* `"bnb_4bit_use_double_quant": true`

- **lora_r**:  
  Rank of the LoRA adapter (controls the number of trainable parameters for LoRA).  
  *Example:* `"lora_r": 8`

- **lora_alpha**:  
  Scaling factor for the LoRA adapter weights.  
  *Example:* `"lora_alpha": 32`

- **lora_dropout**:  
  Dropout rate used in LoRA layers to help prevent overfitting.  
  *Example:* `"lora_dropout": 0.1`

- **num_gpus**:  
  Number of GPUs to be used for training.  
  *Example:* `"num_gpus": 2`

- **warmup_steps**:
  Number of steps to gradually increase the learning rate at the start of training.
  Use: `"warmup_steps": 200`

- **added_tokens**:
  Custom tokens to add to the tokenizer's vocabulary.
  Use: `"added_tokens": "<example>,</example2>"`

- **prompt_template**:  
  Template used to format each training example (inserts instruction, input, and output into a predefined format).  
  *Example:*  
  ```json
  "prompt_template": "<|im_start|>system\n\n{instruction}<|im_end|>\n<|im_start|>user\n\n{input}<|im_end|>\n<|im_start|>assistant\n\n{output}<|im_end|>"
  ```
