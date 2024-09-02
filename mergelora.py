from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch
import os
import json
import argparse
# Note this code defautls to bf16 format. Change this with the --fp16 or --fp32 flags
def main(args):
    base_model_path = 'C:/Base/Model/Path/Folder'
    peft_model_path = 'C:/Lora/aka/Peft/Model/Path/Folder'
    output_model_path = 'C:/Output/Model/Path/Folder'

    # Determine the output precision based on the flag
    if args.fp16:
        output_precision = torch.float16
        precision_name = "FP16"
    elif args.fp32:
        output_precision = torch.float32
        precision_name = "FP32"
    else:
        output_precision = torch.bfloat16  # Default to BF16
        precision_name = "BF16"

    print(f"[1/4] Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        return_dict=True,
        torch_dtype=output_precision,  # Use the determined precision
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"[2/4] Loading adapter from: {peft_model_path}")
    
    # Load the PEFT config
    with open(os.path.join(peft_model_path, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)
    
    # Remove 'layer_replication' if it exists
    adapter_config.pop('layer_replication', None)
    
    # Remove unexpected keyword arguments
    unexpected_keys = ['use_dora', 'use_rslora']
    for key in unexpected_keys:
        adapter_config.pop(key, None)
    
    # Create a new LoraConfig
    lora_config = LoraConfig(**adapter_config)
    
    # Load the PEFT model
    model = PeftModel.from_pretrained(base_model, peft_model_path, config=lora_config, device_map="cpu")

    print("[3/4] Merging base model and adapter")
    model = model.merge_and_unload()
    
    # Convert the model to the desired precision
    model = model.to(output_precision)

    print(f"[4/4] Saving merged model and tokenizer to: {output_model_path}")
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    print(f"Merged model saved to {output_model_path} in {precision_name} format")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument('--fp16', action='store_true', help='Output in FP16 precision')
    parser.add_argument('--fp32', action='store_true', help='Output in FP32 precision')
    # Note: BF16 is default, so no flag is needed

    args = parser.parse_args()

    # Ensure only one precision flag is set
    if args.fp16 and args.fp32:
        raise ValueError("Please specify only one precision flag (--fp16 or --fp32)")

    main(args)