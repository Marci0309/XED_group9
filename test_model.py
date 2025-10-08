# === TEST SCRIPT FOR FINE-TUNED LoRA MODEL ===

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Base model and adapter paths
base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_path = "lora_mistral_emotions/checkpoint-800"  # your fine-tuned adapter folder

# 4-bit config for GPU loading (optional, saves VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto" if device == "cuda" else None,
    quantization_config=bnb_config if device == "cuda" else None,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model and tokenizer loaded successfully.\n")

samples = [
    "Classify the emotions of: 'weren't you reprimanding a seaman for having his shirt-tail out, while the ship turned 360 degrees?'",
    "Classify the emotions of: 'He's such a sweet thing.'",
    "Classify the emotions of: 'Are you sure they're here? CHAVEZ:  They're here.'",
    "Classify the emotions of: 'Oh wow, that’s such a beautiful surprise!'",
]

for text in samples:
    prompt = f"<|user|> {text}\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {text}")
    print(f"Model output: {decoded}\n{'-'*60}\n")

print("Testing complete.")
