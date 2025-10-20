import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Base model and adapter paths
base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_path = "models/lora_mistral_finetune"  # your fine-tuned adapter folder

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

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.eos_token = "<|endoftext|>"  # Set <|endoftext|> as eos token
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)  # Get token ID for <|endoftext|>

print("Model and tokenizer loaded successfully.\n")

samples = [
    "Classify the emotions of: '- That\"s OWynn he\"s sitting with.'",  # trust, sadness
    "Classify the emotions of: 'Please sit down, sober up.'",  # joy, fear
    "Classify the emotions of: 'The prosecutor will explain everything when you see him.'",  # trust, sadness
    "Classify the emotions of: 'No, it isn't. Because right now, I have to decide whether he should stay at the firm.'"  # disgust
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
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {text}")
    print(f"Model output: {decoded}\n{'-'*60}\n")

print("Testing complete.")
