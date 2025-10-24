import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
from accelerate import Accelerator

# Initialize Accelerator
accelerator = Accelerator()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("json", data_files={
    "train": "data/combined/train.jsonl",
    "validation": "data/combined/validation.jsonl",
    "test": "data/combined/test.jsonl"
})

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Bits and bytes configuration for model loading (4-bit quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model with quantization config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically place model on available GPUs
    torch_dtype=torch.bfloat16
)

# LORA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LORA to the model
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing and other optimizations
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# Move the model to the correct device (GPU)
model.to(device)

model.print_trainable_parameters()

# Tokenizer function to process the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments, configured for multi-GPU training
training_args = TrainingArguments(
    output_dir="./lora_mistral_combined",
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Adjust batch size according to GPU memory
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    warmup_ratio=0.05,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    bf16=True,
    tf32=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    save_total_limit=2,
    report_to="none",
    ddp_find_unused_parameters=False,
    dataloader_num_workers=2,
    no_cuda=False  # Use GPU
)

# Use accelerator to prepare model and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Prepare everything using accelerator (model, datasets, dataloaders)
model, tokenized_datasets["train"], tokenized_datasets["validation"] = accelerator.prepare(
    model, tokenized_datasets["train"], tokenized_datasets["validation"]
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./lora_mistral_Large", create_model_card=False)
tokenizer.save_pretrained("./lora_mistral_Large")

# Evaluate the model
results = trainer.evaluate(tokenized_datasets["test"])
print("Test Results:", results)
