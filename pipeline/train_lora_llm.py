import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
import warnings
warnings.filterwarnings("ignore", message="Detected kernel version")

# Ensure GPU is used
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ------------------------------
# Load dataset once
# ------------------------------
dataset = load_dataset("json", data_files={
    "train": "data/en/train.jsonl",
    "validation": "data/en/validation.jsonl",
    "test": "data/en/test.jsonl"
})

# ------------------------------
# Define models to fine-tune
# ------------------------------
models_to_train = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "falcon": "tiiuae/Falcon3-7B-Base",
    "llama2": "meta-llama/Llama-3.1-8B",
    "pythia": "EleutherAI/pythia-6.9b"
}

# ------------------------------
# Shared configurations
# ------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

# ------------------------------
# Train all models sequentially
# ------------------------------
for tag, model_name in models_to_train.items():
    print(f"\n==== Fine-tuning model: {model_name} ====")

    # Output directory
    output_dir = f"models/lora_{tag}_finetune"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model (on GPU with 4-bit quantization) ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # --- Ensure padding tokens are configured after loading model ---
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # --- Tokenize data ---
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- LoRA configuration ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )

    # --- Trainer setup ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --- Train model ---
    trainer.train()

    # --- Save model ---
    model.save_pretrained(output_dir, create_model_card=False)
    tokenizer.save_pretrained(output_dir)

    # --- Evaluate model ---
    results = trainer.evaluate(tokenized_datasets["test"])
    print(f"\n Test Results for {model_name}: {results}")

print("\n All 4 models have been fine-tuned and evaluated successfully on GPU.")
