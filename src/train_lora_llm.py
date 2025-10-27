"""
Fine-tune multiple Mistral-size models with LoRA + 4-bit quantization.
Plug-and-play: just drop your dataset JSONL files into /data/en and run.
"""

import os
import torch
import warnings
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore", message="Detected kernel version")

# ============================================================
#  DEVICE SETUP
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ============================================================
#  LOAD DATASET ONCE
# ============================================================
dataset = load_dataset("json", data_files={
    "train": "data/en/train.jsonl",
    "validation": "data/en/validation.jsonl",
    "test": "data/en/test.jsonl",
})

# ============================================================
#  MODEL CHOICES
# ============================================================
models_to_train = {
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "ministral_3b": "mistralai/Ministral-3B-v0.1",
    "mistral_tiny": "mistralai/Mistral-3B-Instruct-v0.1"
}

# ============================================================
#  SHARED CONFIGS
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def tokenize_function(examples, tokenizer):
    # Combine text + label into one supervised sequence
    full_texts = [
        f"Classify the emotions of: {t}\nAnswer: {l}"
        for t, l in zip(examples["text"], examples["label"])
    ]
    return tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )


# ============================================================
#  TRAIN ALL MODELS
# ============================================================
for tag, model_name in models_to_train.items():
    print(f"\n==============================")
    print(f" Fine-tuning model: {model_name}")
    print(f"==============================")

    # Output directory
    output_dir = f"models/{tag}_lora_finetune"
    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenize data
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text", "label"],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # LoRA setup
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        logging_steps=50,
        learning_rate=2e-4,
        per_device_train_batch_size=1,   # safe default for 7B models
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        tf32=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    results = trainer.evaluate(tokenized_datasets["test"])
    print(f"\n Test Results for {tag}: {results}")

print("\n All Mistral models fine-tuned successfully!")
