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
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
warnings.filterwarnings("ignore", message="Detected kernel version")

# Ensure GPU is used
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Load dataset once
lang = cfg["project"]["language"]
data_root = cfg["project"]["data_root"]
dataset = load_dataset(
    "json",
    dataset_paths = {
    "train": os.path.join(data_root, lang, "train.jsonl"),
    "validation": os.path.join(data_root, lang, "validation.jsonl"),
    "test": os.path.join(data_root, lang, "test.jsonl"),
}
)

# Define models to fine-tune
models_to_train = {k: v["pretrained"] for k, v in cfg["models"].items()}

# Shared configurations
quant_cfg = cfg["quantization"]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=quant_cfg["load_in_4bit"],
    bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
)

def tokenize_function(examples, tokenizer):
    token_cfg = cfg["tokenization"]
    return tokenizer(
        examples["text"],
        truncation=token_cfg["truncation"],
        padding=token_cfg["padding"],
        max_length=token_cfg["max_length"],
    )

# Train all models sequentially
for tag, model_name in models_to_train.items():
    output_dir = cfg["models"][tag]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Output directory
    output_dir = f"models/lora_{tag}_finetune"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if cfg["quantization"]["enabled"] else None,
        device_map=cfg["runtime"]["device_map"],
        torch_dtype=getattr(torch, cfg["runtime"]["dtype"]),
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
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # --- Training arguments ---
    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        tf32=cfg["runtime"]["tf32"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        save_total_limit=train_cfg["save_total_limit"],
        report_to=train_cfg["report_to"],
        dataloader_num_workers=cfg["runtime"]["dataloader_num_workers"],
        ddp_find_unused_parameters=cfg["runtime"]["ddp_find_unused_parameters"],
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
