import os
import re
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GenerationConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers.trainer_utils import set_seed


# -----------------------
# CONFIG
# -----------------------
class Config:
    # Base model
    base_model =  "bigscience/bloomz-560m" 
    # Alternative (bigger): "bigscience/bloomz-1b7", "ai-forever/mGPT"
    data_dir = "data"  # where you put data/en|hu|nl|ro/ splits
    languages = ["en", "hu", "nl", "ro"]  # folders present under data/
    output_root = "outputs"

    # Training
    seed = 42
    num_train_epochs = 2
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    warmup_ratio = 0.03
    logging_steps = 50
    eval_strategy = "epoch"
    save_strategy = "epoch"

    # Max sequence length for instruction samples
    max_length = 512

    # Decoding for evaluation
    gen_max_new_tokens = 16
    gen_temperature = 0.0
    gen_top_p = 1.0

    # Choose mode:
    # "single" -> train one model per language
    # "multi"  -> merge languages and train one multilingual model
    mode = "single"  # or "multi"

    # PEFT options:
    # "lora"  -> standard LoRA (full precision/fp16/bf16)
    # "qlora" -> 4-bit QLoRA (bitsandbytes)
    peft_type = "lora"  # or "qlora"

    # PEFT sweep to compare different configs
    lora_sweep = [
        {"r": 8,  "alpha": 16, "dropout": 0.05},
        {"r": 16, "alpha": 32, "dropout": 0.05},
        {"r": 32, "alpha": 64, "dropout": 0.1},
    ]

    # Target modules to adapt (BLOOMZ)
    target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    # QLoRA settings
    load_in_4bit = True
    bnb_4bit_use_double_quant = True
    bnb_4bit_compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    bnb_4bit_quant_type = "nf4"


cfg = Config()
set_seed(cfg.seed)


# -----------------------
# DATA LOADING
# -----------------------
def load_lang_dataset(lang: str) -> DatasetDict:
    """Load a single language dataset from jsonl files."""
    path = os.path.join(cfg.data_dir, lang)
    return load_dataset(
        "json",
        data_files={
            "train": os.path.join(path, "train.jsonl"),
            "validation": os.path.join(path, "validation.jsonl"),
            "test": os.path.join(path, "test.jsonl"),
        }
    )


def load_multi_dataset(langs: List[str]) -> DatasetDict:
    """Concatenate multiple languages into one merged DatasetDict."""
    dss = [load_lang_dataset(l) for l in langs]
    # Merge splits by concatenation
    train = concatenate_datasets([ds["train"] for ds in dss])
    val = concatenate_datasets([ds["validation"] for ds in dss])
    test = concatenate_datasets([ds["test"] for ds in dss])
    return DatasetDict(train=train, validation=val, test=test)


# -----------------------
# TOKENIZATION
# -----------------------
def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tok.pad_token is None:
        # For causal LMs, set pad_token to eos_token
        tok.pad_token = tok.eos_token
    return tok


def tokenize_fn(example, tokenizer: AutoTokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=cfg.max_length,
        padding=False,  # let DataCollator pad dynamically
    )


# -----------------------
# MODEL + PEFT WRAP
# -----------------------
def load_base_model():
    kwargs = {}
    if cfg.peft_type == "qlora":
        # 4-bit quantized base model for QLoRA
        kwargs.update(
            dict(
                load_in_4bit=cfg.load_in_4bit,
                device_map="auto",
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
            )
        )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    if cfg.peft_type == "qlora":
        model = prepare_model_for_kbit_training(model)
    return model


def wrap_with_lora(model, r: int, alpha: int, dropout: float):
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=cfg.target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


# -----------------------
# COLLATOR
# -----------------------
def build_collator(tokenizer):
    # Standard causal-LM collator (labels = input_ids shifted internally)
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# -----------------------
# EVALUATION (GENERATION-BASED ACCURACY)
# -----------------------
emotions_pattern = re.compile(r"(\d(?:\s*,\s*\d)*)")

def parse_emotion_ids(text: str) -> Optional[str]:
    """
    Extract a comma-separated list of integers as a string (e.g., "1, 4").
    Returns normalized form like "1,4".
    """
    m = emotions_pattern.search(text)
    if not m:
        return None
    # normalize "1, 4" -> "1,4"
    norm = ",".join(x.strip() for x in m.group(1).split(","))
    return norm


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, ds, split_name: str):
    model.eval()
    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.gen_max_new_tokens,
        do_sample=False if cfg.gen_temperature == 0.0 else True,
        temperature=cfg.gen_temperature,
        top_p=cfg.gen_top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    correct = 0
    total = 0
    for ex in ds:
        prompt = ex["text"]
        # The ground-truth labels are already embedded after <|assistant|>
        # Extract gold from the training string
        # Everything after last "<|assistant|>" is the gold answer
        if "<|assistant|>" in prompt:
            gold = prompt.split("<|assistant|>")[-1].strip()
        else:
            # fallback: try end of line
            gold = prompt.splitlines()[-1].strip()

        gold_norm = parse_emotion_ids(gold)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, generation_config=gen_cfg)
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        pred_norm = parse_emotion_ids(gen)

        if gold_norm is not None and pred_norm is not None and gold_norm == pred_norm:
            correct += 1
        total += 1

    acc = correct / max(total, 1)
    print(f"[{split_name}] Generation exact-match accuracy: {acc:.4f} ({correct}/{total})")
    return acc


# -----------------------
# TRAIN ONE EXPERIMENT
# -----------------------
def run_experiment(run_name: str, dataset: DatasetDict, lora_r: int, lora_alpha: int, lora_dropout: float):
    os.makedirs(os.path.join(cfg.output_root, run_name), exist_ok=True)

    tokenizer = build_tokenizer()
    tokenized = dataset.map(lambda ex: tokenize_fn(ex, tokenizer), batched=True, remove_columns=dataset["train"].column_names)

    model = load_base_model()
    model = wrap_with_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    args = TrainingArguments(
        output_dir=os.path.join(cfg.output_root, run_name),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        fp16=torch.cuda.is_available() and cfg.peft_type == "lora",   # fp16 for LoRA
        bf16=torch.cuda.is_available() and cfg.peft_type == "qlora",  # bf16 compute for QLoRA
        report_to="none",
    )

    collator = build_collator(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Evaluate with perplexity (language modeling loss)
    eval_res = trainer.evaluate()
    ppl = math.exp(eval_res["eval_loss"]) if "eval_loss" in eval_res and eval_res["eval_loss"] < 20 else float("inf")
    print(f"[{run_name}] eval_loss={eval_res.get('eval_loss'):.4f} | ppl={ppl:.2f}")

    # Also evaluate with generation-based exact-match accuracy on raw validation/test (needs original text)
    # Reload raw splits to evaluate by generation (not tokenized)
    # NOTE: We kept dataset before tokenization in 'dataset'
    val_acc = evaluate_accuracy(model, tokenizer, dataset["validation"], "validation")
    test_acc = evaluate_accuracy(model, tokenizer, dataset["test"], "test")

    # Save small summary
    summary = {
        "run_name": run_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "peft_type": cfg.peft_type,
        "eval_loss": eval_res.get("eval_loss"),
        "perplexity": ppl,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }
    with open(os.path.join(cfg.output_root, run_name, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", os.path.join(cfg.output_root, run_name, "summary.json"))


# -----------------------
# MAIN: run comparisons
# -----------------------
def main():
    set_seed(cfg.seed)

    if cfg.mode == "single":
        # Train a separate model per language + sweep LoRA configs
        for lang in cfg.languages:
            ds = load_lang_dataset(lang)
            # Shuffle to avoid any ordering bias
            ds = DatasetDict(
                train=ds["train"].shuffle(seed=cfg.seed),
                validation=ds["validation"].shuffle(seed=cfg.seed),
                test=ds["test"].shuffle(seed=cfg.seed),
            )
            for peft in cfg.lora_sweep:
                run_name = f"{cfg.peft_type}_{lang}_r{peft['r']}_a{peft['alpha']}_d{str(peft['dropout']).replace('.','')}"
                run_experiment(run_name, ds, lora_r=peft["r"], lora_alpha=peft["alpha"], lora_dropout=peft["dropout"])

    elif cfg.mode == "multi":
        # Merge languages and train one multilingual model + sweep LoRA configs
        ds = load_multi_dataset(cfg.languages)
        ds = DatasetDict(
            train=ds["train"].shuffle(seed=cfg.seed),
            validation=ds["validation"].shuffle(seed=cfg.seed),
            test=ds["test"].shuffle(seed=cfg.seed),
        )
        for peft in cfg.lora_sweep:
            run_name = f"{cfg.peft_type}_multi_r{peft['r']}_a{peft['alpha']}_d{str(peft['dropout']).replace('.','')}"
            run_experiment(run_name, ds, lora_r=peft["r"], lora_alpha=peft["alpha"], lora_dropout=peft["dropout"])

    else:
        raise ValueError("cfg.mode must be 'single' or 'multi'")


if __name__ == "__main__":
    main()
