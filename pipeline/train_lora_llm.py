# train_lora_emotions_masked.py
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "mistralai/Mistral-Large-Instruct-2407"
DATA_FILES = {
    "train": "data/en/train.jsonl",
    "validation": "data/en/validation.jsonl",
    "test": "data/en/test.jsonl",
}
OUTPUT_DIR = "./mistral_single_masked"
USE_4BIT = True
LR = 2e-4
NUM_EPOCHS = 4
EVAL_STEPS = 50
SAVE_STEPS = 200
GRAD_ACCUM = 8
TRAIN_BSZ = 2
EVAL_BSZ = 2
WARMUP_RATIO = 0.1
SEED = 42

ASSISTANT_TAG = "<|assistant|>"
STOP_STR = "<|endoftext|>"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
dataset = load_dataset("json", data_files=DATA_FILES)

# ---------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id

# ---------------------------------------------------------------------
# Split prompt/answer in TEXT first, then tokenize parts (robust masking)
# ---------------------------------------------------------------------
def split_prompt_answer(rec: str):
    """
    From one training string:
      <|user|> ...\n<|assistant|> joy, sadness <|endoftext|>
    return (prompt_text_ends_with_assistant_tag, answer_text, stop_text)
    """
    # normalize any escaped close tags if present
    if ASSISTANT_TAG not in rec:
        return None, None, None
    left, right = rec.split(ASSISTANT_TAG, 1)

    # answer = content before STOP_STR (if present)
    stop_text = ""
    if STOP_STR in right:
        answer_text, _ = right.split(STOP_STR, 1)
        stop_text = STOP_STR
    else:
        answer_text = right

    prompt_text = left + ASSISTANT_TAG  # keep assistant tag in the context
    return prompt_text, answer_text, stop_text

def preprocess_fn(examples: Dict[str, List[str]]) -> Dict[str, Any]:
    prompts, answers, stops, keep = [], [], [], []
    for s in examples["text"]:
        p, a, st = split_prompt_answer(s)
        if p is None:
            p, a, st = "", "", ""
            keep.append(0)
        else:
            keep.append(1)
        prompts.append(p)
        answers.append(a)
        stops.append(st)

    tok_p = tokenizer(prompts, add_special_tokens=False)
    tok_a = tokenizer(answers, add_special_tokens=False)
    tok_st = tokenizer(stops, add_special_tokens=False)

    return {
        "prompt_ids": tok_p["input_ids"],
        "answer_ids": tok_a["input_ids"],
        "stop_ids": tok_st["input_ids"],
        "keep": keep,
    }

proc_dataset = dataset.map(preprocess_fn, batched=True, remove_columns=["text"])

# ---------------------------------------------------------------------
# Collator: build input_ids = prompt + answer + stop; labels = -100/prompt, answer, -100/stop
# ---------------------------------------------------------------------
@dataclass
class CollatorPromptAnswer:
    tokenizer: AutoTokenizer
    padding: str = "longest"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        packed = []
        for f in features:
            p = f["prompt_ids"]
            a = f["answer_ids"]
            st = f["stop_ids"]
            ids = p + a + st
            # labels: ignore prompt & stop, learn only answer tokens
            lab = [-100]*len(p) + a[:] + [-100]*len(st)
            packed.append({"input_ids": ids, "labels": lab})

        batch = DataCollatorWithPadding(self.tokenizer, padding=self.padding)(packed)
        return batch

data_collator = CollatorPromptAnswer(tokenizer)

# ---------------------------------------------------------------------
# Model (QLoRA 4-bit)
# ---------------------------------------------------------------------
bnb_config = (
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if USE_4BIT and torch.cuda.is_available()
    else None
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if torch.cuda.is_available() else None,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

# Stronger LoRA target set for Mistral/LLaMA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# ---------------------------------------------------------------------
# Training args
# ---------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    logging_steps=EVAL_STEPS,
    learning_rate=LR,
    per_device_train_batch_size=TRAIN_BSZ,
    per_device_eval_batch_size=EVAL_BSZ,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    bf16=True,
    tf32=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0,
    seed=SEED,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
)

# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=proc_dataset["train"],
    eval_dataset=proc_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ---------------------------------------------------------------------
# Train / Save / Eval
# ---------------------------------------------------------------------
trainer.train()

model.config.use_cache = True  # re-enable cache for inference
model.save_pretrained(OUTPUT_DIR, create_model_card=False)
tokenizer.save_pretrained(OUTPUT_DIR)

results = trainer.evaluate(proc_dataset["test"])
print("Test Results:", results)
