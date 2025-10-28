import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import os
import json

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 50

DATA_SINGLE = "data/en/test.jsonl"
DATA_MULTI = "data/combined_simple/test.jsonl"

# Output directory for saving results
os.makedirs("results", exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Loading model: {MODEL_NAME}")

# ============================================================
# LOAD MODEL & TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

metric = evaluate.load("accuracy")

# ============================================================
# PROMPT STRATEGIES
# ============================================================

def build_zero_shot_prompt(text):
    """Basic classification prompt (no examples)."""
    return f"<s>[INST] Classify the emotions of: '{text}' [/INST]"

def build_few_shot_prompt(text, examples):
    """Prompt with a few labeled examples."""
    demo_text = "\n".join([f"Sentence: {ex['text']}\nEmotions: {ex['label']}" for ex in examples])
    return (
        f"<s>[INST] You are an emotion classifier. Here are examples:\n"
        f"{demo_text}\n\nNow classify the emotions of: '{text}' [/INST]"
    )

def custom_instruction_prompt(text):
    """Explicitly guide the model with stronger task instructions."""
    return (
        f"<s>[INST] You are a precise emotion classifier. Identify all emotions in the sentence based on Plutchik's 8 basic emotions and return"
        f" as comma-separated values from: joy, trust, fear, surprise, sadness, disgust, anger, anticipation.\n\n"
        f"Sentence: '{text}' [/INST]"
    )

# ============================================================
# RESPONSE GENERATION
# ============================================================

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,  # Ensures no caching of past states to save memory
        )
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text.strip().lower()

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_strategy(dataset, name, prompt_fn, few_shot_examples=None):
    preds, refs = [], []
    print(f"\n Evaluating strategy: {name} on {len(dataset)} samples")

    for example in tqdm(dataset):
        text = example["text"]
        label = example["label"].lower().strip()
        if name == "few-shot":
            prompt = build_few_shot_prompt(text, few_shot_examples)
        else:
            prompt = prompt_fn(text)

        pred = generate_response(prompt)
        preds.append(pred)
        refs.append(label)

    # Compute simple accuracy (string match)
    acc = sum([r in p for p, r in zip(preds, refs)]) / len(refs)
    print(f"Accuracy ({name}): {acc:.3f}")
    return acc

# ============================================================
# FEW-SHOT EXAMPLES (English)
# ============================================================
few_shot_examples = [
    {"text": "I can't believe this happened!", "label": "surprise"},
    {"text": "I'm so happy for you!", "label": "joy"},
    {"text": "This is so unfair.", "label": "anger"},
]

# ============================================================
# LOAD DATASETS
# ============================================================
print("\n Loading test datasets...")
dataset_single = load_dataset("json", data_files=DATA_SINGLE)["train"]
dataset_multi = load_dataset("json", data_files=DATA_MULTI)["train"]

# ============================================================
# RUN EVALUATIONS
# ============================================================
results = {}

# --- Single-language ---
print("\n Evaluating on single-language dataset...")
results["single_zero"] = evaluate_strategy(dataset_single, "zero-shot", build_zero_shot_prompt)
results["single_few"] = evaluate_strategy(dataset_single, "few-shot", build_few_shot_prompt, few_shot_examples)
results["single_instruct"] = evaluate_strategy(dataset_single, "instruction", custom_instruction_prompt)

# --- Multi-language ---
print("\n Evaluating on multi-language dataset...")
results["multi_zero"] = evaluate_strategy(dataset_multi, "zero-shot", build_zero_shot_prompt)
results["multi_few"] = evaluate_strategy(dataset_multi, "few-shot", build_few_shot_prompt, few_shot_examples)
results["multi_instruct"] = evaluate_strategy(dataset_multi, "instruction", custom_instruction_prompt)

# ============================================================
# SAVE RESULTS
# ============================================================
with open("results/prompting_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n ===== Final Summary =====")
for k, v in results.items():
    print(f"{k}: {v:.3f}")

print("\n Results saved to results/prompting_results.json")
