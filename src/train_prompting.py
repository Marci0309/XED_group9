import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import os
import json
from sklearn.metrics import f1_score, jaccard_score
import numpy as np
import shutil


class PromptTrainer:
    """
    Evaluate prompting strategies (zero-shot, few-shot, and instruction-based)
    for emotion classification tasks across multiple languages using Mistral-family models.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        data_dirs: dict | None = None,
        max_new_tokens: int = 50,
        device: str | None = None,
        output_dir: str = "results",
        instruction_text: str = (
            "You are a precise emotion classifier. Identify all emotions in the "
            "sentence based on Plutchik's 8 basic emotions and return them as "
            "comma-separated values from: joy, trust, fear, surprise, sadness, "
            "disgust, anger, anticipation."
        ),
        few_shot_intro: str = "You are an emotion classifier. Here are examples:",
        few_shot_examples: list[dict] | None = None,
    ):
        """
        Args:
            model_name: Hugging Face model name (e.g., Mistral-7B)
            data_dirs: dict with language keys mapping to test file paths, e.g.
                       {"en": "data/en/test.jsonl", "nl": "data/nl/test.jsonl"}
            max_new_tokens: Max tokens to generate per response
            device: "cuda" or "cpu" (auto-detected if None)
            output_dir: Directory for result JSONs
            instruction_text: Instruction text for instruction-based prompting
            few_shot_intro: Introductory text preceding few-shot examples
            few_shot_examples: List of dicts [{'text': ..., 'label': ...}]
        """
        self.model_name = model_name
        self.data_dirs = data_dirs or {"en": "data/en/test.jsonl"}
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.instruction_text = instruction_text
        self.few_shot_intro = few_shot_intro
        self.few_shot_examples = few_shot_examples or [
            {"text": "I can't believe this happened!", "label": "surprise"},
            {"text": "I'm so happy for you!", "label": "joy"},
            {"text": "This is so unfair.", "label": "anger"},
        ]
        
        # Clear Hugging Face cache to free up space in between prompting runs
        hf_cache = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        if os.path.exists(hf_cache):
            print(f"Clearing Hugging Face cache at {hf_cache}")
            shutil.rmtree(hf_cache, ignore_errors=True)

        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Using device: {self.device}")
        print(f"Loading model: {self.model_name}")

        self._load_model()
        self.metric = evaluate.load("accuracy")

    # ============================================================
    #  LOAD MODEL & TOKENIZER
    # ============================================================
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    # ============================================================
    #  PROMPT BUILDERS
    # ============================================================
    def build_zero_shot_prompt(self, text: str) -> str:
        return f"<s>[INST] Classify the emotions of: '{text}' [/INST]"

    def build_few_shot_prompt(self, text: str, examples=None, intro_text=None) -> str:
        examples = examples or self.few_shot_examples
        intro_text = intro_text or self.few_shot_intro
        demo_text = "\n".join(
            [f"Sentence: {ex['text']}\nEmotions: {ex['label']}" for ex in examples]
        )
        return (
            f"<s>[INST] {intro_text}\n"
            f"{demo_text}\n\nNow classify the emotions of: '{text}' [/INST]"
        )

    def build_instruction_prompt(self, text: str, custom_instruction=None) -> str:
        instruction = custom_instruction or self.instruction_text
        return f"<s>[INST] {instruction}\n\nSentence: '{text}' [/INST]"

    # ============================================================
    #  GENERATE MODEL RESPONSE
    # ============================================================
    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )
        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip().lower()

    # ============================================================
    #  EVALUATION FUNCTION
    # ============================================================
    def evaluate_strategy(
        self,
        dataset,
        name,
        prompt_fn,
        few_shot_examples=None,
        few_shot_intro=None,
    ):
        """
        Evaluate a prompting strategy using micro-F1 and Jaccard similarity.
        Supports multi-label emotion classification.
        """

        preds, refs = [], []
        all_labels = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
        print(f"\nEvaluating strategy: {name} on {len(dataset)} samples")

        for example in tqdm(dataset):
            raw = example["text"]
            try:
                if "<|assistant|>" in raw:
                    user_part, assistant_part = raw.split("<|assistant|>", 1)
                else:
                    user_part, assistant_part = raw, ""

                if "Classify the emotions of:" in user_part:
                    text = user_part.split("Classify the emotions of:")[-1]
                    text = text.replace("<|user|>", "").strip(" :'\n")
                else:
                    text = user_part.strip()

                label = (
                    assistant_part.split("<|endoftext|>")[0]
                    .strip()
                    .lower()
                    .replace("\n", "")
                )
            except Exception as e:
                print(f"⚠️ Could not parse example: {raw[:80]}... ({e})")
                text, label = raw, "unknown"

            if name == "few-shot":
                prompt = self.build_few_shot_prompt(
                    text,
                    examples=few_shot_examples,
                    intro_text=few_shot_intro,
                )
            else:
                prompt = prompt_fn(text)

            pred = self.generate_response(prompt)
            preds.append(pred)
            refs.append(label)

        # helper function to compute multi-label binary vectors
        def to_binary_vector(label_str):
            label_set = set([x.strip() for x in label_str.split(",") if x.strip()])
            return [1 if lbl in label_set else 0 for lbl in all_labels]

        y_true = np.array([to_binary_vector(r) for r in refs])
        y_pred = np.array([to_binary_vector(p) for p in preds])

        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        jaccard = jaccard_score(y_true, y_pred, average="samples", zero_division=0)

        print(f"Micro-F1 ({name}): {micro_f1:.3f}")
        print(f"Jaccard  ({name}): {jaccard:.3f}")

        return {"micro_f1": float(micro_f1), "jaccard": float(jaccard)}


    # ============================================================
    #  RUN ON ONE DATASET
    # ============================================================
    def _run_on_dataset(self, dataset, lang, custom_instruction=None, few_shot_intro=None, few_shot_examples=None):
        results = {}
        results["zero-shot"] = self.evaluate_strategy(dataset, "zero-shot", self.build_zero_shot_prompt)
        results["few-shot"] = self.evaluate_strategy(
            dataset,
            "few-shot",
            self.build_few_shot_prompt,
            few_shot_examples=few_shot_examples or self.few_shot_examples,
            few_shot_intro=few_shot_intro or self.few_shot_intro,
        )
        results["instruction"] = self.evaluate_strategy(
            dataset,
            "instruction",
            lambda text: self.build_instruction_prompt(text, custom_instruction),
        )
        
        result_path = os.path.join(self.output_dir, f"prompt_results_{lang}.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f" Saved results for {lang} → {result_path}")
        return results

    # ============================================================
    #  RUN FULL MULTILINGUAL EVALUATION
    # ============================================================
    def run(self, custom_instruction=None, few_shot_intro=None, few_shot_examples=None):
        all_results = {}
        for lang, path in self.data_dirs.items():
            print(f"\n Evaluating on {lang.upper()} dataset: {path}")
            dataset = load_dataset("json", data_files=path)["train"]
            results = self._run_on_dataset(
                dataset,
                lang,
                custom_instruction,
                few_shot_intro,
                few_shot_examples,
            )
            all_results[lang] = results

        # Save combined summary
        summary_path = os.path.join(self.output_dir, "prompting_results_all.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n All language results saved to: {summary_path}")
        return all_results
