import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import os
import json


class PromptTrainer:
    """
    Evaluate prompting strategies (zero-shot, few-shot, and instruction-based)
    for emotion classification tasks using Mistral-family models.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        data_single: str = "data/en/test.jsonl",
        data_multi: str = "data/combined_simple/test.jsonl",
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
            data_single: Path to single-language test data
            data_multi: Path to multi-language test data
            max_new_tokens: Max tokens to generate per response
            device: "cuda" or "cpu" (auto-detected if None)
            output_dir: Directory for result JSON
            instruction_text: Base instruction text for the instruction prompt
            few_shot_intro: Introductory text preceding few-shot examples
            few_shot_examples: List of dicts [{'text': ..., 'label': ...}]
        """
        self.model_name = model_name
        self.data_single = data_single
        self.data_multi = data_multi
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

    def build_few_shot_prompt(
        self,
        text: str,
        examples: list[dict] | None = None,
        intro_text: str | None = None,
    ) -> str:
        """Build a few-shot prompt with examples and customizable intro."""
        examples = examples or self.few_shot_examples
        intro_text = intro_text or self.few_shot_intro
        demo_text = "\n".join(
            [f"Sentence: {ex['text']}\nEmotions: {ex['label']}" for ex in examples]
        )
        return (
            f"<s>[INST] {intro_text}\n"
            f"{demo_text}\n\nNow classify the emotions of: '{text}' [/INST]"
        )

    def build_instruction_prompt(
        self, text: str, custom_instruction: str | None = None
    ) -> str:
        """Build an instruction-based prompt with an optional custom instruction."""
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
        name: str,
        prompt_fn,
        few_shot_examples=None,
        few_shot_intro=None,
    ):
        preds, refs = [], []
        print(f"\nEvaluating strategy: {name} on {len(dataset)} samples")

        for example in tqdm(dataset):
            text = example["text"]
            label = example["label"].lower().strip()

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

        acc = sum([r in p for p, r in zip(preds, refs)]) / len(refs)
        print(f"Accuracy ({name}): {acc:.3f}")
        return acc

    # ============================================================
    #  LOAD DATASETS
    # ============================================================
    def load_datasets(self):
        print("\nLoading test datasets...")
        dataset_single = load_dataset("json", data_files=self.data_single)["train"]
        dataset_multi = load_dataset("json", data_files=self.data_multi)["train"]
        return dataset_single, dataset_multi

    # ============================================================
    #  RUN FULL EVALUATION
    # ============================================================
    def run(
        self,
        custom_instruction: str | None = None,
        few_shot_intro: str | None = None,
        few_shot_examples: list[dict] | None = None,
    ):
        dataset_single, dataset_multi = self.load_datasets()
        results = {
            "config": {
                "instruction_text": custom_instruction or self.instruction_text,
                "few_shot_intro": few_shot_intro or self.few_shot_intro,
                "few_shot_examples": few_shot_examples or self.few_shot_examples,
            }
        }

        # --- Single-language ---
        print("\nEvaluating on single-language dataset...")
        results["single_zero"] = self.evaluate_strategy(
            dataset_single, "zero-shot", self.build_zero_shot_prompt
        )
        results["single_few"] = self.evaluate_strategy(
            dataset_single,
            "few-shot",
            self.build_few_shot_prompt,
            few_shot_examples=few_shot_examples or self.few_shot_examples,
            few_shot_intro=few_shot_intro or self.few_shot_intro,
        )
        results["single_instruct"] = self.evaluate_strategy(
            dataset_single,
            "instruction",
            lambda text: self.build_instruction_prompt(text, custom_instruction),
        )

        # --- Multi-language ---
        print("\nEvaluating on multi-language dataset...")
        results["multi_zero"] = self.evaluate_strategy(
            dataset_multi, "zero-shot", self.build_zero_shot_prompt
        )
        results["multi_few"] = self.evaluate_strategy(
            dataset_multi,
            "few-shot",
            self.build_few_shot_prompt,
            few_shot_examples=few_shot_examples or self.few_shot_examples,
            few_shot_intro=few_shot_intro or self.few_shot_intro,
        )
        results["multi_instruct"] = self.evaluate_strategy(
            dataset_multi,
            "instruction",
            lambda text: self.build_instruction_prompt(text, custom_instruction),
        )

        # Save results
        result_path = os.path.join(self.output_dir, "prompting_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n===== Final Summary =====")
        for k, v in results.items():
            if k != "config":
                print(f"{k}: {v:.3f}")

        print(f"\nResults saved to: {result_path}")
        return results

