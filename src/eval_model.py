import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
from sklearn.metrics import f1_score, jaccard_score


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for generation."""
    
    def __init__(self, stop_sequence_ids):
        super().__init__()
        self.stop = stop_sequence_ids
        self.k = len(self.stop)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if input_ids.shape[1] < self.k:
            return False
        return torch.equal(
            input_ids[0, -self.k:],
            torch.tensor(self.stop, device=input_ids.device)
        )


class LoRAModelEvaluator:
    """Evaluator for LoRA-finetuned models on emotion classification tasks."""
    
    def __init__(
        self,
        base_model="mistralai/Mistral-7B-Instruct-v0.1",
        adapter_path="models/mistral_7b_lora_finetune",
        batch_size=4,
        max_new_tokens=128,
        stop_str="<|endoftext|>",
        allowed_labels=None,
        device=None,
    ):
        """
        Initialize the LoRA model evaluator.
        
        Args:
            base_model (str): Hugging Face model ID or local path for base model.
            adapter_path (str): Path to the LoRA adapter directory.
            batch_size (int): Number of prompts per generation batch.
            max_new_tokens (int): Maximum generation length.
            stop_str (str): Stop string marking end of assistant response.
            allowed_labels (list): List of valid emotion labels.
            device (str): "cuda" or "cpu". Auto-detected if None.
        """
        if allowed_labels is None:
            allowed_labels = [
                "joy", "trust", "fear", "surprise",
                "sadness", "disgust", "anger", "anticipation"
            ]
        
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.stop_str = stop_str
        self.allowed_labels = allowed_labels
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self._load_model()
        self._load_tokenizer()
        self._setup_stopping_criteria()
    
    def _load_model(self):
        """Load the base model and LoRA adapter."""
        bnb_config = None
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        print(f"Loading base model: {self.base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        
        print(f"Loading LoRA adapter: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()
    
    def _load_tokenizer(self):
        """Load and configure the tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.adapter_path, use_fast=True
            )
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model, use_fast=True
            )
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "</s>"
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
    
    def _setup_stopping_criteria(self):
        """Setup stopping criteria for generation."""
        stop_ids = self.tokenizer.encode(self.stop_str, add_special_tokens=False)
        self.stopper = StoppingCriteriaList([StopOnTokens(stop_ids)])
    
    def normalize_labels(self, s: str):
        """
        Parse and normalize emotion labels from a string.
        
        Args:
            s (str): Comma-separated label string.
        
        Returns:
            list: Sorted list of valid, unique labels.
        """
        labs = []
        for tok in s.split(","):
            lab = tok.strip().lower()
            if lab in self.allowed_labels and lab not in labs:
                labs.append(lab)
        return sorted(labs)
    
    def parse_text_record(self, rec_text: str):
        """
        Parse a text record to extract prompt and gold labels.
        
        Args:
            rec_text (str): Full text record containing prompt and labels.
        
        Returns:
            tuple: (prompt, gold_labels) or (None, None) if parsing fails.
        """
        if "<|assistant|>" not in rec_text:
            return None, None
        
        left, right = rec_text.split("<|assistant|>", 1)
        right = right.strip()
        
        if self.stop_str in right:
            right = right.split(self.stop_str, 1)[0]
        
        gold_raw = right.strip()
        prompt = left.strip() + "\n<|assistant|> "
        gold_labels = self.normalize_labels(gold_raw)
        
        return prompt, gold_labels
    
    def labels_to_multihot(self, labels):
        """Convert label list to multi-hot encoding."""
        return [1 if l in labels else 0 for l in self.allowed_labels]
    
    def decode_and_trim(self, full_ids):
        """Decode token IDs and extract assistant response."""
        txt = self.tokenizer.decode(full_ids, skip_special_tokens=False)
        
        if "<|assistant|>" in txt:
            txt = txt.split("<|assistant|>", 1)[1]
        
        if self.stop_str in txt:
            txt = txt.split(self.stop_str, 1)[0]
        
        return txt.strip()
    
    def predict_batch(self, prompts):
        """
        Generate predictions for a batch of prompts.
        
        Args:
            prompts (list): List of prompt strings.
        
        Returns:
            list: Generated text outputs.
        """
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        tok = {k: v.to(self.model.device) for k, v in tok.items()}
        tok.pop("token_type_ids", None)
        
        with torch.no_grad():
            out = self.model.generate(
                **tok,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=self.stopper,
            )
        
        gens = [self.decode_and_trim(out[i]) for i in range(len(prompts))]
        return gens
    
    def evaluate_language(self, data_root, lang):
        """
        Evaluate model on a specific language dataset.
        
        Args:
            data_root (str): Root directory containing language subfolders.
            lang (str): Language code.
        
        Returns:
            dict: Evaluation metrics for this language.
        """
        test_path = os.path.join(data_root, lang, "test.jsonl")
        assert os.path.exists(test_path), f"Missing {test_path}"
        
        prompts, gold_lists = [], []
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rec = obj.get("text", "")
                p, g = self.parse_text_record(rec)
                if p is None:
                    continue
                prompts.append(p)
                gold_lists.append(g)
        
        preds_lists = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc=f"Predict {lang}"):
            batch_prompts = prompts[i:i + self.batch_size]
            outs = self.predict_batch(batch_prompts)
            preds_lists.extend([self.normalize_labels(o) for o in outs])
        
        print("\n--- Sample Predictions for sanity check ---")
        for p, g, o in zip(prompts[:5], gold_lists[:5], preds_lists[:5]):
            print("\nPROMPT:", p)
            print("GOLD:", g)
            print("PRED:", o)
        print("---------------------------------------------------\n")
        
        y_true = np.array([self.labels_to_multihot(g) for g in gold_lists])
        y_pred = np.array([self.labels_to_multihot(p) for p in preds_lists])
        
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        jacc = jaccard_score(y_true, y_pred, average="micro", zero_division=0)
        
        return {
            "language": lang,
            "n_examples": len(prompts),
            "micro_f1": micro_f1,
            "jaccard_micro": jacc,
        }
    
    def evaluate(self, data_root="data", langs=("en",)):
        """
        Evaluate model on multiple language datasets.
        
        Args:
            data_root (str): Root directory containing language subfolders.
            langs (tuple): Languages to evaluate.
        
        Returns:
            dict: Summary report with per-language and macro-averaged scores.
        """
        reports = [self.evaluate_language(data_root, lang) for lang in langs]
        total = sum(r["n_examples"] for r in reports)
        mf1 = np.mean([r["micro_f1"] for r in reports])
        mj = np.mean([r["jaccard_micro"] for r in reports])
        
        summary = {
            "reports": reports,
            "macro_micro_f1": mf1,
            "macro_jaccard": mj,
            "total_examples": total,
        }
        
        print("\n=== Evaluation Summary ===")
        for r in reports:
            print(
                f"{r['language'].upper():>3} | N={r['n_examples']:4d} | "
                f"micro-F1={r['micro_f1']:.4f} | Jaccard={r['jaccard_micro']:.4f}"
            )
        print(
            f"\nMacro-avg over {len(reports)} languages (N={total}): "
            f"micro-F1={mf1:.4f} | Jaccard={mj:.4f}"
        )
        
        return summary