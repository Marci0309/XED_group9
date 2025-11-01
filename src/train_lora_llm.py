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


class LoraFineTuner:
    """Fine-tunes multiple Mistral-size LLMs using LoRA adapters and 4-bit quantization."""

    def __init__(
        self,
        data_paths: dict,
        models: dict,
        output_dir: str = "models",
        num_epochs: int = 3,
        batch_size: int = 1,
        grad_accum_steps: int = 16,
        learning_rate: float = 2e-4,
        device_ids: str = "0",
    ):
        """
        Args:
            data_paths (dict): dict with keys train/validation/test → file paths.
            models (dict): dict of {tag: model_name} to train.
            output_dir (str): Base directory to save all model outputs.
            num_epochs (int): Number of epochs to train each model.
            batch_size (int): Per-device batch size.
            grad_accum_steps (int): Gradient accumulation steps.
            learning_rate (float): Learning rate for fine-tuning.
            device_ids (str): GPUs to use, e.g. "0" or "0,1".
        """
        self.data_paths = data_paths
        self.models = models
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.learning_rate = learning_rate

        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        warnings.filterwarnings("ignore", message="Detected kernel version")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.dataset = self._load_dataset()

    # ============================================================
    #  DATASET
    # ============================================================
    def _load_dataset(self):
        """Load dataset from JSONL paths (train/validation/test)."""
        print("Loading dataset...")
        dataset = load_dataset("json", data_files=self.data_paths)
        print("Dataset loaded successfully.")
        return dataset

    @staticmethod
    def _tokenize_function(examples, tokenizer):
        """Combine text and labels into model-ready sequences."""
        full_texts = [
            f"Classify the emotions of: {t}\nAnswer: {l}"
            for t, l in zip(examples["text"], examples["label"])
        ]
        return tokenizer(full_texts, truncation=True, padding="max_length", max_length=512)

    # ============================================================
    #  SINGLE MODEL TRAINING
    # ============================================================
    def _train_single_model(self, tag: str, model_name: str):
        print(f"\n==============================")
        print(f" Fine-tuning model: {model_name}")
        print(f"==============================")

        out_dir = os.path.join(self.output_dir, f"{tag}_lora_finetune")
        os.makedirs(out_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        tokenized = self.dataset.map(
            lambda x: self._tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=["text", "label"],
        )

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
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        args = TrainingArguments(
            output_dir=out_dir,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=400,
            logging_steps=50,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum_steps,
            num_train_epochs=self.num_epochs,
            warmup_ratio=0.05,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            tf32=False,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            save_total_limit=2,
            report_to="none",
            dataloader_num_workers=2,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        trainer.train()

        # Save LoRA adapter
        model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)

        # Evaluate
        results = trainer.evaluate(tokenized["test"])
        print(f"\nTest Results for {tag}: {results}")

        return results

    # ============================================================
    #  RUN ALL MODELS
    # ============================================================
    def run_all(self):
        """Fine-tune all models in the dictionary."""
        all_results = {}
        for tag, model_name in self.models.items():
            results = self._train_single_model(tag, model_name)
            all_results[tag] = results
        print("\nAll models fine-tuned successfully!")
        return all_results