import os
import pandas as pd
from datasets import Dataset, DatasetDict

class EmotionDataPreprocessor:
    # Number → Plutchik emotion mapping
    NUM_TO_EMO = {
        "1": "joy",
        "2": "trust",
        "3": "fear",
        "4": "surprise",
        "5": "sadness",
        "6": "disgust",
        "7": "anger",
        "8": "anticipation"
    }

    def __init__(self, data_dir: str = "!OriginalData", output_dir: str = "data", files=None, seed=42, test_size=0.2):
        """
        Args:
            data_dir: directory where the input TSVs live
            output_dir: base directory to write per-language jsonl splits
            files: optional dict lang->relative filepath; defaults to given four languages
            seed: RNG seed for split reproducibility
            test_size: fraction for test split (validation is split from test)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        self.test_size = test_size

        self.files = files or {
            "en": os.path.join(data_dir, "en-projections.tsv"),
            "hu": os.path.join(data_dir, "hu-projections.tsv"),
            "nl": os.path.join(data_dir, "nl-projections.tsv"),
            "ro": os.path.join(data_dir, "ro-projections.tsv"),
        }

        self.results = {}

    @staticmethod
    def _to_str(x):
        return "" if pd.isna(x) else str(x)

    def convert_labels(self, label_str: str) -> str:
        """Convert comma-separated numbers into comma-separated emotion words."""
        
        s = self._to_str(label_str)
        numbers = [n.strip() for n in s.split(",") if n.strip()]
        emotions = [self.NUM_TO_EMO.get(n, n) for n in numbers]
        return ", ".join(emotions)

    def load_tsv(self, path: str) -> pd.DataFrame:
        """Load and preprocess a TSV file into a DataFrame suitable for LLM training."""
        
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels"])
        df["labels"] = df["labels"].apply(self._to_str).apply(self.convert_labels)
        # LLM-friendly single-field records
        df["text"] = df.apply(
            lambda r: (
                f"<|user|> Classify the emotions of: '{r['text']}'\n"
                f"<|assistant|> {r['labels']} \n<|endoftext|>"
            ),
            axis=1
        )
        # only keep the formatted text column
        df = df[["text"]]
        return df

    def split_dataset(self, ds: Dataset) -> DatasetDict:
        """Split a Dataset into train/validation/test sets."""
        
        train_test = ds.train_test_split(test_size=self.test_size, seed=self.seed)
        test_valid = train_test['test'].train_test_split(test_size=0.5, seed=self.seed)
        return DatasetDict({
            'train': train_test['train'],
            'validation': test_valid['train'],
            'test': test_valid['test']
        })

    def process_language(self, lang: str, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file for '{lang}': {path}")

        print(f"Processing {lang} from {path} ...")
        df = self.load_tsv(path)
        ds = Dataset.from_pandas(df)

        splits = self.split_dataset(ds)

        out_dir = os.path.join(self.output_dir, lang)
        os.makedirs(out_dir, exist_ok=True)

        saved = {}
        for split_name, split_data in splits.items():
            outfile = os.path.join(out_dir, f"{split_name}.jsonl")
            split_data.to_json(outfile, orient="records", force_ascii=False, lines=True)
            print(f"  Saved {split_name} → {outfile}")
            saved[split_name] = outfile

        self.results[lang] = saved

    def run(self):
        """Run preprocessing for all configured languages/files."""
        for lang, path in self.files.items():
            self.process_language(lang, path)
        return self.results