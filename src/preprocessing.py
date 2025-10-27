import pandas as pd
from datasets import Dataset, DatasetDict
import os

# ============================================================
# FILE PATHS FOR ALL LANGUAGES
# ============================================================
files = {
    "en": "!OriginalData/en-projections.tsv",
    "hu": "!OriginalData/hu-projections.tsv",
    "nl": "!OriginalData/nl-projections.tsv",
    "ro": "!OriginalData/ro-projections.tsv",
}

# ============================================================
# NUMBER → PLUTCHIK EMOTION MAPPING
# ============================================================
NUM_TO_EMO = {
    "1": "joy",
    "2": "trust",
    "3": "fear",
    "4": "surprise",
    "5": "sadness",
    "6": "disgust",
    "7": "anger",
    "8": "anticipation",
}

def convert_labels(label_str: str) -> str:
    """Convert comma-separated numbers into comma-separated emotion words."""
    numbers = [n.strip() for n in str(label_str).split(",")]
    emotions = [NUM_TO_EMO.get(n, n) for n in numbers if n]
    return ", ".join(emotions)

def load_tsv(path):
    """Load a TSV file and convert to simple text/label format."""
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels"])
    df["labels"] = df["labels"].apply(convert_labels)
    df.rename(columns={"labels": "label"}, inplace=True)
    return df

def split_dataset(ds, seed=42):
    """Split into train, validation, and test sets."""
    train_test = ds.train_test_split(test_size=0.2, seed=seed)
    valid_test = train_test["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict({
        "train": train_test["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"],
    })

# ============================================================
# PROCESS EACH LANGUAGE SEPARATELY
# ============================================================
combined_dataframes = []

for lang, path in files.items():
    print(f"Processing {lang}...")

    df = load_tsv(path)
    df["lang"] = lang
    combined_dataframes.append(df)

    # Convert to Hugging Face Dataset
    ds_lang = Dataset.from_pandas(df)

    # Split and save
    splits = split_dataset(ds_lang)
    out_dir = f"data/{lang}"
    os.makedirs(out_dir, exist_ok=True)

    for name, split in splits.items():
        outfile = os.path.join(out_dir, f"{name}.jsonl")
        split.to_json(outfile, orient="records", force_ascii=False, lines=True)
        print(f"  Saved {lang} {name} → {outfile}")

# ============================================================
# COMBINED DATASET
# ============================================================
combined_df = pd.concat(combined_dataframes, ignore_index=True)
combined_ds = Dataset.from_pandas(combined_df)
combined_splits = split_dataset(combined_ds)

out_dir = "data/combined"
os.makedirs(out_dir, exist_ok=True)

for name, split in combined_splits.items():
    outfile = os.path.join(out_dir, f"{name}.jsonl")
    split.to_json(outfile, orient="records", force_ascii=False, lines=True)
    print(f"  Saved combined {name} → {outfile}")

print("\n Simple-format data saved per language + combined dataset.")
