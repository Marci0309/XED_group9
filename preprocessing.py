import pandas as pd
from datasets import Dataset, DatasetDict
import os

# file paths
files = {
    "en": "Data/!OriginalData/en-projections.tsv",
    "hu": "Data/!OriginalData/hu-projections.tsv",
    "nl": "Data/!OriginalData/nl-projections.tsv",
    "ro": "Data/!OriginalData/ro-projections.tsv"
}

# number → Plutchik emotion mapping
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

def convert_labels(label_str: str) -> str:
    """Convert comma-separated numbers into comma-separated emotion words."""
    numbers = [n.strip() for n in label_str.split(",")]
    emotions = [NUM_TO_EMO.get(n, n) for n in numbers if n]  # skip empty
    return ", ".join(emotions)

def load_tsv(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels"])
    df["labels"] = df["labels"].apply(lambda x: str(x))  # keep as string
    # convert numbers to emotion words
    df["labels"] = df["labels"].apply(convert_labels)
    # turn into LLM-friendly format
    df["text"] = df.apply(
        lambda r: f"<|user|> Classify the emotions of: '{r['text']}'\n<|assistant|> {r['labels']}",
        axis=1
    )
    df = df[["text"]]  # keep only text field
    return df

def split_dataset(ds, seed=42):
    train_test = ds.train_test_split(test_size=0.2, seed=seed)
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict({
        'train': train_test['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })

for lang, path in files.items():
    print(f"Processing {lang}...")

    df = load_tsv(path)
    ds = Dataset.from_pandas(df)
    splits = split_dataset(ds)

    out_dir = os.path.join("data", lang)
    os.makedirs(out_dir, exist_ok=True)

    for split_name, split_data in splits.items():
        outfile = os.path.join(out_dir, f"{split_name}.jsonl")
        split_data.to_json(outfile, orient="records", force_ascii=False, lines=True)
        print(f"  Saved {split_name} to {outfile}")

print("LLM-ready data saved inside Data/<language>/ folders.")

