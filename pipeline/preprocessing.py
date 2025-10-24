import pandas as pd
from datasets import Dataset, DatasetDict
import os

# File paths for all languages
files = {
    "en": "!OriginalData/en-projections.tsv",
    "hu": "!OriginalData/hu-projections.tsv",
    "nl": "!OriginalData/nl-projections.tsv",
    "ro": "!OriginalData/ro-projections.tsv"
}

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

def convert_labels(label_str: str) -> str:
    """Convert comma-separated numbers into comma-separated emotion words."""
    numbers = [n.strip() for n in label_str.split(",")]
    emotions = [NUM_TO_EMO.get(n, n) for n in numbers if n]  # skip empty
    return ", ".join(emotions)

def load_tsv(path, lang):
    """Load a TSV file, process the labels and convert text to LLM-friendly format."""
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels"])
    df["labels"] = df["labels"].apply(lambda x: str(x))  # keep as string
    # Convert numbers to emotion words
    df["labels"] = df["labels"].apply(convert_labels)
    # Turn into LLM-friendly format, exclude 'labels' from final output
    df["text"] = df.apply(
        lambda r: f"<|user|> Classify the emotions of: '{r['text']}'\n<|assistant|> {r['labels']} \n<|endoftext|>",
        axis=1
    )
    df = df[["text"]]  # Keep only the 'text' column, exclude 'labels'
    return df

def split_dataset(ds, seed=42):
    """Split dataset into train, validation, and test sets."""
    train_test = ds.train_test_split(test_size=0.2, seed=seed)
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict({
        'train': train_test['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })

# Load and combine all languages
all_data = []

for lang, path in files.items():
    print(f"Processing {lang}...")
    df = load_tsv(path, lang)
    all_data.append(df)

# Combine data from all languages into a single DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Convert to Hugging Face Dataset
ds = Dataset.from_pandas(combined_df)

# Split combined dataset into train, validation, and test sets
splits = split_dataset(ds)

# Save split datasets
out_dir = "data/combined"
os.makedirs(out_dir, exist_ok=True)

for split_name, split_data in splits.items():
    outfile = os.path.join(out_dir, f"{split_name}.jsonl")
    split_data.to_json(outfile, orient="records", force_ascii=False, lines=True)
    print(f"  Saved {split_name} to {outfile}")

print("LLM-ready combined data saved inside data/combined folder.")
