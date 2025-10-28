import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Mapping from numbers to emotions
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

# Function to convert numbers into emotions
def convert_labels(label_str: str) -> str:
    """Convert comma-separated numbers into comma-separated emotion words."""
    numbers = [n.strip() for n in label_str.split(",")]
    emotions = [NUM_TO_EMO.get(n, n) for n in numbers if n]  # skip empty
    return ", ".join(emotions)

# Load the dataset (replace the path with the actual TSV file path)
data_file = "!OriginalData/ro-projections.tsv"  # Modify with your actual file path

# Initialize emotion counter
emotion_counter = Counter()

# Read the TSV file into a DataFrame
df = pd.read_csv(data_file, sep="\t", header=None, names=["text", "labels"])

# Iterate through the dataset and count the emotions
for _, row in df.iterrows():
    labels = row["labels"]
    emotions = convert_labels(labels).split(", ")  # Handle multiple emotions
    
    for emo in emotions:
        emotion_counter[emo] += 1

# Plot the distribution of emotions
emotion_counts = dict(emotion_counter)
emotions = list(emotion_counts.keys())
counts = list(emotion_counts.values())

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(emotions, counts, color='skyblue')
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Emotion Distribution', fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("emotion_distribution.png")
