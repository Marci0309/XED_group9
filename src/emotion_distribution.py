import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

class EmotionDistribution:
    """Class for loading a TSV dataset, converting numeric labels to emotions,
    counting their occurrences, and plotting the distribution."""

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

    def __init__(self, data_file: str, output_path: str = "emotion_distribution.png"):
        """
        Args:
            data_file (str): Path to the input TSV file.
            output_path (str): Path to save the resulting plot image.
        """
        self.data_file = data_file
        self.output_path = output_path
        self.df = None
        self.emotion_counts = None

    def convert_labels(self, label_str: str) -> str:
        """Convert comma-separated numbers into comma-separated emotion words."""
        numbers = [n.strip() for n in str(label_str).split(",")]
        emotions = [self.NUM_TO_EMO.get(n, n) for n in numbers if n]  # skip empty
        return ", ".join(emotions)

    def load_data(self):
        """Load the TSV file into a pandas DataFrame."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"File not found: {self.data_file}")
        self.df = pd.read_csv(self.data_file, sep="\t", header=None, names=["text", "labels"])
        print(f"Loaded dataset: {self.data_file} ({len(self.df)} rows)")
        return self.df

    def count_emotions(self):
        """Count the frequency of each emotion in the dataset."""
        if self.df is None:
            self.load_data()

        emotion_counter = Counter()

        for _, row in self.df.iterrows():
            emotions = self.convert_labels(row["labels"]).split(", ")
            for emo in emotions:
                if emo:
                    emotion_counter[emo] += 1

        self.emotion_counts = dict(emotion_counter)
        print(f"Counted {len(self.emotion_counts)} unique emotions")
        return self.emotion_counts

    def plot_distribution(self, save: bool = True, show: bool = False):
        """Plot the emotion distribution."""
        if self.emotion_counts is None:
            self.count_emotions()

        emotions = list(self.emotion_counts.keys())
        counts = list(self.emotion_counts.values())

        plt.figure(figsize=(10, 6))
        plt.bar(emotions, counts, color='skyblue')
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Emotion Distribution: {os.path.basename(self.data_file)}', fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save:
            plt.savefig(self.output_path)
            print(f"Saved plot to: {self.output_path}")

        if show:
            plt.show()

        plt.close()

    def run(self, show_plot: bool = False):
        """Convenience method to run the full process end-to-end."""
        self.load_data()
        self.count_emotions()
        self.plot_distribution(show=show_plot)
        return self.emotion_counts
