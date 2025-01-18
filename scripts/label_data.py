import pandas as pd
import os

PROCESSED_DIR = "../data/processed/"
LABELED_DIR = "../data/labeled/"

def label_data():
    os.makedirs(LABELED_DIR, exist_ok=True)
    df = pd.read_csv(f"{PROCESSED_DIR}/aradabrand2.csv")  # Use one dataset for labeling
    with open(f"{LABELED_DIR}/labeled_data.conll", 'w', encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = row['text']
            for token in text.split():
                # Add your logic for labeling
                label = "O"  # Replace with actual logic
                f.write(f"{token}\t{label}\n")
            f.write("\n")

if __name__ == "__main__":
    label_data()
