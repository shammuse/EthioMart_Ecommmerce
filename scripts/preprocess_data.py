import os
import json
import pandas as pd
import re

RAW_DIR = "../data/raw/"
PROCESSED_DIR = "../data/processed/"

def preprocess_message(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.strip()
    return text

def preprocess_data():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for file in os.listdir(RAW_DIR):
        with open(f"{RAW_DIR}/{file}", encoding="utf-8") as f:
            raw_data = json.load(f)

        processed_data = []
        for msg in raw_data:
            if msg['text']:
                processed_data.append({
                    'id': msg['id'],
                    'text': preprocess_message(msg['text']),
                    'date': msg['date'],
                    'sender': msg['sender']
                })

        df = pd.DataFrame(processed_data)
        df.to_csv(f"{PROCESSED_DIR}/{file.split('.')[0]}.csv", index=False)

if __name__ == "__main__":
    preprocess_data()
