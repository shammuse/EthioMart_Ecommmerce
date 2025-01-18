from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
import evaluate
import numpy as np
import torch

MODEL_DIR = "../models/fine_tuned/"
DATASET_DIR = "../data/labeled/labeled_data.conll"

def load_conll_data(file_path):
    """
    Load a custom .conll file and parse it into a dataset.
    """
    tokens, ner_tags, current_tokens, current_tags = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line == "":
                if current_tokens:
                    tokens.append(current_tokens)
                    ner_tags.append(current_tags)
                    current_tokens, current_tags = [], []
            else:
                splits = line.split()
                current_tokens.append(splits[0])
                current_tags.append(splits[1])
        if current_tokens:  # Add last sentence
            tokens.append(current_tokens)
            ner_tags.append(current_tags)
    return Dataset.from_dict({"tokens": tokens, "ner_tags": ner_tags})

def align_predictions(predictions, label_ids):
    """
    Align predictions with true labels.
    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_label_list, preds_list = [], []

    for i in range(batch_size):
        out_label_list.append([label for label in label_ids[i] if label != -100])
        preds_list.append([pred for label, pred in zip(label_ids[i], preds[i]) if label != -100])

    return preds_list, out_label_list

def evaluate_model():
    """
    Evaluate the fine-tuned model on a labeled dataset.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Load dataset
    dataset = load_conll_data(DATASET_DIR)
    metric = evaluate.load("seqeval")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['tokens'], truncation=True, is_split_into_words=True, padding=True, max_length=512
        )
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Tokenize and align labels
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    # Prepare for evaluation
    all_predictions, all_labels = [], []

    for batch in tokenized_dataset:
        inputs = {key: torch.tensor(val).unsqueeze(0) for key, val in batch.items() if key in tokenizer.model_input_names}
        labels = batch["labels"]
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = logits.detach().cpu().numpy()
        labels = np.array(labels)

        preds_list, out_label_list = align_predictions(predictions, labels)
        all_predictions.extend(preds_list)
        all_labels.extend(out_label_list)

    # Compute metrics
    metrics = metric.compute(predictions=all_predictions, references=all_labels)
    print(metrics)

if __name__ == "__main__":
    evaluate_model()
