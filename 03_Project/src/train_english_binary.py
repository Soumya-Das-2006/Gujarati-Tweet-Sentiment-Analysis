"""
Fine-tune a transformer (default: distilbert-base-uncased) for binary
English sentiment (positive/negative) on the balanced Sentiment140 split
built by prepare_english_binary.py. Deliberately an easier task (binary,
English, large clean data) than the 3-class Gujarati work, aimed at
reliably clearing a high accuracy bar.

Usage:
    python src/train_english_binary.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "03_Project" / "data" / "processed_ann" / "sentiment140_binary_87"
RESULTS_OUT = ROOT_DIR / "06_Results"

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=48):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length", max_length=max_length)
        self.labels = [LABEL2ID[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=48)
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, model: {args.model_name}")

    model_slug = args.run_name or args.model_name.split("/")[-1]
    MODEL_OUT = ROOT_DIR / "01_Model" / "english_binary" / model_slug
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    train_ds = TweetDataset(train_df["clean_text"], train_df["sentiment"], tokenizer, args.max_length)
    val_ds = TweetDataset(val_df["clean_text"], val_df["sentiment"], tokenizer, args.max_length)
    test_ds = TweetDataset(test_df["clean_text"], test_df["sentiment"], tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=200,
        fp16=(device == "cuda"),
        report_to=[],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    model.save_pretrained(MODEL_OUT / "final_model")
    tokenizer.save_pretrained(MODEL_OUT / "final_model")

    test_output = trainer.predict(test_ds)
    test_preds = np.argmax(test_output.predictions, axis=1)
    test_labels = test_output.label_ids

    report = classification_report(test_labels, test_preds, target_names=["negative", "positive"], output_dict=True)
    cm = confusion_matrix(test_labels, test_preds).tolist()

    results = {
        "model_name": args.model_name,
        "task": "binary English sentiment (Sentiment140)",
        "test_accuracy": report["accuracy"],
        "test_report": report,
        "test_confusion_matrix": cm,
        "confusion_matrix_labels": ["negative", "positive"],
    }
    with open(RESULTS_OUT / f"english_binary_{model_slug}_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nTest accuracy: {report['accuracy']:.4f}")
    print(f"Saved model to {MODEL_OUT / 'final_model'} and results to {RESULTS_OUT / (model_slug + '_results.json')}")


if __name__ == "__main__":
    main()
