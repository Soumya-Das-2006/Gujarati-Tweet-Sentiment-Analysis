"""
Fine-tune google/muril-base-cased for 3-class Gujarati tweet sentiment
(positive/negative/neutral), on the SAME train/val/test split used by
classical_ml.py so results are directly comparable.

Uses the GPU automatically if available (this machine has an RTX 4050 - make
sure you installed the CUDA build of torch, see requirements.txt / README).

Usage:
    python src/train_muril.py --datadir data/processed --outdir models/muril --epochs 15
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
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_length
        )
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
    parser.add_argument("--datadir", default="data/processed")
    parser.add_argument("--outdir", default="models/muril")
    parser.add_argument("--model_name", default="google/muril-base-cased")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected - fine-tuning MuRIL on CPU will be very slow.")

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(datadir / "train.csv")
    val_df = pd.read_csv(datadir / "val.csv")
    test_df = pd.read_csv(datadir / "test.csv")

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, keep_accents=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    train_ds = TweetDataset(train_df["clean_text"], train_df["sentiment"], tokenizer, args.max_length)
    val_ds = TweetDataset(val_df["clean_text"], val_df["sentiment"], tokenizer, args.max_length)
    test_ds = TweetDataset(test_df["clean_text"], test_df["sentiment"], tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(outdir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=(device == "cuda"),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    model.save_pretrained(outdir / "final_model")
    tokenizer.save_pretrained(outdir / "final_model")

    test_output = trainer.predict(test_ds)
    test_preds = np.argmax(test_output.predictions, axis=1)
    test_labels = test_output.label_ids

    report = classification_report(
        test_labels, test_preds, target_names=["negative", "neutral", "positive"], output_dict=True
    )
    cm = confusion_matrix(test_labels, test_preds).tolist()

    val_output = trainer.predict(val_ds)
    val_preds = np.argmax(val_output.predictions, axis=1)
    val_labels = val_output.label_ids
    val_report = classification_report(val_labels, val_preds, output_dict=True)

    results = {
        "val_accuracy": val_report["accuracy"],
        "test_accuracy": report["accuracy"],
        "test_report": report,
        "test_confusion_matrix": cm,
        "confusion_matrix_labels": ["negative", "neutral", "positive"],
    }
    with open(outdir / "muril_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nTest accuracy: {report['accuracy']:.4f}")
    print(f"Saved fine-tuned model to {outdir / 'final_model'} and results to {outdir / 'muril_results.json'}")


if __name__ == "__main__":
    main()
