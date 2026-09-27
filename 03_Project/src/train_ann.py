"""
Two-stage ANN (Embedding + BiLSTM) sentiment classifier:

Stage 1 - pretrain on Sentiment140 (1.6M English tweets, binary sentiment)
    to learn general tweet-sentiment patterns at scale.
Stage 2 - fine-tune the same backbone (swap in a 3-way head) on real native
    Gujarati text (Raw_NLP_Data.csv, hate-speech labels remapped to
    negative/neutral/positive), which is the actual target task.

Usage:
    python src/train_ann.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from ann_model import BiLSTMClassifier, TextDataset, build_vocab

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "03_Project" / "data" / "processed_ann"
MODEL_OUT = ROOT_DIR / "01_Model" / "ann"
RESULTS_OUT = ROOT_DIR / "06_Results"

S140_LABELS = {"negative": 0, "positive": 1}
GUJ_LABELS = {"negative": 0, "neutral": 1, "positive": 2}


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)


def train_epochs(model, train_loader, val_loader, device, epochs, lr, class_weights=None, patience=3):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        preds, labels = evaluate(model, val_loader, device)
        val_acc = accuracy_score(labels, preds)
        print(f"  epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=40000)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--stage1_epochs", type=int, default=4)
    parser.add_argument("--stage2_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    s140_train = pd.read_csv(DATA_DIR / "sentiment140" / "train.csv")
    s140_val = pd.read_csv(DATA_DIR / "sentiment140" / "val.csv")
    guj_train = pd.read_csv(DATA_DIR / "gujarati" / "train.csv")
    guj_val = pd.read_csv(DATA_DIR / "gujarati" / "val.csv")
    guj_test = pd.read_csv(DATA_DIR / "gujarati" / "test.csv")

    # ---- Shared vocabulary built from BOTH corpora, so the embedding table
    # learned in stage 1 stays meaningful when we fine-tune in stage 2. ----
    print("Building shared vocabulary...")
    vocab = build_vocab(
        list(s140_train["clean_text"]) + list(guj_train["clean_text"]), max_vocab_size=args.vocab_size
    )
    print(f"Vocab size: {len(vocab)}")
    with open(MODEL_OUT / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    model = BiLSTMClassifier(
        vocab_size=len(vocab), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_classes=2
    ).to(device)

    # ---- Stage 1: pretrain on Sentiment140 ----
    print("\n=== Stage 1: pretraining on Sentiment140 ===")
    s1_train_ds = TextDataset(s140_train["clean_text"], s140_train["sentiment"], vocab, S140_LABELS, args.max_len)
    s1_val_ds = TextDataset(s140_val["clean_text"], s140_val["sentiment"], vocab, S140_LABELS, args.max_len)
    s1_train_loader = DataLoader(s1_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    s1_val_loader = DataLoader(s1_val_ds, batch_size=args.batch_size, num_workers=0)

    model, stage1_val_acc = train_epochs(model, s1_train_loader, s1_val_loader, device, args.stage1_epochs, args.lr)
    print(f"Stage 1 best val accuracy: {stage1_val_acc:.4f}")
    torch.save(model.state_dict(), MODEL_OUT / "stage1_pretrained.pt")

    # ---- Stage 2: fine-tune on real Gujarati data (3-class) ----
    print("\n=== Stage 2: fine-tuning on native Gujarati data ===")
    model.replace_head(num_classes=3).to(device)

    class_weights_arr = compute_class_weight(
        "balanced", classes=np.array([0, 1, 2]), y=guj_train["sentiment"].map(GUJ_LABELS).values
    )
    class_weights = torch.tensor(class_weights_arr, dtype=torch.float)
    print(f"Class weights (negative/neutral/positive): {class_weights_arr}")

    s2_train_ds = TextDataset(guj_train["clean_text"], guj_train["sentiment"], vocab, GUJ_LABELS, args.max_len)
    s2_val_ds = TextDataset(guj_val["clean_text"], guj_val["sentiment"], vocab, GUJ_LABELS, args.max_len)
    s2_test_ds = TextDataset(guj_test["clean_text"], guj_test["sentiment"], vocab, GUJ_LABELS, args.max_len)
    s2_train_loader = DataLoader(s2_train_ds, batch_size=64, shuffle=True, num_workers=0)
    s2_val_loader = DataLoader(s2_val_ds, batch_size=64, num_workers=0)
    s2_test_loader = DataLoader(s2_test_ds, batch_size=64, num_workers=0)

    model, stage2_val_acc = train_epochs(
        model, s2_train_loader, s2_val_loader, device, args.stage2_epochs, args.lr, class_weights=class_weights
    )
    print(f"Stage 2 best val accuracy: {stage2_val_acc:.4f}")

    # ---- Final evaluation on held-out Gujarati test set ----
    preds, labels = evaluate(model, s2_test_loader, device)
    id2label = {v: k for k, v in GUJ_LABELS.items()}
    report = classification_report(labels, preds, target_names=["negative", "neutral", "positive"], output_dict=True)
    cm = confusion_matrix(labels, preds).tolist()
    test_acc = accuracy_score(labels, preds)
    print(f"\nFinal Gujarati test accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), MODEL_OUT / "final_model.pt")
    with open(MODEL_OUT / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": len(vocab),
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "max_len": args.max_len,
                "label2id": GUJ_LABELS,
            },
            f,
        )

    results = {
        "stage1_sentiment140_val_accuracy": stage1_val_acc,
        "stage2_gujarati_val_accuracy": stage2_val_acc,
        "test_accuracy": test_acc,
        "test_report": report,
        "test_confusion_matrix": cm,
        "confusion_matrix_labels": ["negative", "neutral", "positive"],
    }
    with open(RESULTS_OUT / "ann_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model to {MODEL_OUT} and results to {RESULTS_OUT / 'ann_results.json'}")


if __name__ == "__main__":
    main()
