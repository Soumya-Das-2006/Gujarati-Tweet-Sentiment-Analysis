"""
Combine classical ML, MuRIL, and ANN results into one comparison table (for
the paper) plus a bar chart.

IMPORTANT CAVEAT: classical ML and MuRIL were evaluated on GujaratiTweetsData
(machine-translated from English), while the ANN was fine-tuned and evaluated
on Raw_NLP_Data (real native Gujarati, hate-speech labels remapped to
sentiment). These are DIFFERENT test sets with different difficulty and class
balance - accuracy numbers across the "dataset" boundary are not directly
comparable, only within the same dataset group.

Usage:
    python src/compare_results.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "06_Results"
CLASSICAL_RESULTS = RESULTS_DIR / "classical_ml_results.json"
MURIL_RESULTS = ROOT_DIR / "01_Model" / "muril" / "muril_results.json"
ANN_RESULTS = RESULTS_DIR / "ann_results.json"
ENGLISH_BINARY_RESULTS = RESULTS_DIR / "english_binary_bertweet-base-v2_results.json"

TRANSLATED_DATASET = "GujaratiTweetsData (machine-translated)"
NATIVE_DATASET = "Raw_NLP_Data (native Gujarati)"
ENGLISH_DATASET = "Sentiment140 (English, binary)"


def main():
    rows = []

    if CLASSICAL_RESULTS.exists():
        with open(CLASSICAL_RESULTS, encoding="utf-8") as f:
            classical = json.load(f)
        for name, r in classical.items():
            report = r["test_report"]
            rows.append({
                "model": name,
                "dataset": TRANSLATED_DATASET,
                "val_accuracy": r.get("val_accuracy"),
                "test_accuracy": r.get("test_accuracy"),
                "macro_f1": report.get("macro avg", {}).get("f1-score"),
                "positive_recall": report.get("positive", {}).get("recall"),
            })
    else:
        print(f"Missing {CLASSICAL_RESULTS} - run classical_ml.py first.")

    if MURIL_RESULTS.exists():
        with open(MURIL_RESULTS, encoding="utf-8") as f:
            muril = json.load(f)
        report = muril["test_report"]
        rows.append({
            "model": "muril",
            "dataset": TRANSLATED_DATASET,
            "val_accuracy": muril.get("val_accuracy"),
            "test_accuracy": muril.get("test_accuracy"),
            "macro_f1": report.get("macro avg", {}).get("f1-score"),
            "positive_recall": report.get("positive", {}).get("recall"),
        })
    else:
        print(f"Missing {MURIL_RESULTS} - run train_muril.py first.")

    if ANN_RESULTS.exists():
        with open(ANN_RESULTS, encoding="utf-8") as f:
            ann = json.load(f)
        report = ann["test_report"]
        rows.append({
            "model": "ann_bilstm",
            "dataset": NATIVE_DATASET,
            "val_accuracy": ann.get("stage2_gujarati_val_accuracy"),
            "test_accuracy": ann.get("test_accuracy"),
            "macro_f1": report.get("macro avg", {}).get("f1-score"),
            "positive_recall": report.get("positive", {}).get("recall"),
        })
    else:
        print(f"Missing {ANN_RESULTS} - run train_ann.py first.")

    if ENGLISH_BINARY_RESULTS.exists():
        with open(ENGLISH_BINARY_RESULTS, encoding="utf-8") as f:
            eng = json.load(f)
        report = eng["test_report"]
        rows.append({
            "model": "bertweet_binary_english",
            "dataset": ENGLISH_DATASET,
            "val_accuracy": None,
            "test_accuracy": eng.get("test_accuracy"),
            "macro_f1": report.get("macro avg", {}).get("f1-score"),
            "positive_recall": report.get("positive", {}).get("recall"),
        })
    else:
        print(f"Missing {ENGLISH_BINARY_RESULTS} - run train_english_binary.py first.")

    if not rows:
        print("No results found yet.")
        return

    df = pd.DataFrame(rows).sort_values(["dataset", "test_accuracy"], ascending=[True, False])
    print(df.to_string(index=False))
    print(
        "\nNOTE: rows are grouped by 'dataset' - only compare test_accuracy "
        "within the same dataset group, not across groups (different test sets)."
    )
    df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {TRANSLATED_DATASET: "#2563eb", NATIVE_DATASET: "#1e8e3e", ENGLISH_DATASET: "#d97706"}
    bar_colors = [colors[d] for d in df["dataset"]]
    ax.bar(df["model"], df["test_accuracy"], color=bar_colors)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Gujarati Sentiment Models - grouped by evaluation dataset (colors)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_chart.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'comparison_table.csv'} and {RESULTS_DIR / 'comparison_chart.png'}")


if __name__ == "__main__":
    main()
