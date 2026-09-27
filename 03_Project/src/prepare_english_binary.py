"""
Build a balanced binary (positive/negative) English sentiment split from the
already-cleaned Sentiment140 data, for fine-tuning a transformer aimed at
87%+ accuracy - a deliberately easier task than 3-class Gujarati (binary,
English, large clean dataset) to reliably hit that bar.

Usage:
    python src/prepare_english_binary.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "03_Project" / "data" / "processed_ann" / "sentiment140"
OUT_DIR = ROOT_DIR / "03_Project" / "data" / "processed_ann" / "sentiment140_binary_87"

TRAIN_PER_CLASS = 100_000
VAL_PER_CLASS = 10_000
TEST_PER_CLASS = 10_000


def main(seed: int = 42):
    train_src = pd.read_csv(SRC_DIR / "train.csv")
    val_src = pd.read_csv(SRC_DIR / "val.csv")  # already disjoint from train_src

    def balanced_sample(df, n_per_class, rng):
        parts = []
        for cls in ["negative", "positive"]:
            subset = df[df["sentiment"] == cls]
            parts.append(subset.sample(n=min(n_per_class, len(subset)), random_state=rng))
        return pd.concat(parts).sample(frac=1.0, random_state=rng).reset_index(drop=True)

    train_df = balanced_sample(train_src, TRAIN_PER_CLASS, seed)
    # val/test both drawn from val_src (disjoint from train_src), split further so they don't overlap each other
    val_pool, test_pool = train_test_split(
        val_src, test_size=0.5, stratify=val_src["sentiment"], random_state=seed
    )
    val_df = balanced_sample(val_pool, VAL_PER_CLASS, seed)
    test_df = balanced_sample(test_pool, TEST_PER_CLASS, seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    print("train dist:\n", train_df["sentiment"].value_counts())
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
