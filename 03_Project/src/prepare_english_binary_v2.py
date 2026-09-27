"""
v2: lighter cleaning that PRESERVES case, punctuation, and emoticons
(";D", "!!!", ":)") - the earlier all-lowercase/letters-only cleaning was
stripping exactly the signals a tweet-pretrained model like BERTweet relies
on most. Also samples more data per class (200k vs 100k) for better
generalization (the v1 model started overfitting by epoch 2).

Usage:
    python src/prepare_english_binary_v2.py
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_PATH = ROOT_DIR / "03_Project" / "data" / "raw" / "TweetsData_English.csv"
OUT_DIR = ROOT_DIR / "03_Project" / "data" / "processed_ann" / "sentiment140_binary_v2"

TRAIN_PER_CLASS = 200_000
VAL_PER_CLASS = 10_000
TEST_PER_CLASS = 10_000


def light_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main(seed: int = 42):
    cols = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(RAW_PATH, header=None, names=cols, encoding="latin-1")
    df = df[df["target"].isin([0, 4])]
    df["sentiment"] = df["target"].map({0: "negative", 4: "positive"})

    df["clean_text"] = df["text"].apply(light_clean)
    df = df[df["clean_text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"])
    print(f"deduplicated {before - len(df)} rows ({before} -> {len(df)})")

    train_df, temp_df = train_test_split(df, test_size=0.03, stratify=df["sentiment"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["sentiment"], random_state=seed)

    def balanced_sample(d, n_per_class):
        parts = [d[d["sentiment"] == c].sample(n=min(n_per_class, len(d[d["sentiment"] == c])), random_state=seed) for c in ["negative", "positive"]]
        return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_out = balanced_sample(train_df, TRAIN_PER_CLASS)
    val_out = balanced_sample(val_df, VAL_PER_CLASS)
    test_out = balanced_sample(test_df, TEST_PER_CLASS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out[["clean_text", "sentiment"]].to_csv(OUT_DIR / "train.csv", index=False)
    val_out[["clean_text", "sentiment"]].to_csv(OUT_DIR / "val.csv", index=False)
    test_out[["clean_text", "sentiment"]].to_csv(OUT_DIR / "test.csv", index=False)

    print(f"train: {len(train_out)}  val: {len(val_out)}  test: {len(test_out)}")
    print("sample cleaned texts:")
    print(train_out["clean_text"].head(5).to_string())
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
