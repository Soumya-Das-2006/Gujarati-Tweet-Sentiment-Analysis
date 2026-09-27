"""
Data prep for the two-stage ANN model:

Stage 1 (pretrain): Sentiment140 - 1.6M English tweets, binary sentiment
    (0=negative, 4=positive -> remapped to 0/1). No header row in the raw file.

Stage 2 (fine-tune): Raw_NLP_Data.csv - real native Gujarati text, originally
    labeled for hate speech, remapped to 3-class sentiment:
        Hate (Accusations/Swearing/Promoting violence/Sexist) -> negative
        Non-Hate + Positive                                    -> positive
        Non-Hate + Neutral                                     -> neutral

Usage:
    python src/ann_data_prep.py
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "03_Project" / "data" / "raw"
OUT_DIR = ROOT_DIR / "03_Project" / "data" / "processed_ann"

HATE_SUBCATEGORIES = {"Accusations", "Swearing", "Promoting violence", "Sexist"}


def clean_gujarati_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[a-zA-Z]+", " ", text)
    text = re.sub(r"[^઀-૿\s]", " ", text)  # keep only Gujarati unicode block
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_english_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_gujarati_sentiment(row) -> str:
    if row["Simple_Labels"] == "Hate":
        return "negative"
    if row["SubCategory"] == "Positive":
        return "positive"
    return "neutral"  # Non-Hate + Neutral, and any stray Non-Hate/Accusations|Swearing noise


def prepare_gujarati(seed: int = 42):
    df = pd.read_csv(RAW_DIR / "Raw_NLP_Data.csv")
    df = df.dropna(subset=["Post", "Simple_Labels", "SubCategory"])

    df["sentiment"] = df.apply(map_gujarati_sentiment, axis=1)
    df["clean_text"] = df["Post"].apply(clean_gujarati_text)
    df = df[df["clean_text"].str.len() > 0]

    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"])
    print(f"Gujarati: deduplicated {before - len(df)} rows ({before} -> {len(df)})")
    print("Gujarati sentiment distribution:\n", df["sentiment"].value_counts())

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["sentiment"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["sentiment"], random_state=seed)

    out = OUT_DIR / "gujarati"
    out.mkdir(parents=True, exist_ok=True)
    train_df[["clean_text", "sentiment"]].to_csv(out / "train.csv", index=False)
    val_df[["clean_text", "sentiment"]].to_csv(out / "val.csv", index=False)
    test_df[["clean_text", "sentiment"]].to_csv(out / "test.csv", index=False)
    print(f"Saved Gujarati split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test -> {out}")


def prepare_sentiment140(seed: int = 42, sample_size: int | None = None):
    cols = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(RAW_DIR / "TweetsData_English.csv", header=None, names=cols, encoding="latin-1")
    df = df[df["target"].isin([0, 4])]
    df["sentiment"] = df["target"].map({0: "negative", 4: "positive"})

    if sample_size and sample_size < len(df):
        df = df.groupby("sentiment", group_keys=False).apply(
            lambda g: g.sample(sample_size // 2, random_state=seed)
        )

    df["clean_text"] = df["text"].apply(clean_english_text)
    df = df[df["clean_text"].str.len() > 0]

    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"])
    print(f"Sentiment140: deduplicated {before - len(df)} rows ({before} -> {len(df)})")
    print("Sentiment140 distribution:\n", df["sentiment"].value_counts())

    train_df, val_df = train_test_split(df, test_size=0.05, stratify=df["sentiment"], random_state=seed)

    out = OUT_DIR / "sentiment140"
    out.mkdir(parents=True, exist_ok=True)
    train_df[["clean_text", "sentiment"]].to_csv(out / "train.csv", index=False)
    val_df[["clean_text", "sentiment"]].to_csv(out / "val.csv", index=False)
    print(f"Saved Sentiment140 split: {len(train_df)} train / {len(val_df)} val -> {out}")


if __name__ == "__main__":
    prepare_gujarati()
    prepare_sentiment140()
