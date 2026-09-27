"""
Preprocessing for Gujarati tweet sentiment analysis.

Expects a raw CSV with columns compatible with GujaratiTweetsData.csv:
    textID, text, tranlate_text, sentiment   (sentiment in {positive, negative, neutral})

Produces a single stratified train/val/test split that is reused by BOTH the
classical ML pipeline and the MuRIL fine-tuning script, so results are
directly comparable and there is no leakage between splits.

Usage:
    python src/preprocess.py --input data/GujaratiTweetsData.csv
"""
import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Indic NLP setup
from indicnlp import common
from indicnlp.tokenize import indic_tokenize
import os
INDIC_NLP_RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "indic_nlp_resources")
common.set_resources_path(INDIC_NLP_RESOURCES)

GUJARATI_STOPWORDS = {
    "છે", "છું", "છીએ", "છો", "હતો", "હતી", "હતા", "હશે", "થી", "ને", "ના", "ની",
    "નો", "નું", "માં", "પર", "એ", "આ", "તે", "તેમ", "જે", "શું", "કે", "અને",
    "પણ", "કારણ", "કે", "જો", "તો", "એટલે", "માટે", "સાથે", "વગેરે",
}

TEXT_COLUMN_CANDIDATES = ["tranlate_text", "translate_text", "text", "tweet"]
LABEL_COLUMN_CANDIDATES = ["sentiment", "label"]


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in columns: {list(df.columns)}")


def clean_gujarati_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                       # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                    # keep hashtag word, drop '#'
    text = re.sub(r"[a-zA-Z]+", " ", text)                   # stray English words
    text = re.sub(r"[^઀-૿\s]", " ", text)          # keep only Gujarati unicode block + whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    tokens = indic_tokenize.trivial_tokenize(text)
    return " ".join(w for w in tokens if w not in GUJARATI_STOPWORDS)


def load_and_clean(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    text_col = detect_column(df, TEXT_COLUMN_CANDIDATES)
    label_col = detect_column(df, LABEL_COLUMN_CANDIDATES)

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "sentiment"})
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[df["sentiment"].isin(["positive", "negative", "neutral"])]

    df = df.dropna(subset=["text"])
    df["clean_text"] = df["text"].apply(clean_gujarati_text).apply(remove_stopwords)
    df = df[df["clean_text"].str.len() > 0]

    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"])
    print(f"Deduplicated {before - len(df)} rows ({before} -> {len(df)})")

    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int = 42):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["sentiment"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["sentiment"], random_state=seed
    )
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw GujaratiTweetsData.csv")
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(input_path)
    print("Class distribution after cleaning:")
    print(df["sentiment"].value_counts())

    train_df, val_df, test_df = stratified_split(df, seed=args.seed)

    train_df.to_csv(outdir / "train.csv", index=False)
    val_df.to_csv(outdir / "val.csv", index=False)
    test_df.to_csv(outdir / "test.csv", index=False)

    print(f"\nSaved: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test rows to {outdir}")
    print("IMPORTANT: use these exact same files for both classical_ml.py and train_muril.py")
    print("so the two approaches are compared on identical, leakage-free splits.")


if __name__ == "__main__":
    main()
