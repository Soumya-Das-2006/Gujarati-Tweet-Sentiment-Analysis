"""
Classical ML baselines for Gujarati tweet sentiment (positive/negative/neutral).

Fixes the leakage risk in the prior draft (which reported a suspicious 100%
Random Forest accuracy): the TF-IDF vectorizer is fit ONLY on the train split
produced by preprocess.py, then applied (transform-only) to val/test. Every
model is also cross-validated on the train split for a more honest estimate.

Usage:
    python src/classical_ml.py --datadir data/processed --outdir results
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=2000, C=5, class_weight="balanced"),
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "svm": LinearSVC(random_state=42, max_iter=5000),
}


def load_split(datadir: Path):
    train = pd.read_csv(datadir / "train.csv")
    val = pd.read_csv(datadir / "val.csv")
    test = pd.read_csv(datadir / "test.csv")
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data/processed")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_split(datadir)

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train["clean_text"])   # fit ONLY on train
    X_val = vectorizer.transform(val["clean_text"])
    X_test = vectorizer.transform(test["clean_text"])
    y_train, y_val, y_test = train["sentiment"], val["sentiment"], test["sentiment"]

    models_dir = Path("models/classical")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, models_dir / "vectorizer.joblib")

    summary = {}
    best_val_acc = 0
    best_model_name = ""
    
    for name, model in MODELS.items():
        print(f"\n=== {name} ===")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        print(f"5-fold CV accuracy on train: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        model.fit(X_train, y_train)
        joblib.dump(model, models_dir / f"{name}.joblib")

        val_report = classification_report(y_val, model.predict(X_val), output_dict=True)
        test_pred = model.predict(X_test)
        test_report = classification_report(y_test, test_pred, output_dict=True)
        cm = confusion_matrix(y_test, test_pred, labels=["positive", "negative", "neutral"]).tolist()

        val_acc = val_report['accuracy']
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_name = name
            joblib.dump(model, models_dir / "best_model.joblib")

        print(f"Val accuracy:  {val_report['accuracy']:.4f}")
        print(f"Test accuracy: {test_report['accuracy']:.4f}")

        summary[name] = {
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "val_accuracy": val_report["accuracy"],
            "test_accuracy": test_report["accuracy"],
            "test_report": test_report,
            "test_confusion_matrix": cm,
            "confusion_matrix_labels": ["positive", "negative", "neutral"],
        }

    with open(outdir / "classical_ml_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved models to {models_dir} and results to {outdir}")
    print(f"Best model based on Val Accuracy: {best_model_name} ({best_val_acc:.4f})")
    print("\nIf any model's accuracy looks implausibly high (e.g. ~100%), check for")
    print("near-duplicate tweets leaking across splits before trusting the number.")


if __name__ == "__main__":
    main()
