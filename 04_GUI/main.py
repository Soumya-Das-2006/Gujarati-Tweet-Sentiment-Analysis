import sys
import time
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "03_Project" / "src"))

from model_utils import (  # noqa: E402
    MODEL_REGISTRY,
    explain_one,
    load_all_registry_models,
    predict_one,
    preprocess_text,
    translate_gu_to_en,
)

app = Flask(__name__)

REGISTRY = {}
EXAMPLES = []
MODEL_ORDER = ["model_1", "model_2", "model_3", "model_4"]


def _load_examples():
    test_path = ROOT_DIR / "03_Project" / "data" / "processed" / "test.csv"
    if not test_path.exists():
        return
    df = pd.read_csv(test_path)
    for sentiment in ["positive", "negative", "neutral"]:
        rows = df[df["sentiment"] == sentiment]
        if len(rows) == 0:
            continue
        row = rows.sample(1, random_state=7).iloc[0]
        EXAMPLES.append({"sentiment": sentiment, "text": row["text"]})


print("Loading all 4 models + translator (this can take a minute)...")
REGISTRY.update(load_all_registry_models())
for key in MODEL_ORDER:
    status = "OK" if REGISTRY[key]["loaded"] else "NOT FOUND"
    print(f"  {key} ({REGISTRY[key]['name']}): {status}")
_load_examples()
print("Ready.")


def _translate(text: str) -> str:
    try:
        return translate_gu_to_en(REGISTRY["translator"], text)
    except Exception as e:  # translation is best-effort for display; never block prediction
        return f"(translation unavailable: {e})"


@app.route("/")
def index():
    model_cards = [
        {
            "key": key,
            "name": REGISTRY[key]["name"],
            "loaded": REGISTRY[key]["loaded"],
            "accuracy": f"{REGISTRY[key]['accuracy'] * 100:.1f}%" if REGISTRY[key]["accuracy"] is not None else "unknown",
            "dataset": REGISTRY[key]["dataset"],
            "classes": REGISTRY[key]["classes"],
            "needs_translation": REGISTRY[key]["needs_translation"],
        }
        for key in MODEL_ORDER
    ]
    return render_template("index.html", models=model_cards, examples=EXAMPLES)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    text = data.get("text", "").strip()
    model_type = data.get("model_type", "all")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    clean_text = preprocess_text(text)
    if not clean_text:
        return jsonify({"error": "No recognizable Gujarati words found after cleaning."}), 400

    t0 = time.perf_counter()
    english_translation = _translate(text)
    translation_ms = round((time.perf_counter() - t0) * 1000, 1)

    keys = MODEL_ORDER if model_type == "all" else [model_type]
    if model_type != "all" and model_type not in MODEL_REGISTRY:
        return jsonify({"error": f"Unknown model_type '{model_type}'"}), 400

    results = {key: predict_one(key, REGISTRY, clean_text, english_translation) for key in keys}

    return jsonify(
        {
            "clean_text": clean_text,
            "english_translation": english_translation,
            "translation_ms": translation_ms,
            "results": results,
        }
    )


@app.route("/explain", methods=["POST"])
def explain():
    data = request.json or {}
    text = data.get("text", "").strip()
    model_type = data.get("model_type", "all")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    clean_text = preprocess_text(text)
    if not clean_text:
        return jsonify({"error": "No recognizable Gujarati words found after cleaning."}), 400

    english_translation = _translate(text)

    keys = MODEL_ORDER if model_type == "all" else [model_type]
    if model_type != "all" and model_type not in MODEL_REGISTRY:
        return jsonify({"error": f"Unknown model_type '{model_type}'"}), 400

    results = {key: explain_one(key, REGISTRY, clean_text, english_translation) for key in keys}
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
