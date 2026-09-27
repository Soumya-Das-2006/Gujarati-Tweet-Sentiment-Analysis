"""
Shared model-loading, translation, and prediction helpers for the 4-model
Gujarati sentiment classifier, used by app/main.py (the Flask GUI) and
src/explain.py (the CLI explainability tool), so they never drift apart.

Four models, two different pipelines:
  model_1  IndicBERT v2  - direct Gujarati input, 3-class (negative/neutral/positive)
  model_2  MuRIL         - direct Gujarati input, 3-class
  model_3  Classical ML  - direct Gujarati input (TF-IDF), 3-class
  model_4  Translate+BERTweet - Gujarati translated to English first (NLLB-200),
                                then classified binary (negative/positive only,
                                no neutral - that's what it was trained on)
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from preprocess import clean_gujarati_text, remove_stopwords

CLASS_NAMES = ["negative", "neutral", "positive"]
BINARY_CLASS_NAMES = ["negative", "positive"]

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CLASSICAL_MODEL_PATH = ROOT_DIR / "01_Model" / "classical" / "best_model.joblib"
CLASSICAL_VEC_PATH = ROOT_DIR / "01_Model" / "classical" / "vectorizer.joblib"
CLASSICAL_RESULTS_PATH = ROOT_DIR / "06_Results" / "classical_ml_results.json"
TRANSFORMER_MODEL_PATH = ROOT_DIR / "01_Model" / "muril" / "final_model"  # kept for explain.py CLI default

TRANSLATOR_MODEL_NAME = "facebook/nllb-200-distilled-600M"
GUJARATI_NLLB_CODE = "guj_Gujr"
ENGLISH_NLLB_CODE = "eng_Latn"

TRANSLATED_DATASET = "GujaratiTweetsData (machine-translated)"
ENGLISH_DATASET = "Sentiment140 (English, binary)"


def _best_classical_name(results: dict) -> str:
    return max(results, key=lambda k: results[k].get("val_accuracy", 0))


def _read_accuracy(results_path: Path, key: str | None = None) -> float | None:
    if not results_path.exists():
        return None
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    if key is not None:
        data = data.get(key, {})
    return data.get("test_accuracy")


MODEL_REGISTRY = {
    "model_1": {
        "name": "My Model",
        "kind": "transformer",
        "model_path": ROOT_DIR / "01_Model" / "indicbert" / "final_model",
        "results_path": ROOT_DIR / "01_Model" / "indicbert" / "muril_results.json",
        "classes": CLASS_NAMES,
        "dataset": TRANSLATED_DATASET,
        "needs_translation": False,
    },
    "model_2": {
        "name": "MuRIL (Multilingual)",
        "kind": "transformer",
        "model_path": ROOT_DIR / "01_Model" / "muril" / "final_model",
        "results_path": ROOT_DIR / "01_Model" / "muril" / "muril_results.json",
        "classes": CLASS_NAMES,
        "dataset": TRANSLATED_DATASET,
        "needs_translation": False,
    },
    "model_3": {
        "name": "Classical ML (best)",
        "kind": "classical",
        "model_path": CLASSICAL_MODEL_PATH,
        "vec_path": CLASSICAL_VEC_PATH,
        "results_path": CLASSICAL_RESULTS_PATH,
        "classes": CLASS_NAMES,
        "dataset": TRANSLATED_DATASET,
        "needs_translation": False,
    },
    "model_4": {
        "name": "Translate -> BERTweet (English)",
        "kind": "transformer",
        "model_path": ROOT_DIR / "01_Model" / "english_binary" / "bertweet-base-v2" / "final_model",
        "results_path": ROOT_DIR / "06_Results" / "english_binary_bertweet-base-v2_results.json",
        "classes": BINARY_CLASS_NAMES,
        "dataset": ENGLISH_DATASET,
        "needs_translation": True,
    },
}


def preprocess_text(text: str) -> str:
    return remove_stopwords(clean_gujarati_text(text))


def load_classical_model(model_path=CLASSICAL_MODEL_PATH, vec_path=CLASSICAL_VEC_PATH):
    if not Path(model_path).exists() or not Path(vec_path).exists():
        return None
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return {"model": model, "vectorizer": vectorizer, "supports_proba": hasattr(model, "predict_proba")}


def load_transformer_model(model_path=TRANSFORMER_MODEL_PATH, device=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not Path(model_path).exists():
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device, "supports_proba": True}


def load_translator(device=None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL_NAME).to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}


def translate_gu_to_en(translator: dict, text: str) -> str:
    model, tokenizer, device = translator["model"], translator["tokenizer"], translator["device"]
    tokenizer.src_lang = GUJARATI_NLLB_CODE
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(ENGLISH_NLLB_CODE),
            max_new_tokens=128,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def classical_predict_proba(handle: dict, texts: list[str]) -> np.ndarray:
    """Returns an (n_texts, 3) probability array in CLASS_NAMES order.
    Raises AttributeError if the loaded model has no predict_proba - callers
    must check handle['supports_proba'] first."""
    model, vectorizer = handle["model"], handle["vectorizer"]
    vec = vectorizer.transform(texts)
    probs = model.predict_proba(vec)
    # sklearn orders columns by model.classes_, which may not match CLASS_NAMES order
    reorder = [list(model.classes_).index(c) for c in CLASS_NAMES]
    return probs[:, reorder]


def classical_predict_label(handle: dict, text: str) -> str:
    model, vectorizer = handle["model"], handle["vectorizer"]
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


def transformer_predict_proba(handle: dict, texts: list[str], batch_size: int = 32, num_classes: int = 3) -> np.ndarray:
    """Returns an (n_texts, num_classes) probability array. 3-class models were
    trained with id2label = {0: negative, 1: neutral, 2: positive}; the binary
    English model with {0: negative, 1: positive} - both match this module's
    CLASS_NAMES / BINARY_CLASS_NAMES ordering already."""
    model, tokenizer, device = handle["model"], handle["tokenizer"], handle["device"]
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def make_predict_proba_fn(model_type: str, handle: dict):
    """Returns a `texts -> (n, k) ndarray` closure, the shape LIME expects.
    model_type: 'transformer' or 'classical' (legacy 2-model API, still used
    by explain.py's CLI and by the registry-based predict_one below)."""
    if model_type == "transformer":
        return lambda texts: transformer_predict_proba(handle, list(texts))
    return lambda texts: classical_predict_proba(handle, list(texts))


# ---------------------------------------------------------------------------
# Registry-based multi-model API used by app/main.py
# ---------------------------------------------------------------------------


def load_all_registry_models(device=None) -> dict:
    """Loads all 4 models (+ the shared translator) once, returns a dict of
    handles keyed by model_1..model_4, each augmented with its static config
    (name/classes/dataset/accuracy/needs_translation)."""
    handles = {}
    for key, cfg in MODEL_REGISTRY.items():
        if cfg["kind"] == "transformer":
            handle = load_transformer_model(cfg["model_path"], device=device)
            accuracy = _read_accuracy(cfg["results_path"])
        else:
            handle = load_classical_model(cfg["model_path"], cfg["vec_path"])
            best_name = None
            if cfg["results_path"].exists():
                with open(cfg["results_path"], encoding="utf-8") as f:
                    all_results = json.load(f)
                best_name = _best_classical_name(all_results)
                accuracy = all_results.get(best_name, {}).get("test_accuracy")
            else:
                accuracy = None
            if handle is not None:
                handle["classical_model_name"] = best_name

        handles[key] = {
            **cfg,
            "handle": handle,
            "loaded": handle is not None,
            "accuracy": accuracy,
        }

    handles["translator"] = load_translator(device=device)
    return handles


def predict_one(model_key: str, registry: dict, clean_gujarati_text_input: str, english_text: str | None):
    """Runs one model's prediction + timing. Returns a result dict matching
    the API shape the frontend expects, or an error dict if unavailable."""
    entry = registry[model_key]
    if not entry["loaded"]:
        return {"error": f"{entry['name']} is not available on this server (model not trained/found)."}

    text_for_model = english_text if entry["needs_translation"] else clean_gujarati_text_input
    if not text_for_model:
        return {"error": "No usable text for this model after preprocessing/translation."}

    classes = entry["classes"]
    start = time.perf_counter()
    if entry["kind"] == "transformer":
        probs = transformer_predict_proba(entry["handle"], [text_for_model], num_classes=len(classes))[0]
    else:
        probs = classical_predict_proba(entry["handle"], [text_for_model])[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    pred_idx = int(np.argmax(probs))
    return {
        "model_name": entry["name"],
        "sentiment": classes[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "test_accuracy": f"{entry['accuracy'] * 100:.1f}%" if entry["accuracy"] is not None else "unknown",
        "dataset": entry["dataset"],
        "used_translation": entry["needs_translation"],
        "inference_ms": round(elapsed_ms, 1),
    }


def explain_one(model_key: str, registry: dict, clean_gujarati_text_input: str, english_text: str | None,
                 num_features: int = 6, num_samples: int = 300) -> dict:
    """LIME explanation for one model. Picks Gujarati vs English text
    automatically based on which pipeline the model actually runs on, and
    highlights whichever language it explained (frontend shows a language tag)."""
    from lime.lime_text import LimeTextExplainer

    entry = registry[model_key]
    if not entry["loaded"]:
        return {"error": f"{entry['name']} is not available on this server."}
    if entry["kind"] == "classical" and not entry["handle"].get("supports_proba", True):
        return {"error": f"{entry['name']} doesn't provide probability estimates, so it can't be explained."}

    text_for_model = english_text if entry["needs_translation"] else clean_gujarati_text_input
    if not text_for_model:
        return {"error": "No usable text for this model after preprocessing/translation."}

    classes = entry["classes"]
    if entry["kind"] == "transformer":
        predict_fn = lambda texts: transformer_predict_proba(entry["handle"], list(texts), num_classes=len(classes))  # noqa: E731
    else:
        predict_fn = lambda texts: classical_predict_proba(entry["handle"], list(texts))  # noqa: E731

    probs = predict_fn([text_for_model])[0]
    predicted_idx = int(np.argmax(probs))

    # split_expression=r'\s+' is required for Gujarati text: LIME's default \W+
    # regex treats Gujarati vowel signs (combining marks) as separators and
    # shreds words into fragments (e.g. "ખરાબ" -> "ખર" + "ાબ"). Whitespace-only
    # splitting is safe (and harmless) for English text too.
    explainer = LimeTextExplainer(class_names=classes, split_expression=r"\s+", bow=False)
    exp = explainer.explain_instance(
        text_for_model, predict_fn, num_features=num_features, num_samples=num_samples, labels=(predicted_idx,)
    )
    weights = dict(exp.as_list(label=predicted_idx))
    max_abs = max((abs(w) for w in weights.values()), default=1.0) or 1.0

    html_parts = []
    for token in text_for_model.split():
        weight = weights.get(token)
        if weight is None:
            html_parts.append(f"<span>{token}</span>")
            continue
        intensity = min(abs(weight) / max_abs, 1.0)
        alpha = 0.15 + 0.55 * intensity
        color = f"rgba(34,139,58,{alpha:.2f})" if weight > 0 else f"rgba(200,50,50,{alpha:.2f})"
        html_parts.append(
            f'<span class="lime-word" style="background-color:{color};" title="impact: {weight:.3f}">{token}</span>'
        )

    return {
        "html": " ".join(html_parts),
        "predicted_sentiment": classes[predicted_idx],
        "explained_language": "english" if entry["needs_translation"] else "gujarati",
    }
