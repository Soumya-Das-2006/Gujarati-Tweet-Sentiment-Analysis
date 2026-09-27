"""
Predict sentiment for a Gujarati sentence using the fine-tuned MuRIL model,
and explain WHY: which words in the sentence pushed the prediction toward
that sentiment.

Explanation method: leave-one-out occlusion. For each word, we remove it,
re-run the model, and measure how much the predicted class's probability
drops. A word whose removal drops the probability a lot is a word the model
is relying on for that sentiment.

Usage:
    python src/predict.py "તમારો ટેક્સ્ટ અહીં લખો"
    python src/predict.py --model models/muril/final_model "તમારો ટેક્સ્ટ અહીં લખો"
"""
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import clean_gujarati_text, remove_stopwords

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def get_probs(text: str, tokenizer, model) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    return F.softmax(logits, dim=-1)[0]


def predict_with_explanation(raw_text: str, tokenizer, model, top_k: int = 5):
    clean = remove_stopwords(clean_gujarati_text(raw_text))
    words = clean.split()

    if not words:
        return {"error": "No Gujarati words found after cleaning. Check the input text."}

    base_probs = get_probs(clean, tokenizer, model)
    pred_idx = int(torch.argmax(base_probs))
    pred_label = ID2LABEL[pred_idx]
    base_prob = float(base_probs[pred_idx])

    contributions = []
    for i, word in enumerate(words):
        without_word = " ".join(words[:i] + words[i + 1 :])
        if not without_word:
            drop = base_prob  # removing the only word drops confidence to "no evidence"
        else:
            probs_without = get_probs(without_word, tokenizer, model)
            drop = base_prob - float(probs_without[pred_idx])
        contributions.append((word, drop))

    contributions.sort(key=lambda x: x[1], reverse=True)

    return {
        "input_text": raw_text,
        "cleaned_text": clean,
        "predicted_sentiment": pred_label,
        "confidence": round(base_prob, 4),
        "all_class_probabilities": {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(base_probs)},
        "top_contributing_words": [
            {"word": w, "impact": round(d, 4)} for w, d in contributions[:top_k] if d > 0
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Gujarati sentence to analyze")
    parser.add_argument("--model", default="models/muril/final_model")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    result = predict_with_explanation(args.text, tokenizer, model, args.top_k)

    print(f"\nInput:      {result.get('input_text')}")
    print(f"Sentiment:  {result.get('predicted_sentiment', '?').upper()}  (confidence: {result.get('confidence')})")
    print(f"All probabilities: {result.get('all_class_probabilities')}")
    print("Why (words that most support this prediction):")
    for item in result.get("top_contributing_words", []):
        print(f"  - {item['word']}  (impact: {item['impact']})")


if __name__ == "__main__":
    main()
