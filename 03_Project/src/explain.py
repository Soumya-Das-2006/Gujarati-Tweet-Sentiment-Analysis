import argparse
from pathlib import Path

from lime.lime_text import LimeTextExplainer

from model_utils import (
    CLASS_NAMES,
    load_classical_model,
    load_transformer_model,
    make_predict_proba_fn,
    preprocess_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Gujarati text to explain")
    parser.add_argument("--model_type", choices=["classical", "transformer"], default="classical")
    args = parser.parse_args()

    clean_text = preprocess_text(args.text)
    print(f"Clean text: {clean_text}")

    if args.model_type == "classical":
        handle = load_classical_model()
        if handle is None:
            print("Classical models not found. Please run classical_ml.py first.")
            return
    else:
        handle = load_transformer_model()
        if handle is None:
            print("Transformer model not found. Please run train_muril.py first.")
            return

    if not handle["supports_proba"]:
        print(f"The loaded {args.model_type} model has no predict_proba - cannot explain.")
        return

    # split_expression=r'\s+' - see app/main.py for why this is required
    # (LIME's default regex fragments Gujarati words at vowel-sign boundaries).
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, split_expression=r"\s+", bow=False)
    predict_fn = make_predict_proba_fn(args.model_type, handle)
    exp = explainer.explain_instance(clean_text, predict_fn, num_features=6, num_samples=300)

    print("\nExplanation:")
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight:.4f}")

    out_path = Path("results/explanation.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exp.save_to_file(out_path)
    print(f"\nSaved explanation visualization to {out_path}")


if __name__ == "__main__":
    main()
