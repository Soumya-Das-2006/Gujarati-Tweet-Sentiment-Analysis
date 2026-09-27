# Gujarati Sentiment Analysis

Ph.D. research code for Gujarati-language sentiment analysis, by **Manish Joshi**
(Faculty of Information Technology and Computer Science, Parul University),
supervised by **Dr. Snehlata Barde**.

**Demo video:** https://youtu.be/ApXYeB1IJUY

This repository's code actually spans **two distinct research studies** that
share a topic but not a dataset or a method. They are kept separate
throughout — never averaged or merged into one set of numbers.

| | Study A | Study B |
|---|---|---|
| Title | Sentiment Analysis on Gujarati Twitter Data | Evaluating ML and Transformer-Based Models for Gujarati Sentiment Analysis |
| Dataset | 26,905 human-annotated Gujarati tweets | IndicSentiment product reviews (1,151) + native movie reviews (453) |
| Method | Fine-tuning (classical ML + transformers, end-to-end) | Frozen-representation probing (encoders never updated) |
| Deployed? | Yes — live Flask web app | No — a benchmarking study |

---

## Repository structure

```
02_Experiments/     Scratch experiment notes
03_Project/         Source code: preprocessing, training, evaluation scripts
  ├─ src/           Core pipeline (preprocess, classical_ml, train_muril, train_ann, ...)
  ├─ scripts/       Data audit / translation / validation utilities
  └─ data/          processed/ (tweet train/val/test splits) is tracked;
                     raw/ and processed_ann/ are not (too large — see below)
04_GUI/              Flask web application (main.py, templates, static)
06_Results/          Metrics (JSON/CSV), comparison charts, small model artifacts
```

This is a **code-and-results repository**: trained model weights, presentation
decks, and paper drafts are kept locally rather than published here.

## What's *not* in this repo, and why

- Model weights and checkpoints — every training run kept every intermediate
  checkpoint; the full set is 73GB on disk, and GitHub hard-caps individual
  files at 100MB, so publishing them isn't practical without separate
  large-file hosting.
- The two ~413MB Random Forest `.joblib` files (unbounded tree depth over a
  20k-feature TF-IDF matrix makes even a "classical" model huge here).
- `03_Project/data/raw/` (240MB, includes a 238MB English translation dataset)
  and `03_Project/data/processed_ann/` (163MB, Sentiment140-derived data).
- Manuscripts, DRC progress reports, and presentation decks — kept as private
  working documents, not published in this code repository.
- `indic_nlp_resources/` and `.venv/` (third-party / environment, not project code).

Everything excluded is **regeneratable** by re-running the scripts below
against the tracked source data and code.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r 03_Project/requirements.txt
```

Also clone the Indic NLP resources used for Gujarati tokenization:

```bash
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
```

## Reproducing Study A (fine-tuned tweet sentiment)

```bash
cd 03_Project
python src/preprocess.py --input data/raw/GujaratiTweetsData.csv
python src/classical_ml.py
python src/train_muril.py --epochs 15
python src/train_ann.py
python src/train_english_binary.py
python src/compare_results.py
```

## Reproducing Study B (frozen-representation probing)

Study B follows an 11-feature-set × 4-task frozen-probing protocol (4 encoders
× 2 pooling strategies + 3 lexical baselines), evaluated with 10×5-fold
resampled cross-validation and Nadeau–Bengio / Holm–Bonferroni correction.
The full experimental writeup is kept in the accompanying paper draft
(not included in this code repository).

## Running the web app

```bash
cd 04_GUI
python main.py
```

Serves all four Study A models (MuRIL, IndicBERT v2 fine-tuned, Classical ML,
Translate→BERTweet) with live model switching, per-class confidence bars, and
one-click LIME explanations.

## Headline results

**Study A** — best native 3-class result: fine-tuned IndicBERT v2 ("My Model")
at 75.3% test accuracy, narrowly ahead of MuRIL (74.75%). Full metrics in
`06_Results/`.

**Study B** — IndicBERT v2 (frozen, mean-pooled) + Logistic Regression reaches
0.9130 macro F1 on translated product reviews; on native movie reviews, no
frozen encoder is statistically distinguishable from character n-grams
(p = 0.4626 after correction).

## Supervisor

Dr. Snehlata Barde, Faculty of Information Technology and Computer Science,
Parul University.
