# Gujarati Tweet Sentiment Analysis — Experiment

3-class sentiment (positive / negative / neutral), comparing classical ML
baselines against MuRIL, on one shared leakage-free train/val/test split.

## Setup

```bash
pip install -r requirements.txt
```

If you have the RTX 4050 GPU available, install the CUDA build of torch instead
of the CPU one in requirements.txt (check https://pytorch.org for the current
command matching your CUDA version), otherwise `train_muril.py` will fall back
to CPU and be much slower.

## Pipeline

1. Put the raw dataset at `data/GujaratiTweetsData.csv` (columns: `textID`,
   `text`, `tranlate_text`, `sentiment`).

2. Preprocess + split (creates `data/processed/{train,val,test}.csv`):
   ```bash
   python src/preprocess.py --input data/GujaratiTweetsData.csv
   ```

3. Train classical ML baselines (Naive Bayes, Random Forest, Gradient
   Boosting, SVM) on TF-IDF features:
   ```bash
   python src/classical_ml.py
   ```

4. Fine-tune MuRIL:
   ```bash
   python src/train_muril.py --epochs 15
   ```

5. Combine into one comparison table + chart for the paper:
   ```bash
   python src/compare_results.py
   ```

## Why this differs from the earlier draft

The paper draft ("Sentiment Analysis on Gujarati Twitter Data Using
Supervised Machine Learning") reported **100% accuracy for Random Forest**,
which is not credible for a 20k-tweet, 3-class social media task and isn't
corroborated by the DRC progress report (which cites MuRIL's 70.79%/71.43%
as the credible result). The likely cause is TF-IDF being fit on the full
corpus before the train/test split, and/or near-duplicate tweets leaking
across splits. `preprocess.py` deduplicates first and produces the split;
`classical_ml.py` fits TF-IDF only on the train partition and 5-fold
cross-validates each model — rerun both scripts to get a trustworthy number
before quoting any classical ML accuracy in the paper.

## Next steps (from the DRC-3 future work plan)

- [ ] Further fine-tune MuRIL, particularly to fix weak positive-sentiment recall
- [ ] Expand dataset toward the ~25,000-tweet target
- [ ] Add LSTM baseline
- [ ] Wrap the best model in a real-time inference API/web app
