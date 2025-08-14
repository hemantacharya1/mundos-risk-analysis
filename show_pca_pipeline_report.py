"""Generate a classification report for a saved PCA pipeline (e.g. best_pca_pipeline.pkl).

Note: The saved pipeline was trained (fitted) on the FULL dataset when created
with --save-best. Producing a fresh train/test split here will yield slightly
optimistic metrics because PCA (and scaler) already saw all samples. For a
strict evaluation, rerun train_pca_pipeline.py without saving best, and use the
reported test metrics from the sweep.

Usage:
  uv run python show_pca_pipeline_report.py --pipeline best_pca_pipeline.pkl \
      --embeddings embeddings.npz --sentiment sentiment.csv

Options:
  --test-size 0.2 --seed 42  (controls the evaluation split)
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
import baseline_model as bm

LOGGER = logging.getLogger("show_pca_pipeline_report")

def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Classification report for saved PCA pipeline")
    ap.add_argument("--pipeline", type=Path, default=Path("best_pca_pipeline.pkl"))
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings.npz"))
    ap.add_argument("--sentiment", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=Path("leads.csv"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if not args.pipeline.exists():
        raise SystemExit("Pipeline pickle not found")
    with open(args.pipeline, "rb") as f:
        pipe = pickle.load(f)

    sentiment_used = pipe.get("sentiment_used", False)
    scaler = pipe.get("scaler")
    pca = pipe["pca"]
    model = pipe["model"]

    if not args.embeddings.exists():
        raise SystemExit("Embeddings NPZ missing")
    X_emb = bm.load_embeddings(args.embeddings)

    X_sent = None
    if sentiment_used:
        if not args.sentiment:
            raise SystemExit("Pipeline expects sentiment; provide --sentiment")
        X_sent = bm.load_sentiment(args.sentiment)
        if len(X_sent) != len(X_emb):
            raise SystemExit("Sentiment rows mismatch embeddings")

    if not args.data.exists():
        raise SystemExit("Data CSV missing")
    y = pd.read_csv(args.data)["conversion_label"].to_numpy()

    if len(y) != len(X_emb):
        raise SystemExit("Label rows mismatch embeddings")

    X_proc = scaler.transform(X_emb) if scaler else X_emb
    X_pca = pca.transform(X_proc)
    X = bm.combine_features(X_pca, X_sent)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, digits=3)
    calibrated = bool(pipe.get("calibrated", False))
    brier = brier_score_loss(y_test, proba) if calibrated else None

    print(f"Pipeline: {args.pipeline}\nComponents: {pipe.get('n_components')} sentiment_used={sentiment_used} calibrated={calibrated}\nSplit test_size={args.test_size} seed={args.seed}\nAUC: {auc:.6f}{' Brier: '+format(brier,'.6f') if brier is not None else ''}\n")
    print(report)

if __name__ == "__main__":  # pragma: no cover
    main()
