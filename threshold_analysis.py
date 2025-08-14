"""Generate precision/recall/lift metrics over thresholds for a trained model.

Supports either:
  * Raw logistic regression pickle (baseline or reduced) expecting full feature matrix from embeddings/sentiment or PCA NPZ.
  * Saved PCA pipeline (best_pca_pipeline.pkl) containing scaler+pca+model+metadata.

Usage examples:
  uv run python threshold_analysis.py --model artifacts/baseline_logreg_*.pkl --embeddings embeddings.npz --sentiment sentiment.csv --out metrics_thresholds.json
  uv run python threshold_analysis.py --pipeline best_pca_pipeline.pkl --embeddings embeddings.npz --sentiment sentiment.csv --out metrics_thresholds.json

Outputs JSON list of operating points along with summary best-F1 and a small lift table.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import baseline_model as bm

LOGGER = logging.getLogger("threshold_analysis")

THRESHOLDS = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]  # 0.05 .. 0.95 step 0.05


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_labels(data_path: Path) -> np.ndarray:
    df = pd.read_csv(data_path)
    return df["conversion_label"].to_numpy()


def compute_operating_points(y_true: np.ndarray, proba: np.ndarray):
    pts = []
    pos_rate = y_true.mean() if len(y_true) else 0.0
    for thr in THRESHOLDS:
        pred = (proba >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        pred_rate = (pred == 1).mean()
        lift = (prec / pos_rate) if pos_rate > 0 and prec > 0 else 0.0
        pts.append({
            "threshold": thr,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "pred_positive_rate": round(float(pred_rate), 4),
            "lift": round(float(lift), 4),
        })
    best = max(pts, key=lambda d: d["f1"])
    return pts, best


def prepare_features(args, pipeline=None):
    if pipeline is not None:
        # Pipeline path: embeddings (+ sentiment) -> scaler -> pca -> combine -> model
        X_emb = bm.load_embeddings(args.embeddings)
        X_sent = None
        if pipeline.get("sentiment_used"):
            if not args.sentiment:
                raise SystemExit("Pipeline expects sentiment; provide --sentiment")
            X_sent = bm.load_sentiment(args.sentiment)
            if len(X_sent) != len(X_emb):
                raise SystemExit("Sentiment rows mismatch embeddings")
        scaler = pipeline.get("scaler")
        pca = pipeline["pca"]
        X_proc = scaler.transform(X_emb) if scaler else X_emb
        X_pca = pca.transform(X_proc)
        return bm.combine_features(X_pca, X_sent)
    # Raw model path: decide which feature source to use
    if args.pca_file:
        return bm.load_pca_features(args.pca_file, use_pca_only=args.pca_only)
    X_emb = bm.load_embeddings(args.embeddings)
    X_sent = bm.load_sentiment(args.sentiment) if args.sentiment else None
    return bm.combine_features(X_emb, X_sent)


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Threshold analysis for classification model")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=Path, help="Pickle of trained logistic regression (baseline or reduced)")
    group.add_argument("--pipeline", type=Path, help="Saved PCA pipeline pickle")
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings.npz"))
    ap.add_argument("--sentiment", type=Path, default=None)
    ap.add_argument("--pca-file", type=Path, default=None)
    ap.add_argument("--pca-only", action="store_true")
    ap.add_argument("--data", type=Path, default=Path("leads.csv"))
    ap.add_argument("--out", type=Path, default=Path("metrics_thresholds.json"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    y = ensure_labels(args.data)

    pipeline = None
    model = None
    if args.pipeline:
        with open(args.pipeline, "rb") as f:
            pipeline = pickle.load(f)
        model = pipeline["model"]
        LOGGER.info("Loaded pipeline (n_components=%s calibrated=%s)", pipeline.get("n_components"), pipeline.get("calibrated"))
    else:
        model = load_model(args.model)
        LOGGER.info("Loaded raw model")

    X = prepare_features(args, pipeline=pipeline)
    if len(X) != len(y):
        raise SystemExit("Feature rows mismatch labels")

    proba = model.predict_proba(X)[:, 1]
    pts, best = compute_operating_points(y, proba)

    summary = {
        "positive_rate": float(y.mean()),
        "n_samples": int(len(y)),
        "best_f1": best,
        "thresholds": pts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Wrote threshold metrics -> %s (best F1 thr=%.2f f1=%.4f)", args.out, best["threshold"], best["f1"])

if __name__ == "__main__":  # pragma: no cover
    main()
