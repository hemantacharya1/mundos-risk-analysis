"""Inference using a saved PCA pipeline (best_pca_pipeline.pkl).

Applies: (optional) StandardScaler -> PCA -> (calibrated) LogisticRegression
on top of precomputed sentence embeddings (+ optional sentiment features).

Usage:
  uv run python inference_pca_pipeline.py \
      --pipeline best_pca_pipeline.pkl \
      --embeddings embeddings.npz \
      --output predictions_pca_pipeline.csv

If the pipeline was trained with sentiment features (sentiment_used=True in
pickle), supply the same sentiment file used in training:
  --sentiment sentiment.csv

Outputs CSV with: lead_id, proba, prediction (thresholded), plus any
existing columns if --include-metadata provided.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import baseline_model as bm

LOGGER = logging.getLogger("inference_pca_pipeline")

def load_pipeline(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Pipeline pickle not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    required = {"pca", "model", "n_components"}
    missing = required - set(obj.keys())
    if missing:
        raise ValueError(f"Pipeline pickle missing keys: {missing}")
    return obj

def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Inference with saved PCA pipeline")
    ap.add_argument("--pipeline", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings.npz"))
    ap.add_argument("--sentiment", type=Path, default=None, help="Sentiment features file if pipeline expects them")
    ap.add_argument("--data", type=Path, default=Path("leads.csv"))
    ap.add_argument("--output", type=Path, default=Path("predictions_pca_pipeline.csv"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--include-metadata", action="store_true", help="Include original columns in output")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    pipeline = load_pipeline(args.pipeline)
    scaler = pipeline.get("scaler")
    pca = pipeline["pca"]
    model = pipeline["model"]
    sentiment_used = pipeline.get("sentiment_used", False)

    if not args.embeddings.exists():
        raise SystemExit("Embeddings NPZ missing; generate with generate_embeddings.py")
    X_emb = bm.load_embeddings(args.embeddings)

    if sentiment_used:
        if not args.sentiment:
            raise SystemExit("Pipeline expects sentiment features; provide --sentiment path")
        X_sent = bm.load_sentiment(args.sentiment)
        if len(X_sent) != len(X_emb):
            raise SystemExit("Sentiment rows mismatch embeddings rows")
    else:
        X_sent = None

    X_proc = scaler.transform(X_emb) if scaler else X_emb
    X_pca = pca.transform(X_proc)
    X_final = bm.combine_features(X_pca, X_sent)

    proba = model.predict_proba(X_final)[:, 1]
    preds = (proba >= args.threshold).astype(int)

    if not args.data.exists():
        raise SystemExit("Data CSV with lead_id missing; required for output alignment")
    df = pd.read_csv(args.data)
    if len(df) != len(proba):
        raise SystemExit("Row mismatch between data and features")
    out_df = pd.DataFrame({
        "lead_id": df["lead_id"],
        "proba": proba,
        "prediction": preds,
    })
    if args.include_metadata:
        meta_cols = [c for c in df.columns if c not in out_df.columns]
        out_df = pd.concat([df[meta_cols], out_df], axis=1)

    out_df.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
    LOGGER.info("Wrote predictions -> %s", args.output)

    meta = {
        "pipeline": str(args.pipeline),
        "embeddings": str(args.embeddings),
        "sentiment": str(args.sentiment) if args.sentiment else None,
        "threshold": args.threshold,
        "calibrated": bool(pipeline.get("calibrated", False)),
        "n_components": pipeline.get("n_components"),
        "variance_sum": pipeline.get("variance_sum"),
        "sentiment_used": sentiment_used,
        "proba_min": float(np.min(proba)),
        "proba_max": float(np.max(proba)),
    }
    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    LOGGER.info("Wrote meta -> %s", meta_path)

if __name__ == "__main__":  # pragma: no cover
    main()
