"""Inference for interest stage model (LogisticRegression pipeline).
Usage:
    uv run python interest_multiclass/inference.py --model interest_multiclass/artifacts/logreg_pipeline.pkl \
            --input leads_1.csv --output predictions_interest.csv
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
try:  # Prefer absolute package import
    from interest_multiclass.config import EMBEDDINGS_FILE, SENTIMENT_FILE, CLASS_LABEL_COLUMN
except Exception:
    try:  # Relative import when executed via -m inside package
        from .config import EMBEDDINGS_FILE, SENTIMENT_FILE, CLASS_LABEL_COLUMN  # type: ignore
    except Exception:  # Final fallback: add parent to sys.path
        import sys, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from interest_multiclass.config import EMBEDDINGS_FILE, SENTIMENT_FILE, CLASS_LABEL_COLUMN  # type: ignore


def load_pipeline(path: Path):
    if not path.exists():
        raise SystemExit(f"Model file missing: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if "model" not in obj:
        raise SystemExit("Invalid pipeline object")
    return obj


def load_embeddings() -> np.ndarray:
    with np.load(EMBEDDINGS_FILE) as npz:
        if 'combined' in npz:
            return npz['combined'].astype(np.float32)
        if 'embeddings' in npz:
            return npz['embeddings'].astype(np.float32)
        raise SystemExit("No suitable array in embeddings NPZ")


def load_sentiment() -> np.ndarray | None:
    if not Path(SENTIMENT_FILE).exists():
        return None
    df = pd.read_csv(SENTIMENT_FILE)
    cols = [c for c in df.columns if c.startswith('sentiment_')]
    if not cols:
        return None
    return df[cols].to_numpy(dtype=np.float32)


def assemble_features() -> np.ndarray:
    X_emb = load_embeddings()
    X_sent = load_sentiment()
    if X_sent is not None:
        if len(X_sent) != len(X_emb):
            raise SystemExit("Sentiment row mismatch")
        return np.hstack([X_emb, X_sent])
    return X_emb


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="Input CSV with customer_summary, agent_summary (for row count alignment)")
    ap.add_argument("--output", type=Path, default=Path("predictions_interest.csv"))
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input data file missing: {args.input}")
    df = pd.read_csv(args.input)

    pipe = load_pipeline(args.model)
    X = assemble_features()
    if len(X) != len(df):
        raise SystemExit("Feature rows mismatch input rows")
    scaler = pipe.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)
    pca = pipe.get("pca")
    if pca is not None:
        X = pca.transform(X)
    model = pipe["model"]
    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)

    out_df = df.copy()
    out_df["predicted_stage"] = preds
    for i in range(proba.shape[1]):
        out_df[f"prob_class_{i}"] = proba[:, i]
    out_df.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)

    meta = {
        "model": str(args.model),
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(df),
        "feature_dim": int(X.shape[1]),
        "classes": pipe.get("classes"),
        "sentiment_used": pipe.get("sentiment_used"),
    }
    with open(args.output.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote predictions -> {args.output}")

if __name__ == "__main__":
    main()
