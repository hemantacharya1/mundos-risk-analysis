"""Inference script for lead conversion prediction.

Steps:
1. Load saved logistic regression model (trained on embeddings with optional sentiment features).
2. Read input CSV containing columns: lead_id, customer_summary, agent_summary.
3. Generate embeddings (customer + agent) with the same transformer model.
4. Optionally compute sentiment features (auto-required if model expects them).
5. Concatenate features to match training dimension; run prediction.
6. Output CSV with: lead_id, probability, prediction, (optionally) threshold used.

Usage examples:
  uv run python inference.py --model artifacts/baseline_logreg_20250814T091331Z.pkl --input leads.csv --output predictions.csv
  uv run python inference.py --model artifacts/latest.pkl --input new_leads.csv --with-sentiment --output preds.csv

Automatically infers need for sentiment if model feature dimension > embedding dimension (768).
"""
from __future__ import annotations
import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import generate_embeddings as ge
import sentiment_features as sf
import baseline_model as bm

EMBED_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL_DEFAULT = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_ORDER = [
    "sentiment_cust",
    "sentiment_agent",
    "sentiment_cust_pos",
    "sentiment_cust_neg",
    "sentiment_agent_pos",
    "sentiment_agent_neg",
    "sentiment_gap",
]

LOGGER = logging.getLogger("inference")


def build_sentiment_features(df: pd.DataFrame, analyzer, batch_size: int) -> pd.DataFrame:
    cust = sf.compute_sentiment(df["customer_summary"].tolist(), analyzer, batch_size=batch_size)
    agent = sf.compute_sentiment(df["agent_summary"].tolist(), analyzer, batch_size=batch_size)
    out = pd.DataFrame({
        "sentiment_cust": cust.signed,
        "sentiment_agent": agent.signed,
        "sentiment_cust_pos": cust.p_pos,
        "sentiment_cust_neg": cust.p_neg,
        "sentiment_agent_pos": agent.p_pos,
        "sentiment_agent_neg": agent.p_neg,
        "sentiment_gap": agent.signed - cust.signed,
    })
    return out


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run inference on lead data")
    ap.add_argument("--model", type=Path, required=True, help="Path to saved model .pkl")
    ap.add_argument("--input", type=Path, required=True, help="Input CSV with lead_id, customer_summary, agent_summary")
    ap.add_argument("--output", type=Path, default=Path("predictions.csv"), help="Output predictions CSV path")
    ap.add_argument("--embed-model", default=EMBED_MODEL_DEFAULT, help="Embedding model name")
    ap.add_argument("--sentiment-model", default=SENTIMENT_MODEL_DEFAULT, help="Sentiment model name")
    ap.add_argument("--with-sentiment", action="store_true", help="Force computing sentiment features (if model used them)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive class")
    ap.add_argument("--id-column", default="lead_id")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--meta-json", type=Path, default=None, help="Optional meta JSON output path")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")
    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "predict_proba"):
        raise SystemExit("Loaded model does not support predict_proba")

    df = pd.read_csv(args.input)
    required_cols = {"customer_summary", "agent_summary", args.id_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in input: {missing}")

    # Generate embeddings
    device = ge.detect_device(prefer_gpu=not args.no_gpu)
    LOGGER.info("Device: %s", device)
    tokenizer, embed_model = ge.load_model(args.embed_model, device)
    cust_emb = ge.encode_texts(df["customer_summary"].astype(str).tolist(), tokenizer, embed_model, device, batch_size=args.batch_size, max_length=args.max_length)
    agent_emb = ge.encode_texts(df["agent_summary"].astype(str).tolist(), tokenizer, embed_model, device, batch_size=args.batch_size, max_length=args.max_length)
    emb = np.concatenate([cust_emb, agent_emb], axis=1).astype(np.float32)
    LOGGER.info("Embeddings shape=%s", emb.shape)

    expected_dim = model.coef_.shape[1]
    sentiment_needed = expected_dim > emb.shape[1]
    if sentiment_needed and not args.with_sentiment:
        LOGGER.info("Model expects %d features; embeddings provide %d -> computing sentiment automatically", expected_dim, emb.shape[1])
    compute_sent = args.with_sentiment or sentiment_needed

    sent_df = None
    if compute_sent:
        analyzer = sf.load_analyzer(args.sentiment_model, device.type if hasattr(device, 'type') else str(device))
        sent_df = build_sentiment_features(df, analyzer, batch_size=args.batch_size)
        # Ensure order
        sent_df = sent_df[SENTIMENT_ORDER[:sent_df.shape[1]]]
        sent_mat = sent_df.to_numpy(dtype=np.float32)
        features = bm.combine_features(emb, sent_mat)
    else:
        features = emb

    if features.shape[1] != expected_dim:
        raise SystemExit(f"Feature dimension {features.shape[1]} != model expected {expected_dim}")

    proba = model.predict_proba(features)[:, 1]
    preds = (proba >= args.threshold).astype(int)

    out_df = pd.DataFrame({
        args.id_column: df[args.id_column],
        "probability": proba,
        "prediction": preds,
    })
    out_df.to_csv(args.output, index=False)
    LOGGER.info("Wrote predictions -> %s", args.output)

    if args.meta_json:
        meta = {
            "model_path": str(args.model),
            "input_path": str(args.input),
            "rows": len(df),
            "embedding_model": args.embed_model,
            "sentiment_used": compute_sent,
            "feature_dim": features.shape[1],
            "threshold": args.threshold,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        with open(args.meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        LOGGER.info("Wrote meta -> %s", args.meta_json)

if __name__ == "__main__":  # pragma: no cover
    main()
