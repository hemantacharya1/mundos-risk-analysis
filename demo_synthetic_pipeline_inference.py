"""Generate random synthetic leads and score them with a saved PCA pipeline.

Usage:
  uv run python demo_synthetic_pipeline_inference.py \
      --pipeline best_pca_pipeline.pkl --n 12 --output synthetic_predictions.csv

Notes:
  * Generates plausible positive / negative intent summaries.
  * Computes fresh embeddings (same model as training script).
  * Adds sentiment features if the pipeline expects them.
  * Outputs CSV with lead_id, customer_summary, agent_summary, proba, prediction.
"""
from __future__ import annotations
import argparse
import logging
import random
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

import generate_embeddings as ge  # reuse encode_texts / model loading
import sentiment_features as sf   # reuse sentiment analyzer if needed
import baseline_model as bm       # for combine_features

LOGGER = logging.getLogger("demo_synth")
MODEL_NAME = ge.MODEL_NAME
SENTIMENT_MODEL = sf.DEFAULT_MODEL

POS_CUSTOMER = [
    "I'd like to upgrade and add more coverage to my policy.",
    "Looking for a premium package; ready to proceed today.",
    "Need faster onboarding; budget approved already.",
    "Happy with trial results, want annual subscription.",
]
NEG_CUSTOMER = [
    "Just comparing prices, not sure if I'll switch.",
    "Probably cancelling; service felt slow last month.",
    "Exploring options but have no budget yet.",
    "Might revisit next quarter if priorities change.",
]
POS_AGENT = [
    "Great news, I can activate that upgrade immediately.",
    "We can finalize the annual plan and include the discount.",
    "Implementation can start this week once you confirm.",
    "I'll prepare the contract; activation is same-day.",
]
NEG_AGENT = [
    "Feel free to reach out if your situation changes.",
    "Understood; you can continue evaluating with no commitment.",
    "Let me know if you find budget and we can revisit.",
    "I'll close the file for now but I'm available for questions.",
]

def build_samples(n: int, seed: int, pos_frac: float):
    """Generate n synthetic leads with approximately pos_frac positives.

    We set a synthetic "label" based on the template source (positive templates => 1).
    """
    random.seed(seed)
    n_pos = int(round(n * pos_frac))
    n_neg = n - n_pos
    rows = []
    # Generate positives
    for i in range(n_pos):
        cust = random.choice(POS_CUSTOMER)
        agent = random.choice(POS_AGENT)
        rows.append({
            "lead_id": 1_000_000 + i,
            "customer_summary": cust,
            "agent_summary": agent,
            "synthetic_label": 1,
        })
    # Generate negatives
    for j in range(n_neg):
        cust = random.choice(NEG_CUSTOMER)
        agent = random.choice(NEG_AGENT)
        rows.append({
            "lead_id": 2_000_000 + j,
            "customer_summary": cust,
            "agent_summary": agent,
            "synthetic_label": 0,
        })
    random.shuffle(rows)
    return pd.DataFrame(rows)

def compute_embeddings(df: pd.DataFrame, batch_size: int = 16):
    import torch
    device = ge.detect_device()
    LOGGER.info("Embedding device: %s", device)
    tokenizer, model = ge.load_model(MODEL_NAME, device)
    cust_emb = ge.encode_texts(df["customer_summary"].tolist(), tokenizer, model, device, batch_size=batch_size)
    agent_emb = ge.encode_texts(df["agent_summary"].tolist(), tokenizer, model, device, batch_size=batch_size)
    combined = np.concatenate([cust_emb, agent_emb], axis=1).astype(np.float32)
    return combined

def compute_sentiment(df: pd.DataFrame, batch_size: int = 32):
    device = sf.detect_device()
    analyzer = sf.load_analyzer(SENTIMENT_MODEL, device)
    cust_res = sf.compute_sentiment(df["customer_summary"].tolist(), analyzer, batch_size=batch_size)
    agent_res = sf.compute_sentiment(df["agent_summary"].tolist(), analyzer, batch_size=batch_size)
    out = np.vstack([
        cust_res.signed,
        agent_res.signed,
        cust_res.p_pos,
        cust_res.p_neg,
        agent_res.p_pos,
        agent_res.p_neg,
        agent_res.signed - cust_res.signed,
    ]).T.astype(np.float32)
    return out

def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Synthetic PCA pipeline inference")
    ap.add_argument("--pipeline", type=Path, default=Path("best_pca_pipeline.pkl"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos-frac", type=float, default=0.5, help="Fraction of positive (conversion-like) samples to generate")
    ap.add_argument("--output", type=Path, default=Path("synthetic_predictions.csv"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if not args.pipeline.exists():
        raise SystemExit("Pipeline pickle not found")
    with open(args.pipeline, "rb") as f:
        pipe = pickle.load(f)
    scaler = pipe.get("scaler")
    pca = pipe["pca"]
    model = pipe["model"]
    sentiment_used = bool(pipe.get("sentiment_used", False))

    if not 0 < args.pos_frac < 1:
        raise SystemExit("--pos-frac must be between 0 and 1 (exclusive)")
    df = build_samples(args.n, args.seed, args.pos_frac)
    X_emb = compute_embeddings(df)
    X_proc = scaler.transform(X_emb) if scaler else X_emb
    X_pca = pca.transform(X_proc)

    if sentiment_used:
        X_sent = compute_sentiment(df)
    else:
        X_sent = None
    X_final = bm.combine_features(X_pca, X_sent)

    proba = model.predict_proba(X_final)[:, 1]
    preds = (proba >= 0.5).astype(int)
    out_df = df.copy()
    out_df["proba"] = proba
    out_df["prediction"] = preds
    out_df.sort_values("proba", ascending=False, inplace=True)
    # If we have synthetic labels, compute quick metrics
    if "synthetic_label" in out_df.columns:
        y_true = out_df["synthetic_label"].to_numpy()
        try:
            auc = roc_auc_score(y_true, out_df["proba"].to_numpy())
        except ValueError:
            auc = None
        report = classification_report(y_true, out_df["prediction"].to_numpy(), digits=3)
        LOGGER.info("Synthetic set size=%d pos_frac=%.3f AUC=%s", len(out_df), args.pos_frac, f"{auc:.4f}" if auc is not None else "NA")
        print(report)
    out_df.to_csv(args.output, index=False)
    LOGGER.info("Wrote synthetic predictions -> %s", args.output)
    print(out_df.head(15))

if __name__ == "__main__":  # pragma: no cover
    main()
