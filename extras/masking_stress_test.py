"""Masking stress test to assess reliance on top predictive tokens.

Procedure:
1. Load dataset and compute token log-odds (presence per doc) using combined summaries.
2. Select top N high positive log-odds tokens and top N high negative log-odds tokens.
3. Mask (replace) those tokens in summaries with a placeholder.
4. Load existing embeddings (original) from NPZ.
5. Recompute embeddings for masked dataset only.
6. Train/test logistic regression (single stratified split) on original vs masked embeddings; report AUC drop.

Usage:
  uv run python masking_stress_test.py --embeddings embeddings.npz --top-n 20 --output mask_results.json

Optional:
  --seed 42  (reproducibility)
  --test-size 0.2
  --model sentence-transformers/all-MiniLM-L6-v2
  --no-gpu  (force CPU)
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from collections import Counter
from typing import List, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import generate_embeddings as ge
import baseline_model as bm

LOGGER = logging.getLogger("mask_stress")

PLACEHOLDER = "[MASK]"


def tokenize(text: str) -> List[str]:
    return [t for t in text.split() if t]


def compute_log_odds_tokens(texts: pd.Series, labels: np.ndarray):
    pos_counts = Counter()
    neg_counts = Counter()
    for txt, y in zip(texts.fillna(""), labels):
        toks = {t.lower() for t in tokenize(str(txt)) if len(t) > 2}
        if y == 1:
            pos_counts.update(toks)
        else:
            neg_counts.update(toks)
    vocab = sorted(set(pos_counts) | set(neg_counts))
    pos_total = sum(pos_counts.values()) + len(vocab)
    neg_total = sum(neg_counts.values()) + len(vocab)
    rows = []
    for tok in vocab:
        pc = pos_counts.get(tok, 0) + 1
        nc = neg_counts.get(tok, 0) + 1
        log_odds = np.log(pc / pos_total) - np.log(nc / neg_total)
        rows.append((tok, pc - 1, nc - 1, log_odds))
    df = pd.DataFrame(rows, columns=["token", "pos_docs", "neg_docs", "log_odds"])
    return df


def mask_tokens(text: str, to_mask: Set[str]) -> str:
    if not text:
        return text
    out = []
    for tok in tokenize(str(text)):
        low = tok.lower()
        if low in to_mask:
            out.append(PLACEHOLDER)
        else:
            out.append(tok)
    return " ".join(out)


def build_masked_dataframe(df: pd.DataFrame, tokens: Set[str]) -> pd.DataFrame:
    masked = df.copy()
    masked["customer_summary"] = masked["customer_summary"].apply(lambda t: mask_tokens(t, tokens))
    masked["agent_summary"] = masked["agent_summary"].apply(lambda t: mask_tokens(t, tokens))
    return masked


def embeddings_from_df(df: pd.DataFrame, model_name: str, batch_size: int, max_length: int, device):
    tokenizer, model = ge.load_model(model_name, device)
    cust_emb = ge.encode_texts(df["customer_summary"].tolist(), tokenizer, model, device, batch_size=batch_size, max_length=max_length)
    agent_emb = ge.encode_texts(df["agent_summary"].tolist(), tokenizer, model, device, batch_size=batch_size, max_length=max_length)
    return np.concatenate([cust_emb, agent_emb], axis=1)


def train_auc(X: np.ndarray, y: np.ndarray, seed: int, test_size: float):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    return auc


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Mask high-impact tokens and measure AUC drop")
    ap.add_argument("--data", type=Path, default=Path("leads.csv"))
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings.npz"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sentiment", type=Path, default=None, help="Optional sentiment features file (ignored for masking, only embedding impact tested)")
    ap.add_argument("--top-n", type=int, default=20, help="Top N positive and N negative tokens to mask (total 2N)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if not args.data.exists() or not args.embeddings.exists():
        raise SystemExit("Required files missing (data or embeddings)")

    df = pd.read_csv(args.data)
    y = df["conversion_label"].to_numpy()

    # Compute token log-odds
    combined_text = df["customer_summary"].astype(str) + " " + df["agent_summary"].astype(str)
    tok_df = compute_log_odds_tokens(combined_text, y)
    top_pos = tok_df.sort_values("log_odds", ascending=False).head(args.top_n)
    top_neg = tok_df.sort_values("log_odds").head(args.top_n)
    mask_set = set(top_pos["token"]).union(top_neg["token"])
    LOGGER.info("Masking %d tokens (top %d pos + top %d neg)", len(mask_set), args.top_n, args.top_n)

    # Load original embeddings
    X_orig = bm.load_embeddings(args.embeddings)

    # Generate masked embeddings
    masked_df = build_masked_dataframe(df, mask_set)
    device = ge.detect_device(prefer_gpu=not args.no_gpu)
    LOGGER.info("Device: %s", device)
    X_masked = embeddings_from_df(masked_df, args.model, args.batch_size, args.max_length, device)

    # Train/test AUCs (embeddings only)
    orig_auc = train_auc(X_orig, y, seed=args.seed, test_size=args.test_size)
    masked_auc = train_auc(X_masked, y, seed=args.seed, test_size=args.test_size)
    delta = orig_auc - masked_auc
    LOGGER.info("Original AUC: %.4f | Masked AUC: %.4f | Delta: %.4f", orig_auc, masked_auc, delta)

    # Average masked token count per doc (before vs after)
    def count_masked(txt: str) -> int:
        return sum(1 for t in tokenize(txt) if t.lower() in mask_set)
    avg_masked_cust = df["customer_summary"].apply(count_masked).mean()
    avg_masked_agent = df["agent_summary"].apply(count_masked).mean()

    result = {
        "top_pos_tokens": top_pos["token"].tolist(),
        "top_neg_tokens": top_neg["token"].tolist(),
        "orig_auc": orig_auc,
        "masked_auc": masked_auc,
        "auc_delta": delta,
        "avg_masked_tokens_customer": avg_masked_cust,
        "avg_masked_tokens_agent": avg_masked_agent,
        "total_masked_vocab": len(mask_set),
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        LOGGER.info("Wrote results -> %s", args.output)
    else:
        LOGGER.info("Result JSON: %s", json.dumps(result, indent=2))

if __name__ == "__main__":  # pragma: no cover
    main()
