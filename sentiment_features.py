"""Generate sentiment scores for customer and agent summaries.

Signed score: +prob_positive, -prob_positive_if_negative (approx via model output).
Also stores raw p_pos / p_neg if requested.

Usage examples (uv):
  uv run python sentiment_features.py --output sentiment.parquet
  uv run python sentiment_features.py --merge-csv --signed-only
  uv run python sentiment_features.py --max-samples 200 --log-level DEBUG

Environment shortcut for tests / offline:
  Set SKIP_SENTIMENT_MODEL=1 to use a dummy neutral model (avoids download).
"""
from __future__ import annotations
import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import numpy as np
import pandas as pd
import torch
from transformers import pipeline

LOGGER = logging.getLogger("sentiment")
DATA_FILE = Path("leads.csv")
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

def detect_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():  # pragma: no cover
        return "cuda"
    return "cpu"

def load_analyzer(model_name: str, device: str):
    if os.environ.get("SKIP_SENTIMENT_MODEL") == "1":  # test / offline mode
        LOGGER.warning("Using dummy neutral sentiment analyzer (SKIP_SENTIMENT_MODEL=1)")
        def dummy(texts: Sequence[str]):  # type: ignore
            return [{"label": "POSITIVE", "score": 0.5} for _ in texts]
        return dummy
    return pipeline("sentiment-analysis", model=model_name, device=0 if device == "cuda" else -1)

@dataclass
class SentimentResult:
    signed: np.ndarray
    p_pos: np.ndarray
    p_neg: np.ndarray
    errors: int
    empty: int

def compute_sentiment(texts: List[str], analyzer, batch_size: int = 32) -> SentimentResult:
    signed_scores: List[float] = []
    p_pos_list: List[float] = []
    p_neg_list: List[float] = []
    errors = 0
    empty = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        mask_empty = [((t is None) or (str(t).strip() == "")) for t in batch]
        to_infer = [t if not m else "neutral" for t, m in zip(batch, mask_empty)]
        try:
            outputs = analyzer(to_infer)
        except Exception:  # pragma: no cover
            outputs = [{"label": "POSITIVE", "score": 0.5}] * len(batch)
            errors += len(batch)
        for o, is_empty in zip(outputs, mask_empty):
            if is_empty:
                empty += 1
            label = o.get("label", "POSITIVE")
            score = float(o.get("score", 0.5))
            if label.upper().startswith("NEG"):
                p_pos = 1 - score
                p_neg = score
                signed = -score
            else:
                p_pos = score
                p_neg = 1 - score
                signed = score
            signed_scores.append(signed)
            p_pos_list.append(p_pos)
            p_neg_list.append(p_neg)
    return SentimentResult(
        signed=np.array(signed_scores, dtype=np.float32),
        p_pos=np.array(p_pos_list, dtype=np.float32),
        p_neg=np.array(p_neg_list, dtype=np.float32),
        errors=errors,
        empty=empty,
    )

def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate sentiment features")
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("sentiment.parquet"))
    parser.add_argument("--merge-csv", action="store_true")
    parser.add_argument("--signed-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}")

    df = pd.read_csv(args.data)
    if args.max_samples:
        df = df.iloc[: args.max_samples].copy()
        LOGGER.info("Using subset: %d rows", len(df))

    device = detect_device(prefer_gpu=not args.no_gpu)
    LOGGER.info("Device: %s", device)
    analyzer = load_analyzer(args.model, device)

    cust_res = compute_sentiment(df["customer_summary"].tolist(), analyzer, batch_size=args.batch_size)
    agent_res = compute_sentiment(df["agent_summary"].tolist(), analyzer, batch_size=args.batch_size)
    LOGGER.info("Finished sentiment scoring. empty_cust=%d empty_agent=%d", cust_res.empty, agent_res.empty)

    out_df = pd.DataFrame({
        "sentiment_cust": cust_res.signed,
        "sentiment_agent": agent_res.signed,
    })
    if not args.signed_only:
        out_df["sentiment_cust_pos"] = cust_res.p_pos
        out_df["sentiment_cust_neg"] = cust_res.p_neg
        out_df["sentiment_agent_pos"] = agent_res.p_pos
        out_df["sentiment_agent_neg"] = agent_res.p_neg
        out_df["sentiment_gap"] = out_df["sentiment_agent"] - out_df["sentiment_cust"]

    # Determine output handling with graceful fallback if parquet engine missing
    meta = {"model": args.model, "rows": len(df), "signed_only": args.signed_only, "device": device}
    if args.merge_csv:
        merged = pd.concat([df.reset_index(drop=True), out_df], axis=1)
        out_csv = args.output.with_suffix(".csv")
        merged.to_csv(out_csv, index=False)
        meta["output_file"] = str(out_csv)
        meta["format"] = "csv"
        LOGGER.info("Wrote merged CSV -> %s", out_csv)
        meta_path = out_csv.with_suffix(".json")
    else:
        out_path = args.output
        wrote = False
        if out_path.suffix.lower() == ".parquet":
            try:
                out_df.to_parquet(out_path, index=False)
                wrote = True
                meta["format"] = "parquet"
                meta["output_file"] = str(out_path)
                LOGGER.info("Wrote sentiment features -> %s", out_path)
            except ImportError:
                LOGGER.warning("pyarrow/fastparquet not installed; falling back to CSV")
        if not wrote:
            # Fallback to CSV (either suffix not parquet or parquet failed)
            out_csv = out_path.with_suffix(".csv") if out_path.suffix.lower() == ".parquet" else out_path.with_suffix(".csv")
            out_df.to_csv(out_csv, index=False)
            meta["format"] = "csv"
            meta["output_file"] = str(out_csv)
            LOGGER.info("Wrote sentiment features CSV -> %s", out_csv)
            out_path = out_csv
        meta_path = Path(meta["output_file"]).with_suffix(".json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    LOGGER.info("Wrote meta -> %s", meta_path)

if __name__ == "__main__":
    main()
