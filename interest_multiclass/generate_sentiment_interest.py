"""Generate sentiment features for leads_1.csv into interest_multiclass folder.
Creates sentiment_interest.csv with 7 sentiment-derived columns mirroring original schema.
Usage:
  uv run python interest_multiclass/generate_sentiment_interest.py --signed-only   # (2 cols + derived gap -> still 3)
  uv run python interest_multiclass/generate_sentiment_interest.py                  # full 7-column set
Environment:
  SKIP_SENTIMENT_MODEL=1  -> dummy neutral scores (fast, for tests)
"""
from __future__ import annotations
import argparse, json, os, logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from config import DATA_FILE, SENTIMENT_FILE, SENT_MODEL

LOGGER = logging.getLogger("sentiment_interest")

REQUIRED_COLS = ["customer_summary", "agent_summary"]


def detect_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():  # pragma: no cover
        return "cuda"
    return "cpu"


def load_analyzer(model_name: str, device: str):
    if os.environ.get("SKIP_SENTIMENT_MODEL") == "1":
        LOGGER.warning("Using dummy neutral analyzer (SKIP_SENTIMENT_MODEL=1)")
        def dummy(texts):  # type: ignore
            return [{"label": "POSITIVE", "score": 0.5} for _ in texts]
        return dummy
    return pipeline("sentiment-analysis", model=model_name, device=0 if device == "cuda" else -1)


def compute(texts, analyzer, batch_size: int = 32):
    signed, p_pos, p_neg = [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        safe = ["neutral" if (t is None or str(t).strip()=="") else str(t) for t in batch]
        try:
            outputs = analyzer(safe)
        except Exception:  # pragma: no cover
            outputs = [{"label": "POSITIVE", "score": 0.5}] * len(batch)
        for o in outputs:
            label = o.get("label", "POSITIVE").upper()
            score = float(o.get("score", 0.5))
            if label.startswith("NEG"):
                signed.append(-score)
                p_pos.append(1-score)
                p_neg.append(score)
            else:
                signed.append(score)
                p_pos.append(score)
                p_neg.append(1-score)
    return np.array(signed, dtype=np.float32), np.array(p_pos, dtype=np.float32), np.array(p_neg, dtype=np.float32)


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_FILE)
    ap.add_argument("--model", default=SENT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--signed-only", action="store_true", help="Only signed sentiment + gap (reduces to 3 derived cols)")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if not args.data.exists():
        raise SystemExit(f"Data file missing: {args.data}")
    df = pd.read_csv(args.data)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise SystemExit(f"Required column missing: {c}")

    device = detect_device(prefer_gpu=not args.no_gpu)
    LOGGER.info("Device: %s", device)
    analyzer = load_analyzer(args.model, device)

    cust_signed, cust_pos, cust_neg = compute(df["customer_summary"].tolist(), analyzer, batch_size=args.batch_size)
    agent_signed, agent_pos, agent_neg = compute(df["agent_summary"].tolist(), analyzer, batch_size=args.batch_size)

    out = {
        "sentiment_cust": cust_signed,
        "sentiment_agent": agent_signed,
    }
    if not args.signed_only:
        out.update({
            "sentiment_cust_pos": cust_pos,
            "sentiment_cust_neg": cust_neg,
            "sentiment_agent_pos": agent_pos,
            "sentiment_agent_neg": agent_neg,
        })
    # gap always included for parity with original 7-col layout
    out["sentiment_gap"] = agent_signed - cust_signed

    out_df = pd.DataFrame(out)
    SENTIMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(SENTIMENT_FILE, index=False)

    meta = {
        "rows": len(out_df),
        "signed_only": args.signed_only,
        "model": args.model,
        "device": device,
        "file": str(SENTIMENT_FILE),
        "columns": out_df.columns.tolist(),
    }
    with open(Path(SENTIMENT_FILE).with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    LOGGER.info("Wrote sentiment -> %s (cols=%d)", SENTIMENT_FILE, len(out_df.columns))

if __name__ == "__main__":
    main()
