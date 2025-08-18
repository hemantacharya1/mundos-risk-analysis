"""Generate sentence embeddings for leads dataset.

Features:
* Loads `leads.csv` via pandas.
* Uses Hugging Face model (default: sentence-transformers/all-MiniLM-L6-v2).
* Computes embeddings for customer_summary and agent_summary with mean pooling (mask-aware).
* Concatenates both embeddings -> final feature vector per row.
* Saves to .npz (NumPy) and/or .parquet if requested.
* Supports batching, limiting samples, device auto-detect.

CLI examples (uv):
    uv run python generate_embeddings.py --output embeddings.npz
    uv run python generate_embeddings.py --output embeddings.npz --parquet embeddings.parquet --batch-size 64
    uv run python generate_embeddings.py --max-samples 200 --log-level DEBUG
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_FILE = Path("leads.csv")
LOGGER = logging.getLogger("embeddings")

def detect_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():  # pragma: no cover
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(model_name: str, device: torch.device) -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def encode_texts(texts: List[str], tokenizer, model, device: torch.device, batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    all_embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked = token_embeddings * attention_mask
        sums = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        mean_pooled = (sums / counts).cpu().numpy()
        all_embs.append(mean_pooled)
        LOGGER.debug("Encoded batch %d-%d", i, i + len(batch))
    return np.vstack(all_embs)

def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate embeddings for leads dataset")
    parser.add_argument("--data", type=Path, default=DATA_FILE, help="Path to leads.csv")
    parser.add_argument("--model", default=MODEL_NAME, help="HF model name")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--output", type=Path, default=Path("embeddings.npz"), help="Output .npz path")
    parser.add_argument("--parquet", type=Path, default=None, help="Optional Parquet output path")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU")
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
    tokenizer, model = load_model(args.model, device)
    LOGGER.info("Model hidden size: %d", model.config.hidden_size)

    cust_emb = encode_texts(df["customer_summary"].tolist(), tokenizer, model, device, batch_size=args.batch_size, max_length=args.max_length)
    agent_emb = encode_texts(df["agent_summary"].tolist(), tokenizer, model, device, batch_size=args.batch_size, max_length=args.max_length)

    combined = np.concatenate([cust_emb, agent_emb], axis=1)
    LOGGER.info("Embeddings shape (cust, agent, combined): %s %s %s", cust_emb.shape, agent_emb.shape, combined.shape)

    np.savez_compressed(args.output, customer=cust_emb, agent=agent_emb, combined=combined, model=args.model)
    LOGGER.info("Saved embeddings NPZ -> %s", args.output)
    if args.parquet:
        emb_df = pd.DataFrame(combined)
        emb_df.to_parquet(args.parquet, index=False)
        LOGGER.info("Saved embeddings Parquet -> %s", args.parquet)

if __name__ == "__main__":
    main()
