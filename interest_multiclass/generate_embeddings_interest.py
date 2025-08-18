"""Wrapper to generate embeddings for leads_1.csv into interest_multiclass folder.
Uses existing root generate_embeddings logic via import to stay DRY.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from config import DATA_FILE, EMBEDDINGS_FILE, EMB_MODEL


def detect_device(prefer_gpu: bool = True):
    if prefer_gpu and torch.cuda.is_available():  # pragma: no cover
        return torch.device("cuda")
    return torch.device("cpu")


def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(1)
    counts = mask.sum(1).clamp(min=1)
    return summed / counts


def encode(texts, tokenizer, model, device, batch_size=32, max_length=256):
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = ["" if t is None else str(t) for t in texts[i:i+batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().numpy()
        outs.append(pooled)
    return np.vstack(outs)


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_FILE)
    ap.add_argument("--model", default=EMB_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data file missing: {args.data}")
    df = pd.read_csv(args.data)
    if "customer_summary" not in df.columns or "agent_summary" not in df.columns:
        raise SystemExit("Required text columns missing")

    device = detect_device(prefer_gpu=not args.no_gpu)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    cust = encode(df["customer_summary"].tolist(), tokenizer, model, device, args.batch_size, args.max_length)
    agent = encode(df["agent_summary"].tolist(), tokenizer, model, device, args.batch_size, args.max_length)
    combined = np.concatenate([cust, agent], axis=1)
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBEDDINGS_FILE, combined=combined, customer=cust, agent=agent, model=args.model)
    print(f"Saved embeddings -> {EMBEDDINGS_FILE} shape={combined.shape}")

if __name__ == "__main__":
    main()
