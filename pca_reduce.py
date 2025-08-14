"""Reduce embedding dimensionality with PCA and optionally append sentiment features.

Usage examples:
  uv run python pca_reduce.py --embeddings embeddings.npz --n-components 50 --output pca_features.npz
  uv run python pca_reduce.py --embeddings embeddings.npz --sentiment sentiment.csv --n-components 50 --output pca_sentiment.npz

Outputs an NPZ containing:
  pca: (n_samples, n_components) float32
  combined: if sentiment provided, horizontal concat of pca + sentiment
  explained_variance_ratio: 1D array of per-component variance ratios
  cumulative_variance: cumulative sum (for convenience)
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import baseline_model as bm

LOGGER = logging.getLogger("pca_reduce")


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="PCA reduction for embeddings")
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings.npz"))
    ap.add_argument("--sentiment", type=Path, default=None, help="Optional sentiment features file (csv/parquet)")
    ap.add_argument("--n-components", type=int, default=50)
    ap.add_argument("--standardize", action="store_true", help="Apply StandardScaler before PCA")
    ap.add_argument("--output", type=Path, default=Path("pca_features.npz"))
    ap.add_argument("--plot", action="store_true", help="Generate variance plot as PNG next to NPZ")
    ap.add_argument("--save-components", action="store_true", help="Save PCA components matrix to separate .npy file")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if not args.embeddings.exists():
        raise SystemExit("Embeddings file missing")

    X_emb = bm.load_embeddings(args.embeddings)
    LOGGER.info("Loaded embeddings shape=%s", X_emb.shape)

    scaler = None
    X_input = X_emb
    if args.standardize:
        scaler = StandardScaler()
        X_input = scaler.fit_transform(X_emb)
        LOGGER.info("Applied StandardScaler")

    n_comp = min(args.n_components, X_input.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_input).astype(np.float32)
    LOGGER.info("PCA reduced shape=%s explained_variance=%.4f", X_pca.shape, pca.explained_variance_ratio_.sum())

    X_sent = None
    combined = None
    if args.sentiment:
        try:
            X_sent = bm.load_sentiment(args.sentiment)
            combined = bm.combine_features(X_pca, X_sent)
            LOGGER.info("Combined feature shape=%s", combined.shape)
        except Exception as e:
            raise SystemExit(f"Failed loading sentiment features: {e}")

    np.savez_compressed(
        args.output,
        pca=X_pca,
        combined=combined if combined is not None else X_pca,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        cumulative_variance=np.cumsum(pca.explained_variance_ratio_).astype(np.float32),
        n_components=np.array([n_comp], dtype=np.int32),
        standardized=np.array([1 if args.standardize else 0], dtype=np.int32),
    )
    LOGGER.info("Saved PCA features -> %s", args.output)

    # Meta JSON
    meta = {
        "embeddings_file": str(args.embeddings),
        "sentiment_file": str(args.sentiment) if args.sentiment else None,
        "output_file": str(args.output),
        "n_components": n_comp,
        "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "standardized": args.standardize,
    }
    if args.save_components:
        comp_path = args.output.with_suffix('.components.npy')
        np.save(comp_path, pca.components_.astype(np.float32))
        meta["components_file"] = str(comp_path)
        LOGGER.info("Saved components -> %s", comp_path)

    if args.plot:
        fig, ax = plt.subplots(figsize=(6,4))
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        ax.bar(range(1, len(evr)+1), evr, color='#4C72B0', label='Per-component')
        ax.plot(range(1, len(evr)+1), cum, color='#55A868', marker='o', label='Cumulative')
        ax.set_xlabel('Component')
        ax.set_ylabel('Variance Ratio')
        ax.set_title(f'PCA Variance (sum={evr.sum():.3f})')
        ax.grid(alpha=0.3, linestyle=':')
        ax.legend()
        plot_path = args.output.with_suffix('.variance.png')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        meta["variance_plot"] = str(plot_path)
        LOGGER.info("Saved variance plot -> %s", plot_path)
    with open(args.output.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    LOGGER.info("Wrote meta -> %s", args.output.with_suffix('.json'))

if __name__ == "__main__":  # pragma: no cover
    main()
