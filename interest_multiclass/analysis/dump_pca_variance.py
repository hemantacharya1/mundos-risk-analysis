"""Dump PCA variance details for the saved PCA transform.
Usage:
  uv run python -m interest_multiclass.analysis.dump_pca_variance \
    --pca-file interest_multiclass/artifacts/pca_transform.pkl \
    --output interest_multiclass/analysis/pca_variance_summary.json
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np

def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument('--pca-file', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--top', type=int, default=10, help='How many individual component variances to list')
    args = ap.parse_args()

    if not args.pca_file.exists():
        raise SystemExit(f"PCA file not found: {args.pca_file}")
    with open(args.pca_file, 'rb') as f:
        pca = pickle.load(f)
    if not hasattr(pca, 'explained_variance_ratio_'):
        raise SystemExit('Loaded object does not appear to be a fitted PCA (missing explained_variance_ratio_).')

    ratios = np.array(pca.explained_variance_ratio_, dtype=float)
    cumulative = np.cumsum(ratios)
    total_components = int(len(ratios))
    ninetyfive_idx = int(np.searchsorted(cumulative, 0.95) + 1)
    ninety_idx = int(np.searchsorted(cumulative, 0.90) + 1)
    ninetynine_idx = int(np.searchsorted(cumulative, 0.99) + 1)

    summary = {
        'total_components_saved': total_components,
        'cumulative_variance_final': float(cumulative[-1]),
        'components_for_90pct': ninety_idx,
        'components_for_95pct': ninetyfive_idx,
        'components_for_99pct': ninetynine_idx,
        'chosen_components': int(pca.n_components_) if hasattr(pca, 'n_components_') else total_components,
        'top_component_variances': ratios[: args.top].round(6).tolist(),
        'top_cumulative': cumulative[: args.top].round(6).tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote PCA variance summary -> {args.output}")

if __name__ == '__main__':  # pragma: no cover
    main()
