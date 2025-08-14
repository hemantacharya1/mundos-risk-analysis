"""Show classification report for a saved logistic regression model.

Supports models trained via:
  * baseline_model.py (embeddings +/- sentiment or PCA features)
  * train_logreg_reduced.py (precomputed PCA NPZ)

It reconstructs the original train/test split using the metrics JSON
(test_size + random_state) so the report matches what was printed during
training (aside from floating point differences).

Usage:
  uv run python show_classification_report.py --model artifacts/baseline_logreg_*.pkl

Optional flags if automatic feature reconstruction fails:
  --embeddings embeddings.npz
  --sentiment sentiment.csv
  --pca-file pca_50_sentiment_std.npz  (if model trained on PCA file)
  --pca-only  (force using 'pca' array in PCA NPZ)

If --model is omitted the script uses the most recent baseline_logreg_*.pkl
in artifacts/.
"""
from __future__ import annotations
import argparse
import glob
import json
import logging
import pickle
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import baseline_model as bm

LOGGER = logging.getLogger("show_report")


def find_latest_model(art_dir: Path) -> Optional[Path]:
    cand = sorted(art_dir.glob("baseline_logreg_*.pkl"))
    return cand[-1] if cand else None


def infer_metrics_path(model_path: Path) -> Path:
    ts = model_path.stem.split("_")[-1]
    m = model_path.parent / f"metrics_{ts}.json"
    return m


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def attempt_pca_match(feature_dim: int) -> tuple[Optional[np.ndarray], Optional[Path]]:
    """Scan pca_*.npz files and return array & path whose 'combined' or 'pca' matches feature_dim."""
    candidates = []
    for p_str in glob.glob("pca_*.npz"):
        p = Path(p_str)
        try:
            with np.load(p) as npz:
                if "combined" in npz and npz["combined"].shape[1] == feature_dim:
                    candidates.append((npz["combined"].astype(np.float32), p, "combined"))
                elif "pca" in npz and npz["pca"].shape[1] == feature_dim:
                    candidates.append((npz["pca"].astype(np.float32), p, "pca"))
        except Exception:  # pragma: no cover
            continue
    if not candidates:
        return None, None
    # Prefer combined over pca if both exist
    candidates.sort(key=lambda t: 0 if t[2] == "combined" else 1)
    arr, path, _ = candidates[0]
    return arr, path


def reconstruct_features(args, clf, metrics) -> tuple[np.ndarray, str]:
    feature_dim = clf.coef_.shape[1]
    # If PCA file explicitly provided
    if args.pca_file:
        Xp = bm.load_pca_features(args.pca_file, use_pca_only=args.pca_only)
        if Xp.shape[1] != feature_dim:
            raise SystemExit(f"Provided PCA file feature dim {Xp.shape[1]} != model dim {feature_dim}")
        return Xp, f"PCA ({'pca' if args.pca_only else 'combined'} from {args.pca_file.name})"

    # Try embeddings + optional sentiment path
    if args.embeddings.exists():
        X_emb = bm.load_embeddings(args.embeddings)
        emb_dim = X_emb.shape[1]
        X_sent = None
        sent_dim = 0
        if args.sentiment and args.sentiment.exists():
            try:
                X_sent = bm.load_sentiment(args.sentiment)
                sent_dim = X_sent.shape[1]
            except Exception as e:  # pragma: no cover
                LOGGER.warning("Failed loading sentiment: %s", e)
        # Exact matches
        if feature_dim == emb_dim and X_sent is None:
            return X_emb, "embeddings only"
        if X_sent is not None and feature_dim == emb_dim + sent_dim:
            return np.hstack([X_emb, X_sent]), "embeddings + sentiment"

    # Last resort: scan PCA files automatically
    Xp, ppath = attempt_pca_match(feature_dim)
    if Xp is not None:
        return Xp, f"PCA auto-match ({ppath.name})"

    raise SystemExit("Could not reconstruct feature matrix; supply --pca-file or matching data")


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Show classification report for saved model")
    ap.add_argument("--model", type=Path, default=None, help="Path to model pickle (defaults to latest baseline model)")
    ap.add_argument("--metrics", type=Path, default=None, help="Metrics JSON path (auto from model if omitted)")
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings.npz"))
    ap.add_argument("--sentiment", type=Path, default=None)
    ap.add_argument("--pca-file", type=Path, default=None)
    ap.add_argument("--pca-only", action="store_true")
    ap.add_argument("--data", type=Path, default=Path("leads.csv"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    model_path = args.model
    if model_path is None:
        model_path = find_latest_model(Path("artifacts"))
        if model_path is None:
            raise SystemExit("No model found in artifacts/")
        LOGGER.info("Using latest model: %s", model_path)

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    metrics_path = args.metrics or infer_metrics_path(model_path)
    metrics = load_metrics(metrics_path)

    if not args.data.exists():
        raise SystemExit("Data CSV not found")
    y = pd.read_csv(args.data)["conversion_label"].to_numpy()

    X, origin = reconstruct_features(args, clf, metrics)
    if len(X) != len(y):
        raise SystemExit("Feature rows mismatch labels")

    rs = metrics.get("random_state", 42)
    test_size = metrics.get("test_size", 0.2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)

    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)
    report = classification_report(y_te, preds, digits=3)
    auc = roc_auc_score(y_te, proba)

    print(f"Model: {model_path.name}\nMetrics: {metrics_path.name}\nOrigin: {origin}\nAUC: {auc:.6f}\n")
    print(report)

if __name__ == "__main__":  # pragma: no cover
    main()
