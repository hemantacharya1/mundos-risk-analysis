"""Train logistic regression on precomputed reduced features (PCA +/- sentiment).

Source features come from a PCA NPZ produced by `pca_reduce.py` (or compatible
file containing arrays `pca` and `combined`). By default we use the `combined`
array which is PCA components plus optional appended sentiment columns. You can
switch to PCA-only with `--pca-only`.

Artifacts saved (if `--save-model`):
  * model pickle: logistic regression classifier
  * metrics JSON: test metrics & configuration

Example:
  uv run python train_logreg_reduced.py --pca-file pca_50_sentiment_std.npz --cv-folds 5 --search --save-model
"""
from __future__ import annotations
import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

import baseline_model as bm

LOGGER = logging.getLogger("train_logreg_reduced")


def load_reduced(path: Path, pca_only: bool = False) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"PCA feature file not found: {path}")
    with np.load(path) as npz:
        if not pca_only and "combined" in npz:
            arr = npz["combined"]
        else:
            arr = npz["pca"]
    return arr.astype(np.float32)


def cross_val_auc(X: np.ndarray, y: np.ndarray, C: float, folds: int, seed: int) -> List[float]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: List[float] = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        scores.append(roc_auc_score(y[te], prob))
    return scores


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train logistic regression on reduced (PCA) features")
    ap.add_argument("--pca-file", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("leads.csv"))
    ap.add_argument("--pca-only", action="store_true", help="Use only PCA components array, ignore combined")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--cv-folds", type=int, default=0, help="If >1 run CV on training set")
    ap.add_argument("--search", action="store_true", help="C grid search (needs CV)")
    ap.add_argument("--c-grid", default="0.1,0.5,1,2,5")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if not args.data.exists():
        raise SystemExit("Data file missing")

    X = load_reduced(args.pca_file, pca_only=args.pca_only)
    y = pd.read_csv(args.data)["conversion_label"].to_numpy()
    if len(X) != len(y):
        raise SystemExit(f"Feature rows {len(X)} != label rows {len(y)}")
    LOGGER.info("Loaded reduced features shape=%s (pca_only=%s)", X.shape, args.pca_only)

    cv_results = None
    best_C = 1.0
    search_details = None
    if args.cv_folds and args.cv_folds > 1:
        base_scores = cross_val_auc(X, y, C=1.0, folds=args.cv_folds, seed=args.random_state)
        LOGGER.info("CV AUC (C=1.0) mean=%.4f std=%.4f folds=%s", np.mean(base_scores), np.std(base_scores), base_scores)
        cv_results = {"C": 1.0, "folds": args.cv_folds, "scores": base_scores, "mean": float(np.mean(base_scores)), "std": float(np.std(base_scores))}
        if args.search:
            c_values = [float(c.strip()) for c in args.c_grid.split(",") if c.strip()]
            best_mean = -1.0
            per_c = []
            for c in c_values:
                scores = cross_val_auc(X, y, C=c, folds=args.cv_folds, seed=args.random_state)
                mean_s = float(np.mean(scores))
                per_c.append({"C": c, "scores": scores, "mean": mean_s})
                LOGGER.info("Search C=%.3g mean=%.4f", c, mean_s)
                if mean_s > best_mean:
                    best_mean = mean_s
                    best_C = c
            search_details = {"results": per_c, "best_C": best_C, "best_mean": best_mean}
            LOGGER.info("Best C=%.3g (mean AUC=%.4f)", best_C, best_mean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=best_C)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, digits=3)
    print(report)
    print(f"ROC AUC: {auc:.4f} (C={best_C})")

    metrics = {
        "roc_auc": float(auc),
        "C": best_C,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "cv": cv_results,
        "search": search_details,
        "feature_shape": X.shape,
        "pca_file": str(args.pca_file),
        "pca_only": args.pca_only,
    }

    if args.save_model:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        model_path = args.out_dir / f"logreg_reduced_{ts}.pkl"
        metrics_path = args.out_dir / f"metrics_reduced_{ts}.json"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        LOGGER.info("Saved reduced model -> %s", model_path)
        LOGGER.info("Saved metrics -> %s", metrics_path)

if __name__ == "__main__":  # pragma: no cover
    main()
