"""Train a PCA + (optional sentiment) + LogisticRegression pipeline with optional calibration and component sweep.

Usage examples:
  uv run python train_pca_pipeline.py --embeddings embeddings.npz --sentiment sentiment.csv \
      --components 10,20,30,40,50 --standardize --calibrate isotonic \
      --results pca_sweep_results.json --save-best best_pca_pipeline.pkl

Outputs:
  * results JSON: per-component metrics (variance_sum, test_auc, brier (if calibrated), feature_dim)
  * best pipeline pickle: dict with keys {scaler, pca, model, sentiment_used, n_components, sentiment_dim}

Selection: smallest n_components whose AUC >= (best_auc - tolerance).
"""
from __future__ import annotations
import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import baseline_model as bm

LOGGER = logging.getLogger("train_pca_pipeline")


def parse_components(comp_str: str) -> List[int]:
    return [int(c.strip()) for c in comp_str.split(",") if c.strip()]


def build_features(X_emb: np.ndarray, X_sent: np.ndarray | None) -> np.ndarray:
    return bm.combine_features(X_emb, X_sent)


def train_single(
    X_emb: np.ndarray,
    y: np.ndarray,
    n_components: int,
    standardize: bool,
    X_sent: np.ndarray | None,
    test_size: float,
    seed: int,
    calibrate: str | None,
    log_level: int,
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X_emb, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    n_comp = min(n_components, X_train.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    if X_sent is not None:
        # Align sentiment rows with original ordering (no shuffling yet) then slice
        # Because we split after passing X_emb; we can rebuild indices
        idx = np.arange(len(X_emb))
        tr_mask = np.zeros(len(X_emb), dtype=bool)
        tr_mask[: len(X_train_pca)] = True  # This approach invalid (ordering lost) -> use indices from train_test_split
        # Better: re-split indices directly
    
    # Re-perform split to capture indices for sentiment slicing
    idx_train, idx_test = train_test_split(np.arange(len(X_emb)), test_size=test_size, random_state=seed, stratify=y)
    X_sent_train = X_sent[idx_train] if X_sent is not None else None
    X_sent_test = X_sent[idx_test] if X_sent is not None else None

    Xtr = build_features(X_train_pca, X_sent_train)
    Xte = build_features(X_test_pca, X_sent_test)

    base_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)

    calibrated = False
    if calibrate and calibrate.lower() in {"isotonic", "sigmoid"}:
        clf = CalibratedClassifierCV(base_clf, method=calibrate.lower(), cv=3)
        clf.fit(Xtr, y_train)
        calibrated = True
    else:
        clf = base_clf
        clf.fit(Xtr, y_train)

    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba) if calibrated else None

    return {
        "n_components": n_comp,
        "variance_sum": float(pca.explained_variance_ratio_.sum()),
        "test_auc": float(auc),
        "brier": float(brier) if brier is not None else None,
        "feature_dim": int(Xtr.shape[1]),
        "calibrated": calibrated,
        "model": clf,
        "scaler": scaler,
        "pca": pca,
        "sentiment_dim": 0 if X_sent is None else int(X_sent.shape[1]),
        "idx_train": idx_train.tolist(),
        "idx_test": idx_test.tolist(),
    }


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train PCA pipeline with sweep")
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings.npz"))
    ap.add_argument("--sentiment", type=Path, default=None)
    ap.add_argument("--components", default="10,20,30,40,50")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", default=None, help="None|isotonic|sigmoid")
    ap.add_argument("--results", type=Path, default=Path("pca_sweep_results.json"))
    ap.add_argument("--save-best", type=Path, default=None)
    ap.add_argument("--tolerance", type=float, default=0.001, help="AUC tolerance when choosing minimal n_components")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    X_emb = bm.load_embeddings(args.embeddings)
    y = pd.read_csv("leads.csv")["conversion_label"].to_numpy()
    if len(X_emb) != len(y):
        raise SystemExit("Embeddings and labels size mismatch")

    X_sent = None
    if args.sentiment:
        X_sent = bm.load_sentiment(args.sentiment)
        if len(X_sent) != len(y):
            raise SystemExit("Sentiment rows mismatch labels")

    comp_list = parse_components(args.components)
    results: List[dict] = []
    for c in comp_list:
        res = train_single(
            X_emb=X_emb,
            y=y,
            n_components=c,
            standardize=args.standardize,
            X_sent=X_sent,
            test_size=args.test_size,
            seed=args.seed,
            calibrate=args.calibrate,
            log_level=logging.getLogger().level,
        )
        LOGGER.info(
            "n=%d variance=%.4f AUC=%.4f%s feat_dim=%d", res["n_components"], res["variance_sum"], res["test_auc"],
            f" brier={res['brier']:.4f}" if res['brier'] is not None else "", res["feature_dim"],
        )
        results.append({k: v for k, v in res.items() if k not in {"model", "scaler", "pca", "idx_train", "idx_test"}})

    best_auc = max(r["test_auc"] for r in results)
    chosen = sorted([r for r in results if r["test_auc"] >= best_auc - args.tolerance], key=lambda r: r["n_components"])[0]
    LOGGER.info("Chosen n_components=%d (AUC=%.4f best=%.4f tolerance=%.4f)", chosen["n_components"], chosen["test_auc"], best_auc, args.tolerance)

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": chosen, "tolerance": args.tolerance}, f, indent=2)
    LOGGER.info("Wrote results -> %s", args.results)

    if args.save_best:
        # Retrain pipeline on FULL dataset for deployment using chosen n_components
        scaler = StandardScaler() if args.standardize else None
        X_proc = scaler.fit_transform(X_emb) if scaler else X_emb
        pca = PCA(n_components=chosen["n_components"], random_state=args.seed)
        X_pca = pca.fit_transform(X_proc)
        X_final = build_features(X_pca, X_sent)
        base_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        if args.calibrate and args.calibrate.lower() in {"isotonic", "sigmoid"}:
            clf = CalibratedClassifierCV(base_clf, method=args.calibrate.lower(), cv=3)
            clf.fit(X_final, y)
        else:
            clf = base_clf
            clf.fit(X_final, y)
        pipeline_obj = {
            "scaler": scaler,
            "pca": pca,
            "model": clf,
            "n_components": chosen["n_components"],
            "variance_sum": chosen["variance_sum"],
            "sentiment_used": X_sent is not None,
            "feature_dim": X_final.shape[1],
            "calibrated": bool(args.calibrate and args.calibrate.lower() in {"isotonic", "sigmoid"}),
        }
        with open(args.save_best, "wb") as f:
            pickle.dump(pipeline_obj, f)
        LOGGER.info("Saved best pipeline -> %s", args.save_best)

if __name__ == "__main__":  # pragma: no cover
    main()
