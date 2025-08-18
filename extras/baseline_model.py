"""Train a simple baseline classifier on precomputed embeddings.

Workflow:
1. Generate embeddings: uv run python generate_embeddings.py --output embeddings.npz
2. Train baseline: uv run python baseline_model.py --embeddings embeddings.npz --save-model

Features added:
* Stratified train/test split.
* Optional k-fold cross-validation AUC.
* Optional hyperparameter search over C values.
* Artifact saving: model.pkl + metrics.json (if --save-model given).
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

EMB_FILE = Path("embeddings.npz")
DATA_FILE = Path("leads.csv")

# Helper load functions kept simple so they can be unit-tested without invoking CLI.


def load_embeddings(path: Path) -> np.ndarray:
	"""Load combined embeddings matrix from NPZ (float32)."""
	with np.load(path) as npz:
		return npz["combined"].astype(np.float32)


def load_pca_features(path: Path, use_pca_only: bool = False) -> np.ndarray:
	"""Load PCA NPZ produced by pca_reduce.py.

	If use_pca_only is False (default), tries 'combined' first (PCA + optional sentiment),
	else loads the raw 'pca' array. Falls back to 'pca' if 'combined' missing.
	Returns float32 array.
	"""
	if not path.exists():
		raise FileNotFoundError(f"PCA file not found: {path}")
	with np.load(path) as npz:
		if not use_pca_only and "combined" in npz:
			arr = npz["combined"]
		else:
			arr = npz["pca"]
	return arr.astype(np.float32)


def load_sentiment(path: Path) -> np.ndarray:
	"""Load sentiment feature matrix from a Parquet or CSV file.

	Expected columns start with 'sentiment_'. Non-numeric columns are ignored.
	Returns 2D float32 numpy array.
	"""
	if not path.exists():
		raise FileNotFoundError(f"Sentiment features file not found: {path}")
	if path.suffix.lower() == ".parquet":  # pragma: no cover (depends on optional engine)
		df = pd.read_parquet(path)
	else:
		df = pd.read_csv(path)
	cols = [c for c in df.columns if c.startswith("sentiment_")]
	if not cols:
		raise ValueError("No sentiment_ columns found in sentiment features file")
	mat = df[cols].select_dtypes(include=["number"]).to_numpy(dtype=np.float32)
	if mat.ndim != 2:
		raise ValueError("Sentiment features matrix is not 2D after selection")
	return mat


def combine_features(emb: np.ndarray, sent: Optional[np.ndarray]) -> np.ndarray:
	"""Horizontally concatenate embeddings with sentiment features (if provided)."""
	if sent is None:
		return emb
	if len(emb) != len(sent):
		raise ValueError(f"Row mismatch embeddings={len(emb)} sentiment={len(sent)}")
	return np.hstack([emb, sent])


def cross_val_auc(X: np.ndarray, y: np.ndarray, C: float, folds: int, seed: int) -> List[float]:
	skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
	scores: List[float] = []
	for train_idx, test_idx in skf.split(X, y):
		clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
		clf.fit(X[train_idx], y[train_idx])
		prob = clf.predict_proba(X[test_idx])[:, 1]
		scores.append(roc_auc_score(y[test_idx], prob))
	return scores


def main():  # pragma: no cover
	parser = argparse.ArgumentParser(description="Train baseline model on embeddings (optionally + sentiment features)")
	parser.add_argument("--embeddings", type=Path, default=EMB_FILE, help="Path to embeddings NPZ (from generate_embeddings.py)")
	parser.add_argument("--data", type=Path, default=DATA_FILE, help="Original data CSV for labels")
	parser.add_argument("--sentiment", type=Path, default=None, help="Optional sentiment features file (csv or parquet)")
	parser.add_argument("--pca-file", type=Path, default=None, help="Optional PCA feature NPZ (overrides --embeddings + --sentiment)")
	parser.add_argument("--pca-use-pca", action="store_true", help="Use only PCA components from PCA NPZ (ignore combined array)")
	parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--log-level", default="INFO")
	parser.add_argument("--cv-folds", type=int, default=0, help="If >1 perform CV AUC estimate")
	parser.add_argument("--search", action="store_true", help="Hyperparameter search over C values (needs CV)")
	parser.add_argument("--c-grid", default="0.1,0.5,1,2,5", help="Comma list of C values for search")
	parser.add_argument("--save-model", action="store_true", help="Persist model + metrics artifacts")
	parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
	args = parser.parse_args()

	logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
	if not args.embeddings.exists():
		raise SystemExit("Embeddings file not found. Run generate_embeddings.py first.")
	if not args.data.exists():
		raise SystemExit("Data file not found.")

	y = pd.read_csv(args.data)["conversion_label"].to_numpy()

	if args.pca_file:
		if args.sentiment:
			logging.warning("--pca-file provided; ignoring separate --sentiment input (PCA NPZ already encodes sentiment if present).")
		try:
			X = load_pca_features(args.pca_file, use_pca_only=args.pca_use_pca)
		except Exception as e:
			raise SystemExit(f"Failed to load PCA features: {e}")
		if len(X) != len(y):
			raise SystemExit(f"PCA feature rows {len(X)} != labels {len(y)}")
		logging.info("Using PCA features shape=%s (pca_file=%s mode=%s)", X.shape, args.pca_file, "pca_only" if args.pca_use_pca else "combined")
	else:
		X_emb = load_embeddings(args.embeddings)
		if len(X_emb) != len(y):
			raise SystemExit(f"Embeddings rows {len(X_emb)} != labels {len(y)}")
		X_sent = None
		if args.sentiment:
			try:
				X_sent = load_sentiment(args.sentiment)
			except Exception as e:
				raise SystemExit(f"Failed to load sentiment features: {e}")
			logging.info("Loaded sentiment features shape=%s", X_sent.shape)
		X = combine_features(X_emb, X_sent)
		logging.info(
			"Final feature matrix shape=%s (emb_dim=%d + sentiment_dim=%d)",
			X.shape,
			X_emb.shape[1],
			0 if X_sent is None else X_sent.shape[1],
		)

	# Optional CV
	cv_results = None
	if args.cv_folds > 1:
		base_auc = cross_val_auc(X, y, C=1.0, folds=args.cv_folds, seed=args.random_state)
		logging.info("CV AUC (C=1.0) mean=%.4f std=%.4f folds=%s", np.mean(base_auc), np.std(base_auc), base_auc)
		cv_results = {"C": 1.0, "folds": args.cv_folds, "scores": base_auc, "mean": float(np.mean(base_auc)), "std": float(np.std(base_auc))}

	# Hyperparameter search
	best_C = 1.0
	search_details = None
	if args.search:
		c_values = [float(c.strip()) for c in args.c_grid.split(",") if c.strip()]
		best_mean = -1.0
		per_c = []
		for c in c_values:
			scores = cross_val_auc(X, y, C=c, folds=max(args.cv_folds, 3), seed=args.random_state)
			mean_score = float(np.mean(scores))
			per_c.append({"C": c, "scores": scores, "mean": mean_score})
			logging.info("Search C=%.3g mean=%.4f", c, mean_score)
			if mean_score > best_mean:
				best_mean = mean_score
				best_C = c
		logging.info("Best C=%.3g (mean AUC=%.4f)", best_C, best_mean)
		search_details = {"results": per_c, "best_C": best_C, "best_mean": best_mean}

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
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
	}

	if args.save_model:
		args.out_dir.mkdir(parents=True, exist_ok=True)
		ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
		model_path = args.out_dir / f"baseline_logreg_{ts}.pkl"
		metrics_path = args.out_dir / f"metrics_{ts}.json"
		with open(model_path, "wb") as f:
			pickle.dump(clf, f)
		with open(metrics_path, "w", encoding="utf-8") as f:
			json.dump(metrics, f, indent=2)
		logging.info("Saved model -> %s", model_path)
		logging.info("Saved metrics -> %s", metrics_path)


if __name__ == "__main__":
	main()

