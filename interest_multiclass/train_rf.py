"""Train RandomForest multiclass interest model.
Usage:
  uv run python interest_multiclass/train_rf.py --search
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, log_loss, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import (
    DATA_FILE, CLASS_LABEL_COLUMN, CLASS_IDS, CLASS_NAMES,
    EMBEDDINGS_FILE, SENTIMENT_FILE, COMBINED_FEATURES_FILE,
    TEST_SIZE, RANDOM_STATE, ARTIFACTS_DIR, DEFAULT_RF_PARAMS
)

OUT_MODEL = ARTIFACTS_DIR / "rf_pipeline.pkl"
OUT_METRICS = ARTIFACTS_DIR / "metrics_rf.json"


def load_embeddings() -> np.ndarray:
    if not EMBEDDINGS_FILE.exists():
        raise SystemExit("Embeddings file missing. Generate embeddings first.")
    with np.load(EMBEDDINGS_FILE) as npz:
        # Accept either 'combined' or 'embeddings' key from upstream scripts
        if 'combined' in npz:
            return npz['combined'].astype(np.float32)
        if 'embeddings' in npz:
            return npz['embeddings'].astype(np.float32)
        raise SystemExit("No suitable array in embeddings NPZ (expected 'combined' or 'embeddings').")


def load_sentiment() -> np.ndarray | None:
    if not SENTIMENT_FILE.exists():
        return None
    df = pd.read_csv(SENTIMENT_FILE)
    cols = [c for c in df.columns if c.startswith('sentiment_')]
    if not cols:
        return None
    return df[cols].to_numpy(dtype=np.float32)


def assemble_features() -> np.ndarray:
    X_emb = load_embeddings()
    X_sent = load_sentiment()
    if X_sent is not None:
        if len(X_sent) != len(X_emb):
            raise SystemExit("Sentiment row count mismatch embeddings.")
        X = np.hstack([X_emb, X_sent])
    else:
        X = X_emb
    np.savez_compressed(COMBINED_FEATURES_FILE, features=X)
    return X


def search_param_grid(X, y):
    # Simple manual grid (avoid external dependencies). Could expand.
    grids = [
        {"n_estimators": n, "max_depth": d, "min_samples_split": s}
        for n in (200, 400)
        for d in (None, 30)
        for s in (2, 4)
    ]
    best = None
    best_score = -1
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    for g in grids:
        clf = RandomForestClassifier(
            n_estimators=g["n_estimators"],
            max_depth=g["max_depth"],
            min_samples_split=g["min_samples_split"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)
        try:
            auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0
        if auc > best_score:
            best_score = auc
            best = g
    return best or {}, best_score


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="Run simple param grid search")
    ap.add_argument("--standardize", action="store_true", help="Apply StandardScaler before RF (optional)")
    args = ap.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit(f"Data file missing: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    if CLASS_LABEL_COLUMN not in df.columns:
        raise SystemExit(f"Label column '{CLASS_LABEL_COLUMN}' missing")
    y = df[CLASS_LABEL_COLUMN].to_numpy()

    X = assemble_features()
    if len(X) != len(y):
        raise SystemExit("Feature / label length mismatch")

    scaler = None
    if args.standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    params = DEFAULT_RF_PARAMS.__dict__.copy()
    search_info = None
    if args.search:
        best_grid, best_auc = search_param_grid(X, y)
        params.update(best_grid)
        search_info = {"best_grid": best_grid, "auc_macro_ovr": best_auc}

    clf = RandomForestClassifier(**params)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    preds = clf.predict(X_test)

    metrics: dict = {}
    try:
        metrics["macro_roc_auc_ovr"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
    except Exception:
        metrics["macro_roc_auc_ovr"] = None
    metrics["macro_f1"] = float(f1_score(y_test, preds, average="macro"))
    metrics["weighted_f1"] = float(f1_score(y_test, preds, average="weighted"))
    try:
        metrics["log_loss"] = float(log_loss(y_test, proba))
    except Exception:
        metrics["log_loss"] = None
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average=None, labels=sorted(set(y_test)))
    per_class = {}
    sorted_labels = sorted(set(y_test))
    for i, lbl in enumerate(sorted_labels):
        name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else str(lbl)
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = confusion_matrix(y_test, preds).tolist()
    metrics["search"] = search_info
    metrics["params"] = params

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    pipeline_obj = {
        "model": clf,
        "scaler": scaler,
        "feature_dim": X.shape[1],
        "sentiment_used": bool(load_sentiment() is not None),
        "algorithm": "RandomForestClassifier",
        "classes": CLASS_IDS,
    }
    import pickle
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(pipeline_obj, f)
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model -> {OUT_MODEL}")
    print(f"Saved metrics -> {OUT_METRICS}")

if __name__ == "__main__":  # pragma: no cover
    main()
