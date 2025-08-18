"""Train multinomial LogisticRegression interest model.
Usage:
  uv run python interest_multiclass/train_logreg.py --standardize --search
Options:
  --standardize : apply StandardScaler
  --search      : small hyperparameter grid search (C, penalty, max_iter)
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, log_loss, precision_recall_fscore_support, confusion_matrix
from config import (
    DATA_FILE, CLASS_LABEL_COLUMN, CLASS_IDS, CLASS_NAMES,
    EMBEDDINGS_FILE, SENTIMENT_FILE, COMBINED_FEATURES_FILE,
    TEST_SIZE, RANDOM_STATE, ARTIFACTS_DIR,
    PCA_MODEL_FILE, PCA_VARIANCE_PLOT, PCA_COMPONENTS_DEFAULT, PCA_VARIANCE_THRESHOLD
)

OUT_MODEL = ARTIFACTS_DIR / "logreg_pipeline.pkl"
OUT_METRICS = ARTIFACTS_DIR / "metrics_logreg.json"


def load_embeddings() -> np.ndarray:
    if not EMBEDDINGS_FILE.exists():
        raise SystemExit("Embeddings file missing. Generate embeddings first.")
    with np.load(EMBEDDINGS_FILE) as npz:
        if 'combined' in npz:
            return npz['combined'].astype(np.float32)
        if 'embeddings' in npz:
            return npz['embeddings'].astype(np.float32)
        raise SystemExit("No suitable array in embeddings NPZ (expected 'combined' or 'embeddings').")


def load_sentiment() -> np.ndarray | None:
    if not SENTIMENT_FILE.exists():
        return None
    import pandas as pd
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


def search_grid(X, y, scaler):
    # multi_class deprecated (always multinomial for these solvers when n_classes>2); omit to silence FutureWarning
    grid = [
        {"C": c, "penalty": pen, "solver": solv, "max_iter": 300}
        for c in (0.1, 1.0, 3.0)
        for pen, solv in [("l2", "lbfgs"), ("l2", "saga")]
    ]
    best = None
    best_score = -1
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    for params in grid:
        model = LogisticRegression(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)
        try:
            auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0
        if auc > best_score:
            best_score = auc
            best = params
    return best, best_score


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--pca", action="store_true", help="Enable PCA dimensionality reduction")
    ap.add_argument("--pca-components", type=int, default=None, help="Fixed number of PCA components (overrides variance threshold)")
    ap.add_argument("--pca-variance", type=float, default=None, help="Variance threshold (e.g. 0.95) if components not given")
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
    search_info = None
    pca_info = None
    params = {"C":1.0, "penalty":"l2", "solver":"lbfgs", "max_iter":300}
    if args.search:
        best, best_auc = search_grid(X, y, scaler)
        if best:
            params = best
            search_info = {"best": best, "auc_macro_ovr": best_auc}

    # Split BEFORE fitting scaler / PCA to avoid leakage
    pca = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # Standardize (fit only on training set)
    if args.standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Optional PCA (fit only on training set output)
    if args.pca:
        desired_components = args.pca_components if args.pca_components is not None else PCA_COMPONENTS_DEFAULT
        if desired_components is not None:
            pca = PCA(n_components=desired_components, random_state=RANDOM_STATE)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            pca_info = {"chosen_components": int(pca.n_components_), "mode": "fixed"}
        else:
            var_threshold = args.pca_variance if args.pca_variance is not None else PCA_VARIANCE_THRESHOLD
            pca_temp = PCA(n_components=min(X_train.shape[0], X_train.shape[1]), svd_solver="full", random_state=RANDOM_STATE)
            pca_temp.fit(X_train)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            k = int(np.searchsorted(cumsum, var_threshold) + 1)
            pca = PCA(n_components=k, random_state=RANDOM_STATE)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            pca_info = {"variance_threshold": var_threshold, "chosen_components": k}
            # Optionally save variance plot
            try:  # pragma: no cover
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6,4))
                plt.plot(cumsum, marker='o', linewidth=1)
                plt.axhline(var_threshold, color='red', linestyle='--', linewidth=1)
                plt.xlabel('Components')
                plt.ylabel('Cumulative Explained Variance')
                plt.title('PCA Variance Accumulation')
                PCA_VARIANCE_PLOT.parent.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(PCA_VARIANCE_PLOT)
                plt.close()
            except Exception:
                pass

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)

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
    for i, lbl in enumerate(sorted(set(y_test))):
        name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else str(lbl)
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = confusion_matrix(y_test, preds).tolist()
    metrics["search"] = search_info
    metrics["pca"] = pca_info
    metrics["params"] = params

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    pipeline_obj = {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "original_feature_dim": X.shape[1],
        "feature_dim": int(X_train.shape[1]) if pca is not None else X.shape[1],
        "sentiment_used": bool(load_sentiment() is not None),
        "algorithm": "LogisticRegression",
        "classes": CLASS_IDS,
    }
    if pca is not None:
        with open(PCA_MODEL_FILE, "wb") as f:
            pickle.dump(pca, f)
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(pipeline_obj, f)
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model -> {OUT_MODEL}")
    print(f"Saved metrics -> {OUT_METRICS}")

if __name__ == "__main__":  # pragma: no cover
    main()
