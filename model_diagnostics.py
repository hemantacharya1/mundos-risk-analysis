"""Model diagnostics and leakage checks.

Steps performed:
1. Shuffle labels test (expect AUC ~0.5 if no leakage in process alone).
2. Duplicate summaries with conflicting labels detection.
3. Token leakage heuristic: log-odds of tokens for positive vs negative labels.
4. Train logistic regression on embeddings only vs embeddings+sentiment; compare AUC.
5. Extract top absolute coefficient features (sentiment dims separated).

Usage:
  uv run python model_diagnostics.py --embeddings embeddings.npz --sentiment sentiment.csv
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import baseline_model  # reuse load functions

SENTIMENT_ORDER = [
    "sentiment_cust",
    "sentiment_agent",
    "sentiment_cust_pos",
    "sentiment_cust_neg",
    "sentiment_agent_pos",
    "sentiment_agent_neg",
    "sentiment_gap",
]

def log_odds_tokens(text_series: pd.Series, labels: np.ndarray, top_n: int = 20):
    # Simple whitespace tokenization; could extend.
    from collections import Counter
    pos_counts = Counter()
    neg_counts = Counter()
    for text, y in zip(text_series.fillna(""), labels):
        tokens = [t.lower() for t in str(text).split() if t and len(t) > 2]
        uniq = set(tokens)  # use presence not frequency per doc
        if y == 1:
            pos_counts.update(uniq)
        else:
            neg_counts.update(uniq)
    vocab = set(pos_counts) | set(neg_counts)
    pos_total = sum(pos_counts.values()) + len(vocab)
    neg_total = sum(neg_counts.values()) + len(vocab)
    rows = []
    for tok in vocab:
        pc = pos_counts.get(tok, 0) + 1
        nc = neg_counts.get(tok, 0) + 1
        log_odds = np.log(pc / pos_total) - np.log(nc / neg_total)
        rows.append((tok, pc - 1, nc - 1, log_odds))
    df = pd.DataFrame(rows, columns=["token", "pos_docs", "neg_docs", "log_odds"])
    top_pos = df.sort_values("log_odds", ascending=False).head(top_n)
    top_neg = df.sort_values("log_odds").head(top_n)
    return top_pos, top_neg


def train_and_auc(X: np.ndarray, y: np.ndarray, seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    return auc, clf


def shuffled_label_auc(X: np.ndarray, y: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_shuff = rng.permutation(y)
    auc, _ = train_and_auc(X, y_shuff, seed=seed)
    return auc


def nested_cv_auc(X: np.ndarray, y: np.ndarray, outer_folds: int = 3, inner_folds: int = 3, seed: int = 42):
    # Simple nested CV for robust estimate; searching only C over a small grid.
    Cs = [0.1, 1.0, 5.0]
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    outer_scores = []
    for ofold, (tr_idx, te_idx) in enumerate(outer.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        best_c = None
        best_mean = -1
        for c in Cs:
            inner_scores = []
            for i_tr, i_te in inner.split(X_tr, y_tr):
                clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=c)
                clf.fit(X_tr[i_tr], y_tr[i_tr])
                prob = clf.predict_proba(X_tr[i_te])[:, 1]
                inner_scores.append(roc_auc_score(y_tr[i_te], prob))
            mean = float(np.mean(inner_scores))
            if mean > best_mean:
                best_mean = mean
                best_c = c
        final_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=best_c)
        final_clf.fit(X_tr, y_tr)
        prob_outer = final_clf.predict_proba(X_te)[:, 1]
        outer_auc = roc_auc_score(y_te, prob_outer)
        outer_scores.append(outer_auc)
    return outer_scores


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings.npz"))
    ap.add_argument("--sentiment", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=Path("leads.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if not args.embeddings.exists():
        raise SystemExit("Embeddings file missing")
    if not args.data.exists():
        raise SystemExit("Data file missing")

    y = pd.read_csv(args.data)["conversion_label"].to_numpy()
    X_emb = baseline_model.load_embeddings(args.embeddings)

    X_sent = None
    if args.sentiment:
        try:
            X_sent = baseline_model.load_sentiment(args.sentiment)
        except Exception as e:
            logging.warning("Could not load sentiment features: %s", e)

    X_full = baseline_model.combine_features(X_emb, X_sent)

    # 1. Shuffle label sanity
    shuffle_auc = shuffled_label_auc(X_full, y, seed=args.seed)
    logging.info("Shuffle label AUC (should be ~0.5): %.4f", shuffle_auc)

    # 2. Duplicates with conflicting labels
    df = pd.read_csv(args.data)
    df['combined_key'] = df['customer_summary'].astype(str) + '||' + df['agent_summary'].astype(str)
    conflict = (df.groupby('combined_key')['conversion_label']
                  .nunique().reset_index())
    conflict = conflict[conflict['conversion_label'] > 1]
    if not conflict.empty:
        logging.warning("Found %d summary text duplicates with conflicting labels", len(conflict))
    else:
        logging.info("No conflicting duplicate summaries detected")

    # 3. Token leakage heuristic
    combined_text = df['customer_summary'].astype(str) + ' ' + df['agent_summary'].astype(str)
    top_pos, top_neg = log_odds_tokens(combined_text, y, top_n=15)
    logging.info("Top tokens skewed to positive (conversion) label:\n%s", top_pos.to_string(index=False))
    logging.info("Top tokens skewed to negative (non-conversion) label:\n%s", top_neg.to_string(index=False))

    # 4. Train embeddings-only
    emb_auc, emb_clf = train_and_auc(X_emb, y, seed=args.seed)
    logging.info("Embeddings-only AUC: %.4f", emb_auc)

    # 5. Train embeddings + sentiment (if provided)
    if X_sent is not None:
        full_auc, full_clf = train_and_auc(X_full, y, seed=args.seed)
        logging.info("Embeddings+sentiment AUC: %.4f (delta=%.4f)", full_auc, full_auc - emb_auc)
    else:
        full_auc = None
        full_clf = emb_clf

    # 6. Coefficients examination (only last trained clf used if sentiment loaded)
    clf = full_clf if X_sent is not None else emb_clf
    coefs = clf.coef_.ravel()
    if X_sent is not None:
        sent_dim = X_sent.shape[1]
        emb_dim = X_emb.shape[1]
        sent_coefs = coefs[-sent_dim:]
        coef_df = pd.DataFrame({"feature": SENTIMENT_ORDER[:sent_dim], "coef": sent_coefs, "abs_coef": np.abs(sent_coefs)}).sort_values("abs_coef", ascending=False)
        logging.info("Top sentiment feature coefficients:\n%s", coef_df.to_string(index=False))
    # Top embedding dims
    top_emb_idx = np.argsort(np.abs(coefs[: X_emb.shape[1]]))[-10:][::-1]
    top_emb_vals = coefs[top_emb_idx]
    logging.info("Top 10 embedding dims by |coef| (index:value): %s", 
                 ", ".join(f"{i}:{v:.4f}" for i, v in zip(top_emb_idx, top_emb_vals)))

    # Nested CV robust check (cheap grid)
    outer_scores = nested_cv_auc(X_full, y, outer_folds=3, inner_folds=3, seed=args.seed)
    logging.info("Nested CV outer AUC scores: %s (mean=%.4f, std=%.4f)", outer_scores, float(np.mean(outer_scores)), float(np.std(outer_scores)))

if __name__ == "__main__":  # pragma: no cover
    main()
