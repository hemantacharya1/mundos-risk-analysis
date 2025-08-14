import os, sys, numpy as np
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_logreg_reduced as tlr  # noqa: E402


def test_load_reduced_prefers_combined(tmp_path):
    # Create NPZ with both pca and combined; ensure combined loaded unless pca_only
    pca = np.random.rand(5, 3).astype(np.float32)
    combined = np.random.rand(5, 5).astype(np.float32)
    path = tmp_path / 'test_pca.npz'
    np.savez_compressed(path, pca=pca, combined=combined)
    arr = tlr.load_reduced(path, pca_only=False)
    assert arr.shape == combined.shape
    arr_pca = tlr.load_reduced(path, pca_only=True)
    assert arr_pca.shape == pca.shape


def test_cross_val_auc_runs():
    X = np.random.rand(30, 4).astype(np.float32)
    y = np.array([0, 1] * 15)
    scores = tlr.cross_val_auc(X, y, C=1.0, folds=3, seed=42)
    assert len(scores) == 3
    assert all(0 <= s <= 1 for s in scores)
