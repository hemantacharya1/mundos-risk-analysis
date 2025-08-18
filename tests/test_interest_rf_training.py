import sys, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression


def test_logreg_training_small():
    # Create tiny synthetic embeddings (5 samples, 768 dims) + sentiment (7 dims)
    n_samples = 12
    emb = np.random.randn(n_samples, 768).astype('float32')
    sentiment = np.random.randn(n_samples, 7).astype('float32')
    X = np.hstack([emb, sentiment])
    y = np.array([0,1,2,1,0,2,1,0,2,1,0,2])

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (n_samples, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
