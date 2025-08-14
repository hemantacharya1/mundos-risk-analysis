import os
import sys
import numpy as np

# Ensure project root on path for direct script imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baseline_model import combine_features  # noqa: E402

def test_combine_features_shapes():
    emb = np.random.rand(10, 5).astype(np.float32)
    sent = np.random.rand(10, 3).astype(np.float32)
    combined = combine_features(emb, sent)
    assert combined.shape == (10, 8)


def test_combine_features_row_mismatch():
    emb = np.random.rand(10, 5).astype(np.float32)
    sent = np.random.rand(9, 3).astype(np.float32)
    try:
        _ = combine_features(emb, sent)
    except ValueError as e:
        assert "Row mismatch" in str(e)
    else:
        assert False, "Expected ValueError"
