import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import baseline_model as bm  # noqa: E402


def test_pca_npz_structure():
    # Require that a PCA file exists from prior step; skip if not.
    pca_path = 'pca_50_sentiment_std.npz'
    if not os.path.exists(pca_path):
        import pytest
        pytest.skip('PCA file not present')
    with np.load(pca_path) as npz:
        assert 'pca' in npz
        assert 'combined' in npz
        pca = npz['pca']
        combined = npz['combined']
        assert pca.shape[1] <= combined.shape[1]
        assert 'explained_variance_ratio' in npz
        assert 'cumulative_variance' in npz
