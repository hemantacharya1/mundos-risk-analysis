import numpy as np
from sentiment_features import compute_sentiment

def test_compute_sentiment_basic():
    def analyzer(texts):
        out = []
        for i, _t in enumerate(texts):
            if i % 2 == 0:
                out.append({"label": "POSITIVE", "score": 0.8})
            else:
                out.append({"label": "NEGATIVE", "score": 0.7})
        return out

    texts = ["good", "bad", "great", "awful", "neutral", " ", "excellent"]
    res = compute_sentiment(texts, analyzer, batch_size=3)
    assert res.signed.shape[0] == len(texts)
    assert (res.signed <= 1).all() and (res.signed >= -1).all()
    assert res.signed[0] > 0 and res.signed[1] < 0
    assert res.empty >= 1
    assert np.allclose(res.p_pos + res.p_neg, 1.0, atol=1e-5)
