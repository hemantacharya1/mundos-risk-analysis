import os
import pickle
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, pipeline as hf_pipeline

# -----------------------------
# Config / ENV
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "best_pca_pipeline.pkl")
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SENT_MODEL_NAME = os.getenv("SENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
SKIP_SENTIMENT = os.getenv("SKIP_SENTIMENT_MODEL", "0") == "1"
PREFER_GPU = os.getenv("USE_GPU", "0") == "1"

device = torch.device("cuda" if (PREFER_GPU and torch.cuda.is_available()) else "cpu")

# -----------------------------
# Lazy loaders
# -----------------------------
@lru_cache(maxsize=1)
def load_pipeline():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    # Expect dict with keys
    for k in ["pca", "model", "n_components"]:
        if k not in obj:
            raise ValueError(f"Pipeline dict missing key '{k}'")
    return obj


@lru_cache(maxsize=1)
def load_embedding_model():
    tok = AutoTokenizer.from_pretrained(EMB_MODEL_NAME)
    mdl = AutoModel.from_pretrained(EMB_MODEL_NAME)
    mdl.to(device)
    mdl.eval()
    return tok, mdl


@lru_cache(maxsize=1)
def load_sentiment_analyzer():  # may be skipped
    if SKIP_SENTIMENT:
        return None
    dev = 0 if (device.type == "cuda") else -1
    return hf_pipeline("sentiment-analysis", model=SENT_MODEL_NAME, device=dev)


# -----------------------------
# Feature helpers
# -----------------------------
@torch.no_grad()
def embed(text: str) -> np.ndarray:
    tokenizer, mdl = load_embedding_model()
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = mdl(**enc)
    # mean pool (attention aware)
    mask = enc["attention_mask"].unsqueeze(-1)
    hidden = out.last_hidden_state * mask
    emb = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return emb.squeeze(0).cpu().numpy()


def sentiment_score(text: str) -> float:
    if SKIP_SENTIMENT:
        return 0.0
    analyzer = load_sentiment_analyzer()
    if analyzer is None:
        return 0.0
    out = analyzer(text)[0]
    label = out.get("label", "POSITIVE").upper()
    score = float(out.get("score", 0.5))
    return score if label.startswith("POS") else -score


def build_feature_vector(cust_text: str, agent_text: str, sentiment_used: bool) -> np.ndarray:
    cust_emb = embed(cust_text)
    agent_emb = embed(agent_text)
    combined = np.concatenate([cust_emb, agent_emb], axis=0)
    pipe = load_pipeline()
    scaler = pipe.get("scaler")
    pca = pipe["pca"]
    # scale -> pca
    Xrow = combined.reshape(1, -1)
    if scaler is not None:
        Xrow = scaler.transform(Xrow)
    X_pca = pca.transform(Xrow)
    if sentiment_used:
        # Determine how many sentiment columns the model was trained with
        feature_dim = pipe.get("feature_dim")  # stored at training time
        n_components = pipe.get("n_components")
        if feature_dim is None:
            # Fallback: assume 2 signed columns
            expected_sent_cols = 2
        else:
            expected_sent_cols = int(feature_dim) - int(n_components)
        if expected_sent_cols not in (2, 7):
            raise ValueError(f"Unexpected sentiment feature width {expected_sent_cols}; expected 2 or 7.")
        if SKIP_SENTIMENT:
            sent_arr = np.zeros((1, expected_sent_cols), dtype=np.float32)
        else:
            # Compute base scores
            # Analyzer outputs -> replicate training logic from sentiment_features.py
            def _analyze(text: str):
                out = load_sentiment_analyzer()
                res = out(text)[0]
                label = res.get("label", "POSITIVE").upper()
                score = float(res.get("score", 0.5))
                if label.startswith("NEG"):
                    signed = -score
                    p_pos = 1 - score
                    p_neg = score
                else:
                    signed = score
                    p_pos = score
                    p_neg = 1 - score
                return signed, p_pos, p_neg
            s_c_signed, s_c_p_pos, s_c_p_neg = sentiment_score(cust_text), 0.0, 0.0  # placeholders
            s_a_signed, s_a_p_pos, s_a_p_neg = sentiment_score(agent_text), 0.0, 0.0
            if expected_sent_cols == 7 and not SKIP_SENTIMENT:
                # Recompute using full analyzer outputs (since sentiment_score only returns signed)
                s_c_signed, s_c_p_pos, s_c_p_neg = _analyze(cust_text)
                s_a_signed, s_a_p_pos, s_a_p_neg = _analyze(agent_text)
                gap = s_a_signed - s_c_signed
                sent_arr = np.array([[
                    s_c_signed,
                    s_a_signed,
                    s_c_p_pos,
                    s_c_p_neg,
                    s_a_p_pos,
                    s_a_p_neg,
                    gap,
                ]], dtype=np.float32)
            elif expected_sent_cols == 2:
                sent_arr = np.array([[s_c_signed, s_a_signed]], dtype=np.float32)
            else:  # unexpected but guarded above
                sent_arr = np.zeros((1, expected_sent_cols), dtype=np.float32)
        X_final = np.concatenate([X_pca, sent_arr], axis=1)
    else:
        X_final = X_pca
    return X_final


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Lead Conversion Scoring API", version="1.1.0")


class LeadData(BaseModel):
    customer_summary: str
    agent_summary: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_file": os.path.basename(MODEL_PATH),
        "device": str(device),
        "embedding_model": EMB_MODEL_NAME,
        "sentiment_model": None if SKIP_SENTIMENT else SENT_MODEL_NAME,
    }


@app.post("/predict")
def predict(data: LeadData, threshold: float = Query(0.5, ge=0.0, le=1.0)):
    pipe = load_pipeline()
    sentiment_used = bool(pipe.get("sentiment_used", False))
    X = build_feature_vector(data.customer_summary, data.agent_summary, sentiment_used)
    proba = float(pipe["model"].predict_proba(X)[0, 1])
    pred = int(proba >= threshold)
    return {
        "conversion_probability": proba,
        "prediction": pred,
        "threshold": threshold,
        "model_file": os.path.basename(MODEL_PATH),
        "n_components": pipe.get("n_components"),
        "variance_sum": pipe.get("variance_sum"),
        "sentiment_used": sentiment_used,
        "calibrated": bool(pipe.get("calibrated", False)),
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
