"""FastAPI application exposing inference for the calibrated PCA logistic regression model.

Endpoints:
  GET /health        -> basic liveness
  GET /metadata      -> model + build metadata
  POST /predict      -> batch predict on (customer_summary, agent_summary)

See README.md in this folder for details.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from functools import lru_cache
from typing import List, Any, Dict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel, pipeline as hf_pipeline

# ---- Configuration defaults ----
DEFAULT_MODEL_PATH = Path(os.environ.get("MODEL_PATH", "../best_interest_pipeline_1.pkl"))
EMB_MODEL_NAME = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SENT_MODEL_NAME = os.environ.get("SENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
PREFER_GPU = os.environ.get("USE_GPU", "0") == "1"
MAX_BATCH = int(os.environ.get("MAX_BATCH", "256"))

app = FastAPI(title="Leads Risk PCA Model API", version="0.1.0")
_START_TIME = time.time()


class PredictItem(BaseModel):
    lead_id: str = Field(..., description="Lead identifier")
    customer_summary: str
    agent_summary: str


class PredictRequest(BaseModel):
    items: List[PredictItem]


class PredictResult(BaseModel):
    lead_id: str
    proba: float
    prediction: int


class PredictResponse(BaseModel):
    model: Dict[str, Any]
    results: List[PredictResult]


def _device() -> torch.device:
    if PREFER_GPU and torch.cuda.is_available():  # pragma: no cover
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_pipeline() -> Dict[str, Any]:
    path = DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Pipeline pickle not found: {path}")
    import pickle
    with open(path, "rb") as f:
        obj = pickle.load(f)
    required = {"pca", "model", "n_components"}
    missing = required - set(obj.keys())
    if missing:
        raise ValueError(f"Pipeline pickle missing keys: {missing}")
    return obj


@lru_cache(maxsize=1)
def load_embedding_model():
    tok = AutoTokenizer.from_pretrained(EMB_MODEL_NAME)
    mdl = AutoModel.from_pretrained(EMB_MODEL_NAME)
    mdl.to(_device())
    mdl.eval()
    return tok, mdl


@lru_cache(maxsize=1)
def load_sentiment_analyzer():
    dev = 0 if (_device().type == "cuda") else -1
    return hf_pipeline("sentiment-analysis", model=SENT_MODEL_NAME, device=dev)


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    masked = token_embeddings * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def encode(texts: List[str], tokenizer, model, max_length: int = 256, batch_size: int = 32) -> np.ndarray:
    embs: List[np.ndarray] = []
    dev = _device()
    with torch.no_grad():  # pragma: no cover
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(dev)
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"]).cpu().numpy()
            embs.append(pooled)
    return np.vstack(embs)


def compute_signed_sentiment(texts: List[str]) -> np.ndarray:
    analyzer = load_sentiment_analyzer()
    vals: List[float] = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        outputs = analyzer(batch)
        for o in outputs:
            label = o.get("label", "POSITIVE")
            score = float(o.get("score", 0.5))
            vals.append(-score if label.upper().startswith("NEG") else score)
    return np.array(vals, dtype=np.float32)


def build_features(items: List[PredictItem]):
    pipe = load_pipeline()
    sentiment_used = pipe.get("sentiment_used", False)
    scaler = pipe.get("scaler")
    pca = pipe["pca"]
    model = pipe["model"]

    tokenizer, emb_model = load_embedding_model()
    cust_texts = [it.customer_summary for it in items]
    agent_texts = [it.agent_summary for it in items]
    cust_emb = encode(cust_texts, tokenizer, emb_model)
    agent_emb = encode(agent_texts, tokenizer, emb_model)
    combined = np.concatenate([cust_emb, agent_emb], axis=1)

    if scaler is not None:
        combined_proc = scaler.transform(combined)
    else:
        combined_proc = combined
    X_pca = pca.transform(combined_proc)

    if sentiment_used:
        cust_sent = compute_signed_sentiment(cust_texts)
        agent_sent = compute_signed_sentiment(agent_texts)
        sent_feats = np.stack([cust_sent, agent_sent], axis=1)
        X_final = np.concatenate([X_pca, sent_feats], axis=1)
    else:
        X_final = X_pca

    proba = model.predict_proba(X_final)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return proba, preds, pipe


@app.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - _START_TIME, 2)}


@app.get("/metadata")
def metadata():
    try:
        pipe = load_pipeline()
        return {
            "n_components": pipe.get("n_components"),
            "variance_sum": pipe.get("variance_sum"),
            "calibrated": bool(pipe.get("calibrated", False)),
            "sentiment_used": bool(pipe.get("sentiment_used", False)),
            "model_path": str(DEFAULT_MODEL_PATH.resolve()),
            "embedding_model": EMB_MODEL_NAME,
            "sentiment_model": SENT_MODEL_NAME,
        }
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    items = req.items
    if not items:
        raise HTTPException(status_code=400, detail="No items provided")
    if len(items) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Batch too large (>{MAX_BATCH})")
    try:
        proba, preds, pipe = build_features(items)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    results = [PredictResult(lead_id=it.lead_id, proba=float(p), prediction=int(y)) for it, p, y in zip(items, proba, preds)]
    model_meta = {
        "n_components": pipe.get("n_components"),
        "variance_sum": pipe.get("variance_sum"),
        "sentiment_used": bool(pipe.get("sentiment_used", False)),
        "calibrated": bool(pipe.get("calibrated", False)),
        "embedding_model": EMB_MODEL_NAME,
        "sentiment_model": SENT_MODEL_NAME if pipe.get("sentiment_used", False) else None,
    }
    return PredictResponse(model=model_meta, results=results)


@app.get("/")
def root():
    return {"service": app.title, "version": app.version, "health": "/health", "predict": "/predict"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("fastapi_backend.app:app", host="0.0.0.0", port=8000)
