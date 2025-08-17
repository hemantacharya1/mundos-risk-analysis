import os
import pickle
from functools import lru_cache
from typing import Optional, Dict, Any
import warnings

import numpy as np
import torch
from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, pipeline as hf_pipeline
try:  # joblib optional
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

# -----------------------------
# Config / ENV
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "best_interest_pipeline_1.pkl")
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SENT_MODEL_NAME = os.getenv("SENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
SKIP_SENTIMENT = os.getenv("SKIP_SENTIMENT_MODEL", "0") == "1"
PREFER_GPU = os.getenv("USE_GPU", "0") == "1"
CLASS_NAME_MAP_ENV = os.getenv("INTEREST_CLASS_NAMES", "0:no_interest,1:mild_interest,2:strong_interest")
# Optional override for sentiment width when feature_dim missing in pickle (allowed values: 2 or 7)
FORCE_SENTIMENT_WIDTH_ENV = os.getenv("FORCE_SENTIMENT_WIDTH", "")

device = torch.device("cuda" if (PREFER_GPU and torch.cuda.is_available()) else "cpu")

# -----------------------------
# Lazy loaders
# -----------------------------
@lru_cache(maxsize=1)
def load_pipeline():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
    load_error = None
    obj = None
    # First attempt: pickle
    try:
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        load_error = e
        # Try joblib if available
        if joblib is not None:
            try:
                obj = joblib.load(MODEL_PATH)
            except Exception as e2:  # still failing
                raise RuntimeError(
                    f"Failed to unpickle model file '{MODEL_PATH}'. Pickle error: {load_error}. Joblib error: {e2}. "
                    "Verify the file is a valid sklearn pipeline or dict."
                ) from e2
        else:
            raise RuntimeError(
                f"Failed to unpickle model file '{MODEL_PATH}'. Error: {load_error}. Install joblib to try alternative loader."
            ) from load_error
    # Normalize to dict
    if isinstance(obj, dict):
        pass
    elif hasattr(obj, "predict_proba"):
        obj = {"model": obj}
    else:
        raise ValueError(
            "Loaded object is neither dict nor estimator with predict_proba(); cannot use as pipeline."
        )
    # Warn if expected keys missing (not fatal for plain estimator)
    expected_optional = ["scaler", "pca", "n_components", "feature_dim", "sentiment_used"]
    missing = [k for k in expected_optional if k not in obj]
    if missing:
        warnings.warn(
            f"Pipeline missing optional keys: {missing}. Proceeding with available 'model' only.",
            RuntimeWarning,
        )
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
    combined = np.concatenate([cust_emb, agent_emb], axis=0).reshape(1, -1)
    pipe = load_pipeline()
    scaler = pipe.get("scaler")
    pca = pipe.get("pca")
    model = pipe.get("model")
    if scaler is not None:
        combined = scaler.transform(combined)
    if pca is not None:
        feats = pca.transform(combined)
    else:
        feats = combined
    # Dynamic inference: if sentiment_used flag not set but model expects wider feature vector
    if not sentiment_used:
        # Try to infer required width from model (sklearn estimator or Pipeline wrapped)
        expected = None
        try:
            if hasattr(model, "n_features_in_"):
                expected = int(getattr(model, "n_features_in_"))
        except Exception:  # pragma: no cover
            expected = None
        base_width = feats.shape[1]
        # If expected width indicates presence of sentiment columns (2 or 7) beyond embeddings/PCA components
        if expected and expected > base_width:
            diff = expected - base_width
            if diff in (2, 7):
                sentiment_used = True
                if FORCE_SENTIMENT_WIDTH_ENV.isdigit() and int(FORCE_SENTIMENT_WIDTH_ENV) in (2, 7):
                    diff = int(FORCE_SENTIMENT_WIDTH_ENV)
            # If mismatch but FORCE_SENTIMENT_WIDTH provided, honor it
        elif FORCE_SENTIMENT_WIDTH_ENV.isdigit() and int(FORCE_SENTIMENT_WIDTH_ENV) in (2, 7):
            diff = int(FORCE_SENTIMENT_WIDTH_ENV)
            sentiment_used = True
        else:
            diff = 0
        if not sentiment_used:
            return feats
        # Use diff as expected sentiment width (fallback 2)
        expected_sent_cols = diff if diff in (2, 7) else 2
    else:
        # sentiment_used already True via metadata
        expected_sent_cols = None  # will recompute below
    # Infer sentiment dimension
    if expected_sent_cols is None:
        feature_dim = pipe.get("feature_dim")
        n_components = pipe.get("n_components") if pca is not None else feats.shape[1]
        if feature_dim is None:
            if FORCE_SENTIMENT_WIDTH_ENV.isdigit() and int(FORCE_SENTIMENT_WIDTH_ENV) in (2, 7):
                expected_sent_cols = int(FORCE_SENTIMENT_WIDTH_ENV)
            else:
                # Derive from model.n_features_in_ if available
                expected_sent_cols = 2
                try:
                    if hasattr(model, "n_features_in_"):
                        total = int(getattr(model, "n_features_in_"))
                        base = feats.shape[1]
                        if total - base in (2, 7):
                            expected_sent_cols = total - base
                except Exception:  # pragma: no cover
                    pass
        else:
            expected_sent_cols = int(feature_dim) - int(n_components)
    if expected_sent_cols not in (2, 7):
        expected_sent_cols = 2  # safe fallback
    if SKIP_SENTIMENT:
        sent_arr = np.zeros((1, expected_sent_cols), dtype=np.float32)
    else:
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
        if expected_sent_cols == 2:
            sent_arr = np.array([[sentiment_score(cust_text), sentiment_score(agent_text)]], dtype=np.float32)
        else:
            c_signed, c_p_pos, c_p_neg = _analyze(cust_text)
            a_signed, a_p_pos, a_p_neg = _analyze(agent_text)
            gap = a_signed - c_signed
            sent_arr = np.array([[c_signed, a_signed, c_p_pos, c_p_neg, a_p_pos, a_p_neg, gap]], dtype=np.float32)
    return np.concatenate([feats, sent_arr], axis=1)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Interest Stage Scoring API", version="2.0.0")


class LeadData(BaseModel):
    customer_summary: str
    agent_summary: str


def _parse_class_name_mapping(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for part in raw.split(','):
        if ':' not in part:
            continue
        k, v = part.split(':', 1)
        mapping[k.strip()] = v.strip()
    return mapping


@app.get("/health")
def health():
    pipe = load_pipeline()
    model = pipe["model"]
    classes = getattr(model, "classes_", np.array([0, 1]))
    class_map = _parse_class_name_mapping(CLASS_NAME_MAP_ENV)
    readable = []
    for c in classes:
        key = str(int(c)) if isinstance(c, (int, np.integer)) else str(c)
        readable.append({"class": key, "label": class_map.get(key, key)})
    return {
        "status": "ok",
        "model_file": os.path.basename(MODEL_PATH),
        "device": str(device),
        "embedding_model": EMB_MODEL_NAME,
        "sentiment_model": None if SKIP_SENTIMENT else SENT_MODEL_NAME,
        "classes": readable,
    }


@app.post("/predict")
def predict(data: LeadData, top_k: int = Query(3, ge=1, le=10)):
    pipe = load_pipeline()
    model = pipe["model"]
    sentiment_used = bool(pipe.get("sentiment_used", False))
    X = build_feature_vector(data.customer_summary, data.agent_summary, sentiment_used)
    probs = model.predict_proba(X)[0]
    classes = getattr(model, "classes_", np.arange(len(probs)))
    class_map = _parse_class_name_mapping(CLASS_NAME_MAP_ENV)
    # Build probability dict
    prob_entries = []
    for c, p in zip(classes, probs):
        key = str(int(c)) if isinstance(c, (int, np.integer)) else str(c)
        prob_entries.append({"class": key, "label": class_map.get(key, key), "prob": float(p)})
    prob_entries_sorted = sorted(prob_entries, key=lambda d: d["prob"], reverse=True)
    best = prob_entries_sorted[0]
    return {
        "predicted_class": best["class"],
        "predicted_label": best["label"],
        "top": prob_entries_sorted[: top_k],
        "all_probabilities": prob_entries_sorted,
        "model_file": os.path.basename(MODEL_PATH),
        "n_components": pipe.get("n_components"),
        "variance_sum": pipe.get("variance_sum"),
        "sentiment_used": sentiment_used,
        "calibrated": bool(pipe.get("calibrated", False)),
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
