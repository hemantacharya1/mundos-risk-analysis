import os
import pickle
from functools import lru_cache
from typing import Optional, Dict, Any
import warnings
from pathlib import Path

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
# Default model filename (must always be used per requirement). This file should contain the latest PCA logistic pipeline.
MODEL_PATH_RAW = os.getenv("MODEL_PATH", "fastapi_backend/best_interest_pipeline_1.pkl")
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
def _resolve_model_path() -> str:
    """Resolve model path robustly whether server started from repo root or fastapi_backend folder.

    Resolution order:
      1. If env provided absolute path -> use directly.
      2. If relative path exists as given (cwd based) -> use.
      3. Try relative to project root (parent of this file's directory) -> use if exists.
      4. Try walking up two parents.
      5. Final: raise FileNotFoundError with attempted candidates listed.
    """
    raw = MODEL_PATH_RAW
    attempted = []
    # Absolute path
    if os.path.isabs(raw):
        if os.path.isfile(raw):
            return raw
        attempted.append(raw)
    else:
        # As given (current working directory)
        if os.path.isfile(raw):
            return raw
        attempted.append(os.path.abspath(raw))
        # Relative to repo root (parent directory of this file)
        this_dir = Path(__file__).resolve().parent
        repo_root = this_dir.parent  # fastapi_backend/ -> project root
        candidate = repo_root / raw
        if candidate.is_file():
            return str(candidate)
        attempted.append(str(candidate))
        # One more level up just in case
        candidate2 = repo_root.parent / raw
        if candidate2.is_file():
            return str(candidate2)
        attempted.append(str(candidate2))
    raise FileNotFoundError(
        f"Model file not found. Provided path='{raw}'. Tried: " + "; ".join(attempted) + ". Set MODEL_PATH env to a valid absolute path."
    )


@lru_cache(maxsize=1)
def load_pipeline():
    model_path = _resolve_model_path()
    load_error = None
    obj = None
    # First attempt: pickle (also guard against obvious non-pickle files)
    try:
        with open(model_path, "rb") as f:
            head = f.read(2)
            f.seek(0)
            if head and head[0] != 0x80:
                warnings.warn(
                    f"File '{model_path}' does not start with pickle magic (0x80); attempting load anyway.",
                    RuntimeWarning,
                )
            obj = pickle.load(f)
    except Exception as e:
        load_error = e
        fallback_candidates = [
            # Prefer latest trained PCA logistic pipeline if available
            "interest_multiclass/artifacts/logreg_pipeline.pkl",
            "../interest_multiclass/artifacts/logreg_pipeline.pkl",
        ]
        loaded_fallback_path = None
        for cand in fallback_candidates:
            cand_path = Path(model_path).parent / cand if not os.path.isabs(cand) else Path(cand)
            if cand_path.is_file():
                try:
                    with open(cand_path, "rb") as f2:
                        obj = pickle.load(f2)
                    loaded_fallback_path = str(cand_path.resolve())
                    warnings.warn(
                        f"Primary model file '{model_path}' could not be loaded ({e!r}). Fallback loaded from '{loaded_fallback_path}'. "
                        "Consider updating or replacing the primary file with this artifact.",
                        RuntimeWarning,
                    )
                    break
                except Exception:
                    continue
        if obj is None:
            # Try joblib last resort on primary file
            if joblib is not None:
                try:
                    obj = joblib.load(model_path)
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load primary model '{model_path}'. Pickle error: {load_error}. Joblib error: {e2}."
                    ) from e2
            else:
                raise RuntimeError(
                    f"Failed to load primary model '{model_path}'. Error: {load_error}. Install joblib for alternative loader."
                ) from load_error
        # Record fallback origin if used
        if obj is not None and loaded_fallback_path:
            obj["_fallback_source"] = loaded_fallback_path
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
    # Augment metadata for convenience
    pca = obj.get("pca")
    if pca is not None and "n_components" not in obj:
        try:
            obj["n_components"] = int(getattr(pca, "n_components_", getattr(pca, "n_components", None)))
        except Exception:  # pragma: no cover
            pass
    if "feature_dim" not in obj:
        # derive from model if possible
        try:
            if hasattr(obj.get("model"), "n_features_in_"):
                obj["feature_dim"] = int(getattr(obj.get("model"), "n_features_in_"))
        except Exception:  # pragma: no cover
            pass
    # Attach resolved path for downstream endpoints
    obj["_resolved_model_path"] = model_path
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
    """Construct feature vector replicating training order:
        (embeddings [+ sentiment]) -> scaler -> PCA.
    Previously sentiment was appended AFTER PCA causing a mismatch for scaler expecting 775 features.
    """
    pipe = load_pipeline()
    scaler = pipe.get("scaler")
    pca = pipe.get("pca")
    original_feature_dim = pipe.get("original_feature_dim")  # e.g. 775

    # 1. Embeddings (customer + agent concatenated)
    cust_emb = embed(cust_text)
    agent_emb = embed(agent_text)
    emb = np.concatenate([cust_emb, agent_emb], axis=0)  # (768,)

    # 2. Sentiment (if used). Determine width: if we have original_feature_dim then
    #    sentiment_width = original_feature_dim - embedding_width (768)
    sentiment_width = 0
    if sentiment_used and not SKIP_SENTIMENT:
        if original_feature_dim is not None:
            sentiment_width = int(original_feature_dim) - emb.shape[0]
        elif FORCE_SENTIMENT_WIDTH_ENV.isdigit() and int(FORCE_SENTIMENT_WIDTH_ENV) in (2, 7):
            sentiment_width = int(FORCE_SENTIMENT_WIDTH_ENV)
        else:  # fallback default
            sentiment_width = 7
        # Build sentiment features
        if sentiment_width == 2:
            sent_arr = np.array([
                sentiment_score(cust_text),
                sentiment_score(agent_text),
            ], dtype=np.float32)
        else:  # rich 7-dim engineer
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
            c_signed, c_p_pos, c_p_neg = _analyze(cust_text)
            a_signed, a_p_pos, a_p_neg = _analyze(agent_text)
            gap = a_signed - c_signed
            sent_arr = np.array([c_signed, a_signed, c_p_pos, c_p_neg, a_p_pos, a_p_neg, gap], dtype=np.float32)
        features = np.concatenate([emb, sent_arr], axis=0)
    else:
        features = emb

    features = features.reshape(1, -1)

    # 3. Scale (if present)
    if scaler is not None:
        features = scaler.transform(features)

    # 4. PCA (if present)
    if pca is not None:
        features = pca.transform(features)

    return features


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
    "model_file": os.path.basename(pipe.get("_resolved_model_path", MODEL_PATH_RAW)),
    "fallback_source": pipe.get("_fallback_source"),
        "device": str(device),
        "embedding_model": EMB_MODEL_NAME,
        "sentiment_model": None if SKIP_SENTIMENT else SENT_MODEL_NAME,
        "classes": readable,
    "pca_used": bool(pipe.get("pca") is not None),
    "n_components": pipe.get("n_components"),
    "feature_dim": pipe.get("feature_dim"),
    "original_feature_dim": pipe.get("original_feature_dim"),
    "sentiment_used": pipe.get("sentiment_used"),
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
    # Infer / fill missing metadata on the fly
    algorithm = pipe.get("algorithm") or type(model).__name__
    feature_dim = pipe.get("feature_dim")
    if feature_dim is None and hasattr(model, "n_features_in_"):
        feature_dim = int(getattr(model, "n_features_in_"))
    pca_obj = pipe.get("pca")
    n_components = pipe.get("n_components")
    if pca_obj is not None and n_components is None:
        try:
            n_components = int(getattr(pca_obj, "n_components_", getattr(pca_obj, "n_components", None)))
        except Exception:  # pragma: no cover
            pass
    original_feature_dim = pipe.get("original_feature_dim")
    if original_feature_dim is None and pca_obj is not None and feature_dim and n_components:
        # Attempt reconstruction: original pre-sentiment + sentiment width
        sent_width = 0
        if sentiment_used:
            # sentiment width is difference between total feature dim and PCA components
            try:
                sent_width = feature_dim - n_components
            except Exception:  # pragma: no cover
                sent_width = 0
        try:
            # Number of embedding (pre-PCA) features is pca_obj.n_features_ if available
            pre_pca = getattr(pca_obj, "n_features_", None)
            if pre_pca is not None:
                original_feature_dim = int(pre_pca) + sent_width
        except Exception:  # pragma: no cover
            pass
    response = {
        "predicted_class": best["class"],
        "predicted_label": best["label"],
        "predicted_prob": best["prob"],
        "probabilities": prob_entries_sorted[: top_k],  # sorted desc, limited to top_k
    "model_file": os.path.basename(pipe.get("_resolved_model_path", MODEL_PATH_RAW)),
    "fallback_source": pipe.get("_fallback_source"),
        "pca_used": bool(pca_obj is not None),
        "n_components": n_components,
        "feature_dim": feature_dim,
        "original_feature_dim": original_feature_dim,
        "sentiment_used": sentiment_used,
        "calibrated": bool(pipe.get("calibrated", False)),
        "algorithm": algorithm,
    }
    return response


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
