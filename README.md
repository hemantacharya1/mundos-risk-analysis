## Mundos Risk Analysis

Production-ready pipeline to transform raw conversational lead summaries into calibrated probabilities of interest stage (multi-class) using transformer embeddings, sentiment features, dimensionality reduction (PCA), and a FastAPI inference service.

---
## 1. High-Level Flow
1. Load & validate dataset (`load_leads.py`).
2. Generate embeddings (`generate_embeddings.py`).
3. Generate sentiment features (`sentiment_features.py`).
4. Train calibrated PCA logistic pipelines over component sweep (`train_pca_pipeline.py`).
5. Select & persist best pipeline pickle (`best_pca_pipeline*.pkl`).
6. Serve model via FastAPI (`fastapi_backend/main.py`).
7. Batch / ad‑hoc inference utilities (`inference.py`, `inference_pca_pipeline.py`).

---
## 2. Current Folder Structure (key items only)
```
mundos_risk_analysis/
├─ artifacts/                 # Timestamped baseline models & metrics JSON
├─ fastapi_backend/           # Production inference backend (FastAPI)
├─ flowcharts/                # (Diagrams / mermaid sources – optional)
├─ interest_multiclass/       # Extended multiclass training experiments
├─ tests/                     # Unit tests (data validation, pooling, PCA, etc.)
├─ baseline_model.py          # Train baseline logistic model (no PCA sweep)
├─ generate_embeddings.py     # Create MiniLM embeddings (customer + agent)
├─ sentiment_features.py      # Build 2- or 7-d sentiment feature set
├─ train_pca_pipeline.py      # Component sweep + calibration + best pipeline save
├─ train_logreg_reduced.py    # Train logistic on pre-reduced feature sets
├─ pca_reduce.py              # Manual PCA reduction & artifact generation
├─ inference.py               # Generic inference using saved baseline model
├─ inference_pca_pipeline.py  # Inference using unified PCA pipeline pickle
├─ threshold_analysis.py      # Precision/recall/F1 grid & threshold export
├─ masking_stress_test.py     # Robustness via token masking
├─ model_diagnostics.py       # Leakage / coefficient / shuffle diagnostics
├─ load_leads.py              # Load + integrity checks & hashing
├─ show_classification_report.py # Convenience reporting utility
├─ show_pca_pipeline_report.py   # PCA pipeline evaluation summary
├─ PROJECT_OVERVIEW.md        # Deep-dive design & metrics document
├─ README.md                  # (This file)
├─ pyproject.toml             # Dependencies & build config (uv / PEP 621)
└─ uv.lock                    # Locked dependency versions
```

---
## 3. Key Files & Purpose
| File / Dir | Description |
|------------|-------------|
| `leads.csv` | Source labeled leads dataset. |
| `generate_embeddings.py` | Produces concatenated MiniLM embeddings (768 dims). |
| `sentiment_features.py` | Sentiment (signed + probs or 2-col variant). |
| `train_pca_pipeline.py` | Fits scaler+PCA+logreg over component grid, calibrates, saves best. |
| `best_pca_pipeline.pkl` | Unified pipeline (scaler, PCA, model, metadata). |
| `baseline_model.py` | Simpler non-sweep baseline logistic training script. |
| `pca_reduce.py` | One-off PCA artifact generation / plotting. |
| `inference_pca_pipeline.py` | Batch inference using best PCA pipeline. |
| `inference.py` | Inference for baseline models; auto-detects sentiment need. |
| `threshold_analysis.py` | Computes metrics over probability thresholds. |
| `model_diagnostics.py` | Leakage tests, token log-odds, nested CV. |
| `masking_stress_test.py` | Removes top tokens to gauge reliance. |
| `artifacts/` | Baseline model pickles + metrics timestamped. |
| `fastapi_backend/` | REST API service (embedding + sentiment at runtime). |
| `tests/` | Unit tests for data integrity & feature logic. |
| `interest_multiclass/` | Experimental multiclass model variants. |
| `PROJECT_OVERVIEW.md` | Extended documentation & analysis. |

---
## 4. Quick Start (PowerShell)
```powershell
# Install dependencies
uv sync

#############################################
# OPTION A (Preferred): interest_multiclass #
#############################################
# 1. Validate data (creates data_integrity_report.json)
uv run python interest_multiclass/data_validation.py

# 2. Generate embeddings (customer + agent concatenated arrays)
uv run python interest_multiclass/generate_embeddings_interest.py --data leads_1.csv

# 3. (Optional but recommended) Sentiment features (7-col rich block; add --signed-only for 3-col light block)
uv run python interest_multiclass/generate_sentiment_interest.py

# 4. Train multinomial LogisticRegression with StandardScaler + PCA (95% variance) + small param search
uv run python interest_multiclass/train_logreg.py --standardize --search --pca --pca-variance 0.95
#   Artifacts: interest_multiclass/artifacts/logreg_pipeline.pkl (+ metrics_logreg.json, pca_transform.pkl, etc.)

# 5. Batch inference using trained multiclass pipeline
uv run python interest_multiclass/inference.py --model interest_multiclass/artifacts/logreg_pipeline.pkl --input leads_1.csv --output predictions_interest.csv

#############################################
# OPTION B (Legacy baseline sweep + calibration)
#############################################
# Generate baseline embeddings & sentiment (shared scripts)
uv run python generate_embeddings.py --output embeddings.npz
uv run python sentiment_features.py --output sentiment.csv

# Train PCA component sweep with calibration; saves best_pca_pipeline.pkl
uv run python train_pca_pipeline.py --embeddings embeddings.npz --sentiment sentiment.csv --components 10,40,50,100 --standardize --calibrate sigmoid --save-best best_pca_pipeline.pkl --results pca_sweep_results.json

# Batch inference for legacy pipeline
uv run python inference_pca_pipeline.py --pipeline best_pca_pipeline.pkl --embeddings embeddings.npz --sentiment sentiment.csv --output predictions_interest.csv

# (Either option) Run test suite
uv run python -m pytest -q
```

---
## 5. Serving the Model
Run the FastAPI backend (supports GPU if `USE_GPU=1` and CUDA available):
```powershell
uv run uvicorn fastapi_backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Health check:
```powershell
curl http://localhost:8000/health
```
Predict (example payload):
```powershell
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"customer_summary":"Customer asked about pricing tiers","agent_summary":"Agent explained premium plan and scheduled a follow-up"}'
```

Environment overrides (optional):
| Variable | Default | Effect |
|----------|---------|--------|
| `MODEL_PATH` | `fastapi_backend/best_interest_pipeline_1.pkl` | Path to unified pipeline. |
| `EMB_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model name. |
| `SENT_MODEL` | `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment model (HF). |
| `SKIP_SENTIMENT_MODEL` | `0` | Set `1` to disable sentiment at runtime. |
| `USE_GPU` | `0` | Set `1` to prefer CUDA. |
| `INTEREST_CLASS_NAMES` | `0:no_interest,1:mild_interest,2:strong_interest` | Class label mapping. |
| `FORCE_SENTIMENT_WIDTH` | (blank) | Force sentiment width (2 or 7) if metadata absent. |

---
## 6. Metadata & Features
Unified pipeline pickle includes (when available):
`{ model, scaler, pca, n_components, sentiment_used, feature_dim, original_feature_dim, calibrated, algorithm }`.

Runtime feature build order in API: embeddings (customer + agent) → optional sentiment (2 or 7 cols) → scaler → PCA → logistic regression.

---
## 7. Common Issues
| Symptom | Cause | Resolution |
|---------|-------|------------|
| Feature dimension mismatch | Sentiment usage differs between train/infer | Use same sentiment setting or set `FORCE_SENTIMENT_WIDTH`. |
| First request slow | HuggingFace model download | Warm by calling `/health` or `/predict` once. |
| GPU not used | CUDA not present / `USE_GPU` unset | Export `USE_GPU=1` and ensure torch with CUDA installed. |

---
## 8. Extending
Ideas: add drift monitoring, isotonic calibration comparison, SHAP explanations on PCA space, batch /predict endpoint, embedding caching layer.

---
## 9. License
Internal / unspecified. Add a LICENSE file before distribution.

---
For deeper methodology & metrics see `PROJECT_OVERVIEW.md`.
