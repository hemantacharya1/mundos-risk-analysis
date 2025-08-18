## Interest Stage FastAPI Backend

Multiclass interest stage inference service for PCA + LogisticRegression pipeline (0 / 1 / 2 classes).

### 🔧 Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Liveness + model metadata |
| POST | /predict | Score a single (customer_summary, agent_summary) pair (multiclass probs) |

### 🧾 Request / Response
POST /predict JSON body:
```json
{
  "customer_summary": "Asked about whitening, wants Saturday appointment.",
  "agent_summary": "Offered Saturday 10am slot and sent booking link."
}
```
Sample response (truncated):
```json
{
  "predicted_class": "2",
  "predicted_label": "strong_interest",
  "top": [
    {"class": "2", "label": "strong_interest", "prob": 0.87},
    {"class": "1", "label": "mild_interest", "prob": 0.11},
    {"class": "0", "label": "no_interest", "prob": 0.02}
  ],
  "pca_used": true,
  "n_components": 58,
  "feature_dim": 58,
  "original_feature_dim": 1543,
  "sentiment_used": true,
  "algorithm": "LogisticRegression"
}
```

### ⚙️ Environment Variables
| Name | Default | Description |
|------|---------|-------------|
| MODEL_PATH | best_interest_pipeline_1.pkl | Path to saved pipeline dict |
| EMB_MODEL | sentence-transformers/all-MiniLM-L6-v2 | HuggingFace embedding model |
| SENT_MODEL | distilbert-base-uncased-finetuned-sst-2-english | Sentiment model (SST-2) |
| SKIP_SENTIMENT_MODEL | 0 | If 1, skip loading sentiment model (fills zeros) |
| INTEREST_CLASS_NAMES | 0:no_interest,1:mild_interest,2:strong_interest | Override class label mapping |
| FORCE_SENTIMENT_WIDTH | (blank) | Force sentiment feature width (2 or 7) if auto-detect fails |
| USE_GPU | 0 | If 1 and CUDA available, run models on GPU |

### 📦 Expected Pipeline Pickle Keys
`{ model, scaler, pca, original_feature_dim, feature_dim, n_components, sentiment_used, algorithm }`

`feature_dim = n_components + (# sentiment columns)`; backend infers 2 vs 7.

### 🚀 Run (Windows PowerShell / uv)
```powershell
uv run uvicorn fastapi_backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Or (plain):
```powershell
uv run python -m uvicorn fastapi_backend.main:app --reload
```

Warm first request: (downloads models + builds caches)
```powershell
curl http://127.0.0.1:8000/health
```

### 🧪 Example Prediction (curl)
```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"customer_summary":"Asked about whitening, wants Saturday appointment.","agent_summary":"Offered Saturday 10am slot and sent booking link."}' | jq
```

PowerShell (escaping quotes):
```powershell
curl -Method POST "http://127.0.0.1:8000/predict" -Body '{"customer_summary":"Needs implant consult","agent_summary":"Provided pricing tiers and offered free x-ray."}' -ContentType 'application/json'
```

### 🔄 Inference Flow
```mermaid
sequenceDiagram
  participant Client
  participant API
  participant HF as HF Models
  participant Pipe as Pipeline
  Client->>API: POST /predict JSON
  API->>HF: Embed customer, agent
  HF-->>API: 2 vectors
  API->>Pipe: scale + PCA (if present)
  API->>HF: (optional) sentiment
  API: concatenate features
  API->>Pipe: model.predict_proba
  Pipe-->>API: probability
  API-->>Client: prob + label
```

### 🧠 Performance Notes
| Aspect | Cold | Warm |
|--------|------|------|
| First /predict | Downloads embedding + sentiment weights | ~ few 10s ms embedding + PCA + LR |
| Sentiment skipped | N/A | Faster (zeros appended) |

To reduce latency: set `SKIP_SENTIMENT_MODEL=1` if sentiment was not used (`sentiment_used=false`).

### 🛡 Error Codes
| Code | Scenario |
|------|----------|
| 422 | Missing required JSON fields |
| 500 | Model file missing / bad pipeline keys |

### 🧩 Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| feature dim mismatch | Wrong MODEL_PATH or changed pickle | Use matching pipeline artifact |
| slow responses | Cold start each request | Keep process warm / preload with /health |
| CUDA not used | `USE_GPU=1` but no device | Verify `torch.cuda.is_available()` |

### 🔮 Planned Enhancements
Batch prediction endpoint • Multi-pipeline selection (10/50/100 comps) • Model card endpoint • Request tracing / timing metrics.

### ✅ Health Check
```json
GET /health -> {
  "status":"ok",
  "model_file":"logreg_pipeline.pkl",
  "device":"cpu",
  "embedding_model":"sentence-transformers/all-MiniLM-L6-v2",
  "sentiment_model":"distilbert-base-uncased-finetuned-sst-2-english"
}
```

---
Need another endpoint or optimization? Ask and it can be added.
