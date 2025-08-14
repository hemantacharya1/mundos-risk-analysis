## Lead Scoring FastAPI Backend

Lightweight inference service for the calibrated PCA + LogisticRegression lead conversion model.

### 🔧 Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Liveness + model metadata |
| POST | /predict | Score a single (customer_summary, agent_summary) pair |

### 🧾 Request / Response
POST /predict JSON body:
```json
{
  "customer_summary": "Asked about whitening, wants Saturday appointment.",
  "agent_summary": "Offered Saturday 10am slot and sent booking link."
}
```
Optional query param: `threshold` (default 0.5) to produce binary label.

Sample response:
```json
{
  "conversion_probability": 0.8734,
  "prediction": 1,
  "threshold": 0.5,
  "model_file": "best_pca_pipeline.pkl",
  "n_components": 10,
  "variance_sum": 0.3308,
  "sentiment_used": true,
  "calibrated": true
}
```

### ⚙️ Environment Variables
| Name | Default | Description |
|------|---------|-------------|
| MODEL_PATH | best_pca_pipeline.pkl | Path to saved pipeline dict |
| EMB_MODEL | sentence-transformers/all-MiniLM-L6-v2 | HuggingFace embedding model |
| SENT_MODEL | distilbert-base-uncased-finetuned-sst-2-english | Sentiment model (SST-2) |
| SKIP_SENTIMENT_MODEL | 0 | If 1, skip loading sentiment model (fills zeros) |
| USE_GPU | 0 | If 1 and CUDA available, run models on GPU |

### 📦 Expected Pipeline Pickle Keys
`{ scaler, pca, model, n_components, variance_sum, sentiment_used, feature_dim, calibrated }`

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
curl -s -X POST "http://127.0.0.1:8000/predict?threshold=0.55" \
  -H "Content-Type: application/json" \
  -d '{"customer_summary":"Asked about whitening, wants Saturday appointment.","agent_summary":"Offered Saturday 10am slot and sent booking link."}' | jq
```

PowerShell (escaping quotes):
```powershell
curl -Method POST "http://127.0.0.1:8000/predict?threshold=0.55" -Body '{"customer_summary":"Needs implant consult","agent_summary":"Provided pricing tiers and offered free x-ray."}' -ContentType 'application/json'
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
  API->>Pipe: scale + PCA
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
  "model_file":"best_pca_pipeline.pkl",
  "device":"cpu",
  "embedding_model":"sentence-transformers/all-MiniLM-L6-v2",
  "sentiment_model":"distilbert-base-uncased-finetuned-sst-2-english"
}
```

---
Need another endpoint or optimization? Ask and it can be added.
