# Interest Multiclass Pipeline

Predict interest stage (0=no_interest, 1=mild_interest, 2=strong_interest) from customer & agent summaries using:
1. Text embeddings (customer + agent concatenated)
2. OPTIONAL sentiment feature block (7 columns by default) for richer signal
3. Multinomial LogisticRegression (sole model now; RandomForest removed)

Nothing optional is skipped below: every auxiliary / optional step is explicitly listed so you can choose deliberately.

## Repository Components (this subfolder)
- `config.py` : Central config (paths, labels, defaults)
- `data_validation.py` : Schema & label checks; writes `data_integrity_report.json`
- `generate_embeddings_interest.py` : Generates dual-embedding (customer+agent) NPZ
- `generate_sentiment_interest.py` : Dedicated sentiment feature generator (mirrors global script; outputs CSV + JSON meta)
- `train_logreg.py` : Trains multinomial LogisticRegression (optional param search) saves `logreg_pipeline.pkl` + metrics
- `inference.py` : Batch inference over a CSV using saved pipeline
- (Optional external) `sentiment_features.py` (root) : Legacy sentiment generator (alternative to local script)
- Tests: `tests/test_interest_*` ensure validation + RF basics

## End-to-End Steps (Mandatory & Optional Flags)
Order is important; optional steps marked (OPTIONAL) but still fully documented.

### 1. Data Validation (MANDATORY)
```powershell
uv run python interest_multiclass/data_validation.py
```
Generates `interest_multiclass/data_integrity_report.json`. Inspect for:
- `missing_required_columns` must be []
- `unexpected_labels` must be []

### 2. Embeddings Generation (MANDATORY before training)
```powershell
uv run python interest_multiclass/generate_embeddings_interest.py --data leads_1.csv
```
Outputs: `interest_multiclass/embeddings_interest.npz` with arrays: combined, customer, agent.

### 3. Sentiment Features (OPTIONAL but RECOMMENDED for 775-dim parity)
Two equivalent ways; prefer the local script for isolation:
Local script:
```powershell
uv run python interest_multiclass/generate_sentiment_interest.py
```
Only signed + gap (reduced 3 cols):
```powershell
uv run python interest_multiclass/generate_sentiment_interest.py --signed-only
```
OR reuse root generic script:
```powershell
uv run python sentiment_features.py --data leads_1.csv --merge-csv --output interest_multiclass/sentiment_interest.csv
```
Result file expected by training: `interest_multiclass/sentiment_interest.csv`

### 4. Feature Assembly (IMPLICIT)
`train_logreg.py` internally assembles by horizontally stacking embeddings + (if present) sentiment CSV. No separate manual step required, but you can inspect saved `features_interest.npz` afterward.

### 5. LogisticRegression Training (MANDATORY; PCA OPTIONAL)
Basic train with defaults:
```powershell
uv run python interest_multiclass/train_logreg.py
```
With simple parameter search (C, solver):
```powershell
uv run python interest_multiclass/train_logreg.py --search
```
Add standardization (OPTIONAL – usually helpful):
```powershell
uv run python interest_multiclass/train_logreg.py --standardize --search
```
Add PCA selecting first k components explicitly:
```powershell
uv run python interest_multiclass/train_logreg.py --standardize --pca --pca-components 200
```
Add PCA choosing minimum components for 95% variance (change with --pca-variance):
```powershell
uv run python interest_multiclass/train_logreg.py --standardize --pca --pca-variance 0.95
```
Outputs:
- `interest_multiclass/artifacts/logreg_pipeline.pkl`
- `interest_multiclass/artifacts/metrics_logreg.json`
- `interest_multiclass/features_interest.npz`
- (if PCA) `interest_multiclass/artifacts/pca_transform.pkl` and `pca_variance.png`

### 6. Inference (MANDATORY for producing predictions)
```powershell
uv run python interest_multiclass/inference.py --model interest_multiclass/artifacts/logreg_pipeline.pkl --input leads_1.csv --output predictions_interest.csv
```
Generates `predictions_interest.csv` with columns: original + predicted_stage + probs JSON column.

### 7. (OPTIONAL) FastAPI Integration
Point the FastAPI backend to `logreg_pipeline.pkl` (update model path / env). Ensure sentiment width inference aligns with whether sentiment was generated.

### 8. (OPTIONAL) Hyperparameter Iteration
Adjust grid inside `train_logreg.py` (edit `search_grid`).

### 9. (OPTIONAL) Benchmark / Diagnostics
Add SHAP / permutation importance scripts if deeper explainability required.

### 10. Tests (RECOMMENDED)
```powershell
uv run pytest -q tests/test_interest_*.py
```

## Artifact Inventory
Under `interest_multiclass/`:
- `embeddings_interest.npz` (MANDATORY for training)
- `sentiment_interest.csv` (OPTIONAL, if sentiment generated)
- `features_interest.npz` (auto-created during training)
- `artifacts/logreg_pipeline.pkl` (model pipeline dict)
- `artifacts/metrics_logreg.json` (evaluation metrics)
- `artifacts/pca_transform.pkl` (PCA object if used)
- `artifacts/pca_variance.png` (variance curve if created)
- `data_integrity_report.json` (validation output)
- `sentiment_interest.json` (sentiment metadata; optional)

## Metrics Reference (metrics_logreg.json keys)
- macro_roc_auc_ovr
- macro_f1
- weighted_f1
- log_loss (may be null if error)
- per_class.{name}.{precision,recall,f1}
- confusion_matrix
- search (contains grid search result if used)

## Class Mapping
```
0 -> no_interest
1 -> mild_interest
2 -> strong_interest
```

## Repro Tips
- Set `PYTHONHASHSEED=0` (optional) for extra determinism.
- LogisticRegression randomness controlled by `random_state` seeds indirectly (data split). Solver stochasticity (saga) also benefits from set seeds.

---
Scaffold + documentation kept exhaustive to avoid silent omissions.
