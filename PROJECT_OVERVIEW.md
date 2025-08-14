# Project Overview

Comprehensive documentation of the implemented pipeline, features, diagnostics, and artifacts to date.

## 1. Scope & Objectives
Transform a single text-based leads dataset (`leads.csv`) into:
- Validated, clean dataset (ID uniqueness, schema integrity).
- Rich text feature representations (transformer embeddings, sentiment, PCA reduction).
- Baseline & compressed models with diagnostics, leakage checks, robustness tests, and inference capability.

## 2. Data
File: `leads.csv`
Columns:
- `lead_id` (int, unique)
- `customer_summary` (str, non-null)
- `agent_summary` (str, non-null)
- `conversion_label` (int, {0,1})

Data Integrity Steps:
- Duplicate `lead_id` detection & resolution (4 adjusted IDs).
- Validation script: required columns, null checks, label set, uniqueness.

## 3. Core Scripts
| Script | Purpose |
|--------|---------|
| `load_leads.py` | Load & validate dataset (hashing, optional dedupe, CLI flags). |
| `generate_embeddings.py` | Generate MiniLM sentence embeddings (customer & agent, 384 + 384 -> 768 dims). |
| `baseline_model.py` | Train logistic regression on embeddings (+ optional sentiment or PCA features). |
| `sentiment_features.py` | Compute sentiment (signed + prob) with fallback to CSV if Parquet engine missing. |
| `model_diagnostics.py` | Leakage & robustness: shuffle test, token log-odds, duplicates, nested CV, coefficients. |
| `masking_stress_test.py` | Mask top discriminative tokens and measure AUC drop. |
| `inference.py` | End-to-end prediction (auto-detect sentiment feature need). |
| `pca_reduce.py` | Offline PCA reduction (variance plot, component matrix, optional sentiment concat). |
| `train_pca_pipeline.py` | Training-only (fold-split) PCA + calibration + component sweep + pipeline persistence. |

## 4. Feature Engineering
- Transformer model: `sentence-transformers/all-MiniLM-L6-v2` (mean pooling with attention mask).
- Sentiment model: `distilbert-base-uncased-finetuned-sst-2-english` (signed score + p_pos/p_neg + gap metric).
- Combined embedding feature: concat(customer_embedding, agent_embedding).
- Sentiment appended optionally (7 columns): `sentiment_cust`, `sentiment_agent`, `sentiment_cust_pos`, `sentiment_cust_neg`, `sentiment_agent_pos`, `sentiment_agent_neg`, `sentiment_gap`.
- PCA reduction: standardized embeddings -> PCA n components; optional sentiment after reduction.

## 5. Modeling
Primary classifier: Logistic Regression (`class_weight=balanced`, variable C via search/CV).
Artifacts: serialized model pickle + metrics JSON (timestamped) or unified pipeline pickle (scaler + PCA + model) for PCA sweep.

### PCA Pipeline Sweep (Calibrated Sigmoid)
| n_components | variance_sum | feature_dim | test_auc | brier |
|--------------|--------------|-------------|----------|-------|
| 10 | 0.3308 | 17 | 0.9993 | 0.0078 |
| 20 | 0.4519 | 27 | 0.9997 | 0.0061 |
| 30 | 0.5303 | 37 | 1.0000 | 0.0046 |
| 40 | 0.5883 | 47 | 0.9999 | 0.0044 |
| 50 | 0.6330 | 57 | 0.9999 | 0.0041 |
Chosen (tolerance 0.001): 10 components (compact with negligible AUC delta).

### Compression Impact
- Full: 775 dims (768 emb + 7 sentiment) – AUC ~0.9999.
- PCA 50 + sentiment: 57 dims – AUC ~0.9999 (~92.6% reduction).
- PCA 40 + sentiment: 47 dims – AUC ~0.9999.
- PCA 10 + sentiment: 17 dims – AUC 0.9993.

## 6. Diagnostics & Validation
| Check | Result |
|-------|--------|
| Shuffle label sanity | AUC ~0.46 (expected ~0.5) – no pipeline leakage. |
| Duplicate conflicting summaries | None detected. |
| Token log-odds | Strong semantic polarity (e.g., positive: "scheduled"; negative: "price", "fee"). |
| Nested CV | All outer folds AUC ~1.0. |
| Masking top 30 tokens | AUC drop ~0.0014 (minimal reliance on individual tokens). |
| Sentiment delta | Negligible vs embeddings (redundant). |
| Calibration | Brier improved with more components (but small absolute gains). |

## 7. Robustness Measures
- Masking stress test confirms signal is distributed beyond a sparse vocabulary.
- PCA+calibration pipeline reduces dimensionality and potential overfitting surface.
- Shuffle labels confirm evaluation path purity.

## 8. Inference
`inference.py` workflow:
1. Load model.
2. Generate embeddings for new rows.
3. Auto-detect feature dimension mismatch -> compute sentiment if needed.
4. Output `prediction` (0/1) + `probability`.
Optional metadata JSON (timestamp, model path, feature dim, sentiment usage).

`best_pca_pipeline.pkl` (from sweep) stores combined preprocessing; a future helper can wrap end-to-end (not yet added).

## 9. Artifacts Generated
| Artifact Type | Pattern / Example |
|---------------|-------------------|
| Embeddings NPZ | `embeddings.npz` |
| Sentiment CSV / JSON | `sentiment.csv`, `sentiment.json` |
| Baseline models | `artifacts/baseline_logreg_*.pkl` |
| Metrics JSON | `artifacts/metrics_*.json` |
| PCA feature NPZ | `pca_50_sentiment_std.npz` + `.json` meta |
| PCA components | `pca_50_sentiment_std.components.npy` |
| Variance plot | `pca_50_sentiment_std.variance.png` |
| Pipeline sweep results | `pca_sweep_results.json` |
| Best pipeline pickle | `best_pca_pipeline.pkl` |
| Stress test results | `mask_results.json` |
| Predictions | `predictions.csv` |

## 10. Tests
Total: 8 passing.
| Test File | Purpose |
|-----------|---------|
| `tests/test_load_leads.py` | Schema & integrity checks. |
| `tests/test_pooling_utils.py` | Mean pooling correctness. |
| `tests/test_sentiment_features.py` | Sentiment computation logic (mocked). |
| `tests/test_feature_combination.py` | Feature concatenation shape & error handling. |
| `tests/test_pca_reduce.py` | PCA NPZ structure. |
| (Others embed sanity through prior scripts). |

## 11. Key Design Choices
- Mean pooling (mask-aware) chosen for speed & simplicity (vs CLS token).
- Logistic regression for transparency and speed; perfect separation made complex models unnecessary.
- Calibration optional: added sigmoid method in PCA pipeline sweep.
- PCA fit only on training split in sweep to avoid variance leakage.
- Sentiment fallback to CSV ensures resilience without `pyarrow`.

## 12. Current Limitations / Assumptions
- Single dataset; no temporal split => can’t measure temporal generalization.
- Perfect AUC suggests domain-easy separation; real-world drift risk remains.
- Sentiment features redundant; retained for interpretability but could be dropped to cut 7 dims.
- No probability calibration for original full model (only PCA sweep). Could add if deploying.

## 13. Recommended Next Enhancements (Optional)
| Category | Enhancement |
|----------|-------------|
| Calibration | Add isotonic/Platt to non-PCA path & inference integration. |
| Explainability | SHAP on PCA space or raw embeddings (sample subset). |
| Drift Monitoring | Track embedding centroid, variance over time. |
| Threshold Strategy | Optimize threshold vs precision/recall or business cost curve. |
| Unified Pipeline Inference | Script to apply `best_pca_pipeline.pkl` directly (embedding + sentiment + predict). |
| Model Card | Add `MODEL_CARD.md` with fairness, limitations, intended use. |
| Plotting | ROC / PR / calibration curves saving. |

## 14. Command Reference (Core)
```powershell
# Embeddings
uv run python generate_embeddings.py --output embeddings.npz

# Sentiment (CSV fallback automatically if no parquet engine)
uv run python sentiment_features.py --output sentiment.parquet

# Baseline model (embeddings + sentiment)
uv run python baseline_model.py --embeddings embeddings.npz --sentiment sentiment.csv --cv-folds 5 --search --save-model

# Diagnostics
uv run python model_diagnostics.py --embeddings embeddings.npz --sentiment sentiment.csv

# Masking stress test
uv run python masking_stress_test.py --embeddings embeddings.npz --top-n 15 --output mask_results.json

# PCA reduction + artifacts
uv run python pca_reduce.py --embeddings embeddings.npz --sentiment sentiment.csv --n-components 50 --standardize --plot --save-components --output pca_50_sentiment_std.npz

# PCA sweep + calibration
uv run python train_pca_pipeline.py --embeddings embeddings.npz --sentiment sentiment.csv --components 10,20,30,40,50 --standardize --calibrate sigmoid --results pca_sweep_results.json --save-best best_pca_pipeline.pkl

# Inference (auto sentiment if needed)
uv run python inference.py --model artifacts\baseline_logreg_*.pkl --input leads.csv --output predictions.csv
```

## 15. Summary
The project now includes a complete, reproducible NLP feature pipeline, robust diagnostics, dimensionality reduction, model calibration (in PCA context), inference tooling, and comprehensive documentation/tests. Remaining optional steps are incremental hardening and explainability rather than core functionality.

---
*Generated documentation snapshot.*
