# Interest Stage Multiclass Model Card

## Overview
Multinomial LogisticRegression classifier predicting customer interest stage:
- 0: no_interest
- 1: mild_interest
- 2: strong_interest

## Data
- Source file: `leads_1.csv`
- Rows (post-validation): 7300
- Features:
  - Text embeddings: concatenated customer + agent MiniLM-L6-v2 (768 * 2 = 1536 dims)
  - Sentiment (optional): 7 engineered sentiment-derived columns (present in this run)
  - PCA reduced components: chosen to retain >=95% variance (58 components after scaling + sentiment concatenation) OR fixed if specified.

## Training Configuration
- Standardization: Yes (per-feature Z-score on train split only)
- PCA: Enabled (variance threshold 0.95)
- Model: `sklearn.linear_model.LogisticRegression` (solver = saga, l2)
- Hyperparameter search: grid over C ∈ {0.1,1.0,3.0} and solvers {lbfgs,saga}
- Best params: see metrics JSON (`metrics_logreg.json`) key `params`
- Random seed: 42 for splitting and PCA

## Metrics (Holdout Test)
Extracted from `metrics_logreg.json`:
| Metric | Value |
| ------ | ----- |
| Macro ROC AUC (OvR) | 0.9977 |
| Macro F1 | 0.9917 |
| Weighted F1 | 0.9918 |
| Log Loss | 0.0591 |

Per-class performance:
| Class | Precision | Recall | F1 |
| ----- | --------- | ------ | --- |
| no_interest | 0.9939 | 0.9919 | 0.9929 |
| mild_interest | 0.9851 | 0.9914 | 0.9883 |
| strong_interest | 0.9960 | 0.9920 | 0.9940 |

Confusion matrix rows = true, cols = predicted:
```
[[490,  4,  0],
 [  2,463,  2],
 [  1,  3,495]]
```

## Calibration
- Log loss low; additional calibration (Platt / isotonic) not currently applied. Consider if decision thresholds become critical.

## Artifacts
- `artifacts/logreg_pipeline.pkl` : serialized dict {model, scaler, pca, feature_dim, original_feature_dim, classes}
- `artifacts/metrics_logreg.json` : metrics & search meta
- `artifacts/pca_transform.pkl` : PCA object (if PCA enabled)
- `artifacts/pca_variance.png` : variance curve (threshold mode)

## Inference Path
1. Load pipeline dict
2. (Optional) StandardScaler transform
3. (Optional) PCA transform
4. LogisticRegression predict_proba → argmax

## Limitations & Risks
- Assumes domain stability; distribution shift (new product categories, channel changes) may degrade performance.
- Sentiment model is English-focused; non-English utterances may reduce quality.
- PCA component count tied to a single variance threshold; large changes in upstream feature distributions may require re-selection.

## Maintenance Recommendations
- Recompute metrics monthly; compare macro F1 drift >1pp triggers retrain.
- Track class balance; if any class share <15%, consider class-weight tuning.
- Add calibration if future business rules require probability thresholds.

## Reproducibility
Command example:
```
uv run python interest_multiclass/train_logreg.py --standardize --pca --pca-variance 0.95 --search
```

## Contact
Model owner: (add name/contact)

*Auto-generated model card; update owner and any governance fields as needed.*
