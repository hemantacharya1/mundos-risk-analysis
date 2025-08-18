"""Create evaluation plots (confusion matrix heatmap + per-class ROC curves) for latest metrics/model.
Usage:
  uv run python interest_multiclass/analysis/plot_evaluation.py \
    --model interest_multiclass/artifacts/logreg_pipeline.pkl \
    --metrics interest_multiclass/artifacts/metrics_logreg.json \
    --data leads_1.csv --output-dir interest_multiclass/analysis
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
try:
    from interest_multiclass.config import CLASS_LABEL_COLUMN, CLASS_NAMES
except ModuleNotFoundError:
    import sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from interest_multiclass.config import CLASS_LABEL_COLUMN, CLASS_NAMES


def load_pipeline(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_features(pipe, df: pd.DataFrame):
    # This is minimal: we rely on precomputed feature shapes; for full regeneration we would re-embed
    # Here we assume predictions were made already and we only visualize using stored confusion matrix.
    # For ROC we need proba; recompute via pipeline if embeddings & sentiment are still present.
    from interest_multiclass.inference import assemble_features  # reuse existing logic
    X = assemble_features()
    scaler = pipe.get('scaler')
    if scaler is not None:
        X = scaler.transform(X)
    pca = pipe.get('pca')
    if pca is not None:
        X = pca.transform(X)
    return X


def plot_confusion(cm: np.ndarray, out: Path):
    plt.figure(figsize=(4.5,4))
    im = plt.imshow(cm, cmap='Blues')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha='right')
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha='center', va='center', fontsize=9)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_roc(y_true, proba, out: Path):
    plt.figure(figsize=(5.5,4.5))
    n_classes = proba.shape[1]
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba[:,c])
        plt.plot(fpr, tpr, label=f"{CLASS_NAMES[c]} AUC={auc(fpr,tpr):.3f}")
    plt.plot([0,1],[0,1],'k--',linewidth=0.8)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('One-vs-Rest ROC Curves')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--metrics', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metrics, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    cm_list = metrics.get('confusion_matrix')
    if cm_list is None:
        raise SystemExit('confusion_matrix not found in metrics file.')
    cm = np.array(cm_list)
    plot_confusion(cm, args.output_dir / 'confusion_matrix.png')

    # Recompute probabilities for ROC curves
    df = pd.read_csv(args.data)
    pipe = load_pipeline(args.model)
    X = build_features(pipe, df)
    model = pipe['model']
    proba = model.predict_proba(X)
    # Align y for the current dataset
    if CLASS_LABEL_COLUMN in df.columns:
        y_true = df[CLASS_LABEL_COLUMN].to_numpy()
        plot_roc(y_true, proba, args.output_dir / 'roc_ovr.png')

    with open(args.output_dir / 'eval_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'confusion_matrix_file': 'confusion_matrix.png',
            'roc_file': 'roc_ovr.png',
            'classes': CLASS_NAMES,
            'n_samples': int(len(df)),
            'pca_used': pipe.get('pca') is not None,
            'feature_dim': pipe.get('feature_dim')
        }, f, indent=2)
    print('Wrote evaluation plots and summary.')

if __name__ == '__main__':  # pragma: no cover
    main()
