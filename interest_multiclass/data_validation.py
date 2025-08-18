"""Validate leads_1.csv for multiclass interest task.
Run:
  uv run python interest_multiclass/data_validation.py
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any
from config import DATA_FILE, CLASS_LABEL_COLUMN, CLASS_IDS, CLASS_NAMES, BASE_DIR

REPORT_FILE = BASE_DIR / "data_integrity_report.json"

REQUIRED_COLUMNS = ["customer_summary", "agent_summary", CLASS_LABEL_COLUMN]


def validate(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "checks": {}
    }

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report["checks"]["missing_required_columns"] = missing

    nulls = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS if c in df.columns}
    report["checks"]["null_counts"] = nulls

    if CLASS_LABEL_COLUMN in df.columns:
        labels = sorted(df[CLASS_LABEL_COLUMN].dropna().unique().tolist())
    else:
        labels = []
    report["checks"]["unique_labels_found"] = labels
    unexpected = [l for l in labels if l not in CLASS_IDS]
    report["checks"]["unexpected_labels"] = unexpected

    counts = (
        df[CLASS_LABEL_COLUMN].value_counts().to_dict()
        if CLASS_LABEL_COLUMN in df.columns else {}
    )
    report["checks"]["label_counts"] = {int(k): int(v) for k, v in counts.items()}

    low = {int(k): int(v) for k, v in counts.items() if v < 5}
    report["checks"]["low_count_labels"] = low

    return report


def main():  # pragma: no cover
    if not DATA_FILE.exists():
        raise SystemExit(f"Data file not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    report = validate(df)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report -> {REPORT_FILE}")
    if report["checks"]["missing_required_columns"]:
        print("WARNING: Missing columns")
    if report["checks"]["unexpected_labels"]:
        print("WARNING: Unexpected labels present")

if __name__ == "__main__":  # pragma: no cover
    main()
