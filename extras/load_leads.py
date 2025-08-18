"""Utility script to load and inspect the leads dataset.

Run directly:
    python load_leads.py [--head 5]

Or import:
    from load_leads import load_leads
    df = load_leads()
"""
from __future__ import annotations
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Iterable
import pandas as pd

DATA_FILE = Path("leads.csv")

EXPECTED_COLUMNS = [
    "lead_id",
    "customer_summary",
    "agent_summary",
    "conversion_label",
]

LOGGER = logging.getLogger("load_leads")


class DataValidationError(Exception):
    """Raised when dataset validation fails."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_leads(path: Path | str = DATA_FILE, *, validate: bool = True, dedupe: bool = False) -> pd.DataFrame:
    """Load the leads CSV into a DataFrame and optionally validate.

    Parameters
    ----------
    path : Path | str
        Path to the CSV file.
    validate : bool
        Whether to run validation checks.
    """
    path = Path(path)
    if not path.exists():  # pragma: no cover
        raise FileNotFoundError(path)
    LOGGER.debug("Loading dataset from %s", path)
    df = pd.read_csv(path)
    if validate:
        try:
            validate_dataset(df, expected_columns=EXPECTED_COLUMNS)
        except DataValidationError as e:
            if dedupe and "Duplicate lead_id" in str(e):
                LOGGER.warning("Duplicates detected; applying dedupe keep=first")
                df = df.drop_duplicates(subset="lead_id", keep="first")
                # Re-run validation (should pass now or raise again)
                validate_dataset(df, expected_columns=EXPECTED_COLUMNS)
            else:
                raise
    # Coerce types after validation (ensures column presence first)
    if df["lead_id"].dtype == object:
        try:
            df["lead_id"] = df["lead_id"].astype(int)
        except ValueError:
            LOGGER.warning("Could not coerce lead_id to int; leaving as object")
    return df


def validate_dataset(df: pd.DataFrame, *, expected_columns: Iterable[str]) -> None:
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    extra = [c for c in df.columns if c not in expected_columns]
    if extra:
        LOGGER.info("Extra columns present (ignored for validation): %s", extra)
    # Null checks
    null_counts = df.isna().sum()
    if (null_counts > 0).any():
        raise DataValidationError(f"Null values found: {null_counts[null_counts>0].to_dict()}")
    # lead_id uniqueness
    if df['lead_id'].duplicated().any():
        dupes = df[df['lead_id'].duplicated()]['lead_id'].tolist()[:5]
        raise DataValidationError(f"Duplicate lead_id values detected (sample): {dupes}")
    # conversion_label binary
    labels = set(df['conversion_label'].unique())
    if not labels.issubset({0, 1}):
        raise DataValidationError(f"Unexpected conversion_label values: {labels}")
    LOGGER.debug("Validation passed: %d rows", len(df))

def summarize(df: pd.DataFrame, head: int = 5, *, show_lengths: bool = True) -> None:
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Class balance (conversion_label):")
    print(df["conversion_label"].value_counts().to_string())
    if show_lengths:
        for col in ["customer_summary", "agent_summary"]:
            lengths = df[col].str.len()
            print(f"Length stats [{col}] min={lengths.min()} mean={lengths.mean():.1f} max={lengths.max()}")
    print("\nHead:")
    print(df.head(head))
    print("\nInfo:")
    print(df.info())

def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect leads.csv")
    parser.add_argument("--path", type=Path, default=DATA_FILE, help="Path to leads.csv")
    parser.add_argument("--head", type=int, default=5, help="Number of rows to show in head()")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation checks")
    parser.add_argument("--validate-only", action="store_true", help="Run validation and exit")
    parser.add_argument("--hash", action="store_true", help="Print SHA256 hash of the raw CSV file")
    parser.add_argument("--dedupe", action="store_true", help="Automatically drop duplicate lead_id rows (keep first)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...)")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    if args.hash:
        print(_sha256(args.path))
    df = load_leads(args.path, validate=not args.no_validate, dedupe=args.dedupe)
    if args.validate_only:
        print("Validation passed.")
        return
    summarize(df, args.head)

if __name__ == "__main__":  # pragma: no cover
    main()
