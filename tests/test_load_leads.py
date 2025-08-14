import sys
from pathlib import Path
import pandas as pd

# Ensure project root is on path for direct script import when not installed as a package
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_leads import load_leads, EXPECTED_COLUMNS  # type: ignore

DATA_PATH = Path('leads.csv')


def test_csv_exists():
    assert DATA_PATH.exists(), 'leads.csv missing'


def test_load_and_validate():
    df = load_leads(DATA_PATH)
    # Basic expectations
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) > 0
    assert df['lead_id'].is_unique
    assert set(df['conversion_label'].unique()) <= {0, 1}


def test_no_nulls():
    df = load_leads(DATA_PATH)
    assert not df.isna().any().any(), 'Null values unexpectedly present'
