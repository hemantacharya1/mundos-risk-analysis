import os, sys, json
from pathlib import Path
import pandas as pd

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INT_DIR = ROOT / 'interest_multiclass'

from interest_multiclass import data_validation  # noqa: E402
from interest_multiclass.config import CLASS_LABEL_COLUMN

def test_validation_basic(tmp_path):
    df = pd.DataFrame({
        'customer_summary': ['a','b','c','d','e'],
        'agent_summary': ['x','y','z','w','v'],
        CLASS_LABEL_COLUMN: [0,1,2,1,0]
    })
    report = data_validation.validate(df)
    assert report['checks']['missing_required_columns'] == []
    assert sorted(report['checks']['unique_labels_found']) == [0,1,2]

