from pathlib import Path
from dataclasses import dataclass
from typing import List

# Base paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR.parent / "leads_1.csv"  # expects columns: customer_summary, agent_summary, stage
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EMBEDDINGS_FILE = BASE_DIR / "embeddings_interest.npz"
SENTIMENT_FILE = BASE_DIR / "sentiment_interest.csv"
COMBINED_FEATURES_FILE = BASE_DIR / "features_interest.npz"
PCA_MODEL_FILE = ARTIFACTS_DIR / "pca_transform.pkl"
PCA_VARIANCE_PLOT = ARTIFACTS_DIR / "pca_variance.png"

# PCA configuration (can be overridden via CLI flags)
PCA_COMPONENTS_DEFAULT: int | None = None  # if None and --pca set, choose by variance threshold
PCA_VARIANCE_THRESHOLD = 0.95  # used if components not explicitly given

CLASS_LABEL_COLUMN = "stage"
CLASS_NAMES = ["no_interest", "mild_interest", "strong_interest"]
CLASS_IDS = [0, 1, 2]

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

RANDOM_STATE = 42
TEST_SIZE = 0.2

@dataclass
class RFParams:
    n_estimators: int = 400
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: str | None = "balanced"
    n_jobs: int = -1
    random_state: int = RANDOM_STATE

DEFAULT_RF_PARAMS = RFParams()

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
