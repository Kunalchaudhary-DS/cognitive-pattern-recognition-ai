"""
App-wide configuration.
Add any new settings here — never hardcode paths or values in routes/services.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # project root
DATASET_FOLDER = str(BASE_DIR / "datasets")
TEMPLATES_DIR = str(BASE_DIR / "templates")

# ── App metadata ───────────────────────────────────────────────────────────────
APP_TITLE = "Cognitive Pattern Recognition System"
APP_VERSION = "1.0.0"

# ── ML settings ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_SPLITS    = 5
CORRELATION_THRESHOLD          = 0.7   # used to flag strong correlations
CLASSIFICATION_UNIQUE_THRESHOLD = 10   # legacy — kept for compute_feature_importance

# ── 2-Stage smart training ──────────────────────────────────────────────────────
LARGE_DATASET_THRESHOLD = 2000   # rows — triggers 2-stage training above this
SCREEN_SAMPLE_SIZE      = 2000   # rows used in Stage-1 screening
TOP_N_FOR_FULL_TRAIN    = 3      # only top N models from screening proceed to full training

# ── Encoding ────────────────────────────────────────────────────────────────────
ONEHOT_CARDINALITY_LIMIT = 15    # categorical cols with ≤ this many unique values → OneHot

# ── Dataset validation ─────────────────────────────────────────────────────────
MIN_ROWS = 5
MIN_COLUMNS = 2