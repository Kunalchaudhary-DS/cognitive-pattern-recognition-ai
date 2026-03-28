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
TEST_SIZE = 0.2
CV_SPLITS = 5
CORRELATION_THRESHOLD = 0.7        # used to flag strong correlations
CLASSIFICATION_UNIQUE_THRESHOLD = 10   # if target has ≤ this many unique values → classification

# ── Dataset validation ─────────────────────────────────────────────────────────
MIN_ROWS = 5
MIN_COLUMNS = 2