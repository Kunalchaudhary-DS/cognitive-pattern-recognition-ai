"""
Training route — runs AutoML model pool and returns all results.
"""

import json
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.core.state import state
from app.services.training_service import run_training

router = APIRouter()


def _json_safe(obj):
    """Recursively make a dict JSON-serialisable (handles None, numpy types)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@router.post("/train/")
async def train_model():
    if state.X is None:
        return JSONResponse(content={"error": "Run preprocessing first"})

    output = run_training(state.X, state.y, state.problem_type)

    # ── Persist to state ──────────────────────────────────────────────────────
    state.best_model       = output["best_model"]
    state.scaler           = output["scaler"]
    state.needs_scaling    = output["needs_scaling"]
    state.X_test           = output["X_test"]
    state.y_test           = output["y_test"]

    # Merge metadata INTO training_results so analysis.py and the frontend can read them
    results_with_meta = dict(output["results"])
    results_with_meta["BestModel"]      = output["best_model_name"]
    results_with_meta["ProblemType"]    = state.problem_type
    results_with_meta["PrimaryMetric"]  = output["primary_metric"]   # key used to rank
    results_with_meta["Imbalanced"]     = output["imbalanced"]
    state.training_results = results_with_meta

    return JSONResponse(content=_json_safe(results_with_meta))