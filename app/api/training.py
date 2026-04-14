"""
Training route — runs AutoML model pool and returns all results.
"""

import json
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.core.state import state
from app.services.training_service import run_training
from app.services.constraint_service import extract_statistical_constraints, merge_constraints
from app.services.ai_service import generate_semantic_constraints

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

    # Persist model to state
    state.best_model       = output["best_model"]
    state.scaler           = output["scaler"]
    state.needs_scaling    = output["needs_scaling"]
    state.X_test           = output["X_test"]
    state.y_test           = output["y_test"]

    # Merge metadata INTO training_results so analysis.py and the frontend can read them
    results_with_meta = dict(output["results"])
    results_with_meta["BestModel"]      = output["best_model_name"]
    results_with_meta["ProblemType"]    = state.problem_type
    results_with_meta["PrimaryMetric"]  = output["primary_metric"]
    results_with_meta["Imbalanced"]     = output["imbalanced"]
    state.training_results = results_with_meta

    # Semantic Prediction Interceptor — Constraint Generation 
    # Only for regression (classification output is already a finite class label)
    constraint_summary = {}
    if state.problem_type == "regression" and state.df is not None and state.target_column:
        try:
            # Layer 1: Statistical extraction from training data
            statistical = extract_statistical_constraints(
                df             = state.df,
                target_column  = state.target_column,
                problem_type   = state.problem_type,
            )

            # Layer 2: Ollama semantic enrichment (gracefully returns {} if offline)
            all_cols  = list(state.df.columns)
            dtype_map = {
                col: str(state.df[col].dtype)
                for col in all_cols
            }
            col_names_no_target = [c for c in all_cols if c != state.target_column]

            semantic = generate_semantic_constraints(
                column_names      = col_names_no_target + [state.target_column],
                target_column     = state.target_column,
                dtype_map         = dtype_map,
                statistical_bounds= statistical,
                problem_type      = state.problem_type,
            )

            # Layer 3: Merge statistical + semantic into final rulebook
            state.constraint_map = merge_constraints(statistical, semantic)

            # Build a human-readable summary for the frontend
            tb = state.constraint_map.get("target_bounds", {})
            if tb:
                constraint_summary = {
                    "effective_min":   tb.get("effective_min"),
                    "effective_max":   tb.get("effective_max"),
                    "reason_min":      tb.get("reason_min"),
                    "reason_max":      tb.get("reason_max"),
                    "relative_rules":  len(state.constraint_map.get("relative_rules", [])),
                    "soft_warnings":   len(state.constraint_map.get("soft_warnings", [])),
                }

            print(f"[Training] Constraint map generated: {state.constraint_map}")

        except Exception as e:
            # Constraint generation must never break training
            print(f"[Training] Constraint generation failed (non-fatal): {e}")
            state.constraint_map = {}

    # Include constraint summary in the training response
    results_with_meta["ConstraintMap"] = constraint_summary

    return JSONResponse(content=_json_safe(results_with_meta))