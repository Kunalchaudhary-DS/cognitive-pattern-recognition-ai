"""
Training route — runs AutoML model pool and returns all results.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.core.state import state
from app.services.training_service import run_training

router = APIRouter()


@router.post("/train/")
async def train_model():
    if state.X is None:
        return JSONResponse(content={"error": "Run preprocessing first"})

    output = run_training(state.X, state.y, state.problem_type)

    # Persist best model and metadata to state
    state.best_model      = output["best_model"]
    state.scaler          = output["scaler"]
    state.needs_scaling   = output["needs_scaling"]
    state.training_results = output["results"]

    response = dict(output["results"])
    response["BestModel"]   = output["best_model_name"]
    response["ProblemType"] = state.problem_type

    return JSONResponse(content=response)