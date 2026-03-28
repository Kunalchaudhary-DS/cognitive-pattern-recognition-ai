"""
Preprocessing routes — feature importance and full preprocessing pipeline.
"""

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from app.core.state import state
from app.services.preprocessing_service import run_preprocessing, compute_feature_importance

router = APIRouter()


@router.post("/feature-importance/")
async def feature_importance(target_column: str = Form(...)):
    if state.df is None:
        return JSONResponse(content={"error": "No dataset uploaded"})

    df = state.df.copy()

    if target_column not in df.columns:
        return JSONResponse(content={"error": "Invalid target column"})

    result = compute_feature_importance(df, target_column)
    return JSONResponse(content=result)


@router.post("/preprocess/")
async def preprocess_data(target_column: str = Form(...)):
    if state.df is None:
        return JSONResponse(content={"error": "No dataset uploaded"})

    df = state.df.copy()

    if target_column not in df.columns:
        return JSONResponse(content={"error": "Invalid target column"})

    result = run_preprocessing(df, target_column)

    # Save processed data to shared state
    state.X             = result.pop("X")
    state.y             = result.pop("y")
    state.preprocessor  = result.pop("preprocessor")
    state.feature_names = result.pop("feature_names")
    state.problem_type  = result["problem_type"]
    state.target_column = target_column
    state.encoding_maps = result.get("encoding_maps", {})

    return JSONResponse(content=result)