"""
Dataset routes — upload, load demo datasets, list demo datasets.
"""

import os
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.core.state import state
from app.core.config import DATASET_FOLDER
from app.services.dataset_service import (
    validate_dataframe,
    load_csv_bytes,
    load_csv_path,
    build_upload_profile,
    build_demo_profile,
    compute_strong_correlations,
)
from app.services.dataset_finder_service import _get_all_datasets

router = APIRouter()

DEMO_DATASETS = [
    {"file": "accounting_dataset.csv",                          "name": "Accounting Dataset",                    "category": "Finance"},
    {"file": "cricket_mini.csv",                                "name": "Cricket Statistics Dataset",            "category": "Sports"},
    {"file": "Global_Education.csv",                            "name": "Global Education Dataset",              "category": "Education"},
    {"file": "healthcare_dataset.csv",                          "name": "Healthcare Dataset",                    "category": "Healthcare"},
    {"file": "healthcare-dataset-stroke-data.csv",              "name": "Stroke Prediction Dataset",             "category": "Healthcare"},
    {"file": "HR_Attrition.csv",                                "name": "HR Attrition Dataset",                  "category": "Human Resources"},
    {"file": "International_Education_Costs.csv",               "name": "International Education Costs Dataset", "category": "Education"},
    {"file": "personal_finance_tracker_dataset.csv",            "name": "Personal Finance Tracker Dataset",      "category": "Finance"},
    {"file": "retail-grocery-customers.csv",                    "name": "Retail Grocery Customers Dataset",      "category": "Retail"},
    {"file": "Smartphone_Usage_Productivity_Dataset_50000.csv", "name": "Smartphone Usage & Productivity Dataset","category": "Technology"},
    {"file": "StudentsPerformance.csv",                         "name": "Students Performance Dataset",          "category": "Education"},
    {"file": "synthetic_personal_finance_dataset.csv",          "name": "Synthetic Personal Finance Dataset",    "category": "Finance"},
]


@router.get("/demo-datasets/")
async def get_demo_datasets():
    """Return all browseable datasets including Kaggle downloads."""
    all_ds = _get_all_datasets()
    # Return only fields the frontend needs (file, name, category)
    return {"datasets": [
        {"file": ds["file"], "name": ds["name"], "category": ds.get("category", "Other")}
        for ds in all_ds
        if os.path.exists(os.path.join(DATASET_FOLDER, ds["file"]))
    ]}


@router.post("/load-demo-dataset/")
async def load_demo_dataset(dataset_name: str = Form(...)):
    file_path = os.path.join(DATASET_FOLDER, dataset_name)

    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "Dataset not found"})

    df = load_csv_path(file_path)
    error = validate_dataframe(df)
    if error:
        return JSONResponse(content={"error": error})

    state.df = df.copy()
    state.strong_correlations = compute_strong_correlations(df)

    return JSONResponse(content=build_demo_profile(df))


@router.post("/upload/")
async def upload_dataset(file: UploadFile = File(...)):
    contents = await file.read()
    df = load_csv_bytes(contents)

    error = validate_dataframe(df)
    if error:
        return JSONResponse(content={"error": error})

    state.df = df.copy()
    state.strong_correlations = compute_strong_correlations(df)

    return JSONResponse(content=build_upload_profile(df))