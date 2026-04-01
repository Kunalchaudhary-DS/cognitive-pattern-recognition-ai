"""
Kaggle routes — search and download datasets from Kaggle.
"""

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from app.services.kaggle_service import search_kaggle_datasets, download_kaggle_dataset

router = APIRouter()


@router.post("/kaggle-search/")
async def kaggle_search(problem_statement: str = Form(...)):
    if not problem_statement.strip():
        return JSONResponse(content={"error": "Please enter a problem statement"})

    results = search_kaggle_datasets(problem_statement)
    return JSONResponse(content={
        "datasets":          results,
        "problem_statement": problem_statement,
        "total_found":       len(results)
    })


@router.post("/kaggle-download/")
async def kaggle_download(
    dataset_ref:   str = Form(...),
    dataset_title: str = Form(...)
):
    if not dataset_ref.strip():
        return JSONResponse(content={"error": "Dataset reference is required"})

    result = download_kaggle_dataset(dataset_ref, dataset_title)
    return JSONResponse(content=result)