# Dataset finder route — matches a problem statement to local datasets by relevance.

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from app.services.dataset_finder_service import find_matching_datasets

router = APIRouter()


@router.post("/find-datasets/")
async def find_datasets(problem_statement: str = Form(...)):
    if not problem_statement.strip():
        return JSONResponse(content={"error": "Please enter a problem statement"})

    try:
        matches = find_matching_datasets(problem_statement, top_n=5)
        no_local_match = len(matches) == 0

        return JSONResponse(content={
            "matches":           matches,
            "problem_statement": problem_statement,
            "total_found":       len(matches),
            "no_local_match":    no_local_match,
            "suggestion":        (
                "No local dataset matched your problem statement. "
                "Try searching Kaggle to find and download a relevant dataset."
            ) if no_local_match else None,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
