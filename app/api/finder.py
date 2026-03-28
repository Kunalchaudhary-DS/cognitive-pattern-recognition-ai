"""
Dataset finder route — takes a problem statement and returns
best matching datasets ranked by AI relevance score.
"""

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
        return JSONResponse(content={
            "matches":           matches,
            "problem_statement": problem_statement,
            "total_found":       len(matches)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)})