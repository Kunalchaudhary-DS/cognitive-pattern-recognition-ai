from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/")
def home():
    return JSONResponse(content={"status": "CPRS Backend running", "version": "1.0.0"})