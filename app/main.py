from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import APP_TITLE, APP_VERSION
from app.api import pages, dataset, preprocessing, training, analysis ,finder, kaggle

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Upload a CSV, auto-preprocess, train multiple ML models, and discover patterns."
)

# ── CORS — allows React (localhost:5173) to talk to FastAPI (localhost:8000) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register all routers ───────────────────────────────────────────────────────
app.include_router(pages.router)
app.include_router(dataset.router)
app.include_router(preprocessing.router)
app.include_router(training.router)
app.include_router(analysis.router)
app.include_router(finder.router)
app.include_router(kaggle.router)