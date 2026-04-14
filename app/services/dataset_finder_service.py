"""
Dataset Finder Service — Smart keyword-based matching engine.
Fast, accurate, no AI needed for scoring.
Phi-3 only used for generating the reason text after matching.

Auto-discovers all CSV files in the datasets/ folder, including
Kaggle-downloaded datasets, so every file is searchable locally.
"""

import os
import re
import json
import pandas as pd
from app.core.config import DATASET_FOLDER


#Persistent registry path
REGISTRY_PATH = os.path.join(DATASET_FOLDER, "_dataset_registry.json")


# Hardcoded demo datasets (original curated list)
DEMO_DATASETS = [
    {
        "file": "accounting_dataset.csv",
        "name": "Accounting Dataset",
        "category": "Finance",
        "keywords": ["accounting", "finance", "financial", "budget", "revenue", "expense",
                     "profit", "loss", "tax", "audit", "balance", "sheet", "income",
                     "money", "cost", "price", "payment", "invoice", "transaction"]
    },
    {
        "file": "cricket_mini.csv",
        "name": "Cricket Statistics Dataset",
        "category": "Sports",
        "keywords": ["cricket", "sport", "player", "match", "score", "batting", "bowling",
                     "wicket", "run", "team", "tournament", "performance", "athlete",
                     "game", "win", "lose", "statistics", "predict"]
    },
    {
        "file": "Global_Education.csv",
        "name": "Global Education Dataset",
        "category": "Education",
        "keywords": ["education", "school", "student", "learning", "global", "country",
                     "literacy", "enrollment", "academic", "university", "college",
                     "graduation", "study", "knowledge", "teacher", "curriculum"]
    },
    {
        "file": "healthcare_dataset.csv",
        "name": "Healthcare Dataset",
        "category": "Healthcare",
        "keywords": ["health", "medical", "patient", "hospital", "disease", "doctor",
                     "treatment", "diagnosis", "medicine", "clinical", "care", "drug",
                     "symptom", "illness", "healthcare", "blood", "test", "report"]
    },
    {
        "file": "healthcare-dataset-stroke-data.csv",
        "name": "Stroke Prediction Dataset",
        "category": "Healthcare",
        "keywords": ["stroke", "brain", "heart", "risk", "predict", "health", "medical",
                     "patient", "age", "blood", "pressure", "glucose", "bmi", "smoke",
                     "disease", "neural", "clinical", "diagnosis", "hypertension"]
    },
    {
        "file": "HR_Attrition.csv",
        "name": "HR Attrition Dataset",
        "category": "Human Resources",
        "keywords": ["employee", "attrition", "hr", "human", "resource", "turnover",
                     "retention", "salary", "job", "work", "resign", "quit", "hire",
                     "department", "manager", "performance", "satisfaction", "company",
                     "workforce", "staff", "leave", "promotion", "role"]
    },
    {
        "file": "International_Education_Costs.csv",
        "name": "International Education Costs Dataset",
        "category": "Education",
        "keywords": ["education", "cost", "tuition", "fee", "university", "international",
                     "study", "abroad", "college", "expense", "scholarship", "afford",
                     "country", "student", "finance", "budget", "school"]
    },
    {
        "file": "personal_finance_tracker_dataset.csv",
        "name": "Personal Finance Tracker Dataset",
        "category": "Finance",
        "keywords": ["personal", "finance", "money", "budget", "expense", "saving",
                     "spending", "income", "track", "transaction", "bank", "credit",
                     "debit", "payment", "monthly", "financial", "plan", "investment"]
    },
    {
        "file": "retail-grocery-customers.csv",
        "name": "Retail Grocery Customers Dataset",
        "category": "Retail",
        "keywords": ["retail", "grocery", "customer", "shopping", "purchase", "product",
                     "store", "buy", "sell", "market", "consumer", "basket", "order",
                     "sales", "revenue", "loyalty", "segment", "behavior", "recommend"]
    },
    {
        "file": "Smartphone_Usage_Productivity_Dataset_50000.csv",
        "name": "Smartphone Usage & Productivity Dataset",
        "category": "Technology",
        "keywords": ["smartphone", "mobile", "phone", "app", "usage", "screen", "time",
                     "productivity", "technology", "digital", "device", "social", "media",
                     "internet", "user", "behavior", "addiction", "notification", "hour"]
    },
    {
        "file": "StudentsPerformance.csv",
        "name": "Students Performance Dataset",
        "category": "Education",
        "keywords": ["student", "performance", "grade", "score", "exam", "test", "academic",
                     "school", "math", "reading", "writing", "education", "pass", "fail",
                     "predict", "result", "gpa", "mark", "class", "subject"]
    },
    {
        "file": "synthetic_personal_finance_dataset.csv",
        "name": "Synthetic Personal Finance Dataset",
        "category": "Finance",
        "keywords": ["finance", "personal", "synthetic", "money", "saving", "income",
                     "expense", "budget", "investment", "loan", "debt", "credit",
                     "bank", "financial", "wealth", "asset", "liability", "tax"]
    },
]


# Registry helpers

def _load_registry() -> list:
    """Load the persistent dataset registry from disk."""
    if not os.path.exists(REGISTRY_PATH):
        return []
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_registry(entries: list):
    """Persist the dataset registry to disk."""
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def _generate_keywords_from_file(file_path: str) -> list:
    """
    Auto-generate search keywords from a CSV's filename + column headers.
    Provides reasonable searchability even without curated keywords.
    """
    keywords = set()

    # Keywords from filename
    basename = os.path.splitext(os.path.basename(file_path))[0]
    name_words = re.findall(r'[a-zA-Z]+', basename.lower())
    noise = {"csv", "data", "dataset", "the", "and", "for", "with", "from"}
    keywords.update(w for w in name_words if len(w) > 2 and w not in noise)

    # Keywords from column headers
    try:
        try:
            df = pd.read_csv(file_path, nrows=0)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, nrows=0, encoding="latin1")

        for col in df.columns:
            col_words = re.findall(r'[a-zA-Z]+', col.lower())
            keywords.update(w for w in col_words if len(w) > 2 and w not in noise)
    except Exception:
        pass

    return sorted(keywords)


def _prettify_filename(filename: str) -> str:
    """Convert 'some_file_name.csv' → 'Some File Name Dataset'."""
    name = os.path.splitext(filename)[0]
    words = re.findall(r'[a-zA-Z0-9]+', name)
    title = " ".join(w.capitalize() for w in words)
    if "dataset" not in title.lower():
        title += " Dataset"
    return title


def register_dataset(
    filenames: list[str],
    kaggle_ref: str = "",
    kaggle_title: str = "",
    category: str = "Kaggle"
) -> None:
    """
    Register one or more newly-downloaded CSV files into the local
    dataset registry so they appear in local search immediately.

    Called by kaggle_service after a successful download.
    """
    registry = _load_registry()
    known_files = {entry["file"] for entry in registry}

    for fname in filenames:
        if fname in known_files:
            continue

        file_path = os.path.join(DATASET_FOLDER, fname)
        keywords = _generate_keywords_from_file(file_path)

        entry = {
            "file":     fname,
            "name":     kaggle_title or _prettify_filename(fname),
            "category": category,
            "keywords": keywords,
            "source":   "kaggle",
            "ref":      kaggle_ref,
        }
        registry.append(entry)

    _save_registry(registry)
    print(f"[Registry] Registered {len(filenames)} file(s) — total entries: {len(registry)}")


#Auto-discover unregistered CSV files 
def _get_all_datasets() -> list:
    """
    Build a unified list of all searchable datasets by merging:
      1. Hardcoded DEMO_DATASETS
      2. Persisted registry (Kaggle downloads)
      3. Auto-discovered CSVs not in either list
    """
    known_files = {ds["file"] for ds in DEMO_DATASETS}
    merged = list(DEMO_DATASETS)

    # Layer 2 — registry entries
    registry = _load_registry()
    for entry in registry:
        if entry["file"] not in known_files:
            merged.append(entry)
            known_files.add(entry["file"])

    # Layer 3 — auto-discover any remaining CSVs on disk
    try:
        all_csvs = [
            f for f in os.listdir(DATASET_FOLDER)
            if f.endswith(".csv") and not f.startswith("_")
        ]
    except FileNotFoundError:
        all_csvs = []

    for csv_file in all_csvs:
        if csv_file in known_files:
            continue

        file_path = os.path.join(DATASET_FOLDER, csv_file)
        keywords = _generate_keywords_from_file(file_path)

        merged.append({
            "file":     csv_file,
            "name":     _prettify_filename(csv_file),
            "category": "Uncategorized",
            "keywords": keywords,
            "source":   "auto-discovered",
        })
        known_files.add(csv_file)

    return merged


# Profiling & scoring (unchanged logic)

def get_dataset_profile(file_path: str) -> dict:
    """Read basic stats from a dataset file."""
    try:
        try:
            df = pd.read_csv(file_path, nrows=5)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, nrows=5, encoding="latin1")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            total_rows = sum(1 for _ in f) - 1

        return {
            "columns":          list(df.columns),
            "numerical_cols":   df.select_dtypes(include=["int64","float64"]).columns.tolist(),
            "categorical_cols": df.select_dtypes(include=["object"]).columns.tolist(),
            "total_rows":       total_rows,
            "total_columns":    len(df.columns),
        }
    except Exception:
        return None


def compute_match_score(problem: str, ds: dict) -> dict:
    """
    Smart keyword scoring — fast and accurate.
    Matches problem words against dataset keywords + columns.
    """
    problem_lower  = problem.lower()
    problem_words  = set(re.findall(r'\b\w+\b', problem_lower))

    # Score 1 — keyword matches
    keyword_matches = sum(
        1 for kw in ds.get("keywords", [])
        if kw in problem_lower or kw in problem_words
    )
    keyword_score = min(keyword_matches * 12, 60)

    # Score 2 — column name matches
    col_matches = 0
    matched_cols = []
    for col in ds.get("columns", []):
        col_words = set(re.findall(r'\b\w+\b', col.lower()))
        overlap   = problem_words & col_words
        if overlap:
            col_matches += len(overlap)
            matched_cols.append(col)
    column_score = min(col_matches * 8, 30)

    # Score 3 — category match bonus
    category_words = set(ds.get("category", "").lower().split())
    category_score = 10 if problem_words & category_words else 0

    total_score = min(keyword_score + column_score + category_score, 100)

    # Generate reason
    if total_score >= 70:
        reason = f"Strong match — dataset contains {keyword_matches} relevant keywords aligned with your problem."
    elif total_score >= 50:
        reason = f"Good match — {keyword_matches} keyword overlaps found with your problem domain."
    elif total_score >= 30:
        reason = f"Partial match — some relevant columns and keywords detected."
    else:
        reason = f"Low relevance — limited overlap with your problem statement."

    return {
        "score":       total_score,
        "reason":      reason,
        "key_columns": matched_cols[:4]
    }


# Main search function
def find_matching_datasets(problem_statement: str, top_n: int = 5) -> list:
    """
    Scores ALL datasets (demo + registry + auto-discovered) using smart
    keyword matching. Returns top N ranked by score.
    """
    all_datasets = _get_all_datasets()
    results = []

    for ds in all_datasets:
        file_path = os.path.join(DATASET_FOLDER, ds["file"])
        if not os.path.exists(file_path):
            continue

        profile = get_dataset_profile(file_path)
        if not profile:
            continue

        # Merge columns into ds for scoring
        ds_with_cols = {**ds, **profile}
        score_result = compute_match_score(problem_statement, ds_with_cols)

        results.append({
            "file":            ds["file"],
            "name":            ds["name"],
            "category":        ds["category"],
            "score":           score_result["score"],
            "reason":          score_result["reason"],
            "key_columns":     score_result["key_columns"],
            "total_rows":      profile["total_rows"],
            "total_columns":   profile["total_columns"],
            "columns":         profile["columns"],
            "numerical_cols":  profile["numerical_cols"],
            "categorical_cols":profile["categorical_cols"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    top = [r for r in results if r["score"] >= 20]
    return top[:top_n]  # returns [] when no dataset meets the threshold — let the API handle it