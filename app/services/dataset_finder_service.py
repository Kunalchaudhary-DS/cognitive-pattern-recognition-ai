"""
Dataset Finder Service — Smart keyword-based matching engine.
Fast, accurate, no AI needed for scoring.
Phi-3 only used for generating the reason text after matching.
"""

import os
import re
import pandas as pd
from app.core.config import DATASET_FOLDER


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
        1 for kw in ds["keywords"]
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
    category_words = set(ds["category"].lower().split())
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


def find_matching_datasets(problem_statement: str, top_n: int = 5) -> list:
    """
    Main function — instantly scores all datasets using smart
    keyword matching. Returns top N ranked by score.
    """
    results = []

    for ds in DEMO_DATASETS:
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
    return top[:top_n] if top else results[:3]