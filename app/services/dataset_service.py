"""
Dataset service — all logic for loading, validating, and profiling datasets.
No FastAPI imports here. Pure Python + pandas.
"""

import os
import io
import pandas as pd
import numpy as np
from app.core.config import (
    MIN_ROWS, MIN_COLUMNS, CORRELATION_THRESHOLD,
    CLASSIFICATION_UNIQUE_THRESHOLD
)


#Validation
def validate_dataframe(df: pd.DataFrame) -> str | None:
    """
    Runs all dataset checks.
    Returns an error string if invalid, or None if the dataset is OK.
    """
    if df.empty:
        return "Uploaded dataset is empty"
    if df.shape[0] < MIN_ROWS:
        return f"Dataset must contain at least {MIN_ROWS} rows"
    if df.shape[1] < MIN_COLUMNS:
        return f"Dataset must contain at least {MIN_COLUMNS} columns"
    if df.dropna(how="all").shape[0] == 0:
        return "Dataset contains only missing values"
    return None


#Loading

def load_csv_bytes(contents: bytes) -> pd.DataFrame:
    """Load a CSV from raw bytes (file upload)."""
    return pd.read_csv(io.BytesIO(contents))


def load_csv_path(file_path: str) -> pd.DataFrame:
    """Load a CSV from disk, handles encoding issues."""
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin1")


#Profiling

def compute_strong_correlations(df: pd.DataFrame) -> list:
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    strong_correlations = []

    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                value = corr_matrix.iloc[i, j]
                if abs(value) > CORRELATION_THRESHOLD:
                    strong_correlations.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(value), 2)
                    })
    return strong_correlations


def suggest_problem_type(df: pd.DataFrame) -> str:
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    unique_counts = df.nunique()
    classification_candidates = [
        col for col in df.columns
        if unique_counts[col] <= CLASSIFICATION_UNIQUE_THRESHOLD and col in categorical_cols
    ]
    return "Classification" if classification_candidates else "Regression"


def build_upload_profile(df: pd.DataFrame) -> dict:
    """
    Full profile used by the /upload/ endpoint.
    Returns everything the frontend needs after a CSV upload.
    """
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    total_rows = len(df)
    total_columns = len(df.columns)

    # Missing values
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / total_rows * 100).round(2)
    total_missing = missing_counts.sum()
    total_cells = total_rows * total_columns
    missing_ratio = total_missing / total_cells
    missing_percent_dataset = round(missing_ratio * 100, 2)

    # Duplicates
    duplicate_count = int(df.duplicated().sum())

    # Dataset nature
    if len(numerical_cols) > len(categorical_cols):
        dataset_nature = "Mostly Numerical Dataset"
    elif len(categorical_cols) > len(numerical_cols):
        dataset_nature = "Mostly Categorical Dataset"
    else:
        dataset_nature = "Balanced Dataset"

    # Class imbalance
    class_imbalance = None
    for col in categorical_cols:
        if df[col].nunique() <= CLASSIFICATION_UNIQUE_THRESHOLD:
            distribution = df[col].value_counts(normalize=True)
            if distribution.max() > 0.8:
                class_imbalance = f"Column '{col}' shows strong class imbalance."
                break

    # Quality score
    quality_score = 100
    if missing_ratio > 0.1:
        quality_score -= 20
    if duplicate_count > 0:
        quality_score -= 10
    if class_imbalance:
        quality_score -= 10
    quality_score = max(0, quality_score)

    # Summary paragraph
    dataset_summary = (
        f"The dataset contains {total_rows} rows and {total_columns} columns. "
        f"It includes {len(numerical_cols)} numerical features and "
        f"{len(categorical_cols)} categorical features. {dataset_nature}. "
    )
    dataset_summary += "Missing values are present. " if total_missing > 0 else "No missing values detected. "
    if duplicate_count > 0:
        dataset_summary += f"There are {duplicate_count} duplicate rows. "
    if class_imbalance:
        dataset_summary += class_imbalance

    # Missing summary
    missing_summary = {
        col: {
            "count": int(missing_counts[col]),
            "percentage": float(missing_percentage[col])
        }
        for col in df.columns if missing_counts[col] > 0
    }

    # Clean preview and full data
    preview_df = df.head().copy().replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)
    full_df = df.replace([np.inf, -np.inf], np.nan)
    full_df = full_df.astype(object).where(pd.notnull(full_df), None)

    suggested_problem = suggest_problem_type(df)
    strong_correlations = compute_strong_correlations(df)

    return {
        "rows": total_rows,
        "total_columns": total_columns,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "missing_summary": missing_summary,
        "duplicate_count": duplicate_count,
        "quality_score": quality_score,
        "dataset_nature": dataset_nature,
        "class_imbalance": class_imbalance,
        "dataset_summary": dataset_summary,
        "basic_statistics": df.describe(include="all").fillna("").to_dict(),
        "strong_correlations": strong_correlations,
        "profile_summary": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_percent": missing_percent_dataset,
            "suggested_problem": suggested_problem,
            "quality_score": quality_score
        },
        "columns": list(df.columns),
        "preview": preview_df.to_dict(orient="records"),
        "full_data": full_df.to_dict(orient="records")
    }


def build_demo_profile(df: pd.DataFrame) -> dict:
    """
    Lighter profile used by the /load-demo-dataset/ endpoint.
    """
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df) * 100).round(2)
    total_missing = missing_counts.sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_ratio = total_missing / total_cells
    missing_percent_dataset = round(missing_ratio * 100, 2)
    quality_score = round(100 - missing_percent_dataset, 2)

    missing_summary = {
        col: {
            "count": int(missing_counts[col]),
            "percentage": float(missing_percentage[col])
        }
        for col in df.columns if missing_counts[col] > 0
    }

    preview_df = df.head().copy().replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)
    full_df = df.replace([np.inf, -np.inf], np.nan)
    full_df = full_df.astype(object).where(pd.notnull(full_df), None)

    strong_correlations = compute_strong_correlations(df)
    suggested_problem = suggest_problem_type(df)

    return {
        "rows": int(len(df)),
        "total_columns": len(df.columns),
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "missing_summary": missing_summary,
        "strong_correlations": strong_correlations,
        "columns": list(df.columns),
        "profile_summary": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_percent": missing_percent_dataset,
            "suggested_problem": suggested_problem,
            "quality_score": quality_score
        },
        "preview": preview_df.to_dict(orient="records"),
        "full_data": full_df.to_dict(orient="records")
    }