"""
Tests for preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from app.services.preprocessing_service import detect_problem_type, run_preprocessing


def sample_classification_df():
    return pd.DataFrame({
        "age":    [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "income": [30000, 45000, 50000, 60000, 70000, 80000, 55000, 90000, 40000, 75000],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "target": ["yes", "no", "yes", "no", "yes", "yes", "no", "yes", "no", "yes"],
    })


def sample_regression_df():
    return pd.DataFrame({
        "area":     [500, 800, 1200, 1500, 2000, 2500, 3000, 3500, 4000, 4500],
        "bedrooms": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        "price":    [100000, 150000, 200000, 250000, 300000,
                     350000, 400000, 450000, 500000, 550000],
    })


# detect_problem_type

def test_detects_classification():
    df = sample_classification_df()
    assert detect_problem_type(df, "target") == "classification"


def test_detects_regression():
    df = sample_regression_df()
    assert detect_problem_type(df, "price") == "regression"


#run_preprocessing

def test_preprocessing_returns_arrays():
    df = sample_classification_df()
    result = run_preprocessing(df, "target")
    assert result["X"] is not None
    assert result["y"] is not None
    assert len(result["y"]) == len(df)


def test_preprocessing_message():
    df = sample_regression_df()
    result = run_preprocessing(df, "price")
    assert result["message"] == "Preprocessing completed successfully"


def test_preprocessing_problem_type():
    df = sample_classification_df()
    result = run_preprocessing(df, "target")
    assert result["problem_type"] == "classification"


def test_preprocessing_feature_names_exist():
    df = sample_regression_df()
    result = run_preprocessing(df, "price")
    assert isinstance(result["feature_names"], list)
    assert len(result["feature_names"]) > 0