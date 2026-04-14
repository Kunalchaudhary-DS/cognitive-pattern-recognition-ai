"""
Tests for dataset loading and validation.
Run with: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
from app.services.dataset_service import validate_dataframe, build_upload_profile, compute_strong_correlations


def make_df(rows=10, cols=3):
    """Helper: creates a simple numeric DataFrame."""
    return pd.DataFrame(np.random.rand(rows, cols), columns=[f"col_{i}" for i in range(cols)])


# validate_dataframe

def test_valid_dataframe_passes():
    df = make_df(10, 3)
    assert validate_dataframe(df) is None


def test_empty_dataframe_fails():
    df = pd.DataFrame()
    assert validate_dataframe(df) == "Uploaded dataset is empty"


def test_too_few_rows_fails():
    df = make_df(rows=2, cols=3)
    error = validate_dataframe(df)
    assert error is not None and "rows" in error


def test_too_few_columns_fails():
    df = make_df(rows=10, cols=1)
    error = validate_dataframe(df)
    assert error is not None and "columns" in error


def test_all_missing_values_fails():
    df = pd.DataFrame([[None, None], [None, None], [None, None],
                        [None, None], [None, None]], columns=["a", "b"])
    error = validate_dataframe(df)
    assert error is not None


# compute_strong_correlations

def test_strong_correlations_detected():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 6, 8, 10],   
        "c": [5, 3, 1, 9, 2],
    })
    correlations = compute_strong_correlations(df)
    assert any(c["feature_1"] == "a" and c["feature_2"] == "b" for c in correlations)


def test_no_false_strong_correlations():
    np.random.seed(0)
    df = pd.DataFrame(np.random.rand(50, 3), columns=["x", "y", "z"])
    correlations = compute_strong_correlations(df)
    # Random data should have no strong correlations
    assert len(correlations) == 0


#build_upload_profile

def test_upload_profile_keys():
    df = make_df(20, 4)
    profile = build_upload_profile(df)
    required_keys = [
        "rows", "total_columns", "numerical_columns", "categorical_columns",
        "quality_score", "dataset_summary", "preview", "columns"
    ]
    for key in required_keys:
        assert key in profile, f"Missing key: {key}"


def test_upload_profile_row_count():
    df = make_df(25, 4)
    profile = build_upload_profile(df)
    assert profile["rows"] == 25