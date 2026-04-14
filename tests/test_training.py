"""
Tests for the training service.
"""

import numpy as np
import pytest
from app.services.training_service import run_training, get_model_pool


# get_model_pool

def test_regression_pool_has_expected_models():
    pool = get_model_pool("regression")
    assert "LinearRegression" in pool
    assert "RandomForest" in pool
    assert "SVR" in pool


def test_classification_pool_has_expected_models():
    pool = get_model_pool("classification")
    assert "LogisticRegression" in pool
    assert "RandomForest" in pool
    assert "NaiveBayes" in pool


#run_training

def make_regression_data():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.rand(100) * 0.1
    return X, y


def make_classification_data():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y


def test_regression_training_returns_best_model():
    X, y = make_regression_data()
    result = run_training(X, y, "regression")
    assert result["best_model"] is not None
    assert result["best_model_name"] in get_model_pool("regression")


def test_classification_training_returns_best_model():
    X, y = make_classification_data()
    result = run_training(X, y, "classification")
    assert result["best_model"] is not None
    assert result["best_model_name"] in get_model_pool("classification")


def test_training_results_have_correct_keys():
    X, y = make_regression_data()
    result = run_training(X, y, "regression")
    for model_name, scores in result["results"].items():
        assert "CV_R2_Mean" in scores
        assert "Test_R2" in scores


def test_training_scaler_set_when_needed():
    X, y = make_classification_data()
    result = run_training(X, y, "classification")
    if result["needs_scaling"]:
        assert result["scaler"] is not None
    else:
        assert result["scaler"] is None