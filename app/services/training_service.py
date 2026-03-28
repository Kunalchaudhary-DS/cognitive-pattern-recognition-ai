"""
Training service — AutoML model pool, cross-validation, best model selection.
All sklearn model logic lives here.
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from app.core.config import RANDOM_STATE, TEST_SIZE, CV_SPLITS


def get_model_pool(problem_type: str) -> dict:
    """Returns the full model pool for regression or classification."""
    if problem_type == "regression":
        return {
            "LinearRegression": {"model": LinearRegression(), "scale": True},
            "Ridge":            {"model": Ridge(), "scale": True},
            "Lasso":            {"model": Lasso(), "scale": True},
            "ElasticNet":       {"model": ElasticNet(), "scale": True},
            "RandomForest":     {"model": RandomForestRegressor(random_state=RANDOM_STATE), "scale": False},
            "GradientBoosting": {"model": GradientBoostingRegressor(), "scale": False},
            "ExtraTrees":       {"model": ExtraTreesRegressor(), "scale": False},
            "KNN":              {"model": KNeighborsRegressor(), "scale": True},
            "SVR":              {"model": SVR(), "scale": True},
        }
    else:
        return {
            "LogisticRegression": {"model": LogisticRegression(max_iter=1000), "scale": True},
            "RandomForest":       {"model": RandomForestClassifier(random_state=RANDOM_STATE), "scale": False},
            "GradientBoosting":   {"model": GradientBoostingClassifier(), "scale": False},
            "ExtraTrees":         {"model": ExtraTreesClassifier(), "scale": False},
            "KNN":                {"model": KNeighborsClassifier(), "scale": True},
            "SVC":                {"model": SVC(probability=True), "scale": True},
            "DecisionTree":       {"model": DecisionTreeClassifier(), "scale": False},
            "NaiveBayes":         {"model": GaussianNB(), "scale": False},
        }


def run_training(X: np.ndarray, y: np.ndarray, problem_type: str) -> dict:
    """
    Trains all models in the pool using cross-validation + hold-out test set.
    Returns results dict, best model name, fitted best model, scaler, needs_scaling flag.
    """

    # Train/test split + CV strategy
    if problem_type == "regression":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        cv_strategy = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scoring_metric = "r2"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        cv_strategy = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scoring_metric = "accuracy"

    # Scale once for scale-sensitive models
    scaler = RobustScaler()
    X_scaled       = scaler.fit_transform(X)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = get_model_pool(problem_type)
    results = {}

    for name, config in models.items():
        model = config["model"]
        needs_scaling = config["scale"]

        # Cross-validation score
        X_cv = X_scaled if needs_scaling else X
        cv_scores = cross_val_score(model, X_cv, y, cv=cv_strategy, scoring=scoring_metric)
        cv_mean = cv_scores.mean()

        # Hold-out evaluation
        if needs_scaling:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        if problem_type == "regression":
            test_score = r2_score(y_test, y_pred)
            results[name] = {
                "CV_R2_Mean":  round(float(cv_mean), 4),
                "Test_R2":     round(float(test_score), 4)
            }
        else:
            test_score = accuracy_score(y_test, y_pred)
            results[name] = {
                "CV_Accuracy_Mean": round(float(cv_mean), 4),
                "Test_Accuracy":    round(float(test_score), 4)
            }

    # Pick best model
    if problem_type == "regression":
        best_model_name = max(results, key=lambda x: results[x]["CV_R2_Mean"])
    else:
        best_model_name = max(results, key=lambda x: results[x]["CV_Accuracy_Mean"])

    best_config = models[best_model_name]
    final_model = best_config["model"]
    needs_scaling = best_config["scale"]

    # Refit best model on full data
    if needs_scaling:
        final_scaler = RobustScaler()
        X_final = final_scaler.fit_transform(X)
        final_model.fit(X_final, y)
    else:
        final_scaler = None
        final_model.fit(X, y)

    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_model": final_model,
        "scaler": final_scaler,
        "needs_scaling": needs_scaling,
    }