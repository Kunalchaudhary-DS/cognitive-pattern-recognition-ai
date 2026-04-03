"""
Insight service — AI-style conclusions, cognitive pattern score,
model performance insights, feature importance extraction.
"""

import numpy as np
import pandas as pd


def extract_feature_importance(
    best_model,
    feature_names: list,
    X_test=None,
    y_test=None,
) -> dict:
    """
    Extract feature importances from any sklearn model.

    Priority:
      1. feature_importances_  (tree-based models: RF, GB, ET, AdaBoost, HistGB)
      2. coef_                 (linear models: LR, Ridge, Lasso, ElasticNet, Logistic)
      3. permutation_importance (fallback for KNN, SVM, NaiveBayes — any model)
    """
    if best_model is None or not feature_names:
        return {}

    importances = None

    # Priority 1 — tree-based native importance
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_

    # Priority 2 — linear model coefficients
    elif hasattr(best_model, "coef_"):
        coef = np.abs(best_model.coef_)
        importances = np.mean(coef, axis=0) if coef.ndim > 1 else coef

    # Priority 3 — permutation importance (works on any model)
    if importances is None and X_test is not None and y_test is not None:
        try:
            from sklearn.inspection import permutation_importance as perm_imp
            result      = perm_imp(best_model, X_test, y_test,
                                   n_repeats=5, random_state=42, n_jobs=-1)
            importances = np.maximum(result.importances_mean, 0)   # clip negatives
        except Exception:
            return {}

    if importances is None:
        return {}

    importances = np.array(importances).flatten()
    n           = min(len(feature_names), len(importances))
    indices     = np.argsort(importances[:n])[::-1][:10]
    return {feature_names[i]: float(importances[i]) for i in indices}



def generate_model_insights(training_results: dict, problem_type: str) -> list:
    """Human-readable insight about model performance."""
    insights = []
    if not training_results:
        return insights

    best_model = training_results.get("BestModel")
    if not best_model:
        return insights

    if problem_type == "regression":
        best_score = training_results[best_model]["CV_R2_Mean"]
        if best_score < 0:
            insights.append("Model performance is poor. Features may not explain target well.")
        elif best_score < 0.5:
            insights.append("Model explains moderate variance in the target variable.")
        else:
            insights.append("Model shows strong predictive performance.")
    else:
        best_score = training_results[best_model]["CV_Accuracy_Mean"]
        if best_score < 0.6:
            insights.append("Classification accuracy is relatively low.")
        elif best_score < 0.8:
            insights.append("Model performs reasonably well.")
        else:
            insights.append("Model achieves strong classification accuracy.")

    return insights


def generate_ai_dataset_conclusion(
    target_column: str,
    problem_type: str,
    feature_importance: dict,
    cluster_patterns: list,
    interaction_patterns: list
) -> str:
    explanation = []

    if feature_importance:
        top_features = list(feature_importance.keys())[:3]
        explanation.append(
            f"The analysis indicates that the target variable '{target_column}' "
            f"is strongly influenced by {', '.join(top_features)}."
        )

    if cluster_patterns:
        explanation.append(
            "Cluster analysis reveals distinct groups in the dataset, "
            "indicating that different feature combinations produce different outcomes."
        )

    if interaction_patterns:
        explanation.append(
            "Feature interaction discovery shows that combinations of multiple "
            "variables significantly influence the target outcome."
        )

    if feature_importance:
        if problem_type == "classification":
            explanation.append(
                "Improving these influential variables may increase the probability "
                "of achieving a positive class prediction."
            )
        else:
            explanation.append(
                "Improving these influential variables may increase the expected "
                "value of the target variable."
            )

    return " ".join(explanation)


def compute_cognitive_pattern_score(
    df: pd.DataFrame,
    feature_importance: dict,
    patterns: list,
    cluster_patterns: list,
    interaction_patterns: list
) -> dict:
    score = 0
    missing_ratio  = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    quality_score  = int((1 - missing_ratio) * 40)
    score         += quality_score
    score         += min(len(patterns) * 5, 20)
    score         += min(len(cluster_patterns) * 5, 20)
    score         += min(len(interaction_patterns) * 5, 20)
    total_score    = min(score, 100)
    level          = "Strong" if total_score > 80 else ("Moderate" if total_score > 60 else "Weak")

    return {
        "score":            total_score,
        "pattern_strength": level,
        "data_quality":     quality_score
    }


def build_prediction_analysis(
    target_column: str,
    problem_type: str,
    feature_importance: dict
) -> str:
    if not feature_importance:
        return ""

    top_features = list(feature_importance.keys())[:3]

    if problem_type == "regression":
        return (
            f"The trained model indicates that the target variable "
            f"'{target_column}' is strongly influenced by "
            f"{', '.join(top_features)}. "
            "Variations in these features are expected to produce the "
            "largest changes in the predicted outcome. "
            "Features with lower importance values contribute less to "
            "the model's predictive behaviour."
        )
    else:
        return (
            f"The classification model suggests that the target class "
            f"'{target_column}' is most affected by "
            f"{', '.join(top_features)}. "
            "Changes in these variables significantly influence "
            "the probability of each class prediction."
        )