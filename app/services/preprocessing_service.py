"""
Preprocessing service — imputation, encoding, scaling.
All sklearn preprocessing logic lives here.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from app.core.config import CLASSIFICATION_UNIQUE_THRESHOLD


def detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
    """Returns 'classification' or 'regression' based on the target column."""
    target_dtype = df[target_column].dtype
    unique_values = df[target_column].nunique()
    if target_dtype == "object" or unique_values <= CLASSIFICATION_UNIQUE_THRESHOLD:
        return "classification"
    return "regression"


def compute_feature_importance(df: pd.DataFrame, target_column: str) -> dict:
    """
    Correlation-based feature importance.
    Used by the /feature-importance/ endpoint.
    """
    problem_type = detect_problem_type(df, target_column)

    if problem_type == "classification":
        le = LabelEncoder()
        df = df.copy()
        df[target_column] = le.fit_transform(df[target_column])

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target_column not in numerical_cols:
        return {"error": "Target encoding failed"}

    correlations = df[numerical_cols].corr()[target_column].drop(target_column)
    ranked = correlations.abs().sort_values(ascending=False)

    result = [
        {
            "feature": feature,
            "correlation": round(float(correlations[feature]), 3)
        }
        for feature in ranked.index
    ]
    return {"feature_importance": result, "problem_type": problem_type}


def run_preprocessing(df: pd.DataFrame, target_column: str) -> dict:
    """
    Full preprocessing pipeline:
      1. Drop rows where target is missing
      2. Impute numerical → median, categorical → most_frequent
      3. Encode: binary → LabelEncoder, high cardinality → frequency, low cardinality → OneHot
      4. Return processed X, y, feature names, preprocessor, problem_type
    """
    problem_type = detect_problem_type(df, target_column)
    original_shape = df.shape

    df = df.dropna(subset=[target_column])
    dropped_rows = original_shape[0] - df.shape[0]

    # Impute
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if numerical_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Categorize categorical columns
    binary_columns, low_cardinality, high_cardinality = [], [], []
    for col in cat_features:
        unique_count = X[col].nunique()
        if unique_count == 2:
            binary_columns.append(col)
        elif unique_count <= 10:
            low_cardinality.append(col)
        else:
            high_cardinality.append(col)

    # Binary → Label Encoding
    encoding_maps = {}
    for col in binary_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoding_maps[col] = {
            str(cls): int(code)
            for code, cls in enumerate(le.classes_)
        }

    # High cardinality → Frequency Encoding
    for col in high_cardinality:
        freq = X[col].value_counts(normalize=True)
        X[col] = X[col].map(freq)
        encoding_maps[col] = {
            str(k): round(float(v), 4)
            for k, v in freq.items()
        }

    # Low cardinality → OneHot via ColumnTransformer
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), low_cardinality)
        ],
        remainder="drop"
    )

    X_processed = preprocessor.fit_transform(X)

    # Extract feature names
    feature_names = list(num_features)
    if low_cardinality:
        onehot_features = preprocessor.named_transformers_["onehot"] \
            .get_feature_names_out(low_cardinality)
        feature_names.extend(onehot_features.tolist())

    # Dense matrix
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    return {
        "X": X_processed,
        "y": y.values,
        "problem_type": problem_type,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "original_shape": original_shape,
        "processed_feature_shape": X_processed.shape,
        "target_shape": y.shape,
        "binary_encoded": binary_columns,
        "onehot_encoded": low_cardinality,
        "frequency_encoded": high_cardinality,
        "dropped_target_rows": dropped_rows,
        "encoding_maps": encoding_maps,
        "message": "Preprocessing completed successfully"
    }