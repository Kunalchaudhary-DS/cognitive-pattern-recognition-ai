"""
Preprocessing service — imputation, encoding, scaling.

5 encoding strategies:
  0. ID-column detection + drop         (unique ratio > 90%)
  1. Ordinal encoding                   (natural-order text like low/med/high)
  2. Binary encoding (LabelEncoder)     (exactly 2 unique values)
  3. OneHot encoding                    (3 – ONEHOT_CARDINALITY_LIMIT unique)
  4. Frequency encoding                 (> ONEHOT_CARDINALITY_LIMIT unique)

Problem-type detection uses 3 signals from the target column alone.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from app.core.config import ONEHOT_CARDINALITY_LIMIT


# ── Known ordinal patterns (values listed lowest → highest) ───────────────────
ORDINAL_PATTERNS = [
    ["low", "medium", "high"],
    ["low", "med", "high"],
    ["low", "medium", "high", "very high"],
    ["none", "low", "medium", "high"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["never", "sometimes", "always"],
    ["poor", "average", "good", "excellent"],
    ["very poor", "poor", "fair", "good", "very good", "excellent"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["disagree", "neutral", "agree"],
    ["no", "maybe", "yes"],
    ["beginner", "intermediate", "advanced", "expert"],
    ["entry", "junior", "mid", "senior", "lead"],
    ["small", "medium", "large"],
    ["xs", "s", "m", "l", "xl"],
    ["xs", "s", "m", "l", "xl", "xxl"],
    ["bronze", "silver", "gold", "platinum"],
    ["q1", "q2", "q3", "q4"],
]


# ── Problem-type detection ─────────────────────────────────────────────────────

def detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
    """
    Multi-signal problem type detection from target column only.
    Uses 3 independent signals; any one is enough to classify as classification.
    Returns 'classification' or 'regression'.
    """
    col      = df[target_column].dropna()
    dtype    = col.dtype
    n_unique = col.nunique()
    n_rows   = max(len(col), 1)

    # Signal 1 — string / bool / categorical dtype
    if dtype == object or str(dtype) in ("bool", "boolean", "category"):
        return "classification"

    # Signal 2 — very few unique numeric values (ratings, labels, flags)
    if n_unique <= 15:
        return "classification"

    # Signal 3 — numeric but low unique-ratio  (e.g. 500 rows, 12 unique → 2.4%)
    unique_ratio = n_unique / n_rows
    if unique_ratio < 0.05 and n_unique <= 30:
        return "classification"

    return "regression"


# ── Ordinal detection helper ───────────────────────────────────────────────────

def _detect_ordinal(col: pd.Series):
    """
    Returns an ordered list of values if the column matches a known ordinal
    pattern, otherwise returns None.
    """
    col_vals = set(col.astype(str).str.lower().dropna().unique())
    for pattern in ORDINAL_PATTERNS:
        if col_vals and col_vals.issubset(set(pattern)):
            # Return only values that actually appear, in correct order
            return [v for v in pattern if v in col_vals]
    return None


# ── Pre-training correlation importance (no model needed) ──────────────────────

def compute_feature_importance(df: pd.DataFrame, target_column: str) -> dict:
    """
    Correlation-based feature importance — used by /feature-importance/ endpoint.
    """
    problem_type = detect_problem_type(df, target_column)
    df = df.copy()

    if problem_type == "classification":
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column].astype(str))

    numerical_cols = df.select_dtypes(include="number").columns.tolist()

    if target_column not in numerical_cols:
        return {"error": "Target encoding failed"}

    correlations = df[numerical_cols].corr()[target_column].drop(target_column)
    ranked = correlations.abs().sort_values(ascending=False)

    result = [
        {"feature": f, "correlation": round(float(correlations[f]), 3)}
        for f in ranked.index
    ]
    return {"feature_importance": result, "problem_type": problem_type}


# ── Main preprocessing pipeline ───────────────────────────────────────────────

def run_preprocessing(df: pd.DataFrame, target_column: str) -> dict:
    """
    Full pipeline:
      1. Drop rows with missing target
      2. Impute: numerical → median, categorical → most_frequent
      3. Drop likely ID columns (unique ratio > 90 %, > 20 unique values)
      4. Ordinal columns → integer rank encoding
      5. Binary columns → LabelEncoder
      6. Low-cardinality (3 – ONEHOT_CARDINALITY_LIMIT) → OneHotEncoder
      7. Higher-cardinality → frequency encoding
    """
    problem_type   = detect_problem_type(df, target_column)
    original_shape = df.shape

    df = df.copy()
    df = df.dropna(subset=[target_column])
    dropped_rows = original_shape[0] - df.shape[0]

    # ── Impute ─────────────────────────────────────────────────────────────────
    numerical_cols   = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if numerical_cols:
        df[numerical_cols] = SimpleImputer(strategy="median").fit_transform(df[numerical_cols])
    if categorical_cols:
        df[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_cols])

    # ── Separate features / target ─────────────────────────────────────────────
    X            = df.drop(columns=[target_column]).copy()
    y            = df[target_column].copy()
    cat_features = X.select_dtypes(include="object").columns.tolist()
    encoding_maps = {}
    id_dropped    = []

    # ── Step 0: Drop ID-like columns ──────────────────────────────────────────
    for col in cat_features:
        n_unique      = X[col].nunique()
        unique_ratio  = n_unique / max(len(X), 1)
        if unique_ratio > 0.9 and n_unique > 20:
            id_dropped.append(col)
    if id_dropped:
        X = X.drop(columns=id_dropped)
    cat_features = [c for c in cat_features if c not in id_dropped]

    # ── Step 1: Ordinal encoding ───────────────────────────────────────────────
    ordinal_encoded = []
    for col in list(cat_features):
        pattern = _detect_ordinal(X[col])
        if pattern:
            mapping = {v: float(i) for i, v in enumerate(pattern)}
            X[col]  = X[col].astype(str).str.lower().map(mapping)
            encoding_maps[col] = {v: i for i, v in enumerate(pattern)}
            ordinal_encoded.append(col)
    cat_features = [c for c in cat_features if c not in ordinal_encoded]

    # ── Categorise remaining categorical columns ───────────────────────────────
    binary_columns  = []
    low_cardinality = []   # 3 – ONEHOT_CARDINALITY_LIMIT  → OneHot
    high_cardinality = []  # > ONEHOT_CARDINALITY_LIMIT    → Frequency

    for col in cat_features:
        n = X[col].nunique()
        if n == 2:
            binary_columns.append(col)
        elif n <= ONEHOT_CARDINALITY_LIMIT:
            low_cardinality.append(col)
        else:
            high_cardinality.append(col)

    # ── Step 2: Binary → LabelEncoder ─────────────────────────────────────────
    for col in binary_columns:
        le     = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoding_maps[col] = {str(cls): int(code) for code, cls in enumerate(le.classes_)}

    # ── Step 3: High-cardinality → Frequency encoding ─────────────────────────
    for col in high_cardinality:
        freq   = X[col].value_counts(normalize=True)
        X[col] = X[col].map(freq)
        encoding_maps[col] = {str(k): round(float(v), 4) for k, v in freq.items()}

    # ── Step 4: Low-cardinality → OneHot via ColumnTransformer ────────────────
    num_features = X.select_dtypes(include="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",    "passthrough",                                               num_features),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"),        low_cardinality),
        ],
        remainder="drop",
    )

    X_processed = preprocessor.fit_transform(X)

    # ── Extract final feature names ────────────────────────────────────────────
    feature_names = list(num_features)
    if low_cardinality:
        onehot_names = preprocessor.named_transformers_["onehot"].get_feature_names_out(low_cardinality)
        feature_names.extend(onehot_names.tolist())

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    return {
        "X":                       X_processed,
        "y":                       y.values,
        "problem_type":            problem_type,
        "preprocessor":            preprocessor,
        "feature_names":           feature_names,
        "original_shape":          original_shape,
        "processed_feature_shape": X_processed.shape,
        "target_shape":            y.shape,
        "id_dropped":              id_dropped,
        "ordinal_encoded":         ordinal_encoded,
        "binary_encoded":          binary_columns,
        "onehot_encoded":          low_cardinality,
        "frequency_encoded":       high_cardinality,
        "dropped_target_rows":     dropped_rows,
        "encoding_maps":           encoding_maps,
        "message":                 "Preprocessing completed successfully",
    }