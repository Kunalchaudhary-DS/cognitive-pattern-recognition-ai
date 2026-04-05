"""
Analysis routes — dashboard data, full analysis after training.
"""

import pandas as pd
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.core.state import state
from app.services.analysis_service import (
    build_auto_graphs,
    discover_patterns,
    discover_clusters,
    discover_feature_interactions,
    generate_pattern_visualizations,
)
from app.services.insight_service import (
    extract_feature_importance,
    generate_model_insights,
    generate_ai_dataset_conclusion,
    compute_cognitive_pattern_score,
    build_prediction_analysis,
)

from app.services.ai_service import (
    generate_dataset_explanation,
    generate_training_explanation,
    generate_pattern_explanation,
    generate_insight_summary,
    generate_panel_insights,
    ask_phi3,
)

router = APIRouter()


@router.get("/dashboard-data/")
async def dashboard_data():
    if state.df is None:
        return JSONResponse(content={"error": "No dataset uploaded"})
    if state.target_column is None:
        return JSONResponse(content={"error": "Run preprocessing first"})

    df = state.df.copy()
    numerical_cols   = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    missing_total    = df.isnull().sum().sum()

    # Dataset summary paragraph
    dataset_summary = (
        f"Dataset contains {len(df)} rows and {len(df.columns)} columns. "
        f"There are {len(numerical_cols)} numerical features and "
        f"{len(categorical_cols)} categorical features. "
    )
    dataset_summary += "Some missing values are present." if missing_total > 0 else "No missing values detected."

    # Feature importance from best model
    feature_importance = extract_feature_importance(state.best_model, state.feature_names, state.X_test, state.y_test)

    # Model comparison table
    model_comparison = {}
    if state.training_results:
        metric_key = "CV_R2_Mean" if state.problem_type == "regression" else "CV_Accuracy_Mean"
        for model_name, metrics in state.training_results.items():
            if model_name in ["BestModel", "ProblemType", "ConfusionMatrix"]:
                continue
            if not isinstance(metrics, dict):
                continue
            metric_value = metrics.get(metric_key)
            if metric_value is not None:
                model_comparison[model_name] = metric_value

    # Pattern / cluster / interaction analysis
    patterns             = discover_patterns(df, state.target_column)
    cluster_patterns     = discover_clusters(df)
    interaction_patterns = discover_feature_interactions(df, state.target_column)

    # ── Key Insights — the most important findings from this dataset ───
    insights = []

    # 1. Top influential feature
    if feature_importance:
        top_features = list(feature_importance.keys())[:3]
        others = ", ".join("'" + f + "'" for f in top_features[1:])
        insights.append(
            f"'{top_features[0]}' is the strongest predictor of '{state.target_column}'"
            + (f", followed by {others}." if others else ".")
        )

    # 2. Strongest correlation pattern
    if state.strong_correlations:
        sc = state.strong_correlations[0]
        direction = "positive" if sc["correlation"] > 0 else "negative"
        insights.append(
            f"Strong {direction} correlation ({sc['correlation']}) between "
            f"'{sc['feature_1']}' and '{sc['feature_2']}' — these features move together."
        )

    # 3. Data quality signal
    missing_pct = round(missing_total / (df.shape[0] * df.shape[1]) * 100, 1) if (df.shape[0] * df.shape[1]) > 0 else 0
    dup_count = int(df.duplicated().sum())
    if missing_pct > 5:
        insights.append(f"{missing_pct}% of data is missing — this may affect model reliability.")
    elif missing_pct == 0 and dup_count == 0:
        insights.append("Dataset has zero missing values and no duplicates — excellent data quality.")

    # 4. Cluster discovery highlight
    if cluster_patterns:
        insights.append(
            f"{len(cluster_patterns)} distinct data clusters found — "
            f"the dataset contains naturally separable groups."
        )

    # 5. Model performance verdict
    model_insights = generate_model_insights(state.training_results, state.problem_type)
    if model_insights:
        insights.append(model_insights[0])

    # 6. Class imbalance / target skew warning
    if state.target_column in categorical_cols:
        dist = df[state.target_column].value_counts(normalize=True)
        if dist.max() > 0.75:
            insights.append(
                f"Target variable '{state.target_column}' is imbalanced — "
                f"'{dist.idxmax()}' dominates at {round(dist.max()*100, 1)}%."
            )
    elif state.target_column in numerical_cols:
        skew = df[state.target_column].skew()
        if abs(skew) > 1.5:
            direction = "right" if skew > 0 else "left"
            insights.append(
                f"Target '{state.target_column}' is heavily {direction}-skewed (skew={skew:.2f}) — "
                f"consider log-transform for better model performance."
            )

    # 7. Outlier warning from patterns
    outlier_patterns = [p for p in patterns if "outlier" in p.lower()]
    if outlier_patterns:
        insights.append(outlier_patterns[0])

    # Fallback — never leave empty
    if not insights:
        insights.append(
            f"Dataset contains {len(df)} rows across {len(df.columns)} features — "
            f"{len(numerical_cols)} numerical, {len(categorical_cols)} categorical."
        )

    # Auto graphs — pass feature_importance so smart selection can prioritise them
    auto_graphs = build_auto_graphs(
        df, state.target_column, state.strong_correlations,
        feature_importance=feature_importance,
    )

    # Target distribution
    target_distribution = df[state.target_column].value_counts().to_dict()

    # Correlation matrix
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr().fillna(0)
        correlation_values = correlation_matrix.values.tolist()
    else:
        correlation_values = []

    # AI conclusions
    ai_conclusion = generate_ai_dataset_conclusion(
        state.target_column, state.problem_type,
        feature_importance, cluster_patterns, interaction_patterns
    )
    prediction_analysis = build_prediction_analysis(
        state.target_column, state.problem_type, feature_importance
    )
    pattern_score = compute_cognitive_pattern_score(
        df, feature_importance, patterns, cluster_patterns, interaction_patterns
    )
    pattern_visualizations = generate_pattern_visualizations(
        df, cluster_patterns, interaction_patterns, numerical_cols
    )

    # Cap at 1000 rows for dashboard display — no need to send 50k rows to the UI
    display_df = df.sample(min(len(df), 1000), random_state=42) if len(df) > 1000 else df
    full_data = (
        display_df.replace([np.inf, -np.inf], np.nan)
          .astype(object)
          .where(pd.notnull(display_df), None)
          .to_dict(orient="records")
    )

    return JSONResponse(content={
        "dataset_summary":      dataset_summary,
        "target_distribution":  target_distribution,
        "correlation_matrix":   correlation_values,
        "correlation_labels":   numerical_cols,
        "feature_importance":   feature_importance,
        "model_comparison":     model_comparison,
        "insights":             insights,
        "patterns":             patterns,
        "clusters":             cluster_patterns,
        "auto_graphs":          auto_graphs,
        "feature_interactions": interaction_patterns,
        "ai_conclusion":        ai_conclusion,
        "prediction_analysis":  prediction_analysis,
        "pattern_score":        pattern_score,
        "pattern_visualizations": pattern_visualizations,
        "full_data":            full_data,
        "best_model_name": state.training_results.get("BestModel", ""),
    })

@router.post("/ai/dataset-explanation/")
async def ai_dataset_explanation():
    if state.df is None:
        return JSONResponse(content={"error": "No dataset uploaded"})

    df = state.df
    numerical_cols   = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    missing_percent  = round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2)

    explanation = generate_dataset_explanation(
        rows              = len(df),
        columns           = len(df.columns),
        numerical_count   = len(numerical_cols),
        categorical_count = len(categorical_cols),
        missing_percent   = missing_percent,
        quality_score     = round(100 - missing_percent, 2),
        suggested_problem = state.problem_type or "Unknown"
    )
    return JSONResponse(content={"explanation": explanation})


@router.post("/ai/training-explanation/")
async def ai_training_explanation():
    if state.training_results is None:
        return JSONResponse(content={"error": "Run training first"})

    from app.services.insight_service import extract_feature_importance
    feature_importance = extract_feature_importance(state.best_model, state.feature_names, state.X_test, state.y_test)
    top_features       = list(feature_importance.keys())[:3]
    best_model_name    = state.training_results.get("BestModel", "Unknown")

    if state.problem_type == "regression":
        best_score = state.training_results.get(best_model_name, {}).get("CV_R2_Mean", 0)
    else:
        best_score = state.training_results.get(best_model_name, {}).get("CV_Accuracy_Mean", 0)

    explanation = generate_training_explanation(
        best_model     = best_model_name,
        problem_type   = state.problem_type,
        best_score     = best_score,
        model_results  = state.training_results,
        top_features   = top_features
    )
    return JSONResponse(content={"explanation": explanation})


@router.post("/ai/pattern-explanation/")
async def ai_pattern_explanation():
    if state.df is None or state.target_column is None:
        return JSONResponse(content={"error": "Run preprocessing first"})

    from app.services.analysis_service import discover_patterns, discover_clusters
    from app.services.insight_service  import compute_cognitive_pattern_score, extract_feature_importance

    df                 = state.df.copy()
    patterns           = discover_patterns(df, state.target_column)
    clusters           = discover_clusters(df)
    feature_importance = extract_feature_importance(state.best_model, state.feature_names, state.X_test, state.y_test)
    pattern_score      = compute_cognitive_pattern_score(df, feature_importance, patterns, clusters, [])

    explanation = generate_pattern_explanation(
        pattern_score   = pattern_score["score"],
        pattern_strength= pattern_score["pattern_strength"],
        patterns        = patterns,
        clusters        = clusters,
        target_column   = state.target_column,
        problem_type    = state.problem_type
    )
    return JSONResponse(content={"explanation": explanation})


@router.get("/ai/insight-summary/")
async def ai_insight_summary():
    if state.training_results is None:
        return JSONResponse(content={"error": "Run training first"})

    from app.services.insight_service  import extract_feature_importance, compute_cognitive_pattern_score
    from app.services.analysis_service import discover_patterns, discover_clusters

    df                 = state.df.copy()
    feature_importance = extract_feature_importance(state.best_model, state.feature_names, state.X_test, state.y_test)
    top_feature        = list(feature_importance.keys())[0] if feature_importance else "N/A"
    patterns           = discover_patterns(df, state.target_column)
    clusters           = discover_clusters(df)
    pattern_score      = compute_cognitive_pattern_score(df, feature_importance, patterns, clusters, [])
    best_model_name    = state.training_results.get("BestModel", "Unknown")

    if state.problem_type == "regression":
        best_score = state.training_results.get(best_model_name, {}).get("CV_R2_Mean", 0)
    else:
        best_score = state.training_results.get(best_model_name, {}).get("CV_Accuracy_Mean", 0)

    summary = generate_insight_summary(
        target_column = state.target_column,
        problem_type  = state.problem_type,
        best_model    = best_model_name,
        best_score    = best_score,
        pattern_score = pattern_score["score"],
        top_feature   = top_feature
    )
    return JSONResponse(content={"summary": summary})


@router.post("/ai/panel-insights/")
async def ai_panel_insights():
    """
    Generates short AI insights using the actual content of the
    'Discovered Patterns' and 'Cluster Analysis' dashboard panels as context.
    """
    if state.df is None or state.target_column is None:
        return JSONResponse(content={"error": "Run preprocessing first"})

    from app.services.analysis_service import discover_patterns, discover_clusters

    df       = state.df.copy()
    patterns = discover_patterns(df, state.target_column)
    clusters = discover_clusters(df)

    insights = generate_panel_insights(
        patterns      = patterns,
        clusters      = clusters,
        target_column = state.target_column,
        problem_type  = state.problem_type or "regression",
    )

    return JSONResponse(content={
        "patterns_insight": insights["patterns_insight"],
        "clusters_insight": insights["clusters_insight"],
    })


@router.post("/predict/")

async def predict(request: Request):
    """
    Takes feature values from user and returns model prediction.
    Applies the Semantic Prediction Interceptor to enforce logical constraints.
    """
    if state.best_model is None:
        return JSONResponse(content={"error": "Train a model first"})

    body = await request.json()
    input_values = body.get("values", {})

    if not input_values:
        return JSONResponse(content={"error": "No input values provided"})

    try:
        # Build feature array in correct order
        feature_names = state.feature_names or []
        input_array   = []

        encoding_maps = state.encoding_maps or {}

        for fname in feature_names:
            val = input_values.get(fname, 0)

            # Check if this feature has an encoding map
            if fname in encoding_maps and isinstance(val, str):
                enc_map = encoding_maps[fname]
                # Try exact match first
                if val in enc_map:
                    input_array.append(float(enc_map[val]))
                # Try case-insensitive match
                else:
                    val_lower = val.lower()
                    matched = next(
                        (enc_map[k] for k in enc_map if k.lower() == val_lower),
                        None
                    )
                    if matched is not None:
                        input_array.append(float(matched))
                    else:
                        input_array.append(0.0)
            else:
                try:
                    input_array.append(float(val))
                except (ValueError, TypeError):
                    input_array.append(0.0)

        X_input = np.array([input_array])

        # Scale if needed
        if state.needs_scaling and state.scaler:
            X_input = state.scaler.transform(X_input)

        # ── Raw model prediction ──────────────────────────────────────────────
        raw_prediction = state.best_model.predict(X_input)[0]

        # ── Semantic Prediction Interceptor (regression only) ─────────────────
        constraints_applied = []
        soft_warnings       = []
        raw_value_display   = None
        was_corrected       = False

        if state.problem_type == "regression":
            from app.services.constraint_service import apply_constraints
            interceptor_result = apply_constraints(
                raw_prediction  = float(raw_prediction),
                input_values    = input_values,
                constraint_map  = state.constraint_map,
                feature_names   = feature_names,
                target_column   = state.target_column or "",
            )
            final_prediction    = interceptor_result["final_value"]
            constraints_applied = interceptor_result["constraints_applied"]
            soft_warnings       = interceptor_result["soft_warnings"]
            was_corrected       = interceptor_result["was_corrected"]
            raw_value_display   = round(interceptor_result["raw_value"], 4) if was_corrected else None
            result              = round(float(final_prediction), 4)
        else:
            result = str(raw_prediction)

        # ── AI explanation ────────────────────────────────────────────────────
        top_features = feature_names[:3] if feature_names else []
        top_values   = [input_values.get(f, 0) for f in top_features]

        correction_note = ""
        if was_corrected:
            correction_note = (
                f"\nNote: The raw model output was {raw_value_display}, "
                f"but it was corrected to {result} due to domain logic constraints."
            )

        prompt = f"""You are an AI prediction system. Explain this prediction in 2 complete sentences.

Target variable: {state.target_column}
Problem type: {state.problem_type}
Prediction result: {result}{correction_note}
Top input features: {dict(zip(top_features, top_values))}

Write 2 complete sentences explaining what this prediction means in real-world terms.
Do not cut off mid-sentence."""

        explanation = ask_phi3(prompt)

        return JSONResponse(content={
            "prediction":           result,
            "raw_prediction":       raw_value_display,
            "was_corrected":        was_corrected,
            "constraints_applied":  constraints_applied,
            "soft_warnings":        soft_warnings,
            "target":               state.target_column,
            "problem_type":         state.problem_type,
            "explanation":          explanation,
            "model_used":           state.training_results.get("BestModel", "Unknown"),
            "encoding_maps": {
                k: v for k, v in (state.encoding_maps or {}).items()
                if k in feature_names
            }
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)})



@router.get("/sample-predictions/")
async def sample_predictions():
    """
    Runs prediction on first 5 rows of the dataset.
    Returns actual vs predicted values.

    FIX: Uses the already-fitted state.preprocessor.transform() instead of
    re-calling run_preprocessing() from scratch on 5 rows (which re-fits the
    OneHotEncoder on a tiny slice → wrong column count → shape mismatch crash).
    """
    if state.best_model is None:
        return JSONResponse(content={"error": "Train a model first"})

    if state.df is None:
        return JSONResponse(content={"error": "No dataset loaded"})

    if state.preprocessor is None:
        return JSONResponse(content={"error": "Run preprocessing first"})

    try:
        df     = state.df.copy()
        target = state.target_column

        # ── Grab 5 sample rows (features only) ───────────────────────────────
        sample_df     = df.head(5).copy()
        actual_values = sample_df[target].tolist()
        X_raw         = sample_df.drop(columns=[target])

        # ── Impute missing values to mirror the training pipeline ─────────────
        #    (SimpleImputer was applied per-column during run_preprocessing;
        #     for 5 rows we just fill NaNs with 0 / "unknown" — safe fallback)
        numerical_cols   = X_raw.select_dtypes(include="number").columns.tolist()
        categorical_cols = X_raw.select_dtypes(include="object").columns.tolist()
        if numerical_cols:
            X_raw[numerical_cols] = X_raw[numerical_cols].fillna(X_raw[numerical_cols].median())
        if categorical_cols:
            X_raw[categorical_cols] = X_raw[categorical_cols].fillna("unknown")

        # ── Apply ordinal / binary / frequency encodings stored in state ──────
        #    The fitted ColumnTransformer (state.preprocessor) handles OneHot
        #    columns but ordinal/binary/frequency cols were encoded BEFORE it
        #    was fitted, so we must replicate those steps first.
        encoding_maps = state.encoding_maps or {}
        for col, enc_map in encoding_maps.items():
            if col not in X_raw.columns:
                continue
            # All encoding_maps store {original_value: numeric_code}
            mapped = X_raw[col].astype(str).str.lower().map(
                {str(k).lower(): v for k, v in enc_map.items()}
            )
            # Unknown categories → 0
            X_raw[col] = mapped.fillna(0).astype(float)

        # ── Use the ALREADY-FITTED preprocessor (handles OneHot correctly) ────
        X_sample = state.preprocessor.transform(X_raw)

        if hasattr(X_sample, "toarray"):
            X_sample = X_sample.toarray()

        # ── Scale with the ALREADY-FITTED scaler ──────────────────────────────
        if state.needs_scaling and state.scaler:
            X_sample = state.scaler.transform(X_sample)

        # ── Predict ───────────────────────────────────────────────────────────
        predictions = state.best_model.predict(X_sample)

        rows = []
        for i in range(len(actual_values)):
            actual    = actual_values[i]
            predicted = predictions[i]

            if state.problem_type == "regression":
                actual    = round(float(actual), 3)
                predicted = round(float(predicted), 3)
                error     = round(abs(actual - predicted), 3)
                match     = error < abs(actual) * 0.1 if actual != 0 else error < 0.1
            else:
                actual    = str(actual)
                predicted = str(predicted)
                error     = 0
                match     = actual == predicted

            rows.append({
                "row":       i + 1,
                "actual":    actual,
                "predicted": predicted,
                "error":     error,
                "match":     match
            })

        return JSONResponse(content={
            "samples":      rows,
            "target":       target,
            "problem_type": state.problem_type,
            "model":        state.training_results.get("BestModel", "Unknown")
        })

    except Exception as e:
        import traceback
        print(f"[sample_predictions] ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e), "detail": traceback.format_exc()})
    
@router.get("/encoding-maps/")
async def get_encoding_maps():
    if not state.encoding_maps:
        return JSONResponse(content={"encoding_maps": {}})
    return JSONResponse(content={"encoding_maps": state.encoding_maps})


@router.get("/confusion-matrix/")
async def confusion_matrix_data():
    """
    Returns the confusion matrix computed during training.
    Already stored in state.training_results["ConfusionMatrix"] — no re-prediction needed.
    """
    if state.training_results is None:
        return JSONResponse(content={"error": "Train a model first"})
    if state.problem_type != "classification":
        return JSONResponse(content={"error": "Only available for classification problems"})

    matrix = state.training_results.get("ConfusionMatrix")
    if not matrix:
        return JSONResponse(content={"error": "No confusion matrix available (model may not have been evaluated on test set)"})

    # Build label list — use label_encoder if available, else 0..N
    if state.label_encoder is not None:
        labels = [str(c) for c in state.label_encoder.classes_]
    else:
        n = len(matrix)
        labels = [str(i) for i in range(n)]

    return JSONResponse(content={
        "matrix": matrix,
        "labels": labels,
        "model":  state.training_results.get("BestModel", "Unknown"),
        "problem_type": state.problem_type,
    })


@router.get("/roc-curve/")
async def roc_curve_data():
    """
    Computes the ROC curve on the held-out test set.
    Only valid for binary classification.
    """
    if state.training_results is None or state.best_model is None:
        return JSONResponse(content={"error": "Train a model first"})
    if state.problem_type != "classification":
        return JSONResponse(content={"error": "Only available for classification problems"})
    if state.X_test is None or state.y_test is None:
        return JSONResponse(content={"error": "No test data available"})

    from sklearn.metrics import roc_curve, roc_auc_score

    y_test = state.y_test
    n_classes = len(np.unique(y_test))
    if n_classes != 2:
        return JSONResponse(content={"error": f"ROC curve requires binary classification (found {n_classes} classes)"})

    try:
        if hasattr(state.best_model, "predict_proba"):
            y_score = state.best_model.predict_proba(state.X_test)[:, 1]
        elif hasattr(state.best_model, "decision_function"):
            y_score = state.best_model.decision_function(state.X_test)
        else:
            return JSONResponse(content={"error": "Model does not support probability estimates"})

        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc          = float(roc_auc_score(y_test, y_score))

        return JSONResponse(content={
            "fpr":   [round(float(v), 4) for v in fpr],
            "tpr":   [round(float(v), 4) for v in tpr],
            "auc":   round(auc, 4),
            "model": state.training_results.get("BestModel", "Unknown"),
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)})