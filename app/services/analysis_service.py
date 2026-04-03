"""
Analysis service — pattern discovery, clustering, feature interactions,
statistical insights, auto graph selection.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── Statistical insight per graph type ────────────────────────────────────────

def generate_statistical_insight(df: pd.DataFrame, graph: dict) -> str:
    insight = ""

    if graph["type"] == "histogram":
        col = graph["x"]
        data = df[col].dropna()
        if len(data) == 0:
            return "No sufficient data available for analysis."
        mean   = data.mean()
        median = data.median()
        skew   = data.skew()
        shape  = "right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "fairly symmetric")
        insight = (
            f"The distribution of {col} has an average value of {mean:.2f} "
            f"with a median of {median:.2f}. The data appears {shape}, "
            f"indicating how values are concentrated across the dataset."
        )

    elif graph["type"] == "scatter":
        x, y = graph["x"], graph["y"]
        data = df[[x, y]].dropna()
        if len(data) == 0:
            return "Not enough data to evaluate relationship."
        corr      = data[x].corr(data[y])
        strength  = "strong" if abs(corr) > 0.7 else ("moderate" if abs(corr) > 0.4 else "weak")
        direction = "positive" if corr > 0 else "negative"
        insight = (
            f"{x} and {y} show a {strength} {direction} correlation "
            f"({corr:.2f}). This suggests that changes in {x} "
            f"are associated with changes in {y}."
        )

    elif graph["type"] == "bar":
        col    = graph["x"]
        counts = df[col].value_counts()
        if len(counts) == 0:
            return "No categorical distribution available."
        top = counts.idxmax()
        insight = (
            f"The category '{top}' appears most frequently in {col}, "
            f"indicating it dominates the dataset distribution."
        )

    elif graph["type"] == "box":
        x, y   = graph["x"], graph["y"]
        groups = df.groupby(x)[y].mean().dropna()
        if len(groups) == 0:
            return "No group comparison insight available."
        top_group = groups.idxmax()
        insight = (
            f"The category '{top_group}' has the highest average {y}, "
            f"suggesting this group tends to produce larger values."
        )

    return insight


# ── Pattern discovery ──────────────────────────────────────────────────────────

def discover_patterns(df: pd.DataFrame, target_column: str) -> list:
    patterns = []
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.75:
                        direction = "positive" if corr > 0 else "negative"
                        patterns.append(
                            f"Strong {direction} relationship detected between "
                            f"'{col1}' and '{col2}' (correlation {corr:.2f})."
                        )

    for col in numerical_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        if len(outliers) > 0:
            percentage = (len(outliers) / len(df)) * 100
            if percentage > 3:
                patterns.append(
                    f"Potential outliers detected in '{col}', "
                    f"representing {percentage:.1f}% of observations."
                )

    if target_column in numerical_cols:
        correlations = df[numerical_cols].corr()[target_column].drop(target_column)
        top_features = correlations.abs().sort_values(ascending=False).head(3)
        for feature in top_features.index:
            corr_value = correlations[feature]
            direction  = "positive" if corr_value > 0 else "negative"
            patterns.append(
                f"'{feature}' shows a {direction} relationship with "
                f"the target variable '{target_column}'."
            )

    return patterns


# ── Cluster discovery ──────────────────────────────────────────────────────────

def discover_clusters(df: pd.DataFrame) -> list:
    cluster_insights = []
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numerical_cols = [
        col for col in numerical_cols
        if not any(x in col.lower() for x in ["id", "sl", "index"])
    ]

    if len(numerical_cols) < 2:
        return cluster_insights

    data = df[numerical_cols].dropna()
    if len(data) < 20:
        return cluster_insights

    # Sample for speed — KMeans on 5000 rows is representative and much faster
    if len(data) > 5000:
        data = data.sample(5000, random_state=42)

    scaler      = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans      = KMeans(n_clusters=3, random_state=42)
    clusters    = kmeans.fit_predict(scaled_data)

    data          = data.copy()
    data["cluster"] = clusters
    cluster_summary = data.groupby("cluster").mean()
    overall_mean    = data[numerical_cols].mean()

    for cluster_id in cluster_summary.index:
        cluster_mean = cluster_summary.loc[cluster_id]
        differences  = (cluster_mean - overall_mean).abs()
        top_features = differences.sort_values(ascending=False).head(2).index
        description  = [
            f"higher {f}" if cluster_mean[f] > overall_mean[f] else f"lower {f}"
            for f in top_features
        ]
        cluster_insights.append(
            f"Cluster {cluster_id + 1} shows {', '.join(description)} compared to the overall dataset."
        )

    return cluster_insights


# ── Feature interactions ───────────────────────────────────────────────────────

def discover_feature_interactions(df: pd.DataFrame, target_column: str) -> list:
    interactions   = []
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = [
        col for col in numerical_cols
        if not any(x in col.lower() for x in ["id", "sl", "index"])
    ]

    # Sample for speed on large datasets — interactions are statistical, sample is sufficient
    sample_df = df.sample(min(len(df), 5000), random_state=42) if len(df) > 5000 else df

    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            col1, col2 = numerical_cols[i], numerical_cols[j]
            data = sample_df[[col1, col2]].dropna()
            if len(data) < 20:
                continue
            corr = data[col1].corr(data[col2])
            if abs(corr) > 0.6:
                direction = "positive" if corr > 0 else "negative"
                interactions.append(
                    f"'{col1}' and '{col2}' show a strong {direction} interaction "
                    f"(correlation {corr:.2f}), indicating these variables move together."
                )

    for num_col in numerical_cols:
        for cat_col in categorical_cols:
            grouped = sample_df.groupby(cat_col)[num_col].mean()
            if len(grouped) < 2:
                continue
            top_category    = grouped.idxmax()
            bottom_category = grouped.idxmin()
            interactions.append(
                f"Category '{top_category}' in '{cat_col}' is associated with higher "
                f"average '{num_col}' compared to '{bottom_category}'."
            )

    return interactions


# ── Auto graph selection ───────────────────────────────────────────────────────

def build_auto_graphs(df: pd.DataFrame, target: str, strong_correlations: list) -> list:
    numerical_cols   = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    auto_graphs = []

    for col in numerical_cols[:5]:
        auto_graphs.append({"type": "histogram", "x": col, "y": None, "title": f"Distribution of {col}"})

    for col in categorical_cols[:5]:
        auto_graphs.append({"type": "bar", "x": col, "y": None, "title": f"Category Counts of {col}"})

    if target in numerical_cols:
        for col in numerical_cols:
            if col != target:
                auto_graphs.append({"type": "scatter", "x": col, "y": target, "title": f"{col} vs {target}"})
        for col in categorical_cols[:3]:
            auto_graphs.append({"type": "box", "x": col, "y": target, "title": f"{target} across {col}"})

    for item in strong_correlations[:5]:
        auto_graphs.append({
            "type": "scatter",
            "x": item["feature_1"],
            "y": item["feature_2"],
            "title": f"{item['feature_1']} vs {item['feature_2']}"
        })

    for graph in auto_graphs:
        graph["insight"] = generate_statistical_insight(df, graph)

    return auto_graphs


# ── Pattern visualizations ─────────────────────────────────────────────────────

def generate_pattern_visualizations(
    df: pd.DataFrame,
    clusters: list,
    interactions: list,
    numerical_cols: list
) -> list:
    pattern_graphs = []

    if len(numerical_cols) >= 2:
        pattern_graphs.append({
            "type": "cluster_scatter",
            "x": numerical_cols[0],
            "y": numerical_cols[1],
            "title": f"Cluster Visualization: {numerical_cols[0]} vs {numerical_cols[1]}"
        })

    for interaction in interactions[:3]:
        cols = [c for c in numerical_cols if c in interaction]
        if len(cols) >= 2:
            pattern_graphs.append({
                "type": "interaction_scatter",
                "x": cols[0],
                "y": cols[1],
                "title": f"Interaction Pattern: {cols[0]} vs {cols[1]}"
            })

    if len(numerical_cols) >= 2:
        pattern_graphs.append({
            "type": "outlier_scatter",
            "x": numerical_cols[0],
            "y": numerical_cols[1],
            "title": f"Outlier Detection: {numerical_cols[0]} vs {numerical_cols[1]}"
        })

    return pattern_graphs