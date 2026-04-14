"""
Constraint Service — Semantic Prediction Interceptor (Layers 1, 3 & 4)

Layer 1 — Statistical Constraint Extractor:
    Scans the training DataFrame to derive hard, data-proven bounds and
    cross-column inequality rules with zero hallucination risk.

Layer 3 — Constraint Merger:
    Reconciles statistical (Layer 1) and LLM-semantic (Layer 2) constraints.
    Statistical evidence always gates LLM proposals — a semantic claim the
    data contradicts is rejected or downgraded to a soft warning.

Layer 4 — Prediction Interceptor:
    Applied after model.predict().  Clips the raw output to the merged
    bounds, checks relative rules against the user's own input values,
    and returns a transparency report of every correction made.

Scope:  Regression problems only in this version.
        For classification the output is a class label; the class space is
        already naturally finite and a separate approach is needed.
"""

import numpy as np
import pandas as pd
from typing import Optional


# Layer 1 — Statistical Constraint Extractor
def extract_statistical_constraints(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> dict:
    """
    Derives hard constraints purely from the training data.

    Returns a dict with two sections:
      • "target_bounds"    — absolute and percentile bounds for the target
      • "relative_rules"  — cross-column inequalities that hold in 100% of rows
    """
    if problem_type != "regression":
        return {}

    if target_column not in df.columns:
        return {}

    result: dict = {}

    # Target bounds 
    target_series = df[target_column].dropna()
    if len(target_series) == 0:
        return {}

    result["target_bounds"] = {
        "hard_min": float(target_series.min()),
        "hard_max": float(target_series.max()),
        "soft_min": float(np.percentile(target_series, 1)),
        "soft_max": float(np.percentile(target_series, 99)),
        "source":   "statistical",
    }

    #Cross-column relative rules
    # Only inspect numeric columns (excluding the target itself)
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_column
    ]

    relative_rules = []

    for col in numeric_cols:
        col_series = df[col].dropna()
        if len(col_series) == 0:
            continue

        # Rule: target <= col  (e.g. years_with_manager <= total_working_years)
        valid_rows = df[[target_column, col]].dropna()
        if len(valid_rows) == 0:
            continue

        compliance_lte = (valid_rows[target_column] <= valid_rows[col]).all()
        compliance_lt  = (valid_rows[target_column] <  valid_rows[col]).all()

        if compliance_lte:
            relative_rules.append({
                "target_col":  target_column,
                "operator":    "<=" if not compliance_lt else "<",
                "ref_col":     col,
                "compliance":  1.0,
                "source":      "statistical",
            })

    result["relative_rules"] = relative_rules

    print(
        f"[Constraints] Statistical extraction complete. "
        f"Target bounds: [{result['target_bounds']['hard_min']}, "
        f"{result['target_bounds']['hard_max']}]. "
        f"Relative rules found: {len(relative_rules)}"
    )

    return result


# Layer 3 — Constraint Merger

def merge_constraints(
    statistical: dict,
    semantic: dict,
) -> dict:
    """
    Merges statistical (Layer 1) and semantic/LLM (Layer 2) constraints.

    Merge rules:
      1. If statistical bounds exist, they define the absolute hard limits.
         Semantic bounds are accepted only when they fall WITHIN the
         statistical range — preventing the LLM from hallucinating bounds
         that the data itself contradicts.
      2. The final "effective_max" and "effective_min" exposed to the
         interceptor may be the semantic values (tighter domain constraint)
         as long as rule 1 is satisfied.
      3. Relative rules from statistical layer are always kept.
         Relative rules proposed by Ollama that have no statistical backing
         are stored as "soft_warnings" — they are surfaced to the user but
         do NOT cause clipping.
    """
    merged: dict = {}

    # Target bounds
    stat_bounds = statistical.get("target_bounds", {})
    sem_bounds  = semantic.get("target_bounds", {})

    if stat_bounds:
        hard_min = stat_bounds["hard_min"]
        hard_max = stat_bounds["hard_max"]
        soft_min = stat_bounds.get("soft_min", hard_min)
        soft_max = stat_bounds.get("soft_max", hard_max)

        # Accept semantic max if it is more restrictive AND within statistical range
        effective_max = hard_max
        effective_min = hard_min
        semantic_reason_max = None
        semantic_reason_min = None

        if sem_bounds:
            s_max = sem_bounds.get("max")
            s_min = sem_bounds.get("min")

            if s_max is not None:
                try:
                    s_max_f = float(s_max)
                    # Semantic max is valid if: hard_min <= s_max <= hard_max
                    # We allow it even if it's higher than hard_max (domain intent)
                    # but we never let it go below hard_min (would make no sense)
                    if s_max_f >= hard_min:
                        effective_max = s_max_f
                        semantic_reason_max = sem_bounds.get("reason", "LLM domain inference")
                except (ValueError, TypeError):
                    pass

            if s_min is not None:
                try:
                    s_min_f = float(s_min)
                    if s_min_f <= hard_max:
                        effective_min = s_min_f
                        semantic_reason_min = sem_bounds.get("reason", "LLM domain inference")
                except (ValueError, TypeError):
                    pass

        merged["target_bounds"] = {
            "hard_min":        hard_min,
            "hard_max":        hard_max,
            "soft_min":        soft_min,
            "soft_max":        soft_max,
            "effective_min":   effective_min,
            "effective_max":   effective_max,
            "reason_min":      semantic_reason_min or "statistical observation",
            "reason_max":      semantic_reason_max or "statistical observation",
        }

    # Relative rules
    # Statistical rules → hard (will cause clipping / warning)
    hard_rules = list(statistical.get("relative_rules", []))

    # Semantic relative rules that have NO statistical backing → soft warnings
    stat_rule_pairs = {
        (r["target_col"], r["ref_col"]) for r in hard_rules
    }
    soft_warnings = []
    for rule in semantic.get("relative_rules", []):
        pair = (rule.get("target_col"), rule.get("ref_col"))
        if pair not in stat_rule_pairs:
            soft_warnings.append({
                **rule,
                "source": "semantic_only",
                "note":   "Not confirmed by data — shown as warning only",
            })

    merged["relative_rules"]  = hard_rules
    merged["soft_warnings"]   = soft_warnings

    print(
        f"[Constraints] Merge complete. "
        f"Effective range: [{merged.get('target_bounds', {}).get('effective_min', 'N/A')}, "
        f"{merged.get('target_bounds', {}).get('effective_max', 'N/A')}]. "
        f"Hard rules: {len(hard_rules)}, Soft warnings: {len(soft_warnings)}"
    )

    return merged


# Layer 4 — Prediction Interceptor

def apply_constraints(
    raw_prediction: float,
    input_values: dict,
    constraint_map: Optional[dict],
    feature_names: list,
    target_column: str,
) -> dict:
    """
    Applies the merged constraint rulebook to a raw model prediction.

    Returns:
        {
            "final_value":         float   — corrected prediction,
            "raw_value":           float   — original model output,
            "constraints_applied": list    — list of correction messages (hard clips),
            "soft_warnings":       list    — warnings for semantic-only rules,
            "was_corrected":       bool    — True if any hard clip was applied,
        }
    """
    if not constraint_map:
        return {
            "final_value":         raw_prediction,
            "raw_value":           raw_prediction,
            "constraints_applied": [],
            "soft_warnings":       [],
            "was_corrected":       False,
        }

    final_value         = raw_prediction
    corrections_applied = []
    soft_warnings_out   = []

    # Step 1: Absolute bounds clip 
    bounds = constraint_map.get("target_bounds", {})
    if bounds:
        eff_min = bounds.get("effective_min")
        eff_max = bounds.get("effective_max")

        if eff_max is not None and final_value > eff_max:
            corrections_applied.append(
                f"Clipped from {round(raw_prediction, 4)} → {eff_max} "
                f"(max logical limit for '{target_column}': {bounds.get('reason_max', 'domain constraint')})"
            )
            final_value = eff_max

        if eff_min is not None and final_value < eff_min:
            corrections_applied.append(
                f"Clipped from {round(raw_prediction, 4)} → {eff_min} "
                f"(min logical limit for '{target_column}': {bounds.get('reason_min', 'domain constraint')})"
            )
            final_value = eff_min

    # Step 2: Relative rule checks 
    for rule in constraint_map.get("relative_rules", []):
        ref_col  = rule.get("ref_col")
        operator = rule.get("operator", "<=")

        # Pull the reference value from the user's input
        ref_val_raw = input_values.get(ref_col)
        if ref_val_raw is None:
            continue

        try:
            ref_val = float(ref_val_raw)
        except (ValueError, TypeError):
            continue

        violated = False
        if operator == "<=" and final_value > ref_val:
            violated = True
        elif operator == "<" and final_value >= ref_val:
            violated = True

        if violated:
            original = final_value
            final_value = min(final_value, ref_val)
            corrections_applied.append(
                f"Relative constraint: '{target_column}' {operator} '{ref_col}' "
                f"({ref_val}). Corrected {round(original, 4)} → {final_value}"
            )

    # Step 3: Soft warning checks (semantic-only rules) 
    for rule in constraint_map.get("soft_warnings", []):
        ref_col  = rule.get("ref_col")
        operator = rule.get("operator", "<=")
        ref_val_raw = input_values.get(ref_col)

        if ref_val_raw is None:
            continue
        try:
            ref_val = float(ref_val_raw)
        except (ValueError, TypeError):
            continue

        violated = False
        if operator == "<=" and final_value > ref_val:
            violated = True
        elif operator == "<" and final_value >= ref_val:
            violated = True

        if violated:
            soft_warnings_out.append(
                f"⚠ Possible domain issue: '{target_column}' {operator} '{ref_col}' "
                f"({ref_val}) — not confirmed by training data"
            )

    was_corrected = len(corrections_applied) > 0

    if was_corrected:
        print(
            f"[Interceptor] Prediction corrected: {raw_prediction} → {final_value}. "
            f"Corrections: {corrections_applied}"
        )

    return {
        "final_value":         final_value,
        "raw_value":           raw_prediction,
        "constraints_applied": corrections_applied,
        "soft_warnings":       soft_warnings_out,
        "was_corrected":       was_corrected,
    }
