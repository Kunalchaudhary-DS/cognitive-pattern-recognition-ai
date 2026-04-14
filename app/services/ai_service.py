"""
AI Service — connects to locally running Ollama (Phi-3) to generate
intelligent, research-quality explanations of ML results.

Ollama must be running before using this service.
Start it with: ollama serve  (runs in background automatically after install)
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tinyllama"


def ask_phi3(prompt: str, num_predict: int = 120, temperature: float = 0.4) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                }
            },
            timeout=120
        )

        if response.status_code == 200:
            text = response.json().get("response", "").strip()
            return ensure_complete_sentences(text)
        else:
            return "AI explanation unavailable at this time."

    except requests.exceptions.ConnectionError:
        return "Ollama is not running. Please start it with: ollama serve"
    except requests.exceptions.Timeout:
        return "AI response timed out. Try again."
    except Exception as e:
        return f"AI service error: {str(e)}"


def ensure_complete_sentences(text: str) -> str:
    """
    Trims text to end at the last complete sentence.
    A complete sentence ends with . ! or ?
    """
    if not text:
        return text

    # Remove numbered list prefixes like "1." "2." at start
    import re
    text = re.sub(r'^\d+\.\s+', '', text)
    text = re.sub(r'\s+\d+\.\s+', ' ', text)

    # Find last sentence-ending punctuation
    last_end = max(
        text.rfind('.'),
        text.rfind('!'),
        text.rfind('?')
    )

    if last_end > 0:
        return text[:last_end + 1].strip()

    return text.strip()


# ── Explanation generators ─────────────────────────────────────────────────────
def generate_dataset_explanation(
    rows: int,
    columns: int,
    numerical_count: int,
    categorical_count: int,
    missing_percent: float,
    quality_score: float,
    suggested_problem: str
) -> str:
    """
    AI narrates the dataset profile — short, punchy, direct.
    """
    quality_verdict = "high-quality" if quality_score >= 90 else "moderate-quality" if quality_score >= 70 else "low-quality"
    missing_note    = f"with {missing_percent}% missing values" if missing_percent > 0 else "with no missing values"

    prompt = f"""Write exactly 2 sentences of ML insight about this dataset. Start directly with the finding. No introduction.

Facts: {rows} rows, {columns} columns ({numerical_count} numerical, {categorical_count} categorical), {missing_note}, quality score {quality_score}/100, task: {suggested_problem}.

Sentence 1: Describe what this dataset is suited for and its key characteristic.
Sentence 2: State one specific strength or risk for ML training based on the numbers above.
No bullet points. No "I" statements. End each sentence with a period."""

    return ask_phi3(prompt, num_predict=100)


def generate_training_explanation(
    best_model: str,
    problem_type: str,
    best_score: float,
    model_results: dict,
    top_features: list
) -> str:
    """
    AI explains training results — direct insight on WHY the best model won.
    """
    metric_name   = "R²" if problem_type == "regression" else "accuracy"
    score_percent = round(best_score * 100, 1)
    features_str  = ", ".join(f"'{f}'" for f in top_features[:3]) if top_features else "N/A"

    # Build a lean model ranking string (top 3 only)
    def _get_score(v):
        try:
            if isinstance(v, dict): return list(v.values())[0]
            return float(v)
        except: return 0.0

    ranking = sorted(
        [(k, _get_score(v)) for k, v in model_results.items()
         if k not in ["BestModel", "ProblemType", "PrimaryMetric", "Imbalanced", "ConstraintMap"]
         and isinstance(v, dict)],
        key=lambda x: x[1], reverse=True
    )[:3]
    ranking_str = ", ".join(f"{n} ({s*100:.1f}%)" for n, s in ranking)

    prompt = f"""Write exactly 2 sentences explaining these AutoML results. Start with the key finding. No introduction.

Best model: {best_model} | {metric_name}: {score_percent}% | Top features: {features_str} | Top 3 models: {ranking_str} | Task: {problem_type}

Sentence 1: State why {best_model} likely outperformed others for this {problem_type} task.
Sentence 2: Explain what a {score_percent}% {metric_name} means practically for real-world deployment.
No bullet points. No "I" statements. Each sentence ends with a period."""

    return ask_phi3(prompt, num_predict=110)


def generate_pattern_explanation(
    pattern_score: int,
    pattern_strength: str,
    patterns: list,
    clusters: list,
    target_column: str,
    problem_type: str
) -> str:
    """
    AI interprets pattern analysis results — focused on what they mean for the target.
    """
    top_pattern  = patterns[0]  if patterns  else "No significant patterns detected"
    cluster_info = f"{len(clusters)} distinct clusters found" if clusters else "No clusters identified"

    prompt = f"""Write exactly 2 sentences interpreting these ML pattern findings. Lead with the most important insight.

Target: '{target_column}' | Score: {pattern_score}/100 ({pattern_strength}) | Strongest pattern: {top_pattern} | Clusters: {cluster_info}

Sentence 1: Interpret what the strongest pattern reveals about the relationship between features and '{target_column}'.
Sentence 2: Explain what the {cluster_info} implies about the underlying data population structure.
No bullet points. No "I" statements. Be specific. Each sentence ends with a period."""

    return ask_phi3(prompt, num_predict=110)


def generate_insight_summary(
    target_column: str,
    problem_type: str,
    best_model: str,
    best_score: float,
    pattern_score: int,
    top_feature: str
) -> str:
    """
    Final research conclusion — direct, impressive, no filler.
    """
    metric        = "R²" if problem_type == "regression" else "accuracy"
    score_percent = round(best_score * 100, 1)
    strength      = "strong" if pattern_score >= 75 else "moderate" if pattern_score >= 50 else "weak"

    prompt = f"""Write exactly 2 sentences as a research conclusion. Open with the most impressive finding.

Study: {problem_type} on '{target_column}' | Best model: {best_model} | {metric}: {score_percent}% | Pattern score: {pattern_score}/100 ({strength}) | Key driver: '{top_feature}'

Sentence 1: State the core finding — model performance and what it means for predicting '{target_column}'.
Sentence 2: Connect '{top_feature}' and the {strength} pattern score to a real-world implication.
No bullet points. No "I" statements. No restatement of raw numbers already visible. Each sentence ends with a period."""

    return ask_phi3(prompt, num_predict=110)


# ── Panel-Level AI Insights (Discovered Patterns + Cluster Analysis) ──────────

def generate_panel_insights(
    patterns: list,
    clusters: list,
    target_column: str,
    problem_type: str,
) -> dict:
    """
    Generates short, specific AI insights for the two dashboard panels:
    'Discovered Patterns' and 'Cluster Analysis'.

    Returns:
        {
            "patterns_insight": str,
            "clusters_insight": str,
        }
    """
    # ── Patterns insight ─────────────────────────────────────────────────────
    if patterns:
        top3 = patterns[:3]
        patterns_context = " | ".join(top3)
        pattern_prompt = f"""Write 1 sentence of expert insight about these discovered ML patterns. Be specific and non-obvious.

Target: '{target_column}' | Patterns found: {patterns_context}

One sentence only. Start with the key finding. No introduction. End with a period."""
        patterns_insight = ask_phi3(pattern_prompt, num_predict=80, temperature=0.3)
    else:
        patterns_insight = f"No statistically significant patterns were detected between features and '{target_column}'." 

    # ── Clusters insight ─────────────────────────────────────────────────────
    if clusters:
        cluster_context = " | ".join(clusters[:3])
        cluster_prompt = f"""Write 1 sentence of expert insight about these data clusters. Be specific and non-obvious.

Target: '{target_column}' | Task: {problem_type} | Clusters: {cluster_context}

One sentence only. Start with the key finding. No introduction. End with a period."""
        clusters_insight = ask_phi3(cluster_prompt, num_predict=80, temperature=0.3)
    else:
        clusters_insight = "The dataset does not exhibit clear sub-group separation, suggesting a homogeneous sample distribution."

    return {
        "patterns_insight": patterns_insight,
        "clusters_insight": clusters_insight,
    }


# Semantic Constraint Inference (Layer 2 of Prediction Interceptor) 

def generate_semantic_constraints(
    column_names: list,
    target_column: str,
    dtype_map: dict,
    statistical_bounds: dict,
    problem_type: str,
) -> dict:
    """
    Asks Ollama to infer real-world domain constraints from column names.

    Uses the statistical_bounds as grounding context so the LLM works within
    observed data reality. Returns a structured constraint dict:
      {
        "target_bounds":  { "min": ..., "max": ..., "reason": "..." },
        "relative_rules": [ { "target_col": ..., "operator": ..., "ref_col": ..., "reason": ... } ]
      }

    Always returns {} gracefully on any failure — system degrades to
    statistical-only constraints without breaking.
    """
    if problem_type != "regression":
        return {}

    if not column_names or not target_column:
        return {}

    # Build a compact schema string for the prompt
    schema_lines = []
    stat_tb = statistical_bounds.get("target_bounds", {})

    for col in column_names:
        dtype = dtype_map.get(col, "numeric")
        prefix = f"  - {col} (dtype: {dtype})"
        if col == target_column:
            prefix += (
                f" [TARGET — observed range: "
                f"{stat_tb.get('hard_min', '?')} to {stat_tb.get('hard_max', '?')}]"
            )
        schema_lines.append(prefix)

    schema_text = "\n".join(schema_lines)

    prompt = f"""You are a domain expert analyzing a machine learning dataset.
Your task: infer logical real-world constraints for the TARGET variable based solely on column names and observed ranges.

COLUMNS:
{schema_text}

TARGET column: "{target_column}"

INSTRUCTIONS:
1. Is there a real-world maximum or minimum for "{target_column}"? (e.g., scores cap at 100, ages cannot be negative)
2. Is "{target_column}" logically bounded by any other column? (e.g., subset durations cannot exceed their parent duration)
3. Only include constraints you are highly confident about from the column names alone.

Return ONLY a valid JSON object. No explanation. No markdown. No extra text.
Format:
{{
  "target_bounds": {{
    "min": <number or null>,
    "max": <number or null>,
    "reason": "<brief reason>"
  }},
  "relative_rules": [
    {{
      "target_col": "{target_column}",
      "operator": "<= or <",
      "ref_col": "<column name from schema>",
      "reason": "<brief reason>"
    }}
  ]
}}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1, 
                    "num_predict": 250,
                },
            },
            timeout=60,
        )

        if response.status_code != 200:
            print("[Constraints-LLM] Ollama returned non-200. Using statistical only.")
            return {}

        raw_text = response.json().get("response", "").strip()

        # Extract JSON from response — handle cases where model wraps in markdown
        import re
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            print(f"[Constraints-LLM] No JSON found in response: {raw_text[:200]}")
            return {}

        parsed = json.loads(json_match.group())

        # Validate structure — must have known keys
        if not isinstance(parsed, dict):
            return {}

        print(f"[Constraints-LLM] Semantic constraints inferred: {parsed}")
        return parsed

    except requests.exceptions.ConnectionError:
        print("[Constraints-LLM] Ollama offline. Using statistical constraints only.")
        return {}
    except requests.exceptions.Timeout:
        print("[Constraints-LLM] Ollama timed out. Using statistical constraints only.")
        return {}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Constraints-LLM] JSON parse error: {e}. Using statistical only.")
        return {}
    except Exception as e:
        print(f"[Constraints-LLM] Unexpected error: {e}. Using statistical only.")
        return {}