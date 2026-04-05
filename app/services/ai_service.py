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


def ask_phi3(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 150,
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
    AI narrates the dataset profile in research-quality language.
    Called after upload or demo dataset load.
    """
    prompt = f"""You are an expert data scientist writing a research paper.
Analyze this dataset profile and write 3-4 sentences of intelligent insight.
Be specific, use proper ML terminology, and sound academic.

Dataset Profile:
- Rows: {rows}
- Columns: {columns}
- Numerical features: {numerical_count}
- Categorical features: {categorical_count}
- Missing data: {missing_percent}%
- Data quality score: {quality_score}/100
- Suggested problem type: {suggested_problem}

Write 2-3 complete sentences about this dataset's characteristics,
data quality, and suitability for machine learning.
Every sentence must be complete. Do not use bullet points. Do not cut off mid-sentence."""

    return ask_phi3(prompt)


def generate_training_explanation(
    best_model: str,
    problem_type: str,
    best_score: float,
    model_results: dict,
    top_features: list
) -> str:
    """
    AI explains the training results — why the best model won,
    what the score means, and what the features suggest.
    """
    # Format model results safely — values can be dict, list, or scalar
    def _fmt_score(scores):
        """Extract first numeric score regardless of type."""
        try:
            if isinstance(scores, dict):
                return list(scores.values())[0]
            if isinstance(scores, (list, tuple)):
                return scores[0]
            return float(scores)
        except Exception:
            return 0.0

    results_text = "\n".join([
        f"  - {name}: {_fmt_score(scores):.4f}"
        for name, scores in model_results.items()
        if name not in ["BestModel", "ProblemType"]
    ])

    metric_name = "R² score" if problem_type == "regression" else "accuracy"
    score_percent = round(best_score * 100, 1)

    prompt = f"""You are an expert ML researcher writing a research paper analysis.
Based on these AutoML training results, write 3-4 sentences of intelligent insight.
Use proper ML terminology. Sound academic and specific.

Training Results:
- Problem type: {problem_type}
- Best model: {best_model} ({metric_name}: {score_percent}%)
- Top influential features: {', '.join(top_features[:3]) if top_features else 'N/A'}

All model scores:
{results_text}

Write 2-3 complete sentences explaining why {best_model} performed best
and what the {score_percent}% {metric_name} implies for real-world use.
Every sentence must be complete. Do not cut off mid-sentence. Do not use bullet points.."""

    return ask_phi3(prompt)


def generate_pattern_explanation(
    pattern_score: int,
    pattern_strength: str,
    patterns: list,
    clusters: list,
    target_column: str,
    problem_type: str
) -> str:
    """
    AI interprets the cognitive pattern analysis results.
    This is the most research-worthy explanation.
    """
    patterns_text = "\n".join([f"  - {p}" for p in patterns[:5]]) if patterns else "  - No significant patterns detected"
    clusters_text = "\n".join([f"  - {c}" for c in clusters[:3]]) if clusters else "  - No clusters identified"

    prompt = f"""You are an expert AI researcher writing about cognitive pattern recognition.
Analyze these ML pattern discovery results and write a research-quality conclusion.
Use academic language, be specific, and relate findings to the target variable.

Cognitive Pattern Analysis:
- Target variable: {target_column}
- Problem type: {problem_type}
- Overall pattern score: {pattern_score}/100
- Pattern strength level: {pattern_strength}

Discovered patterns:
{patterns_text}

Cluster analysis:
{clusters_text}

Write 2-3 complete sentences specifically about the PATTERNS and CLUSTERS discovered
in the data structure. Focus only on what patterns were found and why they matter.
Every sentence must be complete. Do not cut off mid-sentence. Do not use bullet points."""

    return ask_phi3(prompt)


def generate_insight_summary(
    target_column: str,
    problem_type: str,
    best_model: str,
    best_score: float,
    pattern_score: int,
    top_feature: str
) -> str:
    """
    Final overall AI summary — the 'conclusion' section of the analysis.
    Perfect for showing to a mentor or including in a patent document.
    """
    metric = "R²" if problem_type == "regression" else "accuracy"
    score_percent = round(best_score * 100, 1)

    prompt = f"""You are an AI research system that has just completed a full
cognitive pattern recognition analysis. Write an executive summary in 4-5 sentences.
Sound like a published research paper conclusion. Be specific and impressive.

Analysis Summary:
- Target variable analyzed: {target_column}
- ML task: {problem_type}
- Best performing model: {best_model}
- Model {metric}: {score_percent}%
- Cognitive pattern score: {pattern_score}/100 
- Most influential feature: {top_feature}

Write 2-3 complete sentences as a final RESEARCH CONCLUSION summarizing the model
performance, prediction accuracy, and real-world significance of the findings.
Every sentence must be complete. Do not cut off mid-sentence. Do not use bullet points."""

    return ask_phi3(prompt)