"""
Pydantic schemas for dashboard and analysis responses.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class PatternScore(BaseModel):
    score: int
    pattern_strength: str
    data_quality: int


class AutoGraph(BaseModel):
    type: str
    x: str
    y: Optional[str]
    title: str
    insight: Optional[str] = ""


class DashboardResponse(BaseModel):
    dataset_summary: str
    target_distribution: Dict[str, Any]
    correlation_matrix: List[List[float]]
    correlation_labels: List[str]
    feature_importance: Dict[str, float]
    model_comparison: Dict[str, float]
    insights: List[str]
    patterns: List[str]
    clusters: List[str]
    auto_graphs: List[AutoGraph]
    feature_interactions: List[str]
    ai_conclusion: str
    prediction_analysis: str
    pattern_score: PatternScore
    pattern_visualizations: List[Dict[str, Any]]
    full_data: List[Dict[str, Any]]