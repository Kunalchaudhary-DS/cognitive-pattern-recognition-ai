"""
Pydantic schemas for preprocessing and training responses.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple


class PreprocessResponse(BaseModel):
    original_shape: Tuple[int, int]
    processed_feature_shape: Tuple[int, int]
    target_shape: Tuple[int, int]
    problem_type: str
    binary_encoded: List[str]
    onehot_encoded: List[str]
    frequency_encoded: List[str]
    dropped_target_rows: int
    message: str


class FeatureImportanceItem(BaseModel):
    feature: str
    correlation: float


class FeatureImportanceResponse(BaseModel):
    feature_importance: List[FeatureImportanceItem]
    problem_type: str