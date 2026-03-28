"""
Pydantic schemas for dataset-related API responses.
These define the exact shape of JSON your API returns — no more raw dicts.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ProfileSummary(BaseModel):
    rows: int
    columns: int
    missing_percent: float
    suggested_problem: str
    quality_score: float


class MissingSummaryItem(BaseModel):
    count: int
    percentage: float


class StrongCorrelation(BaseModel):
    feature_1: str
    feature_2: str
    correlation: float


class DatasetUploadResponse(BaseModel):
    rows: int
    total_columns: int
    numerical_columns: List[str]
    categorical_columns: List[str]
    missing_summary: Dict[str, MissingSummaryItem]
    duplicate_count: int
    quality_score: float
    dataset_nature: str
    class_imbalance: Optional[str]
    dataset_summary: str
    strong_correlations: List[StrongCorrelation]
    profile_summary: ProfileSummary
    columns: List[str]
    preview: List[Dict[str, Any]]
    full_data: List[Dict[str, Any]]


class DemoDatasetInfo(BaseModel):
    file: str
    name: str
    category: str


class DemoDatasetsResponse(BaseModel):
    datasets: List[DemoDatasetInfo]