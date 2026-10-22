from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class LatentTopicLoading(BaseModel):
    topic_index: int
    loading: float
    absolute_loading: float
    rank: int

class LatentDimension(BaseModel):
    dimension_index: int
    singular_value: float
    squared_singular_value: float
    latent_energy_ratio: float
    cumulative_energy: float
    eigenvalue: float
    normalized_eigen_contribution: float
    cumulative_eigen_contribution: float
    is_low_energy: bool
    dominant_topics: List[LatentTopicLoading] = Field(default_factory=list)

class LatentAnalysisSummary(BaseModel):
    total_dimensions: int
    rank: int
    reconstruction_error: float
    energy_80_dimensions: List[int]
    energy_90_dimensions: List[int]
    low_energy_dimension_indices: List[int] = Field(default_factory=list)
    low_energy_threshold: float

class TemporalCoverage(BaseModel):
    first_observed_year: Optional[int]
    last_observed_year: Optional[int]
    number_of_years_present: Optional[int]

class TemporalTopic(BaseModel):
    year: int
    topic_index: int
    paper_count: int
    dominant_count: int
    dominant_percentage: float
    probability_mass: float
    average_probability: float
    trend: str
    temporal_coverage_metadata: TemporalCoverage

class TemporalIntersection(BaseModel):
    year: int
    topic_pair: List[int]
    support_count: int
    weighted_intersection: float
    normalized_weighted_value: float
    threshold: float

class LatentResultResponse(BaseModel):
    analysis_id: str
    dimensions: List[LatentDimension]

class TemporalTopicResponse(BaseModel):
    analysis_id: str
    topics: List[TemporalTopic]

class TemporalIntersectionResponse(BaseModel):
    analysis_id: str
    intersections: List[TemporalIntersection]

class CombinedLatentTemporalResponse(BaseModel):
    analysis_id: str
    latent: Dict[str, Any]
    temporal: List[Any]
    temporal_intersections: List[Any]

class RunAnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    latent_dimensions: int
    energy_80_dimensions: List[int]
    energy_90_dimensions: List[int]
    temporal_years: List[int]
    low_energy_dimensions: List[int]
