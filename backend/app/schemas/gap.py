from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class GapEngineConfig(BaseModel):
    rarity_weight: float = 0.20
    intersection_weight: float = 0.20
    temporal_weight: float = 0.20
    limitation_weight: float = 0.20
    semantic_weight: float = 0.20
    topic_presence_threshold: float = 0.20
    minimum_temporal_points: int = 3
    minimum_evidence_count: int = 1

class GapEvidence(BaseModel):
    paper_id: Optional[str] = None
    evidence_type: str  # e.g., "limitation_phrase", "temporal_trend", "sparse_intersection"
    topic_index: Optional[int] = None
    topic_pair: Optional[List[int]] = None
    year: Optional[int] = None
    evidence_text: Optional[str] = None
    evidence_metadata: Dict[str, Any] = Field(default_factory=dict)

class GapCandidate(BaseModel):
    candidate_id: str
    topic_indices: List[int] = Field(default_factory=list)
    topic_labels: List[str] = Field(default_factory=list)
    latent_dimension_indices: List[int] = Field(default_factory=list)
    rarity_score: Optional[float] = None
    intersection_score: Optional[float] = None
    temporal_score: Optional[float] = None
    limitation_score: Optional[float] = None
    semantic_support_score: Optional[float] = None
    final_gap_score: float = 0.0
    evidence_count: int = 0
    confidence_level: str = "Low"  # evidence-coverage descriptor, not probability
    evidence_references: List[GapEvidence] = Field(default_factory=list)
    reasoning_signals: List[str] = Field(default_factory=list)
    descriptive_statement: Optional[str] = None
    confidence_rationale: Optional[str] = None
    missing_evidence: List[str] = Field(default_factory=list)

class GapRunResponse(BaseModel):
    analysis_id: str
    candidate_count: int
    threshold: float
    weights: Dict[str, float]
    scoring_status: str = "Complete"
    missing_signals: List[str] = Field(default_factory=list)
    candidates: List[GapCandidate]
