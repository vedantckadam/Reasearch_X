from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class HumanEvaluation(BaseModel):
    candidate_id: str
    reviewer_id: str
    is_relevant: bool
    is_novel: bool
    evidence_sufficient: bool
    comments: Optional[str] = None

class RetrievalMetric(BaseModel):
    analysis_id: str
    precision_at_k: float
    recall_at_k: float
    k: int
    has_ground_truth: bool

class GapAgreementMetric(BaseModel):
    candidate_id: str
    human_agreement_rate: float
    total_reviews: int

class EvidenceClaimMetric(BaseModel):
    candidate_id: str
    supported_claims: int
    total_claims: int
    support_rate: float

class AblationResult(BaseModel):
    analysis_id: str
    config_name: str
    coherence_score: float
    num_topics: int
    gap_candidates_found: int
    average_human_agreement: Optional[float] = None

class BERTopicComparison(BaseModel):
    analysis_id: str
    lda_coherence: float
    bertopic_coherence: Optional[float] = None
    bertopic_available: bool
