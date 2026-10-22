from pydantic import BaseModel, Field
from typing import List, Optional

class SourceReference(BaseModel):
    paper_id: str
    title: str
    year: int
    chunk_id: Optional[str] = None
    evidence_field: str
    similarity_score: Optional[float] = None

class GroundedExplanation(BaseModel):
    candidate_id: str
    candidate_gap_statement: str
    evidence_summary: str
    literature_context: str
    limitation_evidence: str
    temporal_context: str
    counterevidence: str
    evidence_limitations: str
    suggested_direction: str
    source_references: List[SourceReference] = Field(default_factory=list)
    grounding_status: str = "Success"
