from pydantic import BaseModel, Field
from typing import List, Optional

class PaperSchema(BaseModel):
    canonical_id: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    citation_count: int = 0
    url: Optional[str] = None
    publication_venue: Optional[str] = None
    is_open_access: bool = False
    relevance_score: float = 0.0
    openalex_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
