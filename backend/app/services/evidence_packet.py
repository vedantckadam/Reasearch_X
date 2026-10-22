from typing import List, Dict, Any, Tuple
from app.schemas.gap import GapCandidate
from app.services.rag_chunking import RAGChunk

def build_evidence_packet(candidate: GapCandidate, retrieved_chunks: List[Tuple[RAGChunk, float]]) -> Dict[str, Any]:
    """
    Constructs a structured evidence packet to be sent to Nemotron.
    """
    structured_evidence = []
    for ev in candidate.evidence_references:
        structured_evidence.append({
            "type": ev.evidence_type,
            "text": ev.evidence_text,
            "year": ev.year,
            "paper_id": ev.paper_id,
            "metadata": ev.evidence_metadata
        })
        
    chunks = []
    source_metadata = []
    
    for chunk, sim in retrieved_chunks:
        chunks.append({
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "similarity": sim
        })
        source_metadata.append({
            "chunk_id": chunk.chunk_id,
            "paper_id": chunk.paper_id,
            "title": chunk.title,
            "year": chunk.year,
            "source_field": chunk.source_field
        })
        
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_signals": {
            "rarity_score": candidate.rarity_score,
            "intersection_score": candidate.intersection_score,
            "temporal_score": candidate.temporal_score,
            "limitation_score": candidate.limitation_score,
            "semantic_support_score": candidate.semantic_support_score,
            "reasoning_signals": candidate.reasoning_signals
        },
        "structured_evidence": structured_evidence,
        "retrieved_chunks": chunks,
        "source_metadata": source_metadata
    }
