from sqlalchemy.orm import Session
from typing import List, Dict, Any, Tuple
from app.schemas.gap import GapCandidate
from app.models.paper_db import PaperDB, Topic, TopicAnalysis
from app.services.rag_chunking import chunk_paper, RAGChunk
from app.services.vector_store import get_vector_store

def initialize_analysis_store(session: Session, analysis_id: str):
    """
    Chunks all papers in an analysis and loads them into the local vector store.
    """
    store = get_vector_store(analysis_id)
    if store.chunks:
        return # Already initialized
        
    papers = session.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).all()
    all_chunks = []
    for p in papers:
        all_chunks.extend(chunk_paper(p))
        
    store.add_chunks(all_chunks)

def build_retrieval_query(session: Session, analysis_id: str, candidate: GapCandidate) -> str:
    """
    Constructs a retrieval query based on the candidate gap signals.
    """
    query_parts = []
    
    if candidate.topic_labels:
        query_parts.append(" ".join(candidate.topic_labels))
        
    for ev in candidate.evidence_references:
        if ev.evidence_type == "limitation_phrase" and ev.evidence_text:
            query_parts.append(ev.evidence_text)
            
    # Include topic words if topic labels aren't enough
    topic_analysis = session.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).first()
    if topic_analysis and candidate.topic_indices:
        topics = session.query(Topic).filter(
            Topic.topic_analysis_id == topic_analysis.id,
            Topic.topic_idx.in_(candidate.topic_indices)
        ).all()
        for t in topics:
            if t.top_words:
                # top_words might be a dict or list depending on how it was stored
                if isinstance(t.top_words, dict):
                    query_parts.extend(list(t.top_words.keys())[:5])
                elif isinstance(t.top_words, list):
                    query_parts.extend([str(w) for w in t.top_words[:5]])
                
    return " ".join(query_parts)

def retrieve_evidence(
    session: Session, 
    analysis_id: str, 
    candidate: GapCandidate,
    top_k: int = 5,
    min_similarity: float = 0.0
) -> List[Tuple[RAGChunk, float]]:
    """
    Retrieves evidence (both supporting and counterevidence) for a gap candidate.
    """
    initialize_analysis_store(session, analysis_id)
    store = get_vector_store(analysis_id)
    
    query = build_retrieval_query(session, analysis_id, candidate)
    if not query.strip():
        return []
        
    results = store.search(query, top_k=top_k, min_similarity=min_similarity)
    
    # Filter out duplicates (e.g., same paper, multiple chunks) if needed, 
    # but returning chunks is fine for RAG.
    
    return results
