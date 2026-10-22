import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from app.models.paper_db import SessionLocal, Analysis, GapCandidateDB, GapEvidenceDB, RAGChunkDB, RAGRetrievalDB, GroundedExplanationDB
from app.schemas.rag import GroundedExplanation, SourceReference
from app.schemas.gap import GapCandidate, GapEvidence
from app.services.rag_retrieval import retrieve_evidence
from app.services.evidence_packet import build_evidence_packet
from app.services.nemotron_client import NemotronClient
from app.services.rag_prompts import GROUNDED_SYSTEM_PROMPT

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/{analysis_id}/gaps/{candidate_id}/explain", response_model=GroundedExplanation)
def explain_gap_route(analysis_id: str, candidate_id: str, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    c_db = db.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == analysis_id, GapCandidateDB.candidate_id == candidate_id).first()
    if not c_db:
        raise HTTPException(status_code=404, detail="Candidate not found")
        
    ev_db = db.query(GapEvidenceDB).filter(GapEvidenceDB.candidate_id == candidate_id).all()
    evidence_references = [
        GapEvidence(
            paper_id=e.paper_id,
            evidence_type=e.evidence_type,
            topic_index=e.topic_index,
            topic_pair=e.topic_pair,
            year=e.year,
            evidence_text=e.evidence_text,
            evidence_metadata=e.evidence_metadata or {}
        ) for e in ev_db
    ]
    
    candidate = GapCandidate(
        candidate_id=c_db.candidate_id,
        topic_indices=c_db.topic_indices or [],
        topic_labels=c_db.topic_labels or [],
        latent_dimension_indices=c_db.latent_dimension_indices or [],
        rarity_score=c_db.rarity_score,
        intersection_score=c_db.intersection_score,
        temporal_score=c_db.temporal_score,
        limitation_score=c_db.limitation_score,
        semantic_support_score=c_db.semantic_support_score,
        final_gap_score=c_db.final_gap_score,
        evidence_count=c_db.evidence_count,
        confidence_level=c_db.confidence_level,
        reasoning_signals=c_db.reasoning_signals or [],
        evidence_references=evidence_references
    )
    
    # Retrieve evidence
    retrieved_chunks = retrieve_evidence(db, analysis_id, candidate, top_k=5, min_similarity=0.0)
    
    # Calculate Semantic Score (S Component)
    if retrieved_chunks:
        # Simple heuristic: max or average similarity
        avg_sim = sum(sim for _, sim in retrieved_chunks) / len(retrieved_chunks)
        candidate.semantic_support_score = float(avg_sim)
        c_db.semantic_support_score = float(avg_sim)
        
        # Recompute final score if necessary - simplified here
        # Normally would import the GapEngineConfig and recalculate properly
        db.commit()
    
    # Build Evidence Packet
    packet = build_evidence_packet(candidate, retrieved_chunks)
    
    # Call Nemotron
    client = NemotronClient()
    try:
        explanation = client.generate_grounded_explanation(GROUNDED_SYSTEM_PROMPT, packet)
    except ValueError as ve:
        msg = str(ve)
        if "not configured" in msg.lower() or "api_key" in msg.lower():
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI response parsing failed: {msg}")
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI service authentication failed: {str(pe)}")
    except TimeoutError as te:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=f"AI service timeout: {str(te)}")
    except ConnectionError as ce:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI service connection failed: {str(ce)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Nemotron generation failed: {str(e)}")
        
    # Persist Chunks and Retrievals
    for rank, (chunk, sim) in enumerate(retrieved_chunks):
        existing_chunk = db.query(RAGChunkDB).filter(RAGChunkDB.chunk_id == chunk.chunk_id).first()
        if not existing_chunk:
            db.add(RAGChunkDB(
                id=str(uuid.uuid4()),
                analysis_id=analysis_id,
                paper_id=chunk.paper_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_field=chunk.source_field,
                year=chunk.year,
                metadata_json={}
            ))
            
        db.add(RAGRetrievalDB(
            id=str(uuid.uuid4()),
            candidate_id=candidate_id,
            chunk_id=chunk.chunk_id,
            similarity_score=sim,
            retrieval_rank=rank
        ))
        
    # Persist Explanation
    db.query(GroundedExplanationDB).filter(GroundedExplanationDB.candidate_id == candidate_id).delete()
    db.add(GroundedExplanationDB(
        id=str(uuid.uuid4()),
        candidate_id=candidate_id,
        model_name=client.model,
        prompt_version="v1",
        candidate_gap_statement=explanation.candidate_gap_statement,
        evidence_summary=explanation.evidence_summary,
        literature_context=explanation.literature_context,
        limitation_evidence=explanation.limitation_evidence,
        temporal_context=explanation.temporal_context,
        counterevidence=explanation.counterevidence,
        evidence_limitations=explanation.evidence_limitations,
        suggested_direction=explanation.suggested_direction,
        grounding_status=explanation.grounding_status
    ))
    
    db.commit()
    
    return explanation

@router.get("/{analysis_id}/gaps/{candidate_id}/explanation", response_model=GroundedExplanation)
def get_explanation_route(analysis_id: str, candidate_id: str, db: Session = Depends(get_db)):
    exp_db = db.query(GroundedExplanationDB).filter(GroundedExplanationDB.candidate_id == candidate_id).first()
    if not exp_db:
        raise HTTPException(status_code=404, detail="No explanation available")
        
    # Retrieve sources
    retrievals = db.query(RAGRetrievalDB).filter(RAGRetrievalDB.candidate_id == candidate_id).order_by(RAGRetrievalDB.retrieval_rank).all()
    source_refs = []
    for r in retrievals:
        chunk = db.query(RAGChunkDB).filter(RAGChunkDB.chunk_id == r.chunk_id).first()
        if chunk:
            # We don't have title in RAGChunkDB directly unless we fetch PaperDB
            from app.models.paper_db import PaperDB
            paper = db.query(PaperDB).filter(PaperDB.id == chunk.paper_id).first()
            source_refs.append(SourceReference(
                paper_id=chunk.paper_id,
                title=paper.title if paper else "Unknown",
                year=chunk.year,
                chunk_id=chunk.chunk_id,
                evidence_field=chunk.source_field,
                similarity_score=r.similarity_score
            ))
            
    return GroundedExplanation(
        candidate_id=exp_db.candidate_id,
        candidate_gap_statement=exp_db.candidate_gap_statement,
        evidence_summary=exp_db.evidence_summary,
        literature_context=exp_db.literature_context,
        limitation_evidence=exp_db.limitation_evidence,
        temporal_context=exp_db.temporal_context,
        counterevidence=exp_db.counterevidence,
        evidence_limitations=exp_db.evidence_limitations,
        suggested_direction=exp_db.suggested_direction,
        grounding_status=exp_db.grounding_status,
        source_references=source_refs
    )

@router.get("/{analysis_id}/gaps/{candidate_id}/sources")
def get_sources_route(analysis_id: str, candidate_id: str, db: Session = Depends(get_db)):
    retrievals = db.query(RAGRetrievalDB).filter(RAGRetrievalDB.candidate_id == candidate_id).order_by(RAGRetrievalDB.retrieval_rank).all()
    results = []
    for r in retrievals:
        chunk = db.query(RAGChunkDB).filter(RAGChunkDB.chunk_id == r.chunk_id).first()
        if chunk:
            results.append({
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "text": chunk.text,
                "similarity_score": r.similarity_score,
                "retrieval_rank": r.retrieval_rank,
                "source_field": chunk.source_field,
                "year": chunk.year
            })
    return results
