from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from app.models.paper_db import SessionLocal, Analysis, GapCandidateDB, GapEvidenceDB
from app.schemas.gap import GapRunResponse, GapCandidate, GapEvidence, GapEngineConfig
from app.services.gap_engine import run_gap_engine, persist_gap_candidates

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/{analysis_id}/gap/run", response_model=GapRunResponse)
def run_gap_analysis_route(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    config = GapEngineConfig()
    try:
        response = run_gap_engine(db, analysis_id, config)
        persist_gap_candidates(db, analysis_id, response)
        return response
    except ValueError as e:
        raise HTTPException(status_code=409, detail=f"Gap analysis failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap analysis encountered an error: {str(e)}")

@router.get("/{analysis_id}/gaps", response_model=List[GapCandidate])
def get_gaps_route(analysis_id: str, db: Session = Depends(get_db)):
    cands_db = db.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == analysis_id).order_by(GapCandidateDB.final_gap_score.desc(), GapCandidateDB.candidate_id).all()
    
    result = []
    for c in cands_db:
        result.append(GapCandidate(
            candidate_id=c.candidate_id,
            topic_indices=c.topic_indices or [],
            topic_labels=c.topic_labels or [],
            latent_dimension_indices=c.latent_dimension_indices or [],
            rarity_score=c.rarity_score,
            intersection_score=c.intersection_score,
            temporal_score=c.temporal_score,
            limitation_score=c.limitation_score,
            semantic_support_score=c.semantic_support_score,
            final_gap_score=c.final_gap_score,
            evidence_count=c.evidence_count,
            confidence_level=c.confidence_level,
            reasoning_signals=c.reasoning_signals or []
        ))
    return result

@router.get("/{analysis_id}/gaps/{candidate_id}", response_model=GapCandidate)
def get_gap_route(analysis_id: str, candidate_id: str, db: Session = Depends(get_db)):
    c = db.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == analysis_id, GapCandidateDB.candidate_id == candidate_id).first()
    if not c:
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
    
    return GapCandidate(
        candidate_id=c.candidate_id,
        topic_indices=c.topic_indices or [],
        topic_labels=c.topic_labels or [],
        latent_dimension_indices=c.latent_dimension_indices or [],
        rarity_score=c.rarity_score,
        intersection_score=c.intersection_score,
        temporal_score=c.temporal_score,
        limitation_score=c.limitation_score,
        semantic_support_score=c.semantic_support_score,
        final_gap_score=c.final_gap_score,
        evidence_count=c.evidence_count,
        confidence_level=c.confidence_level,
        reasoning_signals=c.reasoning_signals or [],
        evidence_references=evidence_references
    )

@router.get("/{analysis_id}/gaps/{candidate_id}/evidence", response_model=List[GapEvidence])
def get_gap_evidence_route(analysis_id: str, candidate_id: str, db: Session = Depends(get_db)):
    c = db.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == analysis_id, GapCandidateDB.candidate_id == candidate_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Candidate not found")
        
    ev_db = db.query(GapEvidenceDB).filter(GapEvidenceDB.candidate_id == candidate_id).all()
    return [
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
