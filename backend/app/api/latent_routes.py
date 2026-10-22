from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from app.models.paper_db import SessionLocal, Analysis, TopicAnalysis, DocumentTopic, PaperDB
from app.schemas.latent import (
    LatentResultResponse, TemporalTopicResponse, TemporalIntersectionResponse,
    CombinedLatentTemporalResponse, RunAnalysisResponse, LatentAnalysisSummary
)
from app.services.latent_persistence import persist_complete_latent_temporal_analysis
from app.services.latent_results import (
    get_latent_dimensions, get_latent_summary,
    get_temporal_topics, get_temporal_intersections,
    get_complete_latent_temporal_results
)
from app.services.latent_analysis import load_document_topic_matrix, run_latent_math_analysis
from app.services.temporal_analysis import calculate_temporal_analysis, calculate_temporal_intersections

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/{analysis_id}/latent/run", response_model=RunAnalysisResponse)
def run_latent_analysis_route(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    ta = db.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).first()
    if not ta:
        raise HTTPException(status_code=404, detail="Topic Analysis not found")
        
    try:
        theta, paper_ids, topic_ids, topic_labels = load_document_topic_matrix(db, ta.id)
    except Exception as e:
        raise HTTPException(status_code=409, detail=f"Cannot load document topic matrix: {str(e)}")
        
    try:
        latent_results = run_latent_math_analysis(theta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Latent mathematical analysis failed: {str(e)}")
    
    # Prepare paper data for temporal analysis
    papers = db.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).all()
    paper_year_map = {p.id: p.year for p in papers}
    
    paper_data = []
    for i, pid in enumerate(paper_ids):
        paper_data.append({
            "paper_id": pid,
            "year": paper_year_map.get(pid),
            "topics": theta[i].tolist()
        })
        
    num_topics = len(topic_ids)
    raw_temporal = calculate_temporal_analysis(paper_data, num_topics)
    raw_intersections = calculate_temporal_intersections(paper_data, num_topics, threshold=0.20)
    
    # Transform raw_temporal to the expected list format
    temporal_formatted = []
    if raw_temporal and "topics" in raw_temporal:
        for t_idx, t_data in raw_temporal["topics"].items():
            for tl in t_data.get("timeline", []):
                temporal_formatted.append({
                    "year": tl["year"],
                    "topic_index": t_idx,
                    "paper_count": tl["paper_count"],
                    "dominant_count": int(tl["dominant_percentage"] * tl["paper_count"]),
                    "dominant_percentage": tl["dominant_percentage"],
                    "probability_mass": tl["probability_mass"],
                    "average_probability": tl["average_probability"],
                    "trend": t_data["trend"],
                    "temporal_coverage_metadata": {
                        "first_observed_year": t_data["first_year_observed"],
                        "last_observed_year": t_data["last_year_observed"],
                        "number_of_years_present": t_data["number_of_years_present"]
                    }
                })
                
    # Transform raw_intersections
    intersections_formatted = []
    for pair_data in raw_intersections:
        t_a = pair_data["topic_pair"]["topic_a_id"]
        t_b = pair_data["topic_pair"]["topic_b_id"]
        for tl in pair_data["timeline"]:
            intersections_formatted.append({
                "year": tl["year"],
                "topic_pair": [t_a, t_b],
                "support_count": tl["support_count"],
                "weighted_intersection": tl["weighted_intersection"],
                "normalized_weighted_value": tl["weighted_intersection"], # Assuming same for now if not computed
                "threshold": 0.20
            })
            
    persist_complete_latent_temporal_analysis(
        db, analysis_id, latent_results, temporal_formatted, intersections_formatted
    )
    
    # Calculate unique temporal years
    years = list(set([item["year"] for item in temporal_formatted]))
    
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "latent_dimensions": latent_results.get("total_dimensions", 0),
        "energy_80_dimensions": latent_results.get("energy_80_dimensions", []),
        "energy_90_dimensions": latent_results.get("energy_90_dimensions", []),
        "temporal_years": sorted(years),
        "low_energy_dimensions": [d["dimension_index"] for d in latent_results.get("dimensions", []) if d.get("is_low_energy")]
    }

@router.get("/{analysis_id}/latent", response_model=LatentResultResponse)
def get_latent_route(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    dims = get_latent_dimensions(db, analysis_id)
    if not dims:
        raise HTTPException(status_code=404, detail="Latent results not found")
    return {"analysis_id": analysis_id, "dimensions": dims}

@router.get("/{analysis_id}/latent/summary", response_model=LatentAnalysisSummary)
def get_latent_summary_route(analysis_id: str, db: Session = Depends(get_db)):
    summary = get_latent_summary(db, analysis_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Latent summary not found")
    return summary

@router.get("/{analysis_id}/temporal/topics", response_model=TemporalTopicResponse)
def get_temporal_topics_route(analysis_id: str, db: Session = Depends(get_db)):
    topics = get_temporal_topics(db, analysis_id)
    return {"analysis_id": analysis_id, "topics": topics}

@router.get("/{analysis_id}/temporal/intersections", response_model=TemporalIntersectionResponse)
def get_temporal_intersections_route(analysis_id: str, db: Session = Depends(get_db)):
    inters = get_temporal_intersections(db, analysis_id)
    return {"analysis_id": analysis_id, "intersections": inters}

@router.get("/{analysis_id}/latent-temporal", response_model=CombinedLatentTemporalResponse)
def get_combined_route(analysis_id: str, db: Session = Depends(get_db)):
    res = get_complete_latent_temporal_results(db, analysis_id)
    if not res.get("latent", {}).get("summary"):
        raise HTTPException(status_code=404, detail="Results not found")
    return res
