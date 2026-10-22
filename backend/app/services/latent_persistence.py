import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.models.paper_db import (
    LatentDimensionDB, LatentTopicLoadingDB, TemporalTopicDB, 
    TemporalIntersectionDB, LatentAnalysisSummaryDB, DocumentTopic, Topic
)

def persist_latent_dimension_analysis(session: Session, analysis_id: str, results: Dict[str, Any]):
    # Delete existing
    session.query(LatentTopicLoadingDB).filter(
        LatentTopicLoadingDB.latent_dimension_id.in_(
            session.query(LatentDimensionDB.id).filter(LatentDimensionDB.analysis_id == analysis_id)
        )
    ).delete(synchronize_session=False)
    session.query(LatentDimensionDB).filter(LatentDimensionDB.analysis_id == analysis_id).delete()
    session.query(LatentAnalysisSummaryDB).filter(LatentAnalysisSummaryDB.analysis_id == analysis_id).delete()

    summary = LatentAnalysisSummaryDB(
        id=f"las_{analysis_id}",
        analysis_id=analysis_id,
        total_dimensions=results.get("total_dimensions", 0),
        rank=results.get("rank", 0),
        energy_80_dimensions=results.get("energy_80_dimensions", []),
        energy_90_dimensions=results.get("energy_90_dimensions", []),
        low_energy_threshold=results.get("low_energy_threshold", 0.0),
        reconstruction_error=results.get("reconstruction_error", 0.0)
    )
    session.add(summary)
    
    dimensions_data = results.get("dimensions", [])
    for dim in dimensions_data:
        dim_id = f"ld_{analysis_id}_{dim['dimension_index']}"
        dim_db = LatentDimensionDB(
            id=dim_id,
            analysis_id=analysis_id,
            dimension_index=dim["dimension_index"],
            singular_value=dim["singular_value"],
            squared_singular_value=dim["squared_singular_value"],
            latent_energy_ratio=dim["latent_energy_ratio"],
            cumulative_energy=dim["cumulative_energy"],
            eigenvalue=dim["eigenvalue"],
            normalized_eigen_contribution=dim["normalized_eigen_contribution"],
            cumulative_eigen_contribution=dim["cumulative_eigen_contribution"],
            is_low_energy=dim["is_low_energy"]
        )
        session.add(dim_db)
        
        for load in dim.get("dominant_topics", []):
            load_db = LatentTopicLoadingDB(
                id=f"ltl_{dim_id}_{load['topic_index']}",
                latent_dimension_id=dim_id,
                topic_index=load["topic_index"],
                loading=load["loading"],
                absolute_loading=load["absolute_loading"],
                rank=load["rank"]
            )
            session.add(load_db)
            
    session.commit()

def persist_temporal_analysis(session: Session, analysis_id: str, temporal_data: List[Dict[str, Any]]):
    session.query(TemporalTopicDB).filter(TemporalTopicDB.analysis_id == analysis_id).delete()
    for item in temporal_data:
        t_id = f"tt_{analysis_id}_{item['year']}_{item['topic_index']}"
        tdb = TemporalTopicDB(
            id=t_id,
            analysis_id=analysis_id,
            year=item['year'],
            topic_index=item['topic_index'],
            paper_count=item['paper_count'],
            dominant_count=item['dominant_count'],
            dominant_percentage=item['dominant_percentage'],
            probability_mass=item['probability_mass'],
            average_probability=item['average_probability'],
            trend=item['trend']
        )
        meta = item.get('temporal_coverage_metadata', {})
        tdb.first_observed_year = meta.get('first_observed_year')
        tdb.last_observed_year = meta.get('last_observed_year')
        tdb.number_of_years_present = meta.get('number_of_years_present')
        session.add(tdb)
    session.commit()

def persist_temporal_intersections(session: Session, analysis_id: str, intersections_data: List[Dict[str, Any]]):
    session.query(TemporalIntersectionDB).filter(TemporalIntersectionDB.analysis_id == analysis_id).delete()
    for item in intersections_data:
        topic_pair = item["topic_pair"]
        t_i = min(topic_pair[0], topic_pair[1])
        t_j = max(topic_pair[0], topic_pair[1])
        i_id = f"ti_{analysis_id}_{item['year']}_{t_i}_{t_j}"
        idb = TemporalIntersectionDB(
            id=i_id,
            analysis_id=analysis_id,
            year=item['year'],
            topic_i=t_i,
            topic_j=t_j,
            support_count=item['support_count'],
            normalized_weighted_value=item['normalized_weighted_value'],
            weighted_intersection=item['weighted_intersection'],
            threshold=item['threshold']
        )
        session.add(idb)
    session.commit()

def persist_complete_latent_temporal_analysis(
    session: Session, analysis_id: str, 
    latent_results: Dict[str, Any], 
    temporal_results: List[Dict[str, Any]], 
    intersections_results: List[Dict[str, Any]]
):
    try:
        persist_latent_dimension_analysis(session, analysis_id, latent_results)
        persist_temporal_analysis(session, analysis_id, temporal_results)
        persist_temporal_intersections(session, analysis_id, intersections_results)
    except Exception as e:
        session.rollback()
        raise e
