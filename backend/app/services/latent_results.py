from sqlalchemy.orm import Session
from typing import Dict, List, Any
from app.models.paper_db import (
    LatentDimensionDB, LatentTopicLoadingDB, TemporalTopicDB, 
    TemporalIntersectionDB, LatentAnalysisSummaryDB
)

def get_latent_summary(session: Session, analysis_id: str) -> Dict[str, Any]:
    summary = session.query(LatentAnalysisSummaryDB).filter(LatentAnalysisSummaryDB.analysis_id == analysis_id).first()
    if not summary:
        return {}
    return {
        "total_dimensions": summary.total_dimensions,
        "rank": summary.rank,
        "energy_80_dimensions": summary.energy_80_dimensions,
        "energy_90_dimensions": summary.energy_90_dimensions,
        "low_energy_threshold": summary.low_energy_threshold,
        "reconstruction_error": summary.reconstruction_error,
        "low_energy_dimension_indices": [
            d.dimension_index for d in session.query(LatentDimensionDB.dimension_index).filter(
                LatentDimensionDB.analysis_id == analysis_id, LatentDimensionDB.is_low_energy == True
            ).all()
        ]
    }

def get_latent_dimensions(session: Session, analysis_id: str) -> List[Dict[str, Any]]:
    dims = session.query(LatentDimensionDB).filter(LatentDimensionDB.analysis_id == analysis_id).order_by(LatentDimensionDB.dimension_index).all()
    results = []
    for dim in dims:
        loadings = session.query(LatentTopicLoadingDB).filter(LatentTopicLoadingDB.latent_dimension_id == dim.id).order_by(LatentTopicLoadingDB.rank).all()
        dim_data = {
            "dimension_index": dim.dimension_index,
            "singular_value": dim.singular_value,
            "squared_singular_value": dim.squared_singular_value,
            "latent_energy_ratio": dim.latent_energy_ratio,
            "cumulative_energy": dim.cumulative_energy,
            "eigenvalue": dim.eigenvalue,
            "normalized_eigen_contribution": dim.normalized_eigen_contribution,
            "cumulative_eigen_contribution": dim.cumulative_eigen_contribution,
            "is_low_energy": dim.is_low_energy,
            "dominant_topics": [
                {
                    "topic_index": load.topic_index,
                    "loading": load.loading,
                    "absolute_loading": load.absolute_loading,
                    "rank": load.rank
                } for load in loadings
            ]
        }
        results.append(dim_data)
    return results

def get_temporal_topics(session: Session, analysis_id: str) -> List[Dict[str, Any]]:
    topics = session.query(TemporalTopicDB).filter(TemporalTopicDB.analysis_id == analysis_id).order_by(TemporalTopicDB.year, TemporalTopicDB.topic_index).all()
    return [
        {
            "year": t.year,
            "topic_index": t.topic_index,
            "paper_count": t.paper_count,
            "dominant_count": t.dominant_count,
            "dominant_percentage": t.dominant_percentage,
            "probability_mass": t.probability_mass,
            "average_probability": t.average_probability,
            "trend": t.trend,
            "temporal_coverage_metadata": {
                "first_observed_year": t.first_observed_year,
                "last_observed_year": t.last_observed_year,
                "number_of_years_present": t.number_of_years_present
            }
        } for t in topics
    ]

def get_temporal_intersections(session: Session, analysis_id: str) -> List[Dict[str, Any]]:
    intersections = session.query(TemporalIntersectionDB).filter(TemporalIntersectionDB.analysis_id == analysis_id).order_by(TemporalIntersectionDB.year, TemporalIntersectionDB.topic_i, TemporalIntersectionDB.topic_j).all()
    return [
        {
            "year": inter.year,
            "topic_pair": [inter.topic_i, inter.topic_j],
            "support_count": inter.support_count,
            "weighted_intersection": inter.weighted_intersection,
            "normalized_weighted_value": inter.normalized_weighted_value,
            "threshold": inter.threshold
        } for inter in intersections
    ]

def get_complete_latent_temporal_results(session: Session, analysis_id: str) -> Dict[str, Any]:
    return {
        "analysis_id": analysis_id,
        "latent": {
            "summary": get_latent_summary(session, analysis_id),
            "dimensions": get_latent_dimensions(session, analysis_id)
        },
        "temporal": get_temporal_topics(session, analysis_id),
        "temporal_intersections": get_temporal_intersections(session, analysis_id)
    }
