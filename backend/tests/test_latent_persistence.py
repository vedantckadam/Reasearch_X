import pytest
import numpy as np
from sqlalchemy.orm import Session
from app.models.paper_db import (
    SessionLocal, Analysis, TopicAnalysis, LatentDimensionDB, 
    LatentTopicLoadingDB, TemporalTopicDB, TemporalIntersectionDB, LatentAnalysisSummaryDB
)
from app.services.latent_persistence import (
    persist_latent_dimension_analysis,
    persist_temporal_analysis,
    persist_temporal_intersections,
    persist_complete_latent_temporal_analysis
)
from app.services.latent_results import (
    get_latent_dimensions,
    get_latent_summary,
    get_temporal_topics,
    get_temporal_intersections,
    get_complete_latent_temporal_results
)

@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    
    analysis = Analysis(id="test_analysis_latent")
    db.add(analysis)
    
    ta = TopicAnalysis(id="test_ta_latent", analysis_id="test_analysis_latent", num_topics=2, vocabulary_size=10, coherence_score=0.5)
    db.add(ta)
    db.commit()
    
    yield db
    
    db.query(LatentTopicLoadingDB).delete()
    db.query(LatentDimensionDB).delete()
    db.query(TemporalTopicDB).delete()
    db.query(TemporalIntersectionDB).delete()
    db.query(LatentAnalysisSummaryDB).delete()
    db.query(TopicAnalysis).filter(TopicAnalysis.id == "test_ta_latent").delete()
    db.query(Analysis).filter(Analysis.id == "test_analysis_latent").delete()
    db.commit()
    db.close()

def test_latent_dimension_persistence(db_session: Session):
    results = {
        "total_dimensions": 2,
        "rank": 2,
        "energy_80_dimensions": [0],
        "energy_90_dimensions": [0, 1],
        "low_energy_threshold": 0.05,
        "reconstruction_error": 0.01,
        "dimensions": [
            {
                "dimension_index": 0,
                "singular_value": 5.0,
                "squared_singular_value": 25.0,
                "latent_energy_ratio": 0.8,
                "cumulative_energy": 0.8,
                "eigenvalue": 5.0,
                "normalized_eigen_contribution": 0.8,
                "cumulative_eigen_contribution": 0.8,
                "is_low_energy": False,
                "dominant_topics": [
                    {"topic_index": 0, "loading": 0.9, "absolute_loading": 0.9, "rank": 1}
                ]
            }
        ]
    }
    
    persist_latent_dimension_analysis(db_session, "test_analysis_latent", results)
    
    # Verify summary
    summary = get_latent_summary(db_session, "test_analysis_latent")
    assert summary["total_dimensions"] == 2
    
    # Verify dimensions
    dims = get_latent_dimensions(db_session, "test_analysis_latent")
    assert len(dims) == 1
    assert dims[0]["dimension_index"] == 0
    assert dims[0]["dominant_topics"][0]["loading"] == 0.9

def test_temporal_persistence(db_session: Session):
    temporal_data = [
        {
            "year": 2021,
            "topic_index": 0,
            "paper_count": 10,
            "dominant_count": 5,
            "dominant_percentage": 0.5,
            "probability_mass": 7.5,
            "average_probability": 0.75,
            "trend": "increasing",
            "temporal_coverage_metadata": {
                "first_observed_year": 2021,
                "last_observed_year": 2022,
                "number_of_years_present": 2
            }
        }
    ]
    
    persist_temporal_analysis(db_session, "test_analysis_latent", temporal_data)
    topics = get_temporal_topics(db_session, "test_analysis_latent")
    assert len(topics) == 1
    assert topics[0]["trend"] == "increasing"
    assert topics[0]["temporal_coverage_metadata"]["first_observed_year"] == 2021

def test_temporal_intersection_persistence(db_session: Session):
    intersection_data = [
        {
            "year": 2021,
            "topic_pair": [0, 1],
            "support_count": 3,
            "weighted_intersection": 2.5,
            "normalized_weighted_value": 0.8,
            "threshold": 0.20
        }
    ]
    
    persist_temporal_intersections(db_session, "test_analysis_latent", intersection_data)
    inters = get_temporal_intersections(db_session, "test_analysis_latent")
    assert len(inters) == 1
    assert inters[0]["support_count"] == 3
    assert inters[0]["topic_pair"] == [0, 1]

def test_complete_reconstruction(db_session: Session):
    res = get_complete_latent_temporal_results(db_session, "test_analysis_latent")
    assert res["analysis_id"] == "test_analysis_latent"
    assert "summary" in res["latent"]
    assert "dimensions" in res["latent"]
    assert len(res["temporal"]) == 1
    assert len(res["temporal_intersections"]) == 1

def test_duplicate_safe_rerun(db_session: Session):
    intersection_data = [
        {
            "year": 2021,
            "topic_pair": [0, 1],
            "support_count": 5,
            "weighted_intersection": 3.0,
            "normalized_weighted_value": 0.9,
            "threshold": 0.20
        }
    ]
    persist_temporal_intersections(db_session, "test_analysis_latent", intersection_data)
    inters = get_temporal_intersections(db_session, "test_analysis_latent")
    assert len(inters) == 1
    assert inters[0]["support_count"] == 5
