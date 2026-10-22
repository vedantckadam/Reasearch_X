from fastapi.testclient import TestClient
from app.models.paper_db import SessionLocal, Analysis, TopicAnalysis, Topic, DocumentTopic, PaperDB
from main import app
import pytest
import uuid

client = TestClient(app)

global_test_analysis_id = str(uuid.uuid4())
global_test_ta_id = str(uuid.uuid4())

@pytest.fixture(scope="module")
def setup_db():
    db = SessionLocal()
    
    analysis = Analysis(id=global_test_analysis_id)
    db.add(analysis)
    
    ta = TopicAnalysis(id=global_test_ta_id, analysis_id=global_test_analysis_id, num_topics=2, vocabulary_size=10, coherence_score=0.5)
    db.add(ta)
    db.commit()
    
    t1 = Topic(id=str(uuid.uuid4()), topic_analysis_id=global_test_ta_id, topic_idx=0, label="T1")
    t2 = Topic(id=str(uuid.uuid4()), topic_analysis_id=global_test_ta_id, topic_idx=1, label="T2")
    db.add_all([t1, t2])
    db.commit()
    
    p1_id = str(uuid.uuid4())
    p1 = PaperDB(id=p1_id, analysis_id=global_test_analysis_id, year=2021)
    db.add(p1)
    db.commit()
    
    dt1 = DocumentTopic(id=str(uuid.uuid4()), topic_analysis_id=global_test_ta_id, paper_id=p1_id, topic_idx=0, probability=0.8)
    dt2 = DocumentTopic(id=str(uuid.uuid4()), topic_analysis_id=global_test_ta_id, paper_id=p1_id, topic_idx=1, probability=0.2)
    db.add_all([dt1, dt2])
    db.commit()
    
    yield db
    
    # Cleanup
    db.query(DocumentTopic).filter(DocumentTopic.topic_analysis_id == global_test_ta_id).delete()
    db.query(Topic).filter(Topic.topic_analysis_id == global_test_ta_id).delete()
    db.query(PaperDB).filter(PaperDB.analysis_id == global_test_analysis_id).delete()
    db.query(TopicAnalysis).filter(TopicAnalysis.id == global_test_ta_id).delete()
    db.query(Analysis).filter(Analysis.id == global_test_analysis_id).delete()
    db.commit()
    db.close()

def test_run_latent_analysis_invalid(setup_db):
    response = client.post("/api/analysis/invalid_id/latent/run")
    assert response.status_code == 404

def test_run_latent_analysis(setup_db):
    response = client.post(f"/api/analysis/{global_test_analysis_id}/latent/run")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

def test_get_latent_summary(setup_db):
    response = client.get(f"/api/analysis/{global_test_analysis_id}/latent/summary")
    assert response.status_code == 200
    assert response.json()["total_dimensions"] == 1

def test_get_temporal_topics(setup_db):
    response = client.get(f"/api/analysis/{global_test_analysis_id}/temporal/topics")
    assert response.status_code == 200
    assert len(response.json()["topics"]) > 0

def test_get_temporal_intersections(setup_db):
    response = client.get(f"/api/analysis/{global_test_analysis_id}/temporal/intersections")
    assert response.status_code == 200

def test_get_combined(setup_db):
    response = client.get(f"/api/analysis/{global_test_analysis_id}/latent-temporal")
    assert response.status_code == 200
    assert "latent" in response.json()
    assert "temporal" in response.json()

def test_get_latent_summary_not_found(setup_db):
    response = client.get("/api/analysis/invalid_id/latent/summary")
    assert response.status_code == 404

def test_get_temporal_topics_not_found(setup_db):
    response = client.get("/api/analysis/invalid_id/temporal/topics")
    assert response.status_code == 200
    assert response.json() == {"analysis_id": "invalid_id", "topics": []}

def test_get_combined_not_found(setup_db):
    response = client.get("/api/analysis/invalid_id/latent-temporal")
    assert response.status_code == 404
