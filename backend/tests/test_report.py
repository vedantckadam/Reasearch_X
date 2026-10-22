import pytest
import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.models.paper_db import SessionLocal, Analysis, TopicAnalysis, Topic, GapCandidateDB, GapEvidenceDB, GroundedExplanationDB

client = TestClient(app, raise_server_exceptions=True)

def test_missing_topic_analysis():
    # Test an analysis ID with genuinely empty results (no topics)
    response = client.get("/api/analysis/empty_analysis_id/topics")
    assert response.status_code == 404
    assert response.json()["detail"] == "Topic analysis not found"

def test_report_generation_missing_analysis():
    response = client.get("/api/analysis/invalid_analysis_id/report/json")
    assert response.status_code == 404
    
    response_pdf = client.get("/api/analysis/invalid_analysis_id/report/pdf")
    assert response_pdf.status_code == 404

def test_report_and_topics_with_data():
    db = SessionLocal()
    # Mock data setup
    analysis_id = "test_report_analysis"
    analysis = Analysis(id=analysis_id, status="COMPLETED")
    
    ta2 = TopicAnalysis(id="test_ta_2", analysis_id=analysis_id, num_topics=2, vocabulary_size=10, coherence_score=0.5)
    t = Topic(id="test_topic", topic_analysis_id="test_ta_2", topic_idx=0, label="Test", dominant_papers=1, average_probability=0.9, corpus_share=1.0, top_words=[["word", 0.9]])
    
    # Empty top_words
    t_empty = Topic(id="test_topic_empty", topic_analysis_id="test_ta_2", topic_idx=1, label="Empty", dominant_papers=0, average_probability=0.1, corpus_share=0.0, top_words=[])
    
    # Missing top_words
    t_missing = Topic(id="test_topic_missing", topic_analysis_id="test_ta_2", topic_idx=2, label="Missing", dominant_papers=0, average_probability=0.1, corpus_share=0.0, top_words=None)
    
    # Malformed top_words
    t_malformed = Topic(id="test_topic_malformed", topic_analysis_id="test_ta_2", topic_idx=3, label="Malformed", dominant_papers=0, average_probability=0.1, corpus_share=0.0, top_words=[["word_only"], [None, None]])
    
    cand = GapCandidateDB(id="test_cand_id", analysis_id=analysis_id, candidate_id="cand_1", final_gap_score=0.8, evidence_count=1)
    
    db.add_all([analysis, ta2, t, t_empty, t_missing, t_malformed, cand])
    db.commit()
    db.close()
    
    try:
        # Test topics endpoint
        response = client.get(f"/api/analysis/{analysis_id}/topics")
        assert response.status_code == 200
        data = response.json()
        assert len(data["topics"]) == 4
        assert data["topics"][0]["label"] == "Test"
        assert data["topics"][1]["top_words"] == []
        assert data["topics"][2]["top_words"] is None
        assert data["topics"][3]["top_words"] == [["word_only"], [None, None]]
        
        # Test JSON report endpoint
        response_json = client.get(f"/api/analysis/{analysis_id}/report/json")
        assert response_json.status_code == 200
        report_data = response_json.json()
        assert report_data["analysis_id"] == analysis_id
        assert len(report_data["topics"]) == 4
        assert len(report_data["candidate_gap_signals"]) == 1
        
        # Test PDF report endpoint (should not crash on malformed terms)
        response_pdf = client.get(f"/api/analysis/{analysis_id}/report/pdf")
        assert response_pdf.status_code == 200
        assert response_pdf.headers["content-type"] == "application/pdf"
        assert response_pdf.content.startswith(b"%PDF-")
    finally:
        # Cleanup
        db = SessionLocal()
        db.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == analysis_id).delete()
        db.query(Topic).filter(Topic.topic_analysis_id == "test_ta_2").delete()
        db.query(TopicAnalysis).filter(TopicAnalysis.id == "test_ta_2").delete()
        db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        db.commit()
        db.close()
