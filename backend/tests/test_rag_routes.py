import os
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app
from app.models.paper_db import SessionLocal, Analysis, GapCandidateDB, GroundedExplanationDB
from app.schemas.rag import GroundedExplanation, SourceReference

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_candidate():
    db = SessionLocal()
    # Create test analysis and candidate
    analysis_id = "test-rag-analysis"
    candidate_id = "test-rag-cand-1"

    if not db.query(Analysis).filter(Analysis.id == analysis_id).first():
        db.add(Analysis(id=analysis_id, status="COMPLETED"))

    cand = db.query(GapCandidateDB).filter(GapCandidateDB.candidate_id == candidate_id).first()
    if not cand:
        db.add(GapCandidateDB(
            id=str(uuid.uuid4()),
            analysis_id=analysis_id,
            candidate_id=candidate_id,
            topic_indices=[0, 1],
            topic_labels=["Federated Learning", "Differential Privacy"],
            rarity_score=0.8,
            intersection_score=0.7,
            temporal_score=0.6,
            limitation_score=0.9,
            semantic_support_score=None,
            final_gap_score=0.75,
            evidence_count=2,
            confidence_level="High",
            reasoning_signals=["Sparse Co-occurrence"]
        ))
    db.query(GroundedExplanationDB).filter(GroundedExplanationDB.candidate_id == candidate_id).delete()
    db.commit()
    db.close()

    yield analysis_id, candidate_id

    # Cleanup
    db = SessionLocal()
    db.query(GroundedExplanationDB).filter(GroundedExplanationDB.candidate_id == candidate_id).delete()
    db.commit()
    db.close()

def test_explain_gap_missing_analysis():
    resp = client.post("/api/analysis/nonexistent-analysis/gaps/cand-1/explain")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Analysis not found"

def test_explain_gap_missing_candidate(setup_test_candidate):
    analysis_id, _ = setup_test_candidate
    resp = client.post(f"/api/analysis/{analysis_id}/gaps/nonexistent-cand/explain")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Candidate not found"

def test_get_explanation_not_found(setup_test_candidate):
    analysis_id, candidate_id = setup_test_candidate
    resp = client.get(f"/api/analysis/{analysis_id}/gaps/{candidate_id}/explanation")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "No explanation available"

@patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "LLM_PROVIDER": "openrouter"})
def test_explain_gap_unconfigured_key_returns_503(setup_test_candidate):
    analysis_id, candidate_id = setup_test_candidate
    resp = client.post(f"/api/analysis/{analysis_id}/gaps/{candidate_id}/explain")
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]

@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_test_key", "LLM_PROVIDER": "openrouter"})
@patch("app.services.nemotron_client.NemotronClient.generate_grounded_explanation")
def test_explain_gap_success(mock_generate, setup_test_candidate):
    analysis_id, candidate_id = setup_test_candidate
    mock_explanation = GroundedExplanation(
        candidate_id=candidate_id,
        candidate_gap_statement="Integration of differential privacy in edge federated networks.",
        evidence_summary="Found 2 limitation snippets pointing to communication bottlenecks.",
        literature_context="Recent surveys explore local DP but lack communication bounds.",
        limitation_evidence="Explicit author limitations regarding transmission cost.",
        temporal_context="Topic activity has increased 40% since 2022.",
        counterevidence="Centralized DP solutions exist but do not meet edge constraints.",
        evidence_limitations="Evaluated on 100 corpus papers.",
        suggested_direction="Develop adaptive noise injection schemes for heterogeneous clients.",
        source_references=[
            SourceReference(
                paper_id="paper_1",
                title="Privacy-Preserving Federated Systems",
                year=2023,
                evidence_field="abstract"
            )
        ],
        grounding_status="Success"
    )
    mock_generate.return_value = mock_explanation

    resp = client.post(f"/api/analysis/{analysis_id}/gaps/{candidate_id}/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["candidate_id"] == candidate_id
    assert data["candidate_gap_statement"] == mock_explanation.candidate_gap_statement
    assert data["grounding_status"] == "Success"

    # Verify that GET explanation now returns the persisted explanation
    get_resp = client.get(f"/api/analysis/{analysis_id}/gaps/{candidate_id}/explanation")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data["candidate_gap_statement"] == mock_explanation.candidate_gap_statement

@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_test_key", "LLM_PROVIDER": "openrouter"})
@patch("app.services.nemotron_client.NemotronClient.generate_grounded_explanation")
def test_explain_gap_provider_timeout_returns_504(mock_generate, setup_test_candidate):
    analysis_id, candidate_id = setup_test_candidate
    mock_generate.side_effect = TimeoutError("Request timed out after 60s")

    resp = client.post(f"/api/analysis/{analysis_id}/gaps/{candidate_id}/explain")
    assert resp.status_code == 504
    assert "timeout" in resp.json()["detail"].lower()

@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_test_key", "LLM_PROVIDER": "openrouter"})
@patch("app.services.nemotron_client.NemotronClient.generate_grounded_explanation")
def test_explain_gap_provider_auth_error_returns_502(mock_generate, setup_test_candidate):
    analysis_id, candidate_id = setup_test_candidate
    mock_generate.side_effect = PermissionError("openrouter authentication failed (HTTP 401)")

    resp = client.post(f"/api/analysis/{analysis_id}/gaps/{candidate_id}/explain")
    assert resp.status_code == 502
    assert "authentication failed" in resp.json()["detail"]
