import os
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app
from app.models.paper_db import (
    SessionLocal,
    Analysis,
    PaperDB,
    TopicAnalysis,
    Topic,
    GapCandidateDB,
)

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_chat_analysis():
    db = SessionLocal()
    analysis_id = "test-chat-analysis-id"
    
    # 1. Clean or create Analysis
    existing = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not existing:
        db.add(Analysis(
            id=analysis_id,
            status="COMPLETED",
            seed_text="Investigating federated differential privacy in mobile edge computing environments."
        ))

    # 2. Add sample papers
    if not db.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).first():
        db.add(PaperDB(
            id=str(uuid.uuid4()),
            analysis_id=analysis_id,
            title="Privacy in Edge Intelligence",
            year=2023,
        ))
        db.add(PaperDB(
            id=str(uuid.uuid4()),
            analysis_id=analysis_id,
            title="Communication Constraints in Mobile Federated Networks",
            year=2024,
        ))

    # 3. Add topic analysis
    ta = db.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).first()
    if not ta:
        ta = TopicAnalysis(
            id=str(uuid.uuid4()),
            analysis_id=analysis_id,
            num_topics=2,
            vocabulary_size=50,
            coherence_score=0.45,
        )
        db.add(ta)
        db.flush()
        db.add(Topic(
            id=str(uuid.uuid4()),
            topic_analysis_id=ta.id,
            topic_idx=0,
            label="Federated Optimization",
            dominant_papers=10,
            top_words=["federated", "edge", "gradient", "aggregation"]
        ))
        db.add(Topic(
            id=str(uuid.uuid4()),
            topic_analysis_id=ta.id,
            topic_idx=1,
            label="Differential Privacy",
            dominant_papers=8,
            top_words=["privacy", "noise", "epsilon", "budget"]
        ))

    # 4. Add gap candidates
    if not db.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == analysis_id).first():
        db.add(GapCandidateDB(
            id=str(uuid.uuid4()),
            analysis_id=analysis_id,
            candidate_id=f"{analysis_id}-cand-1",
            topic_indices=[0, 1],
            topic_labels=["Federated Optimization", "Differential Privacy"],
            rarity_score=0.88,
            intersection_score=0.92,
            final_gap_score=0.85,
            confidence_level="High",
        ))

    db.commit()
    db.close()

    yield analysis_id

    # Cleanup
    db = SessionLocal()
    db.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == analysis_id).delete()
    db.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).delete()
    t_ids = [t.id for t in db.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).all()]
    for tid in t_ids:
        db.query(Topic).filter(Topic.topic_analysis_id == tid).delete()
    db.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).delete()
    db.query(Analysis).filter(Analysis.id == analysis_id).delete()
    db.commit()
    db.close()


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key", "LLM_PROVIDER": "openrouter"})
@patch("app.services.nemotron_client.NemotronClient.post_chat_completion")
def test_chat_without_analysis_id(mock_post):
    mock_post.return_value = {
        "content": "ResearchGap-X analyzes literature using multi-signal SVD and topic modeling.",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "finish_reason": "stop",
    }

    resp = client.post("/api/chat", json={
        "message": "How does research gap detection work?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "ResearchGap-X analyzes" in data["reply"]
    assert data["grounded_context_used"] is False
    assert data["analysis_id"] is None


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key", "LLM_PROVIDER": "openrouter"})
@patch("app.services.nemotron_client.NemotronClient.post_chat_completion")
def test_chat_with_valid_analysis_id(mock_post, setup_chat_analysis):
    analysis_id = setup_chat_analysis
    mock_post.return_value = {
        "content": "Your active analysis contains 2 papers focusing on Federated Optimization and Differential Privacy.",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "finish_reason": "stop",
    }

    resp = client.post("/api/chat", json={
        "message": "Summarize my uploaded paper and findings",
        "analysis_id": analysis_id,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "Federated Optimization" in data["reply"]
    assert data["grounded_context_used"] is True
    assert data["analysis_id"] == analysis_id

    # Verify context passed to model
    call_args = mock_post.call_args.kwargs
    messages = call_args["messages"]
    system_content = messages[0]["content"]
    assert "Privacy in Edge Intelligence" in system_content
    assert "Differential Privacy" in system_content
    assert f"{analysis_id}-cand-1" in system_content


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key", "LLM_PROVIDER": "openrouter"})
@patch("app.services.nemotron_client.NemotronClient.post_chat_completion")
def test_chat_with_nonexistent_analysis_id(mock_post):
    mock_post.return_value = {
        "content": "I do not see that analysis ID, but I can answer general methodology questions.",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "finish_reason": "stop",
    }

    resp = client.post("/api/chat", json={
        "message": "What gaps did you find?",
        "analysis_id": "nonexistent-analysis-uuid",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["grounded_context_used"] is False


def test_chat_empty_message_returns_400():
    resp = client.post("/api/chat", json={
        "message": "    ",
    })
    assert resp.status_code == 400
    assert "cannot be empty" in resp.json()["detail"]


def test_chat_message_exceeding_max_length_returns_422():
    too_long = "A" * 2500
    resp = client.post("/api/chat", json={
        "message": too_long,
    })
    assert resp.status_code == 422


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "LLM_PROVIDER": "openrouter"})
def test_chat_unconfigured_api_key_returns_503():
    resp = client.post("/api/chat", json={
        "message": "Hello assistant",
    })
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key", "LLM_PROVIDER": "openrouter"})
@patch("app.services.nemotron_client.NemotronClient.post_chat_completion")
def test_chat_timeout_returns_504(mock_post):
    mock_post.side_effect = TimeoutError("Request timed out after 60s")
    resp = client.post("/api/chat", json={
        "message": "Hello assistant",
    })
    assert resp.status_code == 504
    assert "timeout" in resp.json()["detail"].lower()


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key", "LLM_PROVIDER": "openrouter"})
@patch("app.services.nemotron_client.NemotronClient.post_chat_completion")
def test_chat_auth_error_returns_502(mock_post):
    mock_post.side_effect = PermissionError("openrouter authentication failed (HTTP 401)")
    resp = client.post("/api/chat", json={
        "message": "Hello assistant",
    })
    assert resp.status_code == 502
    assert "authentication failed" in resp.json()["detail"]


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key", "LLM_PROVIDER": "openrouter"})
@patch("app.services.nemotron_client.NemotronClient.post_chat_completion")
def test_chat_history_is_limited(mock_post):
    mock_post.return_value = {
        "content": "Conversation acknowledged.",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    }

    # Pass 10 history items
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
        for i in range(10)
    ]

    resp = client.post("/api/chat", json={
        "message": "Latest query",
        "history": history,
    })
    assert resp.status_code == 200

    call_args = mock_post.call_args.kwargs
    messages = call_args["messages"]
    # 1 system message + max 6 history messages + 1 user message = 8 messages max
    assert len(messages) <= 8
    # Ensure oldest messages were pruned
    contents = [m["content"] for m in messages]
    assert "Message 0" not in contents
    assert "Message 9" in contents
    assert "Latest query" in contents
