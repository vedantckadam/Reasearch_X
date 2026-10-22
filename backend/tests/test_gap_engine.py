import pytest
from app.schemas.gap import GapEngineConfig, GapCandidate, GapEvidence
from app.services.gap_engine import (
    calculate_rarity_score, calculate_intersection_sparsity,
    calculate_temporal_score, calculate_limitation_score,
    calculate_semantic_score, run_gap_engine, persist_gap_candidates
)
from app.models.paper_db import (
    SessionLocal, Analysis, TopicAnalysis, Topic, DocumentTopic,
    PaperDB, TemporalTopicDB, TemporalIntersectionDB, GapCandidateDB, GapEvidenceDB
)
import uuid

@pytest.fixture(scope="function")
def db_session():
    db = SessionLocal()
    import uuid
    aid = str(uuid.uuid4())
    ta_id = str(uuid.uuid4())
    
    analysis = Analysis(id=aid, status="COMPLETED")
    db.add(analysis)
    
    ta = TopicAnalysis(id=ta_id, analysis_id=aid, num_topics=2)
    db.add(ta)
    
    t1 = Topic(id=str(uuid.uuid4()), topic_analysis_id=ta_id, topic_idx=0, label="T1", corpus_share=0.5)
    t2 = Topic(id=str(uuid.uuid4()), topic_analysis_id=ta_id, topic_idx=1, label="T2", corpus_share=0.5)
    db.add_all([t1, t2])
    
    p1_id = str(uuid.uuid4())
    p2_id = str(uuid.uuid4())
    p1 = PaperDB(id=p1_id, analysis_id=aid, title="Test Deep Learning", abstract="This paper has a limitation in scope.", year=2021)
    p2 = PaperDB(id=p2_id, analysis_id=aid, title="Test NLP", abstract="Works perfectly well.", year=2022)
    db.add_all([p1, p2])
    
    dt1 = DocumentTopic(id=str(uuid.uuid4()), topic_analysis_id=ta_id, paper_id=p1_id, topic_idx=0, probability=0.8)
    dt2 = DocumentTopic(id=str(uuid.uuid4()), topic_analysis_id=ta_id, paper_id=p2_id, topic_idx=1, probability=0.9)
    db.add_all([dt1, dt2])
    
    tt1 = TemporalTopicDB(id=str(uuid.uuid4()), analysis_id=aid, year=2021, topic_index=0, paper_count=1, trend="emerging")
    tt2 = TemporalTopicDB(id=str(uuid.uuid4()), analysis_id=aid, year=2022, topic_index=0, paper_count=2, trend="growing")
    tt3 = TemporalTopicDB(id=str(uuid.uuid4()), analysis_id=aid, year=2023, topic_index=0, paper_count=1, trend="declining")
    db.add_all([tt1, tt2, tt3])
    
    ti1 = TemporalIntersectionDB(id=str(uuid.uuid4()), analysis_id=aid, year=2021, topic_i=0, topic_j=1, support_count=1, weighted_intersection=0.1, normalized_weighted_value=0.1, threshold=0.1)
    db.add(ti1)
    
    db.commit()
    yield db, aid
    
    # Teardown
    db.query(GapEvidenceDB).filter(GapEvidenceDB.candidate_id.like(f"{aid}%")).delete(synchronize_session=False)
    db.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == aid).delete(synchronize_session=False)
    db.query(TemporalIntersectionDB).filter(TemporalIntersectionDB.analysis_id == aid).delete()
    db.query(TemporalTopicDB).filter(TemporalTopicDB.analysis_id == aid).delete()
    db.query(DocumentTopic).filter(DocumentTopic.topic_analysis_id == ta_id).delete()
    db.query(PaperDB).filter(PaperDB.analysis_id == aid).delete()
    db.query(Topic).filter(Topic.topic_analysis_id == ta_id).delete()
    db.query(TopicAnalysis).filter(TopicAnalysis.id == ta_id).delete()
    db.query(Analysis).filter(Analysis.id == aid).delete()
    db.commit()
    db.close()

def test_gap_engine_config_defaults():
    config = GapEngineConfig()
    assert config.rarity_weight == 0.20
    assert config.topic_presence_threshold == 0.20

def test_rarity_score(db_session):
    db, aid = db_session
    config = GapEngineConfig()
    score, ev = calculate_rarity_score(db, aid, 0, config, 2)
    assert score == 0.5  # 1 out of 2 papers have probability >= 0.2
    assert len(ev) == 1
    assert ev[0].evidence_type == "low_coverage"

def test_intersection_sparsity(db_session):
    db, aid = db_session
    config = GapEngineConfig()
    score, ev = calculate_intersection_sparsity(db, aid, 0, 1, config, 2)
    assert score == 0.5  # max support 1 / 2 total = 0.5 coverage -> sparsity 0.5

def test_temporal_score(db_session):
    db, aid = db_session
    config = GapEngineConfig(minimum_temporal_points=3)
    score, ev = calculate_temporal_score(db, aid, 0, config)
    assert score == 0.8  # declining
    assert len(ev) == 1
    assert ev[0].year == 2023

def test_limitation_score(db_session):
    db, aid = db_session
    config = GapEngineConfig()
    score, ev = calculate_limitation_score(db, aid, 0, config)
    assert score == 1.0  # 1 paper has limitation phrase, out of 1 associated paper = 1.0 proportion. 1.0 / 0.20 > 1.0 -> 1.0
    assert len(ev) == 1
    assert ev[0].evidence_metadata["matched_phrase"] == "limitation"

def test_semantic_score(db_session):
    db, aid = db_session
    config = GapEngineConfig()
    score, ev = calculate_semantic_score(db, aid, 0, config)
    assert score is None

def test_run_gap_engine(db_session):
    db, aid = db_session
    config = GapEngineConfig(minimum_temporal_points=3)
    resp = run_gap_engine(db, aid, config)
    
    assert resp.analysis_id == aid
    assert resp.candidate_count > 0
    assert len(resp.candidates) > 0
    assert resp.scoring_status == "Incomplete (Semantic signal missing)"
    assert "Semantic (S)" in resp.missing_signals
    
    c_T0 = next((c for c in resp.candidates if c.candidate_id == f"{aid}-T0"), None)
    assert c_T0 is not None
    assert c_T0.rarity_score == 0.5
    assert c_T0.temporal_score == 0.8
    assert c_T0.limitation_score == 1.0
    assert c_T0.semantic_support_score is None
    
    # Numerator = (0.2*0.5) + (0.2*0.8) + (0.2*1.0) = 0.1 + 0.16 + 0.2 = 0.46
    # Denominator = 0.6 (Semantic is None, Intersection is None for individual topic)
    # Final score = 0.46 / 0.6 = 0.7666...
    assert pytest.approx(c_T0.final_gap_score, 0.01) == 0.766
    
    # Test persistence
    persist_gap_candidates(db, aid, resp)
    db_cands = db.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == aid).all()
    assert len(db_cands) == len(resp.candidates)
