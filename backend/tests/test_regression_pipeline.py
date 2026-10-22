import pytest
import numpy as np
from app.schemas.paper import PaperSchema
from app.schemas.gap import GapEngineConfig, GapCandidate, GapEvidence
from app.services.query_generation import extract_seed_metadata, generate_queries
from app.services.relevance import rank_papers
from app.services.lda_model import get_topic_words, build_document_term_matrix, train_lda
from app.services.gap_engine import (
    calculate_intersection_sparsity,
    calculate_rarity_score,
    run_gap_engine,
    persist_gap_candidates
)
from app.models.paper_db import (
    SessionLocal, Analysis, TopicAnalysis, Topic, DocumentTopic,
    PaperDB, TemporalTopicDB, TemporalIntersectionDB, GapCandidateDB, GapEvidenceDB
)
import uuid

SEED_PAPER_TEXT = """An Automated Capability–Expression Gap
Analysis Framework for Professional Readiness
Assessment and Personalized Development
Author Name1[0000-0000-0000-0000] and Co-Author
Name2[1111-2222-3333-4444]
1 Department of Computer Science and Engineering, Institute of Advanced
Technology, City, Country
{author.name,coauthor.name}@institution.edu
2 Center for Human-Centered Artificial Intelligence, Research University, City, Country
Abstract. Talent acquisition and career progression systems rely pre-
dominantly on self-reported textual artifacts, such as resumes, portfolios,
and interview transcripts, to assess professional competence. This re-
liance creates a systemic structural vulnerability: candidate evaluation
conflates communicative fluency with underlying operational capability.
Consequently, candidates skilled in self-presentation gain disproportionate access to roles, whereas highly capable practitioners with less polished communication are overlooked. This phenomenon—the capability-expression gap—induces severe misalignment in hiring decisions, misguides personalized development interventions, and diminishes institutional trust. Existing automated recruitment and competence assessment tools fail to address this divergence, evaluating textual signals in isolation without grounding them in verified behavioral or artifactual evidence. This paper introduces an automated framework for capability-expression gap analysis...
Keywords: Competence assessment · Capability-expression gap · Professional readiness · Natural language processing · Human-centered AI · Personalized development.
"""

# ---------------------------------------------------------------------------
# 1. Query Generation & Seed Paper Metadata Extraction Tests
# ---------------------------------------------------------------------------

def test_seed_paper_metadata_extraction():
    meta = extract_seed_metadata(SEED_PAPER_TEXT)
    assert "Capability-Expression Gap Analysis" in meta["title"] or "Capability–Expression Gap Analysis" in meta["title"]
    assert "Professional Readiness" in meta["title"]
    assert len(meta["title_terms"]) >= 4
    assert "Talent acquisition" in meta["abstract"]
    assert len(meta["keywords"]) >= 3

def test_query_generation_seed_paper_domain_anchors():
    queries = generate_queries(SEED_PAPER_TEXT, num_queries=3)
    assert len(queries) == 3
    # Verify that generic unanchored query 'capability expression' is NOT generated
    assert "capability expression" not in queries
    # Verify domain grounded queries are present
    assert any("gap" in q.lower() for q in queries)
    assert any("readiness" in q.lower() or "assessment" in q.lower() or "talent" in q.lower() for q in queries)

# ---------------------------------------------------------------------------
# 2. Corpus Relevance & Domain Filtering Tests
# ---------------------------------------------------------------------------

def test_mixed_corpus_relevance_filtering():
    """
    Verify that unrelated biological and materials-science homonym papers
    cannot dominate the analysis and are rejected (relevance set to 0.0).
    """
    papers = [
        PaperSchema(
            title="Learned but Not Expressed: Capability-Expression Dissociation in Large Language Models",
            abstract="We evaluate capability vs expression in language models.",
            year=2026
        ),
        PaperSchema(
            title="Comprehensive assessment of the graduates readiness for professional work",
            abstract="A framework for evaluating competence and professional readiness.",
            year=2020
        ),
        PaperSchema(
            title="Dynamic Maternal Gradients Control Drosophila Gap Gene Expression",
            abstract="Canalization of gap gene expression in the Drosophila blastoderm embryo.",
            year=2017
        ),
        PaperSchema(
            title="Heparanase expression correlates with metastatic capability in human choriocarcinoma",
            abstract="Metastatic capability of carcinoma cells and heparanase expression.",
            year=2007
        ),
        PaperSchema(
            title="Achieving DFT accuracy with machine-learning interatomic potential in bcc iron",
            abstract="Interatomic potentials for defects in ferromagnetic iron.",
            year=2018
        ),
    ]

    ranked = rank_papers(SEED_PAPER_TEXT, papers)
    by_title = {p.title: p.relevance_score for p in ranked}

    # Relevant papers must have positive relevance
    assert by_title["Learned but Not Expressed: Capability-Expression Dissociation in Large Language Models"] > 0.05
    assert by_title["Comprehensive assessment of the graduates readiness for professional work"] > 0.05

    # Off-domain homonym papers must be rejected (0.0)
    assert by_title["Dynamic Maternal Gradients Control Drosophila Gap Gene Expression"] == 0.0
    assert by_title["Heparanase expression correlates with metastatic capability in human choriocarcinoma"] == 0.0
    assert by_title["Achieving DFT accuracy with machine-learning interatomic potential in bcc iron"] == 0.0

def test_interdisciplinary_work_preserved():
    """
    Verify that legitimate interdisciplinary work (e.g. from education, public policy, management)
    is NOT deleted automatically.
    """
    papers = [
        PaperSchema(
            title="Seeking sustainability competence and capability in higher education",
            abstract="An evaluation of student capabilities, skills, and assessment in educational curricula.",
            year=2018
        ),
        PaperSchema(
            title="Building State Capability: Evidence, Analysis, Action",
            abstract="Organizational capability gaps in public administration and policy readiness.",
            year=2017
        ),
    ]

    ranked = rank_papers(SEED_PAPER_TEXT, papers)
    for p in ranked:
        assert p.relevance_score > 0.0

# ---------------------------------------------------------------------------
# 3. Topic Labeling & "Unresolved" Fallback Tests
# ---------------------------------------------------------------------------

def test_unresolved_topic_label_fallback():
    """
    Never invent a descriptive label when the topic is not interpretable; mark it as unresolved.
    """
    class MockLDA:
        def __init__(self):
            # Topic 0: normal topic with distinctive features
            # Topic 1: uninterpretable / noisy topic with generic academic stop words and negligible weight
            self.components_ = np.array([
                [0.8, 0.7, 0.1, 0.05],
                [0.01, 0.01, 0.01, 0.01]  # degenerate flat weights
            ])
            
    lda = MockLDA()
    feature_names = ["readiness", "competence", "study", "analysis"]
    topics = get_topic_words(lda, feature_names, n_top_words=4)

    assert "Readiness" in topics[0]["label"]
    # Topic 1 has flat/degenerate weights (< 0.04) and should be marked as Unresolved Topic
    assert "Unresolved Topic" in topics[1]["label"]

def test_topic_labeling_academic_stop_words_filtered():
    """
    Verify that generic academic meta-words (study, analysis, based) do not contaminate topic labels.
    """
    class MockLDA:
        def __init__(self):
            # 'study' and 'analysis' have highest weight, but 'talent' and 'analytics' are substantive
            self.components_ = np.array([
                [0.9, 0.8, 0.7, 0.6]
            ])
            
    lda = MockLDA()
    feature_names = ["study", "analysis", "talent", "analytics"]
    topics = get_topic_words(lda, feature_names, n_top_words=4)

    assert "Talent & Analytics" in topics[0]["label"]
    assert "Study" not in topics[0]["label"]
    assert "Analysis" not in topics[0]["label"]

# ---------------------------------------------------------------------------
# 4. Gap Scoring, Empty Intersection & Confidence Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def gap_fixture():
    db = SessionLocal()
    aid = str(uuid.uuid4())
    ta_id = str(uuid.uuid4())
    
    analysis = Analysis(id=aid, status="COMPLETED")
    db.add(analysis)
    
    ta = TopicAnalysis(id=ta_id, analysis_id=aid, num_topics=2)
    db.add(ta)
    
    t0 = Topic(id=str(uuid.uuid4()), topic_analysis_id=ta_id, topic_idx=0, label="Readiness Assessment", corpus_share=0.5, dominant_papers=5)
    t1 = Topic(id=str(uuid.uuid4()), topic_analysis_id=ta_id, topic_idx=1, label="Talent Analytics", corpus_share=0.5, dominant_papers=5)
    db.add_all([t0, t1])
    
    # 10 documents
    p_ids = []
    for i in range(10):
        pid = str(uuid.uuid4())
        p_ids.append(pid)
        p = PaperDB(id=pid, analysis_id=aid, title=f"Paper {i}", abstract="Standard empirical findings with complete scope.", year=2022)
        db.add(p)
        
    # Topics 0 and 1 never co-occur (support = 0)
    for i in range(5):
        db.add(DocumentTopic(id=str(uuid.uuid4()), topic_analysis_id=ta_id, paper_id=p_ids[i], topic_idx=0, probability=0.9))
    for i in range(5, 10):
        db.add(DocumentTopic(id=str(uuid.uuid4()), topic_analysis_id=ta_id, paper_id=p_ids[i], topic_idx=1, probability=0.9))
        
    ti = TemporalIntersectionDB(
        id=str(uuid.uuid4()), analysis_id=aid, year=2022, topic_i=0, topic_j=1,
        support_count=0, weighted_intersection=0.0, normalized_weighted_value=0.0, threshold=0.2
    )
    db.add(ti)
    db.commit()
    
    yield db, aid, ta_id
    
    # Teardown
    db.query(GapEvidenceDB).filter(GapEvidenceDB.candidate_id.like(f"{aid}%")).delete(synchronize_session=False)
    db.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == aid).delete(synchronize_session=False)
    db.query(TemporalIntersectionDB).filter(TemporalIntersectionDB.analysis_id == aid).delete()
    db.query(DocumentTopic).filter(DocumentTopic.topic_analysis_id == ta_id).delete()
    db.query(PaperDB).filter(PaperDB.analysis_id == aid).delete()
    db.query(Topic).filter(Topic.topic_analysis_id == ta_id).delete()
    db.query(TopicAnalysis).filter(TopicAnalysis.id == ta_id).delete()
    db.query(Analysis).filter(Analysis.id == aid).delete()
    db.commit()
    db.close()

def test_empty_intersection_does_not_saturate_to_one(gap_fixture):
    """
    Verify that an unobserved topic intersection (support=0, sparsity=1.0)
    does NOT produce a final gap score of 1.00 when uncorroborated by literature evidence.
    """
    db, aid, _ = gap_fixture
    config = GapEngineConfig()
    
    resp = run_gap_engine(db, aid, config)
    cand = next((c for c in resp.candidates if c.candidate_id == f"{aid}-I0-1"), None)
    
    assert cand is not None
    # Sparsity should be 1.0 (no co-occurrence observed)
    assert cand.intersection_score == 1.0
    # Limitation score should be None (no limitation sentences in corpus)
    assert cand.limitation_score is None
    # Formula: (0.20 * 1.0 + 0) / 0.40 = 0.50
    assert cand.final_gap_score == 0.50
    assert cand.final_gap_score < 1.0

def test_empty_intersection_confidence_is_not_high(gap_fixture):
    """
    Verify that an empty intersection alone does not automatically produce High confidence.
    """
    db, aid, _ = gap_fixture
    config = GapEngineConfig()
    
    resp = run_gap_engine(db, aid, config)
    cand = next((c for c in resp.candidates if c.candidate_id == f"{aid}-I0-1"), None)
    
    assert cand is not None
    assert cand.confidence_level != "High"
    assert cand.confidence_rationale is not None
    assert "Unobserved" in cand.confidence_rationale or "sparsity" in cand.confidence_rationale.lower()

def test_topic_pair_candidate_validity_fields(gap_fixture):
    """
    Verify that candidate contains topic labels, descriptive statement, and transparent signals.
    """
    db, aid, _ = gap_fixture
    config = GapEngineConfig()
    
    resp = run_gap_engine(db, aid, config)
    cand = next((c for c in resp.candidates if c.candidate_id == f"{aid}-I0-1"), None)
    
    assert cand is not None
    # Topic labels must be populated
    assert cand.topic_labels == ["Readiness Assessment", "Talent Analytics"]
    # Descriptive statement must be grounded
    assert "Readiness Assessment" in cand.descriptive_statement
    assert "Talent Analytics" in cand.descriptive_statement
    # Reasoning signals must document formula and absence of limitation evidence
    assert any("Formula" in sig for sig in cand.reasoning_signals)
    assert any("No author limitation" in sig for sig in cand.reasoning_signals)
