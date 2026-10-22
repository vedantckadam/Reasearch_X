import numpy as np
import pytest
from app.services.topic_intersections import (
    calculate_weighted_intersection,
    calculate_support_intersection,
    calculate_normalized_association,
    extract_topic_pairs
)
from app.services.latent_analysis import load_document_topic_matrix
from app.models.paper_db import SessionLocal, TopicAnalysis, DocumentTopic, Topic, Analysis

def test_weighted_intersection():
    theta = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1]
    ])
    
    M, num_topics = calculate_weighted_intersection(theta)
    
    assert num_topics == 3
    expected_M = theta.T @ theta
    np.testing.assert_allclose(M, expected_M, atol=1e-5)
    
    # Test symmetry
    np.testing.assert_allclose(M, M.T, atol=1e-5)

def test_support_intersection():
    theta = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1]
    ])
    
    # Threshold 0.20
    # Doc 0: [1, 0, 0]
    # Doc 1: [1, 1, 0]
    # Doc 2: [0, 1, 0]
    # Doc 3: [1, 1, 0]
    
    S = calculate_support_intersection(theta, threshold=0.20)
    
    # Diagonals: doc counts
    assert S[0, 0] == 3 # docs 0, 1, 3
    assert S[1, 1] == 3 # docs 1, 2, 3
    assert S[2, 2] == 0
    
    # Symmetry
    np.testing.assert_allclose(S, S.T)
    
def test_normalized_association():
    theta = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1]
    ])
    M, _ = calculate_weighted_intersection(theta)
    A = calculate_normalized_association(M)
    
    expected_A01 = M[0, 1] / np.sqrt(M[0, 0] * M[1, 1])
    np.testing.assert_allclose(A[0, 1], expected_A01, atol=1e-5)
    
def test_zero_denominator():
    theta = np.array([
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    M, _ = calculate_weighted_intersection(theta)
    A = calculate_normalized_association(M)
    
    # Topic 1 has 0 variance, so normalized association with it should be 0 safely
    assert not np.isnan(A).any()
    assert not np.isinf(A).any()
    assert A[1, 1] == 0.0
    assert A[0, 1] == 0.0

def test_invalid_matrix():
    with pytest.raises(ValueError, match="2-dimensional"):
        calculate_weighted_intersection(np.array([1, 2, 3]))
        
    with pytest.raises(ValueError, match="at least one topic"):
        calculate_weighted_intersection(np.array([[]]))
        
    with pytest.raises(ValueError, match="NaN"):
        calculate_weighted_intersection(np.array([[1.0, np.nan]]))

def test_invalid_threshold():
    theta = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        calculate_support_intersection(theta, threshold=-0.1)
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        calculate_support_intersection(theta, threshold=1.1)

def test_extract_topic_pairs():
    matrix = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.8],
        [0.2, 0.8, 1.0]
    ])
    labels = {0: "A", 1: "B", 2: "C"}
    pairs = extract_topic_pairs(matrix, labels)
    
    assert len(pairs) == 3
    assert pairs[0]["topic_a"] == "B"
    assert pairs[0]["topic_b"] == "C"
    assert pairs[0]["value"] == 0.8
    
    assert pairs[1]["value"] == 0.5
    assert pairs[2]["value"] == 0.2
    
    pairs_top = extract_topic_pairs(matrix, labels, top_n=1)
    assert len(pairs_top) == 1
    assert pairs_top[0]["value"] == 0.8

def test_integration():
    db = SessionLocal()
    try:
        import uuid
        test_analysis_id = str(uuid.uuid4())
        test_ta_id = str(uuid.uuid4())
        
        analysis = Analysis(id=test_analysis_id)
        db.add(analysis)
        db.commit()
        
        ta = TopicAnalysis(id=test_ta_id, analysis_id=test_analysis_id, num_topics=2, vocabulary_size=10, coherence_score=0.5)
        db.add(ta)
        db.commit()
        
        t1 = Topic(id=str(uuid.uuid4()), topic_analysis_id=test_ta_id, topic_idx=0, label="T1")
        t2 = Topic(id=str(uuid.uuid4()), topic_analysis_id=test_ta_id, topic_idx=1, label="T2")
        db.add_all([t1, t2])
        db.commit()
        
        dt1 = DocumentTopic(id=str(uuid.uuid4()), topic_analysis_id=test_ta_id, paper_id="p1", topic_idx=0, probability=0.8)
        dt2 = DocumentTopic(id=str(uuid.uuid4()), topic_analysis_id=test_ta_id, paper_id="p1", topic_idx=1, probability=0.2)
        db.add_all([dt1, dt2])
        db.commit()
        
        theta, paper_ids, topic_ids, topic_labels = load_document_topic_matrix(db, test_ta_id)
        assert theta.shape == (1, 2)
        
        M, num_topics = calculate_weighted_intersection(theta)
        assert num_topics == 2
        
        S = calculate_support_intersection(theta)
        assert S.shape == (2, 2)
        
        A = calculate_normalized_association(M)
        assert A.shape == (2, 2)
        
        pairs = extract_topic_pairs(M, topic_labels)
        assert len(pairs) == 1
    finally:
        db.rollback()
        # Clean up
        db.query(DocumentTopic).filter(DocumentTopic.topic_analysis_id == test_ta_id).delete()
        db.query(Topic).filter(Topic.topic_analysis_id == test_ta_id).delete()
        db.query(TopicAnalysis).filter(TopicAnalysis.id == test_ta_id).delete()
        db.query(Analysis).filter(Analysis.id == test_analysis_id).delete()
        db.commit()
        db.close()
