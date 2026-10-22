import pytest
from app.services.text_preprocessing import (
    select_and_clean_text,
    preprocess_corpus,
    _fallback_tokenize_and_filter,
    is_spacy_available
)
from app.services.query_generation import generate_queries, _extract_fallback_phrases
from app.schemas.paper import PaperSchema

class MockPaper:
    def __init__(self, id, title=None, abstract=None):
        self.id = id
        self.title = title
        self.abstract = abstract

def test_select_and_clean_text_normal():
    paper = MockPaper(
        id="p1",
        title="Federated Learning in Healthcare: A Comprehensive Survey!",
        abstract="We investigate privacy-preserving deep neural networks (DNNs) with http://example.com/repo and contact info@health.org."
    )
    cleaned = select_and_clean_text(paper)
    assert "federated learning in healthcare a comprehensive survey" in cleaned
    assert "http" not in cleaned
    assert "info@health.org" not in cleaned
    assert "!" not in cleaned

def test_select_and_clean_text_empty_and_none():
    assert select_and_clean_text(None) == ""
    assert select_and_clean_text(MockPaper("p_empty", "", "")) == ""
    assert select_and_clean_text(MockPaper("p_none", None, None)) == ""

def test_fallback_tokenization_scientific_terms():
    text = "Semi-supervised COVID-19 detection using state-of-the-art transformer 12345 models with 98.5% accuracy."
    filtered = _fallback_tokenize_and_filter(text)
    # Scientific terms and hyphens preserved
    assert "semi-supervised" in filtered
    assert "covid-19" in filtered
    assert "state-of-the-art" in filtered
    assert "transformer" in filtered
    # Pure numbers and short stop-words filtered out
    assert "12345" not in filtered
    assert "98" not in filtered
    assert "with" not in filtered

def test_preprocess_corpus_deterministic_and_insufficient():
    papers = [
        MockPaper(id="p1", title="Sparse Representation", abstract=""), # too short
        MockPaper(
            id="p2",
            title="A Multi-Agent System for Quantum Machine Learning in High Energy Physics",
            abstract="Quantum variational algorithms provide exponential computational speedups for entangled particle state classification."
        )
    ]
    res1 = preprocess_corpus(papers, min_length=10)
    res2 = preprocess_corpus(papers, min_length=10)
    
    # Must be deterministic
    assert res1 == res2
    assert res1[0]["status"] == "INSUFFICIENT_TEXT"
    assert res1[0]["text"] == ""
    assert res1[1]["status"] == "OK"
    assert len(res1[1]["text"].split()) > 5

def test_query_generation_fallback():
    # Empty text
    assert generate_queries("", num_queries=3) == ["Research"]
    assert generate_queries("   ", num_queries=3) == ["Research"]
    
    # Scientific text
    abstract = (
        "This paper proposes a novel framework combining federated learning and differential privacy "
        "for edge Internet-of-Things devices in biomedical signal processing. Experimental evaluations "
        "demonstrate robust convergence under non-IID data distributions."
    )
    queries = generate_queries(abstract, num_queries=3)
    assert len(queries) <= 3
    assert len(queries) >= 1
    # Check that substantive phrases are extracted
    for q in queries:
        assert isinstance(q, str)
        assert len(q) > 3

def test_query_generation_deterministic():
    text = "Graph neural networks for molecular drug discovery and binding affinity prediction."
    q1 = generate_queries(text, num_queries=2)
    q2 = generate_queries(text, num_queries=2)
    assert q1 == q2
