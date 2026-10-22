import pytest
from app.services.deduplication import normalize_string, deduplicate_papers
from app.schemas.paper import PaperSchema
from app.services.relevance import rank_papers

def test_normalize_string():
    assert normalize_string("Hello, World!") == "hello world"
    assert normalize_string("  Deep   Learning  ") == "deep learning"
    assert normalize_string(None) == ""

def test_deduplicate_papers():
    papers = [
        PaperSchema(title="Federated Learning", year=2025, doi="10.123/456", sources=["openalex"]),
        PaperSchema(title="Federated Learning!", year=2025, sources=["semantic_scholar"]),
        PaperSchema(title="Different Paper", year=2024, doi="10.123/789")
    ]
    
    deduped = deduplicate_papers(papers)
    assert len(deduped) == 2
    
    # First paper should have merged sources
    merged_paper = next(p for p in deduped if "federated learning" in p.title.lower())
    assert "openalex" in merged_paper.sources
    assert "semantic_scholar" in merged_paper.sources
    assert merged_paper.doi == "10.123/456"

def test_rank_papers():
    seed_text = "Federated learning in healthcare for edge devices."
    papers = [
        PaperSchema(title="Edge Computing in Healthcare", abstract="Federated learning helps."),
        PaperSchema(title="Quantum Computing", abstract="Not related at all."),
    ]
    
    ranked = rank_papers(seed_text, papers)
    assert ranked[0].title == "Edge Computing in Healthcare"
    assert ranked[0].relevance_score > ranked[1].relevance_score
