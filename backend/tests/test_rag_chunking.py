from app.services.rag_chunking import chunk_paper, RAGChunk
from app.models.paper_db import PaperDB

def test_chunking_valid_text():
    paper = PaperDB(
        id="paper1",
        analysis_id="a1",
        title="Test Title",
        abstract="Word " * 600,
        year=2020
    )
    
    chunks = chunk_paper(paper, chunk_size=500, overlap=100)
    
    assert len(chunks) > 0
    # First chunk is title
    assert chunks[0].source_field == "title"
    assert chunks[0].text == "Test Title"
    
    # Second chunk is first 500 words of abstract
    assert chunks[1].source_field == "abstract"
    assert len(chunks[1].text.split()) == 500
    
    # Metadata check
    for c in chunks:
        assert c.paper_id == "paper1"
        assert c.analysis_id == "a1"
        assert c.year == 2020
        assert c.chunk_id.startswith("chunk-paper1")

def test_chunking_empty_text():
    paper = PaperDB(
        id="paper2",
        analysis_id="a1",
        title="",
        abstract=" ",
        year=2021
    )
    
    chunks = chunk_paper(paper)
    assert len(chunks) == 0
