import pytest
from app.services.vector_store import LocalVectorStore
from app.services.rag_chunking import RAGChunk

def test_vector_store_search():
    store = LocalVectorStore()
    
    chunks = [
        RAGChunk("c1", "p1", "a1", "T1", 2020, "abs", "This is about deep learning"),
        RAGChunk("c2", "p2", "a1", "T2", 2020, "abs", "This is about biology and cells"),
        RAGChunk("c3", "p3", "a1", "T3", 2020, "abs", "Deep neural networks are great")
    ]
    
    store.add_chunks(chunks)
    
    # Vector store uses simple hash-based pseudo random vectors, 
    # but we can verify interface and sorting.
    res = store.search("deep learning", top_k=2, min_similarity=-1.0)
    assert len(res) == 2
    assert isinstance(res[0][1], float)
    
    # Sort order
    assert res[0][1] >= res[1][1]
