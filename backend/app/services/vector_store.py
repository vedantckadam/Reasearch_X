import numpy as np
from typing import List, Dict, Any, Tuple
from app.services.rag_chunking import RAGChunk
from app.services.rag_embeddings import get_embedder

class LocalVectorStore:
    """
    A minimal analysis-scoped local vector store using numpy.
    No persistent database required for MVP Phase O.
    """
    def __init__(self):
        self.chunks: List[RAGChunk] = []
        self.vectors: List[List[float]] = []
        self.embedder = get_embedder()
        
    def add_chunks(self, chunks: List[RAGChunk]):
        if not chunks:
            return
            
        texts = [c.text for c in chunks]
        embs = self.embedder.embed_documents(texts)
        
        self.chunks.extend(chunks)
        self.vectors.extend(embs)
        
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Tuple[RAGChunk, float]]:
        if not self.chunks:
            return []
            
        query_vec = np.array(self.embedder.embed_text(query))
        
        # Calculate cosine similarity
        mat = np.array(self.vectors)
        # Assuming vectors are normalized
        similarities = mat @ query_vec
        
        # Filter and sort
        results = []
        for i, sim in enumerate(similarities):
            if sim >= min_similarity:
                results.append((self.chunks[i], float(sim)))
                
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# Simple in-memory cache mapped by analysis_id
_store_cache: Dict[str, LocalVectorStore] = {}

def get_vector_store(analysis_id: str) -> LocalVectorStore:
    if analysis_id not in _store_cache:
        _store_cache[analysis_id] = LocalVectorStore()
    return _store_cache[analysis_id]
