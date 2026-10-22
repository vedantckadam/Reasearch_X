from typing import List
import numpy as np

class BaseEmbedder:
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class MockEmbedder(BaseEmbedder):
    """
    A minimal mock embedder that creates deterministic vectors based on string hash.
    For production, replace with SentenceTransformers, OpenAI, or similar.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        
    def embed_text(self, text: str) -> List[float]:
        # Deterministic pseudo-random vector based on text hash
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(self.dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

# Simple singleton access
_embedder = MockEmbedder()

def get_embedder() -> BaseEmbedder:
    return _embedder
