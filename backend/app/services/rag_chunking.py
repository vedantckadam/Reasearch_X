import uuid
from typing import List, Dict, Any
from app.models.paper_db import PaperDB

class RAGChunk:
    def __init__(self, chunk_id: str, paper_id: str, analysis_id: str, title: str, 
                 year: int, source_field: str, text: str, **kwargs):
        self.chunk_id = chunk_id
        self.paper_id = paper_id
        self.analysis_id = analysis_id
        self.title = title
        self.year = year
        self.source_field = source_field
        self.text = text
        self.metadata = kwargs  # Additional provenance metadata

def chunk_paper(paper: PaperDB, chunk_size: int = 500, overlap: int = 100) -> List[RAGChunk]:
    chunks = []
    
    # Use abstract, or title if abstract is not available. 
    # The Gap Engine currently relies mostly on abstracts as per PaperDB fields.
    fields_to_chunk = []
    if paper.title:
        fields_to_chunk.append(("title", paper.title))
    if paper.abstract:
        fields_to_chunk.append(("abstract", paper.abstract))
        
    for field_name, text in fields_to_chunk:
        if not text.strip():
            continue
            
        words = text.split()
        if not words:
            continue
            
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            
            # Deterministic chunk ID based on paper ID and field/start position
            chunk_id = f"chunk-{paper.id}-{field_name}-{start}"
            
            chunks.append(
                RAGChunk(
                    chunk_id=chunk_id,
                    paper_id=paper.id,
                    analysis_id=paper.analysis_id,
                    title=paper.title or "Unknown Title",
                    year=paper.year or 0,
                    source_field=field_name,
                    text=chunk_text,
                    doi=paper.doi,
                    url=paper.url,
                    char_start=start,
                    char_end=end
                )
            )
            
            start += (chunk_size - overlap)
            
    return chunks
