import requests
import logging
import os
from typing import List
from app.schemas.paper import PaperSchema

logger = logging.getLogger(__name__)

class SemanticScholarClient:
    def __init__(self, base_url="https://api.semanticscholar.org/graph/v1"):
        self.base_url = base_url
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def search_works(self, query: str, limit: int = 25) -> List[PaperSchema]:
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,url,title,abstract,venue,year,authors,citationCount,externalIds"
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
            
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for item in data.get("data", []):
                authors = [auth.get("name", "") for auth in item.get("authors", [])]
                
                paper = PaperSchema(
                    title=item.get("title") or "Untitled",
                    abstract=item.get("abstract"),
                    year=item.get("year"),
                    authors=[a for a in authors if a],
                    citation_count=item.get("citationCount", 0),
                    url=item.get("url"),
                    publication_venue=item.get("venue"),
                    doi=item.get("externalIds", {}).get("DOI"),
                    sources=["semantic_scholar"],
                    semantic_scholar_id=item.get("paperId")
                )
                papers.append(paper)
            return papers
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []
