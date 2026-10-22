import requests
import logging
from typing import List
from app.schemas.paper import PaperSchema

logger = logging.getLogger(__name__)

class OpenAlexClient:
    def __init__(self, base_url="https://api.openalex.org"):
        self.base_url = base_url

    def search_works(self, query: str, limit: int = 25) -> List[PaperSchema]:
        url = f"{self.base_url}/works"
        params = {
            "search": query,
            "per-page": limit,
            "mailto": "researchgapx@example.com" # Best practice for OpenAlex
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for item in data.get("results", []):
                authors = [auth.get("author", {}).get("display_name", "") for auth in item.get("authorships", [])]
                
                # OpenAlex sometimes gives inverted abstracts, but for search results it might not be full text
                # We extract what we can safely.
                
                paper = PaperSchema(
                    title=item.get("title") or "Untitled",
                    year=item.get("publication_year"),
                    doi=item.get("doi"),
                    authors=[a for a in authors if a],
                    citation_count=item.get("cited_by_count", 0),
                    url=item.get("id"), # OpenAlex landing page or id
                    is_open_access=item.get("open_access", {}).get("is_oa", False),
                    sources=["openalex"],
                    openalex_id=item.get("id")
                )
                papers.append(paper)
            return papers
        except Exception as e:
            logger.error(f"OpenAlex search failed: {e}")
            return []
