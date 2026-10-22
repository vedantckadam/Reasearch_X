from typing import List
from app.schemas.paper import PaperSchema
import re
import string

def normalize_string(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s)
    return s

def deduplicate_papers(papers: List[PaperSchema]) -> List[PaperSchema]:
    """
    Deduplicates a list of papers based on DOI, IDs, and Title+Year.
    Merges sources when duplicates are found.
    """
    deduplicated = []
    
    # Fast lookup dictionaries
    seen_doi = {}
    seen_title_year = {}
    seen_openalex = {}
    seen_semantic = {}

    for paper in papers:
        is_duplicate = False
        target_paper = None

        # 1. Check DOI
        if paper.doi:
            norm_doi = paper.doi.lower().strip()
            if norm_doi in seen_doi:
                is_duplicate = True
                target_paper = seen_doi[norm_doi]
        
        # 2. Check Provider IDs
        if not is_duplicate and paper.openalex_id and paper.openalex_id in seen_openalex:
            is_duplicate = True
            target_paper = seen_openalex[paper.openalex_id]
            
        if not is_duplicate and paper.semantic_scholar_id and paper.semantic_scholar_id in seen_semantic:
            is_duplicate = True
            target_paper = seen_semantic[paper.semantic_scholar_id]

        # 3. Check Normalized Title + Year
        if not is_duplicate and paper.title and paper.year:
            norm_title = normalize_string(paper.title)
            key = f"{norm_title}_{paper.year}"
            if key in seen_title_year:
                is_duplicate = True
                target_paper = seen_title_year[key]

        if is_duplicate and target_paper:
            # Merge sources
            for source in paper.sources:
                if source not in target_paper.sources:
                    target_paper.sources.append(source)
            
            # Enrich missing data
            if not target_paper.abstract and paper.abstract:
                target_paper.abstract = paper.abstract
            if not target_paper.doi and paper.doi:
                target_paper.doi = paper.doi
            if not target_paper.openalex_id and paper.openalex_id:
                target_paper.openalex_id = paper.openalex_id
            if not target_paper.semantic_scholar_id and paper.semantic_scholar_id:
                target_paper.semantic_scholar_id = paper.semantic_scholar_id
        else:
            # Add to deduplicated list and populate lookup dictionaries
            deduplicated.append(paper)
            
            if paper.doi:
                seen_doi[paper.doi.lower().strip()] = paper
            if paper.openalex_id:
                seen_openalex[paper.openalex_id] = paper
            if paper.semantic_scholar_id:
                seen_semantic[paper.semantic_scholar_id] = paper
            if paper.title and paper.year:
                norm_title = normalize_string(paper.title)
                seen_title_year[f"{norm_title}_{paper.year}"] = paper

    return deduplicated
