from typing import List
import logging
import numpy as np
import scipy.sparse as sp
from app.schemas.paper import PaperSchema
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

def compute_cosine_similarity(X, Y) -> np.ndarray:
    """
    Computes cosine similarity between matrices X and Y using SciPy and NumPy.
    Avoids importing sklearn.metrics.cluster which attempts to load Cython extensions
    blocked by Windows Application Control policies.
    """
    if sp.issparse(X):
        norm_X = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norm_X[norm_X == 0] = 1.0
        X_norm = X.multiply(1.0 / norm_X[:, None])
    else:
        norm_X = np.linalg.norm(X, axis=1, keepdims=True)
        norm_X[norm_X == 0] = 1.0
        X_norm = X / norm_X
        
    if sp.issparse(Y):
        norm_Y = np.sqrt(np.asarray(Y.multiply(Y).sum(axis=1)).ravel())
        norm_Y[norm_Y == 0] = 1.0
        Y_norm = Y.multiply(1.0 / norm_Y[:, None])
    else:
        norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
        norm_Y[norm_Y == 0] = 1.0
        Y_norm = Y / norm_Y
        
    if sp.issparse(X_norm) or sp.issparse(Y_norm):
        sim = (X_norm @ Y_norm.T)
        if sp.issparse(sim):
            return sim.toarray()
        return np.asarray(sim)
    return np.dot(X_norm, Y_norm.T)

# Off-domain terminology markers that signal biological, oncological, and materials-science homonym false positives
_OFF_DOMAIN_HOMONYM_TERMS = {
    "drosophila", "blastoderm", "metastatic", "choriocarcinoma", "hepatocarcinoma",
    "chloroplasts", "chloroplast", "ferromagnetic", "interatomic", "leishmania",
    "salamandridae", "connexin43", "astrocytic", "syngap", "pancreatic cancer",
    "melanoma", "carcinoma", "nsclc", "xylem", "parenchyma", "interleukin",
    "spindle pole", "bub2p", "bfa1p", "supercooling", "baculovirus", "endothelial",
    "gfp", "mrna", "cd34+", "murine", "choriocarcinoma", "phytoremediation"
}

# Domain terms indicating substantive alignment with evaluation, assessment, computing, management, or education
_DOMAIN_AFFINITY_TERMS = {
    "readiness", "competence", "competency", "talent", "assessment", "readiness",
    "skill", "skills", "education", "learning", "professional", "career",
    "organization", "organizational", "workforce", "hiring", "personnel",
    "student", "students", "training", "evaluation", "framework", "performance",
    "artificial intelligence", "nlp", "language model", "language models",
    "capability gap", "capability gaps"
}

def rank_papers(seed_text: str, papers: List[PaperSchema]) -> List[PaperSchema]:
    """
    Ranks papers based on TF-IDF cosine similarity to the seed text.
    Uses structured seed metadata (title, keywords, abstract) to construct a high-precision
    profile and filters out cross-domain homonym false positives while preserving legitimate
    interdisciplinary literature.
    """
    if not papers:
        return papers
        
    try:
        from app.services.query_generation import extract_seed_metadata
        meta = extract_seed_metadata(seed_text or "")
        
        # Build focused seed representation giving higher weight to title and keywords
        if meta.get("title"):
            seed_profile = f"{meta['title']} {meta['title']} {' '.join(meta.get('keywords', []))} {meta.get('abstract', '')}".strip()
        else:
            seed_profile = seed_text or ""

        # Check whether seed paper is in the human/AI/workforce/capability assessment domain
        seed_lower = (seed_profile or "").lower()
        seed_is_assessment_domain = any(term in seed_lower for term in [
            "readiness", "competence", "assessment", "talent", "career", "professional", "capability-expression"
        ])

        # Prepare corpus: title + abstract for each paper
        corpus = []
        valid_indices = []
        for i, p in enumerate(papers):
            text = f"{p.title or ''} {p.abstract or ''}".strip()
            if text:
                corpus.append(text)
                valid_indices.append(i)
            else:
                p.relevance_score = 0.0

        if not corpus:
            return papers

        # Add seed profile as reference document
        corpus.insert(0, seed_profile)
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate cosine similarity against seed profile (index 0)
        similarities = compute_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        for idx, sim in zip(valid_indices, similarities):
            p = papers[idx]
            paper_text = f"{p.title or ''} {p.abstract or ''}".lower()
            
            # Check for cross-domain homonym false positives
            is_homonym_noise = False
            if seed_is_assessment_domain:
                has_off_domain = any(term in paper_text for term in _OFF_DOMAIN_HOMONYM_TERMS)
                has_core_domain = any(term in paper_text for term in ["readiness", "competence", "talent", "career", "assessment"])
                if has_off_domain and not has_core_domain:
                    is_homonym_noise = True

            if is_homonym_noise:
                # Downweight homonyms to 0.0 to prevent domain contamination
                p.relevance_score = 0.0
            else:
                p.relevance_score = float(sim)
            
        # Sort papers by relevance score in descending order
        papers.sort(key=lambda x: x.relevance_score, reverse=True)
        
    except Exception as e:
        logger.error(f"Relevance ranking failed: {e}")
        
    return papers
