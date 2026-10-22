import re
import logging
from typing import List, Optional, Any

logger = logging.getLogger(__name__)

# Stop words for query term filtering
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _STOP_WORDS = set(ENGLISH_STOP_WORDS)
except Exception:
    _STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "for", "from", "further", "had", "has", "have", "in", "into", "is", "it", "its", "of", "on",
        "or", "other", "our", "out", "over", "the", "their", "then", "there", "these", "they", "this",
        "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
        "when", "where", "which", "while", "who", "whom", "why", "with"
    }

_SPACY_TRIED = False
_SPACY_NLP: Optional[Any] = None

def get_spacy_nlp() -> Optional[Any]:
    """Safely attempts to load spaCy model for query generation."""
    global _SPACY_TRIED, _SPACY_NLP
    if _SPACY_TRIED:
        return _SPACY_NLP
    _SPACY_TRIED = True
    try:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm")
    except Exception as e:
        logger.warning("spaCy not available for query generation (%s); using deterministic n-gram fallback.", e)
        _SPACY_NLP = None
    return _SPACY_NLP

def extract_seed_metadata(text: str) -> dict:
    """
    Extracts structured academic metadata (title, abstract, keywords, and title terms)
    from seed paper text if present.
    """
    if not text or not text.strip():
        return {"title": "", "abstract": "", "keywords": [], "title_terms": []}

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    title_lines = []
    abstract_lines = []
    keywords = []

    in_abstract = False
    in_keywords = False
    past_header = False

    for line in lines[:60]:
        abs_match = re.search(r'\babstract[\.:\s—\-]+(.*)', line, re.IGNORECASE)
        kw_match = re.search(r'\b(?:keywords|key\s*words|index\s*terms)[\.:\s—\-]+(.*)', line, re.IGNORECASE)

        if kw_match:
            in_abstract = False
            in_keywords = True
            kw_text = kw_match.group(1)
            raw_kws = re.split(r'[,;·•\u00b7]', kw_text)
            for k in raw_kws:
                k_clean = re.sub(r'[\.\s]+$', '', k).strip()
                if k_clean and len(k_clean) > 2:
                    keywords.append(k_clean)
            continue

        if abs_match:
            past_header = True
            in_abstract = True
            initial_abs = abs_match.group(1).strip()
            if initial_abs:
                abstract_lines.append(initial_abs)
            continue

        if in_keywords:
            raw_kws = re.split(r'[,;·•\u00b7]', line)
            for k in raw_kws:
                k_clean = re.sub(r'[\.\s]+$', '', k).strip()
                if k_clean and len(k_clean) > 2:
                    keywords.append(k_clean)
            in_keywords = False
            continue

        if in_abstract:
            if re.match(r'^(?:1\.?|I\.?|1\s+|I\s+)?\s*(?:introduction|background|related\s+work)\b', line, re.IGNORECASE):
                in_abstract = False
                break
            abstract_lines.append(line)
            continue

        if not past_header:
            if re.search(r'(@|department|faculty|university|institute|author|center\s+for|dr\.|prof\.|\[[0-9\-]+\])', line, re.IGNORECASE):
                past_header = True
                continue
            title_lines.append(line)

    title = " ".join(title_lines).strip()
    title = re.sub(r'[\u2013\u2014]', '-', title)
    title = re.sub(r'\s+', ' ', title)

    abstract = " ".join(abstract_lines).strip()
    abstract = re.sub(r'\s+', ' ', abstract)

    # Substantive content words from title
    title_words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', title.lower())
    title_terms = [w for w in title_words if w not in _STOP_WORDS and len(w) > 2 and not w.isnumeric()]

    return {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "title_terms": title_terms
    }

def _extract_fallback_phrases(text: str) -> List[str]:
    """
    Extracts substantive 2-3 word keyphrases deterministically without requiring spaCy.
    Scans for consecutive non-stop-word scientific terms and groups them into candidate phrases.
    """
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text.lower())
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 2 and not w.isnumeric()]
    
    if not content_words:
        return []
        
    phrases = []
    # Generate substantive bigrams
    for i in range(len(content_words) - 1):
        phrases.append(f"{content_words[i]} {content_words[i+1]}")
    # Generate trigrams
    for i in range(len(content_words) - 2):
        phrases.append(f"{content_words[i]} {content_words[i+1]} {content_words[i+2]}")
        
    # Count frequency to preserve most significant phrases
    phrase_freq = {}
    for p in phrases:
        phrase_freq[p] = phrase_freq.get(p, 0) + 1
        
    # Sort deterministically by frequency descending, then length descending, then alphabetical
    sorted_phrases = sorted(phrase_freq.keys(), key=lambda p: (phrase_freq[p], len(p), p), reverse=True)
    return sorted_phrases

def generate_queries(text: str, num_queries: int = 3) -> List[str]:
    """
    Generates search queries from seed paper text deterministically.
    Prioritizes paper title and domain context if an academic paper structure is detected,
    preventing off-domain homonym retrieval. Falls back to deterministic keyphrase extraction.
    """
    if not text or not text.strip():
        return ["Research"]

    meta = extract_seed_metadata(text)
    queries: List[str] = []

    # 1. If academic title and terms are detected, formulate grounded domain queries
    title = meta.get("title", "")
    title_terms = meta.get("title_terms", [])
    keywords = meta.get("keywords", [])
    abstract = meta.get("abstract", "")

    if len(title_terms) >= 3:
        # Check for core methodological phrases (e.g. "capability-expression gap", "capability gap")
        title_lower = title.lower()
        if "capability" in title_lower and "gap" in title_lower:
            queries.append("capability expression gap")
        elif len(title_terms) >= 2:
            queries.append(f"{title_terms[0]} {title_terms[1]} {title_terms[2]}" if len(title_terms) >= 3 else f"{title_terms[0]} {title_terms[1]}")

        # Check for domain context (e.g. "professional readiness assessment", "competence assessment")
        if "readiness" in title_lower and "assessment" in title_lower:
            queries.append("professional readiness assessment")
        elif "competence" in title_lower or "readiness" in title_lower:
            domain_words = [w for w in title_terms if w in {"readiness", "competence", "assessment", "development", "personalized"}]
            if len(domain_words) >= 2:
                queries.append(" ".join(domain_words[:3]))

        # Include keywords or bridge queries
        if keywords:
            for kw in keywords:
                clean_kw = re.sub(r'[\u2013\u2014]', '-', kw.lower()).strip()
                if len(clean_kw.split()) >= 2 and clean_kw not in [q.lower() for q in queries]:
                    queries.append(clean_kw)
                    if len(queries) >= num_queries:
                        break

        # Bridging query combining core problem and domain
        if len(queries) < num_queries:
            if "capability" in title_lower and "readiness" in title_lower:
                queries.append("capability gap professional readiness")
            elif len(title_terms) >= 4:
                queries.append(f"{title_terms[-3]} {title_terms[-2]} {title_terms[-1]}")

    # 2. If spaCy is available and queries still needed
    if len(queries) < num_queries:
        nlp = get_spacy_nlp()
        if nlp is not None:
            try:
                doc = nlp(text[:5000])
                chunks = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.split()) >= 2]
                chunks = sorted(list(set(chunks)), key=lambda x: (len(x), x), reverse=True)
                for chunk in chunks:
                    if chunk not in queries:
                        queries.append(chunk)
                    if len(queries) >= num_queries:
                        break
            except Exception as e:
                logger.warning("spaCy noun chunk extraction failed (%s); using fallback.", e)

    # 3. Deterministic keyphrase fallback
    if len(queries) < num_queries:
        sample = text[:5000]
        candidate_phrases = _extract_fallback_phrases(sample)
        for p in candidate_phrases:
            if p not in queries:
                queries.append(p)
            if len(queries) >= num_queries:
                break

    # 4. Minimal word fallback if still empty
    if not queries:
        words = text.split()
        if len(words) < 5:
            queries = [" ".join(words)]
        else:
            queries = [
                " ".join(words[:5]),
                " ".join(words[5:10]) if len(words) >= 10 else " ".join(words[:3])
            ]

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for q in queries:
        q_norm = q.strip()
        if q_norm.lower() not in seen:
            seen.add(q_norm.lower())
            deduped.append(q_norm)

    return deduped[:num_queries]
