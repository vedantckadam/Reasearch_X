import re
import string
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Fallback English stop words from scikit-learn (standard, deterministic 318 words)
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _STOP_WORDS = set(ENGLISH_STOP_WORDS)
except Exception:
    _STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
        "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself", "no", "nor",
        "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
        "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
        "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    }

_SPACY_TRIED = False
_SPACY_NLP: Optional[Any] = None

def get_spacy_nlp() -> Optional[Any]:
    """
    Safely attempts to load the spaCy model without crashing application startup
    if spaCy C extensions are blocked by Windows Application Control policies.
    """
    global _SPACY_TRIED, _SPACY_NLP
    if _SPACY_TRIED:
        return _SPACY_NLP
        
    _SPACY_TRIED = True
    try:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        logger.info("spaCy en_core_web_sm model loaded successfully.")
    except Exception as e:
        logger.warning(
            "spaCy could not be loaded (%s). Using deterministic scikit-learn / regex NLP fallback.",
            e
        )
        _SPACY_NLP = None
    return _SPACY_NLP

def is_spacy_available() -> bool:
    """Returns True if the spaCy pipeline is operational, False otherwise."""
    return get_spacy_nlp() is not None

def select_and_clean_text(paper) -> str:
    """
    Normalizes title and abstract: lowercases, removes URLs, emails,
    punctuation, and collapses excess whitespace. Handles empty fields safely.
    """
    if paper is None:
        return ""
    title = getattr(paper, "title", "") or ""
    abstract = getattr(paper, "abstract", "") or ""
    
    text = f"{title} {abstract}".strip()
    if not text:
        return ""
        
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _fallback_tokenize_and_filter(raw_text: str) -> str:
    """
    Deterministic NLP fallback:
    - Splits text into tokens.
    - Preserves scientific alphanumeric and hyphenated terms.
    - Removes pure numbers, punctuation, short (<3 chars) tokens, and standard stop words.
    - Avoids naive suffix-stripping heuristic that could distort domain terms (e.g., basis -> bas).
    """
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', raw_text.lower())
    filtered = [
        token for token in tokens
        if len(token) > 2
        and token not in _STOP_WORDS
        and not token.isnumeric()
    ]
    return " ".join(filtered)

def preprocess_corpus(papers: List[Any], min_length: int = 10) -> List[Dict[str, Any]]:
    """
    Preprocesses a list of paper objects for topic modeling and text mining.
    If spaCy is available, linguistic lemmatization and stop-word removal are performed.
    Otherwise, deterministic tokenization with scikit-learn standard English stop-word
    filtering is used.
    """
    processed = []
    nlp = get_spacy_nlp()
    
    for paper in papers:
        paper_id = getattr(paper, "id", None)
        raw_text = select_and_clean_text(paper)
        words = raw_text.split()
        if len(words) < min_length:
            processed.append({"paper_id": paper_id, "text": "", "status": "INSUFFICIENT_TEXT"})
            continue
            
        if nlp is not None:
            try:
                doc = nlp(raw_text)
                tokens = [
                    token.lemma_ for token in doc
                    if not token.is_stop and len(token.text) > 2 and not token.is_punct and not token.like_num
                ]
                clean_text = " ".join(tokens)
            except Exception as e:
                logger.warning("spaCy processing failed for paper %s (%s); falling back.", paper_id, e)
                clean_text = _fallback_tokenize_and_filter(raw_text)
        else:
            clean_text = _fallback_tokenize_and_filter(raw_text)
            
        status = "OK" if len(clean_text.split()) > 0 else "INSUFFICIENT_TEXT"
        processed.append({"paper_id": paper_id, "text": clean_text, "status": status})
        
    return processed
