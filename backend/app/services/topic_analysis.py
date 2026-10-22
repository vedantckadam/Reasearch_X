import numpy as np
from itertools import combinations
import logging

logger = logging.getLogger(__name__)

def calculate_coherence(topics, dtm, feature_names):
    """
    Simplified coherence calculation using Pointwise Mutual Information (PMI) proxy
    on the document-term matrix.
    """
    coherence_scores = []
    
    # Convert dtm to dense array of binary occurrences for simpler calculation
    # Only suitable for MVP sized corpora
    try:
        bin_dtm = (dtm.toarray() > 0).astype(int)
        doc_count = bin_dtm.shape[0]
        
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        for topic_idx, topic_data in topics.items():
            words = topic_data["words"][:10] # Use top 10 words for coherence
            word_indices = [feature_to_idx[w] for w in words if w in feature_to_idx]
            
            if len(word_indices) < 2:
                coherence_scores.append(0.0)
                continue
                
            topic_coherence = 0.0
            pairs = list(combinations(word_indices, 2))
            
            for w1, w2 in pairs:
                # Count docs with w1, w2, and both
                df_w1 = bin_dtm[:, w1].sum()
                df_w2 = bin_dtm[:, w2].sum()
                df_both = (bin_dtm[:, w1] * bin_dtm[:, w2]).sum()
                
                if df_both > 0:
                    # PMI = log(p(w1, w2) / (p(w1) * p(w2)))
                    # Using a simplified measure to avoid log(0)
                    pmi = np.log((df_both * doc_count) / (df_w1 * df_w2))
                    topic_coherence += pmi
            
            coherence_scores.append(topic_coherence / len(pairs))
            
        return float(np.mean(coherence_scores))
    except Exception as e:
        logger.error(f"Coherence calculation failed: {e}")
        return 0.0

def calculate_topic_statistics(doc_topics, topics):
    # doc_topics is N x K matrix
    dominant_topics = np.argmax(doc_topics, axis=1)
    
    stats = {}
    for topic_idx in topics.keys():
        stats[topic_idx] = {
            "dominant_papers": int(np.sum(dominant_topics == topic_idx)),
            "average_probability": float(np.mean(doc_topics[:, topic_idx])),
            "corpus_share": float(np.sum(doc_topics[:, topic_idx]) / np.sum(doc_topics))
        }
    return stats
