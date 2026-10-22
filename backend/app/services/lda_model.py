from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Domain-agnostic academic meta-words that dilute topic coherence
ACADEMIC_STOP_WORDS = {
    "study", "analysis", "based", "paper", "approach", "results", "model",
    "proposed", "method", "methods", "using", "presents", "evaluated", "evaluation",
    "data", "performance", "framework", "research", "use", "used", "different",
    "provide", "provides", "including", "new", "demonstrate", "order", "high",
    "low", "level", "levels", "process", "various"
}

_EXTENDED_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(ACADEMIC_STOP_WORDS))

def build_document_term_matrix(processed_texts, max_features=5000, min_df=2, max_df=0.95):
    vectorizer = CountVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        stop_words=_EXTENDED_STOP_WORDS
    )
    dtm = vectorizer.fit_transform(processed_texts)
    return dtm, vectorizer.get_feature_names_out()

def train_lda(dtm, n_topics=8, random_state=42):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state, learning_method="batch")
    lda.fit(dtm)
    return lda

def get_topic_words(lda, feature_names, n_top_words=15):
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        weights = weights / weights.sum()
        
        # Determine informative label candidates
        distinctive_features = [
            w for w in top_features
            if w.lower() not in ACADEMIC_STOP_WORDS and len(w) > 2
        ]
        
        # Check topic interpretability:
        # Require at least 2 distinct substantive features and non-flat weight distribution
        is_flat_noise = (len(weights) >= 2 and float(np.std(weights)) < 1e-4)
        if len(distinctive_features) < 2 or is_flat_noise or (len(weights) > 0 and weights[0] < 0.04):
            label = f"Unresolved Topic {topic_idx + 1}"
        else:
            label = f"{distinctive_features[0].capitalize()} & {distinctive_features[1].capitalize()}"
            
        topics[topic_idx] = {
            "words": top_features,
            "weights": weights.tolist(),
            "label": label
        }
    return topics

def get_document_topics(lda, dtm):
    return lda.transform(dtm)
