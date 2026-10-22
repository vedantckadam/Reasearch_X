import pytest
from app.services.text_preprocessing import select_and_clean_text, preprocess_corpus
from app.schemas.paper import PaperSchema
from app.services.lda_model import build_document_term_matrix, train_lda

class MockPaper:
    def __init__(self, id, title, abstract):
        self.id = id
        self.title = title
        self.abstract = abstract

def test_preprocessing():
    papers = [
        MockPaper(id="1", title="Federated Learning in Healthcare", abstract="This is a deep learning test.")
    ]
    
    text = select_and_clean_text(papers[0])
    assert "federated learning" in text
    
    processed = preprocess_corpus(papers, min_length=2)
    assert len(processed) == 1
    assert processed[0]["status"] == "OK"
    assert "learn" in processed[0]["text"] or "learning" in processed[0]["text"]

def test_dtm_and_lda():
    texts = [
        "federated learning privacy",
        "deep learning healthcare",
        "edge computing security",
        "federated learning edge healthcare"
    ]
    
    dtm, vocab = build_document_term_matrix(texts, min_df=1)
    assert dtm.shape[0] == 4
    assert len(vocab) > 3
    
    lda = train_lda(dtm, n_topics=2)
    assert lda.components_.shape == (2, len(vocab))
