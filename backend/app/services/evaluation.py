from sqlalchemy.orm import Session
from app.models.paper_db import TopicAnalysis
from app.schemas.evaluation import BERTopicComparison, AblationResult, RetrievalMetric, EvidenceClaimMetric

def calculate_retrieval_metrics(session: Session, analysis_id: str, k: int = 5) -> RetrievalMetric:
    # Requires ground truth which we don't naturally have without human input.
    # Return a status indicating evaluation dataset is required.
    return RetrievalMetric(
        analysis_id=analysis_id,
        precision_at_k=0.0,
        recall_at_k=0.0,
        k=k,
        has_ground_truth=False
    )

def compare_lda_bertopic(session: Session, analysis_id: str) -> BERTopicComparison:
    # Fetch LDA coherence if available
    ta = session.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).first()
    lda_coh = ta.coherence_score if ta and ta.coherence_score else 0.0
    
    # We do not fabricate BERTopic results. We check if they exist (they don't yet).
    # Return unavailability.
    return BERTopicComparison(
        analysis_id=analysis_id,
        lda_coherence=lda_coh,
        bertopic_coherence=None,
        bertopic_available=False
    )

def calculate_evidence_claim_rate(session: Session, candidate_id: str) -> EvidenceClaimMetric:
    # In a real environment, this might map sentences in Nemotron output back to retrieved chunks.
    # For now, it's an evaluation stub for the framework.
    return EvidenceClaimMetric(
        candidate_id=candidate_id,
        supported_claims=0,
        total_claims=0,
        support_rate=0.0
    )
