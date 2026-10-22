import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.models.paper_db import (
    Topic, DocumentTopic, TemporalTopicDB, TemporalIntersectionDB, 
    LatentDimensionDB, LatentTopicLoadingDB, PaperDB, GapCandidateDB, GapEvidenceDB,
    TopicAnalysis
)
from app.schemas.gap import GapCandidate, GapEvidence, GapEngineConfig, GapRunResponse

def calculate_rarity_score(
    session: Session, analysis_id: str, topic_idx: int, config: GapEngineConfig,
    total_docs: int
) -> Tuple[Optional[float], List[GapEvidence]]:
    """
    R Component: Rarity / Low Coverage
    R(t) = 1 - (docs_with_topic / total_docs)
    """
    if total_docs == 0:
        return None, []
        
    topic = session.query(Topic).join(TopicAnalysis, Topic.topic_analysis_id == TopicAnalysis.id).filter(
        TopicAnalysis.analysis_id == analysis_id, 
        Topic.topic_idx == topic_idx
    ).first()
    
    if not topic:
        return None, []
        
    coverage = topic.corpus_share  # Assume already calculated as (docs_with_topic / total_docs)
    
    # If not present or we need to recalculate explicitly using threshold:
    docs_count = session.query(DocumentTopic).join(TopicAnalysis, DocumentTopic.topic_analysis_id == TopicAnalysis.id).filter(
        TopicAnalysis.analysis_id == analysis_id,
        DocumentTopic.topic_idx == topic_idx,
        DocumentTopic.probability >= config.topic_presence_threshold
    ).count()
    
    coverage = docs_count / total_docs
    rarity = 1.0 - coverage
    
    evidence = GapEvidence(
        evidence_type="low_coverage",
        topic_index=topic_idx,
        evidence_metadata={
            "coverage": coverage,
            "document_count": docs_count,
            "total_documents": total_docs,
            "threshold": config.topic_presence_threshold
        }
    )
    
    return rarity, [evidence]


def calculate_intersection_sparsity(
    session: Session, analysis_id: str, topic_i: int, topic_j: int, 
    config: GapEngineConfig, total_docs: int
) -> Tuple[Optional[float], List[GapEvidence]]:
    """
    I Component: Intersection Sparsity
    """
    if total_docs == 0:
        return None, []
        
    # Get the latest year intersection for global view, or aggregate across years
    intersections = session.query(TemporalIntersectionDB).filter(
        TemporalIntersectionDB.analysis_id == analysis_id,
        TemporalIntersectionDB.topic_i == min(topic_i, topic_j),
        TemporalIntersectionDB.topic_j == max(topic_i, topic_j)
    ).all()
    
    if not intersections:
        # If there's literally no intersection record, support is 0 -> highly sparse
        evidence = GapEvidence(
            evidence_type="sparse_intersection",
            topic_pair=[topic_i, topic_j],
            evidence_metadata={"support_count": 0, "total_documents": total_docs}
        )
        return 1.0, [evidence]
        
    # Calculate overall support
    max_support = max([i.support_count for i in intersections]) if intersections else 0
    intersection_coverage = max_support / total_docs
    sparsity = 1.0 - intersection_coverage
    
    evidence = GapEvidence(
        evidence_type="sparse_intersection",
        topic_pair=[topic_i, topic_j],
        evidence_metadata={
            "max_yearly_support": max_support,
            "total_documents": total_docs,
            "sparsity": sparsity
        }
    )
    
    return sparsity, [evidence]

def calculate_temporal_score(
    session: Session, analysis_id: str, topic_idx: int, config: GapEngineConfig
) -> Tuple[Optional[float], List[GapEvidence]]:
    """
    T Component: Temporal Signal
    """
    temporal_records = session.query(TemporalTopicDB).filter(
        TemporalTopicDB.analysis_id == analysis_id,
        TemporalTopicDB.topic_index == topic_idx
    ).order_by(TemporalTopicDB.year).all()
    
    if len(temporal_records) < config.minimum_temporal_points:
        return None, []
        
    # Example logic: score higher if trend is "declining" (gap opening) 
    # or "emerging" (new gap)
    trends = [r.trend for r in temporal_records if r.trend]
    if not trends:
        return None, []
        
    latest_trend = temporal_records[-1].trend
    
    score = 0.5 # Neutral
    if latest_trend == "declining":
        score = 0.8
    elif latest_trend == "emerging" or latest_trend == "growing":
        score = 0.7
    elif latest_trend == "stable":
        score = 0.3
        
    evidence = GapEvidence(
        evidence_type="temporal_trend",
        topic_index=topic_idx,
        year=temporal_records[-1].year,
        evidence_text=f"Observed trend: {latest_trend}",
        evidence_metadata={
            "data_points": len(temporal_records),
            "latest_trend": latest_trend
        }
    )
    
    return score, [evidence]


def calculate_limitation_score(
    session: Session, analysis_id: str, topic_idx: int, config: GapEngineConfig
) -> Tuple[Optional[float], List[GapEvidence]]:
    """
    L Component: Limitation / Future-Work Evidence
    Searches paper abstracts associated with this topic for limitation phrases.
    """
    # Find papers strongly associated with this topic
    doc_topics = session.query(DocumentTopic).join(TopicAnalysis, DocumentTopic.topic_analysis_id == TopicAnalysis.id).filter(
        TopicAnalysis.analysis_id == analysis_id,
        DocumentTopic.topic_idx == topic_idx,
        DocumentTopic.probability >= config.topic_presence_threshold
    ).all()
    
    paper_ids = [dt.paper_id for dt in doc_topics]
    if not paper_ids:
        return None, []
        
    papers = session.query(PaperDB).filter(PaperDB.id.in_(paper_ids)).all()
    
    limitation_phrases = [
        "limitation", "limitations", "future work", "future research", 
        "further research", "remains unexplored", "not addressed", 
        "not considered", "requires further investigation", "should be investigated"
    ]
    
    evidence_list = []
    matches = 0
    
    for paper in papers:
        text_to_search = (paper.abstract or "").lower() + " " + (paper.title or "").lower()
        if not text_to_search.strip():
            continue
            
        for phrase in limitation_phrases:
            if phrase in text_to_search:
                matches += 1
                # Extract a small snippet around the phrase
                idx = text_to_search.find(phrase)
                start = max(0, idx - 50)
                end = min(len(text_to_search), idx + 50)
                snippet = text_to_search[start:end].replace('\n', ' ')
                
                evidence_list.append(
                    GapEvidence(
                        paper_id=paper.id,
                        evidence_type="limitation_phrase",
                        topic_index=topic_idx,
                        evidence_text=f"...{snippet}...",
                        evidence_metadata={"matched_phrase": phrase}
                    )
                )
                break # count paper once
                
    if not evidence_list:
        return None, []
        
    # Score based on proportion of documents containing limitations
    # Maxes out around 20% of topic documents having limitation phrases
    proportion = matches / len(papers)
    score = min(1.0, proportion / 0.20)
    
    return score, evidence_list

def calculate_semantic_score(
    session: Session, analysis_id: str, topic_idx: int, config: GapEngineConfig
) -> Tuple[Optional[float], List[GapEvidence]]:
    """
    S Component: Semantic Support
    For now, we return None (insufficient data state) because we don't have 
    a direct semantic vector retrieval baseline implemented natively yet.
    """
    return None, []


def run_gap_engine(session: Session, analysis_id: str, config: GapEngineConfig) -> GapRunResponse:
    # 1. Fetch total documents
    total_docs = session.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).count()
    if total_docs == 0:
        return GapRunResponse(
            analysis_id=analysis_id, candidate_count=0, threshold=config.topic_presence_threshold,
            weights={}, candidates=[]
        )
        
    # 2. Fetch Topics
    ta = session.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).first()
    if not ta:
        raise ValueError("No TopicAnalysis found for analysis_id")
        
    topics = session.query(Topic).filter(Topic.topic_analysis_id == ta.id).all()
    
    candidates = []
    
    # 3. Generate individual topic candidates
    for t in topics:
        c_id = f"{analysis_id}-T{t.topic_idx}"
        
        r_score, r_ev = calculate_rarity_score(session, analysis_id, t.topic_idx, config, total_docs)
        t_score, t_ev = calculate_temporal_score(session, analysis_id, t.topic_idx, config)
        l_score, l_ev = calculate_limitation_score(session, analysis_id, t.topic_idx, config)
        s_score, s_ev = calculate_semantic_score(session, analysis_id, t.topic_idx, config)
        
        all_ev = r_ev + t_ev + l_ev + s_ev
        
        numerator = 0.0
        denominator = 0.0
        
        if r_score is not None:
            numerator += config.rarity_weight * r_score
            denominator += config.rarity_weight
        if t_score is not None:
            numerator += config.temporal_weight * t_score
            denominator += config.temporal_weight
        if l_score is not None:
            numerator += config.limitation_weight * l_score
            denominator += config.limitation_weight
        if s_score is not None:
            numerator += config.semantic_weight * s_score
            denominator += config.semantic_weight
            
        final_score = numerator / denominator if denominator > 0 else 0.0
        
        signals = []
        if r_score and r_score > 0.7: signals.append("High rarity")
        if t_score and t_score > 0.7: signals.append("Temporal change observed")
        if l_score and l_score > 0.5: signals.append("Limitation evidence detected")
        
        confidence = "Low"
        if len(all_ev) > 5: confidence = "Medium"
        if len(all_ev) > 15: confidence = "High"
        
        if denominator > 0 and len(all_ev) >= config.minimum_evidence_count:
            candidates.append(GapCandidate(
                candidate_id=c_id,
                topic_indices=[t.topic_idx],
                topic_labels=[t.label],
                rarity_score=r_score,
                temporal_score=t_score,
                limitation_score=l_score,
                semantic_support_score=s_score,
                final_gap_score=final_score,
                evidence_count=len(all_ev),
                confidence_level=confidence,
                evidence_references=all_ev,
                reasoning_signals=signals
            ))
            
    # 4. Generate Topic Pair Candidates (Intersection)
    topic_map = {t.topic_idx: t for t in topics}
    intersections = session.query(TemporalIntersectionDB).filter(TemporalIntersectionDB.analysis_id == analysis_id).all()
    pairs_seen = set()
    for inter in intersections:
        t_i, t_j = min(inter.topic_i, inter.topic_j), max(inter.topic_i, inter.topic_j)
        if (t_i, t_j) in pairs_seen:
            continue
        pairs_seen.add((t_i, t_j))
        
        c_id = f"{analysis_id}-I{t_i}-{t_j}"
        
        i_score, i_ev = calculate_intersection_sparsity(session, analysis_id, t_i, t_j, config, total_docs)
        
        # Aggregate L-score for both topics
        l_score_i, l_ev_i = calculate_limitation_score(session, analysis_id, t_i, config)
        l_score_j, l_ev_j = calculate_limitation_score(session, analysis_id, t_j, config)
        
        l_score = None
        if l_score_i is not None and l_score_j is not None:
            l_score = (l_score_i + l_score_j) / 2.0
        elif l_score_i is not None:
            l_score = l_score_i
        elif l_score_j is not None:
            l_score = l_score_j
            
        all_ev = i_ev + l_ev_i + l_ev_j
        
        # Populate actual topic labels from topic_map
        label_i = topic_map[t_i].label if t_i in topic_map else f"Topic {t_i}"
        label_j = topic_map[t_j].label if t_j in topic_map else f"Topic {t_j}"
        topic_labels = [label_i, label_j]
        
        # Score calculation:
        # Absence of co-occurrence is an unobserved intersection signal, not confirmation of a gap.
        # Expected evidence denominator spans evaluated components (I and L) so missing signals do not saturate score to 1.00.
        expected_weight = config.intersection_weight + config.limitation_weight
        num_score = (config.intersection_weight * (i_score or 0.0)) + (config.limitation_weight * (l_score or 0.0))
        final_score = num_score / expected_weight if expected_weight > 0 else 0.0
        
        signals = []
        if i_score is not None:
            if i_score >= 0.99:
                signals.append("Unobserved topic intersection (0 co-occurrences in corpus)")
            else:
                signals.append(f"Sparse topic intersection ({i_score*100:.0f}% sparsity)")
                
        missing_evidence = []
        if l_score is not None and l_score > 0.0:
            signals.append(f"Limitation evidence confirmed ({l_score*100:.0f}%)")
        else:
            signals.append("No author limitation statements detected")
            missing_evidence.append("Author limitation statements")
            
        missing_evidence.append("Longitudinal temporal convergence")
        signals.append(f"Formula: ({config.intersection_weight:.2f}*I + {config.limitation_weight:.2f}*L) / {expected_weight:.2f}")

        # Grounding check from dominant paper counts
        dom_i = getattr(topic_map.get(t_i), "dominant_papers", 0) or 0
        dom_j = getattr(topic_map.get(t_j), "dominant_papers", 0) or 0
        total_grounded_docs = dom_i + dom_j

        # Confidence calculation & transparent rationale
        if l_score is not None and l_score > 0.4 and total_grounded_docs >= 4:
            confidence = "High"
            conf_rationale = f"High: Structural hole corroborated by documented author limitations and multi-paper topical support ({total_grounded_docs} dominant papers)."
        elif (l_score is not None and l_score > 0) or total_grounded_docs >= 4:
            confidence = "Medium"
            conf_rationale = f"Medium: Topic sparsity ({i_score*100:.0f}%) grounded across {total_grounded_docs} corpus documents, with partial corroboration."
        else:
            confidence = "Low"
            conf_rationale = f"Low: Unobserved topic intersection without corroborating author limitation excerpts (evidence count={len(all_ev)})."

        descriptive_statement = (
            f"Corpus analysis identified an unaddressed intersection between '{label_i}' and '{label_j}' "
            f"with {0 if (i_score or 0) >= 0.99 else 'sparse'} co-occurrences across {total_docs} analyzed documents. "
            f"Constituent topics exist in separate literature clusters without documented synthesis."
        )
        
        if expected_weight > 0 and len(all_ev) >= config.minimum_evidence_count:
            candidates.append(GapCandidate(
                candidate_id=c_id,
                topic_indices=[t_i, t_j],
                topic_labels=topic_labels,
                intersection_score=i_score,
                limitation_score=l_score,
                final_gap_score=final_score,
                evidence_count=len(all_ev),
                confidence_level=confidence,
                evidence_references=all_ev,
                reasoning_signals=signals,
                descriptive_statement=descriptive_statement,
                confidence_rationale=conf_rationale,
                missing_evidence=missing_evidence
            ))
            
    # Sort candidates deterministically
    candidates.sort(key=lambda x: (-x.final_gap_score, x.candidate_id))
    
    weights = {
        "rarity_weight": config.rarity_weight,
        "intersection_weight": config.intersection_weight,
        "temporal_weight": config.temporal_weight,
        "limitation_weight": config.limitation_weight,
        "semantic_weight": config.semantic_weight
    }
    
    return GapRunResponse(
        analysis_id=analysis_id,
        candidate_count=len(candidates),
        threshold=config.topic_presence_threshold,
        weights=weights,
        scoring_status="Incomplete (Semantic signal missing)",
        missing_signals=["Semantic (S)"],
        candidates=candidates
    )

def persist_gap_candidates(session: Session, analysis_id: str, response: GapRunResponse):
    # Clear existing candidates for this analysis
    session.query(GapEvidenceDB).filter(
        GapEvidenceDB.candidate_id.in_(
            session.query(GapCandidateDB.candidate_id).filter(GapCandidateDB.analysis_id == analysis_id)
        )
    ).delete(synchronize_session=False)
    
    session.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == analysis_id).delete(synchronize_session=False)
    session.commit()
    
    for cand in response.candidates:
        db_cand = GapCandidateDB(
            id=str(uuid.uuid4()),
            analysis_id=analysis_id,
            candidate_id=cand.candidate_id,
            topic_indices=cand.topic_indices,
            topic_labels=cand.topic_labels,
            latent_dimension_indices=cand.latent_dimension_indices,
            rarity_score=cand.rarity_score,
            intersection_score=cand.intersection_score,
            temporal_score=cand.temporal_score,
            limitation_score=cand.limitation_score,
            semantic_support_score=cand.semantic_support_score,
            final_gap_score=cand.final_gap_score,
            evidence_count=cand.evidence_count,
            confidence_level=cand.confidence_level,
            reasoning_signals=cand.reasoning_signals
        )
        session.add(db_cand)
        
        for ev in cand.evidence_references:
            db_ev = GapEvidenceDB(
                id=str(uuid.uuid4()),
                candidate_id=cand.candidate_id,
                paper_id=ev.paper_id,
                evidence_type=ev.evidence_type,
                topic_index=ev.topic_index,
                topic_pair=ev.topic_pair,
                year=ev.year,
                evidence_text=ev.evidence_text,
                evidence_metadata=ev.evidence_metadata
            )
            session.add(db_ev)
            
    session.commit()
