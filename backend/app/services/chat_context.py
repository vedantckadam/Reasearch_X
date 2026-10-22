from typing import Optional, Tuple
from sqlalchemy.orm import Session
from app.models.paper_db import (
    Analysis,
    PaperDB,
    TopicAnalysis,
    Topic,
    GapCandidateDB,
    GroundedExplanationDB,
)

SYSTEM_CHAT_BASE_PROMPT = """You are the ResearchGap-X AI Assistant, an expert research methodology advisor and academic literature assistant integrated directly into the ResearchGap-X platform.

Core Operating Guidelines:
1. Grounding & Truthfulness: When active analysis context is provided below, strictly ground your explanations in the uploaded corpus, discovered topics, and identified gap candidates. Never fabricate paper titles, authors, citations, DOIs, statistical scores, or literature claims.
2. Unanalyzed State: If no active analysis context is available, guide the user on research gap methodology (e.g. topic rarity, SVD latent dimensions, temporal intersection sparsity) and encourage them to upload a seed PDF paper using the New Analysis Workspace on the home page.
3. Distinguish Signals from Facts: Clarify that ResearchGap-X generates analytical gap signals and hypotheses, not definitive proofs of literature absence. Remind researchers to validate gaps with full literature review.
4. Response Style: Keep responses professional, clear, and structured using clean Markdown. Be concise and actionable.
"""

def build_chat_context(db: Session, analysis_id: Optional[str]) -> Tuple[str, bool]:
    """
    Retrieves grounded research context for an analysis from the database.
    Returns (context_str, grounded_context_used).
    """
    if not analysis_id:
        return "Note: No active paper analysis is selected. General research methodology mode.", False

    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        return f"Note: Analysis ID '{analysis_id}' was not found in the database. General mode.", False

    # 1. Corpus overview
    papers = db.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).all()
    paper_count = len(papers)
    years = [p.year for p in papers if p.year]
    year_range = f"{min(years)} - {max(years)}" if years else "Unknown"
    
    sample_papers = []
    for p in papers[:5]:
        if p.title:
            yr = f" ({p.year})" if p.year else ""
            sample_papers.append(f"- '{p.title}'{yr}")

    seed_text = (analysis.seed_text or "").strip()
    seed_snippet = seed_text[:350] + ("..." if len(seed_text) > 350 else "") if seed_text else "None"

    # 2. Topic modeling
    topic_analysis = db.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).first()
    topic_summaries = []
    if topic_analysis:
        topics = db.query(Topic).filter(Topic.topic_analysis_id == topic_analysis.id).order_by(Topic.topic_idx).all()
        for t in topics[:6]:
            clean_words = []
            for item in (t.top_words or [])[:4]:
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    clean_words.append(str(item[0]))
                elif isinstance(item, dict):
                    clean_words.append(str(item.get("word", "")))
                elif isinstance(item, str):
                    clean_words.append(item)
            words = ", ".join(clean_words) if clean_words else "N/A"
            topic_summaries.append(f"- Topic {t.topic_idx} [{t.label}]: {words}")

    # 3. Gap candidates
    candidates = (
        db.query(GapCandidateDB)
        .filter(GapCandidateDB.analysis_id == analysis_id)
        .order_by(GapCandidateDB.final_gap_score.desc())
        .all()
    )
    candidate_summaries = []
    for c in candidates[:4]:
        topics_str = ", ".join(str(lbl) for lbl in (c.topic_labels or [])) if c.topic_labels else "Unlabeled"
        score_str = f"{c.final_gap_score:.2f}" if c.final_gap_score is not None else "N/A"
        conf_str = c.confidence_level or "Normal"
        candidate_summaries.append(
            f"- Candidate {c.candidate_id}: Topics [{topics_str}], Gap Score: {score_str}, Confidence: {conf_str}"
        )

    # 4. Explanations if any
    cand_ids = [c.candidate_id for c in candidates[:4]]
    explanations = (
        db.query(GroundedExplanationDB)
        .filter(GroundedExplanationDB.candidate_id.in_(cand_ids))
        .all()
    )
    explanation_summaries = []
    for e in explanations[:2]:
        explanation_summaries.append(
            f"- {e.candidate_id}: \"{e.candidate_gap_statement[:150]}...\""
        )

    lines = [
        f"=== ACTIVE ANALYSIS CONTEXT (ID: {analysis_id}) ===",
        f"Corpus Size: {paper_count} retrieved documents (Publication Years: {year_range})",
        f"Seed Paper Excerpt: {seed_snippet}",
    ]

    if sample_papers:
        lines.append("Sample Retrieved Papers:")
        lines.extend(sample_papers)

    if topic_summaries:
        lines.append(f"Discovered Topic Clusters ({len(topic_summaries)} topics):")
        lines.extend(topic_summaries)

    if candidate_summaries:
        lines.append(f"Top Hypothesized Research Gaps ({len(candidates)} total candidates):")
        lines.extend(candidate_summaries)

    if explanation_summaries:
        lines.append("Synthesized Grounded Explanations:")
        lines.extend(explanation_summaries)

    lines.append("=== END OF ACTIVE ANALYSIS CONTEXT ===")

    return "\n".join(lines), True
