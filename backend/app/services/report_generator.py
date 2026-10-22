import json
from sqlalchemy.orm import Session
from app.models.paper_db import (
    Analysis, PaperDB, TopicAnalysis, Topic, DocumentTopic,
    LatentDimensionDB, TemporalTopicDB, TemporalIntersectionDB,
    GapCandidateDB, GapEvidenceDB, GroundedExplanationDB
)
from datetime import datetime

def generate_json_report(session: Session, analysis_id: str) -> dict:
    analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise ValueError("Analysis not found")
        
    papers = session.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).all()
    
    ta = session.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).first()
    topics = []
    if ta:
        topics_db = session.query(Topic).filter(Topic.topic_analysis_id == ta.id).all()
        for t in topics_db:
            topics.append({
                "topic_idx": t.topic_idx,
                "label": t.label,
                "prevalence": t.corpus_share,
                "top_words": t.top_words
            })
            
    latent_dims = session.query(LatentDimensionDB).filter(LatentDimensionDB.analysis_id == analysis_id).all()
    latent_data = [{"idx": ld.dimension_index, "energy_ratio": ld.energy_ratio, "is_low_energy": ld.is_low_energy} for ld in latent_dims]
    
    temporal_data = []
    t_topics = session.query(TemporalTopicDB).filter(TemporalTopicDB.analysis_id == analysis_id).all()
    for tt in t_topics:
        temporal_data.append({
            "topic_idx": tt.topic_index,
            "year": tt.year,
            "avg_probability": tt.average_probability,
            "trend": tt.trend
        })
        
    gap_candidates = []
    candidates = session.query(GapCandidateDB).filter(GapCandidateDB.analysis_id == analysis_id).all()
    for c in candidates:
        evs = session.query(GapEvidenceDB).filter(GapEvidenceDB.candidate_id == c.candidate_id).all()
        exp = session.query(GroundedExplanationDB).filter(GroundedExplanationDB.candidate_id == c.candidate_id).first()
        
        gap_candidates.append({
            "candidate_id": c.candidate_id,
            "topics": c.topic_indices,
            "final_score": c.final_gap_score,
            "rarity_score": c.rarity_score,
            "intersection_score": c.intersection_score,
            "temporal_score": c.temporal_score,
            "limitation_score": c.limitation_score,
            "semantic_score": c.semantic_support_score,
            "evidence": [{"type": e.evidence_type, "text": e.evidence_text, "year": e.year} for e in evs],
            "explanation": exp.candidate_gap_statement if exp else None,
            "suggested_direction": exp.suggested_direction if exp else None
        })
        
    report = {
        "report_title": "ResearchGap-X Analysis Report",
        "generated_at": datetime.utcnow().isoformat(),
        "analysis_id": analysis_id,
        "corpus_size": len(papers),
        "topics": topics,
        "latent_dimensions": latent_data,
        "temporal_analysis": temporal_data,
        "candidate_gap_signals": gap_candidates,
        "limitations": "These are potential research-gap signals based on structured evidence, not definitive proofs of a research gap."
    }
    
    return report

def generate_pdf_report(session: Session, analysis_id: str) -> bytes:
    from fpdf import FPDF
    
    json_rep = generate_json_report(session, analysis_id)
    
    class PDF(FPDF):
        def header(self):
            self.set_font('helvetica', 'B', 16)
            self.cell(0, 10, 'ResearchGap-X Analysis Report', 0, 1, 'C')
            self.ln(5)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    
    # Metadata
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, f"Analysis ID: {json_rep['analysis_id']}", 0, 1)
    pdf.cell(0, 10, f"Corpus Size: {json_rep['corpus_size']} papers", 0, 1)
    pdf.cell(0, 10, f"Generated At: {json_rep['generated_at']}", 0, 1)
    
    pdf.ln(5)
    
    # Topics
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Extracted Topics", 0, 1)
    pdf.set_font("helvetica", "", 10)
    if not json_rep['topics']:
        pdf.cell(0, 8, "No topics available.", 0, 1)
    else:
        for t in json_rep['topics']:
            top_words = t.get('top_words')
            if not top_words or not isinstance(top_words, list):
                top_words_str = "No terms available."
            else:
                valid_words = []
                for w in top_words[:5]:
                    if isinstance(w, list) and len(w) >= 2 and isinstance(w[0], str) and isinstance(w[1], (int, float)):
                        valid_words.append(f"{w[0]} ({w[1]:.2f})")
                    elif isinstance(w, str):
                        valid_words.append(w)
                top_words_str = ", ".join(valid_words) if valid_words else "No valid terms."
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 8, f"Topic {t.get('topic_idx', '?')} ({t.get('label', 'Unknown')}) [Share: {t.get('prevalence', 0)*100:.1f}%]: {top_words_str}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Gap Candidates
    pdf.set_x(pdf.l_margin)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Candidate Gap Signals", 0, 1)
    pdf.set_font("helvetica", "", 10)
    
    if not json_rep['candidate_gap_signals']:
        pdf.cell(0, 8, "No gap candidates identified.", 0, 1)
    else:
        for c in json_rep['candidate_gap_signals']:
            pdf.set_x(pdf.l_margin)
            pdf.set_font("helvetica", "B", 11)
            pdf.cell(0, 8, f"Candidate: {c['candidate_id']}", 0, 1)
            pdf.set_font("helvetica", "", 10)
            pdf.cell(0, 8, f"Final Score: {c['final_score']:.3f} | Rarity: {c['rarity_score'] or 0:.3f} | Intersection: {c['intersection_score'] or 0:.3f}", 0, 1)
            if c.get('explanation'):
                pdf.set_font("helvetica", "I", 10)
                pdf.multi_cell(pdf.epw, 8, f"Explanation: {c['explanation']}", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("helvetica", "", 10)
                if c.get('suggested_direction'):
                    pdf.multi_cell(pdf.epw, 8, f"Suggested Direction: {c['suggested_direction']}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    pdf.ln(5)
    
    # Limitations
    pdf.set_x(pdf.l_margin)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "LIMITATIONS:", 0, 1)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(pdf.epw, 8, json_rep['limitations'], new_x="LMARGIN", new_y="NEXT")
    
    return bytes(pdf.output())
