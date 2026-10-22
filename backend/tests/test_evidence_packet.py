from app.schemas.gap import GapCandidate, GapEvidence
from app.services.rag_chunking import RAGChunk
from app.services.evidence_packet import build_evidence_packet

def test_build_evidence_packet():
    cand = GapCandidate(
        candidate_id="c1",
        final_gap_score=0.8,
        evidence_references=[
            GapEvidence(
                evidence_type="limitation",
                evidence_text="Future work should...",
                paper_id="p1"
            )
        ]
    )
    
    chunks = [
        (RAGChunk("ch1", "p1", "a1", "T1", 2020, "abs", "Future work should explore..."), 0.9)
    ]
    
    packet = build_evidence_packet(cand, chunks)
    
    assert packet["candidate_id"] == "c1"
    assert len(packet["structured_evidence"]) == 1
    assert packet["structured_evidence"][0]["text"] == "Future work should..."
    assert len(packet["retrieved_chunks"]) == 1
    assert packet["retrieved_chunks"][0]["similarity"] == 0.9
    assert packet["source_metadata"][0]["paper_id"] == "p1"
