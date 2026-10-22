from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
from pydantic import BaseModel
import uuid
import logging

from app.models.paper_db import SessionLocal, Analysis, PaperDB, TopicAnalysis, Topic, DocumentTopic
from app.services.text_preprocessing import preprocess_corpus
from app.services.lda_model import build_document_term_matrix, train_lda, get_topic_words, get_document_topics
from app.services.topic_analysis import calculate_coherence, calculate_topic_statistics

logger = logging.getLogger(__name__)
router = APIRouter()

class TopicRunRequest(BaseModel):
    num_topics: int = 8

def run_topic_modeling_task(analysis_id: str, num_topics: int):
    session = SessionLocal()
    try:
        analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            return
            
        analysis.status = "TOPIC_MODELING_STARTED"
        session.commit()
        
        papers = session.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).all()
        
        # Preprocessing
        analysis.status = "PREPROCESSING"
        session.commit()
        processed = preprocess_corpus(papers)
        valid_docs = [p for p in processed if p["status"] == "OK"]
        
        if len(valid_docs) < 5:
            analysis.status = "TOPIC_MODELING_FAILED_INSUFFICIENT_TEXT"
            session.commit()
            return
            
        texts = [p["text"] for p in valid_docs]
        paper_ids = [p["paper_id"] for p in valid_docs]
        
        # DTM
        analysis.status = "BUILDING_MATRIX"
        session.commit()
        dtm, feature_names = build_document_term_matrix(texts)
        
        # LDA
        analysis.status = "TRAINING_LDA"
        session.commit()
        lda = train_lda(dtm, n_topics=num_topics)
        
        # Extract topics
        analysis.status = "CALCULATING_TOPICS"
        session.commit()
        topics = get_topic_words(lda, feature_names)
        
        # Doc-Topic matrix
        doc_topics = get_document_topics(lda, dtm)
        
        # Stats & Coherence
        analysis.status = "CALCULATING_COHERENCE"
        session.commit()
        coherence = calculate_coherence(topics, dtm, feature_names)
        stats = calculate_topic_statistics(doc_topics, topics)
        
        # Save to DB
        analysis.status = "SAVING_RESULTS"
        session.commit()
        
        # Clear previous topic analyses if re-running
        old_tas = session.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).all()
        for old_ta in old_tas:
            session.query(DocumentTopic).filter(DocumentTopic.topic_analysis_id == old_ta.id).delete()
            session.query(Topic).filter(Topic.topic_analysis_id == old_ta.id).delete()
            session.delete(old_ta)
        session.commit()
        
        topic_analysis_id = str(uuid.uuid4())
        ta = TopicAnalysis(
            id=topic_analysis_id,
            analysis_id=analysis_id,
            num_topics=num_topics,
            vocabulary_size=len(feature_names),
            coherence_score=coherence
        )
        session.add(ta)
        
        for idx, t_data in topics.items():
            words = t_data.get("words", [])
            weights = t_data.get("weights", [])
            paired_top_words = [
                [w, float(wt)] for w, wt in zip(words, weights)
            ] if weights else [[w, 0.0] for w in words]

            t = Topic(
                id=str(uuid.uuid4()),
                topic_analysis_id=topic_analysis_id,
                topic_idx=idx,
                label=t_data["label"],
                dominant_papers=stats[idx]["dominant_papers"],
                average_probability=stats[idx]["average_probability"],
                corpus_share=stats[idx]["corpus_share"],
                top_words=paired_top_words
            )
            session.add(t)
            
        for doc_idx, paper_id in enumerate(paper_ids):
            for topic_idx in range(num_topics):
                dt = DocumentTopic(
                    id=str(uuid.uuid4()),
                    topic_analysis_id=topic_analysis_id,
                    paper_id=paper_id,
                    topic_idx=topic_idx,
                    probability=float(doc_topics[doc_idx, topic_idx])
                )
                session.add(dt)
                
        analysis.status = "TOPIC_MODELING_COMPLETED"
        session.commit()
        
    except Exception as e:
        logger.error(f"Topic modeling failed: {e}")
        analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            analysis.status = "TOPIC_MODELING_FAILED"
            session.commit()
    finally:
        session.close()

@router.post("/analysis/{analysis_id}/topics/run")
async def run_topics(analysis_id: str, request: TopicRunRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_topic_modeling_task, analysis_id, request.num_topics)
    return {"status": "topic_modeling_started"}

@router.get("/analysis/{analysis_id}/topics")
async def get_topics(analysis_id: str):
    session = SessionLocal()
    ta = session.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).order_by(TopicAnalysis.created_at.desc()).first()
    if not ta:
        session.close()
        raise HTTPException(status_code=404, detail="Topic analysis not found")
        
    topics = session.query(Topic).filter(Topic.topic_analysis_id == ta.id).all()
    session.close()
    
    return {
        "num_topics": ta.num_topics,
        "vocabulary_size": ta.vocabulary_size,
        "coherence": ta.coherence_score,
        "topics": topics
    }

@router.get("/analysis/{analysis_id}/document-topics")
async def get_document_topics_route(analysis_id: str):
    session = SessionLocal()
    ta = session.query(TopicAnalysis).filter(TopicAnalysis.analysis_id == analysis_id).order_by(TopicAnalysis.created_at.desc()).first()
    if not ta:
        session.close()
        raise HTTPException(status_code=404, detail="Topic analysis not found")
        
    doc_topics = session.query(DocumentTopic).filter(DocumentTopic.topic_analysis_id == ta.id).all()
    session.close()
    
    return {"document_topics": doc_topics}
