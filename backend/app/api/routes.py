from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
import uuid
import os
import shutil
from pydantic import BaseModel
from typing import List, Optional

from app.services.pdf_extraction import extract_text_from_pdf
from app.services.literature_retrieval import LiteratureRetrievalOrchestrator
from app.models.paper_db import SessionLocal, Analysis, PaperDB

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class AnalysisStartRequest(BaseModel):
    analysis_id: str
    max_papers: int = 100

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    analysis_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{analysis_id}.pdf")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        extracted_text = extract_text_from_pdf(file_path)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="PDF is empty or scanned without OCR text")
            
        session = SessionLocal()
        analysis = Analysis(id=analysis_id, status="QUEUED", seed_text=extracted_text[:10000])
        session.add(analysis)
        session.commit()
        session.close()
        
        return {"analysis_id": analysis_id, "status": "QUEUED"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analysis/start")
async def start_analysis(request: AnalysisStartRequest, background_tasks: BackgroundTasks):
    session = SessionLocal()
    analysis = session.query(Analysis).filter(Analysis.id == request.analysis_id).first()
    session.close()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
        
    orchestrator = LiteratureRetrievalOrchestrator()
    background_tasks.add_task(orchestrator.run_retrieval, request.analysis_id, analysis.seed_text)
    
    return {"status": "processing_started"}

@router.get("/analysis/{analysis_id}/status")
async def get_status(analysis_id: str):
    session = SessionLocal()
    analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
    session.close()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
        
    return {"status": analysis.status, "progress": 50 if analysis.status in ["RETRIEVING", "DEDUPLICATING", "RANKING"] else (100 if analysis.status == "COMPLETED" else 0)}

@router.get("/analysis/{analysis_id}/papers")
async def get_papers(analysis_id: str):
    session = SessionLocal()
    papers = session.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).order_by(PaperDB.relevance_score.desc()).all()
    session.close()
    
    return {
        "total_papers": len(papers),
        "papers": papers
    }

@router.get("/analysis/{analysis_id}/retrieval")
async def get_retrieval(analysis_id: str):
    # Mocking retrieval stats for MVP endpoint requirement
    session = SessionLocal()
    papers = session.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).all()
    session.close()
    
    return {
        "final_corpus_size": len(papers),
        "retrieval_status": "COMPLETED" if papers else "QUEUED/FAILED",
    }
