from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, JSONResponse
from sqlalchemy.orm import Session
from app.models.paper_db import SessionLocal
from app.services.report_generator import generate_json_report, generate_pdf_report

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/{analysis_id}/report/json")
def get_json_report(analysis_id: str, db: Session = Depends(get_db)):
    try:
        report = generate_json_report(db, analysis_id)
        return JSONResponse(content=report)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{analysis_id}/report/pdf")
def get_pdf_report(analysis_id: str, db: Session = Depends(get_db)):
    try:
        pdf_bytes = generate_pdf_report(db, analysis_id)
        return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=report_{analysis_id}.pdf"})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
