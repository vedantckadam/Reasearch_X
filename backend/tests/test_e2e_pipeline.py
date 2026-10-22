import pytest
import os
import sys
from fastapi.testclient import TestClient

# Make sure we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.models.paper_db import SessionLocal, Analysis, PaperDB
from app.schemas.paper import PaperSchema

client = TestClient(app)

def test_full_pipeline_offline_mode():
    """
    End-to-End test of the ResearchGap-X pipeline using deterministic fixtures
    for the external literature retrieval phase.
    """
    # 1. Seed Paper PDF Upload
    test_pdf_path = "test_e2e.pdf"
    with open(test_pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 700 Td\n(Machine learning has limitations in explainability.) Tj\nET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000219 00000 n \n0000000312 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n410\n%%EOF\n")

    with open(test_pdf_path, "rb") as f:
        response = client.post("/api/upload", files={"file": ("test_e2e.pdf", f, "application/pdf")})
    
    assert response.status_code == 200
    analysis_id = response.json()["analysis_id"]
    os.remove(test_pdf_path)

    # 2. Mock Literature Retrieval (Bypass external API)
    # Instead of calling /analysis/start, we inject deterministic DB records
    db = SessionLocal()
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    analysis.status = "COMPLETED"
    
    # Inject 5 papers
    import uuid
    papers = [
        PaperDB(id=str(uuid.uuid4()), analysis_id=analysis_id, title="Deep Learning", abstract="We present a new deep learning model. A major limitation is explainability.", year=2020),
        PaperDB(id=str(uuid.uuid4()), analysis_id=analysis_id, title="Explainable AI", abstract="We solve the explainability problem in deep learning models.", year=2021),
        PaperDB(id=str(uuid.uuid4()), analysis_id=analysis_id, title="Transformers in NLP", abstract="Transformers are great for NLP but require further research in efficiency.", year=2022),
        PaperDB(id=str(uuid.uuid4()), analysis_id=analysis_id, title="Efficient Transformers", abstract="We present an efficient transformer.", year=2023),
        PaperDB(id=str(uuid.uuid4()), analysis_id=analysis_id, title="Future of AI", abstract="The future work should address both efficiency and explainability simultaneously.", year=2023),
        PaperDB(id=str(uuid.uuid4()), analysis_id=analysis_id, title="Deep Learning 2", abstract="We present a new deep learning model part 2. A major limitation is explainability again.", year=2020),
        PaperDB(id=str(uuid.uuid4()), analysis_id=analysis_id, title="Explainable AI 2", abstract="We solve the explainability problem in deep learning models part 2.", year=2021),
        PaperDB(id=str(uuid.uuid4()), analysis_id=analysis_id, title="Transformers in NLP 2", abstract="Transformers are great for NLP but require further research in efficiency part 2.", year=2022),
        PaperDB(id=str(uuid.uuid4()), analysis_id=analysis_id, title="Efficient Transformers 2", abstract="We present an efficient transformer part 2.", year=2023),
        PaperDB(id=str(uuid.uuid4()), analysis_id=analysis_id, title="Future of AI 2", abstract="The future work should address both efficiency and explainability simultaneously part 2.", year=2023),
    ]
    db.add_all(papers)
    db.commit()
    db.close()

    # 3. Topic Modeling
    from app.api.topic_routes import run_topic_modeling_task
    run_topic_modeling_task(analysis_id, num_topics=2)
    
    # 4. Latent & Temporal Analytics
    response = client.post(f"/api/analysis/{analysis_id}/latent/run")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

    # 5. Gap Generation
    response = client.post(f"/api/analysis/{analysis_id}/gap/run")
    assert response.status_code == 200
    
    # 6. Fetch Candidates
    response = client.get(f"/api/analysis/{analysis_id}/gaps")
    assert response.status_code == 200
    candidates = response.json()
    assert len(candidates) > 0
    with open("candidate_dump.json", "w") as f:
        import json
        json.dump(candidates[0], f, indent=2)

    # Clean up
    db = SessionLocal()
    db.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).delete()
    db.query(Analysis).filter(Analysis.id == analysis_id).delete()
    db.commit()
    db.close()
