import pytest
from fastapi.testclient import TestClient
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to ResearchGap-X API"}

def test_read_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "ResearchGap-X"

def test_upload_pdf():
    # Create a dummy PDF file
    test_pdf_path = "test_dummy.pdf"
    with open(test_pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 700 Td\n(Hello World) Tj\nET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000219 00000 n \n0000000312 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n399\n%%EOF\n")
    
    with open(test_pdf_path, "rb") as f:
        response = client.post("/api/upload", files={"file": ("test_dummy.pdf", f, "application/pdf")})
    
    assert response.status_code == 200
    assert "analysis_id" in response.json()
    assert response.json()["status"] == "QUEUED"
    
    os.remove(test_pdf_path)

def test_get_status():
    from app.models.paper_db import SessionLocal, Analysis
    session = SessionLocal()
    if not session.query(Analysis).filter(Analysis.id == "test-id-123").first():
        session.add(Analysis(id="test-id-123", status="QUEUED"))
        session.commit()
    session.close()
    
    response = client.get("/api/analysis/test-id-123/status")
    assert response.status_code == 200
    assert response.json()["status"] == "QUEUED"
