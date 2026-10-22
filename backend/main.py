from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import load_project_env, get_llm_diagnostic

# Ensure project environment variables are loaded from absolute project location
load_project_env()

from app.api.routes import router as api_router

app = FastAPI(title="ResearchGap-X API", version="0.1.0")

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
from app.api.topic_routes import router as topic_router
app.include_router(topic_router, prefix="/api")
from app.api.latent_routes import router as latent_router
app.include_router(latent_router, prefix="/api/analysis")
from app.api.gap_routes import router as gap_router
app.include_router(gap_router, prefix="/api/analysis")
from app.api.rag_routes import router as rag_router
app.include_router(rag_router, prefix="/api/analysis")
from app.api.report_routes import router as report_router
app.include_router(report_router, prefix="/api/analysis")
from app.api.chat_routes import router as chat_router
app.include_router(chat_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to ResearchGap-X API"}

@app.get("/health")
def read_health():
    return {"status": "ok", "app": "ResearchGap-X", "version": "0.1.0"}

@app.get("/api/llm/status")
@app.get("/api/analysis/llm/status")
def read_llm_status():
    """Safe diagnostic endpoint reporting LLM configuration status without exposing credentials."""
    return get_llm_diagnostic()
