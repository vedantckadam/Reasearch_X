import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict

from app.models.paper_db import SessionLocal
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_context import SYSTEM_CHAT_BASE_PROMPT, build_chat_context
from app.services.nemotron_client import NemotronClient

logger = logging.getLogger(__name__)

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/chat", response_model=ChatResponse)
def chat_route(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Dedicated research-aware AI chat endpoint for the ResearchGap-X assistant.
    Grounded in active paper/analysis context when analysis_id is provided.
    Reuses existing NemotronClient and OpenRouter integration securely.
    """
    clean_message = req.message.strip()
    if not clean_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty or whitespace only.",
        )

    # 1. Retrieve grounded research context if analysis_id is provided
    context_str, grounded_used = build_chat_context(db, req.analysis_id)
    system_prompt = f"{SYSTEM_CHAT_BASE_PROMPT}\n\n{context_str}"

    # 2. Limit history to last 6 messages to preserve token window and fast responses
    recent_history = req.history[-6:] if req.history else []
    
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in recent_history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": clean_message})

    # 3. Call NemotronClient
    client = NemotronClient()
    try:
        completion = client.post_chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
        reply = (completion.get("content") or "").strip()
        if not reply:
            reply = "I could not generate a response for this query. Please try rephrasing your research question."

        return ChatResponse(
            reply=reply,
            model=client.model,
            grounded_context_used=grounded_used,
            analysis_id=req.analysis_id,
        )
    except ValueError as ve:
        msg = str(ve)
        if "not configured" in msg.lower() or "api_key" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=msg,
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI chat error: {msg}",
        )
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service authentication failed: {str(pe)}",
        )
    except TimeoutError as te:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"AI provider timeout: {str(te)}",
        )
    except (RuntimeError, ConnectionError) as ce:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI provider communication error: {str(ce)}",
        )
    except Exception as e:
        logger.error(f"Unexpected chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI assistant encountered an unexpected error processing your request.",
        )
