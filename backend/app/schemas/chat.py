from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list)
    analysis_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    model: str
    grounded_context_used: bool
    analysis_id: Optional[str] = None
