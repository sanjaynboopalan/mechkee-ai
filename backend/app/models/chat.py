from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.models.search import Source, Citation

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: Optional[List[Source]] = []
    citations: Optional[List[Citation]] = []
    message_id: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_sources: bool = True
    max_results: int = 10
    search_type: str = "hybrid"

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    session_id: str
    message: ChatMessage
    sources: List[Source]
    citations: List[Citation]
    model_used: str
    response_time: Optional[float] = None

class ChatHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    total_messages: int
    created_at: datetime
    last_activity: datetime