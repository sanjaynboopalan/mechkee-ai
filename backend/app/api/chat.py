from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uuid
import logging
from datetime import datetime

from app.core.rag_pipeline import RAGPipeline
from app.models.chat import ChatMessage, ChatResponse, ChatSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# In-memory session storage (use Redis in production)
chat_sessions = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context_limit: Optional[int] = 10
    use_live_web: Optional[bool] = True

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    use_live_web: Optional[bool] = True

@router.post("/", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
    ChatGPT-style conversation with context and live web search
    """
    try:
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        # Get or create session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = ChatSession(
                session_id=session_id,
                messages=[],
                created_at=datetime.utcnow()
            )
        
        session = chat_sessions[session_id]
        
        # Add user message to history
        user_message = ChatMessage(
            role="user",
            content=chat_request.message,
            timestamp=datetime.utcnow()
        )
        session.messages.append(user_message)
        
        # Get recent context for better conversation flow
        recent_messages = session.messages[-chat_request.context_limit:] if chat_request.context_limit > 0 else []
        
        # Generate response using RAG with live web search
        result = await rag_pipeline.search_and_generate(
            query=chat_request.message,
            max_results=8,
            include_citations=True,
            search_type="hybrid",
            use_live_web=chat_request.use_live_web
        )
        
        # Add assistant response to history
        assistant_message = ChatMessage(
            role="assistant",
            content=result["answer"],
            timestamp=datetime.utcnow(),
            sources=result.get("sources", []),
            citations=result.get("citations", [])
        )
        session.messages.append(assistant_message)
        
        logger.info(f"✅ Chat completed for session {session_id}")
        
        return ChatResponse(
            session_id=session_id,
            message=assistant_message,
            sources=result.get("sources", []),
            citations=result.get("citations", []),
            model_used=result.get("model_used", "groq"),
            search_time=result.get("search_time", 0),
            total_sources=result.get("total_sources", 0)
        )
        
    except Exception as e:
        logger.error(f"❌ Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.get("/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """
    Retrieve chat session history
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return chat_sessions[session_id]

@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """
    Delete a chat session
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del chat_sessions[session_id]
    return {"message": "Session deleted successfully"}

@router.get("/sessions")
async def list_chat_sessions():
    """
    List all active chat sessions
    """
    return {
        "sessions": [
            {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "message_count": len(session.messages)
            }
            for session in chat_sessions.values()
        ]
    }

@router.post("/search", response_model=Dict[str, Any])
async def perplexity_style_search(search_request: SearchRequest):
    """
    Perplexity-style search with live web content and AI-generated answers
    """
    try:
        logger.info(f"🔍 Starting Perplexity-style search for: {search_request.query}")
        
        # Use RAG pipeline for direct search with live web scraping
        result = await rag_pipeline.search_and_generate(
            query=search_request.query,
            max_results=search_request.max_results,
            include_citations=True,
            search_type="hybrid",
            use_live_web=search_request.use_live_web
        )
        
        logger.info(f"✅ Search completed: {result.get('total_sources', 0)} sources found")
        
        return {
            "query": search_request.query,
            "answer": result["answer"],
            "sources": result["sources"],
            "citations": result["citations"],
            "search_time": result["search_time"],
            "model_used": result["model_used"],
            "search_type": result["search_type"],
            "total_sources_found": result["total_sources"],
            "live_sources": result.get("live_sources", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Perplexity search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")