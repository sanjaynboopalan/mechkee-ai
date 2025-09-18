"""
ChatGPT-style conversation API with memory and context
Enhanced with advanced search, response formatting, and source citations
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import time
from datetime import datetime
import uuid
import groq
import os
import asyncio

# Import enhanced AI components
from ..core.response_formatter import (
    response_formatter, 
    ResponsePreferences, 
    DetailLevel, 
    ResponseStyle
)
from ..core.web_search_engine import web_search_engine

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

# Initialize Groq client for ChatGPT-style responses
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# In-memory conversation storage (in production, use Redis/database)
conversations: Dict[str, List[Dict]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class EnhancedChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    search_web: bool = Field(default=True, description="Search web for current information")
    detail_level: str = Field(default="auto", description="Response detail: minimal, concise, detailed, extensive, auto")
    response_style: str = Field(default="conversational", description="Style: professional, conversational, academic, simple")
    include_sources: bool = Field(default=True, description="Include source citations")
    max_sources: int = Field(default=5, description="Maximum sources to include")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    conversation_length: int
    processing_time: float

class EnhancedChatResponse(BaseModel):
    response: str
    session_id: str
    conversation_length: int
    processing_time: float
    word_count: int
    sources: List[Dict[str, Any]]
    key_points: List[str]
    confidence_score: float
    search_metadata: Optional[Dict[str, Any]] = None

@router.post("/", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """
    ChatGPT-style conversational AI with memory and context
    """
    try:
        print(f"🔄 Received chat request: {request.message}")
        start_time = time.time()
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        print(f"📝 Session ID: {session_id}")
        
        # Get or create conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        conversation_history = conversations[session_id]
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        print(f"💬 Added user message to history")
        
        # Prepare messages for Groq API (Simplified for stability)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user questions."
            }
        ]
        
        # Add conversation history (keep last 15 messages for better context)
        recent_history = conversation_history[-15:]
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Get AI response using Groq with timeout protection
        try:
            print("🤖 Calling Groq API...")
            chat_completion = groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=1000,
                timeout=20  # 20 second timeout
            )
            
            ai_response = chat_completion.choices[0].message.content
            print(f"✅ Got response from Groq: {ai_response[:50]}...")
            
        except Exception as groq_error:
            print(f"❌ Groq API error: {groq_error}")
            logger.error(f"Groq API error: {groq_error}")
            ai_response = "I'm experiencing technical difficulties at the moment. Please try your question again!"
        
        # Add AI response to conversation history
        conversation_history.append({
            "role": "assistant", 
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep conversation history manageable (last 20 messages)
        if len(conversation_history) > 20:
            conversations[session_id] = conversation_history[-20:]
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            conversation_length=len(conversation_history),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat session error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat session failed: {str(e)}")

@router.post("/enhanced", response_model=EnhancedChatResponse)
async def enhanced_chat_with_ai(request: EnhancedChatRequest):
    """
    Enhanced ChatGPT-style AI with web search, source citations, and advanced formatting
    """
    try:
        print(f"🔄 Enhanced chat request: {request.message}")
        start_time = time.time()
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        print(f"📝 Enhanced Session ID: {session_id}")
        
        # Get or create conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        conversation_history = conversations[session_id]
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Determine response preferences
        preferences = ResponsePreferences(
            detail_level=DetailLevel(request.detail_level) if request.detail_level != "auto" else None,
            style=ResponseStyle(request.response_style),
            include_sources=request.include_sources,
            max_paragraphs=10,
            prefer_lists=False,
            include_examples=True
        )
        
        # Auto-determine detail level if set to auto
        if request.detail_level == "auto":
            preferences.detail_level = response_formatter.determine_optimal_detail_level(
                request.message, preferences
            )
        
        # Search web for current information if requested
        search_results = []
        search_metadata = None
        
        if request.search_web:
            try:
                print("🔍 Searching web for current information...")
                async with web_search_engine as search_engine:
                    search_results, search_metadata = await search_engine.search_web(
                        query=request.message,
                        max_results=request.max_sources,
                        search_type='general',
                        include_content=True
                    )
                print(f"✅ Found {len(search_results)} search results")
            except Exception as search_error:
                print(f"⚠️ Search failed: {search_error}")
                logger.warning(f"Web search failed: {search_error}")
        
        # Prepare enhanced context for AI
        context_parts = []
        
        # Add search results to context
        if search_results:
            context_parts.append("Current information from web search:")
            for i, result in enumerate(search_results[:3], 1):
                context_parts.append(f"{i}. {result.title}")
                context_parts.append(f"   Source: {result.domain}")
                context_parts.append(f"   Content: {result.content[:300]}...")
                context_parts.append("")
        
        # Prepare messages for Groq API
        system_prompt = f"""You are an advanced AI assistant that provides accurate, well-researched responses. 

Response Guidelines:
- Detail Level: {preferences.detail_level.value}
- Style: {preferences.style.value}
- Be factual and cite information appropriately
- Remove all markdown formatting (no *, #, **, etc.)
- Write in clean, readable prose
- If web search results are provided, integrate them naturally

{chr(10).join(context_parts) if context_parts else ''}

Provide a clear, informative response without any markdown formatting."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (keep last 10 messages for context)
        recent_history = conversation_history[-10:]
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Get AI response using Groq
        try:
            print("🤖 Calling Groq API with enhanced context...")
            chat_completion = groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=2000,
                timeout=30
            )
            
            raw_response = chat_completion.choices[0].message.content
            print(f"✅ Got enhanced response from Groq")
            
        except Exception as groq_error:
            print(f"❌ Groq API error: {groq_error}")
            logger.error(f"Enhanced Groq API error: {groq_error}")
            raw_response = "I'm experiencing technical difficulties. Please try your question again!"
        
        # Format response using advanced formatter
        print("🎨 Formatting response...")
        formatted_response = response_formatter.format_response_with_sources(
            content=raw_response,
            sources=[{
                'title': result.title,
                'url': result.url,
                'snippet': result.snippet,
                'credibility_score': result.credibility_score,
                'domain': result.domain
            } for result in search_results] if search_results else [],
            preferences=preferences
        )
        
        # Create source citations
        source_citations = []
        if search_results and request.include_sources:
            citations = web_search_engine.create_citations(search_results)
            source_citations = [
                {
                    'id': citation.id,
                    'title': citation.title,
                    'url': citation.url,
                    'snippet': citation.snippet,
                    'credibility': citation.credibility,
                    'cite_number': citation.cite_number
                }
                for citation in citations
            ]
        
        # Add formatted response to conversation history
        conversation_history.append({
            "role": "assistant",
            "content": formatted_response.content,
            "timestamp": datetime.now().isoformat(),
            "sources": source_citations,
            "confidence_score": formatted_response.confidence_score
        })
        
        # Keep conversation manageable
        if len(conversation_history) > 20:
            conversations[session_id] = conversation_history[-20:]
        
        processing_time = time.time() - start_time
        
        return EnhancedChatResponse(
            response=formatted_response.content,
            session_id=session_id,
            conversation_length=len(conversation_history),
            processing_time=processing_time,
            word_count=formatted_response.word_count,
            sources=source_citations,
            key_points=formatted_response.key_points,
            confidence_score=formatted_response.confidence_score,
            search_metadata=search_metadata.__dict__ if search_metadata else None
        )
        
    except Exception as e:
        logger.error(f"Enhanced chat session error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced chat failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for ChatGPT clone service"""
    return {"status": "ok", "service": "chatgpt-clone", "timestamp": datetime.now().isoformat()}

@router.get("/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "conversation": conversations[session_id],
        "message_count": len(conversations[session_id])
    }

@router.delete("/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversations:
        del conversations[session_id]
        return {"message": f"Conversation {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.get("/sessions")
async def list_active_sessions():
    """List all active conversation sessions"""
    return {
        "active_sessions": list(conversations.keys()),
        "total_sessions": len(conversations)
    }