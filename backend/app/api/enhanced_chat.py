"""
Enhanced Chat API with RAG Integration
Advanced conversational AI with document retrieval and context awareness
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import time
from datetime import datetime
import uuid
import asyncio
import os
from pathlib import Path

from ..core.enhanced_rag import get_rag_instance

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/enhanced-chat",
    tags=["enhanced-chat"]
)

# Conversation storage with RAG context
enhanced_conversations: Dict[str, Dict] = {}

class EnhancedChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    use_rag: bool = Field(True, description="Whether to use RAG for context-aware responses")
    search_web: bool = Field(False, description="Whether to search web for additional context")

class SourceInfo(BaseModel):
    title: str
    source: str
    relevance_score: float
    snippet: Optional[str] = None

class EnhancedChatResponse(BaseModel):
    response: str
    session_id: str
    conversation_length: int
    processing_time: float
    context_used: bool
    sources: List[SourceInfo] = []
    rag_enabled: bool
    response_type: str  # "standard", "rag_enhanced", "web_enhanced"

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str
    message: str

class KnowledgeBaseStats(BaseModel):
    total_documents: int
    total_chunks: int
    embedding_dimension: int
    last_updated: str

@router.post("/", response_model=EnhancedChatResponse)
async def enhanced_chat(request: EnhancedChatRequest):
    """
    Enhanced chat with RAG integration and context-aware responses
    """
    try:
        start_time = time.time()
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize session if new
        if session_id not in enhanced_conversations:
            enhanced_conversations[session_id] = {
                'messages': [],
                'rag_enabled': request.use_rag,
                'created_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat()
            }
        
        conversation = enhanced_conversations[session_id]
        
        # Add user message to conversation
        user_message = {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat(),
            "rag_requested": request.use_rag
        }
        conversation['messages'].append(user_message)
        conversation['last_activity'] = datetime.now().isoformat()
        
        # Initialize response variables
        ai_response = ""
        sources = []
        context_used = False
        response_type = "standard"
        
        try:
            if request.use_rag:
                # Use RAG for context-aware response
                rag_instance = get_rag_instance()
                rag_result = await rag_instance.search_and_generate(
                    query=request.message,
                    max_context_length=4000
                )
                
                ai_response = rag_result['response']
                sources = [
                    SourceInfo(
                        title=source['title'],
                        source=source['source'],
                        relevance_score=source['relevance_score'],
                        snippet=source.get('snippet')
                    )
                    for source in rag_result['sources']
                ]
                context_used = rag_result['context_used']
                response_type = "rag_enhanced" if context_used else "standard"
                
            else:
                # Standard chat without RAG
                from ..api.chat_session import groq_client
                
                # Prepare conversation history for context
                messages = [
                    {
                        "role": "system",
                        "content": """You are an advanced AI assistant with exceptional reasoning capabilities. You provide:

🎯 **Precise & Contextual Responses**: Always consider the full conversation context and provide relevant, detailed answers
🧠 **Deep Analysis**: Break down complex topics into understandable components with examples
💡 **Creative Problem-Solving**: Offer multiple approaches and innovative solutions
📚 **Educational Value**: Explain concepts clearly with step-by-step reasoning
🔍 **Critical Thinking**: Question assumptions, provide balanced perspectives, and acknowledge limitations
✨ **Engaging Communication**: Use emojis appropriately, maintain conversational flow, and adapt tone to context

When responding:
- Be thorough but concise
- Use markdown formatting for better readability
- Provide code examples when relevant
- Ask clarifying questions when needed
- Cite sources or acknowledge uncertainty when applicable
- Maintain context awareness throughout the conversation"""
                    }
                ]
                
                # Add recent conversation history (last 10 messages)
                recent_messages = conversation['messages'][-10:]
                for msg in recent_messages:
                    if msg['role'] in ['user', 'assistant']:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                # Generate response
                chat_completion = groq_client.chat.completions.create(
                    messages=messages,
                    model="llama-3.1-8b-instant",
                    temperature=0.8,
                    max_tokens=2000,
                    top_p=0.95,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
                
                ai_response = chat_completion.choices[0].message.content
                response_type = "standard"
                
        except Exception as ai_error:
            logger.error(f"AI generation error: {ai_error}")
            ai_response = "🔧 **System Notice**: I'm experiencing technical difficulties generating a response. Please try again!"
            response_type = "error"
        
        # Add AI response to conversation
        assistant_message = {
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().isoformat(),
            "sources": [source.dict() for source in sources],
            "context_used": context_used,
            "response_type": response_type
        }
        conversation['messages'].append(assistant_message)
        
        # Keep conversation manageable (last 30 messages)
        if len(conversation['messages']) > 30:
            conversation['messages'] = conversation['messages'][-30:]
        
        processing_time = time.time() - start_time
        
        return EnhancedChatResponse(
            response=ai_response,
            session_id=session_id,
            conversation_length=len(conversation['messages']),
            processing_time=processing_time,
            context_used=context_used,
            sources=sources,
            rag_enabled=request.use_rag,
            response_type=response_type
        )
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced chat failed: {str(e)}")

@router.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """
    Upload and process a document for the knowledge base
    """
    try:
        # Validate file type
        allowed_extensions = {'.txt', '.pdf', '.docx', '.md', '.html'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Process content based on file type
        rag_instance = get_rag_instance()
        
        if file_extension == '.txt' or file_extension == '.md':
            text_content = content.decode('utf-8')
        elif file_extension == '.pdf':
            # For now, return error for PDF processing
            # In production, you'd implement proper PDF processing
            raise HTTPException(status_code=400, detail="PDF processing not implemented in this demo")
        else:
            raise HTTPException(status_code=400, detail=f"Processing for {file_extension} not implemented")
        
        # Add document to knowledge base
        metadata = {
            'source_type': 'upload',
            'filename': file.filename,
            'title': title or file.filename,
            'description': description or '',
            'upload_timestamp': datetime.now().isoformat()
        }
        
        document_id = await rag_instance.add_document(text_content, metadata)
        
        # Calculate chunks created (estimate)
        chunks_created = len(text_content) // 1000 + 1
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            chunks_created=chunks_created,
            status="success",
            message=f"Document '{file.filename}' processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@router.post("/add-url")
async def add_url_to_knowledge_base(
    url: str = Form(...),
    title: Optional[str] = Form(None)
):
    """
    Add a web page to the knowledge base
    """
    try:
        rag_instance = get_rag_instance()
        document_id = await rag_instance.add_url(url, title)
        
        return {
            "document_id": document_id,
            "url": url,
            "title": title or url,
            "status": "success",
            "message": f"URL '{url}' added to knowledge base successfully"
        }
        
    except Exception as e:
        logger.error(f"Add URL error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add URL: {str(e)}")

@router.get("/knowledge-base/stats", response_model=KnowledgeBaseStats)
async def get_knowledge_base_stats():
    """
    Get statistics about the knowledge base
    """
    try:
        rag_instance = get_rag_instance()
        stats = rag_instance.get_knowledge_base_stats()
        
        return KnowledgeBaseStats(**stats)
        
    except Exception as e:
        logger.error(f"Knowledge base stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.get("/{session_id}/history")
async def get_enhanced_conversation_history(session_id: str):
    """
    Get enhanced conversation history with RAG context
    """
    if session_id not in enhanced_conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    conversation = enhanced_conversations[session_id]
    
    return {
        "session_id": session_id,
        "conversation": conversation['messages'],
        "message_count": len(conversation['messages']),
        "rag_enabled": conversation.get('rag_enabled', False),
        "created_at": conversation['created_at'],
        "last_activity": conversation['last_activity']
    }

@router.delete("/{session_id}")
async def clear_enhanced_conversation(session_id: str):
    """
    Clear enhanced conversation history
    """
    if session_id in enhanced_conversations:
        del enhanced_conversations[session_id]
        return {"message": f"Enhanced conversation {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.get("/sessions")
async def list_enhanced_sessions():
    """
    List all active enhanced conversation sessions
    """
    sessions = []
    for session_id, conversation in enhanced_conversations.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(conversation['messages']),
            "rag_enabled": conversation.get('rag_enabled', False),
            "created_at": conversation['created_at'],
            "last_activity": conversation['last_activity']
        })
    
    return {
        "sessions": sessions,
        "total_sessions": len(sessions)
    }

@router.get("/health")
async def enhanced_chat_health():
    """Health check for enhanced chat service"""
    try:
        rag_instance = get_rag_instance()
        stats = rag_instance.get_knowledge_base_stats()
        
        return {
            "status": "ok",
            "service": "enhanced-chat-with-rag",
            "timestamp": datetime.now().isoformat(),
            "knowledge_base": {
                "documents": stats['total_documents'],
                "status": "ready"
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "enhanced-chat-with-rag",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }