from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routers
from app.api.chat_session import router as chat_session_router
# from app.api.enhanced_chat import router as enhanced_chat_router  # Temporarily disabled
from app.api.health import router as health_router
from app.api.advanced_ai import router as advanced_ai_router

# Create FastAPI app
app = FastAPI(
    title="BlueMech AI",
    description="Professional AI Assistant with RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(chat_session_router, prefix="/api/v1")  # Standard ChatGPT-style conversations
# app.include_router(enhanced_chat_router, prefix="/api/v1")  # Enhanced RAG-powered conversations - Temporarily disabled
app.include_router(advanced_ai_router, prefix="/api/v1")  # Advanced AI with reasoning, truthfulness, and personalization

@app.get("/")
async def root():
    return {
        "message": "Enhanced ChatGPT Clone with RAG",
        "version": "2.0.0",
        "docs": "/docs",
        "description": "Advanced conversational AI with document retrieval and context awareness",
        "features": [
            "Standard ChatGPT-style conversations",
            "RAG-enhanced responses with document retrieval",
            "Document upload and processing",
            "Web page integration",
            "Context-aware responses"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )