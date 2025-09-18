from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "Perplexity AI Clone",
        "version": "1.0.0"
    }

@router.get("/status")
async def system_status():
    return {
        "api": "operational",
        "database": "connected",
        "vector_db": "connected",
        "search_engine": "operational",
        "llm_service": "available"
    }