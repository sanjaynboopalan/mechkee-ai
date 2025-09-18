"""
Live Search API Endpoint - ChatGPT/Perplexity Style
Simple but effective endpoint that actually retrieves live data
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel
import asyncio

from app.core.live_search import LiveSearchSystem

router = APIRouter(prefix="/live-search", tags=["live-search"])

# Initialize the live search system
live_search = LiveSearchSystem()

class LiveSearchQuery(BaseModel):
    query: str
    max_results: Optional[int] = 5

@router.post("/")
async def live_search_endpoint(search_request: LiveSearchQuery):
    """
    Live search endpoint that actually retrieves and uses web data like ChatGPT/Perplexity
    """
    try:
        result = await live_search.search_and_generate(
            query=search_request.query,
            max_results=search_request.max_results
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live search failed: {str(e)}")

@router.get("/health")
async def live_search_health():
    """
    Check if live search system is working
    """
    return {
        "status": "healthy",
        "service": "Live Search System", 
        "ai_available": live_search.ai_available
    }