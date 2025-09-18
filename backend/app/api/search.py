from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
import asyncio

from app.core.rag_pipeline import RAGPipeline
from app.core.search_engine import SearchEngine
from app.models.search import SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])

# Initialize core components
search_engine = SearchEngine()
rag_pipeline = RAGPipeline()

class SearchQuery(BaseModel):
    query: str
    max_results: Optional[int] = 10
    include_citations: Optional[bool] = True
    search_type: Optional[str] = "hybrid"  # "hybrid", "vector", "keyword", "web"
    use_live_web: Optional[bool] = True  # Enable real-time web scraping

@router.post("/", response_model=SearchResponse)
async def search(search_request: SearchQuery):
    """
    Perform AI-powered search with RAG capabilities
    """
    try:
        # Execute search through RAG pipeline
        result = await rag_pipeline.search_and_generate(
            query=search_request.query,
            max_results=search_request.max_results,
            include_citations=search_request.include_citations,
            search_type=search_request.search_type,
            use_live_web=search_request.use_live_web
        )
        
        return SearchResponse(
            query=search_request.query,
            answer=result["answer"],
            sources=result["sources"],
            citations=result["citations"] if search_request.include_citations else [],
            search_time=result["search_time"],
            model_used=result["model_used"],
            search_type=search_request.search_type,
            total_sources_found=len(result["sources"])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/suggestions")
async def get_search_suggestions(q: str = Query(..., min_length=2)):
    """
    Get search suggestions based on query
    """
    try:
        suggestions = await search_engine.get_suggestions(q)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

@router.post("/feedback")
async def search_feedback(query: str, helpful: bool, session_id: Optional[str] = None):
    """
    Record user feedback for search results
    """
    try:
        # Store feedback for improving search quality
        await rag_pipeline.record_feedback(query, helpful, session_id)
        return {"message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")