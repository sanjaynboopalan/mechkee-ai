from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    include_citations: Optional[bool] = True
    search_type: Optional[str] = "hybrid"
    filters: Optional[Dict[str, Any]] = None

class Source(BaseModel):
    url: str
    title: str
    content: str
    relevance_score: float
    domain: Optional[str] = None
    publish_date: Optional[datetime] = None
    author: Optional[str] = None

class Citation(BaseModel):
    text: str
    source_url: str
    source_title: str
    relevance_score: float
    position: int

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[Source]
    citations: List[Citation]
    search_time: float
    model_used: str
    search_type: str
    total_sources_found: int
    
class SearchMetrics(BaseModel):
    query: str
    response_time: float
    sources_retrieved: int
    model_used: str
    user_feedback: Optional[bool] = None
    timestamp: datetime = datetime.utcnow()