from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_path: str
    file_type: str
    file_size: int
    upload_date: datetime
    processed_date: Optional[datetime] = None
    status: str  # "uploaded", "processing", "indexed", "failed"
    chunk_count: Optional[int] = None
    embedding_model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = {}

class ProcessingResult(BaseModel):
    document_id: str
    status: str
    message: str
    chunk_count: int
    processing_time: float
    errors: Optional[List[str]] = []

class DocumentQuery(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    max_results: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.7

class DocumentSearchResult(BaseModel):
    document_id: str
    filename: str
    relevant_chunks: List[DocumentChunk]
    relevance_score: float
    metadata: Dict[str, Any]