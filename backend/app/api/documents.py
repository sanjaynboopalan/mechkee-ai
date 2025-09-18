from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
from pydantic import BaseModel
import aiofiles
import os
from datetime import datetime

from app.core.document_processor import DocumentProcessor
from app.models.documents import DocumentMetadata, ProcessingResult

router = APIRouter(prefix="/documents", tags=["documents"])

# Initialize document processor
doc_processor = DocumentProcessor()

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process documents for search indexing
    """
    try:
        # Validate file type
        allowed_types = ['.pdf', '.txt', '.docx', '.md', '.html']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed types: {allowed_types}"
            )
        
        # Save uploaded file
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Process document
        result = await doc_processor.process_document(file_path)
        
        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=file.filename,
            status="processed",
            message="Document successfully uploaded and indexed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@router.get("/", response_model=List[DocumentMetadata])
async def list_documents():
    """
    List all indexed documents
    """
    try:
        documents = await doc_processor.list_documents()
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/{document_id}")
async def get_document(document_id: str):
    """
    Get document metadata and content
    """
    try:
        document = await doc_processor.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the index
    """
    try:
        success = await doc_processor.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.post("/reindex")
async def reindex_documents():
    """
    Reindex all documents
    """
    try:
        result = await doc_processor.reindex_all_documents()
        return {
            "message": "Reindexing started",
            "documents_processed": result.get("count", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")