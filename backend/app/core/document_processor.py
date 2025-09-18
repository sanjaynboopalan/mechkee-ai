"""
Document Processor
Handles document upload, processing, and indexing
"""

import asyncio
import os
import uuid
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import aiofiles

# Document processing libraries
import PyPDF2
from docx import Document as DocxDocument
from bs4 import BeautifulSoup

from app.models.documents import DocumentMetadata, DocumentChunk, ProcessingResult
from app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Processes various document types and prepares them for search indexing
    """
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.documents_db = {}  # Simple in-memory storage
        
        # Supported file types
        self.supported_types = {
            '.pdf': self._process_pdf,
            '.txt': self._process_text,
            '.md': self._process_text,
            '.docx': self._process_docx,
            '.html': self._process_html
        }
        
        # Chunking parameters
        self.chunk_size = 1000  # characters per chunk
        self.chunk_overlap = 200  # overlap between chunks
    
    async def process_document(self, file_path: str) -> ProcessingResult:
        """
        Process uploaded document and add to search index
        """
        start_time = datetime.utcnow()
        document_id = str(uuid.uuid4())
        
        try:
            # Extract file information
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            file_type = os.path.splitext(filename)[1].lower()
            
            if file_type not in self.supported_types:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Create document metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                file_type=file_type,
                file_size=file_size,
                upload_date=start_time,
                status="processing"
            )
            
            # Extract text content
            content = await self.supported_types[file_type](file_path)
            
            # Create chunks
            chunks = await self._create_chunks(content, document_id)
            
            # Update metadata
            metadata.chunk_count = len(chunks)
            metadata.processed_date = datetime.utcnow()
            metadata.status = "indexed"
            
            # Store in database
            self.documents_db[document_id] = {
                "metadata": metadata,
                "content": content,
                "chunks": chunks
            }
            
            # Add chunks to vector store
            for chunk in chunks:
                await self.vector_store.add_document(
                    document_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_index": chunk.chunk_index,
                        "file_type": file_type
                    }
                )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ProcessingResult(
                document_id=document_id,
                status="success",
                message="Document processed successfully",
                chunk_count=len(chunks),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return ProcessingResult(
                document_id=document_id,
                status="failed",
                message=f"Processing failed: {str(e)}",
                chunk_count=0,
                processing_time=0,
                errors=[str(e)]
            )
    
    async def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise
        return text.strip()
    
    async def _process_text(self, file_path: str) -> str:
        """Process plain text or markdown files"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()
            return content
        except Exception as e:
            logger.error(f"Text file processing failed: {str(e)}")
            raise
    
    async def _process_docx(self, file_path: str) -> str:
        """Extract text from Word document"""
        try:
            doc = DocxDocument(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return "\n".join(text)
        except Exception as e:
            logger.error(f"DOCX processing failed: {str(e)}")
            raise
    
    async def _process_html(self, file_path: str) -> str:
        """Extract text from HTML file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                html_content = await file.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"HTML processing failed: {str(e)}")
            raise
    
    async def _create_chunks(self, content: str, document_id: str) -> List[DocumentChunk]:
        """Split document content into chunks for indexing"""
        chunks = []
        words = content.split()
        
        chunk_index = 0
        start_pos = 0
        
        while start_pos < len(words):
            # Calculate chunk end position
            end_pos = min(start_pos + self.chunk_size, len(words))
            
            # Create chunk content
            chunk_words = words[start_pos:end_pos]
            chunk_content = " ".join(chunk_words)
            
            # Calculate character positions
            start_char = len(" ".join(words[:start_pos]))
            end_char = start_char + len(chunk_content)
            
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_{chunk_index}",
                document_id=document_id,
                content=chunk_content,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                token_count=len(chunk_words)
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_pos = end_pos - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    async def list_documents(self) -> List[DocumentMetadata]:
        """List all processed documents"""
        documents = []
        for doc_data in self.documents_db.values():
            documents.append(doc_data["metadata"])
        return documents
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        return self.documents_db.get(document_id)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from index"""
        try:
            if document_id in self.documents_db:
                # Remove chunks from vector store
                doc_data = self.documents_db[document_id]
                for chunk in doc_data["chunks"]:
                    await self.vector_store.delete_document(chunk.chunk_id)
                
                # Remove from documents database
                del self.documents_db[document_id]
                
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False
    
    async def reindex_all_documents(self) -> Dict[str, Any]:
        """Reindex all documents"""
        try:
            count = 0
            for document_id, doc_data in self.documents_db.items():
                # Re-add chunks to vector store
                for chunk in doc_data["chunks"]:
                    await self.vector_store.add_document(
                        document_id=chunk.chunk_id,
                        content=chunk.content,
                        metadata={
                            "document_id": document_id,
                            "filename": doc_data["metadata"].filename,
                            "chunk_index": chunk.chunk_index,
                            "file_type": doc_data["metadata"].file_type
                        }
                    )
                count += 1
            
            return {"count": count, "status": "completed"}
        except Exception as e:
            logger.error(f"Reindexing failed: {str(e)}")
            raise