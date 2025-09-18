"""
Vector Store Implementation
Handles embeddings and semantic search
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import os
from datetime import datetime

# For production, use Chroma, Pinecone, or Weaviate
# For now, simple in-memory implementation

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector database for storing and searching embeddings
    """
    
    def __init__(self):
        self.embeddings_store = {}  # document_id -> embedding
        self.documents_store = {}   # document_id -> document metadata
        self.index_path = "data/embeddings/vector_index.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Load existing index
        self._load_index()
    
    def _load_index(self):
        """Load existing vector index from disk"""
        try:
            if os.path.exists(self.index_path):
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                    self.documents_store = data.get("documents", {})
                    # Note: embeddings are not saved in this simple implementation
                    logger.info(f"Loaded {len(self.documents_store)} documents from index")
        except Exception as e:
            logger.error(f"Failed to load vector index: {str(e)}")
    
    def _save_index(self):
        """Save vector index to disk"""
        try:
            data = {
                "documents": self.documents_store,
                "last_updated": datetime.utcnow().isoformat()
            }
            with open(self.index_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save vector index: {str(e)}")
    
    async def add_document(
        self,
        document_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Add document with embedding to vector store
        """
        try:
            # Generate embedding if not provided
            if embedding is None:
                embedding = await self._generate_embedding(content)
            
            # Store embedding and document
            self.embeddings_store[document_id] = embedding
            self.documents_store[document_id] = {
                "content": content,
                "metadata": metadata,
                "added_at": datetime.utcnow().isoformat(),
                "embedding_dimension": len(embedding)
            }
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Added document {document_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document to vector store: {str(e)}")
            return False
    
    async def search_similar(
        self,
        query: str,
        max_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Search for similar documents using vector similarity
        """
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.embeddings_store.items():
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                if similarity >= similarity_threshold:
                    similarities.append((doc_id, similarity))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:max_results]
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document by ID
        """
        return self.documents_store.get(document_id)
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document from vector store
        """
        try:
            if document_id in self.embeddings_store:
                del self.embeddings_store[document_id]
            
            if document_id in self.documents_store:
                del self.documents_store[document_id]
            
            self._save_index()
            logger.info(f"Deleted document {document_id} from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text
        In production, use OpenAI embeddings or sentence transformers
        """
        # Mock embedding - in production, use actual embedding model
        # This creates a simple hash-based "embedding"
        import hashlib
        
        # Create a deterministic but varied embedding based on text
        hash_value = hashlib.md5(text.encode()).hexdigest()
        
        # Convert to vector of floats (384 dimensions)
        embedding = []
        for i in range(0, min(len(hash_value), 32), 2):
            val = int(hash_value[i:i+2], 16) / 255.0
            embedding.extend([val] * 12)  # Repeat to get 384 dimensions
        
        # Ensure exactly 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)
        
        return embedding[:384]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {str(e)}")
            return 0.0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        """
        return {
            "total_documents": len(self.documents_store),
            "total_embeddings": len(self.embeddings_store),
            "index_path": self.index_path,
            "last_updated": datetime.utcnow().isoformat()
        }