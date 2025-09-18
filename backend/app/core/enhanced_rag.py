"""
Enhanced RAG (Retrieval-Augmented Generation) System
Advanced document processing, vector search, and context-aware responses
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
import json
from pathlib import Path

# Vector and embedding libraries
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Document processing
import PyPDF2
import docx
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin

# LLM integration
import groq

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processing with multiple format support"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx', '.html', '.md', '.json']
    
    async def process_text_file(self, file_path: str) -> str:
        """Process plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return ""
    
    async def process_pdf_file(self, file_path: str) -> str:
        """Process PDF files"""
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            return ""
    
    async def process_docx_file(self, file_path: str) -> str:
        """Process Word documents"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing DOCX file {file_path}: {e}")
            return ""
    
    async def process_url(self, url: str) -> str:
        """Process web pages"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return ""

class VectorStore:
    """Advanced vector store with hybrid search capabilities"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.tfidf_matrix = None
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the vector store"""
        try:
            self.documents.extend(documents)
            
            # Generate embeddings
            texts = [doc['content'] for doc in documents]
            new_embeddings = self.embedding_model.encode(texts)
            
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            # Update FAISS index
            if self.index is None:
                dimension = new_embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(new_embeddings)
            self.index.add(new_embeddings)
            
            # Update TF-IDF matrix
            all_texts = [doc['content'] for doc in self.documents]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            logger.info(f"Added {len(documents)} documents to vector store. Total: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform semantic search using embeddings"""
        try:
            if self.index is None or len(self.documents) == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            
            return results
        
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform keyword search using TF-IDF"""
        try:
            if self.tfidf_matrix is None or len(self.documents) == 0:
                return []
            
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold for relevance
                    results.append((self.documents[idx], float(similarities[idx])))
            
            return results
        
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Tuple[Dict, float]]:
        """Perform hybrid search combining semantic and keyword search"""
        try:
            semantic_results = self.semantic_search(query, top_k * 2)
            keyword_results = self.keyword_search(query, top_k * 2)
            
            # Combine and rerank results
            combined_scores = {}
            
            # Add semantic scores
            for doc, score in semantic_results:
                doc_id = doc.get('id', hash(doc['content']))
                combined_scores[doc_id] = {
                    'doc': doc,
                    'semantic_score': score,
                    'keyword_score': 0.0
                }
            
            # Add keyword scores
            for doc, score in keyword_results:
                doc_id = doc.get('id', hash(doc['content']))
                if doc_id in combined_scores:
                    combined_scores[doc_id]['keyword_score'] = score
                else:
                    combined_scores[doc_id] = {
                        'doc': doc,
                        'semantic_score': 0.0,
                        'keyword_score': score
                    }
            
            # Calculate hybrid scores
            final_results = []
            for doc_id, scores in combined_scores.items():
                hybrid_score = (
                    semantic_weight * scores['semantic_score'] + 
                    (1 - semantic_weight) * scores['keyword_score']
                )
                final_results.append((scores['doc'], hybrid_score))
            
            # Sort by hybrid score
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            return final_results[:top_k]
        
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

class EnhancedRAG:
    """Enhanced RAG system with intelligent document processing and context generation"""
    
    def __init__(self, groq_api_key: str):
        self.groq_client = groq.Groq(api_key=groq_api_key)
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.knowledge_base_path = Path("data/knowledge_base")
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
    
    async def add_document(self, content: str, metadata: Dict[str, str]) -> str:
        """Add a document to the knowledge base"""
        try:
            # Generate document ID
            doc_id = hashlib.md5(content.encode()).hexdigest()
            
            # Split content into chunks
            chunks = self._split_into_chunks(content, chunk_size=1000, overlap=200)
            
            # Create document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    'id': f"{doc_id}_chunk_{i}",
                    'content': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'timestamp': datetime.now().isoformat()
                    }
                }
                documents.append(doc)
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            return doc_id
        
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def add_url(self, url: str, title: str = None) -> str:
        """Add a web page to the knowledge base"""
        try:
            content = await self.document_processor.process_url(url)
            if not content:
                raise ValueError(f"Could not extract content from URL: {url}")
            
            metadata = {
                'source_type': 'url',
                'source': url,
                'title': title or url,
                'domain': urlparse(url).netloc
            }
            
            return await self.add_document(content, metadata)
        
        except Exception as e:
            logger.error(f"Error adding URL {url}: {e}")
            raise
    
    async def search_and_generate(self, query: str, max_context_length: int = 4000) -> Dict[str, str]:
        """Search knowledge base and generate context-aware response"""
        try:
            # Perform hybrid search
            search_results = self.vector_store.hybrid_search(query, top_k=5)
            
            if not search_results:
                return {
                    'response': "I don't have specific information about that topic in my knowledge base. Let me provide a general response based on my training.",
                    'sources': [],
                    'context_used': False
                }
            
            # Prepare context from search results
            context_parts = []
            sources = []
            
            for doc, score in search_results:
                if len('\n'.join(context_parts)) < max_context_length:
                    context_parts.append(f"**Source**: {doc['metadata'].get('title', 'Unknown')}\n{doc['content']}")
                    sources.append({
                        'title': doc['metadata'].get('title', 'Unknown'),
                        'source': doc['metadata'].get('source', 'Unknown'),
                        'relevance_score': round(score, 3)
                    })
            
            context = '\n\n---\n\n'.join(context_parts)
            
            # Generate response with context
            response = await self._generate_contextual_response(query, context)
            
            return {
                'response': response,
                'sources': sources,
                'context_used': True
            }
        
        except Exception as e:
            logger.error(f"Error in search and generate: {e}")
            return {
                'response': "I encountered an error while searching my knowledge base. Please try again.",
                'sources': [],
                'context_used': False
            }
    
    async def _generate_contextual_response(self, query: str, context: str) -> str:
        """Generate response using retrieved context"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """🚀 You are BlueMech AI, a sophisticated professional assistant with advanced knowledge retrieval capabilities. Your responses should be engaging, insightful, and exceptionally helpful.

**Professional Guidelines**:
✨ **Tone**: Confident, friendly, and authoritative - like a top consultant
🎯 **Accuracy**: Base responses primarily on provided context with precision
📚 **Citations**: Reference sources naturally within your explanations
💡 **Value-Add**: Supplement with strategic insights when context allows
🎨 **Formatting**: Use rich markdown with emojis, headers, and bullet points
⚡ **Engagement**: Make complex topics accessible and interesting
🔍 **Transparency**: Clearly indicate when information comes from context vs. general knowledge

**Response Structure**:
- Start with a confident, engaging hook
- Provide comprehensive analysis based on context
- Add strategic insights or implications when relevant
- End with actionable takeaways or next steps when appropriate
- Use professional emojis sparingly for visual appeal

Remember: You're not just answering questions - you're providing intelligent business consultation."""
                },
                {
                    "role": "user",
                    "content": f"""📊 **Knowledge Base Context**:
{context}

🎯 **Client Question**: {query}

Please provide your expert analysis and recommendations based on the available context."""
                }
            ]
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
                temperature=0.8,  # Higher temperature for more engaging responses
                max_tokens=2000,  # Allow longer, more comprehensive responses
                top_p=0.9
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            return "I encountered an error while generating a response. Please try again."
    
    def _split_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(min(100, chunk_size // 4)):
                    if end - i > start and text[end - i] in '.!?':
                        end = end - i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def get_knowledge_base_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        return {
            'total_documents': len(self.vector_store.documents),
            'total_chunks': len(self.vector_store.documents),
            'embedding_dimension': self.vector_store.embeddings.shape[1] if self.vector_store.embeddings is not None else 0,
            'last_updated': datetime.now().isoformat()
        }

# Global RAG instance
enhanced_rag = None

def get_rag_instance() -> EnhancedRAG:
    """Get or create the global RAG instance"""
    global enhanced_rag
    if enhanced_rag is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        enhanced_rag = EnhancedRAG(groq_api_key)
    return enhanced_rag