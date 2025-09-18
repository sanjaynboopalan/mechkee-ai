"""
RAG (Retrieval Augmented Generation) Pipeline
Core implementation of the Perplexity-like search and generation system
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

# Use multiple AI providers with fallback
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from transformers import pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from app.core.search_engine import SearchEngine
from app.core.vector_store import VectorStore
from app.core.citation_extractor import CitationExtractor
from app.core.web_scraper import WebScraper
from app.models.search import Source, Citation, SearchResponse
from app.models.chat import ChatMessage
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Main RAG pipeline that orchestrates the entire search-to-answer process
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.search_engine = SearchEngine()
        self.vector_store = VectorStore()
        self.citation_extractor = CitationExtractor()
        self.web_scraper = WebScraper()  # Add web scraper for real-time content
        
        # Initialize AI clients with priority order and fallback
        self.ai_clients = {}
        self.ai_provider = "mock"
        
        # Priority: Groq (free) -> OpenAI -> Anthropic -> HuggingFace -> Mock
        if GROQ_AVAILABLE and hasattr(self.settings, 'groq_api_key') and self.settings.groq_api_key:
            try:
                self.ai_clients['groq'] = Groq(api_key=self.settings.groq_api_key)
                self.ai_provider = "groq"
                logger.info("✅ Groq AI initialized (Primary)")
            except Exception as e:
                logger.error(f"Groq initialization failed: {e}")
        
        if OPENAI_AVAILABLE and self.settings.openai_api_key:
            try:
                self.ai_clients['openai'] = openai.OpenAI(api_key=self.settings.openai_api_key)
                if self.ai_provider == "mock":
                    self.ai_provider = "openai"
                logger.info("✅ OpenAI initialized (Fallback)")
            except Exception as e:
                logger.error(f"OpenAI initialization failed: {e}")
        
        if ANTHROPIC_AVAILABLE and hasattr(self.settings, 'anthropic_api_key') and self.settings.anthropic_api_key:
            try:
                self.ai_clients['anthropic'] = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
                if self.ai_provider == "mock":
                    self.ai_provider = "anthropic"
                logger.info("✅ Anthropic initialized (Fallback)")
            except Exception as e:
                logger.error(f"Anthropic initialization failed: {e}")
        
        if HUGGINGFACE_AVAILABLE:
            try:
                self.ai_clients['huggingface'] = pipeline("text-generation", model="microsoft/DialoGPT-medium")
                if self.ai_provider == "mock":
                    self.ai_provider = "huggingface"
                logger.info("✅ HuggingFace initialized (Local fallback)")
            except Exception as e:
                logger.error(f"HuggingFace initialization failed: {e}")
        
        if self.ai_provider == "mock":
            logger.warning("⚠️ No AI providers available - using mock responses")
        else:
            logger.info(f"🚀 Active AI provider: {self.ai_provider.upper()}")
        
        # RAG prompt template
        self.rag_prompt_template = """
You are an AI assistant that provides accurate, comprehensive answers based on the given context.

Context information:
{context}

Question: {question}

Instructions:
1. Provide a comprehensive answer based on the context
2. If the context doesn't contain enough information, say so
3. Use specific quotes from the context when relevant
4. Be concise but thorough
5. Maintain objectivity and cite sources when making claims

Answer:
"""
    
    async def search_and_generate(
        self,
        query: str,
        max_results: int = 10,
        include_citations: bool = True,
        search_type: str = "hybrid",
        use_live_web: bool = True
    ) -> Dict[str, Any]:
        """
        Main RAG pipeline: search -> retrieve -> generate answer (like Perplexity AI)
        """
        start_time = time.time()
        
        try:
            # Step 1: Query processing and expansion
            processed_query = await self._process_query(query)
            logger.info(f"Processed query: {processed_query}")
            
            # Step 2: Retrieve from multiple sources
            all_results = []
            
            # Get local/cached results
            local_results = await self.search_engine.search(
                query=processed_query,
                max_results=max_results // 2,
                search_type=search_type
            )
            all_results.extend(local_results)
            
            # Get real-time web content (like Perplexity)
            if use_live_web:
                try:
                    logger.info(f"🌐 Starting web scraping for query: {processed_query}")
                    web_results = await self.web_scraper.search_and_scrape(
                        query=processed_query,
                        max_sources=max_results // 2
                    )
                    all_results.extend(web_results)
                    logger.info(f"✅ Fetched {len(web_results)} live web sources")
                    
                    # Debug: log web results
                    for i, result in enumerate(web_results[:2]):
                        logger.info(f"Web result {i+1}: {result.title} from {result.domain}")
                        
                except Exception as e:
                    logger.error(f"❌ Web scraping failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.info("🚫 Live web scraping disabled")
            
            # Step 3: Rank and filter results
            ranked_results = await self._rank_results(all_results, query)
            
            # Step 4: Extract relevant context
            context = await self._extract_context(ranked_results)
            
            # Step 5: Generate answer using LLM
            answer = await self._generate_answer(context, query)
            
            # Step 6: Extract citations if requested
            citations = []
            if include_citations:
                citations = await self.citation_extractor.extract_citations(
                    answer, ranked_results
                )
            
            search_time = time.time() - start_time
            
            return {
                "answer": answer,
                "sources": ranked_results,
                "citations": citations,
                "search_time": search_time,
                "model_used": self.ai_provider,
                "search_type": search_type,
                "total_sources": len(all_results),
                "live_sources": len([r for r in ranked_results if 'wikipedia' in r.domain or 'bbc' in r.domain])
            }
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {str(e)}")
            raise
    
    async def chat_with_context(
        self,
        message: str,
        context_messages: List[ChatMessage]
    ) -> Dict[str, Any]:
        """
        Generate response with conversation context
        """
        try:
            # Build conversation context
            conversation_context = self._build_conversation_context(context_messages)
            
            # Enhance query with conversation context
            enhanced_query = f"Context: {conversation_context}\n\nCurrent question: {message}"
            
            # Use RAG pipeline with enhanced query
            result = await self.search_and_generate(
                query=enhanced_query,
                max_results=8,
                include_citations=True
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Chat with context failed: {str(e)}")
            raise
    
    async def _process_query(self, query: str) -> str:
        """
        Process and expand the user query for better search results
        """
        try:
            if self.ai_clients:
                # Query expansion using AI
                expansion_prompt = f"""
                Expand this search query to include related terms and concepts that would help find relevant information:
                
                Original query: {query}
                
                Expanded query (keep it concise):
                """
                
                messages = [{"role": "user", "content": expansion_prompt}]
                expanded_query = await self._generate_with_ai(messages, max_tokens=100, temperature=0.3)
                
                return expanded_query or query
            else:
                # Simple fallback - just return original query
                return query
        except:
            # Fallback to original query
            return query
    
    async def _rank_results(self, results: List[Source], query: str) -> List[Source]:
        """
        Re-rank search results based on relevance and quality
        """
        # For now, return results as-is (sorted by initial relevance)
        # TODO: Implement advanced re-ranking algorithms
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    async def _extract_context(self, sources: List[Source]) -> str:
        """
        Extract and concatenate relevant context from sources
        """
        context_parts = []
        
        for i, source in enumerate(sources[:5]):  # Use top 5 sources
            context_part = f"Source {i+1} ({source.domain}):\n{source.content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def _generate_with_ai(self, messages: list, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """
        Generate response using any available AI provider with automatic fallback
        """
        # Try providers in order of preference
        providers_to_try = ['groq', 'openai', 'anthropic', 'huggingface']
        
        for provider in providers_to_try:
            if provider in self.ai_clients:
                try:
                    if provider == "groq":
                        response = self.ai_clients[provider].chat.completions.create(
                            model="llama-3.1-8b-instant",  # Updated to working model
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        return response.choices[0].message.content.strip()
                    
                    elif provider == "openai":
                        response = self.ai_clients[provider].chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        return response.choices[0].message.content.strip()
                    
                    elif provider == "anthropic":
                        # Convert messages format for Anthropic
                        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
                        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
                        
                        response = self.ai_clients[provider].messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system=system_msg,
                            messages=[{"role": "user", "content": user_msg}]
                        )
                        return response.content[0].text.strip()
                    
                    elif provider == "huggingface":
                        # Use local HuggingFace model
                        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
                        response = self.ai_clients[provider](user_msg, max_length=max_tokens, temperature=temperature)
                        return response[0]['generated_text'].strip()
                
                except Exception as e:
                    logger.error(f"{provider.upper()} API failed: {str(e)}")
                    continue  # Try next provider
        
        # All providers failed
        return "I apologize, but I'm unable to generate an answer at the moment. All AI providers are currently unavailable."
    
    async def _generate_answer(self, context: str, question: str) -> str:
        """
        Generate answer using LLM with retrieved context
        """
        try:
            prompt = self.rag_prompt_template.format(context=context, question=question)
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that provides accurate information based on given context. Be comprehensive and cite your sources."},
                {"role": "user", "content": prompt}
            ]
            
            return await self._generate_with_ai(messages, max_tokens=500, temperature=0.1)
                
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "I apologize, but I'm unable to generate an answer at the moment. Please try again."
    
    async def close(self):
        """
        Close resources (web scraper session, etc.)
        """
        try:
            if hasattr(self.web_scraper, 'close'):
                await self.web_scraper.close()
        except Exception as e:
            logger.error(f"Error closing resources: {e}")
    
    def _build_conversation_context(self, messages: List[ChatMessage]) -> str:
        """
        Build conversation context from recent messages
        """
        context_parts = []
        
        for msg in messages[-5:]:  # Last 5 messages
            role = "Human" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)
    
    async def record_feedback(self, query: str, helpful: bool, session_id: Optional[str] = None):
        """
        Record user feedback for improving search quality
        """
        # TODO: Implement feedback storage and learning
        logger.info(f"Feedback recorded - Query: {query}, Helpful: {helpful}, Session: {session_id}")
        pass