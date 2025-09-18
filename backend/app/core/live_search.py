"""
Simplified Live Search System - ChatGPT/Perplexity Style
Direct implementation that actually retrieves and uses live web data
"""

import asyncio
import aiohttp
import time
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote
from bs4 import BeautifulSoup
from datetime import datetime

from app.utils.config import get_settings
from app.models.search import Source

logger = logging.getLogger(__name__)

class LiveSearchSystem:
    """
    Simple but effective live search system like ChatGPT/Perplexity
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.session = None
        
        # Initialize Groq for AI generation
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.settings.groq_api_key)
            self.ai_available = True
            logger.info("✅ Live Search System initialized with Groq AI")
        except Exception as e:
            self.ai_available = False
            logger.error(f"AI initialization failed: {e}")
    
    async def search_and_generate(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Main search function - like ChatGPT/Perplexity
        1. Search Wikipedia for reliable content
        2. Generate comprehensive AI response
        3. Return with sources and citations
        """
        start_time = time.time()
        
        try:
            # Step 1: Get live content from Wikipedia
            sources = await self._search_wikipedia_content(query, max_results)
            
            # Step 2: Extract content for AI context
            context = self._extract_context_from_sources(sources)
            
            # Step 3: Generate AI response
            if self.ai_available and context:
                answer = await self._generate_ai_response(query, context)
            else:
                answer = self._generate_fallback_response(query, sources)
            
            # Step 4: Create citations
            citations = self._create_citations(sources)
            
            search_time = time.time() - start_time
            
            return {
                "query": query,
                "answer": answer,
                "sources": [self._source_to_dict(s) for s in sources],
                "citations": citations,
                "search_time": search_time,
                "model_used": "groq" if self.ai_available else "fallback",
                "search_type": "live_web",
                "total_sources_found": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Live search failed: {e}")
            return {
                "query": query,
                "answer": f"I apologize, but I encountered an error while searching for information about '{query}'. Please try again.",
                "sources": [],
                "citations": [],
                "search_time": time.time() - start_time,
                "model_used": "error",
                "search_type": "live_web",
                "total_sources_found": 0
            }
    
    async def _search_wikipedia_content(self, query: str, max_results: int) -> List[Source]:
        """
        Search Wikipedia and extract actual content
        """
        sources = []
        
        if not self.session:
            headers = {
                'User-Agent': 'PerplexityAIClone/1.0 (https://example.com/contact; educational-purpose) Python/3.13'
            }
            self.session = aiohttp.ClientSession(headers=headers)
        
        try:
            logger.info(f"🔍 Searching Wikipedia for: {query}")
            
            # Search Wikipedia API
            search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote(query)}&limit={max_results}&format=json"
            logger.info(f"📡 Wikipedia API URL: {search_url}")
            
            async with self.session.get(search_url) as response:
                logger.info(f"📊 Wikipedia API response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"📋 Wikipedia API data length: {len(data) if data else 0}")
                    
                    if len(data) >= 4:
                        titles = data[1]
                        descriptions = data[2] 
                        urls = data[3]
                        
                        logger.info(f"📖 Found {len(titles)} Wikipedia articles")
                        
                        # Get detailed content for each article
                        for title, desc, url in zip(titles[:max_results], descriptions[:max_results], urls[:max_results]):
                            if title and url:
                                logger.info(f"📄 Fetching content for: {title}")
                                content = await self._fetch_wikipedia_content(title)
                                if content:
                                    source = Source(
                                        url=url,
                                        title=title,
                                        content=content[:1000] + "..." if len(content) > 1000 else content,
                                        relevance_score=0.9,
                                        domain="en.wikipedia.org",
                                        publish_date=datetime.now()
                                    )
                                    sources.append(source)
                                    logger.info(f"✅ Successfully fetched: {title}")
                                else:
                                    logger.warning(f"⚠️ No content for: {title}")
                    else:
                        logger.warning(f"⚠️ Unexpected Wikipedia API response format: {data}")
                else:
                    logger.error(f"❌ Wikipedia API error: {response.status}")
            
            logger.info(f"🎯 Total sources found: {len(sources)}")
            return sources
            
        except Exception as e:
            logger.error(f"❌ Wikipedia search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def _fetch_wikipedia_content(self, title: str) -> str:
        """
        Fetch the actual content of a Wikipedia article
        """
        try:
            # Get page extract from Wikipedia API
            api_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={quote(title)}&prop=extracts&exintro=true&explaintext=true&exsectionformat=plain"
            
            async with self.session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data.get('query', {}).get('pages', {})
                    
                    for page_id, page_data in pages.items():
                        extract = page_data.get('extract', '')
                        if extract:
                            return extract
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to fetch Wikipedia content for {title}: {e}")
            return ""
    
    def _extract_context_from_sources(self, sources: List[Source]) -> str:
        """
        Create context string from sources for AI
        """
        if not sources:
            return ""
        
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"Source {i} ({source.title}):\\n{source.content}\\n")
        
        return "\\n".join(context_parts)
    
    async def _generate_ai_response(self, query: str, context: str) -> str:
        """
        Generate AI response using Groq
        """
        try:
            prompt = f"""Based on the following reliable sources, provide a comprehensive and accurate answer to the question.

Question: {query}

Sources:
{context}

Instructions:
1. Provide a detailed, informative answer based on the sources
2. Reference specific information from the sources
3. Be comprehensive but concise
4. Maintain objectivity and accuracy
5. If the sources don't contain enough information, acknowledge it

Answer:"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return f"Based on the available sources, here's what I found about {query}: " + (context[:500] + "..." if len(context) > 500 else context)
    
    def _generate_fallback_response(self, query: str, sources: List[Source]) -> str:
        """
        Generate a basic response when AI is not available
        """
        if not sources:
            return f"I couldn't find reliable information about '{query}' at the moment. Please try again or rephrase your question."
        
        response = f"Based on my search, here's what I found about {query}:\\n\\n"
        
        for i, source in enumerate(sources[:3], 1):
            response += f"{i}. {source.title}: {source.content[:200]}...\\n\\n"
        
        response += f"\\nFound {len(sources)} relevant sources from Wikipedia."
        
        return response
    
    def _create_citations(self, sources: List[Source]) -> List[Dict[str, Any]]:
        """
        Create citation objects from sources
        """
        citations = []
        
        for i, source in enumerate(sources, 1):
            citation = {
                "text": source.content[:100] + "..." if len(source.content) > 100 else source.content,
                "source_url": source.url,
                "source_title": source.title,
                "relevance_score": source.relevance_score,
                "position": i
            }
            citations.append(citation)
        
        return citations
    
    def _source_to_dict(self, source: Source) -> Dict[str, Any]:
        """
        Convert Source object to dictionary
        """
        return {
            "url": source.url,
            "title": source.title,
            "content": source.content,
            "relevance_score": source.relevance_score,
            "domain": source.domain,
            "publish_date": source.publish_date.isoformat() if source.publish_date else None,
            "author": getattr(source, 'author', None)
        }
    
    async def close(self):
        """
        Clean up resources
        """
        if self.session:
            await self.session.close()