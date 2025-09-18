"""
Citation Extractor
Automatically extracts and formats citations from retrieved sources
"""

import re
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.models.search import Source, Citation

logger = logging.getLogger(__name__)

class CitationExtractor:
    """
    Extracts citations from generated answers and matches them to sources
    """
    
    def __init__(self):
        # Pattern to match potential citation text
        self.citation_patterns = [
            r'"([^"]{10,100})"',  # Quoted text
            r'according to ([^,\.]{10,50})',  # According to statements
            r'research shows ([^,\.]{10,50})',  # Research statements
            r'studies indicate ([^,\.]{10,50})',  # Study statements
        ]
    
    async def extract_citations(
        self,
        answer: str,
        sources: List[Source]
    ) -> List[Citation]:
        """
        Extract citations from the generated answer
        """
        citations = []
        
        try:
            # Find quoted text and key statements in the answer
            potential_citations = self._find_potential_citations(answer)
            
            # Match each potential citation to sources
            for i, citation_text in enumerate(potential_citations):
                best_match = await self._find_best_source_match(citation_text, sources)
                
                if best_match:
                    source, relevance_score = best_match
                    
                    citation = Citation(
                        text=citation_text,
                        source_url=source.url,
                        source_title=source.title,
                        relevance_score=relevance_score,
                        position=i + 1
                    )
                    citations.append(citation)
            
            # If no citations found, create citations for top sources mentioned
            if not citations:
                citations = await self._create_default_citations(answer, sources[:3])
            
            return citations
            
        except Exception as e:
            logger.error(f"Citation extraction failed: {str(e)}")
            return []
    
    def _find_potential_citations(self, text: str) -> List[str]:
        """
        Find potential citation text in the answer
        """
        potential_citations = []
        
        # Look for quoted text
        quotes = re.findall(r'"([^"]{20,200})"', text)
        potential_citations.extend(quotes)
        
        # Look for key factual statements
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30 and any(keyword in sentence.lower() for keyword in 
                ['research', 'study', 'according', 'shows', 'indicates', 'found']):
                potential_citations.append(sentence)
        
        # Remove duplicates and sort by length (prefer longer citations)
        potential_citations = list(set(potential_citations))
        potential_citations.sort(key=len, reverse=True)
        
        return potential_citations[:5]  # Return top 5
    
    async def _find_best_source_match(
        self,
        citation_text: str,
        sources: List[Source]
    ) -> Optional[tuple]:
        """
        Find the best matching source for a citation
        """
        best_match = None
        best_score = 0.0
        
        citation_words = set(citation_text.lower().split())
        
        for source in sources:
            # Calculate overlap with source content
            source_words = set(source.content.lower().split())
            overlap = len(citation_words.intersection(source_words))
            
            # Calculate similarity score
            if len(citation_words) > 0:
                similarity = overlap / len(citation_words)
                
                # Boost score if citation appears in source
                if citation_text.lower() in source.content.lower():
                    similarity += 0.5
                
                if similarity > best_score and similarity > 0.3:
                    best_score = similarity
                    best_match = (source, similarity)
        
        return best_match
    
    async def _create_default_citations(
        self,
        answer: str,
        sources: List[Source]
    ) -> List[Citation]:
        """
        Create default citations when no specific citations are found
        """
        citations = []
        
        for i, source in enumerate(sources):
            # Use first sentence or key phrase from source
            content_sentences = source.content.split('.')
            citation_text = content_sentences[0].strip() if content_sentences else source.title
            
            # Ensure minimum length
            if len(citation_text) < 20 and len(content_sentences) > 1:
                citation_text = content_sentences[0] + '. ' + content_sentences[1]
            
            citation = Citation(
                text=citation_text[:150] + "..." if len(citation_text) > 150 else citation_text,
                source_url=source.url,
                source_title=source.title,
                relevance_score=source.relevance_score,
                position=i + 1
            )
            citations.append(citation)
        
        return citations
    
    async def format_citations_for_display(
        self,
        citations: List[Citation]
    ) -> List[Dict[str, Any]]:
        """
        Format citations for frontend display
        """
        formatted_citations = []
        
        for citation in citations:
            formatted = {
                "id": f"cite_{citation.position}",
                "text": citation.text,
                "source": {
                    "title": citation.source_title,
                    "url": citation.source_url,
                    "display_url": self._format_display_url(citation.source_url)
                },
                "relevance": round(citation.relevance_score, 2),
                "position": citation.position
            }
            formatted_citations.append(formatted)
        
        return formatted_citations
    
    def _format_display_url(self, url: str) -> str:
        """
        Format URL for display (remove protocol, www, etc.)
        """
        # Remove protocol
        display_url = re.sub(r'^https?://', '', url)
        
        # Remove www
        display_url = re.sub(r'^www\.', '', display_url)
        
        # Truncate if too long
        if len(display_url) > 50:
            display_url = display_url[:47] + "..."
        
        return display_url