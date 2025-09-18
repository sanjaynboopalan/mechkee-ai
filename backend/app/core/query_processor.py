"""
Query Processor for intent understanding and query enhancement
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Advanced query processing for intent understanding and enhancement
    """
    
    def __init__(self):
        # Query types and patterns
        self.query_patterns = {
            'factual': [
                r'\b(what|who|when|where|why|how)\b',
                r'\b(define|explain|describe)\b',
                r'\bis\b.*\?',
                r'\bwhat\s+is\b'
            ],
            'comparison': [
                r'\b(vs|versus|compared to|difference between)\b',
                r'\b(better|worse|faster|slower)\b',
                r'\b(pros and cons|advantages|disadvantages)\b'
            ],
            'procedural': [
                r'\bhow\s+to\b',
                r'\bsteps\s+to\b',
                r'\btutorial\b',
                r'\bguide\b'
            ],
            'current_events': [
                r'\b(latest|recent|news|today|yesterday)\b',
                r'\b(current|now|present)\b',
                r'\b\d{4}\b'  # Year mentions
            ],
            'opinion': [
                r'\b(best|worst|favorite|recommend)\b',
                r'\b(opinion|think|believe)\b',
                r'\b(should|would|could)\b'
            ]
        }
        
        # Named entities patterns
        self.entity_patterns = {
            'person': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'organization': r'\b[A-Z][A-Z\s&]+\b',
            'location': r'\bin\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'date': r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b',
            'year': r'\b(19|20)\d{2}\b'
        }
        
        # Query expansion terms
        self.expansion_terms = {
            'AI': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
            'programming': ['coding', 'development', 'software engineering', 'computer science'],
            'business': ['company', 'enterprise', 'corporate', 'commercial'],
            'technology': ['tech', 'digital', 'innovation', 'computing']
        }
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive query processing
        """
        try:
            result = {
                'original_query': query,
                'normalized_query': self._normalize_query(query),
                'intent': await self._classify_intent(query),
                'entities': self._extract_entities(query),
                'expanded_query': await self._expand_query(query),
                'keywords': self._extract_keywords(query),
                'context_type': self._determine_context_type(query),
                'suggested_filters': self._suggest_filters(query)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                'original_query': query,
                'normalized_query': query,
                'intent': 'general',
                'entities': {},
                'expanded_query': query,
                'keywords': query.split(),
                'context_type': 'general',
                'suggested_filters': {}
            }
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text"""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common abbreviations
        abbreviations = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience'
        }
        
        for abbr, full in abbreviations.items():
            normalized = re.sub(rf'\b{abbr}\b', full, normalized)
        
        return normalized
    
    async def _classify_intent(self, query: str) -> str:
        """Classify query intent"""
        query_lower = query.lower()
        
        # Check each intent pattern
        for intent, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return 'general'
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities from query"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    async def _expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        expanded_parts = [query]
        query_lower = query.lower()
        
        # Add related terms
        for term, expansions in self.expansion_terms.items():
            if term.lower() in query_lower:
                # Add most relevant expansion
                expanded_parts.append(expansions[0])
        
        # Add semantic variations
        semantic_expansions = await self._get_semantic_expansions(query)
        expanded_parts.extend(semantic_expansions[:2])  # Add top 2
        
        return ' '.join(expanded_parts)
    
    async def _get_semantic_expansions(self, query: str) -> List[str]:
        """Get semantic expansions (mock implementation)"""
        # In production, use word embeddings or language models
        common_expansions = {
            'search': ['find', 'lookup', 'discover'],
            'create': ['make', 'build', 'develop'],
            'learn': ['study', 'understand', 'master'],
            'solve': ['fix', 'resolve', 'address'],
            'analyze': ['examine', 'study', 'investigate']
        }
        
        expansions = []
        for word in query.split():
            if word.lower() in common_expansions:
                expansions.extend(common_expansions[word.lower()][:1])
        
        return expansions
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by',
            'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of',
            'on', 'that', 'the', 'to', 'was', 'will', 'with', 'the'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _determine_context_type(self, query: str) -> str:
        """Determine the context type of the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['recent', 'latest', 'news', 'today']):
            return 'current'
        elif any(word in query_lower for word in ['history', 'past', 'originally']):
            return 'historical'
        elif any(word in query_lower for word in ['future', 'will', 'prediction']):
            return 'predictive'
        elif any(word in query_lower for word in ['tutorial', 'how to', 'guide']):
            return 'instructional'
        else:
            return 'general'
    
    def _suggest_filters(self, query: str) -> Dict[str, Any]:
        """Suggest search filters based on query"""
        filters = {}
        query_lower = query.lower()
        
        # Time-based filters
        if 'recent' in query_lower or 'latest' in query_lower:
            filters['time_range'] = 'recent'
        elif any(year in query_lower for year in ['2023', '2024']):
            filters['time_range'] = 'year_specific'
        
        # Content type filters
        if any(word in query_lower for word in ['tutorial', 'guide', 'how to']):
            filters['content_type'] = 'tutorial'
        elif any(word in query_lower for word in ['news', 'article']):
            filters['content_type'] = 'news'
        elif any(word in query_lower for word in ['research', 'study', 'paper']):
            filters['content_type'] = 'academic'
        
        # Domain filters
        if any(word in query_lower for word in ['programming', 'code', 'development']):
            filters['domain'] = 'technology'
        elif any(word in query_lower for word in ['business', 'market', 'finance']):
            filters['domain'] = 'business'
        elif any(word in query_lower for word in ['health', 'medical', 'medicine']):
            filters['domain'] = 'health'
        
        return filters