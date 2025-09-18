"""
Enhanced Response Formatter
Intelligently formats AI responses without markdown artifacts and adjusts detail level
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import nltk
from textstat import flesch_reading_ease, automated_readability_index

logger = logging.getLogger(__name__)

class DetailLevel(Enum):
    MINIMAL = "minimal"      # Brief, key points only
    CONCISE = "concise"      # Standard length with essential details
    DETAILED = "detailed"    # Comprehensive with explanations
    EXTENSIVE = "extensive"  # Full depth with examples and context

class ResponseStyle(Enum):
    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"
    SIMPLE = "simple"

@dataclass
class ResponsePreferences:
    detail_level: DetailLevel
    style: ResponseStyle
    include_sources: bool
    max_paragraphs: int
    prefer_lists: bool
    include_examples: bool

@dataclass
class FormattedResponse:
    content: str
    word_count: int
    reading_level: float
    sources: List[Dict[str, str]]
    key_points: List[str]
    confidence_score: float

class EnhancedResponseFormatter:
    """
    Advanced response formatter that creates clean, well-structured responses
    without markdown artifacts and with intelligent detail control
    """
    
    def __init__(self):
        self.default_preferences = ResponsePreferences(
            detail_level=DetailLevel.CONCISE,
            style=ResponseStyle.CONVERSATIONAL,
            include_sources=True,
            max_paragraphs=5,
            prefer_lists=False,
            include_examples=True
        )
        
        # Pattern matching for different content types
        self.content_patterns = {
            'technical': ['algorithm', 'implementation', 'code', 'system', 'technology'],
            'explanatory': ['how', 'why', 'what', 'explain', 'describe'],
            'comparative': ['vs', 'versus', 'compare', 'difference', 'better'],
            'instructional': ['how to', 'steps', 'guide', 'tutorial', 'process']
        }
    
    def analyze_query_complexity(self, query: str) -> Tuple[str, float]:
        """Analyze query to determine appropriate response complexity"""
        
        query_lower = query.lower()
        complexity_indicators = {
            'simple': ['what is', 'define', 'meaning', 'yes or no'],
            'moderate': ['how', 'why', 'explain', 'describe'],
            'complex': ['analyze', 'compare', 'evaluate', 'discuss', 'comprehensive'],
            'technical': ['implement', 'algorithm', 'system', 'technical', 'advanced']
        }
        
        complexity_score = 0.3  # Base complexity
        query_type = 'simple'
        
        # Check for complexity indicators
        for level, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    if level == 'simple':
                        complexity_score = 0.2
                        query_type = 'simple'
                    elif level == 'moderate':
                        complexity_score = 0.5
                        query_type = 'moderate'
                    elif level == 'complex':
                        complexity_score = 0.7
                        query_type = 'complex'
                    elif level == 'technical':
                        complexity_score = 0.9
                        query_type = 'technical'
                    break
        
        # Adjust based on query length
        word_count = len(query.split())
        if word_count > 15:
            complexity_score += 0.1
        elif word_count < 5:
            complexity_score -= 0.1
        
        return query_type, min(1.0, max(0.1, complexity_score))
    
    def determine_optimal_detail_level(
        self, 
        query: str, 
        user_preferences: Optional[ResponsePreferences] = None
    ) -> DetailLevel:
        """Determine optimal detail level based on query and user preferences"""
        
        query_type, complexity = self.analyze_query_complexity(query)
        
        # Default mapping
        if complexity < 0.3:
            detail_level = DetailLevel.MINIMAL
        elif complexity < 0.6:
            detail_level = DetailLevel.CONCISE
        elif complexity < 0.8:
            detail_level = DetailLevel.DETAILED
        else:
            detail_level = DetailLevel.EXTENSIVE
        
        # Apply user preferences if available
        if user_preferences:
            # User preference overrides automatic detection
            detail_level = user_preferences.detail_level
        
        return detail_level
    
    def clean_markdown_artifacts(self, text: str) -> str:
        """Remove all markdown artifacts including stars, asterisks, and formatting"""
        
        # Remove various markdown patterns
        cleaned = text
        
        # Remove bold/italic asterisks and underscores
        cleaned = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', cleaned)
        cleaned = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', cleaned)
        
        # Remove headers (# symbols)
        cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
        
        # Remove horizontal rules
        cleaned = re.sub(r'^---+$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\*{3,}$', '', cleaned, flags=re.MULTILINE)
        
        # Remove code blocks and inline code
        cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
        
        # Remove link formatting but keep the text and URL
        cleaned = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', cleaned)
        
        # Remove image references
        cleaned = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', cleaned)
        
        # Remove blockquotes
        cleaned = re.sub(r'^>\s+', '', cleaned, flags=re.MULTILINE)
        
        # Clean up multiple spaces and newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # Remove bullet points and list markers while preserving content
        cleaned = re.sub(r'^[\s]*[-*+•]\s+', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()
    
    def structure_content_by_detail_level(
        self, 
        content: str, 
        detail_level: DetailLevel,
        style: ResponseStyle = ResponseStyle.CONVERSATIONAL
    ) -> str:
        """Structure content according to specified detail level"""
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if detail_level == DetailLevel.MINIMAL:
            # Keep only the most essential information
            structured = self._create_minimal_response(paragraphs)
        elif detail_level == DetailLevel.CONCISE:
            # Standard response with key points
            structured = self._create_concise_response(paragraphs)
        elif detail_level == DetailLevel.DETAILED:
            # Comprehensive explanation
            structured = self._create_detailed_response(paragraphs)
        else:  # EXTENSIVE
            # Full depth with examples
            structured = self._create_extensive_response(paragraphs)
        
        return self._apply_style(structured, style)
    
    def _create_minimal_response(self, paragraphs: List[str]) -> str:
        """Create minimal response with key points only"""
        if not paragraphs:
            return ""
        
        # Take first paragraph and extract key sentence
        first_para = paragraphs[0]
        sentences = [s.strip() for s in first_para.split('.') if s.strip()]
        
        if sentences:
            key_sentence = sentences[0] + '.'
            # Add one more sentence if available and short
            if len(sentences) > 1 and len(sentences[1]) < 100:
                key_sentence += ' ' + sentences[1] + '.'
            return key_sentence
        
        return first_para[:200] + ('...' if len(first_para) > 200 else '')
    
    def _create_concise_response(self, paragraphs: List[str]) -> str:
        """Create concise response with essential details"""
        if not paragraphs:
            return ""
        
        # Take first 2-3 paragraphs, condensed
        selected_paras = paragraphs[:3]
        condensed = []
        
        for para in selected_paras:
            sentences = [s.strip() for s in para.split('.') if s.strip()]
            # Keep first 2-3 sentences per paragraph
            key_sentences = sentences[:3]
            if key_sentences:
                condensed.append('. '.join(key_sentences) + '.')
        
        return '\n\n'.join(condensed)
    
    def _create_detailed_response(self, paragraphs: List[str]) -> str:
        """Create detailed response with comprehensive explanation"""
        if not paragraphs:
            return ""
        
        # Keep most paragraphs but ensure good flow
        selected_paras = paragraphs[:5]
        
        detailed = []
        for para in selected_paras:
            # Clean up but keep full content
            clean_para = para.strip()
            if clean_para:
                detailed.append(clean_para)
        
        return '\n\n'.join(detailed)
    
    def _create_extensive_response(self, paragraphs: List[str]) -> str:
        """Create extensive response with full depth"""
        # Keep all content but ensure it's well-structured
        return '\n\n'.join(p.strip() for p in paragraphs if p.strip())
    
    def _apply_style(self, content: str, style: ResponseStyle) -> str:
        """Apply specific writing style to content"""
        
        if style == ResponseStyle.PROFESSIONAL:
            # More formal language
            content = re.sub(r'\bi\b', 'I', content, flags=re.IGNORECASE)
            content = re.sub(r'\byou\b', 'you', content)
            
        elif style == ResponseStyle.CONVERSATIONAL:
            # Natural, friendly tone (default)
            pass
            
        elif style == ResponseStyle.ACADEMIC:
            # More scholarly language
            content = re.sub(r'\bcan\'t\b', 'cannot', content, flags=re.IGNORECASE)
            content = re.sub(r'\bdon\'t\b', 'do not', content, flags=re.IGNORECASE)
            content = re.sub(r'\bwon\'t\b', 'will not', content, flags=re.IGNORECASE)
            
        elif style == ResponseStyle.SIMPLE:
            # Simplified language
            pass
        
        return content
    
    def extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content"""
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        key_points = []
        
        # Look for sentences that start with key indicators
        key_indicators = [
            'the main', 'importantly', 'key', 'essential', 'primary',
            'first', 'second', 'third', 'finally', 'most', 'best'
        ]
        
        for sentence in sentences[:10]:  # Check first 10 sentences
            sentence_lower = sentence.lower()
            
            # Check if sentence contains key indicators
            for indicator in key_indicators:
                if indicator in sentence_lower:
                    key_points.append(sentence + '.')
                    break
            
            # Add first and last sentences as likely key points
            if sentence == sentences[0] or (len(sentences) > 1 and sentence == sentences[-1]):
                if sentence + '.' not in key_points:
                    key_points.append(sentence + '.')
        
        return key_points[:5]  # Return top 5 key points
    
    def format_response_with_sources(
        self, 
        content: str, 
        sources: List[Dict[str, Any]] = None,
        preferences: Optional[ResponsePreferences] = None
    ) -> FormattedResponse:
        """Format complete response with sources and metadata"""
        
        if preferences is None:
            preferences = self.default_preferences
        
        # Clean markdown artifacts
        clean_content = self.clean_markdown_artifacts(content)
        
        # Structure according to detail level
        structured_content = self.structure_content_by_detail_level(
            clean_content, 
            preferences.detail_level,
            preferences.style
        )
        
        # Extract key points
        key_points = self.extract_key_points(structured_content)
        
        # Calculate metrics
        word_count = len(structured_content.split())
        
        try:
            reading_level = flesch_reading_ease(structured_content)
        except:
            reading_level = 50.0  # Default reading level
        
        # Process sources
        formatted_sources = []
        if sources and preferences.include_sources:
            for source in sources[:5]:  # Limit to 5 sources
                formatted_source = {
                    'title': source.get('title', 'Unknown Source'),
                    'url': source.get('url', ''),
                    'snippet': source.get('snippet', ''),
                    'credibility': source.get('credibility_score', 0.5)
                }
                formatted_sources.append(formatted_source)
        
        # Calculate confidence score based on content quality
        confidence_score = self._calculate_confidence_score(
            structured_content, 
            formatted_sources, 
            word_count
        )
        
        return FormattedResponse(
            content=structured_content,
            word_count=word_count,
            reading_level=reading_level,
            sources=formatted_sources,
            key_points=key_points,
            confidence_score=confidence_score
        )
    
    def _calculate_confidence_score(
        self, 
        content: str, 
        sources: List[Dict], 
        word_count: int
    ) -> float:
        """Calculate confidence score for the response"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on content length and structure
        if 50 <= word_count <= 300:
            confidence += 0.1
        elif word_count > 300:
            confidence += 0.05
        
        # Increase confidence based on sources
        if sources:
            avg_credibility = sum(s.get('credibility', 0.5) for s in sources) / len(sources)
            confidence += (avg_credibility * 0.2)
        
        # Check for uncertainty indicators
        uncertainty_words = ['might', 'could', 'possibly', 'perhaps', 'maybe', 'uncertain']
        uncertainty_count = sum(1 for word in uncertainty_words if word in content.lower())
        confidence -= (uncertainty_count * 0.05)
        
        # Check for definitive statements
        definitive_words = ['is', 'are', 'will', 'always', 'never', 'definitely']
        definitive_count = sum(1 for word in definitive_words if word in content.lower())
        confidence += (definitive_count * 0.02)
        
        return min(1.0, max(0.1, confidence))
    
    def adapt_to_user_feedback(
        self, 
        user_id: str, 
        response: FormattedResponse, 
        feedback_score: float
    ) -> ResponsePreferences:
        """Adapt formatting preferences based on user feedback"""
        
        # This would integrate with the personalization engine
        # For now, return default preferences
        return self.default_preferences

# Global formatter instance
response_formatter = EnhancedResponseFormatter()