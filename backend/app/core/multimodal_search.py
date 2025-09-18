"""
Advanced Multi-Modal Search Engine
Supports text, images, code, audio, video, and structured data with cross-modal relevance
"""

import numpy as np
import json
import re
import ast
import time
import asyncio
import hashlib
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import logging
import base64
from urllib.parse import urlparse
import mimetypes

logger = logging.getLogger(__name__)

class ContentType(Enum):
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    STRUCTURED_DATA = "structured_data"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"

@dataclass
class MultiModalDocument:
    """Enhanced document supporting multiple content modalities"""
    id: str
    title: str
    content_type: ContentType
    primary_content: str  # Main text content
    
    # Modality-specific content
    code_content: Optional[str] = None
    code_language: Optional[str] = None
    
    image_data: Optional[bytes] = None
    image_url: Optional[str] = None
    image_alt_text: Optional[str] = None
    image_caption: Optional[str] = None
    
    audio_url: Optional[str] = None
    audio_transcript: Optional[str] = None
    audio_duration: Optional[float] = None
    
    video_url: Optional[str] = None
    video_transcript: Optional[str] = None
    video_duration: Optional[float] = None
    video_thumbnails: List[str] = field(default_factory=list)
    
    structured_data: Optional[Dict[str, Any]] = None
    table_data: Optional[List[List[str]]] = None
    
    # Vector representations for different modalities
    text_vector: Optional[np.ndarray] = None
    code_vector: Optional[np.ndarray] = None
    image_vector: Optional[np.ndarray] = None
    audio_vector: Optional[np.ndarray] = None
    video_vector: Optional[np.ndarray] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source_url: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class MultiModalQuery:
    """Query supporting multiple input modalities"""
    text_query: Optional[str] = None
    image_query: Optional[bytes] = None
    code_query: Optional[str] = None
    audio_query: Optional[bytes] = None
    video_query: Optional[bytes] = None
    
    # Query context
    intent: str = "search"  # search, compare, analyze, generate
    preferred_modalities: List[ContentType] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # User context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class CodeAnalyzer:
    """Advanced code analysis and understanding"""
    
    def __init__(self):
        self.language_patterns = {
            'python': [r'def\s+\w+', r'import\s+\w+', r'class\s+\w+', r'if\s+__name__'],
            'javascript': [r'function\s+\w+', r'const\s+\w+', r'let\s+\w+', r'var\s+\w+'],
            'java': [r'public\s+class', r'private\s+\w+', r'public\s+static', r'import\s+java'],
            'c++': [r'#include', r'int\s+main', r'class\s+\w+', r'namespace\s+\w+'],
            'sql': [r'SELECT\s+', r'FROM\s+', r'WHERE\s+', r'INSERT\s+INTO'],
            'html': [r'<html>', r'<div>', r'<script>', r'<!DOCTYPE'],
            'css': [r'\.[\w-]+\s*{', r'#[\w-]+\s*{', r'@media', r'font-family'],
        }
        
        self.code_features = {
            'complexity': self._calculate_complexity,
            'readability': self._calculate_readability,
            'functionality': self._extract_functionality,
            'dependencies': self._extract_dependencies,
            'patterns': self._identify_patterns
        }
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Comprehensive code analysis"""
        
        analysis = {
            'language': self._detect_language(code),
            'lines_of_code': len(code.split('\n')),
            'complexity_score': 0.0,
            'readability_score': 0.0,
            'functions': [],
            'classes': [],
            'imports': [],
            'keywords': [],
            'patterns': [],
            'documentation': self._extract_documentation(code)
        }
        
        # Apply feature extractors
        for feature_name, extractor in self.code_features.items():
            try:
                result = extractor(code)
                if isinstance(result, dict):
                    analysis.update(result)
                else:
                    analysis[feature_name] = result
            except Exception as e:
                logger.warning(f"Code analysis failed for {feature_name}: {e}")
        
        return analysis
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language"""
        scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, code, re.IGNORECASE))
                score += matches
            scores[lang] = score
        
        if not scores or max(scores.values()) == 0:
            return 'unknown'
        
        return max(scores, key=scores.get)
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity"""
        # Simplified complexity calculation
        complexity_keywords = ['if', 'else', 'elif', 'while', 'for', 'try', 'except', 'case', 'switch']
        
        complexity = 1  # Base complexity
        for keyword in complexity_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', code, re.IGNORECASE))
        
        # Normalize by lines of code
        lines = len(code.split('\n'))
        return min(10.0, complexity / max(1, lines / 10))
    
    def _calculate_readability(self, code: str) -> float:
        """Calculate code readability score"""
        lines = code.split('\n')
        
        # Readability factors
        avg_line_length = sum(len(line) for line in lines) / max(1, len(lines))
        comment_ratio = sum(1 for line in lines if line.strip().startswith('#') or '//' in line) / max(1, len(lines))
        blank_line_ratio = sum(1 for line in lines if not line.strip()) / max(1, len(lines))
        
        # Score based on best practices
        readability = 1.0
        
        # Penalize very long lines
        if avg_line_length > 80:
            readability -= 0.3
        
        # Reward comments
        readability += min(0.3, comment_ratio * 2)
        
        # Reward reasonable blank lines
        readability += min(0.2, blank_line_ratio * 5)
        
        return max(0.0, min(1.0, readability))
    
    def _extract_functionality(self, code: str) -> Dict[str, List[str]]:
        """Extract functions and classes"""
        functions = re.findall(r'def\s+(\w+)', code)
        classes = re.findall(r'class\s+(\w+)', code)
        
        return {
            'functions': functions,
            'classes': classes
        }
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract imports and dependencies"""
        imports = []
        
        # Python imports
        imports.extend(re.findall(r'import\s+(\w+)', code))
        imports.extend(re.findall(r'from\s+(\w+)\s+import', code))
        
        # JavaScript imports
        imports.extend(re.findall(r'require\([\'"]([^\'"]+)[\'"]\)', code))
        imports.extend(re.findall(r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]', code))
        
        return list(set(imports))
    
    def _identify_patterns(self, code: str) -> List[str]:
        """Identify common programming patterns"""
        patterns = []
        
        # Design patterns
        if re.search(r'class\s+\w+Factory', code, re.IGNORECASE):
            patterns.append('Factory Pattern')
        
        if re.search(r'class\s+\w+Singleton', code, re.IGNORECASE):
            patterns.append('Singleton Pattern')
        
        if re.search(r'def\s+__init__.*Observer', code, re.IGNORECASE):
            patterns.append('Observer Pattern')
        
        # Common algorithms
        if re.search(r'bubble.*sort|sort.*bubble', code, re.IGNORECASE):
            patterns.append('Bubble Sort')
        
        if re.search(r'binary.*search|search.*binary', code, re.IGNORECASE):
            patterns.append('Binary Search')
        
        # Data structures
        if re.search(r'LinkedList|linked.*list', code, re.IGNORECASE):
            patterns.append('Linked List')
        
        if re.search(r'HashMap|hash.*map', code, re.IGNORECASE):
            patterns.append('Hash Map')
        
        return patterns
    
    def _extract_documentation(self, code: str) -> Dict[str, str]:
        """Extract documentation strings and comments"""
        
        # Python docstrings
        docstrings = re.findall(r'"""(.*?)"""', code, re.DOTALL)
        docstrings.extend(re.findall(r"'''(.*?)'''", code, re.DOTALL))
        
        # Comments
        comments = re.findall(r'#\s*(.*)', code)
        comments.extend(re.findall(r'//\s*(.*)', code))
        
        return {
            'docstrings': docstrings,
            'comments': comments
        }
    
    def calculate_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets"""
        
        # Analyze both code snippets
        analysis1 = self.analyze_code(code1)
        analysis2 = self.analyze_code(code2)
        
        similarity = 0.0
        
        # Language similarity
        if analysis1['language'] == analysis2['language']:
            similarity += 0.3
        
        # Function similarity
        funcs1 = set(analysis1.get('functions', []))
        funcs2 = set(analysis2.get('functions', []))
        if funcs1 or funcs2:
            func_similarity = len(funcs1.intersection(funcs2)) / len(funcs1.union(funcs2))
            similarity += 0.2 * func_similarity
        
        # Pattern similarity
        patterns1 = set(analysis1.get('patterns', []))
        patterns2 = set(analysis2.get('patterns', []))
        if patterns1 or patterns2:
            pattern_similarity = len(patterns1.intersection(patterns2)) / len(patterns1.union(patterns2))
            similarity += 0.2 * pattern_similarity
        
        # Complexity similarity
        comp1 = analysis1.get('complexity_score', 0)
        comp2 = analysis2.get('complexity_score', 0)
        complexity_similarity = 1.0 - abs(comp1 - comp2) / max(comp1, comp2, 1)
        similarity += 0.15 * complexity_similarity
        
        # Text similarity (simplified)
        words1 = set(re.findall(r'\w+', code1.lower()))
        words2 = set(re.findall(r'\w+', code2.lower()))
        if words1 or words2:
            text_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
            similarity += 0.15 * text_similarity
        
        return similarity

class ImageAnalyzer:
    """Advanced image analysis and understanding"""
    
    def __init__(self):
        self.supported_formats = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg', 'bmp']
    
    def analyze_image(self, image_data: bytes, image_url: str = None) -> Dict[str, Any]:
        """Analyze image content (mock implementation)"""
        
        analysis = {
            'format': self._detect_format(image_data),
            'size_bytes': len(image_data),
            'estimated_dimensions': self._estimate_dimensions(image_data),
            'content_type': self._classify_content(image_data),
            'has_text': self._detect_text(image_data),
            'dominant_colors': self._extract_colors(image_data),
            'objects_detected': self._detect_objects(image_data),
            'quality_score': self._assess_quality(image_data),
            'accessibility': {
                'has_alt_text': False,
                'contrast_ratio': 0.0,
                'readability': 'unknown'
            }
        }
        
        return analysis
    
    def _detect_format(self, image_data: bytes) -> str:
        """Detect image format from binary data"""
        
        # Check magic bytes
        if image_data.startswith(b'\xFF\xD8\xFF'):
            return 'jpeg'
        elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png'
        elif image_data.startswith(b'GIF8'):
            return 'gif'
        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
            return 'webp'
        elif image_data.startswith(b'BM'):
            return 'bmp'
        elif b'<svg' in image_data[:100].lower():
            return 'svg'
        else:
            return 'unknown'
    
    def _estimate_dimensions(self, image_data: bytes) -> Tuple[int, int]:
        """Estimate image dimensions (mock implementation)"""
        # In production, use PIL or similar library
        return (800, 600)  # Mock dimensions
    
    def _classify_content(self, image_data: bytes) -> str:
        """Classify image content type"""
        # Mock classification
        size = len(image_data)
        
        if size < 10000:
            return 'icon'
        elif size < 100000:
            return 'photo'
        elif size < 500000:
            return 'detailed_image'
        else:
            return 'high_resolution'
    
    def _detect_text(self, image_data: bytes) -> bool:
        """Detect if image contains text (OCR simulation)"""
        # Mock text detection
        return len(image_data) > 50000  # Assume larger images might have text
    
    def _extract_colors(self, image_data: bytes) -> List[str]:
        """Extract dominant colors"""
        # Mock color extraction
        return ['#FF5733', '#33FF57', '#3357FF']
    
    def _detect_objects(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        # Mock object detection
        return [
            {'object': 'person', 'confidence': 0.85, 'bbox': [10, 20, 100, 200]},
            {'object': 'car', 'confidence': 0.72, 'bbox': [150, 50, 300, 180]}
        ]
    
    def _assess_quality(self, image_data: bytes) -> float:
        """Assess image quality"""
        # Simple quality assessment based on size and format
        size = len(image_data)
        
        if size < 5000:
            return 0.3  # Very low quality
        elif size < 50000:
            return 0.6  # Medium quality
        elif size < 500000:
            return 0.8  # Good quality
        else:
            return 0.9  # High quality
    
    def calculate_similarity(self, image1_data: bytes, image2_data: bytes) -> float:
        """Calculate similarity between two images"""
        
        analysis1 = self.analyze_image(image1_data)
        analysis2 = self.analyze_image(image2_data)
        
        similarity = 0.0
        
        # Format similarity
        if analysis1['format'] == analysis2['format']:
            similarity += 0.1
        
        # Content type similarity
        if analysis1['content_type'] == analysis2['content_type']:
            similarity += 0.2
        
        # Dimension similarity
        dims1 = analysis1['estimated_dimensions']
        dims2 = analysis2['estimated_dimensions']
        
        aspect_ratio1 = dims1[0] / dims1[1] if dims1[1] > 0 else 1
        aspect_ratio2 = dims2[0] / dims2[1] if dims2[1] > 0 else 1
        
        aspect_similarity = 1.0 - abs(aspect_ratio1 - aspect_ratio2) / max(aspect_ratio1, aspect_ratio2)
        similarity += 0.2 * aspect_similarity
        
        # Color similarity (mock)
        colors1 = set(analysis1['dominant_colors'])
        colors2 = set(analysis2['dominant_colors'])
        
        if colors1 or colors2:
            color_similarity = len(colors1.intersection(colors2)) / len(colors1.union(colors2))
            similarity += 0.3 * color_similarity
        
        # Object similarity (mock)
        objects1 = {obj['object'] for obj in analysis1['objects_detected']}
        objects2 = {obj['object'] for obj in analysis2['objects_detected']}
        
        if objects1 or objects2:
            object_similarity = len(objects1.intersection(objects2)) / len(objects1.union(objects2))
            similarity += 0.2 * object_similarity
        
        return similarity

class MultiModalSearchEngine:
    """Advanced multi-modal search engine"""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        
        # Document storage by modality
        self.documents: Dict[str, MultiModalDocument] = {}
        self.text_index: Dict[str, List[str]] = defaultdict(list)
        self.code_index: Dict[str, List[str]] = defaultdict(list)
        self.image_index: Dict[str, List[str]] = defaultdict(list)
        
        # Cross-modal similarity cache
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'modality_usage': defaultdict(int),
            'cross_modal_searches': 0,
            'avg_response_time': 0.0
        }
    
    def add_document(self, document: MultiModalDocument) -> bool:
        """Add a multi-modal document to the search engine"""
        
        try:
            # Store document
            self.documents[document.id] = document
            
            # Index by modality
            if document.primary_content:
                self._index_text_content(document.id, document.primary_content)
            
            if document.code_content:
                self._index_code_content(document.id, document.code_content)
            
            if document.image_data or document.image_url:
                self._index_image_content(document.id, document)
            
            if document.audio_transcript:
                self._index_text_content(document.id, document.audio_transcript)
            
            if document.video_transcript:
                self._index_text_content(document.id, document.video_transcript)
            
            if document.structured_data:
                self._index_structured_data(document.id, document.structured_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {e}")
            return False
    
    def _index_text_content(self, doc_id: str, content: str):
        """Index text content"""
        words = re.findall(r'\w+', content.lower())
        for word in words:
            self.text_index[word].append(doc_id)
    
    def _index_code_content(self, doc_id: str, code: str):
        """Index code content"""
        analysis = self.code_analyzer.analyze_code(code)
        
        # Index by language
        language = analysis.get('language', 'unknown')
        self.code_index[f"lang:{language}"].append(doc_id)
        
        # Index functions
        for func in analysis.get('functions', []):
            self.code_index[f"func:{func}"].append(doc_id)
        
        # Index classes
        for cls in analysis.get('classes', []):
            self.code_index[f"class:{cls}"].append(doc_id)
        
        # Index patterns
        for pattern in analysis.get('patterns', []):
            self.code_index[f"pattern:{pattern}"].append(doc_id)
        
        # Index dependencies
        for dep in analysis.get('dependencies', []):
            self.code_index[f"import:{dep}"].append(doc_id)
    
    def _index_image_content(self, doc_id: str, document: MultiModalDocument):
        """Index image content"""
        if document.image_data:
            analysis = self.image_analyzer.analyze_image(document.image_data)
            
            # Index by content type
            content_type = analysis.get('content_type', 'unknown')
            self.image_index[f"type:{content_type}"].append(doc_id)
            
            # Index by format
            format_type = analysis.get('format', 'unknown')
            self.image_index[f"format:{format_type}"].append(doc_id)
            
            # Index detected objects
            for obj in analysis.get('objects_detected', []):
                self.image_index[f"object:{obj['object']}"].append(doc_id)
        
        # Index alt text and captions
        if document.image_alt_text:
            self._index_text_content(doc_id, document.image_alt_text)
        
        if document.image_caption:
            self._index_text_content(doc_id, document.image_caption)
    
    def _index_structured_data(self, doc_id: str, data: Dict[str, Any]):
        """Index structured data"""
        
        def extract_values(obj, prefix=""):
            """Recursively extract values from structured data"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    extract_values(value, new_prefix)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_values(item, f"{prefix}[{i}]")
            else:
                # Index the value
                value_str = str(obj).lower()
                self.text_index[f"structured:{prefix}:{value_str}"].append(doc_id)
        
        extract_values(data)
    
    async def search(self, query: MultiModalQuery, max_results: int = 20) -> List[Tuple[float, MultiModalDocument]]:
        """Perform multi-modal search"""
        
        start_time = time.time()
        
        try:
            # Track search statistics
            self.search_stats['total_searches'] += 1
            
            # Initialize candidates
            all_candidates: Dict[str, float] = defaultdict(float)
            
            # Search by text query
            if query.text_query:
                self.search_stats['modality_usage']['text'] += 1
                text_candidates = await self._search_text(query.text_query)
                self._merge_candidates(all_candidates, text_candidates, weight=0.4)
            
            # Search by code query
            if query.code_query:
                self.search_stats['modality_usage']['code'] += 1
                code_candidates = await self._search_code(query.code_query)
                self._merge_candidates(all_candidates, code_candidates, weight=0.3)
            
            # Search by image query
            if query.image_query:
                self.search_stats['modality_usage']['image'] += 1
                image_candidates = await self._search_image(query.image_query)
                self._merge_candidates(all_candidates, image_candidates, weight=0.3)
            
            # Apply cross-modal relevance boosting
            if len([q for q in [query.text_query, query.code_query, query.image_query] if q]) > 1:
                self.search_stats['cross_modal_searches'] += 1
                all_candidates = await self._apply_cross_modal_boosting(all_candidates, query)
            
            # Apply filters
            filtered_candidates = self._apply_filters(all_candidates, query.filters)
            
            # Apply modality preferences
            preferred_candidates = self._apply_modality_preferences(
                filtered_candidates, query.preferred_modalities
            )
            
            # Sort and limit results
            sorted_candidates = sorted(preferred_candidates.items(), key=lambda x: x[1], reverse=True)
            top_candidates = sorted_candidates[:max_results]
            
            # Convert to documents
            results = []
            for doc_id, score in top_candidates:
                if doc_id in self.documents:
                    results.append((score, self.documents[doc_id]))
            
            # Update statistics
            response_time = time.time() - start_time
            self.search_stats['avg_response_time'] = (
                (self.search_stats['avg_response_time'] * (self.search_stats['total_searches'] - 1) + 
                 response_time) / self.search_stats['total_searches']
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Multi-modal search failed: {e}")
            return []
    
    async def _search_text(self, text_query: str) -> Dict[str, float]:
        """Search text content"""
        candidates: Dict[str, float] = defaultdict(float)
        
        # Simple keyword search
        query_words = re.findall(r'\w+', text_query.lower())
        
        for word in query_words:
            if word in self.text_index:
                for doc_id in self.text_index[word]:
                    candidates[doc_id] += 1.0
        
        # Normalize scores
        max_score = max(candidates.values()) if candidates else 1.0
        normalized_candidates = {doc_id: score / max_score for doc_id, score in candidates.items()}
        
        return normalized_candidates
    
    async def _search_code(self, code_query: str) -> Dict[str, float]:
        """Search code content"""
        candidates: Dict[str, float] = defaultdict(float)
        
        # Analyze query code
        query_analysis = self.code_analyzer.analyze_code(code_query)
        
        # Search by language
        language = query_analysis.get('language', 'unknown')
        if f"lang:{language}" in self.code_index:
            for doc_id in self.code_index[f"lang:{language}"]:
                candidates[doc_id] += 0.3
        
        # Search by functions
        for func in query_analysis.get('functions', []):
            if f"func:{func}" in self.code_index:
                for doc_id in self.code_index[f"func:{func}"]:
                    candidates[doc_id] += 0.4
        
        # Search by patterns
        for pattern in query_analysis.get('patterns', []):
            if f"pattern:{pattern}" in self.code_index:
                for doc_id in self.code_index[f"pattern:{pattern}"]:
                    candidates[doc_id] += 0.5
        
        # Calculate semantic similarity for code
        for doc_id, document in self.documents.items():
            if document.code_content:
                similarity = self.code_analyzer.calculate_similarity(code_query, document.code_content)
                candidates[doc_id] += similarity * 0.6
        
        return candidates
    
    async def _search_image(self, image_query: bytes) -> Dict[str, float]:
        """Search image content"""
        candidates: Dict[str, float] = defaultdict(float)
        
        # Analyze query image
        query_analysis = self.image_analyzer.analyze_image(image_query)
        
        # Search by content type
        content_type = query_analysis.get('content_type', 'unknown')
        if f"type:{content_type}" in self.image_index:
            for doc_id in self.image_index[f"type:{content_type}"]:
                candidates[doc_id] += 0.3
        
        # Search by detected objects
        for obj in query_analysis.get('objects_detected', []):
            if f"object:{obj['object']}" in self.image_index:
                for doc_id in self.image_index[f"object:{obj['object']}"]:
                    candidates[doc_id] += 0.4
        
        # Calculate visual similarity
        for doc_id, document in self.documents.items():
            if document.image_data:
                similarity = self.image_analyzer.calculate_similarity(image_query, document.image_data)
                candidates[doc_id] += similarity * 0.7
        
        return candidates
    
    def _merge_candidates(self, all_candidates: Dict[str, float], 
                         new_candidates: Dict[str, float], weight: float):
        """Merge candidate results with weighting"""
        for doc_id, score in new_candidates.items():
            all_candidates[doc_id] += score * weight
    
    async def _apply_cross_modal_boosting(self, candidates: Dict[str, float], 
                                        query: MultiModalQuery) -> Dict[str, float]:
        """Apply cross-modal relevance boosting"""
        
        boosted_candidates = candidates.copy()
        
        for doc_id in candidates:
            if doc_id not in self.documents:
                continue
            
            document = self.documents[doc_id]
            boost_factor = 1.0
            
            # Text-Code cross-modal boost
            if query.text_query and document.code_content:
                # Check if text query mentions programming concepts
                programming_terms = ['function', 'class', 'method', 'algorithm', 'code', 'implement']
                if any(term in query.text_query.lower() for term in programming_terms):
                    boost_factor += 0.2
            
            # Text-Image cross-modal boost
            if query.text_query and (document.image_data or document.image_caption):
                # Check if text query mentions visual concepts
                visual_terms = ['image', 'picture', 'diagram', 'chart', 'graph', 'visual']
                if any(term in query.text_query.lower() for term in visual_terms):
                    boost_factor += 0.15
            
            # Code-Image cross-modal boost
            if query.code_query and document.image_data:
                # Boost if image might be a diagram or chart related to code
                image_analysis = self.image_analyzer.analyze_image(document.image_data)
                if image_analysis.get('content_type') in ['diagram', 'chart']:
                    boost_factor += 0.25
            
            boosted_candidates[doc_id] *= boost_factor
        
        return boosted_candidates
    
    def _apply_filters(self, candidates: Dict[str, float], 
                      filters: Dict[str, Any]) -> Dict[str, float]:
        """Apply search filters"""
        
        if not filters:
            return candidates
        
        filtered_candidates = {}
        
        for doc_id, score in candidates.items():
            if doc_id not in self.documents:
                continue
            
            document = self.documents[doc_id]
            include_document = True
            
            # Content type filter
            if 'content_type' in filters:
                allowed_types = filters['content_type']
                if isinstance(allowed_types, str):
                    allowed_types = [allowed_types]
                
                if document.content_type.value not in allowed_types:
                    include_document = False
            
            # Date range filter
            if 'date_range' in filters:
                date_range = filters['date_range']
                if document.timestamp < date_range.get('start', 0) or \
                   document.timestamp > date_range.get('end', float('inf')):
                    include_document = False
            
            # Author filter
            if 'author' in filters:
                if document.author != filters['author']:
                    include_document = False
            
            # Tags filter
            if 'tags' in filters:
                required_tags = set(filters['tags'])
                document_tags = set(document.tags)
                if not required_tags.intersection(document_tags):
                    include_document = False
            
            # Language filter (for code)
            if 'programming_language' in filters and document.code_content:
                analysis = self.code_analyzer.analyze_code(document.code_content)
                if analysis.get('language') != filters['programming_language']:
                    include_document = False
            
            if include_document:
                filtered_candidates[doc_id] = score
        
        return filtered_candidates
    
    def _apply_modality_preferences(self, candidates: Dict[str, float], 
                                  preferred_modalities: List[ContentType]) -> Dict[str, float]:
        """Apply modality preferences"""
        
        if not preferred_modalities:
            return candidates
        
        boosted_candidates = {}
        
        for doc_id, score in candidates.items():
            if doc_id not in self.documents:
                continue
            
            document = self.documents[doc_id]
            boost_factor = 1.0
            
            # Boost documents matching preferred modalities
            if document.content_type in preferred_modalities:
                boost_factor += 0.3
            
            # Additional boost for multi-modal documents
            modality_count = sum([
                bool(document.primary_content),
                bool(document.code_content),
                bool(document.image_data),
                bool(document.audio_transcript),
                bool(document.video_transcript)
            ])
            
            if modality_count > 1:
                boost_factor += 0.1 * (modality_count - 1)
            
            boosted_candidates[doc_id] = score * boost_factor
        
        return boosted_candidates
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        
        return {
            'total_documents': len(self.documents),
            'documents_by_type': {
                content_type.value: sum(1 for doc in self.documents.values() 
                                      if doc.content_type == content_type)
                for content_type in ContentType
            },
            'index_sizes': {
                'text_terms': len(self.text_index),
                'code_terms': len(self.code_index),
                'image_terms': len(self.image_index)
            },
            'search_statistics': self.search_stats.copy(),
            'cache_size': len(self.similarity_cache)
        }
    
    def get_document_by_id(self, doc_id: str) -> Optional[MultiModalDocument]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the search engine"""
        
        if doc_id not in self.documents:
            return False
        
        try:
            # Remove from main storage
            del self.documents[doc_id]
            
            # Remove from indices
            for term_list in self.text_index.values():
                if doc_id in term_list:
                    term_list.remove(doc_id)
            
            for term_list in self.code_index.values():
                if doc_id in term_list:
                    term_list.remove(doc_id)
            
            for term_list in self.image_index.values():
                if doc_id in term_list:
                    term_list.remove(doc_id)
            
            # Clean up empty index entries
            self.text_index = {k: v for k, v in self.text_index.items() if v}
            self.code_index = {k: v for k, v in self.code_index.items() if v}
            self.image_index = {k: v for k, v in self.image_index.items() if v}
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False