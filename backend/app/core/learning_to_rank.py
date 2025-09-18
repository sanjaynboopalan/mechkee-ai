"""
Advanced Learning-to-Rank System
Implements state-of-the-art ranking algorithms with real-time learning
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class FeatureVector:
    """Feature vector for a query-document pair"""
    query_id: str
    document_id: str
    features: np.ndarray
    relevance_score: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class RankingExample:
    """Training example for learning-to-rank"""
    query_id: str
    document_pairs: List[Tuple[str, str]]  # (winner_doc_id, loser_doc_id)
    feature_vectors: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)

class FeatureExtractor:
    """Extract comprehensive features for ranking"""
    
    def __init__(self):
        self.feature_names = [
            # Text similarity features
            'bm25_score', 'tf_idf_score', 'jaccard_similarity', 'cosine_text_sim',
            'edit_distance_norm', 'longest_common_subsequence', 'exact_match_count',
            'partial_match_count', 'phrase_match_count', 'query_coverage',
            
            # Vector similarity features
            'vector_cosine_sim', 'vector_euclidean_dist', 'vector_manhattan_dist',
            'vector_dot_product', 'vector_correlation',
            
            # Document quality features
            'doc_length', 'doc_uniqueness', 'readability_score', 'spelling_errors',
            'grammar_score', 'content_depth', 'information_density',
            
            # Authority and trust features
            'domain_authority', 'page_authority', 'citation_count', 'backlink_count',
            'author_expertise', 'publication_quality', 'fact_check_score',
            
            # Freshness and temporal features
            'document_age', 'last_updated', 'temporal_relevance', 'trend_score',
            'seasonal_relevance',
            
            # User engagement features
            'click_through_rate', 'dwell_time', 'bounce_rate', 'social_shares',
            'comment_count', 'like_count', 'view_count', 'download_count',
            
            # Query-specific features
            'query_type_match', 'intent_alignment', 'entity_overlap', 'topic_match',
            'query_complexity', 'query_specificity',
            
            # Structural features
            'url_depth', 'has_images', 'has_videos', 'has_tables', 'has_code',
            'mobile_friendly', 'load_speed', 'accessibility_score',
            
            # Personalization features
            'user_topic_preference', 'user_source_preference', 'historical_relevance',
            'collaborative_filtering_score'
        ]
        
        self.scaler = StandardScaler()
        self.feature_stats = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'mean': 0.0})
    
    def extract_features(self, query: str, document: Dict[str, Any], 
                        query_vector: np.ndarray, user_context: Dict[str, Any] = None) -> np.ndarray:
        """Extract all features for a query-document pair"""
        
        features = np.zeros(len(self.feature_names))
        doc_content = document.get('content', '')
        doc_metadata = document.get('metadata', {})
        
        # Text similarity features
        features[0] = self._calculate_bm25(query, doc_content)
        features[1] = self._calculate_tf_idf(query, doc_content)
        features[2] = self._calculate_jaccard_similarity(query, doc_content)
        features[3] = self._calculate_text_cosine_similarity(query, doc_content)
        features[4] = self._calculate_normalized_edit_distance(query, doc_content)
        features[5] = self._calculate_lcs_ratio(query, doc_content)
        features[6] = self._count_exact_matches(query, doc_content)
        features[7] = self._count_partial_matches(query, doc_content)
        features[8] = self._count_phrase_matches(query, doc_content)
        features[9] = self._calculate_query_coverage(query, doc_content)
        
        # Vector similarity features
        doc_vector = document.get('vector', np.zeros_like(query_vector))
        features[10] = self._vector_cosine_similarity(query_vector, doc_vector)
        features[11] = self._vector_euclidean_distance(query_vector, doc_vector)
        features[12] = self._vector_manhattan_distance(query_vector, doc_vector)
        features[13] = np.dot(query_vector, doc_vector)
        features[14] = np.corrcoef(query_vector, doc_vector)[0, 1] if len(query_vector) > 1 else 0.0
        
        # Document quality features
        features[15] = len(doc_content.split())
        features[16] = self._calculate_document_uniqueness(doc_content)
        features[17] = self._calculate_readability_score(doc_content)
        features[18] = self._count_spelling_errors(doc_content)
        features[19] = self._calculate_grammar_score(doc_content)
        features[20] = self._calculate_content_depth(doc_content)
        features[21] = self._calculate_information_density(doc_content)
        
        # Authority and trust features
        features[22] = doc_metadata.get('domain_authority', 0.0)
        features[23] = doc_metadata.get('page_authority', 0.0)
        features[24] = doc_metadata.get('citation_count', 0)
        features[25] = doc_metadata.get('backlink_count', 0)
        features[26] = doc_metadata.get('author_expertise', 0.0)
        features[27] = doc_metadata.get('publication_quality', 0.0)
        features[28] = doc_metadata.get('fact_check_score', 0.5)
        
        # Freshness and temporal features
        doc_timestamp = document.get('timestamp', time.time())
        current_time = time.time()
        features[29] = (current_time - doc_timestamp) / (24 * 3600)  # Age in days
        features[30] = doc_metadata.get('last_updated', 0.0)
        features[31] = self._calculate_temporal_relevance(query, doc_timestamp)
        features[32] = doc_metadata.get('trend_score', 0.0)
        features[33] = self._calculate_seasonal_relevance(query, doc_timestamp)
        
        # User engagement features
        features[34] = doc_metadata.get('click_through_rate', 0.0)
        features[35] = doc_metadata.get('avg_dwell_time', 0.0)
        features[36] = doc_metadata.get('bounce_rate', 0.0)
        features[37] = doc_metadata.get('social_shares', 0)
        features[38] = doc_metadata.get('comment_count', 0)
        features[39] = doc_metadata.get('like_count', 0)
        features[40] = doc_metadata.get('view_count', 0)
        features[41] = doc_metadata.get('download_count', 0)
        
        # Query-specific features
        features[42] = self._calculate_query_type_match(query, doc_content)
        features[43] = self._calculate_intent_alignment(query, doc_content)
        features[44] = self._calculate_entity_overlap(query, doc_content)
        features[45] = self._calculate_topic_match(query, doc_content)
        features[46] = self._calculate_query_complexity(query)
        features[47] = self._calculate_query_specificity(query)
        
        # Structural features
        features[48] = doc_metadata.get('url_depth', 0)
        features[49] = float(doc_metadata.get('has_images', False))
        features[50] = float(doc_metadata.get('has_videos', False))
        features[51] = float(doc_metadata.get('has_tables', False))
        features[52] = float(doc_metadata.get('has_code', False))
        features[53] = float(doc_metadata.get('mobile_friendly', True))
        features[54] = doc_metadata.get('load_speed', 1.0)
        features[55] = doc_metadata.get('accessibility_score', 0.5)
        
        # Personalization features (if user context available)
        if user_context:
            features[56] = user_context.get('topic_preference_score', 0.0)
            features[57] = user_context.get('source_preference_score', 0.0)
            features[58] = user_context.get('historical_relevance', 0.0)
            features[59] = user_context.get('collaborative_filtering_score', 0.0)
        
        # Normalize features
        features = self._normalize_features(features)
        
        return features
    
    def _calculate_bm25(self, query: str, document: str, k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 score (simplified version)"""
        query_terms = query.lower().split()
        doc_terms = document.lower().split()
        
        if not query_terms or not doc_terms:
            return 0.0
        
        doc_length = len(doc_terms)
        avg_doc_length = 100  # Simplified assumption
        
        score = 0.0
        for term in query_terms:
            tf = doc_terms.count(term)
            idf = math.log(1000 / (1 + doc_terms.count(term)))  # Simplified IDF
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def _calculate_tf_idf(self, query: str, document: str) -> float:
        """Calculate TF-IDF similarity"""
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        
        if not query_terms or not doc_terms:
            return 0.0
        
        score = 0.0
        for term in query_terms:
            tf = doc_terms.count(term) / len(doc_terms)
            idf = math.log(1000 / (1 + doc_terms.count(term)))
            score += tf * idf
        
        return score / len(query_terms)
    
    def _calculate_jaccard_similarity(self, query: str, document: str) -> float:
        """Calculate Jaccard similarity"""
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        
        if not query_terms or not doc_terms:
            return 0.0
        
        intersection = len(query_terms.intersection(doc_terms))
        union = len(query_terms.union(doc_terms))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_text_cosine_similarity(self, query: str, document: str) -> float:
        """Calculate cosine similarity between text"""
        # Simplified bag-of-words cosine similarity
        query_terms = query.lower().split()
        doc_terms = document.lower().split()
        
        if not query_terms or not doc_terms:
            return 0.0
        
        # Create term frequency vectors
        all_terms = list(set(query_terms + doc_terms))
        query_vector = [query_terms.count(term) for term in all_terms]
        doc_vector = [doc_terms.count(term) for term in all_terms]
        
        # Calculate cosine similarity
        dot_product = sum(q * d for q, d in zip(query_vector, doc_vector))
        query_norm = math.sqrt(sum(q * q for q in query_vector))
        doc_norm = math.sqrt(sum(d * d for d in doc_vector))
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return dot_product / (query_norm * doc_norm)
    
    def _calculate_normalized_edit_distance(self, query: str, document: str) -> float:
        """Calculate normalized edit distance"""
        # Simplified version - in production, use proper edit distance algorithm
        query_clean = query.lower().replace(' ', '')
        doc_start = document.lower()[:len(query_clean)]
        
        if not query_clean:
            return 1.0
        
        # Simple character overlap
        matches = sum(1 for q, d in zip(query_clean, doc_start) if q == d)
        return matches / len(query_clean)
    
    def _calculate_lcs_ratio(self, query: str, document: str) -> float:
        """Calculate longest common subsequence ratio"""
        # Simplified LCS
        query_words = query.lower().split()
        doc_words = document.lower().split()
        
        if not query_words:
            return 0.0
        
        common_words = sum(1 for word in query_words if word in doc_words)
        return common_words / len(query_words)
    
    def _count_exact_matches(self, query: str, document: str) -> float:
        """Count exact phrase matches"""
        return 1.0 if query.lower() in document.lower() else 0.0
    
    def _count_partial_matches(self, query: str, document: str) -> float:
        """Count partial matches"""
        query_words = query.lower().split()
        doc_lower = document.lower()
        
        matches = sum(1 for word in query_words if word in doc_lower)
        return matches / len(query_words) if query_words else 0.0
    
    def _count_phrase_matches(self, query: str, document: str) -> float:
        """Count phrase matches"""
        query_words = query.lower().split()
        if len(query_words) < 2:
            return 0.0
        
        doc_lower = document.lower()
        phrases = [' '.join(query_words[i:i+2]) for i in range(len(query_words)-1)]
        
        matches = sum(1 for phrase in phrases if phrase in doc_lower)
        return matches / len(phrases)
    
    def _calculate_query_coverage(self, query: str, document: str) -> float:
        """Calculate how much of the query is covered by the document"""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        if not query_words:
            return 0.0
        
        covered = len(query_words.intersection(doc_words))
        return covered / len(query_words)
    
    def _vector_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _vector_euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate normalized Euclidean distance"""
        distance = np.linalg.norm(vec1 - vec2)
        max_distance = np.linalg.norm(vec1) + np.linalg.norm(vec2)
        return 1.0 - (distance / max_distance) if max_distance > 0 else 0.0
    
    def _vector_manhattan_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate normalized Manhattan distance"""
        distance = np.sum(np.abs(vec1 - vec2))
        max_distance = np.sum(np.abs(vec1)) + np.sum(np.abs(vec2))
        return 1.0 - (distance / max_distance) if max_distance > 0 else 0.0
    
    def _calculate_document_uniqueness(self, document: str) -> float:
        """Calculate document uniqueness score"""
        words = document.lower().split()
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        return unique_words / len(words)
    
    def _calculate_readability_score(self, document: str) -> float:
        """Calculate readability score (simplified Flesch score)"""
        sentences = document.count('.') + document.count('!') + document.count('?')
        words = len(document.split())
        syllables = sum(max(1, len([c for c in word if c.lower() in 'aeiou'])) 
                       for word in document.split())
        
        if sentences == 0 or words == 0:
            return 0.5
        
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0.0, min(1.0, score / 100.0))
    
    def _count_spelling_errors(self, document: str) -> float:
        """Count spelling errors (simplified)"""
        # In production, use proper spell checker
        words = document.split()
        if not words:
            return 0.0
        
        # Simple heuristic: words with numbers or unusual characters
        errors = sum(1 for word in words if any(c.isdigit() for c in word) or len(word) > 20)
        return errors / len(words)
    
    def _calculate_grammar_score(self, document: str) -> float:
        """Calculate grammar score (simplified)"""
        # Simple heuristics for grammar quality
        sentences = document.count('.') + document.count('!') + document.count('?')
        words = len(document.split())
        
        if sentences == 0:
            return 0.5
        
        avg_sentence_length = words / sentences
        
        # Penalize very short or very long sentences
        if avg_sentence_length < 5 or avg_sentence_length > 30:
            return 0.3
        
        return 0.8  # Default good grammar score
    
    def _calculate_content_depth(self, document: str) -> float:
        """Calculate content depth score"""
        word_count = len(document.split())
        
        # Score based on content length
        if word_count < 100:
            return 0.2
        elif word_count < 300:
            return 0.5
        elif word_count < 1000:
            return 0.8
        else:
            return 1.0
    
    def _calculate_information_density(self, document: str) -> float:
        """Calculate information density"""
        words = document.lower().split()
        if not words:
            return 0.0
        
        # Remove common stop words (simplified)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        content_words = [word for word in words if word not in stop_words]
        
        return len(content_words) / len(words)
    
    def _calculate_temporal_relevance(self, query: str, document_timestamp: float) -> float:
        """Calculate temporal relevance"""
        # Check if query has temporal keywords
        temporal_keywords = ['recent', 'latest', 'new', 'current', 'today', 'now', 'this year']
        has_temporal = any(keyword in query.lower() for keyword in temporal_keywords)
        
        if not has_temporal:
            return 0.5  # Neutral relevance
        
        current_time = time.time()
        age_days = (current_time - document_timestamp) / (24 * 3600)
        
        # Exponential decay for temporal queries
        return math.exp(-age_days / 30)  # 30-day half-life
    
    def _calculate_seasonal_relevance(self, query: str, document_timestamp: float) -> float:
        """Calculate seasonal relevance"""
        # Simple seasonal detection
        seasonal_keywords = {
            'winter': [11, 12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'fall': [9, 10, 11],
            'christmas': [12],
            'easter': [3, 4]
        }
        
        current_month = time.localtime().tm_mon
        
        for season, months in seasonal_keywords.items():
            if season in query.lower() and current_month in months:
                return 1.0
        
        return 0.5  # Neutral seasonal relevance
    
    def _calculate_query_type_match(self, query: str, document: str) -> float:
        """Calculate query type matching"""
        query_lower = query.lower()
        
        # Question types
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            # Look for answers in document
            if any(word in document.lower() for word in ['answer', 'because', 'result', 'explain']):
                return 1.0
        
        # Definition queries
        if any(word in query_lower for word in ['define', 'definition', 'meaning', 'what is']):
            if any(word in document.lower() for word in ['definition', 'defined as', 'means', 'refers to']):
                return 1.0
        
        return 0.5  # Default neutral match
    
    def _calculate_intent_alignment(self, query: str, document: str) -> float:
        """Calculate intent alignment"""
        # Simplified intent detection
        informational_keywords = ['what', 'how', 'why', 'explain', 'guide', 'tutorial']
        navigational_keywords = ['login', 'homepage', 'official', 'website']
        transactional_keywords = ['buy', 'purchase', 'price', 'order', 'shop']
        
        query_lower = query.lower()
        doc_lower = document.lower()
        
        # Check intent alignment
        if any(kw in query_lower for kw in informational_keywords):
            if any(kw in doc_lower for kw in ['guide', 'tutorial', 'explanation', 'how-to']):
                return 1.0
        
        if any(kw in query_lower for kw in transactional_keywords):
            if any(kw in doc_lower for kw in ['price', 'buy', 'order', 'cart', 'checkout']):
                return 1.0
        
        return 0.5
    
    def _calculate_entity_overlap(self, query: str, document: str) -> float:
        """Calculate entity overlap (simplified)"""
        # Simple entity detection - proper names and numbers
        import re
        
        query_entities = set(re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', query))
        doc_entities = set(re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', document))
        
        if not query_entities:
            return 0.5
        
        overlap = len(query_entities.intersection(doc_entities))
        return overlap / len(query_entities)
    
    def _calculate_topic_match(self, query: str, document: str) -> float:
        """Calculate topic matching"""
        # Simplified topic detection using key terms
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_content = query_words - stop_words
        doc_content = doc_words - stop_words
        
        if not query_content:
            return 0.5
        
        overlap = len(query_content.intersection(doc_content))
        return overlap / len(query_content)
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity"""
        words = query.split()
        
        # Complexity factors
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        has_operators = any(op in query for op in ['"', '+', '-', 'AND', 'OR', 'NOT'])
        
        complexity = 0.0
        complexity += min(1.0, word_count / 10)  # More words = more complex
        complexity += min(1.0, avg_word_length / 10)  # Longer words = more complex
        complexity += 0.5 if has_operators else 0.0  # Operators = more complex
        
        return complexity / 2.5  # Normalize to 0-1
    
    def _calculate_query_specificity(self, query: str) -> float:
        """Calculate query specificity"""
        words = query.lower().split()
        
        # Specificity indicators
        specific_words = ['specific', 'exact', 'precise', 'detailed', 'particular']
        vague_words = ['general', 'overview', 'basic', 'simple', 'any']
        
        specificity = 0.5  # Start neutral
        
        for word in words:
            if word in specific_words:
                specificity += 0.2
            elif word in vague_words:
                specificity -= 0.2
        
        # Longer queries tend to be more specific
        specificity += min(0.3, len(words) / 20)
        
        return max(0.0, min(1.0, specificity))
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to prevent any single feature from dominating"""
        # Simple min-max normalization
        normalized = np.zeros_like(features)
        
        for i, value in enumerate(features):
            if np.isnan(value) or np.isinf(value):
                normalized[i] = 0.0
            else:
                # Update feature statistics
                self.feature_stats[i]['min'] = min(self.feature_stats[i]['min'], value)
                self.feature_stats[i]['max'] = max(self.feature_stats[i]['max'], value)
                
                # Normalize to 0-1 range
                min_val = self.feature_stats[i]['min']
                max_val = self.feature_stats[i]['max']
                
                if max_val > min_val:
                    normalized[i] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[i] = 0.5  # Default neutral value
        
        return normalized

class LearningToRankSystem:
    """Advanced Learning-to-Rank system with multiple algorithms"""
    
    def __init__(self, 
                 algorithm: str = "gradient_boosting",
                 enable_online_learning: bool = True,
                 max_training_examples: int = 10000):
        
        self.algorithm = algorithm
        self.enable_online_learning = enable_online_learning
        self.max_training_examples = max_training_examples
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Initialize models
        self.models = self._initialize_models()
        self.current_model = self.models[algorithm]
        
        # Training data storage
        self.training_data = deque(maxlen=max_training_examples)
        self.validation_data = []
        
        # Online learning buffer
        self.online_buffer = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_history = []
        self.model_version = 1
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize different ranking models"""
        return {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
    
    async def add_training_example(self, query: str, documents: List[Dict[str, Any]], 
                                  relevance_scores: List[float], query_vector: np.ndarray,
                                  user_context: Dict[str, Any] = None):
        """Add a training example to the system"""
        
        # Extract features for all documents
        feature_vectors = []
        for doc, relevance in zip(documents, relevance_scores):
            features = self.feature_extractor.extract_features(
                query, doc, query_vector, user_context
            )
            
            feature_vector = FeatureVector(
                query_id=hash(query),
                document_id=doc.get('id', ''),
                features=features,
                relevance_score=relevance,
                user_id=user_context.get('user_id') if user_context else None
            )
            
            feature_vectors.append(feature_vector)
        
        # Add to training data
        self.training_data.append(feature_vectors)
        
        # Add to online learning buffer if enabled
        if self.enable_online_learning:
            self.online_buffer.append(feature_vectors)
            
            # Trigger online update if buffer is full
            if len(self.online_buffer) >= 100:
                await self._perform_online_update()
    
    async def _perform_online_update(self):
        """Perform online model update"""
        try:
            # Prepare data from buffer
            X, y = self._prepare_training_data(list(self.online_buffer))
            
            if len(X) < 10:  # Need minimum data for update
                return
            
            # Partial fit for compatible models
            if hasattr(self.current_model, 'partial_fit'):
                self.current_model.partial_fit(X, y)
            else:
                # Retrain with recent data
                recent_data = list(self.training_data)[-1000:]  # Last 1000 examples
                X_recent, y_recent = self._prepare_training_data(recent_data)
                self.current_model.fit(X_recent, y_recent)
            
            # Clear buffer
            self.online_buffer.clear()
            
            logger.info("Online model update completed")
            
        except Exception as e:
            logger.error(f"Online update failed: {e}")
    
    def _prepare_training_data(self, feature_vectors_list: List[List[FeatureVector]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for scikit-learn models"""
        
        X = []
        y = []
        
        for query_vectors in feature_vectors_list:
            for fv in query_vectors:
                if fv.relevance_score is not None:
                    X.append(fv.features)
                    y.append(fv.relevance_score)
        
        return np.array(X), np.array(y)
    
    async def train_model(self, validation_split: float = 0.2):
        """Train the ranking model"""
        
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data")
            return
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data(list(self.training_data))
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
            
            # Train model
            self.current_model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_predictions = self.current_model.predict(X_val)
            mse = mean_squared_error(y_val, val_predictions)
            
            # Calculate NDCG if possible
            ndcg = 0.0
            try:
                # Group predictions by query for NDCG calculation
                ndcg = ndcg_score([y_val], [val_predictions])
            except:
                pass
            
            # Update performance history
            self.performance_history.append({
                'version': self.model_version,
                'mse': mse,
                'ndcg': ndcg,
                'training_size': len(X_train),
                'timestamp': time.time()
            })
            
            self.model_version += 1
            
            logger.info(f"Model trained - MSE: {mse:.4f}, NDCG: {ndcg:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    async def rank_documents(self, query: str, documents: List[Dict[str, Any]], 
                           query_vector: np.ndarray, user_context: Dict[str, Any] = None) -> List[Tuple[float, Dict[str, Any]]]:
        """Rank documents using the trained model"""
        
        if not hasattr(self.current_model, 'predict'):
            # Return random ranking if model not trained
            import random
            ranked = list(enumerate(documents))
            random.shuffle(ranked)
            return [(float(i), doc) for i, doc in ranked]
        
        try:
            # Extract features for all documents
            document_features = []
            for doc in documents:
                features = self.feature_extractor.extract_features(
                    query, doc, query_vector, user_context
                )
                document_features.append(features)
            
            # Predict relevance scores
            X = np.array(document_features)
            predictions = self.current_model.predict(X)
            
            # Combine predictions with documents
            ranked_results = [(float(score), doc) for score, doc in zip(predictions, documents)]
            
            # Sort by predicted relevance (descending)
            ranked_results.sort(key=lambda x: x[0], reverse=True)
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            # Return documents in original order
            return [(0.5, doc) for doc in documents]
    
    def update_from_feedback(self, query: str, document_id: str, 
                           feedback_type: str, feedback_value: float):
        """Update model based on user feedback"""
        
        feedback_example = {
            'query': query,
            'document_id': document_id,
            'feedback_type': feedback_type,
            'feedback_value': feedback_value,
            'timestamp': time.time()
        }
        
        # Add to online learning buffer
        if self.enable_online_learning:
            # Convert feedback to training signal
            # This would need more sophisticated handling in production
            pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        
        if not hasattr(self.current_model, 'feature_importances_'):
            return {}
        
        importance_scores = self.current_model.feature_importances_
        feature_names = self.feature_extractor.feature_names
        
        return dict(zip(feature_names, importance_scores))
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.current_model,
                'feature_extractor': self.feature_extractor,
                'model_version': self.model_version,
                'performance_history': self.performance_history,
                'algorithm': self.algorithm
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.current_model = model_data['model']
            self.feature_extractor = model_data['feature_extractor']
            self.model_version = model_data.get('model_version', 1)
            self.performance_history = model_data.get('performance_history', [])
            self.algorithm = model_data.get('algorithm', 'gradient_boosting')
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'algorithm': self.algorithm,
            'model_version': self.model_version,
            'training_examples': len(self.training_data),
            'online_buffer_size': len(self.online_buffer),
            'feature_count': len(self.feature_extractor.feature_names),
            'performance_history': self.performance_history[-5:],  # Last 5 performance records
            'online_learning_enabled': self.enable_online_learning
        }