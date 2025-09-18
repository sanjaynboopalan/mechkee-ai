"""
Advanced Hybrid Ranking Algorithm
Combines multiple ranking signals for superior search relevance
"""

import numpy as np
import math
import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timezone
import re
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class RankingSignals:
    """Container for all ranking signals"""
    
    # Text matching signals
    bm25_score: float = 0.0
    tf_idf_score: float = 0.0
    exact_match_bonus: float = 0.0
    phrase_match_bonus: float = 0.0
    
    # Vector similarity signals
    vector_similarity: float = 0.0
    semantic_coherence: float = 0.0
    
    # Content quality signals
    authority_score: float = 0.0
    freshness_score: float = 0.0
    engagement_score: float = 0.0
    click_through_rate: float = 0.0
    
    # Query-specific signals
    query_intent_match: float = 0.0
    entity_relevance: float = 0.0
    topic_alignment: float = 0.0
    
    # Document features
    content_length_score: float = 0.0
    readability_score: float = 0.0
    completeness_score: float = 0.0
    
    # Behavioral signals
    dwell_time_score: float = 0.0
    bounce_rate_penalty: float = 0.0
    social_signals: float = 0.0
    
    # Technical quality
    load_speed_score: float = 0.0
    mobile_friendly_score: float = 0.0
    accessibility_score: float = 0.0

@dataclass
class QueryContext:
    """Enhanced query context with intent and metadata"""
    query: str
    intent: str = "informational"  # informational, navigational, transactional
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    search_history: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    location: Optional[Dict[str, Any]] = None
    device_type: str = "desktop"  # desktop, mobile, tablet

class AdvancedRankingAlgorithm:
    """
    State-of-the-art ranking algorithm combining multiple signals
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 enable_neural_ranking: bool = True,
                 enable_personalization: bool = True):
        
        self.learning_rate = learning_rate
        self.enable_neural_ranking = enable_neural_ranking
        self.enable_personalization = enable_personalization
        
        # Feature weights (learned over time)
        self.feature_weights = {
            'bm25_score': 0.25,
            'vector_similarity': 0.20,
            'authority_score': 0.15,
            'freshness_score': 0.10,
            'engagement_score': 0.10,
            'exact_match_bonus': 0.08,
            'query_intent_match': 0.07,
            'semantic_coherence': 0.05
        }
        
        # Document statistics for BM25
        self.doc_stats = {
            'total_docs': 0,
            'avg_doc_length': 0.0,
            'doc_lengths': {},
            'term_doc_freq': defaultdict(int),
            'idf_cache': {}
        }
        
        # Query processing cache
        self.query_cache = {}
        
        # User interaction data
        self.user_interactions = defaultdict(list)
        self.global_click_data = defaultdict(lambda: {'clicks': 0, 'impressions': 0})
        
        # Neural ranking model (simplified)
        if enable_neural_ranking:
            self.neural_weights = self._initialize_neural_model()
        
        # Personalization data
        if enable_personalization:
            self.user_profiles = defaultdict(lambda: {
                'topic_preferences': defaultdict(float),
                'source_preferences': defaultdict(float),
                'query_patterns': defaultdict(int)
            })
    
    def _initialize_neural_model(self) -> Dict[str, np.ndarray]:
        """Initialize simple neural ranking model"""
        input_size = 20  # Number of features
        hidden_size = 32
        output_size = 1
        
        return {
            'W1': np.random.randn(input_size, hidden_size) * 0.01,
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * 0.01,
            'b2': np.zeros(output_size)
        }
    
    def update_document_stats(self, document_id: str, content: str, terms: List[str]):
        """Update document statistics for BM25 calculation"""
        doc_length = len(terms)
        
        # Update global stats
        self.doc_stats['total_docs'] += 1
        self.doc_stats['doc_lengths'][document_id] = doc_length
        
        # Update average document length
        total_length = sum(self.doc_stats['doc_lengths'].values())
        self.doc_stats['avg_doc_length'] = total_length / self.doc_stats['total_docs']
        
        # Update term document frequencies
        unique_terms = set(terms)
        for term in unique_terms:
            self.doc_stats['term_doc_freq'][term] += 1
        
        # Clear IDF cache as it's now outdated
        self.doc_stats['idf_cache'].clear()
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_terms: List[str], 
                             doc_id: str, k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 score with optimizations"""
        
        if doc_id not in self.doc_stats['doc_lengths']:
            return 0.0
        
        doc_length = self.doc_stats['doc_lengths'][doc_id]
        avg_doc_length = self.doc_stats['avg_doc_length']
        
        # Count term frequencies in document
        doc_term_freq = Counter(doc_terms)
        
        score = 0.0
        for term in query_terms:
            if term not in doc_term_freq:
                continue
            
            # Get or calculate IDF
            if term not in self.doc_stats['idf_cache']:
                df = self.doc_stats['term_doc_freq'][term]
                idf = math.log((self.doc_stats['total_docs'] - df + 0.5) / (df + 0.5))
                self.doc_stats['idf_cache'][term] = max(idf, 0.01)  # Prevent negative IDF
            
            idf = self.doc_stats['idf_cache'][term]
            tf = doc_term_freq[term]
            
            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def _calculate_vector_similarity(self, query_vector: np.ndarray, 
                                   doc_vector: np.ndarray) -> float:
        """Calculate cosine similarity with normalization"""
        
        # Handle zero vectors
        query_norm = np.linalg.norm(query_vector)
        doc_norm = np.linalg.norm(doc_vector)
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
        
        # Apply temperature scaling for better distribution
        temperature = 0.1
        scaled_similarity = math.tanh(similarity / temperature)
        
        return max(0.0, scaled_similarity)
    
    def _calculate_semantic_coherence(self, query_context: QueryContext, 
                                    document_content: str) -> float:
        """Calculate semantic coherence between query and document"""
        
        # Extract key concepts from query
        query_concepts = set(query_context.entities + query_context.topics)
        
        # Simple concept matching (in production, use NER and topic modeling)
        doc_words = set(re.findall(r'\b\w+\b', document_content.lower()))
        
        # Calculate concept overlap
        if not query_concepts:
            return 0.5  # Neutral score
        
        overlap = len(query_concepts.intersection(doc_words))
        coherence = overlap / len(query_concepts)
        
        return coherence
    
    def _calculate_freshness_score(self, document_timestamp: float, 
                                 query_context: QueryContext) -> float:
        """Calculate freshness score with query-dependent decay"""
        
        current_time = query_context.timestamp
        age_seconds = current_time - document_timestamp
        age_days = age_seconds / (24 * 3600)
        
        # Different decay rates based on query intent
        if query_context.intent == "navigational":
            # Navigation queries care less about freshness
            half_life_days = 365
        elif "news" in query_context.query.lower() or "recent" in query_context.query.lower():
            # News queries need very fresh content
            half_life_days = 1
        else:
            # General informational queries
            half_life_days = 30
        
        # Exponential decay
        decay_factor = math.exp(-age_days * math.log(2) / half_life_days)
        
        return decay_factor
    
    def _calculate_authority_score(self, document_metadata: Dict[str, Any]) -> float:
        """Calculate document authority based on various signals"""
        
        authority = 0.0
        
        # Domain authority (simplified)
        domain = document_metadata.get('domain', '')
        domain_scores = {
            'wikipedia.org': 0.9,
            'arxiv.org': 0.8,
            'github.com': 0.7,
            'stackoverflow.com': 0.8,
        }
        
        for trusted_domain, score in domain_scores.items():
            if trusted_domain in domain:
                authority = max(authority, score)
        
        # Author authority
        author = document_metadata.get('author', '')
        if author and len(author) > 0:
            authority += 0.1
        
        # Citation count (if available)
        citations = document_metadata.get('citation_count', 0)
        if citations > 0:
            authority += min(0.3, math.log(citations + 1) / 10)
        
        # Content quality indicators
        word_count = document_metadata.get('word_count', 0)
        if word_count > 500:  # Substantial content
            authority += 0.1
        
        return min(1.0, authority)
    
    def _calculate_engagement_score(self, document_id: str) -> float:
        """Calculate engagement score from user interactions"""
        
        click_data = self.global_click_data[document_id]
        
        if click_data['impressions'] == 0:
            return 0.5  # Neutral score for new documents
        
        # Click-through rate
        ctr = click_data['clicks'] / click_data['impressions']
        
        # Apply smoothing for low impression counts
        smoothed_ctr = (click_data['clicks'] + 1) / (click_data['impressions'] + 2)
        
        # Normalize to 0-1 range
        engagement = min(1.0, smoothed_ctr * 10)  # Assume 10% CTR is excellent
        
        return engagement
    
    def _extract_ranking_signals(self, query_context: QueryContext, 
                                document: Dict[str, Any], 
                                query_vector: np.ndarray) -> RankingSignals:
        """Extract all ranking signals for a document"""
        
        signals = RankingSignals()
        
        # Preprocess query and document
        query_terms = re.findall(r'\b\w+\b', query_context.query.lower())
        doc_content = document.get('content', '')
        doc_terms = re.findall(r'\b\w+\b', doc_content.lower())
        doc_id = document.get('id', '')
        
        # Text matching signals
        signals.bm25_score = self._calculate_bm25_score(query_terms, doc_terms, doc_id)
        
        # Exact match bonus
        if query_context.query.lower() in doc_content.lower():
            signals.exact_match_bonus = 1.0
        
        # Phrase match bonus
        query_phrases = [' '.join(query_terms[i:i+2]) for i in range(len(query_terms)-1)]
        phrase_matches = sum(1 for phrase in query_phrases if phrase in doc_content.lower())
        signals.phrase_match_bonus = min(1.0, phrase_matches / len(query_phrases)) if query_phrases else 0.0
        
        # Vector similarity
        doc_vector = document.get('vector', np.zeros_like(query_vector))
        signals.vector_similarity = self._calculate_vector_similarity(query_vector, doc_vector)
        
        # Semantic coherence
        signals.semantic_coherence = self._calculate_semantic_coherence(query_context, doc_content)
        
        # Document quality signals
        metadata = document.get('metadata', {})
        signals.authority_score = self._calculate_authority_score(metadata)
        signals.freshness_score = self._calculate_freshness_score(
            document.get('timestamp', time.time()), query_context
        )
        signals.engagement_score = self._calculate_engagement_score(doc_id)
        
        # Content quality
        word_count = len(doc_terms)
        signals.content_length_score = min(1.0, word_count / 1000)  # Optimal around 1000 words
        
        # Query intent matching
        intent_keywords = {
            'informational': ['what', 'how', 'why', 'explain', 'define'],
            'navigational': ['login', 'site', 'homepage', 'official'],
            'transactional': ['buy', 'purchase', 'order', 'price', 'cost']
        }
        
        query_lower = query_context.query.lower()
        intent_match = 0.0
        for keyword in intent_keywords.get(query_context.intent, []):
            if keyword in query_lower:
                intent_match += 0.2
        signals.query_intent_match = min(1.0, intent_match)
        
        return signals
    
    def _neural_ranking_score(self, signals: RankingSignals) -> float:
        """Apply neural ranking model to signals"""
        
        if not self.enable_neural_ranking:
            return 0.0
        
        # Convert signals to feature vector
        features = np.array([
            signals.bm25_score,
            signals.vector_similarity,
            signals.authority_score,
            signals.freshness_score,
            signals.engagement_score,
            signals.exact_match_bonus,
            signals.phrase_match_bonus,
            signals.semantic_coherence,
            signals.query_intent_match,
            signals.content_length_score,
            signals.tf_idf_score,
            signals.entity_relevance,
            signals.topic_alignment,
            signals.readability_score,
            signals.completeness_score,
            signals.dwell_time_score,
            signals.bounce_rate_penalty,
            signals.social_signals,
            signals.load_speed_score,
            signals.click_through_rate
        ])
        
        # Forward pass through neural network
        hidden = np.maximum(0, np.dot(features, self.neural_weights['W1']) + self.neural_weights['b1'])  # ReLU
        output = np.dot(hidden, self.neural_weights['W2']) + self.neural_weights['b2']
        
        # Apply sigmoid for 0-1 output
        neural_score = 1.0 / (1.0 + np.exp(-output[0]))
        
        return neural_score
    
    def _personalization_boost(self, query_context: QueryContext, 
                              document: Dict[str, Any]) -> float:
        """Calculate personalization boost based on user history"""
        
        if not self.enable_personalization or not query_context.user_id:
            return 1.0  # No boost
        
        user_profile = self.user_profiles[query_context.user_id]
        boost = 1.0
        
        # Topic preference boost
        doc_topics = document.get('metadata', {}).get('topics', [])
        for topic in doc_topics:
            if topic in user_profile['topic_preferences']:
                boost += user_profile['topic_preferences'][topic] * 0.1
        
        # Source preference boost
        doc_domain = document.get('metadata', {}).get('domain', '')
        if doc_domain in user_profile['source_preferences']:
            boost += user_profile['source_preferences'][doc_domain] * 0.05
        
        return min(1.5, boost)  # Cap the boost
    
    def rank_documents(self, query_context: QueryContext, 
                      documents: List[Dict[str, Any]], 
                      query_vector: np.ndarray) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Rank documents using advanced hybrid algorithm
        """
        
        ranked_results = []
        
        for document in documents:
            # Extract all ranking signals
            signals = self._extract_ranking_signals(query_context, document, query_vector)
            
            # Calculate base relevance score using weighted features
            base_score = 0.0
            for feature, weight in self.feature_weights.items():
                if hasattr(signals, feature):
                    base_score += weight * getattr(signals, feature)
            
            # Apply neural ranking if enabled
            if self.enable_neural_ranking:
                neural_score = self._neural_ranking_score(signals)
                base_score = 0.7 * base_score + 0.3 * neural_score
            
            # Apply personalization boost
            personalization_boost = self._personalization_boost(query_context, document)
            final_score = base_score * personalization_boost
            
            # Apply query-specific adjustments
            final_score = self._apply_query_adjustments(query_context, final_score, signals)
            
            ranked_results.append((final_score, document, signals))
        
        # Sort by score (descending)
        ranked_results.sort(key=lambda x: x[0], reverse=True)
        
        # Return documents with scores
        return [(score, doc) for score, doc, _ in ranked_results]
    
    def _apply_query_adjustments(self, query_context: QueryContext, 
                               base_score: float, signals: RankingSignals) -> float:
        """Apply query-specific score adjustments"""
        
        adjusted_score = base_score
        
        # Boost for query length
        query_length = len(query_context.query.split())
        if query_length > 5:  # Long queries benefit from exact matches
            adjusted_score += signals.exact_match_bonus * 0.1
        
        # Intent-specific adjustments
        if query_context.intent == "navigational":
            # Boost authority for navigational queries
            adjusted_score += signals.authority_score * 0.2
        elif query_context.intent == "informational":
            # Boost semantic coherence for informational queries
            adjusted_score += signals.semantic_coherence * 0.15
        
        # Time-sensitive query detection
        time_keywords = ['recent', 'latest', 'new', 'current', 'today', 'now']
        if any(keyword in query_context.query.lower() for keyword in time_keywords):
            adjusted_score += signals.freshness_score * 0.3
        
        return adjusted_score
    
    def update_from_interaction(self, query_context: QueryContext, 
                               document_id: str, interaction_type: str, 
                               interaction_value: float = 1.0):
        """Update ranking model based on user interactions"""
        
        # Update click data
        if interaction_type == "click":
            self.global_click_data[document_id]['clicks'] += 1
        
        if interaction_type in ["click", "impression"]:
            self.global_click_data[document_id]['impressions'] += 1
        
        # Update user profile if personalization is enabled
        if self.enable_personalization and query_context.user_id:
            user_profile = self.user_profiles[query_context.user_id]
            
            # Update query patterns
            query_lower = query_context.query.lower()
            user_profile['query_patterns'][query_lower] += 1
            
            # Record interaction
            self.user_interactions[query_context.user_id].append({
                'query': query_context.query,
                'document_id': document_id,
                'interaction_type': interaction_type,
                'value': interaction_value,
                'timestamp': time.time()
            })
    
    def learn_from_feedback(self, query_context: QueryContext, 
                          ranked_results: List[Tuple[float, Dict[str, Any]]], 
                          feedback: List[float]):
        """Learn from explicit relevance feedback"""
        
        if not self.enable_neural_ranking or len(feedback) != len(ranked_results):
            return
        
        # Extract features for training
        training_data = []
        for (score, document), relevance in zip(ranked_results, feedback):
            signals = self._extract_ranking_signals(
                query_context, document, np.zeros(1536)  # Placeholder vector
            )
            
            features = np.array([
                signals.bm25_score, signals.vector_similarity, signals.authority_score,
                signals.freshness_score, signals.engagement_score, signals.exact_match_bonus,
                signals.phrase_match_bonus, signals.semantic_coherence, signals.query_intent_match,
                signals.content_length_score, signals.tf_idf_score, signals.entity_relevance,
                signals.topic_alignment, signals.readability_score, signals.completeness_score,
                signals.dwell_time_score, signals.bounce_rate_penalty, signals.social_signals,
                signals.load_speed_score, signals.click_through_rate
            ])
            
            training_data.append((features, relevance))
        
        # Simple gradient descent update (in production, use more sophisticated methods)
        self._update_neural_weights(training_data)
    
    def _update_neural_weights(self, training_data: List[Tuple[np.ndarray, float]]):
        """Update neural network weights using gradient descent"""
        
        for features, target in training_data:
            # Forward pass
            hidden = np.maximum(0, np.dot(features, self.neural_weights['W1']) + self.neural_weights['b1'])
            output = np.dot(hidden, self.neural_weights['W2']) + self.neural_weights['b2']
            prediction = 1.0 / (1.0 + np.exp(-output[0]))
            
            # Backward pass (simplified)
            error = target - prediction
            
            # Update output layer
            d_output = error * prediction * (1 - prediction)
            self.neural_weights['W2'] += self.learning_rate * np.outer(hidden, d_output)
            self.neural_weights['b2'] += self.learning_rate * d_output
            
            # Update hidden layer
            d_hidden = d_output * self.neural_weights['W2'].flatten()
            d_hidden[hidden <= 0] = 0  # ReLU derivative
            
            self.neural_weights['W1'] += self.learning_rate * np.outer(features, d_hidden)
            self.neural_weights['b1'] += self.learning_rate * d_hidden
    
    def get_ranking_explanation(self, query_context: QueryContext, 
                               document: Dict[str, Any], 
                               query_vector: np.ndarray) -> Dict[str, Any]:
        """Provide explanation for why a document was ranked as it was"""
        
        signals = self._extract_ranking_signals(query_context, document, query_vector)
        
        # Calculate contribution of each signal
        contributions = {}
        total_score = 0.0
        
        for feature, weight in self.feature_weights.items():
            if hasattr(signals, feature):
                signal_value = getattr(signals, feature)
                contribution = weight * signal_value
                contributions[feature] = {
                    'value': signal_value,
                    'weight': weight,
                    'contribution': contribution
                }
                total_score += contribution
        
        # Add neural ranking if enabled
        if self.enable_neural_ranking:
            neural_score = self._neural_ranking_score(signals)
            contributions['neural_ranking'] = {
                'value': neural_score,
                'weight': 0.3,
                'contribution': 0.3 * neural_score
            }
        
        return {
            'total_score': total_score,
            'signal_contributions': contributions,
            'top_signals': sorted(contributions.items(), 
                                key=lambda x: x[1]['contribution'], reverse=True)[:5]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ranking algorithm statistics"""
        
        return {
            'total_documents': self.doc_stats['total_docs'],
            'avg_document_length': self.doc_stats['avg_doc_length'],
            'unique_terms': len(self.doc_stats['term_doc_freq']),
            'idf_cache_size': len(self.doc_stats['idf_cache']),
            'total_users': len(self.user_profiles) if self.enable_personalization else 0,
            'feature_weights': self.feature_weights.copy(),
            'neural_ranking_enabled': self.enable_neural_ranking,
            'personalization_enabled': self.enable_personalization
        }