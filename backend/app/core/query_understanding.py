"""
Advanced Query Understanding System
Implements query expansion, intent classification, entity recognition, and semantic rewriting
"""

import re
import json
import time
import asyncio
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    INFORMATIONAL = "informational"  # What is X? How to do Y?
    NAVIGATIONAL = "navigational"   # Find specific website/page
    TRANSACTIONAL = "transactional" # Buy, download, sign up
    COMPARISON = "comparison"        # X vs Y, compare A and B
    DEFINITION = "definition"        # Define X, meaning of Y
    PROCEDURAL = "procedural"        # How to, step by step
    FACTUAL = "factual"             # When, where, who, specific facts
    CREATIVE = "creative"           # Generate, create, design
    ANALYTICAL = "analytical"       # Analyze, evaluate, explain

class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    PERCENTAGE = "percentage"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    EVENT = "event"
    LANGUAGE = "language"

@dataclass
class Entity:
    """Represents a named entity in the query"""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    canonical_form: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)
    context: Optional[str] = None

@dataclass
class QueryExpansion:
    """Query expansion suggestions"""
    synonyms: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    hypernyms: List[str] = field(default_factory=list)  # More general terms
    hyponyms: List[str] = field(default_factory=list)   # More specific terms
    context_terms: List[str] = field(default_factory=list)
    trending_terms: List[str] = field(default_factory=list)

@dataclass
class ProcessedQuery:
    """Processed and enhanced query"""
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    confidence: float
    entities: List[Entity]
    keywords: List[str]
    semantic_concepts: List[str]
    expansion: QueryExpansion
    rewritten_queries: List[str]
    complexity_score: float
    ambiguity_score: float
    timestamp: float = field(default_factory=time.time)

class IntentClassifier:
    """Advanced intent classification system"""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.INFORMATIONAL: [
                r'\b(what|explain|describe|information|about|overview)\b',
                r'\b(tell me|help me understand|I want to know)\b',
                r'\b(overview|summary|introduction)\b'
            ],
            QueryIntent.NAVIGATIONAL: [
                r'\b(website|homepage|official site|login|portal)\b',
                r'\b(go to|navigate to|find the page)\b',
                r'\b(site:|url:|domain:)\b'
            ],
            QueryIntent.TRANSACTIONAL: [
                r'\b(buy|purchase|order|get|download|install)\b',
                r'\b(price|cost|shop|store|marketplace)\b',
                r'\b(subscribe|sign up|register|join)\b'
            ],
            QueryIntent.COMPARISON: [
                r'\b(vs|versus|compare|comparison|difference)\b',
                r'\b(better|best|worse|pros and cons)\b',
                r'\b(alternative|similar|like)\b'
            ],
            QueryIntent.DEFINITION: [
                r'\b(define|definition|meaning|what is|what are)\b',
                r'\b(means|refers to|is defined as)\b',
                r'\b(terminology|glossary)\b'
            ],
            QueryIntent.PROCEDURAL: [
                r'\b(how to|how do|step by step|tutorial|guide)\b',
                r'\b(instructions|process|procedure|method)\b',
                r'\b(learn|teach me|show me)\b'
            ],
            QueryIntent.FACTUAL: [
                r'\b(when|where|who|which|whose)\b',
                r'\b(date|time|location|person|place)\b',
                r'\b(fact|statistic|data|number)\b'
            ],
            QueryIntent.CREATIVE: [
                r'\b(create|generate|make|design|build)\b',
                r'\b(ideas|suggestions|brainstorm)\b',
                r'\b(write|compose|draft)\b'
            ],
            QueryIntent.ANALYTICAL: [
                r'\b(analyze|analysis|evaluate|assessment)\b',
                r'\b(why|reason|cause|effect|impact)\b',
                r'\b(pros|cons|advantages|disadvantages)\b'
            ]
        }
        
        # Intent confidence weights
        self.intent_weights = {
            QueryIntent.INFORMATIONAL: 0.9,
            QueryIntent.NAVIGATIONAL: 0.8,
            QueryIntent.TRANSACTIONAL: 0.85,
            QueryIntent.COMPARISON: 0.8,
            QueryIntent.DEFINITION: 0.9,
            QueryIntent.PROCEDURAL: 0.85,
            QueryIntent.FACTUAL: 0.8,
            QueryIntent.CREATIVE: 0.75,
            QueryIntent.ANALYTICAL: 0.8
        }
    
    def classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify query intent with confidence score"""
        
        query_lower = query.lower()
        intent_scores = {}
        
        # Pattern-based classification
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches * self.intent_weights[intent]
            intent_scores[intent] = score
        
        # Additional heuristics
        intent_scores = self._apply_intent_heuristics(query_lower, intent_scores)
        
        if not intent_scores or max(intent_scores.values()) == 0:
            return QueryIntent.INFORMATIONAL, 0.5  # Default intent
        
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        
        # Normalize confidence
        total_score = sum(intent_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5
        
        return best_intent, min(1.0, confidence)
    
    def _apply_intent_heuristics(self, query: str, intent_scores: Dict[QueryIntent, float]) -> Dict[QueryIntent, float]:
        """Apply additional heuristics for intent classification"""
        
        # Question word heuristics
        if query.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            if query.startswith('what is') or query.startswith('what are'):
                intent_scores[QueryIntent.DEFINITION] += 0.5
            elif query.startswith('how to') or query.startswith('how do'):
                intent_scores[QueryIntent.PROCEDURAL] += 0.5
            elif query.startswith(('when', 'where', 'who')):
                intent_scores[QueryIntent.FACTUAL] += 0.5
            else:
                intent_scores[QueryIntent.INFORMATIONAL] += 0.3
        
        # Comparison indicators
        if ' vs ' in query or ' versus ' in query or ' or ' in query:
            intent_scores[QueryIntent.COMPARISON] += 0.6
        
        # Transactional indicators
        if any(word in query for word in ['$', 'price', 'buy', 'purchase', 'order']):
            intent_scores[QueryIntent.TRANSACTIONAL] += 0.4
        
        # Navigation indicators
        if any(word in query for word in ['.com', '.org', '.net', 'website', 'site']):
            intent_scores[QueryIntent.NAVIGATIONAL] += 0.4
        
        return intent_scores

class EntityRecognizer:
    """Advanced named entity recognition system"""
    
    def __init__(self):
        self.entity_patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
                r'\b(Mr|Mrs|Dr|Prof|CEO|President)\s+[A-Z][a-z]+\b'
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]+\s+(Inc|Corp|Ltd|LLC|Company|Corporation)\b',
                r'\b(Google|Microsoft|Apple|Amazon|Facebook|Tesla|Netflix)\b'
            ],
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # City, State
                r'\b(United States|New York|California|London|Paris|Tokyo)\b'
            ],
            EntityType.DATE: [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{4}\b(?=\s|$)'  # Year
            ],
            EntityType.TIME: [
                r'\b\d{1,2}:\d{2}\s*(AM|PM|am|pm)?\b',
                r'\b(morning|afternoon|evening|night|midnight|noon)\b'
            ],
            EntityType.MONEY: [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
                r'\b\d+\s*(dollars?|cents?|USD|EUR|GBP)\b'
            ],
            EntityType.PERCENTAGE: [
                r'\b\d+(?:\.\d+)?%',
                r'\b\d+(?:\.\d+)?\s*percent\b'
            ],
            EntityType.TECHNOLOGY: [
                r'\b(Python|JavaScript|Java|C\+\+|React|Angular|Vue|Node\.js|Docker|Kubernetes)\b',
                r'\b(AI|ML|machine learning|artificial intelligence|deep learning)\b'
            ],
            EntityType.LANGUAGE: [
                r'\b(English|Spanish|French|German|Chinese|Japanese|Korean|Russian|Arabic)\b'
            ]
        }
        
        # Common entity dictionaries (simplified)
        self.entity_dictionaries = {
            EntityType.PERSON: {
                'elon musk', 'bill gates', 'steve jobs', 'mark zuckerberg', 'jeff bezos'
            },
            EntityType.ORGANIZATION: {
                'google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta', 'tesla', 'openai'
            },
            EntityType.LOCATION: {
                'new york', 'london', 'paris', 'tokyo', 'sydney', 'toronto', 'berlin'
            },
            EntityType.TECHNOLOGY: {
                'artificial intelligence', 'machine learning', 'blockchain', 'quantum computing',
                'cloud computing', 'internet of things', 'virtual reality', 'augmented reality'
            }
        }
    
    def recognize_entities(self, query: str) -> List[Entity]:
        """Recognize entities in the query"""
        
        entities = []
        
        # Pattern-based recognition
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, query, re.IGNORECASE):
                    entity = Entity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8
                    )
                    entities.append(entity)
        
        # Dictionary-based recognition
        query_lower = query.lower()
        for entity_type, dictionary in self.entity_dictionaries.items():
            for term in dictionary:
                if term in query_lower:
                    start_pos = query_lower.find(term)
                    entity = Entity(
                        text=query[start_pos:start_pos + len(term)],
                        entity_type=entity_type,
                        start_pos=start_pos,
                        end_pos=start_pos + len(term),
                        confidence=0.9,
                        canonical_form=term
                    )
                    entities.append(entity)
        
        # Remove overlapping entities (keep highest confidence)
        entities = self._remove_overlapping_entities(entities)
        
        # Add context and related entities
        entities = self._enhance_entities(entities, query)
        
        return entities
    
    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping the most confident ones"""
        
        # Sort by start position
        entities.sort(key=lambda e: e.start_pos)
        
        filtered_entities = []
        for entity in entities:
            # Check for overlaps with existing entities
            overlaps = False
            for existing in filtered_entities:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Overlap detected
                    if entity.confidence <= existing.confidence:
                        overlaps = True
                        break
                    else:
                        # Remove the existing entity
                        filtered_entities.remove(existing)
                        break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _enhance_entities(self, entities: List[Entity], query: str) -> List[Entity]:
        """Enhance entities with context and related information"""
        
        for entity in entities:
            # Add context (words around the entity)
            words = query.split()
            entity_words = entity.text.split()
            
            try:
                # Find entity position in words
                for i in range(len(words) - len(entity_words) + 1):
                    if ' '.join(words[i:i+len(entity_words)]).lower() == entity.text.lower():
                        # Get context words
                        context_start = max(0, i - 2)
                        context_end = min(len(words), i + len(entity_words) + 2)
                        context_words = words[context_start:context_end]
                        entity.context = ' '.join(context_words)
                        break
            except:
                pass
            
            # Add related entities based on type
            entity.related_entities = self._find_related_entities(entity)
        
        return entities
    
    def _find_related_entities(self, entity: Entity) -> List[str]:
        """Find entities related to the given entity"""
        
        related = []
        
        if entity.entity_type == EntityType.PERSON:
            # For persons, add common associations
            person_lower = entity.text.lower()
            if 'elon musk' in person_lower:
                related = ['Tesla', 'SpaceX', 'Twitter', 'X']
            elif 'bill gates' in person_lower:
                related = ['Microsoft', 'Gates Foundation']
            elif 'steve jobs' in person_lower:
                related = ['Apple', 'iPhone', 'iPad']
        
        elif entity.entity_type == EntityType.ORGANIZATION:
            org_lower = entity.text.lower()
            if 'google' in org_lower:
                related = ['Alphabet', 'Android', 'Chrome', 'Gmail']
            elif 'microsoft' in org_lower:
                related = ['Windows', 'Office', 'Azure', 'Xbox']
            elif 'apple' in org_lower:
                related = ['iPhone', 'iPad', 'Mac', 'iOS']
        
        elif entity.entity_type == EntityType.TECHNOLOGY:
            tech_lower = entity.text.lower()
            if 'python' in tech_lower:
                related = ['programming', 'Django', 'Flask', 'NumPy']
            elif 'javascript' in tech_lower:
                related = ['React', 'Node.js', 'Angular', 'Vue']
        
        return related

class QueryExpander:
    """Advanced query expansion system"""
    
    def __init__(self):
        # Synonym dictionary (simplified)
        self.synonyms = {
            'fast': ['quick', 'rapid', 'speedy', 'swift'],
            'big': ['large', 'huge', 'massive', 'enormous'],
            'small': ['tiny', 'little', 'compact', 'miniature'],
            'good': ['excellent', 'great', 'wonderful', 'amazing'],
            'bad': ['poor', 'terrible', 'awful', 'horrible'],
            'help': ['assist', 'support', 'aid', 'guide'],
            'learn': ['study', 'understand', 'master', 'grasp'],
            'create': ['make', 'build', 'develop', 'generate'],
            'find': ['search', 'locate', 'discover', 'identify']
        }
        
        # Domain-specific expansions
        self.domain_expansions = {
            'programming': ['coding', 'development', 'software engineering', 'computer science'],
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
            'business': ['commerce', 'trade', 'enterprise', 'corporate'],
            'science': ['research', 'study', 'investigation', 'analysis'],
            'technology': ['tech', 'digital', 'innovation', 'advancement']
        }
        
        # Trending terms (would be updated from real data)
        self.trending_terms = {
            'ai': ['chatgpt', 'generative ai', 'llm', 'transformer'],
            'crypto': ['bitcoin', 'ethereum', 'blockchain', 'defi'],
            'climate': ['sustainability', 'carbon neutral', 'renewable energy']
        }
    
    def expand_query(self, query: str, entities: List[Entity], 
                    intent: QueryIntent) -> QueryExpansion:
        """Generate query expansions"""
        
        expansion = QueryExpansion()
        query_words = query.lower().split()
        
        # Synonym expansion
        for word in query_words:
            if word in self.synonyms:
                expansion.synonyms.extend(self.synonyms[word])
        
        # Domain-specific expansion
        for domain, terms in self.domain_expansions.items():
            if any(term in query.lower() for term in terms):
                expansion.related_terms.extend(terms)
        
        # Entity-based expansion
        for entity in entities:
            expansion.related_terms.extend(entity.related_entities)
            
            # Add canonical forms
            if entity.canonical_form:
                expansion.related_terms.append(entity.canonical_form)
        
        # Intent-based expansion
        expansion = self._expand_by_intent(expansion, intent, query)
        
        # Trending terms expansion
        for trend_key, trend_terms in self.trending_terms.items():
            if trend_key in query.lower():
                expansion.trending_terms.extend(trend_terms)
        
        # Hierarchical expansion (hypernyms/hyponyms)
        expansion = self._add_hierarchical_terms(expansion, query_words)
        
        # Remove duplicates and original terms
        expansion = self._clean_expansion(expansion, query_words)
        
        return expansion
    
    def _expand_by_intent(self, expansion: QueryExpansion, 
                         intent: QueryIntent, query: str) -> QueryExpansion:
        """Expand based on query intent"""
        
        if intent == QueryIntent.PROCEDURAL:
            expansion.context_terms.extend(['tutorial', 'guide', 'step-by-step', 'instructions'])
        
        elif intent == QueryIntent.COMPARISON:
            expansion.context_terms.extend(['vs', 'versus', 'compare', 'difference', 'pros cons'])
        
        elif intent == QueryIntent.DEFINITION:
            expansion.context_terms.extend(['meaning', 'definition', 'explanation', 'what is'])
        
        elif intent == QueryIntent.TRANSACTIONAL:
            expansion.context_terms.extend(['buy', 'purchase', 'price', 'cost', 'order'])
        
        elif intent == QueryIntent.ANALYTICAL:
            expansion.context_terms.extend(['analysis', 'evaluate', 'assessment', 'review'])
        
        return expansion
    
    def _add_hierarchical_terms(self, expansion: QueryExpansion, 
                               query_words: List[str]) -> QueryExpansion:
        """Add hierarchical terms (hypernyms and hyponyms)"""
        
        # Simple hierarchical relationships
        hierarchies = {
            'python': {
                'hypernyms': ['programming language', 'language', 'technology'],
                'hyponyms': ['django', 'flask', 'pandas', 'numpy']
            },
            'car': {
                'hypernyms': ['vehicle', 'transportation'],
                'hyponyms': ['sedan', 'suv', 'truck', 'coupe']
            },
            'dog': {
                'hypernyms': ['animal', 'pet', 'mammal'],
                'hyponyms': ['labrador', 'poodle', 'german shepherd']
            }
        }
        
        for word in query_words:
            if word in hierarchies:
                expansion.hypernyms.extend(hierarchies[word]['hypernyms'])
                expansion.hyponyms.extend(hierarchies[word]['hyponyms'])
        
        return expansion
    
    def _clean_expansion(self, expansion: QueryExpansion, 
                        original_words: List[str]) -> QueryExpansion:
        """Clean and deduplicate expansion terms"""
        
        original_set = set(original_words)
        
        # Remove duplicates and original terms
        expansion.synonyms = list(set(expansion.synonyms) - original_set)
        expansion.related_terms = list(set(expansion.related_terms) - original_set)
        expansion.hypernyms = list(set(expansion.hypernyms) - original_set)
        expansion.hyponyms = list(set(expansion.hyponyms) - original_set)
        expansion.context_terms = list(set(expansion.context_terms) - original_set)
        expansion.trending_terms = list(set(expansion.trending_terms) - original_set)
        
        # Limit number of terms per category
        expansion.synonyms = expansion.synonyms[:10]
        expansion.related_terms = expansion.related_terms[:15]
        expansion.hypernyms = expansion.hypernyms[:8]
        expansion.hyponyms = expansion.hyponyms[:12]
        expansion.context_terms = expansion.context_terms[:10]
        expansion.trending_terms = expansion.trending_terms[:8]
        
        return expansion

class QueryRewriter:
    """Advanced query rewriting system"""
    
    def __init__(self):
        self.rewriting_rules = {
            # Question normalization
            r'^what is (.+)\?*$': r'\1 definition',
            r'^how to (.+)\?*$': r'\1 tutorial guide',
            r'^why (.+)\?*$': r'\1 reason explanation',
            r'^when (.+)\?*$': r'\1 date time',
            r'^where (.+)\?*$': r'\1 location place',
            
            # Intent clarification
            r'(.+) vs (.+)': r'compare \1 \2 difference',
            r'(.+) or (.+)': r'\1 \2 comparison alternative',
            r'best (.+)': r'\1 top rated recommended',
            r'cheap (.+)': r'\1 affordable budget low cost',
            r'free (.+)': r'\1 no cost open source',
            
            # Technical query enhancement
            r'(.+) tutorial': r'how to learn \1 step by step guide',
            r'(.+) example': r'\1 sample code demonstration',
            r'(.+) error': r'\1 problem solution fix troubleshoot',
        }
    
    def rewrite_query(self, query: str, entities: List[Entity], 
                     intent: QueryIntent, expansion: QueryExpansion) -> List[str]:
        """Generate rewritten versions of the query"""
        
        rewritten_queries = []
        
        # Rule-based rewriting
        for pattern, replacement in self.rewriting_rules.items():
            match = re.match(pattern, query, re.IGNORECASE)
            if match:
                rewritten = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                rewritten_queries.append(rewritten)
        
        # Entity-enhanced rewriting
        for entity in entities:
            if entity.canonical_form and entity.canonical_form != entity.text.lower():
                entity_enhanced = query.replace(entity.text, entity.canonical_form)
                rewritten_queries.append(entity_enhanced)
        
        # Synonym-based rewriting
        query_words = query.split()
        if expansion.synonyms:
            for i, word in enumerate(query_words):
                if word.lower() in [syn.lower() for syn in expansion.synonyms]:
                    # Replace with synonym
                    for syn in expansion.synonyms:
                        if syn.lower() != word.lower():
                            synonym_query = query_words.copy()
                            synonym_query[i] = syn
                            rewritten_queries.append(' '.join(synonym_query))
                            break
        
        # Intent-specific rewriting
        rewritten_queries.extend(self._rewrite_by_intent(query, intent, expansion))
        
        # Add expansion terms
        if expansion.related_terms:
            enhanced_query = query + ' ' + ' '.join(expansion.related_terms[:3])
            rewritten_queries.append(enhanced_query)
        
        # Remove duplicates and original query
        rewritten_queries = list(set(rewritten_queries))
        if query in rewritten_queries:
            rewritten_queries.remove(query)
        
        # Limit number of rewritten queries
        return rewritten_queries[:10]
    
    def _rewrite_by_intent(self, query: str, intent: QueryIntent, 
                          expansion: QueryExpansion) -> List[str]:
        """Rewrite based on detected intent"""
        
        rewritten = []
        
        if intent == QueryIntent.PROCEDURAL:
            rewritten.append(f"how to {query}")
            rewritten.append(f"{query} tutorial")
            rewritten.append(f"{query} step by step")
        
        elif intent == QueryIntent.DEFINITION:
            rewritten.append(f"{query} meaning")
            rewritten.append(f"define {query}")
            rewritten.append(f"{query} explanation")
        
        elif intent == QueryIntent.COMPARISON:
            if ' vs ' not in query.lower():
                words = query.split()
                if len(words) >= 2:
                    rewritten.append(f"{words[0]} vs {words[1]}")
        
        elif intent == QueryIntent.TRANSACTIONAL:
            rewritten.append(f"buy {query}")
            rewritten.append(f"{query} price")
            rewritten.append(f"{query} purchase")
        
        return rewritten

class QueryUnderstandingSystem:
    """Main query understanding system"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_recognizer = EntityRecognizer()
        self.query_expander = QueryExpander()
        self.query_rewriter = QueryRewriter()
        
        # Query processing cache
        self.cache = {}
        self.cache_size_limit = 1000
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0,
            'intent_distribution': defaultdict(int),
            'entity_distribution': defaultdict(int)
        }
    
    async def process_query(self, query: str, use_cache: bool = True) -> ProcessedQuery:
        """Process and understand a query"""
        
        start_time = time.time()
        
        # Check cache
        if use_cache and query in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[query]
        
        try:
            # Clean the query
            cleaned_query = self._clean_query(query)
            
            # Classify intent
            intent, intent_confidence = self.intent_classifier.classify_intent(cleaned_query)
            
            # Recognize entities
            entities = self.entity_recognizer.recognize_entities(cleaned_query)
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_query, entities)
            
            # Extract semantic concepts
            semantic_concepts = self._extract_semantic_concepts(cleaned_query, entities, intent)
            
            # Expand query
            expansion = self.query_expander.expand_query(cleaned_query, entities, intent)
            
            # Rewrite query
            rewritten_queries = self.query_rewriter.rewrite_query(
                cleaned_query, entities, intent, expansion
            )
            
            # Calculate complexity and ambiguity scores
            complexity_score = self._calculate_complexity(cleaned_query, entities)
            ambiguity_score = self._calculate_ambiguity(cleaned_query, entities, intent)
            
            # Create processed query object
            processed_query = ProcessedQuery(
                original_query=query,
                cleaned_query=cleaned_query,
                intent=intent,
                confidence=intent_confidence,
                entities=entities,
                keywords=keywords,
                semantic_concepts=semantic_concepts,
                expansion=expansion,
                rewritten_queries=rewritten_queries,
                complexity_score=complexity_score,
                ambiguity_score=ambiguity_score
            )
            
            # Update statistics
            self.stats['total_queries'] += 1
            self.stats['intent_distribution'][intent.value] += 1
            
            for entity in entities:
                self.stats['entity_distribution'][entity.entity_type.value] += 1
            
            processing_time = time.time() - start_time
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['total_queries'] - 1) + 
                 processing_time) / self.stats['total_queries']
            )
            
            # Cache the result
            if use_cache:
                self._add_to_cache(query, processed_query)
            
            return processed_query
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Return basic processed query on error
            return ProcessedQuery(
                original_query=query,
                cleaned_query=query,
                intent=QueryIntent.INFORMATIONAL,
                confidence=0.5,
                entities=[],
                keywords=query.split(),
                semantic_concepts=[],
                expansion=QueryExpansion(),
                rewritten_queries=[],
                complexity_score=0.5,
                ambiguity_score=0.5
            )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove unnecessary punctuation (keep some for intent classification)
        cleaned = re.sub(r'[^\w\s\?\!\-\.]', ' ', cleaned)
        
        # Normalize case (keep first letter for proper nouns)
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned.lower()
        
        return cleaned
    
    def _extract_keywords(self, query: str, entities: List[Entity]) -> List[str]:
        """Extract keywords from the query"""
        
        # Get all words
        words = re.findall(r'\w+', query.lower())
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add entity texts as important keywords
        for entity in entities:
            entity_words = entity.text.lower().split()
            keywords.extend(entity_words)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _extract_semantic_concepts(self, query: str, entities: List[Entity], 
                                 intent: QueryIntent) -> List[str]:
        """Extract high-level semantic concepts"""
        
        concepts = []
        
        # Intent-based concepts
        if intent == QueryIntent.PROCEDURAL:
            concepts.extend(['tutorial', 'learning', 'instructions'])
        elif intent == QueryIntent.COMPARISON:
            concepts.extend(['comparison', 'evaluation', 'alternatives'])
        elif intent == QueryIntent.TRANSACTIONAL:
            concepts.extend(['commerce', 'purchasing', 'shopping'])
        
        # Entity-based concepts
        for entity in entities:
            if entity.entity_type == EntityType.TECHNOLOGY:
                concepts.append('technology')
            elif entity.entity_type == EntityType.PERSON:
                concepts.append('people')
            elif entity.entity_type == EntityType.ORGANIZATION:
                concepts.append('business')
            elif entity.entity_type == EntityType.LOCATION:
                concepts.append('geography')
        
        # Domain-specific concept detection
        domain_keywords = {
            'programming': ['code', 'function', 'class', 'variable', 'algorithm'],
            'science': ['research', 'study', 'experiment', 'hypothesis', 'theory'],
            'business': ['market', 'sales', 'profit', 'company', 'customer'],
            'health': ['medical', 'doctor', 'treatment', 'symptom', 'disease'],
            'education': ['learn', 'study', 'school', 'university', 'course']
        }
        
        query_lower = query.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                concepts.append(domain)
        
        return list(set(concepts))
    
    def _calculate_complexity(self, query: str, entities: List[Entity]) -> float:
        """Calculate query complexity score"""
        
        complexity = 0.0
        
        # Length factor
        word_count = len(query.split())
        complexity += min(1.0, word_count / 20)  # Normalize to 20 words max
        
        # Entity factor
        complexity += min(0.5, len(entities) / 10)  # Up to 10 entities
        
        # Technical terms factor
        tech_terms = ['algorithm', 'implementation', 'optimization', 'architecture', 'framework']
        tech_count = sum(1 for term in tech_terms if term in query.lower())
        complexity += min(0.3, tech_count / 5)
        
        # Question complexity
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        question_count = sum(1 for word in question_words if word in query.lower())
        complexity += min(0.2, question_count / 3)
        
        return min(1.0, complexity)
    
    def _calculate_ambiguity(self, query: str, entities: List[Entity], 
                           intent: QueryIntent) -> float:
        """Calculate query ambiguity score"""
        
        ambiguity = 0.0
        
        # Short queries are more ambiguous
        word_count = len(query.split())
        if word_count < 3:
            ambiguity += 0.4
        elif word_count < 5:
            ambiguity += 0.2
        
        # Vague terms increase ambiguity
        vague_terms = ['thing', 'stuff', 'something', 'anything', 'it', 'this', 'that']
        vague_count = sum(1 for term in vague_terms if term in query.lower())
        ambiguity += min(0.3, vague_count / 3)
        
        # Multiple possible intents increase ambiguity
        if intent == QueryIntent.INFORMATIONAL:  # Most common/default intent
            ambiguity += 0.1
        
        # Pronouns without clear antecedents
        pronouns = ['it', 'this', 'that', 'they', 'them']
        pronoun_count = sum(1 for pronoun in pronouns if pronoun in query.lower())
        ambiguity += min(0.2, pronoun_count / 2)
        
        return min(1.0, ambiguity)
    
    def _add_to_cache(self, query: str, processed_query: ProcessedQuery):
        """Add processed query to cache"""
        
        if len(self.cache) >= self.cache_size_limit:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[query] = processed_query
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        
        return {
            'total_queries_processed': self.stats['total_queries'],
            'cache_hit_rate': (self.stats['cache_hits'] / max(1, self.stats['total_queries'])) * 100,
            'average_processing_time_ms': self.stats['avg_processing_time'] * 1000,
            'intent_distribution': dict(self.stats['intent_distribution']),
            'entity_distribution': dict(self.stats['entity_distribution']),
            'cache_size': len(self.cache),
            'cache_limit': self.cache_size_limit
        }
    
    def clear_cache(self):
        """Clear the processing cache"""
        self.cache.clear()
        self.stats['cache_hits'] = 0