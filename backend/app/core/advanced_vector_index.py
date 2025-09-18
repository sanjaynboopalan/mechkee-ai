"""
Advanced Vector Indexing System
Implements HNSW (Hierarchical Navigable Small World) algorithm with cutting-edge optimizations
"""

import numpy as np
import heapq
import random
import pickle
import threading
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class VectorDocument:
    """Enhanced document representation with metadata"""
    id: str
    vector: np.ndarray
    content: str
    metadata: Dict[str, Any]
    timestamp: float
    authority_score: float = 0.0
    click_through_rate: float = 0.0
    engagement_score: float = 0.0

class QuantizedVector:
    """Product Quantization for memory-efficient storage"""
    def __init__(self, vector: np.ndarray, codebook_size: int = 256, subvector_count: int = 8):
        self.original_dim = len(vector)
        self.subvector_count = subvector_count
        self.subvector_dim = self.original_dim // subvector_count
        self.codebook_size = codebook_size
        
        # Simple quantization (in production, use k-means clustering)
        self.subvectors = self._split_vector(vector)
        self.codes = self._quantize_subvectors()
    
    def _split_vector(self, vector: np.ndarray) -> List[np.ndarray]:
        """Split vector into subvectors"""
        return [vector[i:i+self.subvector_dim] 
                for i in range(0, self.original_dim, self.subvector_dim)]
    
    def _quantize_subvectors(self) -> List[int]:
        """Quantize each subvector to a code"""
        return [hash(tuple(sv)) % self.codebook_size for sv in self.subvectors]
    
    def approximate_distance(self, other_vector: np.ndarray) -> float:
        """Fast approximate distance computation"""
        other_quantized = QuantizedVector(other_vector, self.codebook_size, self.subvector_count)
        
        # Hamming distance between codes as approximation
        matches = sum(1 for c1, c2 in zip(self.codes, other_quantized.codes) if c1 == c2)
        return 1.0 - (matches / len(self.codes))

class HNSWNode:
    """Node in the HNSW graph"""
    def __init__(self, document: VectorDocument, level: int):
        self.document = document
        self.level = level
        self.connections: Dict[int, Set[str]] = defaultdict(set)  # level -> set of node IDs
        self.lock = threading.RLock()
    
    def add_connection(self, level: int, node_id: str):
        """Thread-safe connection addition"""
        with self.lock:
            self.connections[level].add(node_id)
    
    def remove_connection(self, level: int, node_id: str):
        """Thread-safe connection removal"""
        with self.lock:
            self.connections[level].discard(node_id)

class AdvancedVectorIndex:
    """
    Advanced HNSW-based vector index with modern optimizations
    """
    
    def __init__(self, 
                 dim: int = 1536,
                 max_connections: int = 16,
                 max_connections_0: int = 32,
                 ef_construction: int = 200,
                 ml: float = 1/np.log(2),
                 use_quantization: bool = True,
                 enable_caching: bool = True):
        
        self.dim = dim
        self.max_connections = max_connections  # M
        self.max_connections_0 = max_connections_0  # M0 for level 0
        self.ef_construction = ef_construction
        self.ml = ml  # Level generation parameter
        self.use_quantization = use_quantization
        self.enable_caching = enable_caching
        
        # Core data structures
        self.nodes: Dict[str, HNSWNode] = {}
        self.entry_point: Optional[str] = None
        self.levels: Dict[int, Set[str]] = defaultdict(set)
        
        # Performance optimizations
        self.distance_cache: Dict[Tuple[str, str], float] = {} if enable_caching else None
        self.quantized_vectors: Dict[str, QuantizedVector] = {}
        
        # Thread safety
        self.index_lock = threading.RWLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'cache_hits': 0,
            'distance_computations': 0
        }
    
    def _generate_level(self) -> int:
        """Generate random level for new node"""
        level = 0
        while random.random() < self.ml and level < 16:  # Cap at 16 levels
            level += 1
        return level
    
    def _distance(self, vec1: np.ndarray, vec2: np.ndarray, 
                  use_cache: bool = True, doc1_id: str = None, doc2_id: str = None) -> float:
        """Optimized distance computation with caching"""
        
        # Use cache if enabled and IDs provided
        if use_cache and self.enable_caching and doc1_id and doc2_id:
            cache_key = tuple(sorted([doc1_id, doc2_id]))
            if cache_key in self.distance_cache:
                self.search_stats['cache_hits'] += 1
                return self.distance_cache[cache_key]
        
        # Use quantized approximation for initial filtering
        if self.use_quantization and doc1_id in self.quantized_vectors and len(self.distance_cache) > 1000:
            approx_dist = self.quantized_vectors[doc1_id].approximate_distance(vec2)
            if approx_dist > 0.8:  # Early pruning for very dissimilar vectors
                return approx_dist
        
        # Compute exact cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            distance = 1.0
        else:
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            distance = 1.0 - similarity
        
        # Cache the result
        if use_cache and self.enable_caching and doc1_id and doc2_id:
            cache_key = tuple(sorted([doc1_id, doc2_id]))
            self.distance_cache[cache_key] = distance
        
        self.search_stats['distance_computations'] += 1
        return distance
    
    def _search_layer(self, query_vector: np.ndarray, entry_points: Set[str], 
                     num_closest: int, level: int) -> List[Tuple[float, str]]:
        """Search within a specific layer"""
        visited = set()
        candidates = []
        w = []  # Dynamic list of closest elements
        
        # Initialize with entry points
        for ep_id in entry_points:
            if ep_id not in self.nodes:
                continue
            
            ep_node = self.nodes[ep_id]
            dist = self._distance(query_vector, ep_node.document.vector, 
                                doc2_id=ep_id)
            heapq.heappush(candidates, (-dist, ep_id))  # Max heap for candidates
            heapq.heappush(w, (dist, ep_id))  # Min heap for results
            visited.add(ep_id)
        
        while candidates:
            current_dist, current_id = heapq.heappop(candidates)
            current_dist = -current_dist  # Convert back from max heap
            
            # Stop if current distance is worse than furthest in w
            if len(w) >= num_closest and current_dist > w[0][0]:
                break
            
            current_node = self.nodes[current_id]
            
            # Explore connections at this level
            for neighbor_id in current_node.connections.get(level, set()):
                if neighbor_id not in visited and neighbor_id in self.nodes:
                    visited.add(neighbor_id)
                    neighbor_node = self.nodes[neighbor_id]
                    
                    dist = self._distance(query_vector, neighbor_node.document.vector,
                                        doc2_id=neighbor_id)
                    
                    if len(w) < num_closest:
                        heapq.heappush(candidates, (-dist, neighbor_id))
                        heapq.heappush(w, (dist, neighbor_id))
                    elif dist < w[0][0]:  # Better than worst in w
                        heapq.heappush(candidates, (-dist, neighbor_id))
                        heapq.heapreplace(w, (dist, neighbor_id))
        
        return sorted(w)
    
    def add_document(self, document: VectorDocument) -> bool:
        """Add document to the index"""
        try:
            with self.index_lock.writer():
                level = self._generate_level()
                node = HNSWNode(document, level)
                
                # Add quantized version if enabled
                if self.use_quantization:
                    self.quantized_vectors[document.id] = QuantizedVector(document.vector)
                
                # Handle first insertion
                if not self.nodes:
                    self.entry_point = document.id
                    self.nodes[document.id] = node
                    for lev in range(level + 1):
                        self.levels[lev].add(document.id)
                    return True
                
                # Find closest nodes at each level
                entry_points = {self.entry_point}
                
                # Search from top level down to level+1
                for lev in range(self._get_max_level(), level, -1):
                    entry_points = {node_id for _, node_id in 
                                  self._search_layer(document.vector, entry_points, 1, lev)}
                
                # Search and connect at levels from level down to 0
                for lev in range(min(level, self._get_max_level()), -1, -1):
                    candidates = self._search_layer(document.vector, entry_points, 
                                                  self.ef_construction, lev)
                    
                    max_conn = self.max_connections_0 if lev == 0 else self.max_connections
                    selected = self._select_neighbors_heuristic(document.vector, candidates, max_conn)
                    
                    # Add bidirectional connections
                    for _, neighbor_id in selected:
                        node.add_connection(lev, neighbor_id)
                        self.nodes[neighbor_id].add_connection(lev, document.id)
                        
                        # Prune connections if needed
                        self._prune_connections(neighbor_id, lev)
                    
                    entry_points = {neighbor_id for _, neighbor_id in selected}
                
                # Update data structures
                self.nodes[document.id] = node
                for lev in range(level + 1):
                    self.levels[lev].add(document.id)
                
                # Update entry point if this node has higher level
                if level > self._get_max_level():
                    self.entry_point = document.id
                
                return True
                
        except Exception as e:
            logger.error(f"Error adding document {document.id}: {e}")
            return False
    
    def _get_max_level(self) -> int:
        """Get maximum level in the index"""
        if not self.entry_point or self.entry_point not in self.nodes:
            return -1
        return self.nodes[self.entry_point].level
    
    def _select_neighbors_heuristic(self, query_vector: np.ndarray, 
                                   candidates: List[Tuple[float, str]], 
                                   num_select: int) -> List[Tuple[float, str]]:
        """Heuristic neighbor selection to maintain connectivity"""
        if len(candidates) <= num_select:
            return candidates
        
        selected = []
        candidates_copy = candidates.copy()
        
        while len(selected) < num_select and candidates_copy:
            # Select closest candidate
            candidates_copy.sort()
            closest_dist, closest_id = candidates_copy.pop(0)
            selected.append((closest_dist, closest_id))
            
            if closest_id not in self.nodes:
                continue
                
            closest_vector = self.nodes[closest_id].document.vector
            
            # Remove candidates that are too close to selected one (diversity)
            candidates_copy = [
                (dist, node_id) for dist, node_id in candidates_copy
                if node_id not in self.nodes or 
                self._distance(closest_vector, self.nodes[node_id].document.vector) > 0.3
            ]
        
        return selected
    
    def _prune_connections(self, node_id: str, level: int):
        """Prune excess connections while maintaining connectivity"""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        max_conn = self.max_connections_0 if level == 0 else self.max_connections
        
        connections = list(node.connections[level])
        if len(connections) <= max_conn:
            return
        
        # Evaluate all connections and keep the best ones
        candidates = []
        node_vector = node.document.vector
        
        for conn_id in connections:
            if conn_id in self.nodes:
                dist = self._distance(node_vector, self.nodes[conn_id].document.vector,
                                    doc1_id=node_id, doc2_id=conn_id)
                candidates.append((dist, conn_id))
        
        # Select best connections using heuristic
        selected = self._select_neighbors_heuristic(node_vector, candidates, max_conn)
        
        # Update connections
        new_connections = {conn_id for _, conn_id in selected}
        removed_connections = node.connections[level] - new_connections
        
        # Remove bidirectional connections
        for removed_id in removed_connections:
            node.remove_connection(level, removed_id)
            if removed_id in self.nodes:
                self.nodes[removed_id].remove_connection(level, node_id)
        
        node.connections[level] = new_connections
    
    async def search(self, query_vector: np.ndarray, k: int = 10, 
                    ef: int = None) -> List[Tuple[float, VectorDocument]]:
        """Advanced search with performance optimizations"""
        start_time = time.time()
        
        if ef is None:
            ef = max(self.ef_construction, k)
        
        if not self.entry_point or self.entry_point not in self.nodes:
            return []
        
        try:
            with self.index_lock.reader():
                # Search from top level down to level 1
                entry_points = {self.entry_point}
                
                for level in range(self._get_max_level(), 0, -1):
                    entry_points = {node_id for _, node_id in 
                                  self._search_layer(query_vector, entry_points, 1, level)}
                
                # Search level 0 with larger ef
                candidates = self._search_layer(query_vector, entry_points, ef, 0)
                
                # Convert to documents and apply additional ranking
                results = []
                for dist, node_id in candidates[:k]:
                    if node_id in self.nodes:
                        doc = self.nodes[node_id].document
                        
                        # Apply additional scoring factors
                        enhanced_score = self._enhance_score(dist, doc)
                        results.append((enhanced_score, doc))
                
                # Update statistics
                search_time = time.time() - start_time
                self.search_stats['total_searches'] += 1
                self.search_stats['avg_search_time'] = (
                    (self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1) + 
                     search_time) / self.search_stats['total_searches']
                )
                
                return sorted(results)
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _enhance_score(self, base_distance: float, document: VectorDocument) -> float:
        """Enhance scoring with document metadata"""
        
        # Time decay factor (newer documents get slight boost)
        current_time = time.time()
        age_hours = (current_time - document.timestamp) / 3600
        time_decay = np.exp(-age_hours / (24 * 7))  # Week half-life
        
        # Authority and engagement factors
        authority_boost = 1.0 + (document.authority_score * 0.1)
        engagement_boost = 1.0 + (document.engagement_score * 0.05)
        ctr_boost = 1.0 + (document.click_through_rate * 0.1)
        
        # Combined score (lower is better for distance)
        enhanced_distance = base_distance / (authority_boost * engagement_boost * ctr_boost * time_decay)
        
        return enhanced_distance
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        with self.index_lock.reader():
            return {
                'total_documents': len(self.nodes),
                'total_levels': len(self.levels),
                'max_level': self._get_max_level(),
                'cache_size': len(self.distance_cache) if self.distance_cache else 0,
                'search_stats': self.search_stats.copy(),
                'avg_connections_level_0': np.mean([
                    len(node.connections.get(0, set())) 
                    for node in self.nodes.values()
                ]) if self.nodes else 0,
                'memory_usage_mb': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        base_size = 0
        
        # Estimate node storage
        for node in self.nodes.values():
            base_size += node.document.vector.nbytes
            base_size += len(str(node.connections)) * 8  # Rough estimate
        
        # Cache storage
        if self.distance_cache:
            base_size += len(self.distance_cache) * 16  # 8 bytes per float + overhead
        
        # Quantized vectors
        base_size += len(self.quantized_vectors) * 64  # Rough estimate
        
        return base_size / (1024 * 1024)  # Convert to MB
    
    def save_index(self, filepath: str):
        """Save index to disk"""
        try:
            with self.index_lock.reader():
                save_data = {
                    'nodes': self.nodes,
                    'entry_point': self.entry_point,
                    'levels': dict(self.levels),
                    'quantized_vectors': self.quantized_vectors,
                    'config': {
                        'dim': self.dim,
                        'max_connections': self.max_connections,
                        'max_connections_0': self.max_connections_0,
                        'ef_construction': self.ef_construction,
                        'ml': self.ml
                    }
                }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(save_data, f)
                    
                logger.info(f"Index saved to {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def load_index(self, filepath: str):
        """Load index from disk"""
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            with self.index_lock.writer():
                self.nodes = save_data['nodes']
                self.entry_point = save_data['entry_point']
                self.levels = defaultdict(set, save_data['levels'])
                self.quantized_vectors = save_data.get('quantized_vectors', {})
                
                # Restore config
                config = save_data.get('config', {})
                self.dim = config.get('dim', self.dim)
                self.max_connections = config.get('max_connections', self.max_connections)
                self.max_connections_0 = config.get('max_connections_0', self.max_connections_0)
                self.ef_construction = config.get('ef_construction', self.ef_construction)
                self.ml = config.get('ml', self.ml)
                
                logger.info(f"Index loaded from {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")

# Thread-safe RWLock implementation
class RWLock:
    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(threading.RLock())
        self._write_ready = threading.Condition(threading.RLock())
    
    def reader(self):
        return self._ReaderLock(self)
    
    def writer(self):
        return self._WriterLock(self)
    
    class _ReaderLock:
        def __init__(self, rwlock):
            self.rwlock = rwlock
        
        def __enter__(self):
            with self.rwlock._read_ready:
                while self.rwlock._writers > 0:
                    self.rwlock._read_ready.wait()
                self.rwlock._readers += 1
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            with self.rwlock._read_ready:
                self.rwlock._readers -= 1
                if self.rwlock._readers == 0:
                    self.rwlock._read_ready.notify_all()
    
    class _WriterLock:
        def __init__(self, rwlock):
            self.rwlock = rwlock
        
        def __enter__(self):
            with self.rwlock._write_ready:
                while self.rwlock._writers > 0 or self.rwlock._readers > 0:
                    self.rwlock._write_ready.wait()
                self.rwlock._writers += 1
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            with self.rwlock._write_ready:
                self.rwlock._writers -= 1
                self.rwlock._write_ready.notify_all()

# Patch threading module to include our RWLock
threading.RWLock = RWLock