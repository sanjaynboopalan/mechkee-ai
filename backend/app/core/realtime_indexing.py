"""
Real-time Indexing Pipeline
Incremental indexing with background re-ranking, cache invalidation, and streaming updates
"""

import asyncio
import time
import json
import threading
import queue
from typing import List, Dict, Tuple, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import hashlib
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

class IndexOperationType(Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    RERANK = "rerank"
    INVALIDATE_CACHE = "invalidate_cache"

class IndexPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class IndexOperation:
    """Represents an indexing operation"""
    operation_type: IndexOperationType
    document_id: str
    document_data: Optional[Dict[str, Any]] = None
    priority: IndexPriority = IndexPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IndexingStats:
    """Indexing pipeline statistics"""
    total_operations: int = 0
    operations_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    operations_by_priority: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_processing_time: float = 0.0
    successful_operations: int = 0
    failed_operations: int = 0
    queue_size: int = 0
    cache_invalidations: int = 0
    background_reranks: int = 0
    throughput_per_second: float = 0.0
    last_update: float = field(default_factory=time.time)

class CacheManager:
    """Advanced cache management with intelligent invalidation"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # Multi-level cache
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.result_cache: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        self.vector_cache: Dict[str, np.ndarray] = {}
        
        # Cache metadata
        self.access_times: Dict[str, float] = {}
        self.creation_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        
        # Dependency tracking for smart invalidation
        self.document_dependencies: Dict[str, Set[str]] = defaultdict(set)  # doc_id -> cache_keys
        self.query_dependencies: Dict[str, Set[str]] = defaultdict(set)    # query_hash -> doc_ids
        
        # Cache locks for thread safety
        self.cache_locks = {
            'query': threading.RLock(),
            'result': threading.RLock(),
            'vector': threading.RLock(),
            'metadata': threading.RLock()
        }
    
    def get_query_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        with self.cache_locks['query']:
            if query_hash in self.query_cache:
                # Check TTL
                if time.time() - self.creation_times.get(query_hash, 0) < self.ttl_seconds:
                    self.access_times[query_hash] = time.time()
                    self.access_counts[query_hash] += 1
                    return self.query_cache[query_hash]
                else:
                    # Expired
                    self._remove_from_cache(query_hash, 'query')
        return None
    
    def set_query_result(self, query_hash: str, result: Dict[str, Any], 
                        dependent_docs: Set[str] = None):
        """Cache query result with dependency tracking"""
        with self.cache_locks['query']:
            # Ensure cache size limit
            self._ensure_cache_size('query')
            
            self.query_cache[query_hash] = result
            current_time = time.time()
            self.creation_times[query_hash] = current_time
            self.access_times[query_hash] = current_time
            self.access_counts[query_hash] = 1
            
            # Track dependencies
            if dependent_docs:
                self.query_dependencies[query_hash] = dependent_docs
                for doc_id in dependent_docs:
                    self.document_dependencies[doc_id].add(query_hash)
    
    def invalidate_document_cache(self, document_id: str):
        """Invalidate all cache entries dependent on a document"""
        
        invalidated_keys = set()
        
        # Get all cache keys dependent on this document
        with self.cache_locks['metadata']:
            dependent_keys = self.document_dependencies.get(document_id, set()).copy()
        
        # Invalidate query cache entries
        with self.cache_locks['query']:
            for key in dependent_keys:
                if key in self.query_cache:
                    del self.query_cache[key]
                    invalidated_keys.add(key)
                
                # Clean up metadata
                self.access_times.pop(key, None)
                self.creation_times.pop(key, None)
                self.access_counts.pop(key, None)
        
        # Invalidate result cache
        with self.cache_locks['result']:
            keys_to_remove = []
            for key in self.result_cache:
                if document_id in key:  # Simple heuristic
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.result_cache[key]
                invalidated_keys.add(key)
        
        # Clean up dependencies
        with self.cache_locks['metadata']:
            self.document_dependencies[document_id].clear()
            
            # Remove from query dependencies
            for query_hash in dependent_keys:
                if query_hash in self.query_dependencies:
                    self.query_dependencies[query_hash].discard(document_id)
                    if not self.query_dependencies[query_hash]:
                        del self.query_dependencies[query_hash]
        
        logger.info(f"Invalidated {len(invalidated_keys)} cache entries for document {document_id}")
        return len(invalidated_keys)
    
    def _ensure_cache_size(self, cache_type: str):
        """Ensure cache doesn't exceed size limit using LRU eviction"""
        
        cache_map = {
            'query': self.query_cache,
            'result': self.result_cache,
            'vector': self.vector_cache
        }
        
        cache = cache_map.get(cache_type)
        if not cache or len(cache) < self.max_size:
            return
        
        # Find LRU items
        lru_items = sorted(
            cache.keys(),
            key=lambda k: self.access_times.get(k, 0)
        )
        
        # Remove oldest 10% of items
        remove_count = max(1, int(len(cache) * 0.1))
        for key in lru_items[:remove_count]:
            self._remove_from_cache(key, cache_type)
    
    def _remove_from_cache(self, key: str, cache_type: str):
        """Remove item from cache and clean up metadata"""
        
        cache_map = {
            'query': self.query_cache,
            'result': self.result_cache,
            'vector': self.vector_cache
        }
        
        cache = cache_map.get(cache_type)
        if cache and key in cache:
            del cache[key]
        
        # Clean up metadata
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
        self.access_counts.pop(key, None)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        return {
            'query_cache_size': len(self.query_cache),
            'result_cache_size': len(self.result_cache),
            'vector_cache_size': len(self.vector_cache),
            'total_dependencies': len(self.document_dependencies),
            'avg_access_count': np.mean(list(self.access_counts.values())) if self.access_counts else 0,
            'cache_utilization': {
                'query': len(self.query_cache) / self.max_size * 100,
                'result': len(self.result_cache) / self.max_size * 100,
                'vector': len(self.vector_cache) / self.max_size * 100
            }
        }

class BackgroundReranker:
    """Background re-ranking system for improved relevance"""
    
    def __init__(self, rerank_interval: int = 3600):  # 1 hour
        self.rerank_interval = rerank_interval
        self.last_rerank = time.time()
        
        # Re-ranking strategies
        self.rerank_strategies = {
            'user_feedback': self._rerank_by_user_feedback,
            'click_through': self._rerank_by_ctr,
            'temporal_decay': self._rerank_by_temporal_decay,
            'content_freshness': self._rerank_by_content_freshness
        }
        
        # Tracking data
        self.user_interactions = defaultdict(list)
        self.click_data = defaultdict(lambda: {'clicks': 0, 'impressions': 0})
        self.document_scores = defaultdict(float)
        
        # Background task management
        self.rerank_tasks = set()
        self.is_running = False
    
    async def schedule_reranking(self, document_ids: List[str], 
                               strategy: str = 'user_feedback') -> bool:
        """Schedule background re-ranking for documents"""
        
        if strategy not in self.rerank_strategies:
            logger.error(f"Unknown reranking strategy: {strategy}")
            return False
        
        try:
            task = asyncio.create_task(
                self._execute_reranking(document_ids, strategy)
            )
            self.rerank_tasks.add(task)
            
            # Clean up completed tasks
            self.rerank_tasks = {t for t in self.rerank_tasks if not t.done()}
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule reranking: {e}")
            return False
    
    async def _execute_reranking(self, document_ids: List[str], strategy: str):
        """Execute re-ranking strategy"""
        
        try:
            rerank_func = self.rerank_strategies[strategy]
            new_scores = await rerank_func(document_ids)
            
            # Update document scores
            for doc_id, score in new_scores.items():
                old_score = self.document_scores[doc_id]
                self.document_scores[doc_id] = score
                
                # Log significant score changes
                if abs(score - old_score) > 0.1:
                    logger.info(f"Document {doc_id} score changed: {old_score:.3f} -> {score:.3f}")
            
            logger.info(f"Completed {strategy} reranking for {len(document_ids)} documents")
            
        except Exception as e:
            logger.error(f"Reranking execution failed: {e}")
    
    async def _rerank_by_user_feedback(self, document_ids: List[str]) -> Dict[str, float]:
        """Re-rank based on user feedback signals"""
        
        new_scores = {}
        
        for doc_id in document_ids:
            base_score = self.document_scores.get(doc_id, 0.5)
            
            # Aggregate user feedback
            interactions = self.user_interactions.get(doc_id, [])
            if interactions:
                # Calculate feedback score
                positive_feedback = sum(1 for i in interactions if i.get('feedback', 0) > 0.5)
                total_feedback = len(interactions)
                
                feedback_ratio = positive_feedback / total_feedback
                feedback_boost = (feedback_ratio - 0.5) * 0.2  # Max ±0.1 adjustment
                
                new_scores[doc_id] = max(0.0, min(1.0, base_score + feedback_boost))
            else:
                new_scores[doc_id] = base_score
        
        return new_scores
    
    async def _rerank_by_ctr(self, document_ids: List[str]) -> Dict[str, float]:
        """Re-rank based on click-through rates"""
        
        new_scores = {}
        
        for doc_id in document_ids:
            base_score = self.document_scores.get(doc_id, 0.5)
            
            click_info = self.click_data.get(doc_id, {'clicks': 0, 'impressions': 0})
            
            if click_info['impressions'] > 0:
                ctr = click_info['clicks'] / click_info['impressions']
                
                # Apply CTR boost (0.05 CTR = neutral, higher = boost, lower = penalty)
                ctr_adjustment = (ctr - 0.05) * 2.0  # Scale factor
                ctr_adjustment = max(-0.15, min(0.15, ctr_adjustment))  # Cap adjustment
                
                new_scores[doc_id] = max(0.0, min(1.0, base_score + ctr_adjustment))
            else:
                new_scores[doc_id] = base_score
        
        return new_scores
    
    async def _rerank_by_temporal_decay(self, document_ids: List[str]) -> Dict[str, float]:
        """Apply temporal decay to document scores"""
        
        new_scores = {}
        current_time = time.time()
        
        for doc_id in document_ids:
            base_score = self.document_scores.get(doc_id, 0.5)
            
            # Mock document age (in production, get from document metadata)
            doc_age_days = 30  # Assume 30 days for simplicity
            
            # Apply exponential decay (half-life of 90 days)
            decay_factor = np.exp(-doc_age_days * np.log(2) / 90)
            
            new_scores[doc_id] = base_score * decay_factor
        
        return new_scores
    
    async def _rerank_by_content_freshness(self, document_ids: List[str]) -> Dict[str, float]:
        """Re-rank based on content freshness"""
        
        new_scores = {}
        
        for doc_id in document_ids:
            base_score = self.document_scores.get(doc_id, 0.5)
            
            # Mock freshness score (in production, analyze content updates)
            freshness_score = 0.8  # Assume relatively fresh content
            
            # Boost fresh content slightly
            freshness_boost = (freshness_score - 0.5) * 0.1
            
            new_scores[doc_id] = max(0.0, min(1.0, base_score + freshness_boost))
        
        return new_scores
    
    def record_user_interaction(self, document_id: str, interaction_data: Dict[str, Any]):
        """Record user interaction for re-ranking"""
        
        interaction_data['timestamp'] = time.time()
        self.user_interactions[document_id].append(interaction_data)
        
        # Limit interaction history
        if len(self.user_interactions[document_id]) > 100:
            self.user_interactions[document_id] = self.user_interactions[document_id][-100:]
    
    def record_click_data(self, document_id: str, clicked: bool):
        """Record click data for CTR-based re-ranking"""
        
        self.click_data[document_id]['impressions'] += 1
        if clicked:
            self.click_data[document_id]['clicks'] += 1

class StreamingIndexer:
    """High-performance streaming indexer for real-time updates"""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Streaming buffers
        self.add_buffer: List[Dict[str, Any]] = []
        self.update_buffer: List[Dict[str, Any]] = []
        self.delete_buffer: List[str] = []
        
        # Buffer locks
        self.buffer_lock = threading.RLock()
        
        # Periodic flush task
        self.flush_task = None
        self.last_flush = time.time()
        
        # Index update callbacks
        self.index_callbacks: List[Callable] = []
        
    def add_document_stream(self, document: Dict[str, Any]):
        """Add document to streaming buffer"""
        
        with self.buffer_lock:
            self.add_buffer.append(document)
            
            if len(self.add_buffer) >= self.batch_size:
                asyncio.create_task(self._flush_add_buffer())
    
    def update_document_stream(self, document: Dict[str, Any]):
        """Update document in streaming buffer"""
        
        with self.buffer_lock:
            self.update_buffer.append(document)
            
            if len(self.update_buffer) >= self.batch_size:
                asyncio.create_task(self._flush_update_buffer())
    
    def delete_document_stream(self, document_id: str):
        """Delete document from streaming buffer"""
        
        with self.buffer_lock:
            self.delete_buffer.append(document_id)
            
            if len(self.delete_buffer) >= self.batch_size:
                asyncio.create_task(self._flush_delete_buffer())
    
    async def _flush_add_buffer(self):
        """Flush add buffer to index"""
        
        with self.buffer_lock:
            if not self.add_buffer:
                return
            
            documents = self.add_buffer.copy()
            self.add_buffer.clear()
        
        try:
            # Process documents in parallel
            tasks = []
            for callback in self.index_callbacks:
                for doc in documents:
                    task = asyncio.create_task(callback('add', doc))
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info(f"Flushed {len(documents)} documents to index")
            
        except Exception as e:
            logger.error(f"Failed to flush add buffer: {e}")
    
    async def _flush_update_buffer(self):
        """Flush update buffer to index"""
        
        with self.buffer_lock:
            if not self.update_buffer:
                return
            
            documents = self.update_buffer.copy()
            self.update_buffer.clear()
        
        try:
            tasks = []
            for callback in self.index_callbacks:
                for doc in documents:
                    task = asyncio.create_task(callback('update', doc))
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info(f"Flushed {len(documents)} updates to index")
            
        except Exception as e:
            logger.error(f"Failed to flush update buffer: {e}")
    
    async def _flush_delete_buffer(self):
        """Flush delete buffer to index"""
        
        with self.buffer_lock:
            if not self.delete_buffer:
                return
            
            doc_ids = self.delete_buffer.copy()
            self.delete_buffer.clear()
        
        try:
            tasks = []
            for callback in self.index_callbacks:
                for doc_id in doc_ids:
                    task = asyncio.create_task(callback('delete', {'id': doc_id}))
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info(f"Flushed {len(doc_ids)} deletions from index")
            
        except Exception as e:
            logger.error(f"Failed to flush delete buffer: {e}")
    
    async def start_periodic_flush(self):
        """Start periodic buffer flushing"""
        
        while True:
            await asyncio.sleep(self.flush_interval)
            
            current_time = time.time()
            if current_time - self.last_flush >= self.flush_interval:
                await self._flush_all_buffers()
                self.last_flush = current_time
    
    async def _flush_all_buffers(self):
        """Flush all buffers"""
        
        tasks = [
            self._flush_add_buffer(),
            self._flush_update_buffer(),
            self._flush_delete_buffer()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def register_index_callback(self, callback: Callable):
        """Register callback for index operations"""
        self.index_callbacks.append(callback)

class RealTimeIndexingPipeline:
    """Main real-time indexing pipeline"""
    
    def __init__(self, 
                 max_queue_size: int = 10000,
                 num_workers: int = 4,
                 enable_caching: bool = True,
                 enable_background_reranking: bool = True):
        
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.enable_caching = enable_caching
        self.enable_background_reranking = enable_background_reranking
        
        # Priority queue for operations
        self.operation_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        
        # Components
        self.cache_manager = CacheManager() if enable_caching else None
        self.background_reranker = BackgroundReranker() if enable_background_reranking else None
        self.streaming_indexer = StreamingIndexer()
        
        # Statistics
        self.stats = IndexingStats()
        
        # Index operation handlers
        self.operation_handlers = {
            IndexOperationType.ADD: self._handle_add_operation,
            IndexOperationType.UPDATE: self._handle_update_operation,
            IndexOperationType.DELETE: self._handle_delete_operation,
            IndexOperationType.RERANK: self._handle_rerank_operation,
            IndexOperationType.INVALIDATE_CACHE: self._handle_cache_invalidation
        }
        
        # External index systems (to be set by user)
        self.index_systems: List[Any] = []
        
        # Performance monitoring
        self.performance_window = deque(maxlen=1000)  # Last 1000 operations
    
    async def start(self):
        """Start the indexing pipeline"""
        
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start streaming indexer
        asyncio.create_task(self.streaming_indexer.start_periodic_flush())
        
        # Start statistics collection
        asyncio.create_task(self._collect_statistics())
        
        logger.info(f"Started indexing pipeline with {self.num_workers} workers")
    
    async def stop(self):
        """Stop the indexing pipeline"""
        
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Flush remaining operations
        await self._flush_remaining_operations()
        
        logger.info("Stopped indexing pipeline")
    
    async def submit_operation(self, operation: IndexOperation) -> bool:
        """Submit an indexing operation"""
        
        if not self.is_running:
            logger.error("Pipeline not running")
            return False
        
        try:
            # Priority queue uses negative priority for high-priority items
            priority = -operation.priority.value
            
            await self.operation_queue.put((priority, time.time(), operation))
            
            self.stats.total_operations += 1
            self.stats.operations_by_type[operation.operation_type.value] += 1
            self.stats.operations_by_priority[operation.priority.name] += 1
            
            return True
            
        except asyncio.QueueFull:
            logger.error("Operation queue full")
            return False
        except Exception as e:
            logger.error(f"Failed to submit operation: {e}")
            return False
    
    async def _worker(self, worker_name: str):
        """Worker task for processing operations"""
        
        logger.info(f"Started worker {worker_name}")
        
        while self.is_running:
            try:
                # Get operation from queue with timeout
                priority, timestamp, operation = await asyncio.wait_for(
                    self.operation_queue.get(), timeout=1.0
                )
                
                start_time = time.time()
                
                # Process operation
                success = await self._process_operation(operation)
                
                # Update statistics
                processing_time = time.time() - start_time
                self.performance_window.append(processing_time)
                
                if success:
                    self.stats.successful_operations += 1
                    
                    # Execute callback if provided
                    if operation.callback:
                        try:
                            await operation.callback(operation, success)
                        except Exception as e:
                            logger.error(f"Callback execution failed: {e}")
                else:
                    self.stats.failed_operations += 1
                    
                    # Retry on failure
                    if operation.retry_count < operation.max_retries:
                        operation.retry_count += 1
                        await self.submit_operation(operation)
                        logger.info(f"Retrying operation {operation.operation_type.value} for {operation.document_id}")
                
                # Mark task as done
                self.operation_queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # No operations in queue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
    
    async def _process_operation(self, operation: IndexOperation) -> bool:
        """Process a single indexing operation"""
        
        try:
            handler = self.operation_handlers.get(operation.operation_type)
            if not handler:
                logger.error(f"No handler for operation type: {operation.operation_type}")
                return False
            
            return await handler(operation)
            
        except Exception as e:
            logger.error(f"Operation processing failed: {e}")
            return False
    
    async def _handle_add_operation(self, operation: IndexOperation) -> bool:
        """Handle document addition"""
        
        try:
            # Add to all registered index systems
            tasks = []
            for index_system in self.index_systems:
                if hasattr(index_system, 'add_document'):
                    task = asyncio.create_task(
                        index_system.add_document(operation.document_data)
                    )
                    tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success = all(r is True or (not isinstance(r, Exception)) for r in results)
            else:
                success = True
            
            # Invalidate related cache entries
            if self.cache_manager and success:
                self.cache_manager.invalidate_document_cache(operation.document_id)
            
            logger.debug(f"Added document {operation.document_id}")
            return success
            
        except Exception as e:
            logger.error(f"Add operation failed for {operation.document_id}: {e}")
            return False
    
    async def _handle_update_operation(self, operation: IndexOperation) -> bool:
        """Handle document update"""
        
        try:
            # Update in all registered index systems
            tasks = []
            for index_system in self.index_systems:
                if hasattr(index_system, 'update_document'):
                    task = asyncio.create_task(
                        index_system.update_document(operation.document_data)
                    )
                    tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success = all(r is True or (not isinstance(r, Exception)) for r in results)
            else:
                success = True
            
            # Invalidate cache
            if self.cache_manager and success:
                invalidated = self.cache_manager.invalidate_document_cache(operation.document_id)
                self.stats.cache_invalidations += invalidated
            
            # Schedule re-ranking
            if self.background_reranker and success:
                await self.background_reranker.schedule_reranking([operation.document_id])
                self.stats.background_reranks += 1
            
            logger.debug(f"Updated document {operation.document_id}")
            return success
            
        except Exception as e:
            logger.error(f"Update operation failed for {operation.document_id}: {e}")
            return False
    
    async def _handle_delete_operation(self, operation: IndexOperation) -> bool:
        """Handle document deletion"""
        
        try:
            # Delete from all registered index systems
            tasks = []
            for index_system in self.index_systems:
                if hasattr(index_system, 'delete_document'):
                    task = asyncio.create_task(
                        index_system.delete_document(operation.document_id)
                    )
                    tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success = all(r is True or (not isinstance(r, Exception)) for r in results)
            else:
                success = True
            
            # Invalidate cache
            if self.cache_manager and success:
                invalidated = self.cache_manager.invalidate_document_cache(operation.document_id)
                self.stats.cache_invalidations += invalidated
            
            logger.debug(f"Deleted document {operation.document_id}")
            return success
            
        except Exception as e:
            logger.error(f"Delete operation failed for {operation.document_id}: {e}")
            return False
    
    async def _handle_rerank_operation(self, operation: IndexOperation) -> bool:
        """Handle re-ranking operation"""
        
        try:
            if not self.background_reranker:
                return True
            
            document_ids = operation.metadata.get('document_ids', [operation.document_id])
            strategy = operation.metadata.get('strategy', 'user_feedback')
            
            success = await self.background_reranker.schedule_reranking(document_ids, strategy)
            
            if success:
                self.stats.background_reranks += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Rerank operation failed: {e}")
            return False
    
    async def _handle_cache_invalidation(self, operation: IndexOperation) -> bool:
        """Handle cache invalidation"""
        
        try:
            if not self.cache_manager:
                return True
            
            invalidated = self.cache_manager.invalidate_document_cache(operation.document_id)
            self.stats.cache_invalidations += invalidated
            
            return True
            
        except Exception as e:
            logger.error(f"Cache invalidation failed for {operation.document_id}: {e}")
            return False
    
    async def _collect_statistics(self):
        """Collect performance statistics"""
        
        while self.is_running:
            await asyncio.sleep(10)  # Update every 10 seconds
            
            # Update queue size
            self.stats.queue_size = self.operation_queue.qsize()
            
            # Update average processing time
            if self.performance_window:
                self.stats.avg_processing_time = np.mean(list(self.performance_window))
            
            # Calculate throughput
            current_time = time.time()
            time_diff = current_time - self.stats.last_update
            
            if time_diff > 0:
                operations_diff = self.stats.successful_operations
                self.stats.throughput_per_second = operations_diff / time_diff
            
            self.stats.last_update = current_time
    
    async def _flush_remaining_operations(self):
        """Flush remaining operations in queue"""
        
        try:
            while not self.operation_queue.empty():
                _, _, operation = await asyncio.wait_for(
                    self.operation_queue.get(), timeout=0.1
                )
                await self._process_operation(operation)
                self.operation_queue.task_done()
        except asyncio.TimeoutError:
            pass
    
    def register_index_system(self, index_system: Any):
        """Register an index system"""
        self.index_systems.append(index_system)
        
        # Register with streaming indexer
        async def index_callback(operation_type: str, document: Dict[str, Any]):
            if operation_type == 'add' and hasattr(index_system, 'add_document'):
                await index_system.add_document(document)
            elif operation_type == 'update' and hasattr(index_system, 'update_document'):
                await index_system.update_document(document)
            elif operation_type == 'delete' and hasattr(index_system, 'delete_document'):
                await index_system.delete_document(document.get('id'))
        
        self.streaming_indexer.register_index_callback(index_callback)
    
    # Convenience methods for common operations
    
    async def add_document(self, document_id: str, document_data: Dict[str, Any], 
                          priority: IndexPriority = IndexPriority.NORMAL) -> bool:
        """Add a document to the index"""
        
        operation = IndexOperation(
            operation_type=IndexOperationType.ADD,
            document_id=document_id,
            document_data=document_data,
            priority=priority
        )
        
        return await self.submit_operation(operation)
    
    async def update_document(self, document_id: str, document_data: Dict[str, Any],
                             priority: IndexPriority = IndexPriority.NORMAL) -> bool:
        """Update a document in the index"""
        
        operation = IndexOperation(
            operation_type=IndexOperationType.UPDATE,
            document_id=document_id,
            document_data=document_data,
            priority=priority
        )
        
        return await self.submit_operation(operation)
    
    async def delete_document(self, document_id: str,
                             priority: IndexPriority = IndexPriority.NORMAL) -> bool:
        """Delete a document from the index"""
        
        operation = IndexOperation(
            operation_type=IndexOperationType.DELETE,
            document_id=document_id,
            priority=priority
        )
        
        return await self.submit_operation(operation)
    
    async def trigger_reranking(self, document_ids: List[str], strategy: str = 'user_feedback',
                               priority: IndexPriority = IndexPriority.LOW) -> bool:
        """Trigger background re-ranking"""
        
        operation = IndexOperation(
            operation_type=IndexOperationType.RERANK,
            document_id=document_ids[0] if document_ids else 'batch',
            priority=priority,
            metadata={'document_ids': document_ids, 'strategy': strategy}
        )
        
        return await self.submit_operation(operation)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        
        stats = {
            'pipeline': {
                'total_operations': self.stats.total_operations,
                'successful_operations': self.stats.successful_operations,
                'failed_operations': self.stats.failed_operations,
                'success_rate': (self.stats.successful_operations / 
                               max(1, self.stats.total_operations)) * 100,
                'queue_size': self.stats.queue_size,
                'avg_processing_time_ms': self.stats.avg_processing_time * 1000,
                'throughput_per_second': self.stats.throughput_per_second,
                'operations_by_type': dict(self.stats.operations_by_type),
                'operations_by_priority': dict(self.stats.operations_by_priority),
                'cache_invalidations': self.stats.cache_invalidations,
                'background_reranks': self.stats.background_reranks
            }
        }
        
        if self.cache_manager:
            stats['cache'] = self.cache_manager.get_cache_stats()
        
        if self.background_reranker:
            stats['reranking'] = {
                'active_tasks': len(self.background_reranker.rerank_tasks),
                'user_interactions': len(self.background_reranker.user_interactions),
                'click_data_entries': len(self.background_reranker.click_data)
            }
        
        return stats
    
    # Streaming interface
    
    def stream_add_document(self, document: Dict[str, Any]):
        """Add document via streaming interface"""
        self.streaming_indexer.add_document_stream(document)
    
    def stream_update_document(self, document: Dict[str, Any]):
        """Update document via streaming interface"""
        self.streaming_indexer.update_document_stream(document)
    
    def stream_delete_document(self, document_id: str):
        """Delete document via streaming interface"""
        self.streaming_indexer.delete_document_stream(document_id)