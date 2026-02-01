"""
Search Performance Optimizer for Production-Scale RAG Systems

This module provides advanced search optimization capabilities including
query analysis, index optimization, and performance monitoring for
large-scale document retrieval systems.
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import hashlib
import pickle
import sqlite3
from pathlib import Path

# External dependencies
import faiss
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import redis

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query: str
    execution_time: float
    results_count: int
    cache_hit: bool
    index_used: str
    timestamp: float
    memory_usage: int
    cpu_usage: float

@dataclass
class IndexStats:
    """Index performance statistics"""
    index_name: str
    total_documents: int
    index_size_bytes: int
    avg_query_time: float
    cache_hit_rate: float
    last_updated: float
    fragmentation_ratio: float

@dataclass
class OptimizationConfig:
    """Configuration for search optimizer"""
    # Query processing
    enable_query_expansion: bool = True
    enable_spell_correction: bool = True
    enable_stemming: bool = True
    remove_stopwords: bool = True
    
    # Index optimization
    faiss_index_type: str = "IVF"
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    
    # Caching
    query_cache_size: int = 1000
    result_cache_ttl: int = 3600
    index_cache_size: int = 500
    
    # Performance thresholds
    max_query_time: float = 5.0
    target_recall: float = 0.95
    max_memory_usage: int = 4 * 1024 * 1024 * 1024  # 4GB
    
    # Optimization intervals
    auto_optimize_interval: int = 3600  # 1 hour
    statistics_window: int = 86400  # 24 hours

class QueryProcessor:
    """Advanced query processing and expansion"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.stemmer = PorterStemmer() if config.enable_stemming else None
        self.stop_words = set(stopwords.words('english')) if config.remove_stopwords else set()
        self.nlp = None
        self._load_models()
    
    def _load_models(self):
        """Load NLP models"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Load spaCy model for advanced processing
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic processing")
                self.nlp = None
        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process and expand query for optimal search
        
        Returns:
            Dict containing original query, processed query, expansions, etc.
        """
        result = {
            "original": query,
            "processed": query.lower().strip(),
            "tokens": [],
            "expanded_terms": [],
            "entities": [],
            "intent": "search"
        }
        
        # Tokenization
        if self.nlp:
            doc = self.nlp(query)
            result["tokens"] = [token.lemma_ for token in doc if not token.is_space]
            result["entities"] = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Intent detection (basic)
            if any(token.pos_ == "VERB" for token in doc):
                result["intent"] = "action"
            elif any(token.ent_type_ == "PERSON" for token in doc):
                result["intent"] = "person_search"
        else:
            result["tokens"] = word_tokenize(query.lower())
        
        # Remove stopwords
        if self.config.remove_stopwords:
            result["tokens"] = [token for token in result["tokens"] 
                              if token not in self.stop_words]
        
        # Stemming
        if self.config.enable_stemming and self.stemmer:
            result["tokens"] = [self.stemmer.stem(token) for token in result["tokens"]]
        
        # Reconstruct processed query
        result["processed"] = " ".join(result["tokens"])
        
        # Query expansion
        if self.config.enable_query_expansion:
            result["expanded_terms"] = self._expand_query(result["tokens"])
        
        return result
    
    def _expand_query(self, tokens: List[str]) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded = []
        
        # Simple synonym expansion (in production, use WordNet or word embeddings)
        synonym_map = {
            "python": ["programming", "code", "script"],
            "machine": ["automated", "artificial", "computer"],
            "learning": ["training", "education", "study"],
            "data": ["information", "dataset", "analytics"]
        }
        
        for token in tokens:
            if token in synonym_map:
                expanded.extend(synonym_map[token])
        
        return expanded

class IndexOptimizer:
    """FAISS index optimization and management"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.indexes: Dict[str, faiss.Index] = {}
        self.index_metadata: Dict[str, Dict] = {}
        self._lock = threading.RLock()
    
    def create_optimized_index(self, vectors: np.ndarray, index_name: str) -> faiss.Index:
        """Create optimized FAISS index based on data characteristics"""
        dimension = vectors.shape[1]
        n_vectors = vectors.shape[0]
        
        logger.info(f"Creating optimized index for {n_vectors} vectors of dimension {dimension}")
        
        if n_vectors < 1000:
            # Use flat index for small datasets
            index = faiss.IndexFlatIP(dimension)
            index_type = "Flat"
        elif n_vectors < 100000:
            # Use IVF for medium datasets
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            index_type = "IVF"
        else:
            # Use IVF-PQ for large datasets
            quantizer = faiss.IndexFlatIP(dimension)
            m = min(64, dimension // 4)  # Number of sub-quantizers
            index = faiss.IndexIVFPQ(quantizer, dimension, self.config.nlist, m, 8)
            index_type = "IVF-PQ"
        
        # Train index if needed
        if hasattr(index, 'is_trained') and not index.is_trained:
            logger.info(f"Training {index_type} index...")
            index.train(vectors)
        
        # Add vectors
        index.add(vectors)
        
        # Store index and metadata
        with self._lock:
            self.indexes[index_name] = index
            self.index_metadata[index_name] = {
                "type": index_type,
                "dimension": dimension,
                "total_vectors": n_vectors,
                "created_at": time.time(),
                "nlist": self.config.nlist if "IVF" in index_type else None
            }
        
        logger.info(f"Created {index_type} index '{index_name}' with {n_vectors} vectors")
        return index
    
    def optimize_search_params(self, index_name: str, target_recall: float = 0.95) -> Dict[str, Any]:
        """Optimize search parameters for target recall"""
        if index_name not in self.indexes:
            raise ValueError(f"Index '{index_name}' not found")
        
        index = self.indexes[index_name]
        metadata = self.index_metadata[index_name]
        
        params = {"nprobe": self.config.nprobe}
        
        if "IVF" in metadata["type"]:
            # Optimize nprobe parameter
            best_nprobe = self.config.nprobe
            
            # Simple heuristic: increase nprobe for higher recall
            if target_recall >= 0.95:
                best_nprobe = min(metadata["nlist"], self.config.nprobe * 2)
            elif target_recall >= 0.90:
                best_nprobe = min(metadata["nlist"], self.config.nprobe * 1.5)
            
            params["nprobe"] = best_nprobe
            
            # Set search parameters
            faiss.ParameterSpace().set_index_parameters(index, f"nprobe={best_nprobe}")
        
        logger.info(f"Optimized search params for '{index_name}': {params}")
        return params
    
    def get_index_stats(self, index_name: str) -> IndexStats:
        """Get detailed index statistics"""
        if index_name not in self.indexes:
            raise ValueError(f"Index '{index_name}' not found")
        
        index = self.indexes[index_name]
        metadata = self.index_metadata[index_name]
        
        # Calculate index size (approximation)
        size_bytes = index.ntotal * metadata["dimension"] * 4  # Assuming float32
        
        return IndexStats(
            index_name=index_name,
            total_documents=index.ntotal,
            index_size_bytes=size_bytes,
            avg_query_time=0.0,  # Would be calculated from query metrics
            cache_hit_rate=0.0,  # Would be calculated from cache stats
            last_updated=metadata.get("created_at", 0.0),
            fragmentation_ratio=0.0  # Would be calculated from index structure
        )

class PerformanceMonitor:
    """Monitor and analyze search performance"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.query_metrics: deque = deque(maxlen=10000)
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self._db_path = "search_performance.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent metrics"""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    execution_time REAL,
                    results_count INTEGER,
                    cache_hit BOOLEAN,
                    index_used TEXT,
                    timestamp REAL,
                    memory_usage INTEGER,
                    cpu_usage REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON query_metrics(timestamp)")
            conn.commit()
            conn.close()
            logger.info("Initialized performance monitoring database")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def record_query(self, metrics: QueryMetrics):
        """Record query performance metrics"""
        self.query_metrics.append(metrics)
        
        # Store in database
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                INSERT INTO query_metrics 
                (query, execution_time, results_count, cache_hit, index_used, 
                 timestamp, memory_usage, cpu_usage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.query, metrics.execution_time, metrics.results_count,
                metrics.cache_hit, metrics.index_used, metrics.timestamp,
                metrics.memory_usage, metrics.cpu_usage
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            # Get metrics from database
            cursor.execute("""
                SELECT execution_time, results_count, cache_hit, memory_usage, cpu_usage
                FROM query_metrics 
                WHERE timestamp > ?
            """, (cutoff_time,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {"message": "No metrics available for the specified time period"}
            
            exec_times = [row[0] for row in rows]
            result_counts = [row[1] for row in rows]
            cache_hits = [row[2] for row in rows]
            memory_usage = [row[3] for row in rows]
            cpu_usage = [row[4] for row in rows]
            
            return {
                "time_period_hours": hours,
                "total_queries": len(rows),
                "avg_execution_time": np.mean(exec_times),
                "p95_execution_time": np.percentile(exec_times, 95),
                "p99_execution_time": np.percentile(exec_times, 99),
                "avg_results_count": np.mean(result_counts),
                "cache_hit_rate": np.mean(cache_hits),
                "avg_memory_usage": np.mean(memory_usage),
                "max_memory_usage": np.max(memory_usage),
                "avg_cpu_usage": np.mean(cpu_usage),
                "queries_over_threshold": sum(1 for t in exec_times if t > self.config.max_query_time)
            }
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """Detect potential performance issues"""
        issues = []
        
        try:
            summary = self.get_performance_summary(1)  # Last hour
            
            if summary.get("avg_execution_time", 0) > self.config.max_query_time:
                issues.append({
                    "type": "slow_queries",
                    "severity": "high",
                    "message": f"Average query time {summary['avg_execution_time']:.3f}s exceeds threshold {self.config.max_query_time}s",
                    "recommendation": "Consider optimizing indexes or increasing cache size"
                })
            
            if summary.get("cache_hit_rate", 1.0) < 0.5:
                issues.append({
                    "type": "low_cache_hit_rate",
                    "severity": "medium",
                    "message": f"Cache hit rate {summary['cache_hit_rate']:.3f} is below 50%",
                    "recommendation": "Increase cache size or adjust TTL settings"
                })
            
            if summary.get("max_memory_usage", 0) > self.config.max_memory_usage:
                issues.append({
                    "type": "high_memory_usage",
                    "severity": "high",
                    "message": f"Memory usage exceeds {self.config.max_memory_usage / 1024**3:.1f}GB",
                    "recommendation": "Optimize index size or increase available memory"
                })
        
        except Exception as e:
            logger.error(f"Failed to detect performance issues: {e}")
        
        return issues

class SearchOptimizer:
    """Main search optimization engine"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.query_processor = QueryProcessor(config)
        self.index_optimizer = IndexOptimizer(config)
        self.performance_monitor = PerformanceMonitor(config)
        
        # Caching
        self._query_cache: Dict[str, Any] = {}
        self._result_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_lock = threading.RLock()
        
        # Auto-optimization
        self._optimization_task = None
        self._running = False
    
    async def optimize_query(self, query: str) -> Dict[str, Any]:
        """Optimize query for better search performance"""
        start_time = time.time()
        
        # Check query cache
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        with self._cache_lock:
            if query_hash in self._query_cache:
                return self._query_cache[query_hash]
        
        # Process query
        processed = self.query_processor.process_query(query)
        
        # Store in cache
        with self._cache_lock:
            if len(self._query_cache) >= self.config.query_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            
            self._query_cache[query_hash] = processed
        
        processing_time = time.time() - start_time
        logger.debug(f"Query optimization took {processing_time:.3f}s")
        
        return processed
    
    async def search_with_optimization(self, query: str, index_name: str, 
                                     top_k: int = 10) -> Tuple[List[Any], QueryMetrics]:
        """Perform optimized search with performance monitoring"""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # Optimize query
        optimized_query = await self.optimize_query(query)
        
        # Check result cache
        cache_key = f"{query}:{index_name}:{top_k}"
        cache_hit = False
        
        with self._cache_lock:
            if cache_key in self._result_cache:
                results, cached_time = self._result_cache[cache_key]
                if time.time() - cached_time < self.config.result_cache_ttl:
                    cache_hit = True
                    execution_time = time.time() - start_time
                    
                    # Record metrics
                    metrics = QueryMetrics(
                        query=query,
                        execution_time=execution_time,
                        results_count=len(results),
                        cache_hit=True,
                        index_used=index_name,
                        timestamp=time.time(),
                        memory_usage=self._get_memory_usage() - memory_before,
                        cpu_usage=0.0
                    )
                    
                    self.performance_monitor.record_query(metrics)
                    return results, metrics
        
        # Perform search
        if index_name not in self.index_optimizer.indexes:
            raise ValueError(f"Index '{index_name}' not found")
        
        index = self.index_optimizer.indexes[index_name]
        
        # Optimize search parameters
        search_params = self.index_optimizer.optimize_search_params(
            index_name, self.config.target_recall
        )
        
        # Simulate search (in practice, this would use the actual vector query)
        # For demonstration, we'll return mock results
        results = [f"result_{i}" for i in range(min(top_k, 10))]
        
        execution_time = time.time() - start_time
        memory_after = self._get_memory_usage()
        
        # Cache results
        with self._cache_lock:
            if len(self._result_cache) >= self.config.query_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[cache_key] = (results, time.time())
        
        # Record metrics
        metrics = QueryMetrics(
            query=query,
            execution_time=execution_time,
            results_count=len(results),
            cache_hit=False,
            index_used=index_name,
            timestamp=time.time(),
            memory_usage=memory_after - memory_before,
            cpu_usage=0.0  # Would calculate actual CPU usage
        )
        
        self.performance_monitor.record_query(metrics)
        
        return results, metrics
    
    def start_auto_optimization(self):
        """Start automatic optimization background task"""
        if self._optimization_task and not self._optimization_task.done():
            logger.warning("Auto-optimization already running")
            return
        
        self._running = True
        self._optimization_task = asyncio.create_task(self._auto_optimization_loop())
        logger.info("Started auto-optimization")
    
    async def _auto_optimization_loop(self):
        """Background task for automatic optimization"""
        while self._running:
            try:
                await asyncio.sleep(self.config.auto_optimize_interval)
                
                # Detect performance issues
                issues = self.performance_monitor.detect_performance_issues()
                
                if issues:
                    logger.info(f"Detected {len(issues)} performance issues")
                    await self._apply_optimizations(issues)
                
                # Clear old cache entries
                await self._cleanup_caches()
                
            except Exception as e:
                logger.error(f"Auto-optimization error: {e}")
    
    async def _apply_optimizations(self, issues: List[Dict[str, Any]]):
        """Apply automatic optimizations based on detected issues"""
        for issue in issues:
            if issue["type"] == "slow_queries":
                # Increase cache sizes
                self.config.query_cache_size = min(self.config.query_cache_size * 2, 5000)
                logger.info("Increased query cache size due to slow queries")
            
            elif issue["type"] == "low_cache_hit_rate":
                # Increase cache TTL
                self.config.result_cache_ttl = min(self.config.result_cache_ttl * 1.5, 7200)
                logger.info("Increased cache TTL due to low hit rate")
            
            elif issue["type"] == "high_memory_usage":
                # Reduce cache sizes
                self.config.query_cache_size = max(self.config.query_cache_size // 2, 100)
                logger.info("Reduced cache size due to high memory usage")
    
    async def _cleanup_caches(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        
        with self._cache_lock:
            # Clean result cache
            expired_keys = [
                key for key, (_, cached_time) in self._result_cache.items()
                if current_time - cached_time > self.config.result_cache_ttl
            ]
            
            for key in expired_keys:
                del self._result_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        import psutil
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0
    
    def stop_auto_optimization(self):
        """Stop automatic optimization"""
        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
        logger.info("Stopped auto-optimization")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        return {
            "config": {
                "query_cache_size": self.config.query_cache_size,
                "result_cache_ttl": self.config.result_cache_ttl,
                "max_query_time": self.config.max_query_time,
                "target_recall": self.config.target_recall
            },
            "cache_stats": {
                "query_cache_entries": len(self._query_cache),
                "result_cache_entries": len(self._result_cache)
            },
            "indexes": {
                name: asdict(self.index_optimizer.get_index_stats(name))
                for name in self.index_optimizer.indexes.keys()
            },
            "performance": self.performance_monitor.get_performance_summary(),
            "issues": self.performance_monitor.detect_performance_issues(),
            "auto_optimization_running": self._running
        }

# Factory function
def create_search_optimizer(
    enable_query_expansion: bool = True,
    faiss_index_type: str = "IVF",
    query_cache_size: int = 1000,
    max_query_time: float = 5.0,
    auto_optimize: bool = True
) -> SearchOptimizer:
    """Factory function to create search optimizer with sensible defaults"""
    
    config = OptimizationConfig(
        enable_query_expansion=enable_query_expansion,
        faiss_index_type=faiss_index_type,
        query_cache_size=query_cache_size,
        max_query_time=max_query_time
    )
    
    optimizer = SearchOptimizer(config)
    
    if auto_optimize:
        optimizer.start_auto_optimization()
    
    return optimizer

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create optimizer
        optimizer = create_search_optimizer()
        
        # Create sample index
        vectors = np.random.random((1000, 128)).astype('float32')
        index = optimizer.index_optimizer.create_optimized_index(vectors, "test_index")
        
        # Perform optimized searches
        queries = ["python programming", "machine learning", "data science"]
        
        for query in queries:
            results, metrics = await optimizer.search_with_optimization(
                query, "test_index", top_k=10
            )
            print(f"Query: {query}")
            print(f"Results: {len(results)}, Time: {metrics.execution_time:.3f}s, Cache: {metrics.cache_hit}")
            print()
        
        # Get optimization stats
        stats = optimizer.get_optimization_stats()
        print("Optimization Stats:")
        print(json.dumps(stats, indent=2, default=str))
        
        # Stop optimizer
        optimizer.stop_auto_optimization()
    
    asyncio.run(main())