"""
Advanced Reranking Engine for Production-Scale RAG Systems

This module provides sophisticated reranking capabilities with multiple algorithms,
caching, and performance optimization for large-scale document retrieval.
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import hashlib
import pickle
from functools import lru_cache
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import redis
import json

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document representation for reranking"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    initial_score: float = 0.0
    rerank_scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0

@dataclass
class RerankingConfig:
    """Configuration for reranking engine"""
    # Model configurations
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bi_encoder_model: str = "all-MiniLM-L6-v2"
    
    # Performance settings
    batch_size: int = 32
    max_workers: int = 4
    use_gpu: bool = True
    
    # Caching settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    redis_url: Optional[str] = None
    
    # Reranking algorithms
    algorithms: List[str] = field(default_factory=lambda: [
        "cross_encoder", "semantic_similarity", "hybrid_tfidf", "bm25_plus"
    ])
    
    # Weighting for ensemble
    algorithm_weights: Dict[str, float] = field(default_factory=lambda: {
        "cross_encoder": 0.4,
        "semantic_similarity": 0.3,
        "hybrid_tfidf": 0.2,
        "bm25_plus": 0.1
    })
    
    # Performance thresholds
    max_documents: int = 1000
    timeout_seconds: float = 30.0
    min_score_threshold: float = 0.1

class RerankingAlgorithm(ABC):
    """Abstract base class for reranking algorithms"""
    
    @abstractmethod
    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on query"""
        pass
    
    @abstractmethod
    def get_cache_key(self, query: str, doc_ids: List[str]) -> str:
        """Generate cache key for results"""
        pass

class CrossEncoderRanker(RerankingAlgorithm):
    """Cross-encoder based reranking using transformer models"""
    
    def __init__(self, model_name: str, batch_size: int = 32, use_gpu: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model"""
        try:
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            self._model = CrossEncoder(self.model_name, device=device)
            logger.info(f"Loaded cross-encoder model {self.model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise
    
    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using cross-encoder scores"""
        if not self._model:
            raise RuntimeError("Cross-encoder model not loaded")
        
        # Prepare query-document pairs
        pairs = [(query, doc.content) for doc in documents]
        
        # Batch prediction
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self._model.predict(batch_pairs)
            scores.extend(batch_scores)
        
        # Update document scores
        for doc, score in zip(documents, scores):
            doc.rerank_scores["cross_encoder"] = float(score)
        
        return documents
    
    def get_cache_key(self, query: str, doc_ids: List[str]) -> str:
        """Generate cache key for cross-encoder results"""
        content = f"cross_encoder:{self.model_name}:{query}:{':'.join(sorted(doc_ids))}"
        return hashlib.md5(content.encode()).hexdigest()

class SemanticSimilarityRanker(RerankingAlgorithm):
    """Semantic similarity based reranking using bi-encoders"""
    
    def __init__(self, model_name: str, use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """Load bi-encoder model"""
        try:
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Loaded bi-encoder model {self.model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load bi-encoder model: {e}")
            raise
    
    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using semantic similarity"""
        if not self._model:
            raise RuntimeError("Bi-encoder model not loaded")
        
        # Encode query and documents
        query_embedding = self._model.encode([query])
        doc_embeddings = self._model.encode([doc.content for doc in documents])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Update document scores
        for doc, similarity in zip(documents, similarities):
            doc.rerank_scores["semantic_similarity"] = float(similarity)
        
        return documents
    
    def get_cache_key(self, query: str, doc_ids: List[str]) -> str:
        """Generate cache key for semantic similarity results"""
        content = f"semantic:{self.model_name}:{query}:{':'.join(sorted(doc_ids))}"
        return hashlib.md5(content.encode()).hexdigest()

class HybridTFIDFRanker(RerankingAlgorithm):
    """Hybrid TF-IDF based reranking with query expansion"""
    
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer = None
    
    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using hybrid TF-IDF approach"""
        if not documents:
            return documents
        
        # Prepare corpus
        corpus = [query] + [doc.content for doc in documents]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = tfidf_matrix[0]
        doc_vectors = tfidf_matrix[1:]
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        
        # Update document scores
        for doc, similarity in zip(documents, similarities):
            doc.rerank_scores["hybrid_tfidf"] = float(similarity)
        
        return documents
    
    def get_cache_key(self, query: str, doc_ids: List[str]) -> str:
        """Generate cache key for TF-IDF results"""
        content = f"tfidf:{query}:{':'.join(sorted(doc_ids))}"
        return hashlib.md5(content.encode()).hexdigest()

class BM25PlusRanker(RerankingAlgorithm):
    """BM25+ algorithm implementation for document reranking"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()
    
    def _calculate_bm25_plus(self, query_tokens: List[str], doc_tokens: List[str], 
                           avg_doc_length: float) -> float:
        """Calculate BM25+ score"""
        score = 0.0
        doc_length = len(doc_tokens)
        
        for token in query_tokens:
            tf = doc_tokens.count(token)
            if tf > 0:
                # BM25+ formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                score += (numerator / denominator) + self.delta
        
        return score
    
    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using BM25+ algorithm"""
        if not documents:
            return documents
        
        query_tokens = self._tokenize(query)
        doc_tokens_list = [self._tokenize(doc.content) for doc in documents]
        avg_doc_length = sum(len(tokens) for tokens in doc_tokens_list) / len(doc_tokens_list)
        
        # Calculate BM25+ scores
        for doc, doc_tokens in zip(documents, doc_tokens_list):
            score = self._calculate_bm25_plus(query_tokens, doc_tokens, avg_doc_length)
            doc.rerank_scores["bm25_plus"] = score
        
        return documents
    
    def get_cache_key(self, query: str, doc_ids: List[str]) -> str:
        """Generate cache key for BM25+ results"""
        content = f"bm25plus:{query}:{':'.join(sorted(doc_ids))}"
        return hashlib.md5(content.encode()).hexdigest()

class RerankingCache:
    """Caching layer for reranking results"""
    
    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600):
        self.ttl = ttl
        self._redis = None
        self._memory_cache = {}
        
        if redis_url:
            try:
                self._redis = redis.from_url(redis_url)
                self._redis.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, using memory cache")
    
    async def get(self, key: str) -> Optional[List[Document]]:
        """Get cached reranking results"""
        try:
            if self._redis:
                data = self._redis.get(key)
                if data:
                    return pickle.loads(data)
            else:
                return self._memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, documents: List[Document]):
        """Cache reranking results"""
        try:
            if self._redis:
                data = pickle.dumps(documents)
                self._redis.setex(key, self.ttl, data)
            else:
                self._memory_cache[key] = documents
                # Simple TTL for memory cache
                asyncio.create_task(self._expire_memory_cache(key))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def _expire_memory_cache(self, key: str):
        """Expire memory cache entries"""
        await asyncio.sleep(self.ttl)
        self._memory_cache.pop(key, None)

class AdvancedRerankingEngine:
    """Advanced reranking engine with multiple algorithms and optimization"""
    
    def __init__(self, config: RerankingConfig):
        self.config = config
        self.algorithms: Dict[str, RerankingAlgorithm] = {}
        self.cache = RerankingCache(config.redis_url, config.cache_ttl) if config.cache_enabled else None
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "algorithm_usage": {},
            "avg_processing_time": 0.0
        }
        
        self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """Initialize reranking algorithms"""
        if "cross_encoder" in self.config.algorithms:
            self.algorithms["cross_encoder"] = CrossEncoderRanker(
                self.config.cross_encoder_model,
                self.config.batch_size,
                self.config.use_gpu
            )
        
        if "semantic_similarity" in self.config.algorithms:
            self.algorithms["semantic_similarity"] = SemanticSimilarityRanker(
                self.config.bi_encoder_model,
                self.config.use_gpu
            )
        
        if "hybrid_tfidf" in self.config.algorithms:
            self.algorithms["hybrid_tfidf"] = HybridTFIDFRanker()
        
        if "bm25_plus" in self.config.algorithms:
            self.algorithms["bm25_plus"] = BM25PlusRanker()
        
        logger.info(f"Initialized {len(self.algorithms)} reranking algorithms")
    
    async def rerank(self, query: str, documents: List[Document], 
                    algorithms: Optional[List[str]] = None) -> List[Document]:
        """
        Rerank documents using specified algorithms with ensemble scoring
        
        Args:
            query: Search query
            documents: List of documents to rerank
            algorithms: Specific algorithms to use (defaults to config)
        
        Returns:
            List of reranked documents sorted by final score
        """
        start_time = time.time()
        self._stats["total_requests"] += 1
        
        if not documents:
            return documents
        
        # Limit documents if needed
        if len(documents) > self.config.max_documents:
            documents = documents[:self.config.max_documents]
            logger.warning(f"Limited documents to {self.config.max_documents}")
        
        # Use specified algorithms or default
        active_algorithms = algorithms or self.config.algorithms
        
        # Check cache
        doc_ids = [doc.id for doc in documents]
        cache_key = self._get_ensemble_cache_key(query, doc_ids, active_algorithms)
        
        if self.cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self._stats["cache_hits"] += 1
                logger.debug("Cache hit for reranking request")
                return cached_result
        
        # Run reranking algorithms
        tasks = []
        for algo_name in active_algorithms:
            if algo_name in self.algorithms:
                algorithm = self.algorithms[algo_name]
                task = asyncio.create_task(
                    self._run_algorithm_with_timeout(algorithm, query, documents.copy())
                )
                tasks.append((algo_name, task))
                self._stats["algorithm_usage"][algo_name] = self._stats["algorithm_usage"].get(algo_name, 0) + 1
        
        # Wait for all algorithms to complete
        results = []
        for algo_name, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=self.config.timeout_seconds)
                results.append((algo_name, result))
            except asyncio.TimeoutError:
                logger.warning(f"Algorithm {algo_name} timed out")
            except Exception as e:
                logger.error(f"Algorithm {algo_name} failed: {e}")
        
        # Ensemble scoring
        final_documents = self._ensemble_scoring(documents, results)
        
        # Filter by minimum score threshold
        final_documents = [
            doc for doc in final_documents 
            if doc.final_score >= self.config.min_score_threshold
        ]
        
        # Sort by final score
        final_documents.sort(key=lambda x: x.final_score, reverse=True)
        
        # Cache results
        if self.cache:
            await self.cache.set(cache_key, final_documents)
        
        # Update stats
        processing_time = time.time() - start_time
        self._stats["avg_processing_time"] = (
            (self._stats["avg_processing_time"] * (self._stats["total_requests"] - 1) + processing_time) 
            / self._stats["total_requests"]
        )
        
        logger.info(f"Reranked {len(documents)} documents in {processing_time:.3f}s")
        return final_documents
    
    async def _run_algorithm_with_timeout(self, algorithm: RerankingAlgorithm, 
                                        query: str, documents: List[Document]) -> List[Document]:
        """Run reranking algorithm with timeout protection"""
        return await algorithm.rerank(query, documents)
    
    def _ensemble_scoring(self, original_documents: List[Document], 
                         results: List[Tuple[str, List[Document]]]) -> List[Document]:
        """Combine scores from multiple algorithms using weighted ensemble"""
        # Create document lookup
        doc_lookup = {doc.id: doc for doc in original_documents}
        
        # Initialize final scores
        for doc in original_documents:
            doc.final_score = 0.0
        
        # Combine scores from all algorithms
        total_weight = 0.0
        for algo_name, scored_docs in results:
            weight = self.config.algorithm_weights.get(algo_name, 1.0)
            total_weight += weight
            
            # Create score lookup for this algorithm
            score_lookup = {doc.id: doc.rerank_scores.get(algo_name, 0.0) for doc in scored_docs}
            
            # Add weighted scores
            for doc in original_documents:
                score = score_lookup.get(doc.id, 0.0)
                doc.final_score += weight * score
        
        # Normalize by total weight
        if total_weight > 0:
            for doc in original_documents:
                doc.final_score /= total_weight
        
        return original_documents
    
    def _get_ensemble_cache_key(self, query: str, doc_ids: List[str], 
                              algorithms: List[str]) -> str:
        """Generate cache key for ensemble results"""
        content = f"ensemble:{':'.join(sorted(algorithms))}:{query}:{':'.join(sorted(doc_ids))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranking engine statistics"""
        return {
            **self._stats,
            "cache_hit_rate": self._stats["cache_hits"] / max(self._stats["total_requests"], 1),
            "algorithms_loaded": list(self.algorithms.keys()),
            "config": {
                "max_documents": self.config.max_documents,
                "timeout_seconds": self.config.timeout_seconds,
                "min_score_threshold": self.config.min_score_threshold,
                "cache_enabled": self.config.cache_enabled
            }
        }
    
    async def warm_up(self, sample_queries: List[str], sample_documents: List[Document]):
        """Warm up models and cache with sample data"""
        logger.info("Warming up reranking engine...")
        
        for i, query in enumerate(sample_queries[:3]):  # Limit warm-up
            sample_docs = sample_documents[:min(10, len(sample_documents))]
            try:
                await self.rerank(query, sample_docs)
                logger.debug(f"Warm-up query {i+1} completed")
            except Exception as e:
                logger.warning(f"Warm-up query {i+1} failed: {e}")
        
        logger.info("Reranking engine warm-up completed")

# Factory function for easy initialization
def create_reranking_engine(
    algorithms: Optional[List[str]] = None,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    bi_encoder_model: str = "all-MiniLM-L6-v2",
    redis_url: Optional[str] = None,
    max_documents: int = 1000,
    use_gpu: bool = True
) -> AdvancedRerankingEngine:
    """Factory function to create reranking engine with sensible defaults"""
    
    config = RerankingConfig(
        algorithms=algorithms or ["cross_encoder", "semantic_similarity", "hybrid_tfidf"],
        cross_encoder_model=cross_encoder_model,
        bi_encoder_model=bi_encoder_model,
        redis_url=redis_url,
        max_documents=max_documents,
        use_gpu=use_gpu
    )
    
    return AdvancedRerankingEngine(config)

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create sample documents
        documents = [
            Document(id="1", content="Python is a programming language", initial_score=0.8),
            Document(id="2", content="Machine learning with Python libraries", initial_score=0.7),
            Document(id="3", content="Data science and analytics", initial_score=0.6),
        ]
        
        # Create reranking engine
        engine = create_reranking_engine()
        
        # Warm up
        await engine.warm_up(["python programming"], documents)
        
        # Rerank documents
        query = "python machine learning"
        reranked = await engine.rerank(query, documents)
        
        # Print results
        for doc in reranked:
            print(f"Doc {doc.id}: {doc.final_score:.3f} - {doc.content[:50]}...")
        
        # Print stats
        print("\nEngine Stats:", engine.get_stats())
    
    asyncio.run(main())