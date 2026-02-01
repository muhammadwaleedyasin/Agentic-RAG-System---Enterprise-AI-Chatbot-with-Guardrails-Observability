"""Retrieval engine with support for vector search and reranking."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import cohere
from rank_bm25 import BM25Okapi

from ..config.settings import get_settings
from ..embeddings.providers import EmbeddingService
from ..models.base import (
    Chunk,
    RetrievalConfig,
    SearchQuery,
    SearchResult,
    VectorDBConfig,
)
from ..utils.exceptions import RetrievalError
from ..utils.logging import log_search, RAGLogger
from ..vector_db.interfaces import BaseVectorDB, VectorDBFactory


class BM25Reranker:
    """BM25-based reranking for hybrid search."""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
    
    def fit(self, documents: List[str]) -> None:
        """Fit BM25 on document collection."""
        self.documents = documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def score(self, query: str, top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """Score documents against query using BM25."""
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores


class CohereReranker:
    """Cohere-based neural reranking."""
    
    def __init__(self, model: str = "rerank-english-v2.0", api_key: Optional[str] = None):
        self.model = model
        self.client = cohere.Client(api_key)
    
    async def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank documents using Cohere API."""
        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_k=top_k or len(documents)
            )
            
            return [(result.index, result.relevance_score) for result in response.results]
            
        except Exception as e:
            raise RetrievalError(f"Cohere reranking failed: {str(e)}")


class HybridSearchEngine:
    """Hybrid search combining vector similarity and BM25."""
    
    def __init__(
        self, 
        vector_db: BaseVectorDB,
        embedding_service: EmbeddingService,
        config: RetrievalConfig
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.config = config
        self.bm25_reranker = BM25Reranker()
        self.cohere_reranker = None
        
        # Initialize Cohere reranker if enabled
        if config.enable_reranking:
            try:
                self.cohere_reranker = CohereReranker(config.rerank_model)
            except Exception:
                # Fall back to BM25 if Cohere is not available
                pass
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        enable_reranking: Optional[bool] = None
    ) -> List[SearchResult]:
        """Perform hybrid search with optional reranking."""
        
        if top_k is None:
            top_k = self.config.top_k
        
        if enable_reranking is None:
            enable_reranking = self.config.enable_reranking
        
        start_time = time.time()
        
        try:
            # Step 1: Vector similarity search
            with RAGLogger("vector_search", {"query": query, "top_k": top_k}):
                query_embedding = await self.embedding_service.embed_single_text(query)
                
                vector_results = await self.vector_db.search_similar(
                    query_embedding=query_embedding,
                    top_k=top_k * 2,  # Get more results for reranking
                    filters=filters
                )
            
            # Step 2: Hybrid scoring (if enabled)
            if self.config.hybrid_search and vector_results:
                with RAGLogger("hybrid_scoring", {"result_count": len(vector_results)}):
                    vector_results = await self._apply_hybrid_scoring(
                        query, vector_results
                    )
            
            # Step 3: Apply similarity threshold
            filtered_results = [
                (chunk, score) for chunk, score in vector_results
                if score >= self.config.similarity_threshold
            ]
            
            # Step 4: Reranking (if enabled)
            if enable_reranking and len(filtered_results) > 1:
                with RAGLogger("reranking", {"candidate_count": len(filtered_results)}):
                    filtered_results = await self._apply_reranking(
                        query, filtered_results, self.config.rerank_top_k
                    )
            
            # Step 5: Convert to SearchResult objects
            search_results = []
            for rank, (chunk, score) in enumerate(filtered_results[:top_k]):
                search_result = SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=rank + 1
                )
                search_results.append(search_result)
            
            duration = time.time() - start_time
            
            log_search(
                query=query,
                result_count=len(search_results),
                duration=duration,
                filters=filters,
                reranked=enable_reranking
            )
            
            return search_results
            
        except Exception as e:
            raise RetrievalError(f"Hybrid search failed: {str(e)}")
    
    async def _apply_hybrid_scoring(
        self,
        query: str,
        vector_results: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        """Apply hybrid scoring combining vector similarity and BM25."""
        
        # Extract documents for BM25
        documents = [chunk.content for chunk, _ in vector_results]
        
        # Fit BM25 on retrieved documents
        self.bm25_reranker.fit(documents)
        
        # Get BM25 scores
        bm25_scores = self.bm25_reranker.score(query)
        bm25_score_dict = {idx: score for idx, score in bm25_scores}
        
        # Normalize scores to [0, 1]
        if bm25_scores:
            max_bm25_score = max(score for _, score in bm25_scores)
            if max_bm25_score > 0:
                bm25_score_dict = {
                    idx: score / max_bm25_score 
                    for idx, score in bm25_score_dict.items()
                }
        
        # Combine scores
        hybrid_results = []
        for idx, (chunk, vector_score) in enumerate(vector_results):
            bm25_score = bm25_score_dict.get(idx, 0.0)
            
            # Weighted combination
            hybrid_score = (
                (1 - self.config.bm25_weight) * vector_score +
                self.config.bm25_weight * bm25_score
            )
            
            hybrid_results.append((chunk, hybrid_score))
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_results
    
    async def _apply_reranking(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """Apply neural reranking to candidate results."""
        
        if len(candidates) <= 1:
            return candidates
        
        documents = [chunk.content for chunk, _ in candidates]
        
        try:
            # Try Cohere reranking first
            if self.cohere_reranker:
                rerank_results = await self.cohere_reranker.rerank(
                    query=query,
                    documents=documents,
                    top_k=top_k
                )
                
                # Apply rerank scores
                reranked_results = []
                for original_idx, rerank_score in rerank_results:
                    chunk, original_score = candidates[original_idx]
                    
                    # Create new result with rerank score
                    search_result = (chunk, rerank_score)
                    reranked_results.append(search_result)
                
                return reranked_results
            
            else:
                # Fall back to BM25 reranking
                bm25_scores = self.bm25_reranker.score(query, top_k)
                
                reranked_results = []
                for original_idx, bm25_score in bm25_scores:
                    if original_idx < len(candidates):
                        chunk, original_score = candidates[original_idx]
                        reranked_results.append((chunk, bm25_score))
                
                return reranked_results
        
        except Exception:
            # If reranking fails, return original results
            return candidates[:top_k]


class RetrievalEngine:
    """Main retrieval engine orchestrating search and reranking."""
    
    def __init__(
        self,
        vector_db_config: VectorDBConfig,
        retrieval_config: RetrievalConfig
    ):
        self.vector_db_config = vector_db_config
        self.retrieval_config = retrieval_config
        self.settings = get_settings()
        
        # Initialize components
        self.vector_db = VectorDBFactory.create_vector_db(vector_db_config)
        self.embedding_service = EmbeddingService()
        self.search_engine = None
    
    async def initialize(self) -> None:
        """Initialize the retrieval engine."""
        try:
            # Connect to vector database
            await self.vector_db.connect()
            
            # Initialize search engine
            self.search_engine = HybridSearchEngine(
                vector_db=self.vector_db,
                embedding_service=self.embedding_service,
                config=self.retrieval_config
            )
            
        except Exception as e:
            raise RetrievalError(f"Failed to initialize retrieval engine: {str(e)}")
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search using the retrieval engine."""
        if not self.search_engine:
            await self.initialize()
        
        try:
            return await self.search_engine.search(
                query=query.query,
                top_k=query.top_k,
                filters=query.filters,
                enable_reranking=query.rerank
            )
            
        except Exception as e:
            raise RetrievalError(f"Search failed: {str(e)}")
    
    async def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the vector database."""
        if not self.vector_db.client:
            await self.vector_db.connect()
        
        try:
            # Generate embeddings for chunks
            texts = [chunk.content for chunk in chunks]
            embedding_response = await self.embedding_service.embed_texts(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embedding_response.embeddings):
                chunk.embedding = embedding
            
            # Store chunks in vector database
            await self.vector_db.upsert_chunks(chunks)
            
        except Exception as e:
            raise RetrievalError(f"Failed to add chunks: {str(e)}")
    
    async def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks from the vector database."""
        try:
            await self.vector_db.delete_chunks(chunk_ids)
            
        except Exception as e:
            raise RetrievalError(f"Failed to delete chunks: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get retrieval engine statistics."""
        try:
            vector_stats = await self.vector_db.get_collection_stats()
            
            return {
                "vector_db": vector_stats,
                "embedding_provider": self.settings.embeddings.default_provider,
                "retrieval_config": {
                    "top_k": self.retrieval_config.top_k,
                    "rerank_top_k": self.retrieval_config.rerank_top_k,
                    "similarity_threshold": self.retrieval_config.similarity_threshold,
                    "enable_reranking": self.retrieval_config.enable_reranking,
                    "hybrid_search": self.retrieval_config.hybrid_search
                }
            }
            
        except Exception as e:
            raise RetrievalError(f"Failed to get stats: {str(e)}")
    
    async def close(self) -> None:
        """Close the retrieval engine and cleanup resources."""
        if self.vector_db:
            await self.vector_db.disconnect()