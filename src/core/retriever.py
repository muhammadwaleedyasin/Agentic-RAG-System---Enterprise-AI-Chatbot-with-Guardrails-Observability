"""
Retrieval service for finding relevant document chunks.
"""
import asyncio
from typing import List, Dict, Any, Optional
import logging
import numpy as np

from ..config.settings import settings
from ..models.rag import RAGQuery, RAGConfig, RetrievedChunk, RetrievalStrategy
from .vector_store import VectorStore
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class Retriever:
    """Service for retrieving relevant document chunks."""
    
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_service: Embedding service instance
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def retrieve(self, query: RAGQuery) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: RAG query with configuration
            
        Returns:
            List of retrieved chunks
        """
        try:
            # Get configuration with defaults
            config = query.config or RAGConfig()
            
            # Handle empty queries
            if not query.query or not query.query.strip():
                return []
            
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_single(query.query)
            
            # Build search filters
            where_filter = self._build_filters(query.filters)
            
            # Retrieve initial candidates
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=config.top_k * 2,  # Get more candidates for reranking
                where=where_filter
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results
                if result["similarity_score"] >= config.similarity_threshold
            ]
            
            # Apply retrieval strategy
            if config.retrieval_strategy == RetrievalStrategy.MMR:
                final_results = await self._apply_mmr(
                    query_embedding, 
                    filtered_results, 
                    config.top_k,
                    config.diversity_lambda
                )
            elif config.retrieval_strategy == RetrievalStrategy.DIVERSITY:
                final_results = await self._apply_diversity_ranking(
                    filtered_results, 
                    config.top_k
                )
            else:  # SIMILARITY
                final_results = filtered_results[:config.top_k]
            
            # Convert to RetrievedChunk objects
            retrieved_chunks = []
            for i, result in enumerate(final_results):
                chunk = RetrievedChunk(
                    chunk_id=result["chunk_id"],
                    document_id=result["metadata"]["document_id"],
                    content=result["content"],
                    similarity_score=result["similarity_score"],
                    metadata=result["metadata"],
                    rank=i + 1
                )
                retrieved_chunks.append(chunk)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {str(e)}")
            raise
    
    async def retrieve_by_document(self, document_id: str, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve chunks from a specific document.
        
        Args:
            document_id: Document ID to search within
            query: Search query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks from the document
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_single(query)
            
            # Search within specific document
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                where={"document_id": document_id}
            )
            
            # Convert to RetrievedChunk objects
            retrieved_chunks = []
            for i, result in enumerate(search_results):
                chunk = RetrievedChunk(
                    chunk_id=result["chunk_id"],
                    document_id=result["metadata"]["document_id"],
                    content=result["content"],
                    similarity_score=result["similarity_score"],
                    metadata=result["metadata"],
                    rank=i + 1
                )
                retrieved_chunks.append(chunk)
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks from document {document_id}: {str(e)}")
            raise
    
    async def _apply_mmr(
        self, 
        query_embedding: List[float], 
        candidates: List[Dict[str, Any]], 
        top_k: int,
        lambda_param: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance (MMR) for diverse retrieval.
        
        Args:
            query_embedding: Query embedding vector
            candidates: List of candidate results
            top_k: Number of results to select
            lambda_param: Balance between relevance and diversity (0-1)
            
        Returns:
            Reranked results using MMR
        """
        try:
            if not candidates:
                return []
            
            selected = []
            remaining = candidates.copy()
            
            # Select first document (highest similarity)
            if remaining:
                best = max(remaining, key=lambda x: x["similarity_score"])
                selected.append(best)
                remaining.remove(best)
            
            # Select remaining documents using MMR
            while len(selected) < top_k and remaining:
                mmr_scores = []
                
                for candidate in remaining:
                    # Calculate relevance score (similarity to query)
                    relevance = candidate["similarity_score"]
                    
                    # Calculate maximum similarity to already selected documents
                    max_sim_to_selected = 0.0
                    if selected:
                        candidate_embedding = await self._get_embedding_for_result(candidate)
                        for selected_doc in selected:
                            selected_embedding = await self._get_embedding_for_result(selected_doc)
                            similarity = await self.embedding_service.similarity(
                                candidate_embedding, selected_embedding
                            )
                            max_sim_to_selected = max(max_sim_to_selected, similarity)
                    
                    # Calculate MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
                    mmr_scores.append((candidate, mmr_score))
                
                # Select document with highest MMR score
                best_candidate, best_score = max(mmr_scores, key=lambda x: x[1])
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            
            return selected
            
        except Exception as e:
            logger.error(f"Failed to apply MMR: {str(e)}")
            return candidates[:top_k]  # Fallback to simple ranking
    
    async def _apply_diversity_ranking(
        self, 
        candidates: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Apply diversity-based ranking to avoid redundant results.
        
        Args:
            candidates: List of candidate results
            top_k: Number of results to select
            
        Returns:
            Diversified results
        """
        try:
            if not candidates:
                return []
            
            # Group by document to ensure diversity across documents
            doc_groups = {}
            for candidate in candidates:
                doc_id = candidate["metadata"]["document_id"]
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(candidate)
            
            # Select best chunk from each document, round-robin style
            selected = []
            doc_iterators = {doc_id: iter(sorted(chunks, key=lambda x: x["similarity_score"], reverse=True)) 
                           for doc_id, chunks in doc_groups.items()}
            
            while len(selected) < top_k and doc_iterators:
                for doc_id in list(doc_iterators.keys()):
                    try:
                        candidate = next(doc_iterators[doc_id])
                        selected.append(candidate)
                        if len(selected) >= top_k:
                            break
                    except StopIteration:
                        del doc_iterators[doc_id]
                
                if not doc_iterators:
                    break
            
            return selected
            
        except Exception as e:
            logger.error(f"Failed to apply diversity ranking: {str(e)}")
            return candidates[:top_k]  # Fallback to simple ranking
    
    async def _get_embedding_for_result(self, result: Dict[str, Any]) -> List[float]:
        """Get or generate embedding for a search result."""
        # For now, generate embedding from content
        # In a production system, you might cache embeddings
        return await self.embedding_service.embed_single(result["content"])
    
    def _build_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where filters from query filters."""
        if not filters:
            return None
        
        # Map query filters to ChromaDB format
        where_filter = {}
        
        # Document ID filter
        if "document_ids" in filters:
            where_filter["document_id"] = {"$in": filters["document_ids"]}
        
        # Document type filter
        if "document_types" in filters:
            where_filter["document_type"] = {"$in": filters["document_types"]}
        
        # Tags filter
        if "tags" in filters:
            # Assuming tags are stored in metadata
            where_filter["tags"] = {"$in": filters["tags"]}
        
        # Custom metadata filters
        if "metadata" in filters:
            where_filter.update(filters["metadata"])
        
        return where_filter if where_filter else None
    
    async def health_check(self) -> bool:
        """Check if the retriever is healthy."""
        try:
            # Check dependencies
            vector_health = await self.vector_store.health_check()
            embedding_health = await self.embedding_service.health_check()
            
            return vector_health and embedding_health
            
        except Exception:
            return False
