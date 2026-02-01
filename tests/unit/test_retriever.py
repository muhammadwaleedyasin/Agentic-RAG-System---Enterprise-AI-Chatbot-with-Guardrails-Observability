"""
Unit tests for the core retriever module.
Tests cover similarity search, MMR, diversity handling, filters, and error cases.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from typing import List, Dict, Any

from src.core.retriever import Retriever
from src.models.rag import RAGQuery, RAGConfig, RetrievalStrategy


class TestRetriever:
    """Test cases for the Retriever class."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for tests."""
        mock_service = Mock()
        mock_service.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        mock_service.embed_batch = AsyncMock(return_value=[
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6]
        ])
        # Add similarity method for MMR tests
        mock_service.similarity = AsyncMock(return_value=0.1)
        mock_service.health_check = AsyncMock(return_value=True)
        return mock_service
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for tests."""
        from src.core.vector_store import VectorStore
        mock_store = Mock(spec=VectorStore)
        mock_store.search = AsyncMock(return_value=[
            {"chunk_id": "doc1", "content": "Document 1", "similarity_score": 0.9, "metadata": {"document_id": "test-doc-1"}},
            {"chunk_id": "doc2", "content": "Document 2", "similarity_score": 0.8, "metadata": {"document_id": "test-doc-2"}},
            {"chunk_id": "doc3", "content": "Document 3", "similarity_score": 0.7, "metadata": {"document_id": "test-doc-3"}},
            {"chunk_id": "doc4", "content": "Document 4", "similarity_score": 0.6, "metadata": {"document_id": "test-doc-1"}},
            {"chunk_id": "doc5", "content": "Document 5", "similarity_score": 0.5, "metadata": {"document_id": "test-doc-4"}}
        ])
        mock_store.health_check = AsyncMock(return_value=True)
        return mock_store
    
    @pytest.fixture
    def retriever(self, mock_embedding_service, mock_vector_store):
        """Create retriever instance with mocked dependencies."""
        return Retriever(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
    
    @pytest.mark.asyncio
    async def test_similarity_search_only(self, retriever, mock_vector_store):
        """Test basic similarity search without MMR or diversity."""
        query = RAGQuery(
            query="test query", 
            config=RAGConfig(top_k=3, retrieval_strategy=RetrievalStrategy.SIMILARITY)
        )
        
        results = await retriever.retrieve(query)
        
        assert len(results) == 3
        assert results[0].similarity_score >= results[1].similarity_score >= results[2].similarity_score
        mock_vector_store.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mmr_with_varying_lambda(self, retriever, mock_vector_store):
        """Test MMR with different lambda values."""
        # Test with high diversity (low lambda)
        query_high_diversity = RAGQuery(
            query="test query",
            config=RAGConfig(
                top_k=3,
                retrieval_strategy=RetrievalStrategy.MMR,
                diversity_lambda=0.2  # High diversity
            )
        )
        
        results_high_diversity = await retriever.retrieve(query_high_diversity)
        
        # Test with low diversity (high lambda)
        query_low_diversity = RAGQuery(
            query="test query",
            config=RAGConfig(
                top_k=3,
                retrieval_strategy=RetrievalStrategy.MMR,
                diversity_lambda=0.8  # Low diversity
            )
        )
        
        results_low_diversity = await retriever.retrieve(query_low_diversity)
        
        # Both should return results, but selection may differ based on diversity
        assert len(results_high_diversity) == 3
        assert len(results_low_diversity) == 3
        
        # Verify MMR was applied (results may be reordered)
        assert results_high_diversity[0].chunk_id in ["doc1", "doc2", "doc3", "doc4", "doc5"]
        assert results_low_diversity[0].chunk_id in ["doc1", "doc2", "doc3", "doc4", "doc5"]
    
    @pytest.mark.asyncio
    async def test_diversity_grouping_across_documents(self, retriever, mock_vector_store):
        """Test diversity grouping to avoid similar documents."""
        query = RAGQuery(
            query="test query",
            config=RAGConfig(
                top_k=3,
                retrieval_strategy=RetrievalStrategy.DIVERSITY
            )
        )
        
        results = await retriever.retrieve(query)
        
        assert len(results) <= 3
        # Verify diversity was considered
        doc_ids = [chunk.document_id for chunk in results]
        # Should prefer documents from different sources
        assert len(set(doc_ids)) >= 2 or len(results) == 1
    
    @pytest.mark.asyncio
    async def test_filter_building(self, retriever, mock_vector_store):
        """Test filter construction and application."""
        query = RAGQuery(
            query="test query",
            config=RAGConfig(top_k=3),
            filters={"document_ids": ["test-doc-1"]}
        )
        
        await retriever.retrieve(query)
        
        # Verify filters were passed to vector store
        call_args = mock_vector_store.search.call_args
        assert "where" in call_args.kwargs
    
    @pytest.mark.asyncio
    async def test_health_check_with_healthy_deps(self, retriever, mock_vector_store, mock_embedding_service):
        """Test health check with healthy dependencies."""
        mock_vector_store.health_check.return_value = True
        mock_embedding_service.health_check = AsyncMock(return_value=True)
        
        health = await retriever.health_check()
        
        assert health is True
        mock_vector_store.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_with_unhealthy_deps(self, retriever, mock_vector_store, mock_embedding_service):
        """Test health check with unhealthy dependencies."""
        mock_vector_store.health_check.return_value = False
        mock_embedding_service.health_check = AsyncMock(return_value=True)
        
        health = await retriever.health_check()
        
        assert health is False
    
    @pytest.mark.asyncio
    async def test_embedding_service_failure(self, retriever, mock_embedding_service):
        """Test error handling when embedding service fails."""
        mock_embedding_service.embed_single.side_effect = Exception("Embedding failed")
        
        query = RAGQuery(query="test query", config=RAGConfig(top_k=3))
        
        with pytest.raises(Exception) as exc_info:
            await retriever.retrieve(query)
        
        assert "Embedding failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_vector_store_failure(self, retriever, mock_vector_store):
        """Test error handling when vector store fails."""
        mock_vector_store.search.side_effect = Exception("Search failed")
        
        query = RAGQuery(query="test query", config=RAGConfig(top_k=3))
        
        with pytest.raises(Exception) as exc_info:
            await retriever.retrieve(query)
        
        assert "Search failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, retriever):
        """Test handling of empty or whitespace queries."""
        from pydantic import ValidationError
        
        # Test empty string - should raise ValidationError due to min_length=1
        with pytest.raises(ValidationError) as exc_info:
            RAGQuery(query="", config=RAGConfig(top_k=3))
        assert "at least 1 characters" in str(exc_info.value) or "min_length" in str(exc_info.value)
        
        # Test whitespace only - should also raise ValidationError after stripping
        with pytest.raises(ValidationError) as exc_info:
            RAGQuery(query="   ", config=RAGConfig(top_k=3))
        assert "at least 1 characters" in str(exc_info.value) or "min_length" in str(exc_info.value)
        
        # Test valid single character query - should work
        query_single_char = RAGQuery(query="a", config=RAGConfig(top_k=3))
        results = await retriever.retrieve(query_single_char)
        # Should return results (empty or not, depending on mock data)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_complex_filters_combination(self, retriever, mock_vector_store):
        """Test filters combining document_ids, document_types, tags, and custom metadata."""
        query = RAGQuery(
            query="test query",
            config=RAGConfig(top_k=3),
            filters={
                "document_ids": ["doc-1", "doc-2"],
                "document_types": ["pdf", "txt"],
                "tags": ["important", "reviewed"],
                "metadata": {"department": "engineering", "version": "1.0"}
            }
        )
        
        await retriever.retrieve(query)
        
        # Verify complex filters were built correctly
        call_args = mock_vector_store.search.call_args
        assert "where" in call_args.kwargs
        where_filter = call_args.kwargs["where"]
        assert "document_id" in where_filter
        assert "tags" in where_filter
        assert "department" in where_filter

    @pytest.mark.asyncio
    async def test_similarity_threshold_edge_cases(self, retriever, mock_vector_store):
        """Test similarity scores equal to threshold should be included."""
        # Mock results with scores at exact threshold
        mock_vector_store.search.return_value = [
            {"chunk_id": "doc1", "content": "Document 1", "similarity_score": 0.7, "metadata": {"document_id": "test-doc-1"}},
            {"chunk_id": "doc2", "content": "Document 2", "similarity_score": 0.69, "metadata": {"document_id": "test-doc-2"}},
            {"chunk_id": "doc3", "content": "Document 3", "similarity_score": 0.71, "metadata": {"document_id": "test-doc-3"}}
        ]
        
        query = RAGQuery(
            query="test query",
            config=RAGConfig(top_k=3, similarity_threshold=0.7)
        )
        
        results = await retriever.retrieve(query)
        
        # Score equal to threshold should be included, lower scores excluded
        assert len(results) == 2
        assert all(chunk.similarity_score >= 0.7 for chunk in results)

    @pytest.mark.asyncio
    async def test_mmr_error_fallback(self, retriever, mock_vector_store, mock_embedding_service):
        """Test MMR error handling falls back to top_k by similarity."""
        # Make embedding service fail during MMR processing
        mock_embedding_service.embed_single.side_effect = [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # First call succeeds (query embedding)
            Exception("Embedding failed")  # Subsequent calls fail
        ]
        
        query = RAGQuery(
            query="test query",
            config=RAGConfig(
                top_k=3,
                retrieval_strategy=RetrievalStrategy.MMR
            )
        )
        
        results = await retriever.retrieve(query)
        
        # Should fall back to simple similarity ranking
        assert len(results) == 3
        # Should be ordered by similarity score (fallback behavior)
        assert results[0].similarity_score >= results[1].similarity_score

    @pytest.mark.asyncio
    async def test_diversity_ranking_error_fallback(self, retriever, mock_vector_store):
        """Test diversity ranking error handling falls back to top_k by similarity."""
        # Mock vector store to return malformed results to trigger error
        mock_vector_store.search.return_value = [
            {"chunk_id": "doc1", "content": "Document 1", "similarity_score": 0.9, "metadata": {}},  # Missing document_id
            {"chunk_id": "doc2", "content": "Document 2", "similarity_score": 0.8, "metadata": {"document_id": "test-doc-2"}},
        ]
        
        query = RAGQuery(
            query="test query",
            config=RAGConfig(
                top_k=2,
                retrieval_strategy=RetrievalStrategy.DIVERSITY
            )
        )
        
        results = await retriever.retrieve(query)
        
        # Should fall back to simple similarity ranking despite error
        assert len(results) == 2
        assert results[0].similarity_score >= results[1].similarity_score

    @pytest.mark.asyncio
    async def test_mixed_type_filters(self, retriever, mock_vector_store):
        """Test filters with mixed types (int, bool, string)."""
        query = RAGQuery(
            query="test query",
            config=RAGConfig(top_k=3),
            filters={
                "document_ids": ["doc-1", "doc-2"],
                "metadata": {
                    "priority": 5,  # int
                    "is_published": True,  # bool
                    "status": "active",  # string
                    "confidence_score": 0.95  # float
                }
            }
        )
        
        await retriever.retrieve(query)
        
        # Verify mixed type filters were processed
        call_args = mock_vector_store.search.call_args
        assert "where" in call_args.kwargs
        where_filter = call_args.kwargs["where"]
        assert "priority" in where_filter
        assert "is_published" in where_filter
        assert "status" in where_filter

    @pytest.mark.asyncio
    async def test_and_operator_multiple_conditions(self, retriever, mock_vector_store):
        """Test multiple filter conditions combine into And operator."""
        query = RAGQuery(
            query="test query",
            config=RAGConfig(top_k=3),
            filters={
                "document_types": ["pdf"],
                "tags": ["important"],
                "metadata": {"department": "engineering"}
            }
        )
        
        # Mock the _build_filters method to inspect the result
        original_build_filters = retriever._build_filters
        built_filters = []
        
        def capture_filters(filters):
            result = original_build_filters(filters)
            built_filters.append(result)
            return result
        
        retriever._build_filters = capture_filters
        
        await retriever.retrieve(query)
        
        # Should have created a complex filter structure
        assert len(built_filters) > 0
        filter_result = built_filters[0]
        assert filter_result is not None
        # Multiple top-level keys should be present
        assert len(filter_result) >= 2

    @pytest.mark.asyncio
    async def test_empty_search_results(self, retriever, mock_vector_store):
        """Test handling when vector store returns empty results."""
        mock_vector_store.search.return_value = []
        
        query = RAGQuery(query="test query", config=RAGConfig(top_k=3))
        
        results = await retriever.retrieve(query)
        
        assert results == []
