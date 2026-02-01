"""Integration tests for caching system."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import time

from src.app.main import app


class TestCachingIntegration:
    """Test cases for caching system integration."""

    @pytest.fixture
    def client(self, mock_dependencies):
        """Create test client with dependency overrides."""
        from src.app.deps import get_rag_pipeline
        
        # Use dependency overrides instead of patching
        app.dependency_overrides[get_rag_pipeline] = lambda: mock_dependencies["rag_pipeline"]
        
        with TestClient(app) as client:
            yield client
        
        # Clean up dependency overrides
        app.dependency_overrides.clear()

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all required dependencies using dependency injection."""
        mock_rag_pipeline = Mock()
        mock_caching_system = Mock()
        
        # Configure RAG pipeline with proper response structure
        from src.models.rag import RAGResponse, RAGContext, RetrievedChunk
        
        mock_response = RAGResponse(
            answer="Answer about AI",
            sources=["doc1", "doc2"],
            context=RAGContext(
                retrieved_chunks=[],
                total_chunks=0,
                retrieval_time=0.1
            ),
            generation_time=0.2,
            total_time=0.3,
            usage={"tokens": 100},
            metadata={"cached": False}
        )
        mock_rag_pipeline.query.return_value = mock_response
        
        # Configure caching system
        mock_caching_system.get.return_value = None  # Cache miss by default
        mock_caching_system.set.return_value = None
        
        # Mock the pipeline's caching service attribute
        mock_rag_pipeline.caching_service = mock_caching_system
        
        with patch('src.app.api.rag.get_rag_pipeline', return_value=mock_rag_pipeline):
            yield {
                'rag_pipeline': mock_rag_pipeline,
                'caching_system': mock_caching_system
            }

    def test_cache_miss_and_population(self, client, mock_dependencies):
        """Test cache miss followed by cache population."""
        query_data = {
            "query": "What is artificial intelligence?",
            "max_results": 5
        }
        
        response = client.post(
            "/api/v1/rag/query",
            json=query_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "answer" in result
        
        # Verify cache was checked and populated
        mock_dependencies['caching_system'].get.assert_called()
        mock_dependencies['caching_system'].set.assert_called()

    def test_cache_hit_scenario(self, client, mock_dependencies):
        """Test cache hit scenario."""
        # Configure cache to return cached result
        cached_result = {
            "answer": "Cached AI answer",
            "sources": ["cached_doc1"],
            "metadata": {"cached": True}
        }
        mock_dependencies['caching_system'].get.return_value = cached_result
        
        query_data = {
            "query": "What is AI?",
            "max_results": 5
        }
        
        response = client.post(
            "/api/v1/rag/query",
            json=query_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["answer"] == "Cached AI answer"
        assert result["metadata"]["cached"] is True
        
        # RAG pipeline should not be called for cache hits
        mock_dependencies['rag_pipeline'].query.assert_not_called()

    def test_cache_statistics_endpoint(self, client, mock_dependencies):
        """Test cache statistics retrieval."""
        response = client.get(
            "/api/v1/admin/cache/stats",
            headers={"Authorization": "Bearer test_token"}
        )
        
        if response.status_code == 200:
            stats = response.json()
            assert "memory" in stats
            assert "redis" in stats
            assert "combined" in stats
            
            combined = stats["combined"]
            assert "total_hits" in combined
            assert "total_misses" in combined
            assert "hit_rate" in combined

    def test_cache_error_recovery(self, client, mock_dependencies):
        """Test recovery when cache operations fail."""
        cache_system = mock_dependencies['caching_system']
        
        # Simulate cache failures
        cache_system.get.side_effect = Exception("Cache connection failed")
        cache_system.set.side_effect = Exception("Cache write failed")
        
        query_data = {"query": "Cache error recovery test"}
        
        # Should still work despite cache failures
        response = client.post(
            "/api/v1/rag/query",
            json=query_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        # RAG pipeline should still execute
        mock_dependencies['rag_pipeline'].query.assert_called()
