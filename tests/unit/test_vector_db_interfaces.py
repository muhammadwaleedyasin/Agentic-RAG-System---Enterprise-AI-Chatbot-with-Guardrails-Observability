"""
Unit tests for vector database interfaces and factory.
Tests provider-specific implementations, filter conversion, and error handling.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.vector_db.interfaces import (
    VectorDBFactory, 
    BaseVectorDB,
    WeaviateVectorDB,
    PineconeVectorDB
)
from src.models.base import VectorDBConfig


class TestVectorDBFactory:
    """Test cases for VectorDBFactory."""
    
    def test_create_weaviate_provider(self):
        """Test creating Weaviate provider."""
        config = VectorDBConfig(provider="weaviate", connection_string="localhost:8080")
        
        with patch('src.vector_db.interfaces.weaviate'):
            db = VectorDBFactory.create_vector_db(config)
            assert isinstance(db, WeaviateVectorDB)
    
    def test_create_pinecone_provider(self):
        """Test creating Pinecone provider."""
        config = VectorDBConfig(provider="pinecone", connection_string="us-west1-gcp")
        config.config = {"api_key": "test-key", "index_name": "test-index"}
        
        with patch('src.vector_db.interfaces.pinecone'):
            db = VectorDBFactory.create_vector_db(config)
            assert isinstance(db, PineconeVectorDB)
    
    def test_create_unsupported_provider(self):
        """Test creating unsupported provider raises Pydantic validation error."""
        from pydantic import ValidationError
        
        # Test Pydantic validation error for unsupported provider
        with pytest.raises(ValidationError) as exc_info:
            VectorDBConfig(provider="unsupported", connection_string="test")
        
        # Should contain validation error message about enum values
        assert "Input should be" in str(exc_info.value) or "is not a valid enumeration member" in str(exc_info.value)
    
    def test_factory_error_for_unsupported_enum_value(self):
        """Test VectorDBFactory.create_vector_db raises VectorDBError for unsupported enum value."""
        from src.utils.exceptions import VectorDBError
        from src.models.base import VectorDBProvider
        
        # Create a valid config but temporarily remove the provider from factory
        config = VectorDBConfig(provider="weaviate", connection_string="test")
        
        # Temporarily patch the factory to have empty providers to simulate unsupported provider
        original_providers = VectorDBFactory._providers.copy()
        VectorDBFactory._providers = {}
        
        try:
            with pytest.raises(VectorDBError) as exc_info:
                VectorDBFactory.create_vector_db(config)
            
            assert "Unknown vector database provider" in str(exc_info.value)
            
        finally:
            # Restore original providers
            VectorDBFactory._providers = original_providers
    
    def test_get_available_providers(self):
        """Test VectorDBFactory.get_available_providers returns correct list."""
        providers = VectorDBFactory.get_available_providers()
        
        # Should return the registered providers
        assert isinstance(providers, list)
        assert "weaviate" in providers
        assert "pinecone" in providers
        assert len(providers) >= 2  # At least weaviate and pinecone


class TestWeaviateVectorDB:
    """Test cases for Weaviate vector database implementation."""
    
    @pytest.fixture
    def weaviate_config(self):
        """Weaviate configuration for tests."""
        config = VectorDBConfig(provider="weaviate", connection_string="localhost:8080")
        config.config = {"class_name": "TestDocument"}
        return config
    
    @pytest.fixture
    def mock_weaviate_client(self):
        """Mock Weaviate client."""
        with patch('src.vector_db.interfaces.weaviate.Client', create=True) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock schema operations
            mock_instance.schema.get.return_value = {"classes": []}
            mock_instance.schema.create_class = Mock()
            
            # Mock query builder
            mock_query = Mock()
            mock_instance.query = mock_query
            mock_query.get.return_value = mock_query
            mock_query.with_near_vector.return_value = mock_query
            mock_query.with_limit.return_value = mock_query
            mock_query.with_additional.return_value = mock_query
            mock_query.with_where.return_value = mock_query
            
            yield mock_instance
    
    def test_init_with_config(self, weaviate_config, mock_weaviate_client):
        """Test initialization with configuration."""
        db = WeaviateVectorDB(weaviate_config)
        
        assert db.class_name == "TestDocument"
        assert db.config == weaviate_config
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, weaviate_config, mock_weaviate_client):
        """Test search with filter conversion."""
        # Mock search response with correct structure
        mock_weaviate_client.query.get.return_value.with_near_vector.return_value.with_limit.return_value.with_additional.return_value.do.return_value = {
            "data": {
                "Get": {
                    "TestDocument": [
                        {
                            "content": "Test document",
                            "document_id": "test-doc-1",
                            "chunk_index": 0,
                            "start_char": 0,
                            "end_char": 100,
                            "app": "test-app",
                            "version": "1.0",
                            "audience": "internal",
                            "department": "engineering",
                            "sensitivity": "internal",
                            "tags": ["test"],
                            "_additional": {"id": "doc1", "distance": 0.1}
                        }
                    ]
                }
            }
        }
        
        db = WeaviateVectorDB(weaviate_config)
        db.client = mock_weaviate_client
        
        filters = {"department": "engineering", "app": "test-app"}
        results = await db.search_similar([0.1, 0.2, 0.3], top_k=5, filters=filters)
        
        assert len(results) == 1
        chunk, score = results[0]
        assert chunk.id == "doc1"
        assert score == 0.9  # 1 - distance


class TestProviderSpecificErrorHandling:
    """Test error handling across different providers."""
    
    @pytest.mark.asyncio
    async def test_weaviate_connection_error(self):
        """Test Weaviate connection error handling."""
        from src.utils.exceptions import VectorDBError
        
        config = VectorDBConfig(provider="weaviate", connection_string="invalid:8080")
        
        with patch('src.vector_db.interfaces.weaviate.Client', create=True) as mock_client:
            mock_client.side_effect = Exception("Connection failed")
            
            db = WeaviateVectorDB(config)
            
            with pytest.raises(VectorDBError) as exc_info:
                await db.connect()
            
            assert "Failed to connect to Weaviate" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_pinecone_connection_error(self):
        """Test Pinecone connection error handling."""
        from src.utils.exceptions import VectorDBError
        
        config = VectorDBConfig(provider="pinecone", connection_string="us-west1-gcp")
        config.config = {"api_key": "invalid-key", "index_name": "test-index"}
        
        with patch('src.vector_db.interfaces.pinecone.init', create=True) as mock_init:
            mock_init.side_effect = Exception("Authentication failed")
            
            db = PineconeVectorDB(config)
            
            with pytest.raises(VectorDBError) as exc_info:
                await db.connect()
            
            assert "Failed to connect to Pinecone" in str(exc_info.value)


class TestVectorDBSearchMethods:
    """Test search method compatibility across providers."""
    
    @pytest.mark.asyncio
    async def test_weaviate_search_similar_method(self):
        """Test that Weaviate uses search_similar method."""
        config = VectorDBConfig(provider="weaviate", connection_string="localhost:8080")
        config.config = {"class_name": "TestDocument"}
        
        with patch('src.vector_db.interfaces.weaviate.Client', create=True) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock query response
            mock_instance.query.get.return_value.with_near_vector.return_value.with_limit.return_value.with_additional.return_value.do.return_value = {
                "data": {"Get": {"TestDocument": []}}
            }
            
            db = WeaviateVectorDB(config)
            db.client = mock_instance
            
            results = await db.search_similar([0.1, 0.2, 0.3], top_k=5)
            
            assert isinstance(results, list)
            # Verify the query was built correctly
            mock_instance.query.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pinecone_search_similar_method(self):
        """Test that Pinecone uses search_similar method."""
        config = VectorDBConfig(provider="pinecone", connection_string="us-west1-gcp")
        config.config = {"api_key": "test-key", "index_name": "test-index"}
        
        with patch('src.vector_db.interfaces.pinecone', create=True) as mock_pinecone:
            mock_index = Mock()
            mock_pinecone.Index.return_value = mock_index
            mock_index.query.return_value.matches = []
            
            db = PineconeVectorDB(config)
            db.index = mock_index
            
            results = await db.search_similar([0.1, 0.2, 0.3], top_k=5)
            
            assert isinstance(results, list)
            mock_index.query.assert_called_once()
