"""Unit tests for EmbeddingService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.core.embedding_service import EmbeddingService
from src.utils.exceptions import EmbeddingError


class TestEmbeddingService:
    """Test cases for EmbeddingService."""

    @pytest.fixture
    def embedding_service(self):
        """Create EmbeddingService instance for testing."""
        with patch('src.core.embedding_service.SentenceTransformer'):
            service = EmbeddingService(model_name="all-MiniLM-L6-v2")
            yield service

    @pytest.mark.unit
    def test_embedding_service_initialization(self, embedding_service):
        """Test embedding service initialization."""
        assert embedding_service.model_name == "all-MiniLM-L6-v2"
        assert embedding_service.model is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embed_text_success(self, embedding_service):
        """Test successful text embedding."""
        # Mock the model encode method
        mock_embedding = np.array([[0.1, 0.2, 0.3, 0.4]])
        embedding_service.model.encode = MagicMock(return_value=mock_embedding)
        
        text = "This is a test document."
        result = await embedding_service.embed_text(text)
        
        assert isinstance(result, list)
        assert len(result) == 4
        assert result == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embed_text_list_success(self, embedding_service):
        """Test successful embedding of text list."""
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        embedding_service.model.encode = MagicMock(return_value=mock_embeddings)
        
        texts = ["First document.", "Second document."]
        result = await embedding_service.embed_texts(texts)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embed_text_empty_string(self, embedding_service):
        """Test embedding empty string raises error."""
        with pytest.raises(EmbeddingError, match="Text cannot be empty"):
            await embedding_service.embed_text("")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embed_text_none(self, embedding_service):
        """Test embedding None raises error."""
        with pytest.raises(EmbeddingError, match="Text cannot be None"):
            await embedding_service.embed_text(None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embed_text_model_error(self, embedding_service):
        """Test handling of model encoding errors."""
        embedding_service.model.encode = MagicMock(side_effect=Exception("Model error"))
        
        with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
            await embedding_service.embed_text("test text")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embed_texts_batch_processing(self, embedding_service):
        """Test batch processing of multiple texts."""
        # Mock large batch processing
        texts = [f"Document {i}" for i in range(100)]
        mock_embeddings = np.random.rand(100, 384)
        embedding_service.model.encode = MagicMock(return_value=mock_embeddings)
        
        result = await embedding_service.embed_texts(texts, batch_size=32)
        
        assert len(result) == 100
        assert all(len(emb) == 384 for emb in result)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_embedding_dimension(self, embedding_service):
        """Test getting embedding dimension."""
        # Mock model encode for dimension test
        mock_embedding = np.array([[0.1] * 384])
        embedding_service.model.encode = MagicMock(return_value=mock_embedding)
        
        dimension = await embedding_service.get_embedding_dimension()
        assert dimension == 384

    @pytest.mark.unit
    def test_embedding_service_model_loading_error(self):
        """Test handling of model loading errors."""
        with patch('src.core.embedding_service.SentenceTransformer') as mock_st:
            mock_st.side_effect = Exception("Model loading failed")
            
            with pytest.raises(EmbeddingError, match="Failed to load embedding model"):
                EmbeddingService(model_name="invalid-model")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embed_text_normalization(self, embedding_service):
        """Test text normalization before embedding."""
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        embedding_service.model.encode = MagicMock(return_value=mock_embedding)
        
        # Test with text that needs normalization
        text_with_whitespace = "  This is a test document.  \n\n"
        result = await embedding_service.embed_text(text_with_whitespace)
        
        # Check that model was called with normalized text
        embedding_service.model.encode.assert_called_once()
        called_text = embedding_service.model.encode.call_args[0][0]
        assert called_text.strip() == "This is a test document."

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_similarity_calculation(self, embedding_service):
        """Test similarity calculation between embeddings."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        embedding3 = [1.0, 0.0, 0.0]  # Same as embedding1
        
        # Test cosine similarity
        similarity_different = embedding_service.calculate_similarity(embedding1, embedding2)
        similarity_same = embedding_service.calculate_similarity(embedding1, embedding3)
        
        assert similarity_different == pytest.approx(0.0, abs=1e-6)
        assert similarity_same == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embedding_caching(self, embedding_service):
        """Test embedding caching mechanism."""
        # Enable caching
        embedding_service.enable_caching = True
        embedding_service._cache = {}
        
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        embedding_service.model.encode = MagicMock(return_value=mock_embedding)
        
        text = "This is a test document."
        
        # First call should hit the model
        result1 = await embedding_service.embed_text(text)
        assert embedding_service.model.encode.call_count == 1
        
        # Second call should use cache
        result2 = await embedding_service.embed_text(text)
        assert embedding_service.model.encode.call_count == 1  # No additional calls
        assert result1 == result2

    @pytest.mark.unit
    def test_supported_models_list(self, embedding_service):
        """Test getting list of supported models."""
        supported_models = embedding_service.get_supported_models()
        
        assert isinstance(supported_models, list)
        assert "all-MiniLM-L6-v2" in supported_models
        assert "all-mpnet-base-v2" in supported_models