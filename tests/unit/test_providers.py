"""Unit tests for LLM Providers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.providers.openrouter_provider import OpenRouterProvider
from src.providers.vllm_provider import VLLMProvider
from src.providers.provider_factory import LLMProviderFactory, ProviderMode
from src.utils.exceptions import LLMError


class TestOpenRouterProvider:
    """Test cases for OpenRouterProvider."""

    @pytest.fixture
    def openrouter_provider(self):
        """Create OpenRouterProvider instance for testing."""
        with patch('aiohttp.ClientSession'):
            return OpenRouterProvider(
                api_key="test-api-key",
                base_url="https://openrouter.ai/api/v1"
            )

    @pytest.mark.unit
    def test_openrouter_provider_initialization(self, openrouter_provider):
        """Test OpenRouterProvider initialization."""
        assert openrouter_provider.api_key == "test-api-key"
        assert openrouter_provider.base_url == "https://openrouter.ai/api/v1"
        assert openrouter_provider.model_name is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_response_success(self, openrouter_provider):
        """Test successful response generation."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": "This is a test response from OpenRouter."
                }
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mock_response
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            messages = [{"role": "user", "content": "Hello, test message"}]
            response = await openrouter_provider.generate_response(messages)
            
            assert response["content"] == "This is a test response from OpenRouter."
            assert response["usage"]["total_tokens"] == 75

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_response_api_error(self, openrouter_provider):
        """Test API error handling."""
        error_response = {
            "error": {
                "message": "API rate limit exceeded",
                "code": "rate_limit_exceeded"
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=error_response
            )
            mock_post.return_value.__aenter__.return_value.status = 429
            
            messages = [{"role": "user", "content": "Test message"}]
            
            with pytest.raises(LLMError, match="API rate limit exceeded"):
                await openrouter_provider.generate_response(messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_stream_success(self, openrouter_provider):
        """Test successful streaming response."""
        stream_chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
            'data: [DONE]\n\n'
        ]
        
        async def mock_stream():
            for chunk in stream_chunks:
                yield chunk.encode()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.content.iter_chunked = mock_stream
            mock_post.return_value.__aenter__.return_value.status = 200
            
            messages = [{"role": "user", "content": "Stream test"}]
            chunks = []
            
            async for chunk in openrouter_provider.generate_stream(messages):
                chunks.append(chunk)
            
            assert len(chunks) == 3  # Excluding [DONE]
            assert chunks[0]["content"] == "Hello"
            assert chunks[1]["content"] == " world"
            assert chunks[2]["content"] == "!"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_available_models(self, openrouter_provider):
        """Test getting available models."""
        mock_models_response = {
            "data": [
                {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
                {"id": "anthropic/claude-2", "name": "Claude 2"},
                {"id": "meta-llama/llama-2-7b", "name": "Llama 2 7B"}
            ]
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mock_models_response
            )
            mock_get.return_value.__aenter__.return_value.status = 200
            
            models = await openrouter_provider.get_available_models()
            
            assert len(models) == 3
            assert models[0]["id"] == "openai/gpt-3.5-turbo"
            assert models[1]["id"] == "anthropic/claude-2"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_token_counting(self, openrouter_provider):
        """Test token counting functionality."""
        text = "This is a test message for token counting."
        
        # Mock the tokenizer
        with patch('tiktoken.encoding_for_model') as mock_tokenizer:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens
            mock_tokenizer.return_value = mock_encoder
            
            token_count = await openrouter_provider.count_tokens(text)
            assert token_count == 8


class TestVLLMProvider:
    """Test cases for VLLMProvider."""

    @pytest.fixture
    def vllm_provider(self):
        """Create VLLMProvider instance for testing."""
        return VLLMProvider(
            model_name="test-model",
            base_url="http://localhost:8000"
        )

    @pytest.mark.unit
    def test_vllm_provider_initialization(self, vllm_provider):
        """Test VLLMProvider initialization."""
        assert vllm_provider.model_name == "test-model"
        assert vllm_provider.base_url == "http://localhost:8000"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_response_success(self, vllm_provider):
        """Test successful response generation with VLLM."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": "This is a response from VLLM."
                }
            }],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 15,
                "total_tokens": 45
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mock_response
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            messages = [{"role": "user", "content": "Hello VLLM"}]
            response = await vllm_provider.generate_response(messages)
            
            assert response["content"] == "This is a response from VLLM."
            assert response["usage"]["total_tokens"] == 45

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_success(self, vllm_provider):
        """Test VLLM health check."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.status = 200
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"status": "healthy"}
            )
            
            is_healthy = await vllm_provider.health_check()
            assert is_healthy is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_failure(self, vllm_provider):
        """Test VLLM health check failure."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.status = 500
            
            is_healthy = await vllm_provider.health_check()
            assert is_healthy is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_model_info(self, vllm_provider):
        """Test getting model information."""
        mock_model_info = {
            "model_name": "test-model",
            "model_type": "llama",
            "max_tokens": 4096,
            "context_length": 2048
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mock_model_info
            )
            mock_get.return_value.__aenter__.return_value.status = 200
            
            model_info = await vllm_provider.get_model_info()
            
            assert model_info["model_name"] == "test-model"
            assert model_info["max_tokens"] == 4096

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_with_system_message(self, vllm_provider):
        """Test response generation with system message."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": "Response following system instructions."
                }
            }],
            "usage": {"total_tokens": 50}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mock_response
            )
            mock_post.return_value.__aenter__.return_value.status = 200
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
            
            response = await vllm_provider.generate_response(messages)
            assert response["content"] == "Response following system instructions."


class TestLLMProviderFactory:
    """Test cases for LLMProviderFactory."""

    @pytest.fixture
    def factory(self):
        """Create LLMProviderFactory instance for testing."""
        return LLMProviderFactory()

    @pytest.mark.unit
    def test_factory_initialization(self, factory):
        """Test LLMProviderFactory initialization."""
        assert factory is not None
        assert factory._provider_cache == {}
        assert factory._config_cache == {}
        assert factory._unified_provider is None

    @pytest.mark.unit
    def test_create_unified_provider(self, factory):
        """Test creating unified provider."""
        with patch('src.providers.llm_providers.UnifiedLLMProvider') as mock_unified:
            provider = factory.create_provider(mode=ProviderMode.UNIFIED)
            mock_unified.assert_called_once()

    @pytest.mark.unit
    def test_validate_config(self, factory):
        """Test configuration validation."""
        from src.config.settings import LLMProvider
        
        # Valid config
        config = {
            "model_name": "test-model",
            "max_tokens": 2048,
            "temperature": 0.7,
            "timeout": 60
        }
        
        validated = factory.validate_config(LLMProvider.VLLM, config)
        assert validated["model_name"] == "test-model"

    @pytest.mark.unit 
    def test_get_available_providers(self, factory):
        """Test getting available providers."""
        with patch.object(factory, '_create_single_provider') as mock_create:
            with patch.object(factory, '_get_provider_config') as mock_config:
                mock_provider = MagicMock()
                mock_provider.get_model_info.return_value = {"model": "test"}
                mock_create.return_value = mock_provider
                mock_config.return_value = {}
                
                providers = factory.get_available_providers()
                assert isinstance(providers, dict)

    @pytest.mark.unit
    def test_clear_cache(self, factory):
        """Test cache clearing."""
        # Add some items to cache
        factory._provider_cache["test"] = "provider"
        factory._config_cache["test"] = "config"
        
        factory.clear_cache()
        
        assert factory._provider_cache == {}
        assert factory._config_cache == {}
        assert factory._unified_provider is None
