"""
Tests for LLM provider implementations.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.providers.base_provider import (
    BaseLLMProvider, LLMMessage, LLMResponse, LLMStreamChunk,
    ProviderStatus, ProviderConfigError, ProviderUnavailableError
)
from src.providers.vllm_provider import VLLMProvider
from src.providers.openrouter_provider import OpenRouterProvider
from src.providers.llm_providers import UnifiedLLMProvider
from src.providers.provider_factory import LLMProviderFactory, ProviderMode
from src.config.settings import LLMProvider


class TestBaseLLMProvider:
    """Test base provider functionality."""
    
    def test_provider_initialization(self):
        """Test provider initialization."""
        config = {
            "model_name": "test-model",
            "max_tokens": 100,
            "temperature": 0.7,
            "timeout": 30
        }
        
        # Create a concrete implementation for testing
        class TestProvider(BaseLLMProvider):
            async def _generate_impl(self, messages, max_tokens=None, temperature=None, **kwargs):
                return LLMResponse(content="test response", provider="test")
            
            async def _stream_impl(self, messages, max_tokens=None, temperature=None, **kwargs):
                yield LLMStreamChunk(content="test", is_final=True, provider="test")
            
            async def _health_check_impl(self):
                return True
            
            def get_model_info(self):
                return {"provider": "test"}
        
        provider = TestProvider(config)
        assert provider.model_name == "test-model"
        assert provider.max_tokens == 100
        assert provider.temperature == 0.7
        assert provider.timeout == 30
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        config = {"model_name": "test"}
        
        class TestProvider(BaseLLMProvider):
            async def _generate_impl(self, messages, max_tokens=None, temperature=None, **kwargs):
                return LLMResponse(content="test")
            async def _stream_impl(self, messages, max_tokens=None, temperature=None, **kwargs):
                yield LLMStreamChunk(content="test", is_final=True)
            async def _health_check_impl(self):
                return True
            def get_model_info(self):
                return {}
        
        provider = TestProvider(config)
        
        # Test valid parameters
        max_tokens, temperature = provider.validate_parameters(100, 0.5)
        assert max_tokens == 100
        assert temperature == 0.5
        
        # Test invalid parameters
        with pytest.raises(ProviderConfigError):
            provider.validate_parameters(-1, 0.5)  # Invalid max_tokens
        
        with pytest.raises(ProviderConfigError):
            provider.validate_parameters(100, 3.0)  # Invalid temperature


class TestVLLMProvider:
    """Test vLLM provider implementation."""
    
    @pytest.fixture
    def vllm_config(self):
        """Fixture for vLLM configuration."""
        return {
            "model_name": "test-model",
            "base_url": "http://localhost:8001",
            "max_tokens": 100,
            "temperature": 0.7,
            "timeout": 30
        }
    
    @pytest.fixture
    def vllm_provider(self, vllm_config):
        """Fixture for vLLM provider."""
        return VLLMProvider(vllm_config)
    
    def test_vllm_initialization(self, vllm_provider):
        """Test vLLM provider initialization."""
        assert vllm_provider.model_name == "test-model"
        assert vllm_provider.base_url == "http://localhost:8001"
        assert vllm_provider.get_provider_name() == "vllm"
    
    @pytest.mark.asyncio
    async def test_vllm_health_check(self, vllm_provider):
        """Test vLLM health check."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful health check
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await vllm_provider._health_check_impl()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_vllm_generate(self, vllm_provider):
        """Test vLLM generation."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"content": "Test response"},
                    "finish_reason": "stop"
                }],
                "usage": {"total_tokens": 50},
                "model": "test-model"
            }
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            messages = [LLMMessage(role="user", content="Test message")]
            response = await vllm_provider._generate_impl(messages)
            
            assert response.content == "Test response"
            assert response.model == "test-model"
            assert response.provider == "vllm"


class TestOpenRouterProvider:
    """Test OpenRouter provider implementation."""
    
    @pytest.fixture
    def openrouter_config(self):
        """Fixture for OpenRouter configuration."""
        return {
            "model_name": "anthropic/claude-3-haiku",
            "api_key": "test-api-key",
            "base_url": "https://openrouter.ai/api/v1",
            "max_tokens": 100,
            "temperature": 0.7,
            "timeout": 60,
            "fallback_models": ["openai/gpt-3.5-turbo"]
        }
    
    @pytest.fixture
    def openrouter_provider(self, openrouter_config):
        """Fixture for OpenRouter provider."""
        return OpenRouterProvider(openrouter_config)
    
    def test_openrouter_initialization(self, openrouter_provider):
        """Test OpenRouter provider initialization."""
        assert openrouter_provider.model_name == "anthropic/claude-3-haiku"
        assert openrouter_provider.api_key == "test-api-key"
        assert openrouter_provider.fallback_models == ["openai/gpt-3.5-turbo"]
        assert openrouter_provider.get_provider_name() == "openrouter"
    
    def test_openrouter_missing_api_key(self):
        """Test OpenRouter provider without API key."""
        config = {"model_name": "test-model"}
        with pytest.raises(ValueError):
            OpenRouterProvider(config)
    
    @pytest.mark.asyncio
    async def test_openrouter_health_check(self, openrouter_provider):
        """Test OpenRouter health check."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful health check
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await openrouter_provider._health_check_impl()
            assert result is True


class TestUnifiedLLMProvider:
    """Test unified LLM provider implementation."""
    
    @pytest.fixture
    def unified_config(self):
        """Fixture for unified provider configuration."""
        return {
            "providers": {
                "vllm": {
                    "type": "vllm",
                    "model_name": "test-model",
                    "base_url": "http://localhost:8001",
                    "priority": 1
                },
                "openrouter": {
                    "type": "openrouter",
                    "model_name": "anthropic/claude-3-haiku",
                    "api_key": "test-key",
                    "priority": 2
                }
            },
            "failover_strategy": "priority",
            "load_balancing": "health_weighted"
        }
    
    @pytest.fixture
    def unified_provider(self, unified_config):
        """Fixture for unified provider."""
        return UnifiedLLMProvider(unified_config)
    
    def test_unified_initialization(self, unified_provider):
        """Test unified provider initialization."""
        assert len(unified_provider.providers) == 2
        assert "vllm" in unified_provider.providers
        assert "openrouter" in unified_provider.providers
        assert unified_provider.primary_provider == "vllm"
    
    @pytest.mark.asyncio
    async def test_unified_health_check(self, unified_provider):
        """Test unified provider health check."""
        # Mock individual provider health checks
        for provider in unified_provider.providers.values():
            provider.health_check = AsyncMock(return_value=Mock(status=ProviderStatus.HEALTHY))
        
        health = await unified_provider.health_check()
        assert health["overall_status"] == "healthy"
        assert health["healthy_providers"] == 2


class TestLLMProviderFactory:
    """Test LLM provider factory."""
    
    @pytest.fixture
    def factory(self):
        """Fixture for provider factory."""
        return LLMProviderFactory()
    
    def test_factory_initialization(self, factory):
        """Test factory initialization."""
        assert len(factory._provider_cache) == 0
        assert factory._unified_provider is None
    
    def test_get_available_providers(self, factory):
        """Test getting available providers."""
        with patch('src.providers.provider_factory.settings') as mock_settings:
            mock_settings.vllm_model_name = "test-model"
            mock_settings.vllm_base_url = "http://localhost:8001"
            mock_settings.openrouter_api_key = None  # Not available
            
            available = factory.get_available_providers()
            assert LLMProvider.VLLM.value in available
            assert LLMProvider.OPENROUTER.value in available
            assert available[LLMProvider.OPENROUTER.value]["available"] is False
    
    def test_config_validation(self, factory):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            "model_name": "test-model",
            "base_url": "http://localhost:8001",
            "max_tokens": 100,
            "temperature": 0.7,
            "timeout": 30
        }
        
        validated = factory.validate_config(LLMProvider.VLLM, valid_config)
        assert validated["model_name"] == "test-model"
        
        # Invalid config - empty model name
        invalid_config = {
            "model_name": "",
            "base_url": "http://localhost:8001"
        }
        
        with pytest.raises(ProviderConfigError):
            factory.validate_config(LLMProvider.VLLM, invalid_config)
    
    def test_cache_functionality(self, factory):
        """Test cache functionality."""
        # Test cache stats
        stats = factory.get_cache_stats()
        assert stats["provider_cache_size"] == 0
        assert stats["unified_provider_cached"] is False
        
        # Test cache clearing
        factory.clear_cache()
        assert len(factory._provider_cache) == 0


class TestProviderIntegration:
    """Integration tests for provider system."""
    
    @pytest.mark.asyncio
    async def test_provider_fallback(self):
        """Test provider fallback functionality."""
        # Create unified provider with mocked providers
        config = {
            "providers": {
                "primary": {
                    "type": "vllm",
                    "model_name": "test-model",
                    "base_url": "http://localhost:8001",
                    "priority": 1
                },
                "fallback": {
                    "type": "openrouter",
                    "model_name": "test-model",
                    "api_key": "test-key",
                    "priority": 2
                }
            },
            "failover_strategy": "priority"
        }
        
        unified = UnifiedLLMProvider(config)
        
        # Mock primary provider to fail
        unified.providers["primary"].generate = AsyncMock(
            side_effect=ProviderUnavailableError("Provider down")
        )
        
        # Mock fallback provider to succeed
        unified.providers["fallback"].generate = AsyncMock(
            return_value=LLMResponse(
                content="Fallback response",
                provider="fallback",
                fallback_used=True
            )
        )
        
        messages = [LLMMessage(role="user", content="Test")]
        response = await unified.generate(messages)
        
        assert response.content == "Fallback response"
        assert response.fallback_used is True
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        config = {
            "providers": {
                "test": {
                    "type": "vllm",
                    "model_name": "test-model",
                    "base_url": "http://localhost:8001"
                }
            },
            "circuit_breaker_enabled": True,
            "circuit_breaker_threshold": 2
        }
        
        unified = UnifiedLLMProvider(config)
        
        # Mock provider to always fail
        unified.providers["test"].generate = AsyncMock(
            side_effect=ProviderUnavailableError("Always fails")
        )
        
        messages = [LLMMessage(role="user", content="Test")]
        
        # First few failures should attempt the provider
        for _ in range(3):
            try:
                await unified.generate(messages)
            except:
                pass
        
        # Circuit breaker should now be open
        usage = unified.provider_usage["test"]
        assert usage["failures"] >= 2


@pytest.mark.asyncio
async def test_end_to_end_provider_usage():
    """End-to-end test of provider usage."""
    # Test factory creation
    factory = LLMProviderFactory()
    
    # Mock settings for testing
    with patch('src.providers.provider_factory.settings') as mock_settings:
        mock_settings.vllm_model_name = "test-model"
        mock_settings.vllm_base_url = "http://localhost:8001"
        mock_settings.use_unified_provider = True
        
        # Test provider creation
        try:
            provider = factory.create_provider(mode=ProviderMode.SINGLE)
            assert provider is not None
        except Exception as e:
            # Expected if no actual providers are available
            assert "not available" in str(e) or "Provider creation failed" in str(e)


if __name__ == "__main__":
    pytest.main([__file__])
