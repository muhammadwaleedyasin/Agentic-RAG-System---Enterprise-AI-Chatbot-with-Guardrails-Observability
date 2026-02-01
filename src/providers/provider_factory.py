"""
Consolidated LLM provider factory with comprehensive configuration and switching logic.
Enhanced factory for creating LLM provider instances with comprehensive fallback support.
"""
import logging
from typing import Dict, Any, Optional, Union, List
from enum import Enum

from .base_provider import BaseLLMProvider, ProviderConfigError
from .vllm_provider import VLLMProvider
from .openrouter_provider import OpenRouterProvider
from .llm_providers import UnifiedLLMProvider
from src.config.settings import LLMProvider as ProviderType, settings

logger = logging.getLogger(__name__)


class ProviderMode(str, Enum):
    """LLM provider operation modes."""
    SINGLE = "single"           # Use single provider only
    UNIFIED = "unified"         # Use unified provider with failover
    FALLBACK = "fallback"       # Primary with explicit fallback
    LOAD_BALANCED = "load_balanced"  # Load balance across providers


class LLMProviderFactory:
    """Enhanced factory for creating and managing LLM provider instances."""
    
    def __init__(self):
        """Initialize the provider factory."""
        self._provider_cache: Dict[str, BaseLLMProvider] = {}
        self._unified_provider: Optional[UnifiedLLMProvider] = None
        self._config_cache: Dict[str, Dict[str, Any]] = {}
    
    def create_provider(
        self,
        provider_type: Optional[Union[ProviderType, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: ProviderMode = ProviderMode.UNIFIED,
        cache_enabled: bool = True
    ) -> Union[BaseLLMProvider, UnifiedLLMProvider]:
        """
        Create an LLM provider instance.
        
        Args:
            provider_type: Type of provider to create
            config: Custom configuration
            mode: Provider operation mode (currently only SINGLE and UNIFIED are implemented)
            cache_enabled: Whether to use provider caching
            
        Returns:
            Provider instance
            
        Raises:
            ProviderConfigError: If configuration is invalid
        """
        # Check for unimplemented modes
        if mode not in [ProviderMode.SINGLE, ProviderMode.UNIFIED]:
            raise ProviderConfigError(
                f"Provider mode '{mode.value}' is not yet implemented. "
                f"Only '{ProviderMode.SINGLE.value}' and '{ProviderMode.UNIFIED.value}' modes are currently supported."
            )
            
        try:
            # Handle unified mode
            if mode == ProviderMode.UNIFIED:
                return self._create_unified_provider(config, cache_enabled)
            
            # Handle single provider mode
            if provider_type is None:
                provider_type = settings.llm_provider
            
            if isinstance(provider_type, str):
                provider_type = ProviderType(provider_type.lower())
            
            # Check cache
            cache_key = f"{provider_type.value}_{mode.value}"
            if cache_enabled and cache_key in self._provider_cache:
                logger.debug(f"Using cached provider: {cache_key}")
                return self._provider_cache[cache_key]
            
            # Get configuration
            if config is None:
                config = self._get_provider_config(provider_type)
            
            # Create provider
            provider = self._create_single_provider(provider_type, config)
            
            # Cache if enabled
            if cache_enabled:
                self._provider_cache[cache_key] = provider
            
            logger.info(f"Created {provider_type.value} provider in {mode.value} mode")
            return provider
            
        except Exception as e:
            logger.error(f"Failed to create provider: {str(e)}")
            raise ProviderConfigError(f"Provider creation failed: {str(e)}")
    
    def _create_unified_provider(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache_enabled: bool = True
    ) -> UnifiedLLMProvider:
        """Create unified provider with all available providers."""
        if cache_enabled and self._unified_provider is not None:
            return self._unified_provider
        
        if config is None:
            config = self._get_unified_config()
        
        provider = UnifiedLLMProvider(config)
        
        if cache_enabled:
            self._unified_provider = provider
        
        logger.info("Created unified LLM provider")
        return provider
    
    def _create_single_provider(
        self,
        provider_type: ProviderType,
        config: Dict[str, Any]
    ) -> BaseLLMProvider:
        """Create a single provider instance."""
        if provider_type == ProviderType.VLLM:
            return VLLMProvider(config)
        elif provider_type == ProviderType.OPENROUTER:
            return OpenRouterProvider(config)
        else:
            raise ProviderConfigError(f"Unsupported provider type: {provider_type}")
    
    def _get_provider_config(self, provider_type: ProviderType) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        # Check cache
        cache_key = provider_type.value
        if cache_key in self._config_cache:
            return self._config_cache[cache_key].copy()
        
        config = {}
        
        if provider_type == ProviderType.VLLM:
            config = {
                "model_name": settings.vllm_model_name,
                "base_url": settings.vllm_base_url,
                "api_key": settings.vllm_api_key,
                "max_tokens": settings.vllm_max_tokens,
                "temperature": settings.vllm_temperature,
                "timeout": settings.vllm_timeout,
                "max_retries": settings.vllm_max_retries,
                "retry_delay": settings.vllm_retry_delay,
                "connect_timeout": settings.vllm_connect_timeout,
                "read_timeout": settings.vllm_read_timeout,
                # Advanced parameters
                "top_p": settings.vllm_top_p,
                "presence_penalty": settings.vllm_presence_penalty,
                "frequency_penalty": settings.vllm_frequency_penalty,
                "repetition_penalty": settings.vllm_repetition_penalty,
                "stop_tokens": settings.vllm_stop_tokens,
                "best_of": settings.vllm_best_of,
                "use_beam_search": settings.vllm_use_beam_search,
                "top_k": settings.vllm_top_k,
                "length_penalty": settings.vllm_length_penalty,
                "max_model_length": settings.vllm_max_model_length
            }
        
        elif provider_type == ProviderType.OPENROUTER:
            if not settings.openrouter_api_key:
                raise ProviderConfigError("OpenRouter API key is required")
            
            config = {
                "model_name": settings.openrouter_model_name,
                "api_key": settings.openrouter_api_key,
                "base_url": settings.openrouter_base_url,
                "max_tokens": settings.openrouter_max_tokens,
                "temperature": settings.openrouter_temperature,
                "timeout": settings.openrouter_timeout,
                "max_retries": settings.openrouter_max_retries,
                "retry_delay": settings.openrouter_retry_delay,
                "app_name": settings.app_name,
                "http_referer": settings.openrouter_http_referer,
                "site_url": settings.openrouter_site_url,
                # Advanced parameters
                "top_p": settings.openrouter_top_p,
                "presence_penalty": settings.openrouter_presence_penalty,
                "frequency_penalty": settings.openrouter_frequency_penalty,
                "repetition_penalty": settings.openrouter_repetition_penalty,
                "top_k": settings.openrouter_top_k,
                "min_p": settings.openrouter_min_p,
                "seed": settings.openrouter_seed,
                "logit_bias": settings.openrouter_logit_bias,
                "response_format": settings.openrouter_response_format,
                "fallback_models": settings.openrouter_fallback_models,
                "max_cost_per_request": settings.openrouter_max_cost_per_request
            }
        
        # Cache configuration
        self._config_cache[cache_key] = config.copy()
        return config
    
    def _get_unified_config(self) -> Dict[str, Any]:
        """Get configuration for unified provider."""
        providers = {}
        enabled_providers = []
        
        # Add vLLM if available
        try:
            vllm_config = self._get_provider_config(ProviderType.VLLM)
            vllm_config["type"] = "vllm"
            vllm_config["priority"] = 1
            providers["vllm"] = vllm_config
            enabled_providers.append("vllm")
            logger.debug("vLLM provider added to unified configuration")
        except Exception as e:
            logger.warning(f"vLLM provider not available: {str(e)}")
        
        # Add OpenRouter if API key is available
        try:
            if settings.openrouter_api_key:
                openrouter_config = self._get_provider_config(ProviderType.OPENROUTER)
                openrouter_config["type"] = "openrouter"
                openrouter_config["priority"] = 2
                providers["openrouter"] = openrouter_config
                enabled_providers.append("openrouter")
                logger.debug("OpenRouter provider added to unified configuration")
        except Exception as e:
            logger.warning(f"OpenRouter provider not available: {str(e)}")
        
        if not providers:
            raise ProviderConfigError("No providers available for unified configuration")
        
        return {
            "providers": providers,
            "enabled_providers": enabled_providers,
            "failover_strategy": settings.failover_strategy,
            "load_balancing": settings.load_balancing_strategy,
            "circuit_breaker_enabled": settings.circuit_breaker_enabled,
            "circuit_breaker_threshold": settings.circuit_breaker_threshold,
            "circuit_breaker_timeout": settings.circuit_breaker_timeout,
            "health_check_interval": settings.provider_health_check_interval
        }
    
    def create_from_config(self, config: Dict[str, Any]) -> BaseLLMProvider:
        """
        Create a provider from a configuration dictionary.
        
        Args:
            config: Configuration with 'type' field
            
        Returns:
            Provider instance
        """
        provider_type_str = config.get("type")
        if not provider_type_str:
            raise ProviderConfigError("Provider type not specified in config")
        
        try:
            provider_type = ProviderType(provider_type_str.lower())
        except ValueError:
            raise ProviderConfigError(f"Unknown provider type: {provider_type_str}")
        
        # Validate configuration
        validated_config = self.validate_config(provider_type, config)
        
        return self._create_single_provider(provider_type, validated_config)
    
    def validate_config(
        self,
        provider_type: ProviderType,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate provider configuration.
        
        Args:
            provider_type: Provider type
            config: Configuration to validate
            
        Returns:
            Validated configuration
            
        Raises:
            ProviderConfigError: If configuration is invalid
        """
        validated_config = config.copy()
        
        # Common validations
        if not validated_config.get("model_name"):
            raise ProviderConfigError("model_name is required")
        
        max_tokens = validated_config.get("max_tokens", 2048)
        if max_tokens <= 0:
            raise ProviderConfigError("max_tokens must be positive")
        
        temperature = validated_config.get("temperature", 0.7)
        if not (0.0 <= temperature <= 2.0):
            raise ProviderConfigError("temperature must be between 0.0 and 2.0")
        
        timeout = validated_config.get("timeout", 60)
        if timeout <= 0:
            raise ProviderConfigError("timeout must be positive")
        
        # Provider-specific validations
        if provider_type == ProviderType.VLLM:
            base_url = validated_config.get("base_url")
            if not base_url:
                raise ProviderConfigError("base_url is required for vLLM")
            
            # Ensure URL format is correct
            base_url = base_url.rstrip("/")
            if not base_url.startswith(("http://", "https://")):
                raise ProviderConfigError("base_url must start with http:// or https://")
            validated_config["base_url"] = base_url
            
        elif provider_type == ProviderType.OPENROUTER:
            if not validated_config.get("api_key"):
                raise ProviderConfigError("api_key is required for OpenRouter")
            
            if not validated_config.get("app_name"):
                validated_config["app_name"] = settings.app_name
            
            if not validated_config.get("http_referer"):
                validated_config["http_referer"] = "https://github.com/enterprise-rag"
        
        return validated_config
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available providers."""
        providers = {}
        
        for provider_type in ProviderType:
            try:
                config = self._get_provider_config(provider_type)
                provider = self._create_single_provider(provider_type, config)
                info = provider.get_model_info()
                info["available"] = True
                info["config_valid"] = True
                providers[provider_type.value] = info
                logger.debug(f"Provider {provider_type.value} is available")
            except Exception as e:
                providers[provider_type.value] = {
                    "error": str(e),
                    "available": False,
                    "config_valid": False,
                    "provider": provider_type.value
                }
                logger.debug(f"Provider {provider_type.value} not available: {str(e)}")
        
        return providers
    
    def clear_cache(self):
        """Clear all cached providers and configurations."""
        self._provider_cache.clear()
        self._config_cache.clear()
        self._unified_provider = None
        logger.info("Provider cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "provider_cache_size": len(self._provider_cache),
            "config_cache_size": len(self._config_cache),
            "unified_provider_cached": self._unified_provider is not None,
            "cached_providers": list(self._provider_cache.keys()),
            "cached_configs": list(self._config_cache.keys())
        }


# Global factory instance
llm_factory = LLMProviderFactory()


def create_llm_provider(
    provider_type: Optional[Union[ProviderType, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    unified: bool = True
) -> Union[BaseLLMProvider, UnifiedLLMProvider]:
    """
    Convenience function to create LLM provider instances.
    
    Args:
        provider_type: Type of provider to create
        config: Custom configuration
        unified: Whether to create unified provider
        
    Returns:
        Provider instance
    """
    mode = ProviderMode.UNIFIED if unified else ProviderMode.SINGLE
    return llm_factory.create_provider(provider_type, config, mode)


def create_unified_provider(config: Optional[Dict[str, Any]] = None) -> UnifiedLLMProvider:
    """
    Create a unified provider with all available providers and fallback support.
    
    Args:
        config: Custom configuration for the unified provider
        
    Returns:
        UnifiedLLMProvider instance with all configured providers
    """
    return llm_factory.create_provider(mode=ProviderMode.UNIFIED, config=config)



def validate_provider_config(provider_type: ProviderType, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate provider configuration.
    
    Args:
        provider_type: The provider type
        config: Configuration to validate
        
    Returns:
        Validated configuration with any necessary defaults
        
    Raises:
        ProviderConfigError: If configuration is invalid
    """
    return llm_factory.validate_config(provider_type, config)


def create_provider_from_config(config: Dict[str, Any]) -> BaseLLMProvider:
    """
    Create a provider from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'type' field
        
    Returns:
        Provider instance
        
    Raises:
        ProviderConfigError: If configuration is invalid
    """
    provider_type_str = config.get("type")
    if not provider_type_str:
        raise ProviderConfigError("Provider type not specified in config")
    
    try:
        provider_type = ProviderType(provider_type_str.lower())
    except ValueError:
        raise ProviderConfigError(f"Unknown provider type: {provider_type_str}")
    
    # Validate configuration
    validated_config = validate_provider_config(provider_type, config)
    
    # Create provider
    return create_llm_provider(provider_type, validated_config, unified=False)


def get_provider_factory() -> LLMProviderFactory:
    """Get the global provider factory instance."""
    return llm_factory
