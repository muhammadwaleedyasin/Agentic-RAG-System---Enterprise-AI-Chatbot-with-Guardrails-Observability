"""
API endpoints for LLM provider management and monitoring.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import logging

from ...providers.provider_factory import llm_factory, ProviderMode
from ...providers.base_provider import LLMMessage, ProviderStatus
from ...config.settings import LLMProvider, settings
from ...models.base import BaseResponse

router = APIRouter()
logger = logging.getLogger(__name__)


class ProviderHealth(BaseResponse):
    """Provider health status response."""
    status: str
    healthy_providers: int
    total_providers: int
    providers: Dict[str, Any]


class ProviderStats(BaseResponse):
    """Provider statistics response."""
    providers: Dict[str, Any]
    unified_stats: Optional[Dict[str, Any]] = None


class ProviderInfo(BaseResponse):
    """Provider information response."""
    name: str
    available: bool
    config_valid: bool
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.get("/providers/available", response_model=Dict[str, ProviderInfo])
async def get_available_providers():
    """
    Get information about all available LLM providers.
    """
    try:
        available = llm_factory.get_available_providers()
        
        result = {}
        for name, info in available.items():
            result[name] = ProviderInfo(
                name=name,
                available=info.get("available", False),
                config_valid=info.get("config_valid", False),
                model_info=info if info.get("available") else None,
                error=info.get("error")
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting available providers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")


@router.get("/providers/health", response_model=ProviderHealth)
async def get_provider_health():
    """
    Get health status of all providers.
    """
    try:
        # Get unified provider for health check
        provider = llm_factory.create_provider(mode=ProviderMode.UNIFIED)
        health_status = await provider.health_check()
        
        return ProviderHealth(
            status=health_status["overall_status"],
            healthy_providers=health_status["healthy_providers"],
            total_providers=health_status["total_providers"],
            providers=health_status["providers"]
        )
        
    except Exception as e:
        logger.error(f"Error getting provider health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")


@router.get("/providers/stats", response_model=ProviderStats)
async def get_provider_statistics():
    """
    Get detailed statistics for all providers.
    """
    try:
        provider = llm_factory.create_provider(mode=ProviderMode.UNIFIED)
        stats = provider.get_provider_statistics()
        
        return ProviderStats(
            providers=stats["providers"],
            unified_stats=stats.get("unified_stats")
        )
        
    except Exception as e:
        logger.error(f"Error getting provider statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/providers/test/{provider_name}")
async def test_provider(provider_name: str, message: str = "Hello, this is a test message."):
    """
    Test a specific provider with a simple message.
    """
    try:
        # Validate provider name
        if provider_name not in [p.value for p in LLMProvider]:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")
        
        # Create single provider
        provider = llm_factory.create_provider(
            provider_type=LLMProvider(provider_name),
            mode=ProviderMode.SINGLE
        )
        
        # Test health first
        health = await provider.health_check()
        if health.status != ProviderStatus.HEALTHY:
            raise HTTPException(
                status_code=503, 
                detail=f"Provider {provider_name} is not healthy: {health.status}"
            )
        
        # Test generation
        messages = [LLMMessage(role="user", content=message)]
        response = await provider.generate(messages, max_tokens=100)
        
        return {
            "provider": provider_name,
            "status": "success",
            "response": {
                "content": response.content,
                "model": response.model,
                "response_time": response.response_time,
                "usage": response.usage
            },
            "health": health.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing provider {provider_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Provider test failed: {str(e)}")


@router.post("/providers/cache/clear")
async def clear_provider_cache():
    """
    Clear the provider cache.
    """
    try:
        cache_stats_before = llm_factory.get_cache_stats()
        llm_factory.clear_cache()
        cache_stats_after = llm_factory.get_cache_stats()
        
        return {
            "status": "success",
            "message": "Provider cache cleared",
            "before": cache_stats_before,
            "after": cache_stats_after
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/providers/cache/stats")
async def get_cache_statistics():
    """
    Get provider cache statistics.
    """
    try:
        stats = llm_factory.get_cache_stats()
        return {
            "status": "success",
            "cache_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.get("/providers/config")
async def get_provider_configuration():
    """
    Get current provider configuration from settings.
    """
    try:
        config = {
            "primary_provider": settings.llm_provider.value,
            "use_unified_provider": settings.use_unified_provider,
            "failover_strategy": settings.failover_strategy,
            "load_balancing_strategy": settings.load_balancing_strategy,
            "circuit_breaker_enabled": settings.circuit_breaker_enabled,
            "circuit_breaker_threshold": settings.circuit_breaker_threshold,
            "circuit_breaker_timeout": settings.circuit_breaker_timeout,
            "health_check_interval": settings.provider_health_check_interval,
            "providers": {
                "vllm": {
                    "model_name": settings.vllm_model_name,
                    "base_url": settings.vllm_base_url,
                    "max_tokens": settings.vllm_max_tokens,
                    "temperature": settings.vllm_temperature,
                    "timeout": settings.vllm_timeout,
                    "configured": bool(settings.vllm_model_name and settings.vllm_base_url)
                },
                "openrouter": {
                    "model_name": settings.openrouter_model_name,
                    "base_url": settings.openrouter_base_url,
                    "max_tokens": settings.openrouter_max_tokens,
                    "temperature": settings.openrouter_temperature,
                    "timeout": settings.openrouter_timeout,
                    "fallback_models": settings.openrouter_fallback_models,
                    "configured": bool(settings.openrouter_api_key)
                }
            }
        }
        
        return {
            "status": "success",
            "configuration": config
        }
        
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.post("/providers/unified/failover-test")
async def test_unified_failover():
    """
    Test unified provider failover functionality.
    """
    try:
        provider = llm_factory.create_provider(mode=ProviderMode.UNIFIED)
        
        # Get initial health
        initial_health = await provider.health_check()
        
        # Test with a simple message
        messages = [LLMMessage(role="user", content="Test failover with this message.")]
        response = await provider.generate(messages, max_tokens=50)
        
        # Get final statistics
        final_stats = provider.get_provider_statistics()
        
        return {
            "status": "success",
            "message": "Failover test completed",
            "initial_health": initial_health,
            "response": {
                "content": response.content,
                "provider_used": response.provider,
                "fallback_used": response.fallback_used,
                "response_time": response.response_time
            },
            "final_stats": final_stats["unified_stats"]
        }
        
    except Exception as e:
        logger.error(f"Error in failover test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failover test failed: {str(e)}")


@router.get("/providers/models/{provider_name}")
async def get_provider_models(provider_name: str):
    """
    Get available models for a specific provider (OpenRouter only).
    """
    try:
        if provider_name != "openrouter":
            raise HTTPException(
                status_code=400, 
                detail="Model listing is only available for OpenRouter provider"
            )
        
        from ...providers.openrouter_provider import OpenRouterProvider
        
        # Create OpenRouter provider
        provider = llm_factory.create_provider(
            provider_type=LLMProvider.OPENROUTER,
            mode=ProviderMode.SINGLE
        )
        
        if not isinstance(provider, OpenRouterProvider):
            raise HTTPException(status_code=500, detail="Failed to create OpenRouter provider")
        
        # Get available models
        models = await provider.get_available_models()
        
        return {
            "status": "success",
            "provider": provider_name,
            "models": models,
            "total_models": len(models)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models for {provider_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")
