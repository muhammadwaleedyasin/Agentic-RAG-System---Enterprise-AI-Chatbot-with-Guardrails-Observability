"""
Unified LLM interface with provider switching and comprehensive fallback support.
"""
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from enum import Enum

from .base_provider import (
    BaseLLMProvider, LLMMessage, LLMResponse, LLMStreamChunk, 
    ProviderHealth, ProviderStatus, BaseProviderError, 
    ProviderUnavailableError, ProviderConfigError
)
from .vllm_provider import VLLMProvider
from .openrouter_provider import OpenRouterProvider
from src.config.settings import LLMProvider as ProviderType, settings

logger = logging.getLogger(__name__)


class FailoverStrategy(str, Enum):
    """Failover strategies for provider switching."""
    NONE = "none"              # No failover, fail immediately
    MANUAL = "manual"          # Manual provider switching
    AUTO = "auto"             # Automatic failover based on health
    ROUND_ROBIN = "round_robin"  # Distribute load across providers
    PRIORITY = "priority"      # Try providers in priority order


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    HEALTH_WEIGHTED = "health_weighted"


class UnifiedLLMProvider:
    """
    Unified LLM interface that manages multiple providers with automatic failover,
    load balancing, and comprehensive error handling.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the unified provider."""
        self.config = config or {}
        
        # Provider management
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.primary_provider: Optional[str] = None
        self.provider_priorities: List[str] = []
        
        # Failover configuration
        self.failover_strategy = FailoverStrategy(
            self.config.get("failover_strategy", FailoverStrategy.AUTO)
        )
        self.load_balancing = LoadBalancingStrategy(
            self.config.get("load_balancing", LoadBalancingStrategy.HEALTH_WEIGHTED)
        )
        
        # Circuit breaker configuration
        self.circuit_breaker_enabled = self.config.get("circuit_breaker_enabled", True)
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 5)
        self.circuit_breaker_timeout = self.config.get("circuit_breaker_timeout", 60)
        
        # Health monitoring
        self.health_check_interval = self.config.get("health_check_interval", 300)  # 5 minutes
        self.last_health_check = 0
        
        # Statistics and monitoring
        self.total_requests = 0
        self.provider_usage = {}
        self.failover_count = 0
        
        # Current provider tracking
        self.current_provider_index = 0
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers."""
        # Get provider configurations
        provider_configs = self.config.get("providers", {})
        
        # If no specific config provided, use settings
        if not provider_configs:
            provider_configs = self._get_default_provider_configs()
        
        # Initialize each provider
        for provider_name, provider_config in provider_configs.items():
            try:
                provider = self._create_provider(provider_name, provider_config)
                self.providers[provider_name] = provider
                
                # Initialize usage tracking
                self.provider_usage[provider_name] = {
                    "requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_response_time": 0.0,
                    "circuit_breaker_open": False,
                    "circuit_breaker_opened_at": 0
                }
                
                logger.info(f"Initialized provider: {provider_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {str(e)}")
        
        # Set up provider priorities
        self.provider_priorities = list(self.providers.keys())
        if self.provider_priorities:
            self.primary_provider = self.provider_priorities[0]
            logger.info(f"Primary provider set to: {self.primary_provider}")
    
    def _get_default_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default provider configurations from settings."""
        configs = {}
        
        # vLLM configuration
        if settings.llm_provider == ProviderType.VLLM or "vllm" in self.config.get("enabled_providers", ["vllm"]):
            configs["vllm"] = {
                "type": "vllm",
                "model_name": settings.vllm_model_name,
                "base_url": settings.vllm_base_url,
                "max_tokens": settings.vllm_max_tokens,
                "temperature": settings.vllm_temperature,
                "timeout": settings.vllm_timeout,
                "priority": 1
            }
        
        # OpenRouter configuration
        if settings.openrouter_api_key and (
            settings.llm_provider == ProviderType.OPENROUTER or 
            "openrouter" in self.config.get("enabled_providers", ["openrouter"])
        ):
            configs["openrouter"] = {
                "type": "openrouter",
                "model_name": settings.openrouter_model_name,
                "api_key": settings.openrouter_api_key,
                "base_url": settings.openrouter_base_url,
                "max_tokens": settings.openrouter_max_tokens,
                "temperature": settings.openrouter_temperature,
                "timeout": settings.openrouter_timeout,
                "app_name": settings.app_name,
                "priority": 2
            }
        
        # Sort by priority
        sorted_configs = sorted(configs.items(), key=lambda x: x[1].get("priority", 999))
        return dict(sorted_configs)
    
    def _create_provider(self, name: str, config: Dict[str, Any]) -> BaseLLMProvider:
        """Create a provider instance based on configuration."""
        provider_type = config.get("type", name).lower()
        
        if provider_type == "vllm":
            return VLLMProvider(config)
        elif provider_type == "openrouter":
            return OpenRouterProvider(config)
        else:
            raise ProviderConfigError(f"Unknown provider type: {provider_type}")
    
    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        provider_preference: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using the best available provider.
        
        Args:
            messages: List of messages for the conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            provider_preference: Preferred provider name (optional)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLM response with provider information
        """
        self.total_requests += 1
        start_time = time.time()
        
        # Check health if needed
        await self._check_health_if_needed()
        
        # Get providers to try
        providers_to_try = self._get_providers_to_try(provider_preference)
        
        if not providers_to_try:
            raise ProviderUnavailableError("No healthy providers available")
        
        last_error = None
        
        # Try each provider
        for provider_name in providers_to_try:
            if self._is_circuit_breaker_open(provider_name):
                logger.debug(f"Circuit breaker open for {provider_name}, skipping")
                continue
            
            provider = self.providers[provider_name]
            
            try:
                logger.debug(f"Trying provider: {provider_name}")
                
                # Track usage
                self.provider_usage[provider_name]["requests"] += 1
                
                # Generate response
                response = await provider.generate(messages, max_tokens, temperature, **kwargs)
                
                # Update success statistics
                response_time = time.time() - start_time
                self.provider_usage[provider_name]["successes"] += 1
                self.provider_usage[provider_name]["total_response_time"] += response_time
                
                # Mark fallback if not primary provider
                if provider_name != self.primary_provider:
                    response.fallback_used = True
                    self.failover_count += 1
                
                logger.debug(f"Successfully generated response using {provider_name}")
                return response
                
            except Exception as e:
                last_error = e
                self.provider_usage[provider_name]["failures"] += 1
                
                # Check if we should open circuit breaker
                self._check_circuit_breaker(provider_name)
                
                logger.warning(f"Provider {provider_name} failed: {str(e)}")
                
                # Don't try more providers for config errors
                if isinstance(e, ProviderConfigError):
                    break
                
                continue
        
        # All providers failed
        error_msg = f"All providers failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise ProviderUnavailableError(error_msg)
    
    async def stream(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        provider_preference: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """
        Generate a streaming response using the best available provider.
        """
        self.total_requests += 1
        
        # Check health if needed
        await self._check_health_if_needed()
        
        # Get providers to try
        providers_to_try = self._get_providers_to_try(provider_preference)
        
        if not providers_to_try:
            yield LLMStreamChunk(
                content="",
                is_final=True,
                error="No healthy providers available"
            )
            return
        
        # Try each provider
        for provider_name in providers_to_try:
            if self._is_circuit_breaker_open(provider_name):
                continue
            
            provider = self.providers[provider_name]
            
            try:
                logger.debug(f"Trying streaming with provider: {provider_name}")
                
                # Track usage
                self.provider_usage[provider_name]["requests"] += 1
                
                chunk_count = 0
                async for chunk in provider.stream(messages, max_tokens, temperature, **kwargs):
                    chunk_count += 1
                    
                    # Mark fallback if not primary provider
                    if provider_name != self.primary_provider:
                        chunk.provider = provider_name + " (fallback)"
                        if chunk_count == 1:  # Only count failover once
                            self.failover_count += 1
                    
                    yield chunk
                    
                    if chunk.is_final:
                        break
                
                # Update success statistics
                if chunk_count > 0:
                    self.provider_usage[provider_name]["successes"] += 1
                    logger.debug(f"Successfully streamed response using {provider_name}")
                
                return  # Successfully streamed
                
            except Exception as e:
                self.provider_usage[provider_name]["failures"] += 1
                self._check_circuit_breaker(provider_name)
                
                logger.warning(f"Streaming with provider {provider_name} failed: {str(e)}")
                continue
        
        # All providers failed
        yield LLMStreamChunk(
            content="",
            is_final=True,
            error="All streaming providers failed"
        )
    
    def _get_providers_to_try(self, preference: Optional[str] = None) -> List[str]:
        """Get list of providers to try based on strategy."""
        available_providers = [
            name for name, provider in self.providers.items()
            if provider.health.status != ProviderStatus.UNHEALTHY
        ]
        
        if not available_providers:
            return list(self.providers.keys())  # Try all as last resort
        
        # If specific preference requested and available
        if preference and preference in available_providers:
            # Put preferred provider first, then others
            others = [p for p in available_providers if p != preference]
            return [preference] + others
        
        # Apply strategy
        if self.failover_strategy == FailoverStrategy.NONE:
            return [self.primary_provider] if self.primary_provider else []
        
        elif self.failover_strategy == FailoverStrategy.PRIORITY:
            # Use provider priorities
            return sorted(available_providers, key=lambda x: self.provider_priorities.index(x))
        
        elif self.failover_strategy == FailoverStrategy.ROUND_ROBIN:
            # Round robin selection
            if not available_providers:
                return []
            
            provider = available_providers[self.current_provider_index % len(available_providers)]
            self.current_provider_index += 1
            
            # Put selected provider first, then others
            others = [p for p in available_providers if p != provider]
            return [provider] + others
        
        elif self.failover_strategy == FailoverStrategy.AUTO:
            # Auto selection based on load balancing strategy
            return self._get_providers_by_load_balancing(available_providers)
        
        else:  # MANUAL or default
            return [self.primary_provider] if self.primary_provider in available_providers else available_providers
    
    def _get_providers_by_load_balancing(self, providers: List[str]) -> List[str]:
        """Get providers ordered by load balancing strategy."""
        if not providers:
            return []
        
        if self.load_balancing == LoadBalancingStrategy.ROUND_ROBIN:
            # Simple round robin
            provider = providers[self.current_provider_index % len(providers)]
            self.current_provider_index += 1
            others = [p for p in providers if p != provider]
            return [provider] + others
        
        elif self.load_balancing == LoadBalancingStrategy.RESPONSE_TIME:
            # Sort by average response time (ascending)
            def avg_response_time(provider_name):
                usage = self.provider_usage[provider_name]
                if usage["successes"] == 0:
                    return 0  # Prioritize untested providers
                return usage["total_response_time"] / usage["successes"]
            
            return sorted(providers, key=avg_response_time)
        
        elif self.load_balancing == LoadBalancingStrategy.SUCCESS_RATE:
            # Sort by success rate (descending)
            def success_rate(provider_name):
                usage = self.provider_usage[provider_name]
                if usage["requests"] == 0:
                    return 1.0  # Prioritize untested providers
                return usage["successes"] / usage["requests"]
            
            return sorted(providers, key=success_rate, reverse=True)
        
        elif self.load_balancing == LoadBalancingStrategy.HEALTH_WEIGHTED:
            # Weight by health status and success rate
            def health_weight(provider_name):
                provider = self.providers[provider_name]
                usage = self.provider_usage[provider_name]
                
                # Base weight by health status
                if provider.health.status == ProviderStatus.HEALTHY:
                    weight = 1.0
                elif provider.health.status == ProviderStatus.DEGRADED:
                    weight = 0.5
                else:
                    weight = 0.1
                
                # Adjust by success rate
                if usage["requests"] > 0:
                    success_rate = usage["successes"] / usage["requests"]
                    weight *= success_rate
                
                return weight
            
            return sorted(providers, key=health_weight, reverse=True)
        
        else:  # RANDOM or default
            import random
            shuffled = providers.copy()
            random.shuffle(shuffled)
            return shuffled
    
    def _is_circuit_breaker_open(self, provider_name: str) -> bool:
        """Check if circuit breaker is open for a provider."""
        if not self.circuit_breaker_enabled:
            return False
        
        usage = self.provider_usage[provider_name]
        
        # If circuit breaker is open, check if timeout has passed
        if usage["circuit_breaker_open"]:
            if time.time() - usage["circuit_breaker_opened_at"] > self.circuit_breaker_timeout:
                # Reset circuit breaker
                usage["circuit_breaker_open"] = False
                logger.info(f"Circuit breaker reset for {provider_name}")
                return False
            return True
        
        return False
    
    def _check_circuit_breaker(self, provider_name: str):
        """Check if circuit breaker should be opened."""
        if not self.circuit_breaker_enabled:
            return
        
        usage = self.provider_usage[provider_name]
        
        # Check failure rate in recent requests
        recent_requests = min(usage["requests"], 10)  # Last 10 requests
        if recent_requests >= self.circuit_breaker_threshold:
            recent_failures = usage["failures"]  # Simplified - should track recent failures
            
            if recent_failures >= self.circuit_breaker_threshold:
                usage["circuit_breaker_open"] = True
                usage["circuit_breaker_opened_at"] = time.time()
                logger.warning(f"Circuit breaker opened for {provider_name}")
    
    async def _check_health_if_needed(self):
        """Check provider health if interval has passed."""
        current_time = time.time()
        
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        logger.debug("Performing health checks on all providers")
        
        # Check health of all providers concurrently
        health_tasks = {
            name: provider.health_check()
            for name, provider in self.providers.items()
        }
        
        health_results = await asyncio.gather(*health_tasks.values(), return_exceptions=True)
        
        for (name, task), result in zip(health_tasks.items(), health_results):
            if isinstance(result, Exception):
                logger.error(f"Health check failed for {name}: {str(result)}")
            else:
                logger.debug(f"Health check for {name}: {result.status}")
        
        self.last_health_check = current_time
    
    async def health_check(self) -> Dict[str, Any]:
        """Get comprehensive health status of all providers."""
        health_results = {}
        
        # Check each provider
        for name, provider in self.providers.items():
            try:
                health = await provider.health_check()
                usage = self.provider_usage[name]
                
                health_results[name] = {
                    "status": health.status.value,
                    "response_time": health.response_time,
                    "last_check": health.last_check,
                    "error_count": health.error_count,
                    "consecutive_failures": health.consecutive_failures,
                    "last_error": health.last_error,
                    "usage": {
                        "requests": usage["requests"],
                        "successes": usage["successes"],
                        "failures": usage["failures"],
                        "success_rate": (
                            usage["successes"] / max(usage["requests"], 1) * 100
                        ),
                        "avg_response_time": (
                            usage["total_response_time"] / max(usage["successes"], 1)
                        )
                    },
                    "circuit_breaker_open": usage["circuit_breaker_open"]
                }
            except Exception as e:
                health_results[name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Overall status
        healthy_providers = [
            name for name, health in health_results.items()
            if health.get("status") == "healthy"
        ]
        
        return {
            "overall_status": "healthy" if healthy_providers else "unhealthy",
            "healthy_providers": len(healthy_providers),
            "total_providers": len(self.providers),
            "primary_provider": self.primary_provider,
            "failover_strategy": self.failover_strategy.value,
            "load_balancing": self.load_balancing.value,
            "total_requests": self.total_requests,
            "failover_count": self.failover_count,
            "providers": health_results
        }
    
    def get_provider_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics for all providers."""
        stats = {}
        
        for name, provider in self.providers.items():
            provider_stats = provider.get_statistics()
            usage = self.provider_usage[name]
            
            stats[name] = {
                **provider_stats,
                "unified_usage": usage,
                "circuit_breaker_open": usage["circuit_breaker_open"]
            }
        
        return {
            "providers": stats,
            "unified_stats": {
                "total_requests": self.total_requests,
                "failover_count": self.failover_count,
                "failover_rate": (
                    self.failover_count / max(self.total_requests, 1) * 100
                ),
                "primary_provider": self.primary_provider,
                "failover_strategy": self.failover_strategy.value,
                "load_balancing": self.load_balancing.value
            }
        }
    
    def set_primary_provider(self, provider_name: str):
        """Set the primary provider."""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found")
        
        self.primary_provider = provider_name
        logger.info(f"Primary provider set to: {provider_name}")
    
    def add_provider(self, name: str, provider: BaseLLMProvider):
        """Add a new provider."""
        self.providers[name] = provider
        self.provider_usage[name] = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "total_response_time": 0.0,
            "circuit_breaker_open": False,
            "circuit_breaker_opened_at": 0
        }
        
        if name not in self.provider_priorities:
            self.provider_priorities.append(name)
        
        logger.info(f"Added provider: {name}")
    
    def remove_provider(self, name: str):
        """Remove a provider."""
        if name in self.providers:
            del self.providers[name]
            del self.provider_usage[name]
            
            if name in self.provider_priorities:
                self.provider_priorities.remove(name)
            
            if self.primary_provider == name:
                self.primary_provider = self.provider_priorities[0] if self.provider_priorities else None
            
            logger.info(f"Removed provider: {name}")