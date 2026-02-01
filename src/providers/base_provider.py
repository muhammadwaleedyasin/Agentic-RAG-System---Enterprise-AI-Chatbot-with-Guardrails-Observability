"""
Enhanced base LLM provider with comprehensive error handling and fallback mechanisms.
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from pydantic import BaseModel
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderStatus(str, Enum):
    """Provider status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class LLMMessage(BaseModel):
    """Message model for LLM interactions."""
    role: str  # "user", "assistant", "system"
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    """Response model from LLM providers."""
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    response_time: Optional[float] = None
    provider: Optional[str] = None
    fallback_used: bool = False
    error_count: int = 0


class LLMStreamChunk(BaseModel):
    """Streaming chunk from LLM providers."""
    content: str
    is_final: bool = False
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    provider: Optional[str] = None
    error: Optional[str] = None


class ProviderHealth(BaseModel):
    """Provider health information."""
    status: ProviderStatus
    response_time: Optional[float] = None
    last_check: float
    error_count: int = 0
    last_error: Optional[str] = None
    consecutive_failures: int = 0


class BaseProviderError(Exception):
    """Base exception for provider errors."""
    def __init__(self, message: str, retryable: bool = True):
        self.message = message
        self.retryable = retryable
        super().__init__(message)


class ProviderUnavailableError(BaseProviderError):
    """Provider is temporarily unavailable."""
    pass


class ProviderConfigError(BaseProviderError):
    """Provider configuration error."""
    def __init__(self, message: str):
        super().__init__(message, retryable=False)


class BaseLLMProvider(ABC):
    """Enhanced abstract base class for LLM providers with fallback support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
        self.model_name = config.get("model_name")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 60)
        
        # Reliability configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.health_check_interval = config.get("health_check_interval", 300)  # 5 minutes
        self.failure_threshold = config.get("failure_threshold", 5)
        
        # Provider state
        self.health = ProviderHealth(
            status=ProviderStatus.UNKNOWN,
            last_check=0,
            error_count=0,
            consecutive_failures=0
        )
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
    
    @abstractmethod
    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Internal implementation of generate method."""
        pass
    
    @abstractmethod
    async def _stream_impl(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Internal implementation of stream method."""
        pass
    
    @abstractmethod
    async def _health_check_impl(self) -> bool:
        """Internal implementation of health check."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass
    
    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response with retry logic and error handling."""
        start_time = time.time()
        self.total_requests += 1
        
        for attempt in range(self.max_retries + 1):
            try:
                # Validate parameters
                max_tokens, temperature = self.validate_parameters(max_tokens, temperature)
                
                # Check provider health before request
                if self.health.status == ProviderStatus.UNHEALTHY:
                    raise ProviderUnavailableError("Provider is marked as unhealthy")
                
                # Call implementation
                response = await self._generate_impl(messages, max_tokens, temperature, **kwargs)
                
                # Update statistics on success
                response.response_time = time.time() - start_time
                response.provider = self.get_provider_name()
                
                self.successful_requests += 1
                self.total_response_time += response.response_time
                
                # Reset consecutive failures on success
                self.health.consecutive_failures = 0
                if self.health.status != ProviderStatus.HEALTHY:
                    self.health.status = ProviderStatus.HEALTHY
                
                return response
                
            except Exception as e:
                self.failed_requests += 1
                self.health.error_count += 1
                self.health.consecutive_failures += 1
                self.health.last_error = str(e)
                
                # Update health status based on failures
                if self.health.consecutive_failures >= self.failure_threshold:
                    self.health.status = ProviderStatus.UNHEALTHY
                elif self.health.consecutive_failures >= 2:
                    self.health.status = ProviderStatus.DEGRADED
                
                # Check if we should retry
                if attempt < self.max_retries and isinstance(e, BaseProviderError) and e.retryable:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Provider failed after {attempt + 1} attempts: {str(e)}")
                    raise e
        
        # This should never be reached
        raise Exception("Maximum retries exceeded")
    
    async def stream(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Generate a streaming response with error handling."""
        self.total_requests += 1
        
        try:
            # Validate parameters
            max_tokens, temperature = self.validate_parameters(max_tokens, temperature)
            
            # Check provider health
            if self.health.status == ProviderStatus.UNHEALTHY:
                yield LLMStreamChunk(
                    content="",
                    is_final=True,
                    error="Provider is currently unhealthy"
                )
                return
            
            # Stream implementation
            chunk_count = 0
            async for chunk in self._stream_impl(messages, max_tokens, temperature, **kwargs):
                chunk.provider = self.get_provider_name()
                chunk_count += 1
                yield chunk
                
                if chunk.is_final:
                    break
            
            # Update statistics on success
            if chunk_count > 0:
                self.successful_requests += 1
                self.health.consecutive_failures = 0
                
        except Exception as e:
            self.failed_requests += 1
            self.health.error_count += 1
            self.health.consecutive_failures += 1
            self.health.last_error = str(e)
            
            logger.error(f"Streaming failed: {str(e)}")
            yield LLMStreamChunk(
                content="",
                is_final=True,
                error=str(e)
            )
    
    async def health_check(self) -> ProviderHealth:
        """Check provider health with caching."""
        current_time = time.time()
        
        # Use cached result if recent
        if (current_time - self.health.last_check) < self.health_check_interval:
            return self.health
        
        # Perform health check
        start_time = current_time
        try:
            is_healthy = await self._health_check_impl()
            response_time = time.time() - start_time
            
            if is_healthy:
                self.health.status = ProviderStatus.HEALTHY
                self.health.response_time = response_time
                # Don't reset error count, but reduce consecutive failures
                if self.health.consecutive_failures > 0:
                    self.health.consecutive_failures = max(0, self.health.consecutive_failures - 1)
            else:
                self.health.status = ProviderStatus.UNHEALTHY
                self.health.consecutive_failures += 1
                
        except Exception as e:
            self.health.status = ProviderStatus.UNHEALTHY
            self.health.consecutive_failures += 1
            self.health.last_error = str(e)
            logger.error(f"Health check failed: {str(e)}")
        
        self.health.last_check = current_time
        return self.health
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.__class__.__name__.lower().replace('provider', '')
    
    def format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Format messages for the provider's expected format."""
        formatted = []
        for msg in messages:
            formatted_msg = {"role": msg.role, "content": msg.content}
            if msg.name:
                formatted_msg["name"] = msg.name
            if msg.function_call:
                formatted_msg["function_call"] = msg.function_call
            formatted.append(formatted_msg)
        return formatted
    
    def validate_parameters(self, max_tokens: Optional[int], temperature: Optional[float]):
        """Validate and return parameters with defaults."""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        
        # Validate ranges
        if max_tokens <= 0:
            raise ProviderConfigError("max_tokens must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise ProviderConfigError("temperature must be between 0.0 and 2.0")
            
        return max_tokens, temperature
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get provider statistics."""
        avg_response_time = (
            self.total_response_time / max(self.successful_requests, 1)
        )
        
        success_rate = (
            self.successful_requests / max(self.total_requests, 1) * 100
        )
        
        return {
            "provider": self.get_provider_name(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(success_rate, 2),
            "average_response_time": round(avg_response_time, 3),
            "health": self.health.dict(),
            "model_info": self.get_model_info()
        }
    
    def reset_statistics(self):
        """Reset provider statistics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        self.health.error_count = 0
        self.health.consecutive_failures = 0
