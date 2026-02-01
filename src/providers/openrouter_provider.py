"""
OpenRouter provider implementation for cloud LLM inference.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import aiohttp
import json
import logging

from .base_provider import BaseLLMProvider, LLMMessage, LLMResponse, LLMStreamChunk, BaseProviderError, ProviderUnavailableError, ProviderConfigError

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider for cloud LLM inference."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        # Ensure base_url doesn't end with slash
        self.base_url = self.base_url.rstrip("/")
        
        # OpenRouter specific parameters
        self.top_p = config.get("top_p", 1.0)
        self.presence_penalty = config.get("presence_penalty", 0.0)
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.repetition_penalty = config.get("repetition_penalty", 1.0)
        self.app_name = config.get("app_name", "Enterprise RAG Chatbot")
        self.http_referer = config.get("http_referer", "https://github.com/enterprise-rag")
        self.site_url = config.get("site_url")
        
        # Fallback models for auto-failover
        self.fallback_models = config.get("fallback_models", [])
        self.max_cost_per_request = config.get("max_cost_per_request")
        
        # Current model tracking for fallbacks
        self.current_model = self.model_name
        self.model_attempts = 0
    
    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Internal implementation of generate method for OpenRouter."""
        return await self._generate_with_fallback(messages, max_tokens, temperature, **kwargs)
    
    async def _generate_with_fallback(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate with automatic model fallback support."""
        models_to_try = [self.current_model] + self.fallback_models
        last_error = None
        
        for model in models_to_try:
            try:
                return await self._generate_with_model(model, messages, max_tokens, temperature, **kwargs)
                    
            except BaseProviderError as e:
                last_error = e
                logger.warning(f"Model {model} failed: {str(e)}")
                
                # Check if we should try fallback
                if not e.retryable or model == models_to_try[-1]:
                    raise e
                continue
                
        raise last_error or BaseProviderError("All fallback models failed")
    
    async def _stream_with_fallback(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Stream with automatic model fallback support."""
        models_to_try = [self.current_model] + self.fallback_models
        last_error = None
        
        for model in models_to_try:
            try:
                async for chunk in self._stream_with_model(model, messages, max_tokens, temperature, **kwargs):
                    yield chunk
                return
                    
            except BaseProviderError as e:
                last_error = e
                logger.warning(f"Model {model} failed in streaming: {str(e)}")
                
                # Check if we should try fallback
                if not e.retryable or model == models_to_try[-1]:
                    yield LLMStreamChunk(
                        content="",
                        is_final=True,
                        error=str(e)
                    )
                    return
                continue
                
        # If we get here, all models failed
        yield LLMStreamChunk(
            content="",
            is_final=True,
            error="All fallback models failed"
        )
    
    async def _generate_with_model(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate with a specific model."""
        start_time = time.time()
        
        # Format request for OpenRouter API
        request_data = {
            "model": model,
            "messages": self.format_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "stream": False,
            **kwargs
        }
        
        # Add cost control if configured
        if self.max_cost_per_request:
            request_data["max_cost"] = self.max_cost_per_request
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                headers = self._get_headers()
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=request_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        choice = data["choices"][0]
                        
                        return LLMResponse(
                            content=choice["message"]["content"],
                            usage=data.get("usage"),
                            model=data.get("model", model),
                            finish_reason=choice.get("finish_reason"),
                            response_time=time.time() - start_time,
                            fallback_used=(model != self.model_name)
                        )
                    elif response.status == 402:
                        raise ProviderConfigError("OpenRouter: Insufficient credits or cost limit exceeded")
                    elif response.status == 429:
                        raise ProviderUnavailableError("OpenRouter: Rate limit exceeded")
                    elif response.status in [502, 503, 504]:
                        error_text = await response.text()
                        raise ProviderUnavailableError(f"OpenRouter API temporarily unavailable: {error_text}")
                    elif response.status == 400:
                        error_text = await response.text()
                        raise BaseProviderError(f"OpenRouter API bad request: {error_text}", retryable=False)
                    else:
                        error_text = await response.text()
                        raise BaseProviderError(f"OpenRouter API error {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            raise ProviderUnavailableError(f"OpenRouter request timed out after {self.timeout} seconds")
        except aiohttp.ClientConnectorError:
            raise ProviderUnavailableError("Cannot connect to OpenRouter API")
        except (BaseProviderError, ProviderUnavailableError, ProviderConfigError):
            raise
        except Exception as e:
            logger.error(f"OpenRouter generation error: {str(e)}")
            raise BaseProviderError(f"OpenRouter generation failed: {str(e)}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.http_referer,
            "X-Title": self.app_name
        }
        
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
            
        return headers
    
    async def _stream_impl(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Internal implementation of stream method for OpenRouter."""
        async for chunk in self._stream_with_fallback(messages, max_tokens, temperature, **kwargs):
            yield chunk
    
    async def _stream_with_model(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Stream with a specific model."""
        request_data = {
            "model": model,
            "messages": self.format_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "stream": True,
            **kwargs
        }
        
        # Add cost control if configured
        if self.max_cost_per_request:
            request_data["max_cost"] = self.max_cost_per_request
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                headers = self._get_headers()
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=request_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                line = line[6:]  # Remove 'data: ' prefix
                                
                                if line == '[DONE]':
                                    yield LLMStreamChunk(
                                        content="", 
                                        is_final=True
                                    )
                                    break
                                
                                try:
                                    data = json.loads(line)
                                    if "choices" in data and data["choices"]:
                                        choice = data["choices"][0]
                                        delta = choice.get("delta", {})
                                        content = delta.get("content", "")
                                        
                                        if content:
                                            yield LLMStreamChunk(
                                                content=content,
                                                is_final=choice.get("finish_reason") is not None
                                            )
                                        
                                        if choice.get("finish_reason"):
                                            usage = data.get("usage")
                                            yield LLMStreamChunk(
                                                content="",
                                                is_final=True,
                                                usage=usage,
                                                finish_reason=choice.get("finish_reason")
                                            )
                                            break
                                            
                                except json.JSONDecodeError:
                                    continue  # Skip malformed lines
                    elif response.status == 402:
                        raise ProviderConfigError("OpenRouter: Insufficient credits or cost limit exceeded")
                    elif response.status == 429:
                        raise ProviderUnavailableError("OpenRouter: Rate limit exceeded")
                    elif response.status in [502, 503, 504]:
                        error_text = await response.text()
                        raise ProviderUnavailableError(f"OpenRouter streaming temporarily unavailable: {error_text}")
                    else:
                        error_text = await response.text()
                        raise BaseProviderError(f"OpenRouter streaming error {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            raise ProviderUnavailableError(f"OpenRouter streaming timed out after {self.timeout} seconds")
        except aiohttp.ClientConnectorError:
            raise ProviderUnavailableError("Cannot connect to OpenRouter API")
        except (BaseProviderError, ProviderUnavailableError, ProviderConfigError):
            raise
        except Exception as e:
            logger.error(f"OpenRouter streaming error: {str(e)}")
            raise BaseProviderError(f"OpenRouter streaming failed: {str(e)}")
    
    async def _health_check_impl(self) -> bool:
        """Internal implementation of health check for OpenRouter."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                headers = self._get_headers()
                
                async with session.get(f"{self.base_url}/models", headers=headers) as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"OpenRouter health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenRouter model information."""
        return {
            "provider": "openrouter",
            "model_name": self.model_name,
            "current_model": self.current_model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "supports_streaming": True,
            "supports_function_calling": True,  # Many OpenRouter models support this
            "fallback_models": self.fallback_models,
            "max_cost_per_request": self.max_cost_per_request,
            "model_attempts": self.model_attempts
        }
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = self._get_headers()
                
                async with session.get(f"{self.base_url}/models", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", [])
                    else:
                        logger.error(f"Failed to fetch models: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching available models: {str(e)}")
            return []
    
    def set_fallback_models(self, models: List[str]):
        """Update fallback models list."""
        self.fallback_models = models
        logger.info(f"Updated fallback models: {models}")
    
    def reset_model_attempts(self):
        """Reset model attempt counter."""
        self.model_attempts = 0
        self.current_model = self.model_name
