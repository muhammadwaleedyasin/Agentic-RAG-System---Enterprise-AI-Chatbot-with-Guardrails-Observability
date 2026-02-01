"""
vLLM provider implementation for local LLM inference.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import aiohttp
import json
import logging

from .base_provider import BaseLLMProvider, LLMMessage, LLMResponse, LLMStreamChunk, BaseProviderError, ProviderUnavailableError

logger = logging.getLogger(__name__)


class VLLMProvider(BaseLLMProvider):
    """vLLM provider for local LLM inference."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:8001")
        self.api_key = config.get("api_key")  # Optional for local vLLM
        
        # Ensure base_url doesn't end with slash
        self.base_url = self.base_url.rstrip("/")
        
        # vLLM specific parameters
        self.presence_penalty = config.get("presence_penalty", 0.0)
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.top_p = config.get("top_p", 1.0)
        self.stop_tokens = config.get("stop_tokens", [])
    
    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Internal implementation of generate method for vLLM."""
        start_time = time.time()
        
        # Format request for vLLM OpenAI-compatible API
        request_data = {
            "model": self.model_name,
            "messages": self.format_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop_tokens if self.stop_tokens else None,
            "stream": False,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout,
                    connect=self.config.get("connect_timeout", 10)
                )
            ) as session:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=request_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        choice = data["choices"][0]
                        
                        return LLMResponse(
                            content=choice["message"]["content"],
                            usage=data.get("usage"),
                            model=data.get("model"),
                            finish_reason=choice.get("finish_reason"),
                            response_time=time.time() - start_time
                        )
                    elif response.status == 503:
                        raise ProviderUnavailableError("vLLM server is temporarily unavailable")
                    elif response.status in [429, 502, 504]:
                        error_text = await response.text()
                        raise BaseProviderError(f"vLLM API error {response.status}: {error_text}", retryable=True)
                    else:
                        error_text = await response.text()
                        raise BaseProviderError(f"vLLM API error {response.status}: {error_text}", retryable=False)
                        
        except asyncio.TimeoutError:
            raise ProviderUnavailableError(f"vLLM request timed out after {self.timeout} seconds")
        except aiohttp.ClientConnectorError:
            raise ProviderUnavailableError("Cannot connect to vLLM server")
        except (aiohttp.ClientError, BaseProviderError):
            raise
        except Exception as e:
            logger.error(f"vLLM generation error: {str(e)}")
            raise BaseProviderError(f"vLLM generation failed: {str(e)}")
    
    async def _stream_impl(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Internal implementation of stream method for vLLM."""
        request_data = {
            "model": self.model_name,
            "messages": self.format_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop_tokens if self.stop_tokens else None,
            "stream": True,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout,
                    connect=self.config.get("connect_timeout", 10)
                )
            ) as session:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
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
                    elif response.status == 503:
                        raise ProviderUnavailableError("vLLM server is temporarily unavailable")
                    else:
                        error_text = await response.text()
                        raise BaseProviderError(f"vLLM streaming error {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            raise ProviderUnavailableError(f"vLLM streaming timed out after {self.timeout} seconds")
        except aiohttp.ClientConnectorError:
            raise ProviderUnavailableError("Cannot connect to vLLM server")
        except (BaseProviderError, ProviderUnavailableError):
            raise
        except Exception as e:
            logger.error(f"vLLM streaming error: {str(e)}")
            raise BaseProviderError(f"vLLM streaming failed: {str(e)}")
    
    async def _health_check_impl(self) -> bool:
        """Internal implementation of health check for vLLM."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # First try the health endpoint
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            return True
                except:
                    pass
                
                # Fallback: try to get models endpoint
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                async with session.get(f"{self.base_url}/v1/models", headers=headers) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.debug(f"vLLM health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM model information."""
        return {
            "provider": "vllm",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "supports_streaming": True,
            "supports_function_calling": False  # Most vLLM models don't support this
        }
