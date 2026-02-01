"""
Example usage of LLM providers with the Enterprise RAG system.
"""
import asyncio
import logging
from typing import Dict, Any

from src.providers.provider_factory import create_llm_provider, llm_factory, ProviderMode
from src.providers.base_provider import LLMMessage, ProviderStatus
from src.config.settings import LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_single_provider():
    """Example using a single provider."""
    print("\n=== Single Provider Example ===")
    
    # Create vLLM provider
    try:
        provider = create_llm_provider(
            provider_type=LLMProvider.VLLM,
            unified=False  # Single provider mode
        )
        
        # Test health check
        health = await provider.health_check()
        print(f"Provider health: {health.status}")
        
        if health.status == ProviderStatus.HEALTHY:
            # Generate response
            messages = [
                LLMMessage(role="user", content="What is machine learning?")
            ]
            
            response = await provider.generate(messages, max_tokens=100)
            print(f"Response: {response.content}")
            print(f"Model: {response.model}")
            print(f"Response time: {response.response_time:.2f}s")
            
        # Get provider statistics
        stats = provider.get_statistics()
        print(f"Provider stats: {stats}")
        
    except Exception as e:
        print(f"Single provider error: {str(e)}")


async def example_unified_provider():
    """Example using unified provider with fallback."""
    print("\n=== Unified Provider Example ===")
    
    try:
        # Create unified provider
        provider = create_llm_provider(unified=True)
        
        # Check overall health
        health_status = await provider.health_check()
        print(f"Overall status: {health_status['overall_status']}")
        print(f"Healthy providers: {health_status['healthy_providers']}/{health_status['total_providers']}")
        
        # Generate response with automatic failover
        messages = [
            LLMMessage(role="user", content="Explain the benefits of vector databases.")
        ]
        
        response = await provider.generate(messages, max_tokens=150)
        print(f"Response: {response.content}")
        print(f"Provider used: {response.provider}")
        print(f"Fallback used: {response.fallback_used}")
        
        # Get detailed statistics
        stats = provider.get_provider_statistics()
        print(f"\nUnified stats: {stats['unified_stats']}")
        
    except Exception as e:
        print(f"Unified provider error: {str(e)}")


async def example_streaming():
    """Example of streaming responses."""
    print("\n=== Streaming Example ===")
    
    try:
        provider = create_llm_provider(unified=True)
        
        messages = [
            LLMMessage(role="user", content="Write a short poem about AI.")
        ]
        
        print("Streaming response:")
        full_content = ""
        
        async for chunk in provider.stream(messages, max_tokens=200):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_content += chunk.content
            
            if chunk.is_final:
                print(f"\n\nFinal chunk - Provider: {chunk.provider}")
                if chunk.usage:
                    print(f"Usage: {chunk.usage}")
                break
        
    except Exception as e:
        print(f"Streaming error: {str(e)}")


async def example_provider_management():
    """Example of provider management and configuration."""
    print("\n=== Provider Management Example ===")
    
    # Get available providers
    available = llm_factory.get_available_providers()
    print("Available providers:")
    for name, info in available.items():
        print(f"  {name}: {'✓' if info['available'] else '✗'} - {info.get('error', 'OK')}")
    
    # Custom configuration example
    custom_config = {
        "providers": {
            "vllm_custom": {
                "type": "vllm",
                "model_name": "custom-model",
                "base_url": "http://localhost:8001",
                "max_tokens": 1024,
                "temperature": 0.8,
                "timeout": 30,
                "priority": 1
            },
            "openrouter_custom": {
                "type": "openrouter",
                "model_name": "anthropic/claude-3-haiku",
                "api_key": "your-api-key",
                "max_tokens": 2048,
                "temperature": 0.7,
                "fallback_models": ["openai/gpt-3.5-turbo"],
                "priority": 2
            }
        },
        "failover_strategy": "priority",
        "load_balancing": "response_time",
        "circuit_breaker_enabled": True
    }
    
    try:
        custom_provider = llm_factory.create_provider(
            mode=ProviderMode.UNIFIED,
            config=custom_config
        )
        print("Custom unified provider created successfully")
        
        # Test with custom configuration
        health = await custom_provider.health_check()
        print(f"Custom provider health: {health}")
        
    except Exception as e:
        print(f"Custom provider error: {str(e)}")
    
    # Cache statistics
    cache_stats = llm_factory.get_cache_stats()
    print(f"Cache stats: {cache_stats}")


async def example_error_handling():
    """Example of error handling and recovery."""
    print("\n=== Error Handling Example ===")
    
    # Test with invalid configuration
    try:
        invalid_config = {
            "model_name": "",  # Invalid empty model name
            "base_url": "invalid-url",
            "api_key": "fake-key"
        }
        
        provider = llm_factory.create_from_config({
            "type": "vllm",
            **invalid_config
        })
        
    except Exception as e:
        print(f"Expected error with invalid config: {str(e)}")
    
    # Test provider resilience
    try:
        provider = create_llm_provider(unified=True)
        
        # Simulate a request that might fail
        messages = [LLMMessage(role="user", content="Test message")]
        
        # The unified provider should handle failures gracefully
        response = await provider.generate(messages, max_tokens=50)
        print(f"Resilient response: {response.content[:100]}...")
        
    except Exception as e:
        print(f"Error in resilience test: {str(e)}")


async def example_cost_optimization():
    """Example of cost-optimized configuration."""
    print("\n=== Cost Optimization Example ===")
    
    # Cost-optimized configuration
    cost_config = {
        "providers": {
            "vllm_local": {
                "type": "vllm",
                "model_name": "microsoft/DialoGPT-medium",  # Smaller model
                "base_url": "http://localhost:8001",
                "max_tokens": 512,  # Reduced for cost
                "temperature": 0.7,
                "priority": 1  # Prefer local first
            },
            "openrouter_cheap": {
                "type": "openrouter",
                "model_name": "meta-llama/llama-2-7b-chat",  # Cheaper model
                "api_key": "your-api-key",
                "max_tokens": 512,
                "max_cost_per_request": 0.01,  # Cost limit
                "fallback_models": ["openai/gpt-3.5-turbo"],
                "priority": 2
            }
        },
        "failover_strategy": "priority",  # Always try local first
        "circuit_breaker_enabled": True
    }
    
    try:
        cost_provider = llm_factory.create_provider(
            mode=ProviderMode.UNIFIED,
            config=cost_config
        )
        
        print("Cost-optimized provider created")
        
        # Test cost-conscious generation
        messages = [
            LLMMessage(role="user", content="Briefly explain what is Python programming?")
        ]
        
        response = await cost_provider.generate(messages, max_tokens=100)
        print(f"Cost-optimized response: {response.content}")
        print(f"Provider used: {response.provider}")
        
    except Exception as e:
        print(f"Cost optimization error: {str(e)}")


async def main():
    """Run all examples."""
    print("LLM Provider Usage Examples")
    print("=" * 50)
    
    # Run examples
    await example_single_provider()
    await example_unified_provider()
    await example_streaming()
    await example_provider_management()
    await example_error_handling()
    await example_cost_optimization()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
