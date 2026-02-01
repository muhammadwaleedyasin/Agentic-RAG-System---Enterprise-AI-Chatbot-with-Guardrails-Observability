"""
Utility functions and helpers for LLM providers.
"""
import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base_provider import BaseLLMProvider, LLMMessage, LLMResponse, ProviderHealth, ProviderStatus

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of provider benchmark test."""
    provider_name: str
    requests_per_second: float
    average_response_time: float
    p95_response_time: float
    success_rate: float
    total_requests: int
    failed_requests: int
    errors: List[str]


@dataclass
class ProviderMetrics:
    """Comprehensive provider metrics."""
    provider_name: str
    uptime_percentage: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    p95_response_time: float
    p99_response_time: float
    tokens_generated: int
    cost_estimate: float
    last_24h_requests: int
    health_status: ProviderStatus


class ProviderBenchmark:
    """Benchmark utility for testing provider performance."""
    
    def __init__(self):
        self.results = {}
        self.response_times = {}
    
    async def run_benchmark(
        self,
        provider: BaseLLMProvider,
        test_messages: List[LLMMessage],
        num_requests: int = 10,
        concurrent_requests: int = 3,
        timeout: float = 60.0
    ) -> BenchmarkResult:
        """
        Run comprehensive benchmark test on a provider.
        
        Args:
            provider: Provider to benchmark
            test_messages: Test messages to use
            num_requests: Total number of requests to send
            concurrent_requests: Number of concurrent requests
            timeout: Overall timeout for the benchmark
            
        Returns:
            BenchmarkResult with performance metrics
        """
        provider_name = provider.get_provider_name()
        logger.info(f"Starting benchmark for {provider_name} with {num_requests} requests")
        
        start_time = time.time()
        response_times = []
        errors = []
        successful_requests = 0
        failed_requests = 0
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def make_request():
            nonlocal successful_requests, failed_requests
            
            async with semaphore:
                request_start = time.time()
                try:
                    response = await asyncio.wait_for(
                        provider.generate(test_messages),
                        timeout=timeout / num_requests
                    )
                    request_time = time.time() - request_start
                    response_times.append(request_time)
                    successful_requests += 1
                    
                except Exception as e:
                    failed_requests += 1
                    errors.append(str(e))
                    logger.debug(f"Request failed: {str(e)}")
        
        # Run all requests
        tasks = [make_request() for _ in range(num_requests)]
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Benchmark timeout for {provider_name}")
            failed_requests += len([t for t in tasks if not t.done()])
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        requests_per_second = num_requests / total_time if total_time > 0 else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        p95_response_time = (
            sorted(response_times)[int(len(response_times) * 0.95)] 
            if response_times else 0
        )
        success_rate = successful_requests / num_requests * 100 if num_requests > 0 else 0
        
        result = BenchmarkResult(
            provider_name=provider_name,
            requests_per_second=requests_per_second,
            average_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            success_rate=success_rate,
            total_requests=num_requests,
            failed_requests=failed_requests,
            errors=errors[:10]  # Limit error list
        )
        
        logger.info(
            f"Benchmark complete for {provider_name}: "
            f"{requests_per_second:.2f} RPS, {success_rate:.1f}% success rate"
        )
        
        return result
    
    async def compare_providers(
        self,
        providers: Dict[str, BaseLLMProvider],
        test_messages: List[LLMMessage],
        num_requests: int = 10
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare performance across multiple providers.
        
        Args:
            providers: Dictionary of provider name to provider instance
            test_messages: Test messages to use
            num_requests: Number of requests per provider
            
        Returns:
            Dictionary of provider names to benchmark results
        """
        logger.info(f"Comparing {len(providers)} providers")
        
        # Run benchmarks concurrently
        tasks = {
            name: self.run_benchmark(provider, test_messages, num_requests)
            for name, provider in providers.items()
        }
        
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Benchmark failed for {name}: {str(e)}")
                results[name] = BenchmarkResult(
                    provider_name=name,
                    requests_per_second=0,
                    average_response_time=0,
                    p95_response_time=0,
                    success_rate=0,
                    total_requests=0,
                    failed_requests=num_requests,
                    errors=[str(e)]
                )
        
        return results


class ProviderMonitor:
    """Monitor provider health and performance over time."""
    
    def __init__(self):
        self.metrics_history = {}
        self.alert_thresholds = {
            "response_time": 5.0,  # seconds
            "error_rate": 10.0,    # percentage
            "availability": 95.0    # percentage
        }
    
    def record_request(
        self,
        provider_name: str,
        response_time: float,
        success: bool,
        tokens: int = 0,
        cost: float = 0.0
    ):
        """Record a request for monitoring."""
        if provider_name not in self.metrics_history:
            self.metrics_history[provider_name] = []
        
        record = {
            "timestamp": time.time(),
            "response_time": response_time,
            "success": success,
            "tokens": tokens,
            "cost": cost
        }
        
        self.metrics_history[provider_name].append(record)
        
        # Keep only last 24 hours of data
        cutoff_time = time.time() - 86400  # 24 hours
        self.metrics_history[provider_name] = [
            r for r in self.metrics_history[provider_name]
            if r["timestamp"] > cutoff_time
        ]
    
    def get_metrics(self, provider_name: str, hours: int = 24) -> Optional[ProviderMetrics]:
        """Get metrics for a provider over the specified time period."""
        if provider_name not in self.metrics_history:
            return None
        
        cutoff_time = time.time() - (hours * 3600)
        records = [
            r for r in self.metrics_history[provider_name]
            if r["timestamp"] > cutoff_time
        ]
        
        if not records:
            return None
        
        # Calculate metrics
        total_requests = len(records)
        successful_requests = sum(1 for r in records if r["success"])
        failed_requests = total_requests - successful_requests
        
        response_times = [r["response_time"] for r in records if r["success"]]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        sorted_times = sorted(response_times)
        p95_response_time = (
            sorted_times[int(len(sorted_times) * 0.95)] 
            if sorted_times else 0
        )
        p99_response_time = (
            sorted_times[int(len(sorted_times) * 0.99)] 
            if sorted_times else 0
        )
        
        uptime_percentage = successful_requests / total_requests * 100 if total_requests > 0 else 0
        tokens_generated = sum(r["tokens"] for r in records)
        cost_estimate = sum(r["cost"] for r in records)
        
        # Determine health status
        error_rate = failed_requests / total_requests * 100 if total_requests > 0 else 0
        
        if error_rate < 1 and avg_response_time < self.alert_thresholds["response_time"]:
            health_status = ProviderStatus.HEALTHY
        elif error_rate < 5 and avg_response_time < self.alert_thresholds["response_time"] * 2:
            health_status = ProviderStatus.DEGRADED
        else:
            health_status = ProviderStatus.UNHEALTHY
        
        return ProviderMetrics(
            provider_name=provider_name,
            uptime_percentage=uptime_percentage,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            tokens_generated=tokens_generated,
            cost_estimate=cost_estimate,
            last_24h_requests=total_requests,
            health_status=health_status
        )
    
    def get_alerts(self, provider_name: str) -> List[Dict[str, Any]]:
        """Get active alerts for a provider."""
        metrics = self.get_metrics(provider_name, hours=1)  # Last hour
        if not metrics:
            return []
        
        alerts = []
        
        # Response time alert
        if metrics.average_response_time > self.alert_thresholds["response_time"]:
            alerts.append({
                "type": "high_response_time",
                "message": f"Average response time {metrics.average_response_time:.2f}s exceeds threshold {self.alert_thresholds['response_time']}s",
                "severity": "warning"
            })
        
        # Error rate alert
        error_rate = metrics.failed_requests / max(metrics.total_requests, 1) * 100
        if error_rate > self.alert_thresholds["error_rate"]:
            alerts.append({
                "type": "high_error_rate",
                "message": f"Error rate {error_rate:.1f}% exceeds threshold {self.alert_thresholds['error_rate']}%",
                "severity": "critical"
            })
        
        # Availability alert
        if metrics.uptime_percentage < self.alert_thresholds["availability"]:
            alerts.append({
                "type": "low_availability",
                "message": f"Availability {metrics.uptime_percentage:.1f}% below threshold {self.alert_thresholds['availability']}%",
                "severity": "critical"
            })
        
        return alerts


def create_test_messages() -> List[LLMMessage]:
    """Create standard test messages for benchmarking."""
    return [
        LLMMessage(
            role="system",
            content="You are a helpful assistant. Respond concisely to user questions."
        ),
        LLMMessage(
            role="user",
            content="What is the capital of France? Please provide a brief answer."
        )
    ]


def create_stress_test_messages() -> List[LLMMessage]:
    """Create messages for stress testing with longer context."""
    long_context = "Here's a long document about artificial intelligence: " + "AI is transforming industries. " * 100
    
    return [
        LLMMessage(
            role="system",
            content="You are an AI expert. Analyze the provided document and answer questions."
        ),
        LLMMessage(
            role="user",
            content=f"{long_context}\n\nPlease summarize the main points about AI transformation."
        )
    ]


async def test_provider_streaming(provider: BaseLLMProvider, messages: List[LLMMessage]) -> Dict[str, Any]:
    """Test streaming functionality of a provider."""
    start_time = time.time()
    chunks_received = 0
    total_content_length = 0
    errors = []
    
    try:
        async for chunk in provider.stream(messages):
            chunks_received += 1
            if chunk.content:
                total_content_length += len(chunk.content)
            if chunk.error:
                errors.append(chunk.error)
            if chunk.is_final:
                break
        
        response_time = time.time() - start_time
        
        return {
            "success": True,
            "response_time": response_time,
            "chunks_received": chunks_received,
            "total_content_length": total_content_length,
            "streaming_speed": total_content_length / response_time if response_time > 0 else 0,
            "errors": errors
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response_time": time.time() - start_time,
            "chunks_received": chunks_received,
            "errors": errors + [str(e)]
        }


def format_benchmark_report(results: Dict[str, BenchmarkResult]) -> str:
    """Format benchmark results into a readable report."""
    report = ["Provider Performance Benchmark Report", "=" * 40, ""]
    
    # Sort providers by success rate, then by RPS
    sorted_providers = sorted(
        results.items(),
        key=lambda x: (x[1].success_rate, x[1].requests_per_second),
        reverse=True
    )
    
    for name, result in sorted_providers:
        report.extend([
            f"Provider: {name}",
            f"  Requests/second: {result.requests_per_second:.2f}",
            f"  Average response time: {result.average_response_time:.3f}s",
            f"  95th percentile: {result.p95_response_time:.3f}s",
            f"  Success rate: {result.success_rate:.1f}%",
            f"  Failed requests: {result.failed_requests}/{result.total_requests}",
            ""
        ])
        
        if result.errors:
            report.append("  Recent errors:")
            for error in result.errors[:3]:  # Show first 3 errors
                report.append(f"    - {error[:80]}{'...' if len(error) > 80 else ''}")
            report.append("")
    
    return "\n".join(report)


def estimate_token_cost(
    provider_name: str,
    input_tokens: int,
    output_tokens: int,
    model_name: str = None
) -> float:
    """
    Estimate cost for token usage (simplified pricing model).
    
    Note: This is a simplified estimation. Real pricing should be fetched
    from provider APIs or configuration files.
    """
    # Simplified pricing per 1K tokens (USD)
    pricing = {
        "openrouter": {
            "default": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "claude-3": {"input": 0.015, "output": 0.075},
            "llama-2": {"input": 0.0007, "output": 0.0009}
        },
        "vllm": {
            "default": {"input": 0.0, "output": 0.0}  # Local models are free
        }
    }
    
    provider_pricing = pricing.get(provider_name.lower(), pricing["openrouter"])
    
    # Try to find model-specific pricing
    model_pricing = provider_pricing["default"]
    if model_name:
        for key in provider_pricing:
            if key in model_name.lower():
                model_pricing = provider_pricing[key]
                break
    
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    
    return input_cost + output_cost


# Global monitor instance
provider_monitor = ProviderMonitor()