"""Performance and load testing for Enterprise RAG Chatbot."""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch
import concurrent.futures
import psutil
import sys

from tests.performance.load_tester import RAGLoadTester, PerformanceMetrics
from tests.performance.benchmarker import RAGBenchmarker


class TestLoadTesting:
    """Test cases for RAG system load testing."""

    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create mock RAG pipeline for load testing."""
        pipeline = MagicMock()
        pipeline.generate_response = AsyncMock()
        return pipeline

    @pytest.fixture
    def load_tester(self, mock_rag_pipeline):
        """Create load tester instance."""
        return RAGLoadTester(rag_pipeline=mock_rag_pipeline)

    @pytest.fixture
    def sample_queries(self):
        """Sample queries for load testing."""
        return [
            "What is machine learning?",
            "Explain deep learning concepts",
            "How does natural language processing work?",
            "What are neural networks?",
            "Describe artificial intelligence applications",
            "What is computer vision?",
            "How do recommendation systems work?",
            "Explain reinforcement learning",
            "What is data science?",
            "How does cloud computing work?"
        ]

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_request_load(self, load_tester, sample_queries):
        """Test system performance under concurrent request load."""
        # Mock response generation with variable latency
        async def mock_generate_response(query, **kwargs):
            # Simulate realistic response time
            await asyncio.sleep(0.1 + (hash(query) % 100) / 1000)  # 0.1-0.2s
            return type('RAGResponse', (), {
                'answer': f'Response to: {query}',
                'sources': [{'content': 'Mock source', 'score': 0.8}],
                'confidence_score': 0.85,
                'response_time': 0.15
            })
        
        load_tester.rag_pipeline.generate_response.side_effect = mock_generate_response
        
        # Test with increasing concurrent users
        concurrent_users = [1, 5, 10, 20, 50]
        results = {}
        
        for users in concurrent_users:
            metrics = await load_tester.run_concurrent_load_test(
                queries=sample_queries[:users],
                concurrent_users=users,
                duration_seconds=5
            )
            
            results[users] = metrics
            
            # Verify basic performance metrics
            assert metrics.total_requests > 0
            assert metrics.successful_requests >= 0
            assert metrics.failed_requests >= 0
            assert metrics.average_response_time > 0
            assert metrics.requests_per_second >= 0
        
        # Verify performance degrades gracefully under load
        assert results[1].average_response_time <= results[50].average_response_time * 2

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, load_tester, sample_queries):
        """Test system performance under sustained load."""
        # Mock consistent response generation
        load_tester.rag_pipeline.generate_response.side_effect = lambda q, **kwargs: AsyncMock(return_value=type('RAGResponse', (), {
            'answer': f'Sustained response to: {q}',
            'sources': [],
            'confidence_score': 0.8,
            'response_time': 0.12
        }))()
        
        # Run sustained load test
        metrics = await load_tester.run_sustained_load_test(
            queries=sample_queries,
            requests_per_second=10,
            duration_seconds=10
        )
        
        # Verify sustained performance
        assert metrics.total_requests >= 90  # Should complete most requests
        assert metrics.successful_requests >= metrics.total_requests * 0.9  # 90% success rate
        assert metrics.average_response_time < 1.0  # Under 1 second average
        assert metrics.p95_response_time < 2.0  # 95th percentile under 2 seconds

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_spike_load_handling(self, load_tester, sample_queries):
        """Test system handling of traffic spikes."""
        response_times = []
        
        async def mock_spike_response(query, **kwargs):
            # Simulate degraded performance under spike
            spike_delay = min(0.5, len(response_times) * 0.01)  # Increasing delay
            await asyncio.sleep(0.1 + spike_delay)
            response_time = 0.1 + spike_delay
            response_times.append(response_time)
            
            return type('RAGResponse', (), {
                'answer': f'Spike response to: {query}',
                'sources': [],
                'confidence_score': 0.75,
                'response_time': response_time
            })
        
        load_tester.rag_pipeline.generate_response.side_effect = mock_spike_response
        
        # Test traffic spike
        metrics = await load_tester.run_spike_load_test(
            queries=sample_queries,
            baseline_rps=5,
            spike_rps=50,
            spike_duration_seconds=3,
            total_duration_seconds=10
        )
        
        # Verify system handles spike reasonably
        assert metrics.total_requests > 0
        assert metrics.successful_requests >= metrics.total_requests * 0.8  # 80% success rate
        assert metrics.average_response_time < 3.0  # Should not exceed 3 seconds average

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, load_tester, sample_queries):
        """Test memory usage during load testing."""
        # Mock response with memory tracking
        memory_measurements = []
        
        async def mock_memory_response(query, **kwargs):
            # Measure memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_measurements.append(memory_mb)
            
            await asyncio.sleep(0.05)
            return type('RAGResponse', (), {
                'answer': f'Memory test response: {query}',
                'sources': [],
                'confidence_score': 0.8
            })
        
        load_tester.rag_pipeline.generate_response.side_effect = mock_memory_response
        
        # Run memory test
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        metrics = await load_tester.run_memory_load_test(
            queries=sample_queries,
            concurrent_users=20,
            duration_seconds=5
        )
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Verify memory usage is reasonable
        memory_growth = final_memory - initial_memory
        assert memory_growth < 500  # Should not grow by more than 500MB
        
        # Verify metrics include memory information
        assert hasattr(metrics, 'peak_memory_usage_mb')
        assert hasattr(metrics, 'memory_growth_mb')

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_error_rate_under_load(self, load_tester, sample_queries):
        """Test error handling under high load."""
        request_count = 0
        
        async def mock_error_prone_response(query, **kwargs):
            nonlocal request_count
            request_count += 1
            
            # Simulate increasing error rate under load
            if request_count > 30:  # Fail after 30 requests
                if request_count % 3 == 0:  # Every 3rd request fails
                    raise Exception("Service temporarily unavailable")
            
            await asyncio.sleep(0.08)
            return type('RAGResponse', (), {
                'answer': f'Response: {query}',
                'sources': [],
                'confidence_score': 0.8
            })
        
        load_tester.rag_pipeline.generate_response.side_effect = mock_error_prone_response
        
        # Test error handling under load
        metrics = await load_tester.run_error_rate_test(
            queries=sample_queries * 5,  # 50 queries
            concurrent_users=10,
            duration_seconds=8
        )
        
        # Verify error handling
        assert metrics.total_requests > 40
        assert metrics.failed_requests > 0  # Should have some failures
        assert metrics.error_rate <= 0.5  # Error rate should not exceed 50%
        assert metrics.successful_requests > 0  # Should still have successes

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_response_time_distribution(self, load_tester, sample_queries):
        """Test response time distribution analysis."""
        # Mock response with variable timing
        response_times = []
        
        async def mock_variable_response(query, **kwargs):
            # Simulate realistic response time distribution
            base_time = 0.1
            variation = (hash(query) % 100) / 500  # 0-0.2s variation
            response_time = base_time + variation
            response_times.append(response_time)
            
            await asyncio.sleep(response_time)
            return type('RAGResponse', (), {
                'answer': f'Variable time response: {query}',
                'sources': [],
                'confidence_score': 0.8,
                'response_time': response_time
            })
        
        load_tester.rag_pipeline.generate_response.side_effect = mock_variable_response
        
        # Test response time distribution
        metrics = await load_tester.run_response_time_analysis(
            queries=sample_queries * 3,  # 30 queries
            concurrent_users=5,
            duration_seconds=10
        )
        
        # Verify response time metrics
        assert metrics.average_response_time > 0
        assert metrics.median_response_time > 0
        assert metrics.p95_response_time >= metrics.median_response_time
        assert metrics.p99_response_time >= metrics.p95_response_time
        assert metrics.min_response_time <= metrics.average_response_time
        assert metrics.max_response_time >= metrics.average_response_time

    @pytest.mark.performance
    def test_throughput_measurement(self, load_tester, sample_queries):
        """Test throughput measurement capabilities."""
        # Mock high-throughput responses
        load_tester.rag_pipeline.generate_response = AsyncMock(
            return_value=type('RAGResponse', (), {
                'answer': 'Fast response',
                'sources': [],
                'confidence_score': 0.8
            })
        )
        
        # Measure throughput
        throughput_metrics = asyncio.run(
            load_tester.measure_max_throughput(
                queries=sample_queries,
                max_concurrent_users=100,
                ramp_up_time=2
            )
        )
        
        # Verify throughput metrics
        assert throughput_metrics['max_requests_per_second'] > 0
        assert throughput_metrics['max_concurrent_users'] > 0
        assert throughput_metrics['optimal_concurrent_users'] > 0
        assert throughput_metrics['saturation_point'] > 0

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_endurance_testing(self, load_tester, sample_queries):
        """Test system endurance over extended periods."""
        # Mock stable long-running responses
        request_counter = 0
        
        async def mock_endurance_response(query, **kwargs):
            nonlocal request_counter
            request_counter += 1
            
            # Simulate slight performance degradation over time
            degradation_factor = min(0.1, request_counter / 10000)
            response_time = 0.1 + degradation_factor
            
            await asyncio.sleep(response_time)
            return type('RAGResponse', (), {
                'answer': f'Endurance response #{request_counter}: {query}',
                'sources': [],
                'confidence_score': max(0.7, 0.9 - degradation_factor),
                'response_time': response_time
            })
        
        load_tester.rag_pipeline.generate_response.side_effect = mock_endurance_response
        
        # Run endurance test (shorter duration for testing)
        metrics = await load_tester.run_endurance_test(
            queries=sample_queries,
            concurrent_users=10,
            duration_seconds=15,  # Reduced for testing
            measurement_interval=5
        )
        
        # Verify endurance metrics
        assert metrics.total_requests > 100  # Should handle many requests
        assert metrics.successful_requests >= metrics.total_requests * 0.95  # 95% success
        assert metrics.average_response_time < 0.5  # Reasonable response time
        assert len(metrics.time_series_data) >= 3  # Multiple measurement points

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_utilization_monitoring(self, load_tester, sample_queries):
        """Test resource utilization monitoring during load tests."""
        # Mock resource-intensive responses
        async def mock_resource_response(query, **kwargs):
            # Simulate some CPU work
            _ = sum(range(1000))
            await asyncio.sleep(0.05)
            
            return type('RAGResponse', (), {
                'answer': f'Resource response: {query}',
                'sources': [],
                'confidence_score': 0.8
            })
        
        load_tester.rag_pipeline.generate_response.side_effect = mock_resource_response
        
        # Test with resource monitoring
        metrics = await load_tester.run_resource_monitoring_test(
            queries=sample_queries,
            concurrent_users=15,
            duration_seconds=8
        )
        
        # Verify resource metrics are captured
        assert hasattr(metrics, 'cpu_utilization')
        assert hasattr(metrics, 'memory_utilization')
        assert hasattr(metrics, 'peak_cpu_usage')
        assert hasattr(metrics, 'peak_memory_usage')
        
        # Basic sanity checks
        assert 0 <= metrics.cpu_utilization <= 100
        assert metrics.memory_utilization > 0

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_scalability_analysis(self, load_tester, sample_queries):
        """Test scalability analysis across different load levels."""
        # Mock scalable responses
        load_tester.rag_pipeline.generate_response = AsyncMock(
            return_value=type('RAGResponse', (), {
                'answer': 'Scalable response',
                'sources': [],
                'confidence_score': 0.8
            })
        )
        
        # Test scalability across different user counts
        user_levels = [1, 5, 10, 20, 30]
        scalability_results = {}
        
        for users in user_levels:
            metrics = await load_tester.run_concurrent_load_test(
                queries=sample_queries,
                concurrent_users=users,
                duration_seconds=3
            )
            scalability_results[users] = metrics
        
        # Analyze scalability
        analysis = load_tester.analyze_scalability(scalability_results)
        
        # Verify analysis results
        assert 'linear_scalability_coefficient' in analysis
        assert 'bottleneck_threshold' in analysis
        assert 'scalability_grade' in analysis
        assert analysis['scalability_grade'] in ['A', 'B', 'C', 'D', 'F']

    @pytest.mark.performance
    @pytest.mark.asyncio 
    async def test_cache_performance_impact(self, load_tester, sample_queries):
        """Test impact of caching on performance."""
        # Mock cache behavior
        cache = {}
        
        async def mock_cached_response(query, **kwargs):
            if query in cache:
                # Cache hit - very fast response
                await asyncio.sleep(0.01)
                return cache[query]
            else:
                # Cache miss - slower response
                await asyncio.sleep(0.15)
                response = type('RAGResponse', (), {
                    'answer': f'Cached response: {query}',
                    'sources': [],
                    'confidence_score': 0.8
                })
                cache[query] = response
                return response
        
        load_tester.rag_pipeline.generate_response.side_effect = mock_cached_response
        
        # Test with cache (repeated queries)
        repeated_queries = sample_queries * 3  # Each query repeated 3 times
        
        metrics = await load_tester.run_cache_performance_test(
            queries=repeated_queries,
            concurrent_users=10,
            duration_seconds=8
        )
        
        # Verify cache performance impact
        assert metrics.average_response_time < 0.1  # Should be fast with caching
        assert metrics.cache_hit_rate > 0.6  # Should have good cache hit rate
        assert metrics.performance_improvement > 1.5  # Should show improvement

    @pytest.mark.performance
    def test_performance_regression_detection(self, load_tester, sample_queries):
        """Test performance regression detection."""
        # Simulate baseline performance
        baseline_metrics = PerformanceMetrics(
            total_requests=100,
            successful_requests=98,
            failed_requests=2,
            average_response_time=0.15,
            median_response_time=0.14,
            p95_response_time=0.25,
            p99_response_time=0.35,
            min_response_time=0.08,
            max_response_time=0.45,
            requests_per_second=50.0,
            error_rate=0.02,
            start_time=time.time() - 10,
            end_time=time.time()
        )
        
        # Simulate current performance (degraded)
        current_metrics = PerformanceMetrics(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            average_response_time=0.25,  # 67% slower
            median_response_time=0.22,
            p95_response_time=0.40,
            p99_response_time=0.55,
            min_response_time=0.12,
            max_response_time=0.75,
            requests_per_second=35.0,  # 30% lower throughput
            error_rate=0.05,  # Higher error rate
            start_time=time.time() - 5,
            end_time=time.time()
        )
        
        # Detect regression
        regression_analysis = load_tester.detect_performance_regression(
            baseline_metrics, current_metrics
        )
        
        # Verify regression detection
        assert regression_analysis['has_regression'] is True
        assert 'response_time_regression' in regression_analysis
        assert 'throughput_regression' in regression_analysis
        assert 'error_rate_regression' in regression_analysis
        assert regression_analysis['severity'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']