"""Load testing utilities for RAG system performance evaluation."""

import asyncio
import time
import statistics
import psutil
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import csv
from datetime import datetime
import logging


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    median_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)
    
    # Extended metrics
    peak_memory_usage_mb: float = 0.0
    memory_growth_mb: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    peak_cpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    performance_improvement: float = 1.0
    time_series_data: List[Dict] = field(default_factory=list)


@dataclass 
class RequestResult:
    """Individual request result."""
    query: str
    success: bool
    response_time: float
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    memory_usage_mb: float = 0.0


class ResourceMonitor:
    """Monitor system resources during load testing."""
    
    def __init__(self):
        self.monitoring = False
        self.measurements = []
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring."""
        self.monitoring = True
        self.measurements = []
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,)
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Resource monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                self.measurements.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_percent': process.memory_percent()
                })
                
            except Exception as e:
                logging.warning(f"Error monitoring resources: {e}")
            
            time.sleep(interval)
    
    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak resource usage metrics."""
        if not self.measurements:
            return {'peak_cpu': 0.0, 'peak_memory_mb': 0.0, 'avg_cpu': 0.0, 'avg_memory_mb': 0.0}
        
        cpu_values = [m['cpu_percent'] for m in self.measurements]
        memory_values = [m['memory_mb'] for m in self.measurements]
        
        return {
            'peak_cpu': max(cpu_values),
            'peak_memory_mb': max(memory_values),
            'avg_cpu': statistics.mean(cpu_values),
            'avg_memory_mb': statistics.mean(memory_values)
        }


class RAGLoadTester:
    """Load tester for RAG systems."""
    
    def __init__(self, rag_pipeline):
        """
        Initialize load tester.
        
        Args:
            rag_pipeline: RAG pipeline to test
        """
        self.rag_pipeline = rag_pipeline
        self.resource_monitor = ResourceMonitor()
        self.logger = logging.getLogger(__name__)
        
    async def run_concurrent_load_test(self, 
                                     queries: List[str],
                                     concurrent_users: int,
                                     duration_seconds: int) -> PerformanceMetrics:
        """
        Run concurrent load test.
        
        Args:
            queries: List of test queries
            concurrent_users: Number of concurrent users to simulate
            duration_seconds: Test duration in seconds
            
        Returns:
            Performance metrics
        """
        start_time = time.time()
        end_time = start_time + duration_seconds
        results = []
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Create tasks for concurrent execution
            tasks = []
            query_index = 0
            
            while time.time() < end_time:
                # Launch concurrent requests
                for _ in range(min(concurrent_users, len(queries))):
                    query = queries[query_index % len(queries)]
                    task = asyncio.create_task(self._execute_request(query))
                    tasks.append(task)
                    query_index += 1
                
                # Wait for a batch to complete
                if len(tasks) >= concurrent_users:
                    completed_tasks = await asyncio.gather(*tasks[:concurrent_users], return_exceptions=True)
                    for i, result in enumerate(completed_tasks):
                        if isinstance(result, RequestResult):
                            results.append(result)
                        elif isinstance(result, Exception):
                            results.append(RequestResult(
                                query=queries[i % len(queries)],
                                success=False,
                                response_time=0.0,
                                error_message=str(result)
                            ))
                    tasks = tasks[concurrent_users:]
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            # Wait for remaining tasks
            if tasks:
                remaining_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in remaining_results:
                    if isinstance(result, RequestResult):
                        results.append(result)
            
        finally:
            self.resource_monitor.stop_monitoring()
        
        return self._calculate_metrics(results, start_time, time.time())
    
    async def run_sustained_load_test(self,
                                    queries: List[str],
                                    requests_per_second: int,
                                    duration_seconds: int) -> PerformanceMetrics:
        """
        Run sustained load test with fixed RPS.
        
        Args:
            queries: List of test queries
            requests_per_second: Target requests per second
            duration_seconds: Test duration
            
        Returns:
            Performance metrics
        """
        start_time = time.time()
        results = []
        request_interval = 1.0 / requests_per_second
        
        self.resource_monitor.start_monitoring()
        
        try:
            query_index = 0
            next_request_time = start_time
            
            while time.time() < start_time + duration_seconds:
                current_time = time.time()
                
                if current_time >= next_request_time:
                    query = queries[query_index % len(queries)]
                    result = await self._execute_request(query)
                    results.append(result)
                    
                    query_index += 1
                    next_request_time += request_interval
                else:
                    # Wait until next request time
                    await asyncio.sleep(min(0.001, next_request_time - current_time))
                    
        finally:
            self.resource_monitor.stop_monitoring()
        
        return self._calculate_metrics(results, start_time, time.time())
    
    async def run_spike_load_test(self,
                                queries: List[str],
                                baseline_rps: int,
                                spike_rps: int,
                                spike_duration_seconds: int,
                                total_duration_seconds: int) -> PerformanceMetrics:
        """
        Run spike load test.
        
        Args:
            queries: List of test queries
            baseline_rps: Baseline requests per second
            spike_rps: Spike requests per second
            spike_duration_seconds: Duration of spike
            total_duration_seconds: Total test duration
            
        Returns:
            Performance metrics
        """
        start_time = time.time()
        results = []
        
        self.resource_monitor.start_monitoring()
        
        try:
            query_index = 0
            spike_start_time = start_time + (total_duration_seconds - spike_duration_seconds) / 2
            spike_end_time = spike_start_time + spike_duration_seconds
            
            next_request_time = start_time
            
            while time.time() < start_time + total_duration_seconds:
                current_time = time.time()
                
                # Determine current RPS based on whether we're in spike period
                current_rps = spike_rps if spike_start_time <= current_time <= spike_end_time else baseline_rps
                request_interval = 1.0 / current_rps
                
                if current_time >= next_request_time:
                    query = queries[query_index % len(queries)]
                    result = await self._execute_request(query)
                    results.append(result)
                    
                    query_index += 1
                    next_request_time += request_interval
                else:
                    await asyncio.sleep(0.001)
                    
        finally:
            self.resource_monitor.stop_monitoring()
        
        return self._calculate_metrics(results, start_time, time.time())
    
    async def run_memory_load_test(self,
                                 queries: List[str],
                                 concurrent_users: int,
                                 duration_seconds: int) -> PerformanceMetrics:
        """
        Run load test with memory monitoring.
        
        Args:
            queries: List of test queries
            concurrent_users: Number of concurrent users
            duration_seconds: Test duration
            
        Returns:
            Performance metrics with memory information
        """
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        metrics = await self.run_concurrent_load_test(
            queries, concurrent_users, duration_seconds
        )
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        resource_metrics = self.resource_monitor.get_peak_metrics()
        
        # Add memory metrics
        metrics.peak_memory_usage_mb = resource_metrics['peak_memory_mb']
        metrics.memory_growth_mb = final_memory - initial_memory
        
        return metrics
    
    async def run_error_rate_test(self,
                                queries: List[str],
                                concurrent_users: int,
                                duration_seconds: int) -> PerformanceMetrics:
        """
        Run test focused on error rate analysis.
        
        Args:
            queries: List of test queries
            concurrent_users: Number of concurrent users
            duration_seconds: Test duration
            
        Returns:
            Performance metrics with error analysis
        """
        return await self.run_concurrent_load_test(
            queries, concurrent_users, duration_seconds
        )
    
    async def run_response_time_analysis(self,
                                       queries: List[str],
                                       concurrent_users: int,
                                       duration_seconds: int) -> PerformanceMetrics:
        """
        Run test focused on response time distribution analysis.
        
        Args:
            queries: List of test queries
            concurrent_users: Number of concurrent users
            duration_seconds: Test duration
            
        Returns:
            Performance metrics with detailed response time analysis
        """
        return await self.run_concurrent_load_test(
            queries, concurrent_users, duration_seconds
        )
    
    async def measure_max_throughput(self,
                                   queries: List[str],
                                   max_concurrent_users: int = 100,
                                   ramp_up_time: int = 5) -> Dict[str, Any]:
        """
        Measure maximum system throughput.
        
        Args:
            queries: List of test queries
            max_concurrent_users: Maximum concurrent users to test
            ramp_up_time: Time to ramp up users
            
        Returns:
            Throughput metrics
        """
        throughput_results = {}
        best_rps = 0
        optimal_users = 1
        
        for users in [1, 2, 5, 10, 20, 50, max_concurrent_users]:
            metrics = await self.run_concurrent_load_test(
                queries, users, ramp_up_time
            )
            
            throughput_results[users] = metrics.requests_per_second
            
            if metrics.requests_per_second > best_rps and metrics.error_rate < 0.05:
                best_rps = metrics.requests_per_second
                optimal_users = users
        
        return {
            'max_requests_per_second': best_rps,
            'optimal_concurrent_users': optimal_users,
            'max_concurrent_users': max_concurrent_users,
            'saturation_point': self._find_saturation_point(throughput_results),
            'throughput_curve': throughput_results
        }
    
    async def run_endurance_test(self,
                               queries: List[str],
                               concurrent_users: int,
                               duration_seconds: int,
                               measurement_interval: int = 60) -> PerformanceMetrics:
        """
        Run endurance test with periodic measurements.
        
        Args:
            queries: List of test queries
            concurrent_users: Number of concurrent users
            duration_seconds: Test duration
            measurement_interval: Interval between measurements
            
        Returns:
            Performance metrics with time series data
        """
        start_time = time.time()
        all_results = []
        time_series_data = []
        
        self.resource_monitor.start_monitoring()
        
        try:
            while time.time() < start_time + duration_seconds:
                # Run measurement interval
                interval_end = min(
                    time.time() + measurement_interval,
                    start_time + duration_seconds
                )
                interval_duration = interval_end - time.time()
                
                if interval_duration > 0:
                    interval_metrics = await self.run_concurrent_load_test(
                        queries, concurrent_users, int(interval_duration)
                    )
                    
                    time_series_data.append({
                        'timestamp': time.time(),
                        'requests_per_second': interval_metrics.requests_per_second,
                        'average_response_time': interval_metrics.average_response_time,
                        'error_rate': interval_metrics.error_rate
                    })
                    
                    all_results.extend(self._extract_results_from_metrics(interval_metrics))
                    
        finally:
            self.resource_monitor.stop_monitoring()
        
        final_metrics = self._calculate_metrics(all_results, start_time, time.time())
        final_metrics.time_series_data = time_series_data
        
        return final_metrics
    
    async def run_resource_monitoring_test(self,
                                         queries: List[str],
                                         concurrent_users: int,
                                         duration_seconds: int) -> PerformanceMetrics:
        """
        Run test with comprehensive resource monitoring.
        
        Args:
            queries: List of test queries
            concurrent_users: Number of concurrent users
            duration_seconds: Test duration
            
        Returns:
            Performance metrics with resource utilization data
        """
        metrics = await self.run_concurrent_load_test(
            queries, concurrent_users, duration_seconds
        )
        
        # Add resource metrics
        resource_metrics = self.resource_monitor.get_peak_metrics()
        metrics.cpu_utilization = resource_metrics['avg_cpu']
        metrics.memory_utilization = resource_metrics['avg_memory_mb']
        metrics.peak_cpu_usage = resource_metrics['peak_cpu']
        metrics.peak_memory_usage_mb = resource_metrics['peak_memory_mb']
        
        return metrics
    
    async def run_cache_performance_test(self,
                                       queries: List[str],
                                       concurrent_users: int,
                                       duration_seconds: int) -> PerformanceMetrics:
        """
        Run test to measure cache performance impact.
        
        Args:
            queries: List of test queries (should include repeats)
            concurrent_users: Number of concurrent users
            duration_seconds: Test duration
            
        Returns:
            Performance metrics with cache-related metrics
        """
        # Run test without cache first (if possible)
        baseline_metrics = await self.run_concurrent_load_test(
            queries[:len(queries)//3], concurrent_users, duration_seconds//2
        )
        
        # Run test with cache (repeated queries)
        cached_metrics = await self.run_concurrent_load_test(
            queries, concurrent_users, duration_seconds//2
        )
        
        # Calculate cache performance
        cache_hit_rate = self._estimate_cache_hit_rate(queries)
        performance_improvement = (
            baseline_metrics.average_response_time / cached_metrics.average_response_time
            if cached_metrics.average_response_time > 0 else 1.0
        )
        
        cached_metrics.cache_hit_rate = cache_hit_rate
        cached_metrics.performance_improvement = performance_improvement
        
        return cached_metrics
    
    async def _execute_request(self, query: str) -> RequestResult:
        """
        Execute a single request and measure performance.
        
        Args:
            query: Query to execute
            
        Returns:
            Request result
        """
        start_time = time.time()
        
        try:
            # Measure memory before request
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Execute request
            response = await self.rag_pipeline.generate_response(query)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Measure memory after request
            memory_after = process.memory_info().rss / 1024 / 1024
            
            return RequestResult(
                query=query,
                success=True,
                response_time=response_time,
                memory_usage_mb=memory_after - memory_before
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return RequestResult(
                query=query,
                success=False,
                response_time=response_time,
                error_message=str(e)
            )
    
    def _calculate_metrics(self, results: List[RequestResult], start_time: float, end_time: float) -> PerformanceMetrics:
        """Calculate performance metrics from results."""
        if not results:
            return PerformanceMetrics()
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        response_times = [r.response_time for r in successful_results]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = self._percentile(response_times, 95)
            p99_response_time = self._percentile(response_times, 99)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = median_response_time = 0.0
            p95_response_time = p99_response_time = 0.0
            min_response_time = max_response_time = 0.0
        
        total_time = end_time - start_time
        requests_per_second = len(results) / total_time if total_time > 0 else 0
        error_rate = len(failed_results) / len(results) if results else 0
        
        return PerformanceMetrics(
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            start_time=start_time,
            end_time=end_time
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            
            if upper_index >= len(sorted_data):
                return sorted_data[lower_index]
            
            return (
                sorted_data[lower_index] * (1 - weight) +
                sorted_data[upper_index] * weight
            )
    
    def _find_saturation_point(self, throughput_results: Dict[int, float]) -> int:
        """Find the saturation point where throughput stops increasing."""
        users = sorted(throughput_results.keys())
        
        for i in range(1, len(users)):
            current_rps = throughput_results[users[i]]
            previous_rps = throughput_results[users[i-1]]
            
            # If throughput increase is less than 10%, consider it saturation
            if current_rps < previous_rps * 1.1:
                return users[i-1]
        
        return users[-1] if users else 0
    
    def _estimate_cache_hit_rate(self, queries: List[str]) -> float:
        """Estimate cache hit rate based on query repetition."""
        unique_queries = set(queries)
        total_queries = len(queries)
        unique_count = len(unique_queries)
        
        # Simple estimation: (total - unique) / total
        return (total_queries - unique_count) / total_queries if total_queries > 0 else 0.0
    
    def _extract_results_from_metrics(self, metrics: PerformanceMetrics) -> List[RequestResult]:
        """Extract request results from metrics (mock implementation)."""
        # This is a simplified mock implementation
        # In real implementation, you'd store actual results
        results = []
        
        for i in range(metrics.successful_requests):
            results.append(RequestResult(
                query=f"query_{i}",
                success=True,
                response_time=metrics.average_response_time
            ))
        
        for i in range(metrics.failed_requests):
            results.append(RequestResult(
                query=f"query_{i}",
                success=False,
                response_time=0.0,
                error_message="Mock error"
            ))
        
        return results
    
    def analyze_scalability(self, scalability_results: Dict[int, PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze scalability based on load test results."""
        user_counts = sorted(scalability_results.keys())
        rps_values = [scalability_results[u].requests_per_second for u in user_counts]
        
        # Calculate linear scalability coefficient
        if len(user_counts) >= 2:
            # Simple linear regression coefficient
            x_mean = statistics.mean(user_counts)
            y_mean = statistics.mean(rps_values)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(user_counts, rps_values))
            denominator = sum((x - x_mean) ** 2 for x in user_counts)
            
            coefficient = numerator / denominator if denominator != 0 else 0
        else:
            coefficient = 0
        
        # Find bottleneck threshold
        bottleneck_threshold = self._find_saturation_point({
            u: scalability_results[u].requests_per_second for u in user_counts
        })
        
        # Grade scalability
        if coefficient > 0.8:
            grade = 'A'
        elif coefficient > 0.6:
            grade = 'B'
        elif coefficient > 0.4:
            grade = 'C'
        elif coefficient > 0.2:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'linear_scalability_coefficient': coefficient,
            'bottleneck_threshold': bottleneck_threshold,
            'scalability_grade': grade,
            'peak_throughput': max(rps_values) if rps_values else 0,
            'scalability_analysis': {
                u: {
                    'rps': scalability_results[u].requests_per_second,
                    'avg_response_time': scalability_results[u].average_response_time,
                    'error_rate': scalability_results[u].error_rate
                } for u in user_counts
            }
        }
    
    def detect_performance_regression(self,
                                    baseline_metrics: PerformanceMetrics,
                                    current_metrics: PerformanceMetrics,
                                    thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Detect performance regression by comparing metrics.
        
        Args:
            baseline_metrics: Baseline performance metrics
            current_metrics: Current performance metrics
            thresholds: Custom regression thresholds
            
        Returns:
            Regression analysis results
        """
        if thresholds is None:
            thresholds = {
                'response_time_threshold': 1.2,  # 20% increase
                'throughput_threshold': 0.8,     # 20% decrease
                'error_rate_threshold': 2.0      # 100% increase
            }
        
        analysis = {
            'has_regression': False,
            'severity': 'LOW',
            'issues': []
        }
        
        # Response time regression
        if current_metrics.average_response_time > baseline_metrics.average_response_time * thresholds['response_time_threshold']:
            analysis['has_regression'] = True
            analysis['response_time_regression'] = {
                'baseline': baseline_metrics.average_response_time,
                'current': current_metrics.average_response_time,
                'degradation_factor': current_metrics.average_response_time / baseline_metrics.average_response_time
            }
            analysis['issues'].append('Response time degradation detected')
        
        # Throughput regression
        if current_metrics.requests_per_second < baseline_metrics.requests_per_second * thresholds['throughput_threshold']:
            analysis['has_regression'] = True
            analysis['throughput_regression'] = {
                'baseline': baseline_metrics.requests_per_second,
                'current': current_metrics.requests_per_second,
                'degradation_factor': current_metrics.requests_per_second / baseline_metrics.requests_per_second
            }
            analysis['issues'].append('Throughput degradation detected')
        
        # Error rate regression
        if current_metrics.error_rate > baseline_metrics.error_rate * thresholds['error_rate_threshold']:
            analysis['has_regression'] = True
            analysis['error_rate_regression'] = {
                'baseline': baseline_metrics.error_rate,
                'current': current_metrics.error_rate,
                'increase_factor': current_metrics.error_rate / baseline_metrics.error_rate if baseline_metrics.error_rate > 0 else float('inf')
            }
            analysis['issues'].append('Error rate increase detected')
        
        # Determine severity
        if len(analysis['issues']) >= 3:
            analysis['severity'] = 'CRITICAL'
        elif len(analysis['issues']) >= 2:
            analysis['severity'] = 'HIGH'
        elif len(analysis['issues']) >= 1:
            analysis['severity'] = 'MEDIUM'
        
        return analysis
    
    def export_results(self, metrics: PerformanceMetrics, filepath: str, format: str = 'json'):
        """Export performance results to file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'average_response_time': metrics.average_response_time,
                'median_response_time': metrics.median_response_time,
                'p95_response_time': metrics.p95_response_time,
                'p99_response_time': metrics.p99_response_time,
                'requests_per_second': metrics.requests_per_second,
                'error_rate': metrics.error_rate,
                'test_duration': metrics.end_time - metrics.start_time
            }
        }
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format.lower() == 'csv':
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data['metrics'].keys())
                writer.writeheader()
                writer.writerow(data['metrics'])
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results exported to {filepath}")