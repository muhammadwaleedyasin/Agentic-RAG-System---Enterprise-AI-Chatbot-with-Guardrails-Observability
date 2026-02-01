"""
Comprehensive Benchmark Suite for Production-Scale RAG Systems

This module provides extensive benchmarking capabilities for testing
performance, scalability, and reliability of RAG systems under various
load conditions and configurations.
"""

import asyncio
import time
import json
import logging
import statistics
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from optimization.reranking_engine import create_reranking_engine, Document
from optimization.search_optimizer import create_search_optimizer
from optimization.caching_layer import create_caching_system
from optimization.scaling_manager import create_scaling_manager
from monitoring.performance_analyzer import create_performance_analyzer

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    # Test parameters
    num_documents: int = 10000
    num_queries: int = 1000
    concurrent_users: int = 10
    test_duration_seconds: int = 300
    
    # Document parameters
    doc_length_range: Tuple[int, int] = (100, 1000)
    query_length_range: Tuple[int, int] = (5, 50)
    
    # Performance thresholds
    max_response_time: float = 5.0
    min_throughput: float = 10.0  # requests per second
    max_error_rate: float = 0.01  # 1%
    
    # Test types to run
    test_types: List[str] = field(default_factory=lambda: [
        "single_query", "concurrent_queries", "load_test", 
        "stress_test", "scalability_test", "memory_test"
    ])

@dataclass
class BenchmarkResult:
    """Results from a benchmark test"""
    test_name: str
    test_type: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    throughput: float
    error_rate: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataGenerator:
    """Generate synthetic test data"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.word_list = self._load_word_list()
    
    def _load_word_list(self) -> List[str]:
        """Load word list for generating content"""
        # Common English words for generating realistic text
        return [
            "the", "be", "to", "of", "and", "a", "in", "that", "have",
            "i", "it", "for", "not", "on", "with", "he", "as", "you",
            "do", "at", "this", "but", "his", "by", "from", "they",
            "we", "say", "her", "she", "or", "an", "will", "my",
            "one", "all", "would", "there", "their", "what", "so",
            "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just",
            "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than",
            "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how",
            "our", "work", "first", "well", "way", "even", "new",
            "want", "because", "any", "these", "give", "day", "most",
            "us", "is", "was", "are", "been", "has", "had", "were",
            "said", "each", "which", "she", "do", "how", "their",
            "if", "will", "up", "other", "about", "out", "many",
            "then", "them", "these", "so", "some", "her", "would",
            "make", "like", "into", "him", "time", "has", "two",
            "more", "very", "what", "know", "just", "first", "get",
            "over", "think", "where", "much", "go", "well", "were",
            "python", "machine", "learning", "artificial", "intelligence",
            "algorithm", "data", "science", "computer", "technology",
            "programming", "software", "development", "neural", "network",
            "deep", "learning", "natural", "language", "processing",
            "retrieval", "augmented", "generation", "embedding", "vector",
            "search", "query", "document", "index", "ranking", "semantic"
        ]
    
    def generate_document(self, min_length: int = 100, max_length: int = 1000) -> str:
        """Generate a synthetic document"""
        length = np.random.randint(min_length, max_length)
        words = np.random.choice(self.word_list, size=length, replace=True)
        return " ".join(words)
    
    def generate_query(self, min_length: int = 5, max_length: int = 50) -> str:
        """Generate a synthetic query"""
        length = np.random.randint(min_length, max_length)
        words = np.random.choice(self.word_list, size=length, replace=True)
        return " ".join(words)
    
    def generate_documents(self, count: int, min_length: int = 100, max_length: int = 1000) -> List[Document]:
        """Generate multiple synthetic documents"""
        documents = []
        for i in range(count):
            content = self.generate_document(min_length, max_length)
            doc = Document(
                id=f"doc_{i}",
                content=content,
                metadata={"category": np.random.choice(["tech", "science", "general"])},
                initial_score=np.random.random()
            )
            documents.append(doc)
        return documents
    
    def generate_queries(self, count: int, min_length: int = 5, max_length: int = 50) -> List[str]:
        """Generate multiple synthetic queries"""
        return [self.generate_query(min_length, max_length) for _ in range(count)]

class BenchmarkRunner:
    """Main benchmark runner class"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data_generator = DataGenerator()
        self.results: List[BenchmarkResult] = []
        
        # Initialize components
        self.reranking_engine = None
        self.search_optimizer = None
        self.caching_system = None
        self.scaling_manager = None
        self.performance_analyzer = None
        
        # Test data
        self.test_documents: List[Document] = []
        self.test_queries: List[str] = []
    
    async def setup(self):
        """Setup benchmark environment"""
        logger.info("Setting up benchmark environment...")
        
        # Initialize components
        self.reranking_engine = create_reranking_engine()
        self.search_optimizer = create_search_optimizer()
        self.caching_system = create_caching_system()
        self.scaling_manager = create_scaling_manager()
        self.performance_analyzer = create_performance_analyzer()
        
        # Generate test data
        logger.info(f"Generating {self.config.num_documents} test documents...")
        self.test_documents = self.data_generator.generate_documents(
            self.config.num_documents,
            self.config.doc_length_range[0],
            self.config.doc_length_range[1]
        )
        
        logger.info(f"Generating {self.config.num_queries} test queries...")
        self.test_queries = self.data_generator.generate_queries(
            self.config.num_queries,
            self.config.query_length_range[0],
            self.config.query_length_range[1]
        )
        
        # Warm up components
        logger.info("Warming up components...")
        await self._warmup()
        
        logger.info("Benchmark setup complete")
    
    async def _warmup(self):
        """Warm up components with sample data"""
        sample_docs = self.test_documents[:10]
        sample_queries = self.test_queries[:5]
        
        # Warm up reranking engine
        await self.reranking_engine.warm_up(sample_queries, sample_docs)
        
        # Warm up other components
        for query in sample_queries[:3]:
            await self.caching_system.set(f"warmup_{query}", sample_docs[:5])
            _ = await self.caching_system.get(f"warmup_{query}")
    
    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all configured benchmark tests"""
        logger.info(f"Starting benchmark suite with {len(self.config.test_types)} test types")
        
        for test_type in self.config.test_types:
            try:
                logger.info(f"Running {test_type} benchmark...")
                result = await self._run_test_type(test_type)
                self.results.append(result)
                
                # Log immediate results
                logger.info(f"{test_type} completed: "
                          f"{result.throughput:.2f} req/s, "
                          f"avg: {result.avg_response_time:.3f}s, "
                          f"p95: {result.p95_response_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Test {test_type} failed: {e}")
        
        return self.results
    
    async def _run_test_type(self, test_type: str) -> BenchmarkResult:
        """Run a specific type of benchmark test"""
        if test_type == "single_query":
            return await self._run_single_query_test()
        elif test_type == "concurrent_queries":
            return await self._run_concurrent_queries_test()
        elif test_type == "load_test":
            return await self._run_load_test()
        elif test_type == "stress_test":
            return await self._run_stress_test()
        elif test_type == "scalability_test":
            return await self._run_scalability_test()
        elif test_type == "memory_test":
            return await self._run_memory_test()
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    async def _run_single_query_test(self) -> BenchmarkResult:
        """Test single query performance"""
        test_name = "Single Query Performance"
        response_times = []
        errors = 0
        
        start_time = time.time()
        
        for i, query in enumerate(self.test_queries[:100]):  # Test 100 queries
            try:
                query_start = time.time()
                
                # Run reranking
                await self.reranking_engine.rerank(query, self.test_documents[:50])
                
                response_time = time.time() - query_start
                response_times.append(response_time)
                
            except Exception as e:
                errors += 1
                logger.debug(f"Query {i} failed: {e}")
        
        duration = time.time() - start_time
        
        return self._create_result(
            test_name, "single_query", duration, response_times, errors
        )
    
    async def _run_concurrent_queries_test(self) -> BenchmarkResult:
        """Test concurrent query performance"""
        test_name = "Concurrent Queries Performance"
        response_times = []
        errors = 0
        total_requests = 200
        
        start_time = time.time()
        
        # Create tasks for concurrent execution
        async def query_task(query_id: int, query: str):
            try:
                query_start = time.time()
                await self.reranking_engine.rerank(query, self.test_documents[:30])
                return time.time() - query_start
            except Exception as e:
                logger.debug(f"Concurrent query {query_id} failed: {e}")
                raise
        
        # Run concurrent queries
        tasks = []
        for i in range(total_requests):
            query = self.test_queries[i % len(self.test_queries)]
            task = asyncio.create_task(query_task(i, query))
            tasks.append(task)
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                errors += 1
            else:
                response_times.append(result)
        
        duration = time.time() - start_time
        
        return self._create_result(
            test_name, "concurrent_queries", duration, response_times, errors
        )
    
    async def _run_load_test(self) -> BenchmarkResult:
        """Test system under sustained load"""
        test_name = "Load Test"
        response_times = []
        errors = 0
        
        start_time = time.time()
        end_time = start_time + self.config.test_duration_seconds
        
        request_count = 0
        
        while time.time() < end_time:
            batch_tasks = []
            
            # Create batch of concurrent requests
            for _ in range(self.config.concurrent_users):
                if time.time() >= end_time:
                    break
                
                query = self.test_queries[request_count % len(self.test_queries)]
                docs = self.test_documents[:20]  # Smaller doc set for load test
                
                async def load_task(q, d):
                    try:
                        task_start = time.time()
                        await self.reranking_engine.rerank(q, d)
                        return time.time() - task_start
                    except Exception as e:
                        raise e
                
                task = asyncio.create_task(load_task(query, docs))
                batch_tasks.append(task)
                request_count += 1
            
            # Wait for batch completion
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        errors += 1
                    else:
                        response_times.append(result)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        duration = time.time() - start_time
        
        return self._create_result(
            test_name, "load_test", duration, response_times, errors
        )
    
    async def _run_stress_test(self) -> BenchmarkResult:
        """Test system under stress conditions"""
        test_name = "Stress Test"
        response_times = []
        errors = 0
        
        # Stress test with higher concurrency and larger document sets
        stress_concurrent_users = self.config.concurrent_users * 3
        stress_doc_count = min(len(self.test_documents), 200)
        
        start_time = time.time()
        
        async def stress_task(task_id: int):
            try:
                query = self.test_queries[task_id % len(self.test_queries)]
                docs = self.test_documents[:stress_doc_count]
                
                task_start = time.time()
                await self.reranking_engine.rerank(query, docs)
                return time.time() - task_start
            
            except Exception as e:
                raise e
        
        # Run stress test with high concurrency
        tasks = []
        for i in range(stress_concurrent_users * 10):  # 10 rounds of high concurrency
            task = asyncio.create_task(stress_task(i))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                errors += 1
            else:
                response_times.append(result)
        
        duration = time.time() - start_time
        
        return self._create_result(
            test_name, "stress_test", duration, response_times, errors
        )
    
    async def _run_scalability_test(self) -> BenchmarkResult:
        """Test system scalability with increasing load"""
        test_name = "Scalability Test"
        response_times = []
        errors = 0
        
        start_time = time.time()
        
        # Test with increasing document counts
        doc_counts = [10, 50, 100, 200, 500]
        
        for doc_count in doc_counts:
            if doc_count > len(self.test_documents):
                continue
            
            logger.info(f"Testing scalability with {doc_count} documents")
            
            # Run queries with current document count
            for i in range(20):  # 20 queries per doc count
                try:
                    query = self.test_queries[i % len(self.test_queries)]
                    docs = self.test_documents[:doc_count]
                    
                    query_start = time.time()
                    await self.reranking_engine.rerank(query, docs)
                    response_time = time.time() - query_start
                    response_times.append(response_time)
                    
                except Exception as e:
                    errors += 1
                    logger.debug(f"Scalability test query failed: {e}")
        
        duration = time.time() - start_time
        
        return self._create_result(
            test_name, "scalability_test", duration, response_times, errors
        )
    
    async def _run_memory_test(self) -> BenchmarkResult:
        """Test memory usage and garbage collection"""
        test_name = "Memory Test"
        response_times = []
        errors = 0
        
        import gc
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        start_time = time.time()
        
        # Run memory-intensive operations
        for round_num in range(10):
            logger.info(f"Memory test round {round_num + 1}/10")
            
            # Create large document set
            large_doc_set = self.test_documents * 3  # Triple the document set
            
            for i in range(20):
                try:
                    query = self.test_queries[i % len(self.test_queries)]
                    
                    query_start = time.time()
                    await self.reranking_engine.rerank(query, large_doc_set[:100])
                    response_time = time.time() - query_start
                    response_times.append(response_time)
                    
                except Exception as e:
                    errors += 1
                    logger.debug(f"Memory test query failed: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            current_memory = process.memory_info().rss
            memory_increase = (current_memory - initial_memory) / 1024 / 1024  # MB
            logger.debug(f"Memory increase: {memory_increase:.2f} MB")
        
        duration = time.time() - start_time
        final_memory = process.memory_info().rss
        
        result = self._create_result(
            test_name, "memory_test", duration, response_times, errors
        )
        
        # Add memory usage metadata
        result.metadata.update({
            "initial_memory_mb": initial_memory / 1024 / 1024,
            "final_memory_mb": final_memory / 1024 / 1024,
            "memory_increase_mb": (final_memory - initial_memory) / 1024 / 1024
        })
        
        return result
    
    def _create_result(self, test_name: str, test_type: str, duration: float, 
                      response_times: List[float], errors: int) -> BenchmarkResult:
        """Create benchmark result from test data"""
        total_requests = len(response_times) + errors
        successful_requests = len(response_times)
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = 0.0
            p95_response_time = 0.0
            p99_response_time = 0.0
            min_response_time = 0.0
            max_response_time = 0.0
        
        throughput = successful_requests / duration if duration > 0 else 0.0
        error_rate = errors / total_requests if total_requests > 0 else 0.0
        
        # Get memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_usage = {
                "rss_mb": process.memory_info().rss / 1024 / 1024,
                "vms_mb": process.memory_info().vms / 1024 / 1024
            }
            cpu_usage = process.cpu_percent()
        except:
            memory_usage = {"rss_mb": 0.0, "vms_mb": 0.0}
            cpu_usage = 0.0
        
        return BenchmarkResult(
            test_name=test_name,
            test_type=test_type,
            duration=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=errors,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            throughput=throughput,
            error_rate=error_rate,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Summary statistics
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.failed_requests for r in self.results)
        
        avg_throughput = statistics.mean([r.throughput for r in self.results])
        avg_response_time = statistics.mean([r.avg_response_time for r in self.results])
        max_p99_time = max([r.p99_response_time for r in self.results])
        
        # Performance assessment
        passed_tests = []
        failed_tests = []
        
        for result in self.results:
            test_passed = (
                result.avg_response_time <= self.config.max_response_time and
                result.throughput >= self.config.min_throughput and
                result.error_rate <= self.config.max_error_rate
            )
            
            if test_passed:
                passed_tests.append(result.test_name)
            else:
                failed_tests.append({
                    "test": result.test_name,
                    "issues": self._identify_issues(result)
                })
        
        return {
            "benchmark_config": {
                "num_documents": self.config.num_documents,
                "num_queries": self.config.num_queries,
                "concurrent_users": self.config.concurrent_users,
                "test_duration_seconds": self.config.test_duration_seconds
            },
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests),
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_failed,
                "overall_success_rate": total_successful / total_requests if total_requests > 0 else 0.0
            },
            "performance_metrics": {
                "avg_throughput": avg_throughput,
                "avg_response_time": avg_response_time,
                "max_p99_response_time": max_p99_time
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "test_type": r.test_type,
                    "throughput": r.throughput,
                    "avg_response_time": r.avg_response_time,
                    "p95_response_time": r.p95_response_time,
                    "p99_response_time": r.p99_response_time,
                    "error_rate": r.error_rate,
                    "passed": r.test_name in passed_tests
                }
                for r in self.results
            ],
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "recommendations": self._generate_recommendations()
        }
    
    def _identify_issues(self, result: BenchmarkResult) -> List[str]:
        """Identify performance issues in test result"""
        issues = []
        
        if result.avg_response_time > self.config.max_response_time:
            issues.append(f"Average response time ({result.avg_response_time:.3f}s) exceeds threshold ({self.config.max_response_time}s)")
        
        if result.throughput < self.config.min_throughput:
            issues.append(f"Throughput ({result.throughput:.2f} req/s) below threshold ({self.config.min_throughput} req/s)")
        
        if result.error_rate > self.config.max_error_rate:
            issues.append(f"Error rate ({result.error_rate:.3f}) exceeds threshold ({self.config.max_error_rate})")
        
        return issues
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on results"""
        recommendations = []
        
        # Analyze response times
        avg_response_times = [r.avg_response_time for r in self.results]
        if avg_response_times and statistics.mean(avg_response_times) > self.config.max_response_time:
            recommendations.append("Consider optimizing reranking algorithms or implementing better caching")
        
        # Analyze throughput
        throughputs = [r.throughput for r in self.results]
        if throughputs and statistics.mean(throughputs) < self.config.min_throughput:
            recommendations.append("Consider horizontal scaling or performance optimization")
        
        # Analyze error rates
        error_rates = [r.error_rate for r in self.results]
        if error_rates and statistics.mean(error_rates) > self.config.max_error_rate:
            recommendations.append("Investigate and fix sources of errors")
        
        # Memory analysis
        memory_test_result = next((r for r in self.results if r.test_type == "memory_test"), None)
        if memory_test_result and "memory_increase_mb" in memory_test_result.metadata:
            if memory_test_result.metadata["memory_increase_mb"] > 500:  # 500MB increase
                recommendations.append("Investigate memory leaks and optimize memory usage")
        
        if not recommendations:
            recommendations.append("Performance meets all thresholds - system is performing well")
        
        return recommendations
    
    def save_results(self, filepath: str):
        """Save benchmark results to file"""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filepath}")

async def run_benchmark_suite(config: BenchmarkConfig = None) -> Dict[str, Any]:
    """Run the complete benchmark suite"""
    if config is None:
        config = BenchmarkConfig()
    
    runner = BenchmarkRunner(config)
    
    try:
        # Setup and run benchmarks
        await runner.setup()
        await runner.run_all_benchmarks()
        
        # Generate and return report
        report = runner.generate_report()
        
        # Save results
        timestamp = int(time.time())
        results_file = f"benchmark_results_{timestamp}.json"
        runner.save_results(results_file)
        
        return report
    
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG System Benchmark Suite")
    parser.add_argument("--documents", type=int, default=1000, help="Number of test documents")
    parser.add_argument("--queries", type=int, default=100, help="Number of test queries")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        num_documents=args.documents,
        num_queries=args.queries,
        concurrent_users=args.concurrent,
        test_duration_seconds=args.duration
    )
    
    # Run benchmark suite
    async def main():
        report = await run_benchmark_suite(config)
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        if "error" in report:
            print(f"Error: {report['error']}")
            return
        
        summary = report["summary"]
        metrics = report["performance_metrics"]
        
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"Average Throughput: {metrics['avg_throughput']:.2f} req/s")
        print(f"Average Response Time: {metrics['avg_response_time']:.3f}s")
        print(f"Max P99 Response Time: {metrics['max_p99_response_time']:.3f}s")
        
        if report["failed_tests"]:
            print("\nFailed Tests:")
            for failed in report["failed_tests"]:
                print(f"  - {failed['test']}: {', '.join(failed['issues'])}")
        
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
        
        print(f"\nDetailed results saved to: {args.output}")
    
    asyncio.run(main())