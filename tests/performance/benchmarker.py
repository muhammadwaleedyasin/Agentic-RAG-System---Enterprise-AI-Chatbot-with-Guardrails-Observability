"""Performance benchmarking utilities for RAG system components."""

import asyncio
import time
import statistics
import memory_profiler
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    component_name: str
    test_name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    throughput: float
    iterations: int
    timestamp: float
    metadata: Dict[str, Any] = None


class ComponentBenchmarker:
    """Base class for component benchmarking."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.results = []
    
    def benchmark(self, 
                 func: Callable, 
                 test_name: str,
                 iterations: int = 100,
                 warmup_iterations: int = 10,
                 **kwargs) -> BenchmarkResult:
        """
        Benchmark a function.
        
        Args:
            func: Function to benchmark
            test_name: Name of the test
            iterations: Number of iterations
            warmup_iterations: Number of warmup iterations
            **kwargs: Additional arguments for the function
            
        Returns:
            Benchmark result
        """
        # Warmup
        for _ in range(warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(func):
                    asyncio.run(func(**kwargs))
                else:
                    func(**kwargs)
            except Exception:
                pass  # Ignore warmup errors
        
        # Actual benchmark
        execution_times = []
        memory_usage = []
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = initial_memory
        
        for _ in range(iterations):
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                if asyncio.iscoroutinefunction(func):
                    asyncio.run(func(**kwargs))
                else:
                    func(**kwargs)
            except Exception as e:
                # Record failed iteration but continue
                execution_times.append(float('inf'))
                continue
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            execution_times.append(end_time - start_time)
            memory_usage.append(memory_after - memory_before)
            peak_memory = max(peak_memory, memory_after)
        
        # Calculate metrics
        valid_times = [t for t in execution_times if t != float('inf')]
        avg_execution_time = statistics.mean(valid_times) if valid_times else float('inf')
        avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 0
        throughput = 1.0 / avg_execution_time if avg_execution_time > 0 else 0
        
        result = BenchmarkResult(
            component_name=self.component_name,
            test_name=test_name,
            execution_time=avg_execution_time,
            memory_usage_mb=avg_memory_usage,
            peak_memory_mb=peak_memory - initial_memory,
            throughput=throughput,
            iterations=len(valid_times),
            timestamp=time.time()
        )
        
        self.results.append(result)
        return result


class EmbeddingBenchmarker(ComponentBenchmarker):
    """Benchmarker for embedding service."""
    
    def __init__(self, embedding_service):
        super().__init__("EmbeddingService")
        self.embedding_service = embedding_service
    
    async def benchmark_single_text_embedding(self, text: str) -> BenchmarkResult:
        """Benchmark single text embedding."""
        async def embed_text():
            return await self.embedding_service.embed_text(text)
        
        return self.benchmark(
            embed_text,
            "single_text_embedding",
            iterations=100,
            warmup_iterations=10
        )
    
    async def benchmark_batch_embedding(self, texts: List[str], batch_size: int = 32) -> BenchmarkResult:
        """Benchmark batch text embedding."""
        async def embed_batch():
            return await self.embedding_service.embed_texts(texts, batch_size=batch_size)
        
        return self.benchmark(
            embed_batch,
            "batch_text_embedding",
            iterations=20,
            warmup_iterations=3
        )
    
    async def benchmark_embedding_scalability(self, 
                                            base_text: str, 
                                            scale_factors: List[int]) -> List[BenchmarkResult]:
        """Benchmark embedding scalability with different text sizes."""
        results = []
        
        for factor in scale_factors:
            scaled_text = base_text * factor
            result = await self.benchmark_single_text_embedding(scaled_text)
            result.test_name = f"embedding_scale_{factor}x"
            result.metadata = {"scale_factor": factor, "text_length": len(scaled_text)}
            results.append(result)
        
        return results


class VectorStoreBenchmarker(ComponentBenchmarker):
    """Benchmarker for vector store operations."""
    
    def __init__(self, vector_store):
        super().__init__("VectorStore")
        self.vector_store = vector_store
    
    async def benchmark_document_insertion(self, 
                                         documents: List[Dict], 
                                         embeddings: List[List[float]]) -> BenchmarkResult:
        """Benchmark document insertion."""
        async def insert_documents():
            return await self.vector_store.add_documents(documents, embeddings)
        
        return self.benchmark(
            insert_documents,
            "document_insertion",
            iterations=20,
            warmup_iterations=2
        )
    
    async def benchmark_similarity_search(self, 
                                        query_embedding: List[float], 
                                        top_k: int = 10) -> BenchmarkResult:
        """Benchmark similarity search."""
        async def search_similar():
            return await self.vector_store.search_documents(query_embedding, top_k=top_k)
        
        return self.benchmark(
            search_similar,
            "similarity_search",
            iterations=100,
            warmup_iterations=10
        )
    
    async def benchmark_batch_search(self, 
                                   query_embeddings: List[List[float]], 
                                   top_k: int = 10) -> BenchmarkResult:
        """Benchmark batch similarity search."""
        results = []
        
        async def batch_search():
            for embedding in query_embeddings:
                results.append(await self.vector_store.search_documents(embedding, top_k=top_k))
        
        return self.benchmark(
            batch_search,
            "batch_similarity_search",
            iterations=20,
            warmup_iterations=3
        )
    
    async def benchmark_vector_store_scalability(self, 
                                               document_counts: List[int]) -> List[BenchmarkResult]:
        """Benchmark vector store performance with different document counts."""
        results = []
        
        for count in document_counts:
            # Generate test documents
            test_docs = [
                {"id": f"doc_{i}", "content": f"Test document {i}"} 
                for i in range(count)
            ]
            test_embeddings = [[0.1] * 384 for _ in range(count)]
            
            result = await self.benchmark_document_insertion(test_docs, test_embeddings)
            result.test_name = f"insertion_scale_{count}_docs"
            result.metadata = {"document_count": count}
            results.append(result)
        
        return results


class LLMProviderBenchmarker(ComponentBenchmarker):
    """Benchmarker for LLM provider operations."""
    
    def __init__(self, llm_provider):
        super().__init__("LLMProvider")
        self.llm_provider = llm_provider
    
    async def benchmark_response_generation(self, messages: List[Dict[str, str]]) -> BenchmarkResult:
        """Benchmark response generation."""
        async def generate_response():
            return await self.llm_provider.generate_response(messages)
        
        return self.benchmark(
            generate_response,
            "response_generation",
            iterations=20,
            warmup_iterations=2
        )
    
    async def benchmark_streaming_response(self, messages: List[Dict[str, str]]) -> BenchmarkResult:
        """Benchmark streaming response generation."""
        async def generate_stream():
            if hasattr(self.llm_provider, 'generate_stream'):
                chunks = []
                async for chunk in self.llm_provider.generate_stream(messages):
                    chunks.append(chunk)
                return chunks
            else:
                return await self.llm_provider.generate_response(messages)
        
        return self.benchmark(
            generate_stream,
            "streaming_response_generation",
            iterations=10,
            warmup_iterations=1
        )
    
    async def benchmark_concurrent_requests(self, 
                                          messages: List[Dict[str, str]], 
                                          concurrent_count: int = 5) -> BenchmarkResult:
        """Benchmark concurrent request handling."""
        async def concurrent_requests():
            tasks = [
                self.llm_provider.generate_response(messages) 
                for _ in range(concurrent_count)
            ]
            return await asyncio.gather(*tasks)
        
        return self.benchmark(
            concurrent_requests,
            "concurrent_requests",
            iterations=10,
            warmup_iterations=1
        )


class RAGPipelineBenchmarker(ComponentBenchmarker):
    """Benchmarker for complete RAG pipeline."""
    
    def __init__(self, rag_pipeline):
        super().__init__("RAGPipeline")
        self.rag_pipeline = rag_pipeline
    
    async def benchmark_end_to_end_query(self, query: str) -> BenchmarkResult:
        """Benchmark end-to-end query processing."""
        async def process_query():
            return await self.rag_pipeline.generate_response(query)
        
        return self.benchmark(
            process_query,
            "end_to_end_query",
            iterations=50,
            warmup_iterations=5
        )
    
    async def benchmark_query_complexity(self, 
                                       queries: Dict[str, str]) -> List[BenchmarkResult]:
        """Benchmark queries of different complexity levels."""
        results = []
        
        for complexity_level, query in queries.items():
            result = await self.benchmark_end_to_end_query(query)
            result.test_name = f"query_complexity_{complexity_level}"
            result.metadata = {"complexity_level": complexity_level, "query_length": len(query)}
            results.append(result)
        
        return results
    
    async def benchmark_conversation_context(self, 
                                           query: str, 
                                           context_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark performance with different conversation context sizes."""
        results = []
        
        for context_size in context_sizes:
            # Generate mock conversation history
            conversation_history = [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"}
                for i in range(context_size)
            ]
            
            async def query_with_context():
                return await self.rag_pipeline.generate_response(
                    query, conversation_history=conversation_history
                )
            
            result = self.benchmark(
                query_with_context,
                f"conversation_context_{context_size}",
                iterations=20,
                warmup_iterations=2
            )
            result.metadata = {"context_size": context_size}
            results.append(result)
        
        return results


class RAGBenchmarker:
    """Main benchmarker for RAG system components."""
    
    def __init__(self, 
                 rag_pipeline=None,
                 embedding_service=None, 
                 vector_store=None,
                 llm_provider=None):
        """
        Initialize RAG benchmarker.
        
        Args:
            rag_pipeline: Complete RAG pipeline
            embedding_service: Embedding service
            vector_store: Vector store
            llm_provider: LLM provider
        """
        self.rag_pipeline = rag_pipeline
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        
        # Initialize component benchmarkers
        self.benchmarkers = {}
        
        if embedding_service:
            self.benchmarkers['embedding'] = EmbeddingBenchmarker(embedding_service)
        
        if vector_store:
            self.benchmarkers['vector_store'] = VectorStoreBenchmarker(vector_store)
        
        if llm_provider:
            self.benchmarkers['llm'] = LLMProviderBenchmarker(llm_provider)
        
        if rag_pipeline:
            self.benchmarkers['pipeline'] = RAGPipelineBenchmarker(rag_pipeline)
        
        self.all_results = []
    
    async def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across all components."""
        results = {}
        
        # Benchmark embedding service
        if 'embedding' in self.benchmarkers:
            embedding_results = []
            
            # Single text embedding
            single_result = await self.benchmarkers['embedding'].benchmark_single_text_embedding(
                "This is a test document for benchmarking embedding performance."
            )
            embedding_results.append(single_result)
            
            # Batch embedding
            test_texts = [f"Test document {i} for batch embedding." for i in range(50)]
            batch_result = await self.benchmarkers['embedding'].benchmark_batch_embedding(test_texts)
            embedding_results.append(batch_result)
            
            # Scalability test
            scalability_results = await self.benchmarkers['embedding'].benchmark_embedding_scalability(
                "Base text for scalability testing. ", [1, 2, 5, 10]
            )
            embedding_results.extend(scalability_results)
            
            results['embedding'] = embedding_results
        
        # Benchmark vector store
        if 'vector_store' in self.benchmarkers:
            vector_results = []
            
            # Document insertion
            test_docs = [{"id": f"doc_{i}", "content": f"Test document {i}"} for i in range(100)]
            test_embeddings = [[0.1] * 384 for _ in range(100)]
            
            insertion_result = await self.benchmarkers['vector_store'].benchmark_document_insertion(
                test_docs, test_embeddings
            )
            vector_results.append(insertion_result)
            
            # Similarity search
            query_embedding = [0.2] * 384
            search_result = await self.benchmarkers['vector_store'].benchmark_similarity_search(
                query_embedding, top_k=10
            )
            vector_results.append(search_result)
            
            results['vector_store'] = vector_results
        
        # Benchmark LLM provider
        if 'llm' in self.benchmarkers:
            llm_results = []
            
            test_messages = [{"role": "user", "content": "What is machine learning?"}]
            
            # Response generation
            response_result = await self.benchmarkers['llm'].benchmark_response_generation(test_messages)
            llm_results.append(response_result)
            
            # Streaming response
            stream_result = await self.benchmarkers['llm'].benchmark_streaming_response(test_messages)
            llm_results.append(stream_result)
            
            # Concurrent requests
            concurrent_result = await self.benchmarkers['llm'].benchmark_concurrent_requests(
                test_messages, concurrent_count=3
            )
            llm_results.append(concurrent_result)
            
            results['llm'] = llm_results
        
        # Benchmark RAG pipeline
        if 'pipeline' in self.benchmarkers:
            pipeline_results = []
            
            # End-to-end query
            e2e_result = await self.benchmarkers['pipeline'].benchmark_end_to_end_query(
                "What is artificial intelligence and how does it work?"
            )
            pipeline_results.append(e2e_result)
            
            # Query complexity
            complexity_queries = {
                "simple": "What is AI?",
                "medium": "Explain the differences between machine learning and deep learning.",
                "complex": "How do transformer architectures work in natural language processing, and what are the key innovations that make them effective for understanding context?"
            }
            
            complexity_results = await self.benchmarkers['pipeline'].benchmark_query_complexity(
                complexity_queries
            )
            pipeline_results.extend(complexity_results)
            
            results['pipeline'] = pipeline_results
        
        # Store all results
        for component_results in results.values():
            self.all_results.extend(component_results)
        
        return results
    
    async def run_performance_comparison(self, 
                                       configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare performance across different configurations.
        
        Args:
            configurations: List of configuration dictionaries
            
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        for i, config in enumerate(configurations):
            config_name = config.get('name', f'config_{i}')
            
            # Apply configuration (mock implementation)
            # In real implementation, you would reconfigure components
            
            # Run benchmarks for this configuration
            config_results = await self.run_comprehensive_benchmark()
            comparison_results[config_name] = config_results
        
        # Analyze differences
        analysis = self._analyze_configuration_differences(comparison_results)
        
        return {
            'configurations': comparison_results,
            'analysis': analysis
        }
    
    def _analyze_configuration_differences(self, 
                                         comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze differences between configurations."""
        analysis = {
            'best_configuration': None,
            'performance_rankings': {},
            'key_insights': []
        }
        
        # Simple analysis based on overall performance
        config_scores = {}
        
        for config_name, results in comparison_results.items():
            total_score = 0
            count = 0
            
            for component_results in results.values():
                for result in component_results:
                    if result.execution_time > 0:
                        # Lower execution time is better
                        score = 1.0 / result.execution_time
                        total_score += score
                        count += 1
            
            config_scores[config_name] = total_score / count if count > 0 else 0
        
        # Rank configurations
        ranked_configs = sorted(config_scores.items(), key=lambda x: x[1], reverse=True)
        analysis['best_configuration'] = ranked_configs[0][0] if ranked_configs else None
        analysis['performance_rankings'] = dict(ranked_configs)
        
        return analysis
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.all_results:
            return {"error": "No benchmark results available"}
        
        # Group results by component
        component_results = {}
        for result in self.all_results:
            if result.component_name not in component_results:
                component_results[result.component_name] = []
            component_results[result.component_name].append(result)
        
        # Calculate summary statistics
        summary = {}
        for component, results in component_results.items():
            execution_times = [r.execution_time for r in results if r.execution_time != float('inf')]
            memory_usage = [r.memory_usage_mb for r in results]
            throughput = [r.throughput for r in results if r.throughput > 0]
            
            summary[component] = {
                'test_count': len(results),
                'avg_execution_time': statistics.mean(execution_times) if execution_times else 0,
                'median_execution_time': statistics.median(execution_times) if execution_times else 0,
                'avg_memory_usage_mb': statistics.mean(memory_usage) if memory_usage else 0,
                'avg_throughput': statistics.mean(throughput) if throughput else 0,
                'peak_throughput': max(throughput) if throughput else 0
            }
        
        # Identify performance bottlenecks
        bottlenecks = []
        for component, stats in summary.items():
            if stats['avg_execution_time'] > 1.0:  # More than 1 second average
                bottlenecks.append({
                    'component': component,
                    'issue': 'High execution time',
                    'value': stats['avg_execution_time']
                })
            
            if stats['avg_memory_usage_mb'] > 100:  # More than 100MB average
                bottlenecks.append({
                    'component': component,
                    'issue': 'High memory usage',
                    'value': stats['avg_memory_usage_mb']
                })
        
        # Generate recommendations
        recommendations = []
        if bottlenecks:
            recommendations.append("Performance bottlenecks detected. Consider optimization.")
        
        for component, stats in summary.items():
            if stats['avg_throughput'] < 10:  # Less than 10 ops/sec
                recommendations.append(f"Consider optimizing {component} for better throughput")
        
        return {
            'timestamp': time.time(),
            'summary_statistics': summary,
            'total_tests': len(self.all_results),
            'performance_bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'detailed_results': [
                {
                    'component': r.component_name,
                    'test': r.test_name,
                    'execution_time': r.execution_time,
                    'memory_usage_mb': r.memory_usage_mb,
                    'throughput': r.throughput,
                    'iterations': r.iterations
                }
                for r in self.all_results
            ]
        }
    
    def export_benchmark_results(self, filepath: str, format: str = 'json'):
        """Export benchmark results to file."""
        report = self.generate_benchmark_report()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Benchmark results exported to {filepath}")
    
    def get_performance_trends(self, 
                             time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        current_time = time.time()
        cutoff_time = current_time - (time_window_hours * 3600)
        
        recent_results = [
            r for r in self.all_results 
            if r.timestamp >= cutoff_time
        ]
        
        if not recent_results:
            return {"message": "No recent results for trend analysis"}
        
        # Group by time intervals
        interval_duration = 3600  # 1 hour intervals
        intervals = {}
        
        for result in recent_results:
            interval_key = int((result.timestamp - cutoff_time) // interval_duration)
            if interval_key not in intervals:
                intervals[interval_key] = []
            intervals[interval_key].append(result)
        
        # Calculate trends
        trends = {}
        for interval, results in intervals.items():
            avg_execution_time = statistics.mean([
                r.execution_time for r in results 
                if r.execution_time != float('inf')
            ])
            avg_memory = statistics.mean([r.memory_usage_mb for r in results])
            
            trends[interval] = {
                'avg_execution_time': avg_execution_time,
                'avg_memory_usage': avg_memory,
                'test_count': len(results)
            }
        
        return {
            'time_window_hours': time_window_hours,
            'trends': trends,
            'analysis': "Performance trend analysis completed"
        }