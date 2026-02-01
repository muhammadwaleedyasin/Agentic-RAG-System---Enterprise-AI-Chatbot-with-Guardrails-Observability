"""
Observability mixin for RAG pipeline components.

Provides a common interface for adding tracing and metrics
to core RAG components like document processing, embedding,
vector storage, and query processing.
"""

import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import logging

from ..observability.langfuse_integration import get_tracer as get_langfuse_tracer
from ..observability.phoenix_integration import get_phoenix_tracer
from ..observability.metrics_collector import get_metrics_collector
from ..config.observability import get_observability_config

logger = logging.getLogger(__name__)


class ObservabilityMixin:
    """Mixin class for adding observability to RAG components."""
    
    def __init__(self):
        """Initialize observability components."""
        self.obs_config = get_observability_config()
        self.langfuse_tracer = get_langfuse_tracer() if self.obs_config.langfuse.enabled else None
        self.phoenix_tracer = get_phoenix_tracer() if self.obs_config.phoenix.enabled else None
        self.metrics_collector = get_metrics_collector() if self.obs_config.metrics.enabled else None
        
        # Component-specific metrics prefix
        self.component_name = getattr(self, '__class__', type(self)).__name__.lower()
    
    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing operations."""
        start_time = time.time()
        
        # Start Langfuse tracing
        langfuse_trace = None
        if self.langfuse_tracer:
            langfuse_trace = await self.langfuse_tracer.trace_context(
                name=f"{self.component_name}_{operation_name}",
                input_data=input_data,
                metadata=metadata
            )
        
        # Start Phoenix tracing
        phoenix_span = None
        if self.phoenix_tracer:
            phoenix_span = await self.phoenix_tracer.trace_span(
                name=f"{self.component_name}_{operation_name}",
                attributes={
                    "component": self.component_name,
                    "operation": operation_name,
                    **(metadata or {})
                }
            )
        
        try:
            if langfuse_trace:
                async with langfuse_trace as trace:
                    if phoenix_span:
                        async with phoenix_span as span:
                            yield {"langfuse_trace": trace, "phoenix_span": span}
                    else:
                        yield {"langfuse_trace": trace, "phoenix_span": None}
            elif phoenix_span:
                async with phoenix_span as span:
                    yield {"langfuse_trace": None, "phoenix_span": span}
            else:
                yield {"langfuse_trace": None, "phoenix_span": None}
                
        except Exception as e:
            # Record error metrics
            if self.metrics_collector:
                self.metrics_collector.increment(
                    f"{self.component_name}.errors",
                    tags={
                        "operation": operation_name,
                        "error_type": type(e).__name__
                    }
                )
            raise
        finally:
            # Record timing metrics
            duration = time.time() - start_time
            if self.metrics_collector:
                self.metrics_collector.histogram(
                    f"{self.component_name}.operation.duration",
                    duration,
                    tags={"operation": operation_name}
                )
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: str = "histogram",
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric with component prefix."""
        if not self.metrics_collector:
            return
        
        full_metric_name = f"{self.component_name}.{metric_name}"
        component_tags = {"component": self.component_name}
        if tags:
            component_tags.update(tags)
        
        if metric_type == "histogram":
            self.metrics_collector.histogram(full_metric_name, value, component_tags)
        elif metric_type == "counter":
            self.metrics_collector.increment(full_metric_name, value, component_tags)
        elif metric_type == "gauge":
            self.metrics_collector.gauge(full_metric_name, value, component_tags)
    
    def log_to_langfuse(
        self,
        trace_id: str,
        operation: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log operation details to Langfuse."""
        if not self.langfuse_tracer:
            return
        
        try:
            self.langfuse_tracer.client.span(
                trace_id=trace_id,
                name=f"{self.component_name}_{operation}",
                input=input_data,
                output=output_data,
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Failed to log to Langfuse: {e}")
    
    def log_to_phoenix(
        self,
        operation: str,
        attributes: Dict[str, Any]
    ):
        """Log operation details to Phoenix."""
        if not self.phoenix_tracer:
            return
        
        try:
            component_attributes = {
                "component": self.component_name,
                "operation": operation,
                **attributes
            }
            
            with self.phoenix_tracer.tracer_instance.start_as_current_span(
                f"{self.component_name}_{operation}"
            ) as span:
                span.set_attributes(component_attributes)
                
        except Exception as e:
            logger.error(f"Failed to log to Phoenix: {e}")


class DocumentProcessorObservability(ObservabilityMixin):
    """Observability for document processing operations."""
    
    def trace_document_chunking(
        self,
        document_size: int,
        chunk_size: int,
        chunk_overlap: int,
        chunks_created: int,
        processing_time: float
    ):
        """Trace document chunking operation."""
        self.record_metric("document.size_bytes", document_size)
        self.record_metric("document.chunks_created", chunks_created)
        self.record_metric("document.processing_time", processing_time)
        
        if self.phoenix_tracer:
            self.phoenix_tracer.trace_document_ingestion(
                document_path="",  # Will be filled by caller
                chunk_count=chunks_created,
                processing_time=processing_time
            )
    
    def trace_text_extraction(
        self,
        file_type: str,
        extraction_time: float,
        text_length: int,
        success: bool = True
    ):
        """Trace text extraction from documents."""
        self.record_metric("text_extraction.time", extraction_time)
        self.record_metric("text_extraction.length", text_length)
        
        if not success:
            self.record_metric(
                "text_extraction.errors",
                1,
                "counter",
                tags={"file_type": file_type}
            )


class EmbeddingServiceObservability(ObservabilityMixin):
    """Observability for embedding operations."""
    
    def trace_embedding_generation(
        self,
        text_count: int,
        model: str,
        dimensions: int,
        processing_time: float,
        cost: Optional[float] = None
    ):
        """Trace embedding generation."""
        if self.metrics_collector:
            self.metrics_collector.record_embedding_metrics(
                text_count=text_count,
                model=model,
                processing_time=processing_time,
                cost=cost
            )
        
        if self.phoenix_tracer:
            self.phoenix_tracer.trace_embedding_generation(
                text_count=text_count,
                model=model,
                dimensions=dimensions,
                processing_time=processing_time
            )
    
    def trace_batch_embedding(
        self,
        batch_size: int,
        total_batches: int,
        batch_processing_time: float,
        model: str
    ):
        """Trace batch embedding processing."""
        self.record_metric("embedding.batch_size", batch_size)
        self.record_metric("embedding.batch_time", batch_processing_time)
        self.record_metric("embedding.total_batches", total_batches, "gauge")


class VectorStoreObservability(ObservabilityMixin):
    """Observability for vector store operations."""
    
    def trace_vector_insertion(
        self,
        vector_count: int,
        insertion_time: float,
        index_size: Optional[int] = None
    ):
        """Trace vector insertion operations."""
        self.record_metric("vectors.inserted", vector_count)
        self.record_metric("vectors.insertion_time", insertion_time)
        
        if index_size is not None:
            self.record_metric("index.size", index_size, "gauge")
    
    def trace_vector_search(
        self,
        query_embedding_time: float,
        search_time: float,
        results_count: int,
        similarity_scores: List[float],
        top_k: int
    ):
        """Trace vector similarity search."""
        if self.metrics_collector:
            self.metrics_collector.record_retrieval_metrics(
                query_embedding_time=query_embedding_time,
                search_time=search_time,
                results_count=results_count,
                top_k=top_k
            )
        
        if self.phoenix_tracer:
            self.phoenix_tracer.trace_vector_search(
                query="",  # Will be filled by caller
                top_k=top_k,
                results_count=results_count,
                search_time=search_time,
                similarity_scores=similarity_scores
            )


class QueryProcessorObservability(ObservabilityMixin):
    """Observability for query processing operations."""
    
    def trace_query_analysis(
        self,
        query: str,
        analysis_time: float,
        query_type: str,
        complexity_score: Optional[float] = None
    ):
        """Trace query analysis and understanding."""
        self.record_metric("query.analysis_time", analysis_time)
        self.record_metric("query.length", len(query))
        
        if complexity_score is not None:
            self.record_metric("query.complexity", complexity_score)
        
        self.log_to_phoenix(
            "query_analysis",
            {
                "query_type": query_type,
                "query_length": len(query),
                "complexity_score": complexity_score
            }
        )
    
    def trace_response_generation(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        generation_time: float,
        cost: Optional[float] = None
    ):
        """Trace LLM response generation."""
        if self.metrics_collector:
            self.metrics_collector.record_llm_metrics(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency=generation_time,
                cost=cost
            )
        
        if self.phoenix_tracer:
            self.phoenix_tracer.trace_llm_generation(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency=generation_time,
                cost=cost
            )
    
    def trace_response_quality(
        self,
        relevance_score: Optional[float] = None,
        hallucination_score: Optional[float] = None,
        coherence_score: Optional[float] = None
    ):
        """Trace response quality metrics."""
        if self.metrics_collector:
            self.metrics_collector.record_quality_metrics(
                relevance_score=relevance_score,
                hallucination_score=hallucination_score,
                coherence_score=coherence_score
            )


# Factory function to get appropriate observability mixin
def get_observability_mixin(component_type: str) -> ObservabilityMixin:
    """Get appropriate observability mixin for component type."""
    mixins = {
        "document_processor": DocumentProcessorObservability,
        "embedding_service": EmbeddingServiceObservability,
        "vector_store": VectorStoreObservability,
        "query_processor": QueryProcessorObservability
    }
    
    mixin_class = mixins.get(component_type, ObservabilityMixin)
    return mixin_class()