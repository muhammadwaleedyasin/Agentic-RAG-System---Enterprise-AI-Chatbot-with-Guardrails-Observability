"""
Phoenix integration for RAG pipeline observability and tracing.

Provides comprehensive monitoring using Phoenix by Arize AI:
- Application performance monitoring
- LLM evaluation and tracking
- Embedding visualization
- Data drift detection
- Real-time monitoring dashboard
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging

try:
    import phoenix as px
    from phoenix.trace import tracer
    from phoenix.trace.openai import OpenAIInstrumentor
    from phoenix.trace.langchain import LangChainInstrumentor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

logger = logging.getLogger(__name__)


class PhoenixTracer:
    """Phoenix integration for RAG pipeline observability."""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        enabled: bool = True,
        launch_app: bool = False,
        port: int = 6006
    ):
        """Initialize Phoenix tracer.
        
        Args:
            endpoint: Phoenix endpoint URL
            enabled: Whether tracing is enabled
            launch_app: Whether to launch Phoenix app locally
            port: Port for local Phoenix app
        """
        self.enabled = enabled and PHOENIX_AVAILABLE
        self.endpoint = endpoint or os.getenv("PHOENIX_ENDPOINT")
        self.port = port
        self.session = None
        self.tracer_instance = None
        
        if not self.enabled:
            if not PHOENIX_AVAILABLE:
                logger.warning("Phoenix not available. Install with: pip install arize-phoenix")
            else:
                logger.info("Phoenix tracing disabled")
            return
        
        try:
            self._initialize_phoenix(launch_app)
            self._setup_tracing()
            logger.info("Phoenix tracer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix: {e}")
            self.enabled = False
    
    def _initialize_phoenix(self, launch_app: bool):
        """Initialize Phoenix session."""
        if launch_app:
            # Launch local Phoenix app
            self.session = px.launch_app(port=self.port)
            logger.info(f"Phoenix app launched at http://localhost:{self.port}")
        elif self.endpoint:
            # Connect to remote Phoenix instance
            self.session = px.Client(endpoint=self.endpoint)
            logger.info(f"Connected to Phoenix at {self.endpoint}")
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing with Phoenix."""
        if self.endpoint:
            # Use OTLP exporter for remote Phoenix
            exporter = OTLPSpanExporter(endpoint=f"{self.endpoint}/v1/traces")
        else:
            # Use local Phoenix exporter
            from phoenix.trace.exporter import PhoenixSpanExporter
            exporter = PhoenixSpanExporter()
        
        # Setup tracer provider
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(tracer_provider)
        
        # Get tracer instance
        self.tracer_instance = trace.get_tracer(__name__)
        
        # Auto-instrument OpenAI and LangChain if available
        try:
            OpenAIInstrumentor().instrument()
            logger.info("OpenAI auto-instrumentation enabled")
        except Exception as e:
            logger.debug(f"OpenAI instrumentation failed: {e}")
        
        try:
            LangChainInstrumentor().instrument()
            logger.info("LangChain auto-instrumentation enabled")
        except Exception as e:
            logger.debug(f"LangChain instrumentation failed: {e}")
    
    @asynccontextmanager
    async def trace_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        span_kind: str = "INTERNAL"
    ):
        """Create a traced span context."""
        if not self.enabled or not self.tracer_instance:
            yield DummyPhoenixSpan()
            return
        
        with self.tracer_instance.start_as_current_span(
            name,
            attributes=attributes or {}
        ) as span:
            phoenix_span = PhoenixSpan(span)
            try:
                yield phoenix_span
            except Exception as e:
                phoenix_span.set_status("ERROR", str(e))
                phoenix_span.set_attribute("error.type", type(e).__name__)
                raise
    
    def trace_document_ingestion(
        self,
        document_path: str,
        chunk_count: int,
        processing_time: float
    ):
        """Trace document ingestion process."""
        if not self.enabled:
            return
        
        with self.tracer_instance.start_as_current_span("document_ingestion") as span:
            span.set_attributes({
                "document.path": document_path,
                "document.chunk_count": chunk_count,
                "processing.time_seconds": processing_time,
                "operation.type": "ingestion"
            })
    
    def trace_embedding_generation(
        self,
        text_count: int,
        model: str,
        dimensions: int,
        processing_time: float
    ):
        """Trace embedding generation."""
        if not self.enabled:
            return
        
        with self.tracer_instance.start_as_current_span("embedding_generation") as span:
            span.set_attributes({
                "embedding.model": model,
                "embedding.dimensions": dimensions,
                "embedding.text_count": text_count,
                "processing.time_seconds": processing_time,
                "operation.type": "embedding"
            })
    
    def trace_vector_search(
        self,
        query: str,
        top_k: int,
        results_count: int,
        search_time: float,
        similarity_scores: Optional[List[float]] = None
    ):
        """Trace vector similarity search."""
        if not self.enabled:
            return
        
        with self.tracer_instance.start_as_current_span("vector_search") as span:
            attributes = {
                "search.query": query[:200],  # Truncate long queries
                "search.top_k": top_k,
                "search.results_count": results_count,
                "search.time_seconds": search_time,
                "operation.type": "retrieval"
            }
            
            if similarity_scores:
                attributes.update({
                    "search.max_score": max(similarity_scores),
                    "search.min_score": min(similarity_scores),
                    "search.avg_score": sum(similarity_scores) / len(similarity_scores)
                })
            
            span.set_attributes(attributes)
    
    def trace_llm_generation(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency: float,
        cost: Optional[float] = None
    ):
        """Trace LLM generation with detailed metrics."""
        if not self.enabled:
            return
        
        with self.tracer_instance.start_as_current_span("llm_generation") as span:
            attributes = {
                "llm.model": model,
                "llm.prompt_tokens": prompt_tokens,
                "llm.completion_tokens": completion_tokens,
                "llm.total_tokens": total_tokens,
                "llm.latency_seconds": latency,
                "operation.type": "generation"
            }
            
            if cost is not None:
                attributes["llm.cost_usd"] = cost
            
            span.set_attributes(attributes)
    
    def trace_rag_pipeline(
        self,
        query: str,
        retrieved_docs: int,
        total_latency: float,
        user_id: Optional[str] = None
    ):
        """Trace complete RAG pipeline execution."""
        if not self.enabled:
            return
        
        with self.tracer_instance.start_as_current_span("rag_pipeline") as span:
            attributes = {
                "rag.query": query[:200],
                "rag.retrieved_docs": retrieved_docs,
                "rag.total_latency_seconds": total_latency,
                "operation.type": "rag_pipeline"
            }
            
            if user_id:
                attributes["user.id"] = user_id
            
            span.set_attributes(attributes)
    
    def log_evaluation_metrics(
        self,
        query: str,
        response: str,
        relevance_score: Optional[float] = None,
        hallucination_score: Optional[float] = None,
        toxicity_score: Optional[float] = None
    ):
        """Log evaluation metrics for response quality."""
        if not self.enabled:
            return
        
        with self.tracer_instance.start_as_current_span("evaluation") as span:
            attributes = {
                "eval.query": query[:200],
                "eval.response": response[:500],
                "operation.type": "evaluation"
            }
            
            if relevance_score is not None:
                attributes["eval.relevance_score"] = relevance_score
            if hallucination_score is not None:
                attributes["eval.hallucination_score"] = hallucination_score
            if toxicity_score is not None:
                attributes["eval.toxicity_score"] = toxicity_score
            
            span.set_attributes(attributes)
    
    def get_dashboard_url(self) -> Optional[str]:
        """Get URL for Phoenix dashboard."""
        if not self.enabled:
            return None
        
        if self.endpoint:
            return self.endpoint
        else:
            return f"http://localhost:{self.port}"
    
    def close(self):
        """Close Phoenix session."""
        if self.session and hasattr(self.session, 'close'):
            self.session.close()


class PhoenixSpan:
    """Wrapper for Phoenix/OpenTelemetry span."""
    
    def __init__(self, span):
        self.span = span
    
    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.span.set_attribute(key, value)
    
    def set_attributes(self, attributes: Dict[str, Any]):
        """Set multiple span attributes."""
        for key, value in attributes.items():
            self.span.set_attribute(key, value)
    
    def set_status(self, status: str, description: Optional[str] = None):
        """Set span status."""
        from opentelemetry.trace import Status, StatusCode
        
        if status == "OK":
            self.span.set_status(Status(StatusCode.OK, description))
        elif status == "ERROR":
            self.span.set_status(Status(StatusCode.ERROR, description))
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span."""
        self.span.add_event(name, attributes or {})


class DummyPhoenixSpan:
    """Dummy span for when Phoenix is disabled."""
    
    def set_attribute(self, key: str, value: Any):
        pass
    
    def set_attributes(self, attributes: Dict[str, Any]):
        pass
    
    def set_status(self, status: str, description: Optional[str] = None):
        pass
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        pass


class RAGPhoenixTracer:
    """High-level Phoenix tracing utilities for RAG systems."""
    
    def __init__(self, tracer: PhoenixTracer):
        self.tracer = tracer
    
    async def trace_query_flow(
        self,
        query: str,
        user_id: Optional[str] = None
    ):
        """Trace complete query processing flow."""
        async with self.tracer.trace_span(
            "query_flow",
            attributes={
                "query": query[:200],
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        ) as span:
            return span
    
    async def trace_retrieval_augmentation(
        self,
        query: str,
        context_docs: List[str],
        augmented_prompt: str
    ):
        """Trace retrieval augmentation process."""
        async with self.tracer.trace_span(
            "retrieval_augmentation",
            attributes={
                "query": query[:200],
                "context_docs_count": len(context_docs),
                "augmented_prompt_length": len(augmented_prompt)
            }
        ) as span:
            return span
    
    def log_performance_metrics(
        self,
        operation: str,
        latency: float,
        throughput: Optional[float] = None,
        error_rate: Optional[float] = None
    ):
        """Log performance metrics."""
        self.tracer.tracer_instance.start_as_current_span(
            f"performance_{operation}"
        ).set_attributes({
            "performance.operation": operation,
            "performance.latency_seconds": latency,
            "performance.throughput": throughput,
            "performance.error_rate": error_rate
        })


# Global tracer instance
_global_phoenix_tracer: Optional[PhoenixTracer] = None


def get_phoenix_tracer() -> PhoenixTracer:
    """Get global Phoenix tracer instance."""
    global _global_phoenix_tracer
    if _global_phoenix_tracer is None:
        _global_phoenix_tracer = PhoenixTracer()
    return _global_phoenix_tracer


def configure_phoenix_tracer(
    endpoint: Optional[str] = None,
    enabled: bool = True,
    launch_app: bool = False,
    port: int = 6006
) -> PhoenixTracer:
    """Configure global Phoenix tracer."""
    global _global_phoenix_tracer
    _global_phoenix_tracer = PhoenixTracer(
        endpoint=endpoint,
        enabled=enabled,
        launch_app=launch_app,
        port=port
    )
    return _global_phoenix_tracer