"""
Langfuse integration for RAG pipeline tracing and observability.

Provides comprehensive tracing for all RAG components including:
- Document ingestion and chunking
- Vector embeddings and storage
- Retrieval and ranking
- LLM generation and synthesis
- Cost and latency tracking
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from functools import wraps
from contextlib import asynccontextmanager
import json

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe
    from langfuse.openai import openai
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Langfuse integration for RAG pipeline observability."""
    
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: bool = True
    ):
        """Initialize Langfuse tracer.
        
        Args:
            public_key: Langfuse public key (or from LANGFUSE_PUBLIC_KEY env)
            secret_key: Langfuse secret key (or from LANGFUSE_SECRET_KEY env)
            host: Langfuse host URL (or from LANGFUSE_HOST env)
            enabled: Whether tracing is enabled
        """
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self.client = None
        
        if not self.enabled:
            if not LANGFUSE_AVAILABLE:
                logger.warning("Langfuse not available. Install with: pip install langfuse")
            else:
                logger.info("Langfuse tracing disabled")
            return
            
        try:
            self.client = Langfuse(
                public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
                host=host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )
            logger.info("Langfuse tracer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.enabled = False
    
    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Create a new trace for RAG pipeline execution."""
        if not self.enabled:
            return DummyTrace()
            
        try:
            return self.client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or []
            )
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return DummyTrace()
    
    def create_span(
        self,
        trace_id: str,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: str = "DEFAULT"
    ):
        """Create a span within a trace."""
        if not self.enabled:
            return DummySpan()
            
        try:
            return self.client.span(
                trace_id=trace_id,
                name=name,
                input=input_data,
                metadata=metadata or {},
                level=level
            )
        except Exception as e:
            logger.error(f"Failed to create span: {e}")
            return DummySpan()
    
    def log_generation(
        self,
        trace_id: str,
        name: str,
        input_data: Dict[str, Any],
        output: str,
        model: str,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log LLM generation details."""
        if not self.enabled:
            return
            
        try:
            self.client.generation(
                trace_id=trace_id,
                name=name,
                input=input_data,
                output=output,
                model=model,
                usage=usage,
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Failed to log generation: {e}")
    
    def log_retrieval(
        self,
        trace_id: str,
        query: str,
        results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log retrieval operation details."""
        if not self.enabled:
            return
            
        try:
            self.client.span(
                trace_id=trace_id,
                name="vector_retrieval",
                input={"query": query},
                output={"results": results, "count": len(results)},
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Failed to log retrieval: {e}")
    
    def log_embedding(
        self,
        trace_id: str,
        text: str,
        model: str,
        dimensions: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log embedding generation."""
        if not self.enabled:
            return
            
        try:
            self.client.span(
                trace_id=trace_id,
                name="embedding_generation",
                input={"text": text[:200] + "..." if len(text) > 200 else text},
                output={"model": model, "dimensions": dimensions},
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Failed to log embedding: {e}")
    
    def log_document_processing(
        self,
        trace_id: str,
        document_name: str,
        chunks_created: int,
        processing_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log document ingestion and processing."""
        if not self.enabled:
            return
            
        try:
            self.client.span(
                trace_id=trace_id,
                name="document_processing",
                input={"document": document_name},
                output={
                    "chunks_created": chunks_created,
                    "processing_time_seconds": processing_time
                },
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Failed to log document processing: {e}")
    
    def trace_rag_pipeline(self, session_id: Optional[str] = None):
        """Decorator for tracing complete RAG pipeline execution."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                    
                trace = self.create_trace(
                    name=f"rag_pipeline_{func.__name__}",
                    session_id=session_id,
                    metadata={"function": func.__name__}
                )
                
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    trace.update(
                        output={"success": True, "execution_time": execution_time}
                    )
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    trace.update(
                        output={
                            "success": False,
                            "error": str(e),
                            "execution_time": execution_time
                        }
                    )
                    raise
                finally:
                    if self.client:
                        self.client.flush()
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                    
                trace = self.create_trace(
                    name=f"rag_pipeline_{func.__name__}",
                    session_id=session_id,
                    metadata={"function": func.__name__}
                )
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    trace.update(
                        output={"success": True, "execution_time": execution_time}
                    )
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    trace.update(
                        output={
                            "success": False,
                            "error": str(e),
                            "execution_time": execution_time
                        }
                    )
                    raise
                finally:
                    if self.client:
                        self.client.flush()
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    @asynccontextmanager
    async def trace_context(
        self,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing operations."""
        if not self.enabled:
            yield DummyTrace()
            return
            
        trace = self.create_trace(
            name=name,
            metadata=metadata
        )
        
        start_time = time.time()
        try:
            if input_data:
                trace.update(input=input_data)
            yield trace
        except Exception as e:
            execution_time = time.time() - start_time
            trace.update(
                output={
                    "success": False,
                    "error": str(e),
                    "execution_time": execution_time
                }
            )
            raise
        else:
            execution_time = time.time() - start_time
            trace.update(
                output={"success": True, "execution_time": execution_time}
            )
        finally:
            if self.client:
                self.client.flush()
    
    def get_trace_url(self, trace_id: str) -> Optional[str]:
        """Get URL for viewing trace in Langfuse dashboard."""
        if not self.enabled or not self.client:
            return None
            
        host = getattr(self.client, 'host', 'https://cloud.langfuse.com')
        return f"{host}/trace/{trace_id}"
    
    def flush(self):
        """Flush any pending traces."""
        if self.enabled and self.client:
            self.client.flush()


class DummyTrace:
    """Dummy trace for when Langfuse is disabled."""
    
    def __init__(self):
        self.id = "dummy"
    
    def update(self, **kwargs):
        pass
    
    def span(self, **kwargs):
        return DummySpan()


class DummySpan:
    """Dummy span for when Langfuse is disabled."""
    
    def __init__(self):
        self.id = "dummy"
    
    def update(self, **kwargs):
        pass
    
    def end(self, **kwargs):
        pass


class RAGTracer:
    """High-level RAG tracing utilities."""
    
    def __init__(self, tracer: LangfuseTracer):
        self.tracer = tracer
    
    async def trace_document_ingestion(
        self,
        document_path: str,
        chunk_size: int,
        chunk_overlap: int
    ):
        """Trace document ingestion process."""
        async with self.tracer.trace_context(
            name="document_ingestion",
            input_data={
                "document_path": document_path,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        ) as trace:
            return trace
    
    async def trace_query_processing(
        self,
        query: str,
        user_id: Optional[str] = None
    ):
        """Trace query processing and response generation."""
        async with self.tracer.trace_context(
            name="query_processing",
            input_data={"query": query, "user_id": user_id}
        ) as trace:
            return trace
    
    def trace_vector_search(
        self,
        trace_id: str,
        query: str,
        top_k: int,
        results: List[Dict[str, Any]]
    ):
        """Trace vector similarity search."""
        self.tracer.log_retrieval(
            trace_id=trace_id,
            query=query,
            results=results,
            metadata={"top_k": top_k, "results_count": len(results)}
        )
    
    def trace_llm_generation(
        self,
        trace_id: str,
        prompt: str,
        response: str,
        model: str,
        usage: Optional[Dict[str, Any]] = None
    ):
        """Trace LLM generation."""
        self.tracer.log_generation(
            trace_id=trace_id,
            name="llm_generation",
            input_data={"prompt": prompt},
            output=response,
            model=model,
            usage=usage
        )


# Global tracer instance
_global_tracer: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    """Get global Langfuse tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = LangfuseTracer()
    return _global_tracer


def configure_tracer(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
    enabled: bool = True
) -> LangfuseTracer:
    """Configure global Langfuse tracer."""
    global _global_tracer
    _global_tracer = LangfuseTracer(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        enabled=enabled
    )
    return _global_tracer