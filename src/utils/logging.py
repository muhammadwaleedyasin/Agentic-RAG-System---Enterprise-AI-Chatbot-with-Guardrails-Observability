"""Logging utilities for the RAG system."""

import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from ..config.settings import get_settings


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    if settings.debug:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler for structured logs
    structured_format = json.dumps({
        "timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSS}",
        "level": "{level}",
        "module": "{name}",
        "function": "{function}",
        "line": "{line}",
        "message": "{message}",
        "extra": "{extra}"
    })
    
    logger.add(
        "logs/rag_system.log",
        format=structured_format,
        level="INFO",
        rotation="1 day",
        retention="30 days",
        compression="gz",
        serialize=True
    )
    
    # Error logs
    logger.add(
        "logs/errors.log",
        format=structured_format,
        level="ERROR",
        rotation="1 week",
        retention="4 weeks",
        compression="gz",
        serialize=True
    )


def log_operation(
    operation: str,
    status: str,
    duration: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> None:
    """Log an operation with structured metadata."""
    log_data = {
        "operation": operation,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if duration is not None:
        log_data["duration_seconds"] = duration
    
    if metadata:
        log_data["metadata"] = metadata
    
    if error:
        log_data["error"] = error
    
    if status == "success":
        logger.info(f"Operation completed: {operation}", extra=log_data)
    elif status == "error":
        logger.error(f"Operation failed: {operation}", extra=log_data)
    else:
        logger.info(f"Operation {status}: {operation}", extra=log_data)


def log_ingestion(
    file_path: str,
    status: str,
    document_count: int = 0,
    chunk_count: int = 0,
    duration: Optional[float] = None,
    error: Optional[str] = None
) -> None:
    """Log document ingestion operation."""
    metadata = {
        "file_path": file_path,
        "document_count": document_count,
        "chunk_count": chunk_count,
    }
    
    log_operation(
        operation="document_ingestion",
        status=status,
        duration=duration,
        metadata=metadata,
        error=error
    )


def log_search(
    query: str,
    result_count: int,
    duration: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    reranked: bool = False
) -> None:
    """Log search operation."""
    metadata = {
        "query": query,
        "result_count": result_count,
        "filters": filters,
        "reranked": reranked,
    }
    
    log_operation(
        operation="vector_search",
        status="success",
        duration=duration,
        metadata=metadata
    )


def log_rag_response(
    query: str,
    answer_length: int,
    source_count: int,
    confidence: float,
    model: str,
    duration: Optional[float] = None
) -> None:
    """Log RAG response generation."""
    metadata = {
        "query": query,
        "answer_length": answer_length,
        "source_count": source_count,
        "confidence": confidence,
        "model": model,
    }
    
    log_operation(
        operation="rag_response",
        status="success",
        duration=duration,
        metadata=metadata
    )


def log_embedding(
    text_count: int,
    provider: str,
    model: str,
    token_count: int,
    duration: Optional[float] = None,
    error: Optional[str] = None
) -> None:
    """Log embedding generation."""
    metadata = {
        "text_count": text_count,
        "provider": provider,
        "model": model,
        "token_count": token_count,
    }
    
    status = "error" if error else "success"
    
    log_operation(
        operation="embedding_generation",
        status=status,
        duration=duration,
        metadata=metadata,
        error=error
    )


def log_vector_db_operation(
    operation: str,
    provider: str,
    collection: str,
    count: int,
    duration: Optional[float] = None,
    error: Optional[str] = None
) -> None:
    """Log vector database operation."""
    metadata = {
        "provider": provider,
        "collection": collection,
        "count": count,
    }
    
    status = "error" if error else "success"
    
    log_operation(
        operation=f"vector_db_{operation}",
        status=status,
        duration=duration,
        metadata=metadata,
        error=error
    )


class RAGLogger:
    """Context manager for logging RAG operations."""
    
    def __init__(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        logger.debug(f"Starting operation: {self.operation}", extra=self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            log_operation(
                operation=self.operation,
                status="success",
                duration=duration,
                metadata=self.metadata
            )
        else:
            log_operation(
                operation=self.operation,
                status="error",
                duration=duration,
                metadata=self.metadata,
                error=str(exc_val)
            )
    
    def update_metadata(self, **kwargs):
        """Update operation metadata."""
        self.metadata.update(kwargs)