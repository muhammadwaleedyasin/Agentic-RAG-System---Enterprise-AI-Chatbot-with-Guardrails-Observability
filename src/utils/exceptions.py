"""Custom exceptions for the RAG system."""


class RAGException(Exception):
    """Base exception for RAG system."""
    pass


class ConfigurationError(RAGException):
    """Raised when there's a configuration issue."""
    pass


class DocumentProcessingError(RAGException):
    """Raised when document processing fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class VectorDBError(RAGException):
    """Raised when vector database operations fail."""
    pass


class RetrievalError(RAGException):
    """Raised when retrieval operations fail."""
    pass


class LLMError(RAGException):
    """Raised when LLM operations fail."""
    pass


class GuardrailsError(RAGException):
    """Raised when guardrails validation fails."""
    pass


class MemoryError(RAGException):
    """Raised when memory operations fail."""
    pass


class IngestionError(RAGException):
    """Raised when ingestion operations fail."""
    pass


class ChunkingError(RAGException):
    """Raised when chunking operations fail."""
    pass


class ValidationError(RAGException):
    """Raised when validation fails."""
    pass


class AuthenticationError(RAGException):
    """Raised when authentication fails."""
    pass


class RateLimitError(RAGException):
    """Raised when rate limits are exceeded."""
    pass


class TimeoutError(RAGException):
    """Raised when operations timeout."""
    pass