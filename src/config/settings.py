"""
Application configuration settings with dual LLM provider support.
"""
import os
from enum import Enum
from typing import Optional, List, Dict
from pydantic_settings import BaseSettings
from pydantic import validator


class LLMProvider(str, Enum):
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    # Application
    app_name: str = "Enterprise RAG Chatbot"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: LogLevel = LogLevel.INFO
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["*"]
    
    # LLM Provider Configuration
    llm_provider: LLMProvider = LLMProvider.VLLM
    use_unified_provider: bool = True
    
    # Provider Failover Configuration
    failover_strategy: str = "auto"  # none, manual, auto, round_robin, priority
    load_balancing_strategy: str = "health_weighted"  # round_robin, random, response_time, success_rate, health_weighted
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    provider_health_check_interval: int = 300
    
    # vLLM Configuration
    vllm_base_url: str = "http://localhost:8001"
    vllm_model_name: str = "microsoft/DialoGPT-medium"
    vllm_api_key: Optional[str] = None  # Optional for local vLLM
    vllm_max_tokens: int = 2048
    vllm_temperature: float = 0.7
    vllm_timeout: int = 60
    vllm_max_retries: int = 3
    vllm_retry_delay: float = 1.0
    vllm_connect_timeout: int = 10
    vllm_read_timeout: int = 60
    
    # vLLM Advanced Parameters
    vllm_top_p: float = 1.0
    vllm_presence_penalty: float = 0.0
    vllm_frequency_penalty: float = 0.0
    vllm_repetition_penalty: float = 1.0
    vllm_stop_tokens: List[str] = []
    vllm_best_of: int = 1
    vllm_use_beam_search: bool = False
    vllm_top_k: int = -1
    vllm_length_penalty: float = 1.0
    vllm_max_model_length: int = 4096
    
    # OpenRouter Configuration
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model_name: str = "anthropic/claude-3-sonnet"
    openrouter_max_tokens: int = 4096
    openrouter_temperature: float = 0.7
    openrouter_timeout: int = 120
    openrouter_max_retries: int = 3
    openrouter_retry_delay: float = 1.0
    openrouter_http_referer: str = "https://github.com/enterprise-rag"
    openrouter_site_url: Optional[str] = None
    
    # OpenRouter Advanced Parameters
    openrouter_top_p: float = 1.0
    openrouter_presence_penalty: float = 0.0
    openrouter_frequency_penalty: float = 0.0
    openrouter_repetition_penalty: float = 1.0
    openrouter_top_k: int = 0
    openrouter_min_p: float = 0.0
    openrouter_seed: Optional[int] = None
    openrouter_logit_bias: Dict[str, float] = {}
    openrouter_response_format: Optional[str] = None
    openrouter_fallback_models: List[str] = [
        "anthropic/claude-3-haiku",
        "openai/gpt-3.5-turbo",
        "meta-llama/llama-2-70b-chat"
    ]
    openrouter_max_cost_per_request: Optional[float] = None
    openrouter_cost_threshold_warning: Optional[float] = None
    openrouter_models_cache_ttl: int = 3600
    
    # Vector Database Configuration
    vector_db_type: str = "chroma"  # chroma, pinecone, weaviate
    vector_db_path: str = "./data/vectordb"
    vector_dimension: int = 384
    
    # ChromaDB Configuration
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "rag_documents"
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # cpu, cuda
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # RAG Configuration
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    
    # File Processing
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = [".txt", ".pdf", ".docx", ".md"]
    upload_path: str = os.getenv("UPLOAD_PATH", "./data/uploads")
    cache_path: str = os.getenv("CACHE_PATH", "./data/cache")
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Caching
    cache_ttl: int = 3600  # seconds
    redis_url: Optional[str] = None
    
    # Monitoring
    enable_metrics: bool = True
    metrics_endpoint: str = "/metrics"
    
    # Zep Memory Configuration
    zep_api_url: str = "http://localhost:8000"
    zep_api_key: Optional[str] = None
    zep_auto_summarize: bool = True
    zep_memory_window_size: int = 12
    zep_max_messages: int = 100
    zep_max_tokens: int = 8000
    zep_enable_embeddings: bool = True
    zep_timeout: int = 30
    zep_retry_attempts: int = 3
    zep_retry_delay: float = 1.0
    
    # Memory Scoping Configuration
    memory_scope_by_role: bool = True
    memory_admin_access_all: bool = True
    memory_user_isolation: bool = True
    memory_session_ttl: int = 86400  # 24 hours in seconds
    memory_auto_cleanup_enabled: bool = True
    memory_cleanup_interval: int = 3600  # 1 hour in seconds
    
    @validator("openrouter_api_key")
    def validate_openrouter_key(cls, v, values):
        # Only require API key if OpenRouter is the primary provider or unified provider is enabled
        if values.get("llm_provider") == LLMProvider.OPENROUTER and not v:
            raise ValueError("OpenRouter API key is required when using OpenRouter provider")
        if values.get("use_unified_provider", True) and not v:
            # Just warn if unified provider is used without OpenRouter key
            import warnings
            warnings.warn("OpenRouter API key not provided - OpenRouter provider will not be available")
        return v
    
    @validator("openrouter_fallback_models", pre=True)
    def parse_fallback_models(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(",") if model.strip()]
        return v
    
    @validator("vllm_stop_tokens", pre=True)
    def parse_stop_tokens(cls, v):
        if isinstance(v, str):
            return [token.strip() for token in v.split(",") if token.strip()]
        return v
    
    @validator("openrouter_logit_bias", pre=True)
    def parse_logit_bias(cls, v):
        if isinstance(v, str):
            try:
                import json
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v or {}
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()