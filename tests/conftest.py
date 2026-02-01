"""Pytest configuration and shared fixtures for Enterprise RAG Chatbot tests."""

import asyncio
import os
import sys
import types
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

# Stub heavy external libs before any project imports
# Stub chromadb with complete interface
if 'chromadb' not in sys.modules:
    chromadb_stub = types.SimpleNamespace()
    chromadb_stub.Client = Mock
    chromadb_stub.config = types.SimpleNamespace(Settings=types.SimpleNamespace)
    chromadb_stub.utils = types.SimpleNamespace(
        embedding_functions=types.SimpleNamespace(
            DefaultEmbeddingFunction=Mock,
            OpenAIEmbeddingFunction=Mock
        )
    )
    sys.modules['chromadb'] = chromadb_stub

# Stub faiss with common classes
if 'faiss' not in sys.modules:
    faiss_stub = types.SimpleNamespace()
    faiss_stub.IndexFlatIP = Mock
    faiss_stub.IndexIVFFlat = Mock
    faiss_stub.IndexIVFPQ = Mock
    faiss_stub.IndexFlatL2 = Mock
    faiss_stub.METRIC_INNER_PRODUCT = 1
    faiss_stub.METRIC_L2 = 2
    sys.modules['faiss'] = faiss_stub

# Stub pymilvus with required attributes
if 'pymilvus' not in sys.modules:
    sys.modules['pymilvus'] = types.SimpleNamespace(
        Collection=Mock,
        connections=Mock,
        utility=Mock,
        DataType=Mock,
        FieldSchema=Mock,
        CollectionSchema=Mock,
        MilvusException=Exception
    )

# Stub weaviate with complete client interface
if 'weaviate' not in sys.modules:
    weaviate_stub = types.SimpleNamespace()
    weaviate_stub.Client = Mock
    weaviate_stub.AuthApiKey = Mock
    weaviate_stub.AuthClientPassword = Mock
    weaviate_stub.Config = Mock
    sys.modules['weaviate'] = weaviate_stub

# Stub pinecone with complete interface
if 'pinecone' not in sys.modules:
    pinecone_stub = types.SimpleNamespace()
    pinecone_stub.init = Mock
    pinecone_stub.list_indexes = Mock
    pinecone_stub.Index = Mock
    pinecone_stub.Pinecone = Mock
    pinecone_stub.ServerlessSpec = Mock
    sys.modules['pinecone'] = pinecone_stub

# Stub external deps used by caching_layer and other components
external_deps = {
    "memcache": types.SimpleNamespace(Cache=Mock),
    "diskcache": types.SimpleNamespace(Cache=Mock, Index=Mock),
    "aiofiles": types.SimpleNamespace(open=AsyncMock, tempfile=Mock),
    "redis": types.SimpleNamespace(Redis=Mock, from_url=Mock),
    "qdrant_client": types.SimpleNamespace(QdrantClient=Mock, models=Mock),
    "sentence_transformers": types.SimpleNamespace(SentenceTransformer=Mock),
    "transformers": types.SimpleNamespace(AutoTokenizer=Mock, AutoModel=Mock)
}

for name, stub in external_deps.items():
    if name not in sys.modules:
        sys.modules[name] = stub

from src.app.main import app
from src.config.settings import settings


# Remove custom event_loop fixture to avoid conflicts with pytest-asyncio
# Use pytest-asyncio's auto mode instead


@pytest.fixture(scope="session")
def test_config():
    """Load test configuration."""
    os.environ.setdefault("ENVIRONMENT", "test")
    return settings


@pytest.fixture
def client(test_config) -> TestClient:
    """Create a test client for the FastAPI application."""
    app.dependency_overrides = {}
    return TestClient(app)


@pytest.fixture
async def async_client(test_config) -> AsyncGenerator:
    """Create an async test client for the FastAPI application."""
    from httpx import AsyncClient
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline for testing."""
    mock_pipeline = Mock()
    mock_pipeline.query = AsyncMock()
    mock_pipeline.health_check = AsyncMock(return_value=True)
    return mock_pipeline


@pytest.fixture 
def mock_access_controller():
    """Mock access controller for testing."""
    mock_controller = Mock()
    mock_controller.authenticate_user = AsyncMock()
    mock_controller.check_permission = AsyncMock(return_value=True)
    mock_controller.create_session = AsyncMock(return_value={"session_id": "test-session"})
    return mock_controller


@pytest.fixture
def safe_test_client(mock_rag_pipeline, mock_access_controller):
    """Create TestClient with dependency overrides applied before startup."""
    from src.app.deps import get_rag_pipeline, get_access_controller
    
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag_pipeline
    app.dependency_overrides[get_access_controller] = lambda: mock_access_controller
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def test_db_engine():
    """Create a test database engine."""
    test_db_url = "sqlite:///./test.db"
    engine = create_engine(test_db_url, echo=False)
    yield engine
    # Cleanup
    if os.path.exists("./test.db"):
        os.remove("./test.db")


@pytest.fixture(scope="session")
def test_async_db_engine():
    """Create a test async database engine."""
    test_db_url = "sqlite+aiosqlite:///./test_async.db"
    engine = create_async_engine(test_db_url, echo=False)
    yield engine
    # Cleanup
    if os.path.exists("./test_async.db"):
        os.remove("./test_async.db")


@pytest.fixture
def db_session(test_db_engine):
    """Create a database session for testing."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_db_engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest_asyncio.fixture
async def async_db_session(test_async_db_engine):
    """Create an async database session for testing."""
    async_session = async_sessionmaker(
        test_async_db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    mock_client.embeddings.create = AsyncMock()
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock()
    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = MagicMock()
    mock_store.add_documents = AsyncMock()
    mock_store.similarity_search = AsyncMock()
    mock_store.similarity_search_with_score = AsyncMock()
    return mock_store


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock()
    mock_redis.set = AsyncMock()
    mock_redis.delete = AsyncMock()
    mock_redis.exists = AsyncMock()
    return mock_redis


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return {
        "id": "test-doc-1",
        "title": "Test Document",
        "content": "This is a test document with some content for testing purposes.",
        "metadata": {
            "author": "Test Author",
            "created_at": "2024-01-01T00:00:00Z",
            "file_type": "txt",
            "file_size": 1024
        }
    }


@pytest.fixture
def sample_chat_message():
    """Sample chat message for testing."""
    return {
        "message": "What is the main topic discussed in the documents?",
        "conversation_id": "test-conv-1",
        "user_id": "test-user-1"
    }


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "id": "chunk-1",
            "content": "This is the first chunk of the document.",
            "metadata": {"chunk_index": 0, "document_id": "test-doc-1"}
        },
        {
            "id": "chunk-2", 
            "content": "This is the second chunk of the document.",
            "metadata": {"chunk_index": 1, "document_id": "test-doc-1"}
        }
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    import numpy as np
    return np.random.rand(2, 1536).tolist()  # 2 embeddings of dimension 1536


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[[0.1] * 1536, [0.2] * 1536])
    return mock_model


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup test files after each test."""
    yield
    # Cleanup any test files created during tests
    test_files = ["test.db", "test_async.db", "test_documents.json"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    # Cleanup cache directories
    import shutil
    cache_dirs = ["./cache", "test_cache", ".pytest_cache"]
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def api_headers():
    """Standard API headers for testing."""
    return {
        "Content-Type": "application/json",
        "X-API-Key": "test-api-key"
    }


@pytest.fixture
def mock_jwt_token():
    """Mock JWT token for testing."""
    return "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.token"


# Pytest configuration
pytest_plugins = ["pytest_asyncio"]

# Mock Redis client  
@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = Mock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.flushall.return_value = True
    redis_mock.ping.return_value = True
    return redis_mock

# Mock FAISS operations
@pytest.fixture
def mock_faiss():
    """Mock FAISS operations."""
    with patch('faiss.IndexFlatIP', create=True) as mock_flat, \
         patch('faiss.IndexIVFFlat', create=True) as mock_ivf, \
         patch('faiss.IndexIVFPQ', create=True) as mock_ivfpq:
        
        # Configure mock indexes
        for mock_index in [mock_flat, mock_ivf, mock_ivfpq]:
            instance = Mock()
            instance.ntotal = 100
            instance.d = 384
            instance.is_trained = True
            instance.search.return_value = ([0.95, 0.85], [[0, 1]])
            mock_index.return_value = instance
            
        yield {
            'flat': mock_flat,
            'ivf': mock_ivf, 
            'ivfpq': mock_ivfpq
        }
