"""Unit tests for RAG Pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from src.core.rag_pipeline import RAGPipeline
from src.models.rag import RAGQuery, RAGResponse, RAGConfig, RAGContext, RetrievedChunk
from src.models.documents import Document, DocumentChunk
from src.providers.base_provider import LLMMessage


class TestRAGPipeline:
    """Test cases for RAGPipeline."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for RAGPipeline."""
        # Mock EmbeddingService
        mock_embedding_service = MagicMock()
        mock_embedding_service.initialize = AsyncMock()
        mock_embedding_service.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_embedding_service.embed_batch = AsyncMock(return_value=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        mock_embedding_service.health_check = AsyncMock(return_value=True)
        mock_embedding_service.get_model_info = MagicMock(return_value={
            "model_name": "test-embedding-model",
            "dimension": 384
        })
        
        # Mock VectorStore
        mock_vector_store = MagicMock()
        mock_vector_store.initialize = AsyncMock()
        mock_vector_store.add_chunks = AsyncMock()
        mock_vector_store.search = AsyncMock(return_value=[
            {
                "chunk_id": "chunk1",
                "document_id": "doc1",
                "content": "Test chunk content",
                "similarity_score": 0.9,
                "metadata": {"source": "test.pdf"}
            }
        ])
        mock_vector_store.delete_document = AsyncMock()
        mock_vector_store.health_check = AsyncMock(return_value=True)
        mock_vector_store.get_stats = AsyncMock(return_value={
            "total_documents": 10,
            "total_chunks": 100
        })
        
        # Mock DocumentProcessor
        mock_document_processor = MagicMock()
        mock_document_processor.process_file = AsyncMock(return_value=Document(
            document_id="doc1",
            filename="test.pdf",
            content="Test document content",
            metadata={"source": "test.pdf"}
        ))
        mock_document_processor.process_text = AsyncMock(return_value=Document(
            document_id="doc2",
            filename="test.txt",
            content="Test text content",
            metadata={"source": "text"}
        ))
        mock_document_processor.chunk_document = AsyncMock(return_value=[
            DocumentChunk(
                chunk_id="chunk1",
                document_id="doc1",
                content="Test chunk content",
                chunk_index=0,
                metadata={"source": "test.pdf"}
            )
        ])
        
        # Mock Retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=[
            RetrievedChunk(
                chunk_id="chunk1",
                document_id="doc1",
                content="Test chunk content for retrieval",
                similarity_score=0.9,
                metadata={"source": "test.pdf"},
                rank=1
            )
        ])
        mock_retriever.health_check = AsyncMock(return_value=True)
        mock_retriever._build_filters = MagicMock(return_value={})
        
        # Mock LLM Provider
        mock_llm_provider = MagicMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Generated answer based on context"
        mock_llm_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}
        mock_llm_provider.generate = AsyncMock(return_value=mock_llm_response)
        mock_llm_provider.health_check = AsyncMock(return_value=True)
        mock_llm_provider.get_model_info = MagicMock(return_value={
            "model_name": "test-llm-model",
            "provider": "test"
        })
        
        return {
            "embedding_service": mock_embedding_service,
            "vector_store": mock_vector_store,
            "document_processor": mock_document_processor,
            "retriever": mock_retriever,
            "llm_provider": mock_llm_provider
        }

    @pytest.fixture
    @patch('src.core.rag_pipeline.create_llm_provider')
    @patch('src.core.rag_pipeline.Retriever')
    @patch('src.core.rag_pipeline.DocumentProcessor')
    @patch('src.core.rag_pipeline.VectorStore')
    @patch('src.core.rag_pipeline.EmbeddingService')
    def rag_pipeline(self, mock_embedding_cls, mock_vector_cls, mock_processor_cls, 
                     mock_retriever_cls, mock_llm_factory, mock_components):
        """Create RAGPipeline instance with mocked components."""
        # Setup mocks
        mock_embedding_cls.return_value = mock_components["embedding_service"]
        mock_vector_cls.return_value = mock_components["vector_store"]
        mock_processor_cls.return_value = mock_components["document_processor"]
        mock_retriever_cls.return_value = mock_components["retriever"]
        mock_llm_factory.return_value = mock_components["llm_provider"]
        
        return RAGPipeline()

    @pytest.fixture
    def sample_rag_query(self):
        """Create a sample RAG query for testing."""
        return RAGQuery(
            query="What is machine learning?",
            config=RAGConfig(top_k=5, similarity_threshold=0.7),
            filters={"domain": "AI"}
        )

    @pytest.mark.unit
    def test_rag_pipeline_initialization_components(self, rag_pipeline):
        """Test RAGPipeline component initialization."""
        # Test that components are created but not initialized
        assert rag_pipeline.embedding_service is not None
        assert rag_pipeline.vector_store is not None
        assert rag_pipeline.document_processor is not None
        assert rag_pipeline.retriever is None  # Not initialized yet
        assert rag_pipeline.llm_provider is None  # Not initialized yet
        assert rag_pipeline.initialized is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_success(self, rag_pipeline, mock_components):
        """Test successful pipeline initialization."""
        await rag_pipeline.initialize()
        
        # Verify components were initialized
        mock_components["embedding_service"].initialize.assert_called_once()
        mock_components["vector_store"].initialize.assert_called_once()
        
        # Verify retriever and LLM provider were set
        assert rag_pipeline.retriever is not None
        assert rag_pipeline.llm_provider is not None
        assert rag_pipeline.initialized is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_embedding_service_failure(self, rag_pipeline, mock_components):
        """Test initialization failure in embedding service."""
        mock_components["embedding_service"].initialize.side_effect = Exception("Embedding init failed")
        
        with pytest.raises(Exception, match="Embedding init failed"):
            await rag_pipeline.initialize()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_success(self, rag_pipeline, sample_rag_query, mock_components):
        """Test successful query processing."""
        await rag_pipeline.initialize()
        
        response = await rag_pipeline.query(sample_rag_query)
        
        # Verify response structure
        assert isinstance(response, RAGResponse)
        assert response.status == "success"
        assert response.message == "Query processed successfully"
        assert response.query == sample_rag_query.query
        assert response.answer == "Generated answer based on context"
        assert isinstance(response.context, RAGContext)
        assert len(response.sources) == 1
        assert response.generation_time > 0
        assert response.total_time > 0
        assert response.usage is not None
        
        # Verify retriever was called
        mock_components["retriever"].retrieve.assert_called_once_with(sample_rag_query)
        
        # Verify LLM was called
        mock_components["llm_provider"].generate.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_auto_initialize(self, rag_pipeline, sample_rag_query, mock_components):
        """Test that query auto-initializes if not initialized."""
        # Don't manually initialize
        assert rag_pipeline.initialized is False
        
        response = await rag_pipeline.query(sample_rag_query)
        
        # Should auto-initialize
        assert rag_pipeline.initialized is True
        assert isinstance(response, RAGResponse)
        
        # Verify initialization was called
        mock_components["embedding_service"].initialize.assert_called_once()
        mock_components["vector_store"].initialize.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_retrieval_failure(self, rag_pipeline, sample_rag_query, mock_components):
        """Test query processing with retrieval failure."""
        await rag_pipeline.initialize()
        mock_components["retriever"].retrieve.side_effect = Exception("Retrieval failed")
        
        with pytest.raises(Exception, match="Retrieval failed"):
            await rag_pipeline.query(sample_rag_query)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_llm_generation_failure(self, rag_pipeline, sample_rag_query, mock_components):
        """Test query processing with LLM generation failure."""
        await rag_pipeline.initialize()
        mock_components["llm_provider"].generate.side_effect = Exception("LLM generation failed")
        
        with pytest.raises(Exception, match="LLM generation failed"):
            await rag_pipeline.query(sample_rag_query)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ingest_document_file_path(self, rag_pipeline, mock_components):
        """Test document ingestion from file path."""
        file_path = "/path/to/test.pdf"
        metadata = {"source": "test", "type": "pdf"}
        
        document_id = await rag_pipeline.ingest_document(file_path=file_path, metadata=metadata)
        
        assert document_id == "doc1"
        
        # Verify processing chain
        mock_components["document_processor"].process_file.assert_called_once_with(file_path, metadata)
        mock_components["document_processor"].chunk_document.assert_called_once()
        mock_components["embedding_service"].embed_batch.assert_called_once()
        mock_components["vector_store"].add_chunks.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ingest_document_text_content(self, rag_pipeline, mock_components):
        """Test document ingestion from text content."""
        text_content = "This is test text content"
        filename = "test.txt"
        metadata = {"source": "text"}
        
        document_id = await rag_pipeline.ingest_document(
            text_content=text_content, 
            filename=filename, 
            metadata=metadata
        )
        
        assert document_id == "doc2"
        
        # Verify processing chain
        mock_components["document_processor"].process_text.assert_called_once_with(
            text_content, filename, metadata
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ingest_document_no_content(self, rag_pipeline):
        """Test document ingestion with no file path or text content."""
        with pytest.raises(ValueError, match="Either file_path or text_content must be provided"):
            await rag_pipeline.ingest_document()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ingest_document_processing_failure(self, rag_pipeline, mock_components):
        """Test document ingestion with processing failure."""
        mock_components["document_processor"].process_file.side_effect = Exception("Processing failed")
        
        with pytest.raises(Exception, match="Processing failed"):
            await rag_pipeline.ingest_document(file_path="/path/to/test.pdf")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_document_success(self, rag_pipeline, mock_components):
        """Test successful document deletion."""
        document_id = "doc1"
        
        await rag_pipeline.delete_document(document_id)
        
        mock_components["vector_store"].delete_document.assert_called_once_with(document_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_document_failure(self, rag_pipeline, mock_components):
        """Test document deletion failure."""
        document_id = "doc1"
        mock_components["vector_store"].delete_document.side_effect = Exception("Delete failed")
        
        with pytest.raises(Exception, match="Delete failed"):
            await rag_pipeline.delete_document(document_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_documents_success(self, rag_pipeline, mock_components):
        """Test successful document search."""
        query = "machine learning"
        top_k = 10
        filters = {"domain": "AI"}
        
        results = await rag_pipeline.search_documents(query, top_k, filters)
        
        assert len(results) == 1
        assert results[0]["chunk_id"] == "chunk1"
        
        # Verify calls
        mock_components["embedding_service"].embed_single.assert_called_once_with(query)
        mock_components["retriever"]._build_filters.assert_called_once_with(filters)
        mock_components["vector_store"].search.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_documents_failure(self, rag_pipeline, mock_components):
        """Test document search failure."""
        mock_components["embedding_service"].embed_single.side_effect = Exception("Embedding failed")
        
        with pytest.raises(Exception, match="Embedding failed"):
            await rag_pipeline.search_documents("test query")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_stats_success(self, rag_pipeline, mock_components):
        """Test getting pipeline statistics."""
        stats = await rag_pipeline.get_stats()
        
        assert "vector_store" in stats
        assert "embedding_service" in stats
        assert "llm_provider" in stats
        assert "initialized" in stats
        assert stats["initialized"] is True
        
        # Verify calls
        mock_components["vector_store"].get_stats.assert_called_once()
        mock_components["embedding_service"].get_model_info.assert_called_once()
        mock_components["llm_provider"].get_model_info.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_stats_failure(self, rag_pipeline, mock_components):
        """Test getting statistics with component failure."""
        mock_components["vector_store"].get_stats.side_effect = Exception("Stats failed")
        
        stats = await rag_pipeline.get_stats()
        
        assert "error" in stats
        assert "Stats failed" in stats["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, rag_pipeline, mock_components):
        """Test health check with all components healthy."""
        health = await rag_pipeline.health_check()
        
        assert health["overall"] is True
        assert health["embedding_service"] is True
        assert health["vector_store"] is True
        assert health["retriever"] is True
        assert health["llm_provider"] is True
        
        # Verify all health checks were called
        mock_components["embedding_service"].health_check.assert_called_once()
        mock_components["vector_store"].health_check.assert_called_once()
        mock_components["retriever"].health_check.assert_called_once()
        mock_components["llm_provider"].health_check.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_embedding_unhealthy(self, rag_pipeline, mock_components):
        """Test health check with embedding service unhealthy."""
        mock_components["embedding_service"].health_check.return_value = False
        
        health = await rag_pipeline.health_check()
        
        assert health["overall"] is False
        assert health["embedding_service"] is False
        assert health["vector_store"] is True
        assert health["retriever"] is True
        assert health["llm_provider"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, rag_pipeline, mock_components):
        """Test health check with exception during check."""
        mock_components["embedding_service"].health_check.side_effect = Exception("Health check failed")
        
        health = await rag_pipeline.health_check()
        
        # Should handle exception gracefully and return False for overall health
        assert health["overall"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_build_context_success(self, rag_pipeline):
        """Test context building from retrieved chunks."""
        await rag_pipeline.initialize()
        
        query = "What is AI?"
        chunks = [
            RetrievedChunk(
                chunk_id="chunk1",
                document_id="doc1",
                content="AI is artificial intelligence",
                similarity_score=0.9,
                metadata={"source": "doc1.pdf"},
                rank=1
            ),
            RetrievedChunk(
                chunk_id="chunk2",
                document_id="doc2",
                content="Machine learning is a subset of AI",
                similarity_score=0.8,
                metadata={"source": "doc2.pdf"},
                rank=2
            )
        ]
        
        context = await rag_pipeline._build_context(query, chunks, 0.1)
        
        assert isinstance(context, RAGContext)
        assert context.query == query
        assert len(context.retrieved_chunks) == 2
        assert context.total_chunks == 2
        assert "[1] AI is artificial intelligence" in context.context_text
        assert "[2] Machine learning is a subset of AI" in context.context_text
        assert context.retrieval_time == 0.1
        assert context.total_tokens > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_build_context_truncation(self, rag_pipeline):
        """Test context truncation when content is too long."""
        await rag_pipeline.initialize()
        
        # Create a very long chunk
        long_content = "This is a very long content. " * 1000
        chunks = [
            RetrievedChunk(
                chunk_id="chunk1",
                document_id="doc1",
                content=long_content,
                similarity_score=0.9,
                metadata={"source": "doc1.pdf"},
                rank=1
            )
        ]
        
        with patch('src.core.rag_pipeline.settings') as mock_settings:
            mock_settings.max_context_length = 100
            
            context = await rag_pipeline._build_context("query", chunks, 0.1)
            
            assert len(context.context_text) <= 103  # 100 + "..."
            assert context.context_text.endswith("...")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_answer_success(self, rag_pipeline, mock_components):
        """Test answer generation from context."""
        await rag_pipeline.initialize()
        
        query = "What is AI?"
        context = RAGContext(
            query=query,
            retrieved_chunks=[],
            total_chunks=0,
            context_text="AI is artificial intelligence",
            retrieval_time=0.1,
            total_tokens=10
        )
        
        answer = await rag_pipeline._generate_answer(query, context)
        
        assert answer.content == "Generated answer based on context"
        assert answer.usage is not None
        
        # Verify LLM was called with proper messages
        mock_components["llm_provider"].generate.assert_called_once()
        call_args = mock_components["llm_provider"].generate.call_args[0]
        messages = call_args[0]
        
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert query in messages[1].content

    @pytest.mark.unit
    def test_build_sources_success(self, rag_pipeline):
        """Test building sources information from chunks."""
        chunks = [
            RetrievedChunk(
                chunk_id="chunk1",
                document_id="doc1",
                content="This is a test chunk content that is longer than 200 characters to test the preview truncation functionality of the build sources method.",
                similarity_score=0.9,
                metadata={"source": "doc1.pdf", "page": 1},
                rank=1
            ),
            RetrievedChunk(
                chunk_id="chunk2",
                document_id="doc2",
                content="Short content",
                similarity_score=0.8,
                metadata={"source": "doc2.pdf"},
                rank=2
            )
        ]
        
        sources = rag_pipeline._build_sources(chunks)
        
        assert len(sources) == 2
        
        # Check first source (long content)
        source1 = sources[0]
        assert source1["chunk_id"] == "chunk1"
        assert source1["document_id"] == "doc1"
        assert source1["similarity_score"] == 0.9
        assert source1["rank"] == 1
        assert source1["content_preview"].endswith("...")
        assert len(source1["content_preview"]) == 203  # 200 + "..."
        assert source1["metadata"]["source"] == "doc1.pdf"
        
        # Check second source (short content)
        source2 = sources[1]
        assert source2["chunk_id"] == "chunk2"
        assert source2["content_preview"] == "Short content"
        assert not source2["content_preview"].endswith("...")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_different_configs(self, rag_pipeline, mock_components):
        """Test query processing with different RAG configurations."""
        await rag_pipeline.initialize()
        
        # Test with different config
        query = RAGQuery(
            query="What is deep learning?",
            config=RAGConfig(
                top_k=3,
                similarity_threshold=0.8,
                max_context_length=2000
            )
        )
        
        response = await rag_pipeline.query(query)
        
        assert isinstance(response, RAGResponse)
        assert response.query == query.query
        
        # Verify retriever was called with the query (which contains config)
        mock_components["retriever"].retrieve.assert_called_once_with(query)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ingest_document_empty_chunks(self, rag_pipeline, mock_components):
        """Test document ingestion when no chunks are generated."""
        # Mock empty chunks
        mock_components["document_processor"].chunk_document.return_value = []
        
        document_id = await rag_pipeline.ingest_document(file_path="/path/to/test.pdf")
        
        assert document_id == "doc1"
        
        # Should not call embedding or vector store operations
        mock_components["embedding_service"].embed_batch.assert_not_called()
        mock_components["vector_store"].add_chunks.assert_not_called()
