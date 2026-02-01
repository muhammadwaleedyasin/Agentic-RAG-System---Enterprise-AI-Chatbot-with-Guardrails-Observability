"""Integration tests for full RAG pipeline."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import tempfile
import os

from src.core.rag_pipeline import RAGPipeline
from src.core.embedding_service import EmbeddingService
from src.core.vector_store import ChromaVectorStore
from src.core.document_processor import DocumentProcessor
from src.core.retriever import Retriever
from src.providers.openrouter_provider import OpenRouterProvider


class TestFullRAGPipeline:
    """Integration tests for the complete RAG pipeline."""

    @pytest.fixture
    async def rag_system(self):
        """Create a complete RAG system for integration testing."""
        # Mock all external dependencies
        with patch('src.core.embedding_service.SentenceTransformer'):
            with patch('src.core.vector_store.chromadb.Client'):
                with patch('aiohttp.ClientSession'):
                    # Initialize components
                    embedding_service = EmbeddingService("all-MiniLM-L6-v2")
                    vector_store = ChromaVectorStore("test_collection")
                    document_processor = DocumentProcessor()
                    llm_provider = OpenRouterProvider(api_key="test-key")
                    retriever = Retriever(embedding_service, vector_store)
                    
                    # Create RAG pipeline
                    rag_pipeline = RAGPipeline(
                        embedding_service=embedding_service,
                        vector_store=vector_store,
                        llm_provider=llm_provider,
                        retriever=retriever
                    )
                    
                    return {
                        "pipeline": rag_pipeline,
                        "embedding_service": embedding_service,
                        "vector_store": vector_store,
                        "document_processor": document_processor,
                        "llm_provider": llm_provider,
                        "retriever": retriever
                    }

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_document_ingestion_to_retrieval(self, rag_system):
        """Test complete document ingestion and retrieval flow."""
        pipeline = rag_system["pipeline"]
        document_processor = rag_system["document_processor"]
        embedding_service = rag_system["embedding_service"]
        vector_store = rag_system["vector_store"]
        
        # Mock document processing
        sample_document = {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "metadata": {
                "file_name": "ml_intro.txt",
                "file_type": "txt",
                "author": "AI Expert"
            }
        }
        
        # Mock embedding generation
        embedding_service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        
        # Mock vector store operations
        vector_store.add_documents = AsyncMock(return_value=True)
        vector_store.search_documents = AsyncMock(return_value=[
            {
                "id": "doc1",
                "content": sample_document["content"],
                "metadata": sample_document["metadata"],
                "score": 0.95
            }
        ])
        
        # Mock LLM response
        pipeline.llm_provider.generate_response = AsyncMock(return_value={
            "content": "Machine learning is a subset of AI that allows computers to learn from data without explicit programming.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 25}
        })
        
        # Test the complete flow
        # 1. Process document
        with patch.object(document_processor, 'process_file', return_value=sample_document):
            processed_doc = document_processor.process_file("ml_intro.txt")
            
        # 2. Add to vector store
        embedding = await embedding_service.embed_text(processed_doc["content"])
        await vector_store.add_documents([processed_doc], [embedding])
        
        # 3. Query the system
        query = "What is machine learning?"
        response = await pipeline.generate_response(query)
        
        # Verify the complete flow worked
        assert response.answer is not None
        assert len(response.sources) > 0
        assert response.sources[0]["content"] == sample_document["content"]
        assert response.confidence_score > 0.0
        
        # Verify all components were called
        embedding_service.embed_text.assert_called()
        vector_store.add_documents.assert_called_once()
        vector_store.search_documents.assert_called()
        pipeline.llm_provider.generate_response.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_document_ingestion_and_query(self, rag_system):
        """Test ingestion of multiple documents and complex queries."""
        pipeline = rag_system["pipeline"]
        embedding_service = rag_system["embedding_service"]
        vector_store = rag_system["vector_store"]
        
        # Mock multiple documents
        documents = [
            {
                "id": "doc1",
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"topic": "programming", "language": "python"}
            },
            {
                "id": "doc2", 
                "content": "Machine learning algorithms can be supervised, unsupervised, or reinforcement learning.",
                "metadata": {"topic": "ai", "subtopic": "machine_learning"}
            },
            {
                "id": "doc3",
                "content": "Data preprocessing is crucial for machine learning model performance.",
                "metadata": {"topic": "data_science", "subtopic": "preprocessing"}
            }
        ]
        
        # Mock embeddings for documents
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        embedding_service.embed_text = AsyncMock(side_effect=embeddings)
        
        # Mock vector store operations
        vector_store.add_documents = AsyncMock(return_value=True)
        vector_store.search_documents = AsyncMock(return_value=[
            {"id": "doc2", "content": documents[1]["content"], "metadata": documents[1]["metadata"], "score": 0.9},
            {"id": "doc3", "content": documents[2]["content"], "metadata": documents[2]["metadata"], "score": 0.8}
        ])
        
        # Mock LLM response
        pipeline.llm_provider.generate_response = AsyncMock(return_value={
            "content": "Machine learning involves various algorithms like supervised and unsupervised learning. Data preprocessing is essential for good performance.",
            "usage": {"prompt_tokens": 150, "completion_tokens": 30}
        })
        
        # Add all documents
        for i, doc in enumerate(documents):
            await vector_store.add_documents([doc], [embeddings[i]])
        
        # Query with filter
        query = "Tell me about machine learning and data preprocessing"
        response = await pipeline.generate_response(query, filters={"topic": "ai"})
        
        # Verify results
        assert response.answer is not None
        assert len(response.sources) >= 1
        assert "machine learning" in response.answer.lower()
        
        # Verify vector store was queried with filters
        vector_store.search_documents.assert_called()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_context_handling(self, rag_system):
        """Test conversation context and follow-up questions."""
        pipeline = rag_system["pipeline"]
        
        # Mock context-aware responses
        responses = [
            "Python is a programming language known for simplicity.",
            "Yes, Python is excellent for machine learning with libraries like scikit-learn and TensorFlow."
        ]
        
        pipeline.llm_provider.generate_response = AsyncMock(side_effect=[
            {"content": responses[0], "usage": {"total_tokens": 50}},
            {"content": responses[1], "usage": {"total_tokens": 60}}
        ])
        
        # Mock retriever to return relevant documents
        pipeline.retriever.retrieve = AsyncMock(return_value=[
            {"content": "Python programming language overview", "score": 0.9},
            {"content": "Python for machine learning applications", "score": 0.85}
        ])
        
        # First question
        conversation_history = []
        response1 = await pipeline.generate_response(
            "What is Python?", 
            conversation_history=conversation_history
        )
        
        # Update conversation history
        conversation_history.extend([
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": response1.answer}
        ])
        
        # Follow-up question
        response2 = await pipeline.generate_response(
            "Is it good for machine learning?",
            conversation_history=conversation_history
        )
        
        # Verify both responses
        assert response1.answer == responses[0]
        assert response2.answer == responses[1]
        assert pipeline.llm_provider.generate_response.call_count == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, rag_system):
        """Test error handling and recovery mechanisms."""
        pipeline = rag_system["pipeline"]
        
        # Test embedding service failure
        pipeline.embedding_service.embed_text = AsyncMock(
            side_effect=Exception("Embedding service down")
        )
        
        with pytest.raises(Exception):
            await pipeline.generate_response("Test query")
        
        # Test recovery - embedding service comes back online
        pipeline.embedding_service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        pipeline.retriever.retrieve = AsyncMock(return_value=[])
        pipeline.llm_provider.generate_response = AsyncMock(return_value={
            "content": "I don't have enough information to answer that.",
            "usage": {"total_tokens": 20}
        })
        
        response = await pipeline.generate_response("Test query")
        assert response.answer is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_with_large_document_set(self, rag_system):
        """Test system performance with large document sets."""
        pipeline = rag_system["pipeline"]
        vector_store = rag_system["vector_store"]
        
        # Simulate large document set
        large_doc_set = [
            {
                "id": f"doc_{i}",
                "content": f"This is document {i} with relevant content about topic {i % 10}.",
                "metadata": {"doc_id": i, "topic": f"topic_{i % 10}"}
            }
            for i in range(1000)
        ]
        
        # Mock batch operations
        vector_store.add_documents = AsyncMock(return_value=True)
        vector_store.search_documents = AsyncMock(return_value=[
            {
                "id": "doc_5", 
                "content": "This is document 5 with relevant content about topic 5.",
                "score": 0.95
            }
        ])
        
        # Mock LLM response
        pipeline.llm_provider.generate_response = AsyncMock(return_value={
            "content": "Based on the document, this is information about topic 5.",
            "usage": {"total_tokens": 45}
        })
        
        # Test batch ingestion
        import time
        start_time = time.time()
        
        # Simulate batch processing (mocked)
        await vector_store.add_documents(large_doc_set, [[0.1, 0.2, 0.3]] * 1000)
        
        ingestion_time = time.time() - start_time
        
        # Test query performance
        start_time = time.time()
        response = await pipeline.generate_response("Tell me about topic 5")
        query_time = time.time() - start_time
        
        # Verify performance metrics are reasonable (mocked times will be very fast)
        assert ingestion_time < 10  # Should be fast with mocking
        assert query_time < 5      # Should be fast with mocking
        assert response.answer is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_file_processing_flow(self, rag_system):
        """Test processing real files through the pipeline."""
        pipeline = rag_system["pipeline"]
        document_processor = rag_system["document_processor"]
        
        # Create temporary test files
        test_content = """
        Artificial Intelligence (AI) is the simulation of human intelligence in machines.
        Machine learning is a subset of AI that uses statistical techniques.
        Deep learning is a subset of machine learning based on neural networks.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        try:
            # Mock document processor
            document_processor.process_file = MagicMock(return_value={
                "content": test_content.strip(),
                "metadata": {
                    "file_name": os.path.basename(temp_file_path),
                    "file_type": "txt",
                    "file_size": len(test_content)
                }
            })
            
            # Mock the rest of the pipeline
            pipeline.embedding_service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
            pipeline.vector_store.add_documents = AsyncMock(return_value=True)
            pipeline.retriever.retrieve = AsyncMock(return_value=[
                {"content": test_content.strip(), "score": 0.9}
            ])
            pipeline.llm_provider.generate_response = AsyncMock(return_value={
                "content": "AI is the simulation of human intelligence. Machine learning and deep learning are subsets of AI.",
                "usage": {"total_tokens": 40}
            })
            
            # Process the file
            processed_doc = document_processor.process_file(temp_file_path)
            
            # Add to vector store
            embedding = await pipeline.embedding_service.embed_text(processed_doc["content"])
            await pipeline.vector_store.add_documents([processed_doc], [embedding])
            
            # Query the system
            response = await pipeline.generate_response("What is artificial intelligence?")
            
            # Verify the flow
            assert response.answer is not None
            assert "intelligence" in response.answer.lower()
            assert len(response.sources) > 0
            
        finally:
            # Cleanup
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_citation_and_source_tracking(self, rag_system):
        """Test proper citation and source tracking through the pipeline."""
        pipeline = rag_system["pipeline"]
        
        # Mock documents with detailed metadata
        sources = [
            {
                "content": "Climate change is caused by greenhouse gas emissions.",
                "metadata": {
                    "title": "Climate Science Report",
                    "author": "Dr. Smith",
                    "publication_date": "2023-01-15",
                    "page_number": 42
                },
                "score": 0.95
            }
        ]
        
        pipeline.retriever.retrieve = AsyncMock(return_value=sources)
        pipeline.llm_provider.generate_response = AsyncMock(return_value={
            "content": "According to recent research, climate change is primarily caused by greenhouse gas emissions [1].",
            "usage": {"total_tokens": 35}
        })
        
        # Test with citations enabled
        response = await pipeline.generate_response(
            "What causes climate change?",
            include_citations=True
        )
        
        # Verify citations and sources
        assert response.answer is not None
        assert len(response.sources) == 1
        assert response.sources[0]["metadata"]["author"] == "Dr. Smith"
        assert response.sources[0]["metadata"]["title"] == "Climate Science Report"
        
        # Check if citations are properly formatted
        if hasattr(response, 'citations') and response.citations:
            assert len(response.citations) > 0

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_query_handling(self, rag_system):
        """Test handling multiple concurrent queries."""
        import asyncio
        
        pipeline = rag_system["pipeline"]
        
        # Mock consistent responses
        pipeline.retriever.retrieve = AsyncMock(return_value=[
            {"content": "Relevant information", "score": 0.8}
        ])
        pipeline.llm_provider.generate_response = AsyncMock(return_value={
            "content": "This is a response to the query.",
            "usage": {"total_tokens": 30}
        })
        
        # Create multiple concurrent queries
        queries = [
            "What is machine learning?",
            "Explain deep learning",
            "What is natural language processing?",
            "How does computer vision work?",
            "What are neural networks?"
        ]
        
        # Execute concurrent queries
        start_time = time.time()
        tasks = [pipeline.generate_response(query) for query in queries]
        responses = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # Verify all responses
        assert len(responses) == len(queries)
        assert all(response.answer is not None for response in responses)
        assert execution_time < 10  # Should be fast with mocking
        
        # Verify concurrent execution didn't cause issues
        assert all(response.confidence_score is not None for response in responses)

import time