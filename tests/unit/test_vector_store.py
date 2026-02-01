"""Unit tests for VectorStore."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.core.vector_store import VectorStore, ChromaVectorStore
from src.utils.exceptions import VectorStoreError


class TestVectorStore:
    """Test cases for VectorStore base class."""

    @pytest.mark.unit
    def test_vector_store_abstract_methods(self):
        """Test that VectorStore is abstract and requires implementation."""
        with pytest.raises(TypeError):
            VectorStore()


class TestChromaVectorStore:
    """Test cases for ChromaVectorStore implementation."""

    @pytest.fixture
    def chroma_store(self):
        """Create ChromaVectorStore instance for testing."""
        with patch('src.core.vector_store.chromadb.Client') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            store = ChromaVectorStore(
                collection_name="test_collection",
                persist_directory="./test_chroma_db"
            )
            store.collection = mock_collection
            yield store

    @pytest.mark.unit
    def test_chroma_store_initialization(self, chroma_store):
        """Test ChromaVectorStore initialization."""
        assert chroma_store.collection_name == "test_collection"
        assert chroma_store.persist_directory == "./test_chroma_db"
        assert chroma_store.collection is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_documents_success(self, chroma_store):
        """Test successful addition of documents."""
        documents = [
            {"id": "doc1", "content": "First document", "metadata": {"type": "test"}},
            {"id": "doc2", "content": "Second document", "metadata": {"type": "test"}}
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        chroma_store.collection.add = MagicMock()
        
        result = await chroma_store.add_documents(documents, embeddings)
        
        assert result is True
        chroma_store.collection.add.assert_called_once()
        
        # Verify the call arguments
        call_args = chroma_store.collection.add.call_args
        assert len(call_args[1]['ids']) == 2
        assert len(call_args[1]['documents']) == 2
        assert len(call_args[1]['embeddings']) == 2
        assert len(call_args[1]['metadatas']) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_documents_error(self, chroma_store):
        """Test error handling during document addition."""
        documents = [{"id": "doc1", "content": "Test document"}]
        embeddings = [[0.1, 0.2, 0.3]]
        
        chroma_store.collection.add = MagicMock(side_effect=Exception("Database error"))
        
        with pytest.raises(VectorStoreError, match="Failed to add documents"):
            await chroma_store.add_documents(documents, embeddings)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_documents_success(self, chroma_store):
        """Test successful document search."""
        query_embedding = [0.1, 0.2, 0.3]
        mock_results = {
            'ids': [['doc1', 'doc2']],
            'documents': [['First document', 'Second document']],
            'metadatas': [[{'type': 'test'}, {'type': 'test'}]],
            'distances': [[0.1, 0.3]]
        }
        
        chroma_store.collection.query = MagicMock(return_value=mock_results)
        
        results = await chroma_store.search_documents(query_embedding, top_k=2)
        
        assert len(results) == 2
        assert results[0]['id'] == 'doc1'
        assert results[0]['content'] == 'First document'
        assert results[0]['metadata']['type'] == 'test'
        assert results[0]['score'] == 0.1
        
        chroma_store.collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=2,
            include=['documents', 'metadatas', 'distances']
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_documents_with_filter(self, chroma_store):
        """Test document search with metadata filters."""
        query_embedding = [0.1, 0.2, 0.3]
        filter_criteria = {"type": "important"}
        
        mock_results = {
            'ids': [['doc1']],
            'documents': [['Important document']],
            'metadatas': [[{'type': 'important'}]],
            'distances': [[0.1]]
        }
        
        chroma_store.collection.query = MagicMock(return_value=mock_results)
        
        results = await chroma_store.search_documents(
            query_embedding,
            top_k=5,
            where=filter_criteria
        )
        
        assert len(results) == 1
        assert results[0]['metadata']['type'] == 'important'
        
        chroma_store.collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=5,
            where=filter_criteria,
            include=['documents', 'metadatas', 'distances']
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_documents_error(self, chroma_store):
        """Test error handling during document search."""
        query_embedding = [0.1, 0.2, 0.3]
        chroma_store.collection.query = MagicMock(side_effect=Exception("Query error"))
        
        with pytest.raises(VectorStoreError, match="Failed to search documents"):
            await chroma_store.search_documents(query_embedding)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_documents_success(self, chroma_store):
        """Test successful document deletion."""
        document_ids = ["doc1", "doc2"]
        chroma_store.collection.delete = MagicMock()
        
        result = await chroma_store.delete_documents(document_ids)
        
        assert result is True
        chroma_store.collection.delete.assert_called_once_with(ids=document_ids)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_documents_error(self, chroma_store):
        """Test error handling during document deletion."""
        document_ids = ["doc1"]
        chroma_store.collection.delete = MagicMock(side_effect=Exception("Delete error"))
        
        with pytest.raises(VectorStoreError, match="Failed to delete documents"):
            await chroma_store.delete_documents(document_ids)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_document_count(self, chroma_store):
        """Test getting document count."""
        chroma_store.collection.count = MagicMock(return_value=42)
        
        count = await chroma_store.get_document_count()
        
        assert count == 42
        chroma_store.collection.count.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_document_success(self, chroma_store):
        """Test successful document update."""
        document_id = "doc1"
        new_content = "Updated document content"
        new_embedding = [0.7, 0.8, 0.9]
        new_metadata = {"updated": True}
        
        chroma_store.collection.update = MagicMock()
        
        result = await chroma_store.update_document(
            document_id, new_content, new_embedding, new_metadata
        )
        
        assert result is True
        chroma_store.collection.update.assert_called_once_with(
            ids=[document_id],
            documents=[new_content],
            embeddings=[new_embedding],
            metadatas=[new_metadata]
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_operations(self, chroma_store):
        """Test batch operations performance."""
        # Test large batch addition
        documents = [{"id": f"doc{i}", "content": f"Document {i}"} for i in range(1000)]
        embeddings = [[0.1, 0.2, 0.3] for _ in range(1000)]
        
        chroma_store.collection.add = MagicMock()
        
        result = await chroma_store.add_documents(documents, embeddings, batch_size=100)
        
        assert result is True
        # Should be called multiple times for batching
        assert chroma_store.collection.add.call_count >= 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_similarity_search_with_threshold(self, chroma_store):
        """Test similarity search with score threshold."""
        query_embedding = [0.1, 0.2, 0.3]
        mock_results = {
            'ids': [['doc1', 'doc2', 'doc3']],
            'documents': [['Good match', 'OK match', 'Poor match']],
            'metadatas': [[{}, {}, {}]],
            'distances': [[0.1, 0.5, 0.9]]  # Lower distance = higher similarity
        }
        
        chroma_store.collection.query = MagicMock(return_value=mock_results)
        
        # Set threshold to filter out poor matches
        results = await chroma_store.search_documents(
            query_embedding, 
            top_k=10, 
            score_threshold=0.6
        )
        
        # Should only return documents with distance <= 0.6
        assert len(results) == 2
        assert all(result['score'] <= 0.6 for result in results)

    @pytest.mark.unit
    def test_collection_creation_with_custom_embedding_function(self):
        """Test collection creation with custom embedding function."""
        with patch('src.core.vector_store.chromadb.Client') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Test with custom embedding function
            custom_embedding_fn = MagicMock()
            store = ChromaVectorStore(
                collection_name="custom_collection",
                embedding_function=custom_embedding_fn
            )
            
            # Verify collection was created with custom embedding function
            mock_client.return_value.get_or_create_collection.assert_called_once_with(
                name="custom_collection",
                embedding_function=custom_embedding_fn
            )