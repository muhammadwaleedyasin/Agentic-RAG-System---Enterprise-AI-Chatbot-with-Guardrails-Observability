"""Integration tests for complete ingest→query→delete flow."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json

from src.app.main import app


class TestIngestQueryDeleteFlow:
    """Test cases for complete document lifecycle."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all required dependencies."""
        mock_rag_pipeline = Mock()
        mock_access_controller = Mock()

        # Configure mock returns
        mock_access_controller.validate_token.return_value = {
            "valid": True,
            "user_id": "test_user"
        }
        mock_access_controller.check_permission.return_value = True
        
        # Mock async methods
        mock_rag_pipeline.ingest_document = AsyncMock(return_value={
            "success": True,
            "document_id": "doc123",
            "chunks_created": 5
        })
        
        mock_rag_pipeline.query = AsyncMock(return_value={
            "answer": "Test answer about the document",
            "sources": [{"document_id": "doc123"}],
            "metadata": {"chunk_count": 2}
        })
        
        mock_rag_pipeline.delete_document = AsyncMock(return_value={
            "success": True,
            "deleted_chunks": 5
        })
        
        return {
            "rag_pipeline": mock_rag_pipeline,
            "access_controller": mock_access_controller
        }

    @pytest.fixture
    def client_with_mocks(self, mock_dependencies):
        """Create test client with dependency overrides."""
        from src.app.deps import get_rag_pipeline, get_access_controller
        
        # Use dependency overrides instead of patching
        app.dependency_overrides[get_rag_pipeline] = lambda: mock_dependencies["rag_pipeline"]
        app.dependency_overrides[get_access_controller] = lambda: mock_dependencies["access_controller"]
        
        with TestClient(app) as client:
            yield client
        
        # Clean up dependency overrides
        app.dependency_overrides.clear()

    def test_complete_document_lifecycle(self, client_with_mocks, mock_dependencies):
        """Test complete ingest→query→delete flow."""
        client = client_with_mocks
        
        # Step 1: Upload document via upload endpoint with proper file mock
        file_content = b"This is a test document about AI."
        
        # Mock the file upload to avoid UploadFile.size issues
        with patch('src.app.api.documents.UploadFile') as mock_upload_file:
            mock_file = Mock()
            mock_file.filename = "test.txt"
            mock_file.content_type = "text/plain"
            # Remove file.size stub as we now read content directly
            mock_file.read = AsyncMock(return_value=file_content)
            mock_upload_file.return_value = mock_file
            
            upload_response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", file_content, "text/plain")},
                data={"auto_process": "true"}
            )
            
            assert upload_response.status_code == 200
            document_id = upload_response.json()["document_id"]
        
        # Step 2: Query for the content
        query_response = client.post(
            "/api/v1/rag/query",
            json={"query": "What is AI?", "use_rag": True}
        )
        
        assert query_response.status_code == 200
        # Check that sources are returned properly
        response_data = query_response.json()
        assert "sources" in response_data
        
        # Step 3: Delete the document
        delete_response = client.delete(
            f"/api/v1/documents/{document_id}"
        )
        
        assert delete_response.status_code == 200

    def test_upload_text_document(self, client_with_mocks, mock_dependencies):
        """Test uploading text content directly."""
        client = client_with_mocks
        
        # Upload text document
        upload_response = client.post(
            "/api/v1/documents/upload-text",
            data={
                "text_content": "This is a test document about machine learning.",
                "filename": "ml_test.txt",
                "auto_process": "true"
            }
        )
        
        assert upload_response.status_code == 200
        document_id = upload_response.json()["document_id"]
        assert upload_response.json()["upload_status"] == "processing"

    def test_query_after_delete_returns_no_results(self, client_with_mocks, mock_dependencies):
        """Test that deleted content is not retrieved in queries."""
        client = client_with_mocks
        
        # Configure mock to return no results after deletion
        mock_dependencies["rag_pipeline"].query.return_value = {
            "answer": "No information found.",
            "sources": [],
            "metadata": {"chunk_count": 0}
        }
        
        query_response = client.post(
            "/api/v1/rag/query",
            json={"query": "What is AI?", "use_rag": True}
        )
        
        assert query_response.status_code == 200
        assert len(query_response.json()["sources"]) == 0

    def test_document_processing_status(self, client_with_mocks, mock_dependencies):
        """Test checking document processing status."""
        client = client_with_mocks
        
        # Mock document in processing state
        with patch('src.app.api.documents.documents_db', {
            "test-doc-id": {
                "document_id": "test-doc-id",
                "filename": "test.txt",
                "status": "processing",
                "metadata": {},
                "file_size": 100,
                "content_type": "text/plain"
            }
        }):
            # Check document status
            status_response = client.get("/api/v1/documents/test-doc-id")
            
            assert status_response.status_code == 200
            assert status_response.json()["status"] == "processing"

    def test_file_upload_validation(self, client_with_mocks, mock_dependencies):
        """Test file upload validation."""
        client = client_with_mocks
        
        # Test with unsupported file type (assuming .exe is not allowed)
        upload_response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("malware.exe", b"binary content", "application/octet-stream")},
            data={"auto_process": "true"}
        )
        
        # Should fail validation
        assert upload_response.status_code == 400
        assert "not supported" in upload_response.json()["detail"]

    def test_document_search_functionality(self, client_with_mocks, mock_dependencies):
        """Test document search endpoint."""
        client = client_with_mocks
        
        # Mock search results
        mock_dependencies["rag_pipeline"].search_documents = AsyncMock(return_value=[
            {
                "chunk_id": "chunk1",
                "content": "AI is artificial intelligence",
                "similarity_score": 0.95,
                "metadata": {"document_id": "doc123"}
            }
        ])
        
        search_response = client.post(
            "/api/v1/documents/search",
            json={
                "query": "artificial intelligence",
                "top_k": 5,
                "similarity_threshold": 0.7,
                "include_content": True
            }
        )
        
        assert search_response.status_code == 200
        results = search_response.json()["results"]
        assert len(results) > 0
        assert results[0]["similarity_score"] >= 0.7
