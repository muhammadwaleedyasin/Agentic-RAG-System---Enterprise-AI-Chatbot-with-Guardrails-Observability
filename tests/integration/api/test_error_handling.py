import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from src.app.main import app


class TestErrorHandling:
    """API integration tests for error handling and edge cases."""

    @pytest.fixture
    def test_client(self):
        """Test client for FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for testing."""
        return {"Authorization": "Bearer mock_test_token"}

    @pytest.fixture
    def mock_dependencies_with_errors(self):
        """Mock dependencies that can trigger various error conditions."""
        mocks = {}
        
        # Mock RAG pipeline with error scenarios
        mock_rag_pipeline = Mock()
        mock_rag_pipeline.query.side_effect = Exception("RAG pipeline error")
        mock_rag_pipeline.ingest_document.side_effect = ValueError("Invalid document format")
        mock_rag_pipeline.delete_document.side_effect = FileNotFoundError("Document not found")
        mock_rag_pipeline.health_check.return_value = False
        mocks['rag_pipeline'] = mock_rag_pipeline
        
        # Mock access controller with various auth scenarios
        mock_access_controller = Mock()
        mock_access_controller.authenticate_user.return_value = None  # Auth failure
        mock_access_controller.validate_session.return_value = None   # Invalid session
        mock_access_controller.check_permission.return_value = False  # Permission denied
        mocks['access_controller'] = mock_access_controller
        
        return mocks

    def test_validation_errors_missing_required_fields(self, test_client, auth_headers):
        """Test validation errors with missing required fields."""
        # Test chat endpoint without query
        response = test_client.post(
            "/api/v1/chat",
            json={},  # Missing required 'query' field
            headers=auth_headers
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        
        # Validation error should specify missing field
        error_details = data["detail"]
        assert isinstance(error_details, list)
        assert any("query" in str(error).lower() for error in error_details)

    def test_validation_errors_invalid_data_types(self, test_client, auth_headers):
        """Test validation errors with invalid data types."""
        # Test with invalid data types
        invalid_payloads = [
            {"query": 123},  # Should be string
            {"query": "test", "k": "not_a_number"},  # k should be integer
            {"query": "test", "temperature": "not_a_float"},  # temperature should be float
            {"query": ["list", "not", "string"]},  # query should be string
        ]
        
        for payload in invalid_payloads:
            response = test_client.post(
                "/api/v1/chat",
                json=payload,
                headers=auth_headers
            )
            
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data

    def test_malformed_json_requests(self, test_client, auth_headers):
        """Test handling of malformed JSON requests."""
        malformed_json_strings = [
            '{"query": "test",}',      # Trailing comma
            '{"query": test}',         # Unquoted value
            '{"query": "unclosed',     # Unclosed string
            '{query: "test"}',         # Unquoted key
            '{"query": "test" "extra": "field"}',  # Missing comma
        ]
        
        for malformed_json in malformed_json_strings:
            response = test_client.post(
                "/api/v1/chat",
                data=malformed_json,
                headers={**auth_headers, "Content-Type": "application/json"}
            )
            
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data

    def test_file_upload_errors_invalid_file_types(self, test_client, auth_headers):
        """Test file upload errors with invalid file types."""
        # Test with unsupported file types
        invalid_files = [
            ("malware.exe", b"MZ\x90\x00", "application/octet-stream"),
            ("image.jpg", b"\xff\xd8\xff", "image/jpeg"),
            ("binary.bin", b"\x00\x01\x02\x03", "application/octet-stream"),
        ]
        
        with patch('src.app.deps.get_access_controller') as mock_controller:
            mock_controller.return_value.validate_session.return_value = {"username": "test", "role": "employee"}
            mock_controller.return_value.check_permission.return_value = True
            
            for filename, content, content_type in invalid_files:
                response = test_client.post(
                    "/api/v1/documents/upload",
                    files={"file": (filename, content, content_type)},
                    headers=auth_headers
                )
                
                # Should reject invalid file types
                assert response.status_code in [400, 415, 422]
                data = response.json()
                assert "detail" in data or "error" in data

    def test_internal_server_errors_rag_pipeline_failure(self, test_client, mock_dependencies_with_errors, auth_headers):
        """Test internal server errors when RAG pipeline fails."""
        with patch('src.app.deps.get_rag_pipeline', return_value=mock_dependencies_with_errors['rag_pipeline']), \
             patch('src.app.deps.get_access_controller') as mock_controller:
            
            mock_controller.return_value.validate_session.return_value = {"username": "test", "role": "employee"}
            mock_controller.return_value.check_permission.return_value = True
            
            response = test_client.post(
                "/api/v1/chat",
                json={"query": "test query"},
                headers=auth_headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data or "error" in data
            
            # Should not expose internal error details
            error_message = str(data)
            assert "RAG pipeline error" not in error_message or "Internal Server Error" in error_message

    def test_authentication_errors_expired_tokens(self, test_client):
        """Test authentication errors with expired tokens."""
        with patch('src.app.deps.get_access_controller') as mock_controller:
            mock_controller.return_value.validate_session.return_value = None  # Expired token
            
            response = test_client.get(
                "/api/v1/documents",
                headers={"Authorization": "Bearer expired_token"}
            )
            
            assert response.status_code == 401
            data = response.json()
            assert "detail" in data

    def test_authentication_errors_malformed_tokens(self, test_client):
        """Test authentication errors with malformed tokens."""
        malformed_tokens = [
            "Bearer invalid.jwt.token",
            "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid",
            "Bearer not_a_jwt_at_all",
            "Bearer ",  # Empty token
        ]
        
        with patch('src.app.deps.get_access_controller') as mock_controller:
            mock_controller.return_value.validate_session.return_value = None
            
            for token in malformed_tokens:
                response = test_client.get(
                    "/api/v1/documents",
                    headers={"Authorization": token}
                )
                
                assert response.status_code == 401
                data = response.json()
                assert "detail" in data

    def test_authorization_errors_insufficient_permissions(self, test_client):
        """Test authorization errors with insufficient permissions."""
        with patch('src.app.deps.get_access_controller') as mock_controller:
            mock_controller.return_value.validate_session.return_value = {"username": "readonly", "role": "readonly"}
            mock_controller.return_value.check_permission.return_value = False
            
            # Try to access admin endpoint
            response = test_client.get(
                "/api/v1/admin/users",
                headers={"Authorization": "Bearer readonly_token"}
            )
            
            assert response.status_code == 403
            data = response.json()
            assert "detail" in data

    def test_resource_not_found_errors(self, test_client, auth_headers):
        """Test 404 errors for non-existent resources."""
        with patch('src.app.deps.get_access_controller') as mock_controller:
            mock_controller.return_value.validate_session.return_value = {"username": "test", "role": "employee"}
            mock_controller.return_value.check_permission.return_value = True
            
            # Test non-existent document
            response = test_client.get(
                "/api/v1/documents/nonexistent_document_id",
                headers=auth_headers
            )
            
            assert response.status_code == 404
            data = response.json()
            assert "detail" in data

    def test_method_not_allowed_errors(self, test_client, auth_headers):
        """Test 405 errors for unsupported HTTP methods."""
        # Try unsupported methods on endpoints
        unsupported_requests = [
            ("DELETE", "/api/v1/health"),
            ("PUT", "/api/v1/auth/login"),
            ("PATCH", "/api/v1/health"),
        ]
        
        for method, endpoint in unsupported_requests:
            response = getattr(test_client, method.lower())(endpoint, headers=auth_headers)
            
            assert response.status_code == 405
            data = response.json()
            assert "detail" in data

    def test_unsupported_media_type_errors(self, test_client, auth_headers):
        """Test 415 errors for unsupported content types."""
        unsupported_content_types = [
            "text/plain",
            "application/xml",
            "application/x-www-form-urlencoded",
            "multipart/form-data",  # For non-file endpoints
        ]
        
        with patch('src.app.deps.get_access_controller') as mock_controller:
            mock_controller.return_value.validate_session.return_value = {"username": "test", "role": "employee"}
            
            for content_type in unsupported_content_types:
                response = test_client.post(
                    "/api/v1/chat",
                    data="query=test",
                    headers={**auth_headers, "Content-Type": content_type}
                )
                
                assert response.status_code in [415, 422, 400]

    def test_edge_case_empty_requests(self, test_client, auth_headers):
        """Test edge cases with empty requests."""
        # Empty JSON body
        response = test_client.post(
            "/api/v1/chat",
            json={},
            headers=auth_headers
        )
        
        assert response.status_code in [422, 400]
        
        # No body at all
        response = test_client.post(
            "/api/v1/chat",
            headers=auth_headers
        )
        
        assert response.status_code in [422, 400]

    def test_edge_case_very_long_requests(self, test_client, auth_headers):
        """Test edge cases with very long requests."""
        # Very long query
        very_long_query = "a" * 10000  # 10KB query
        
        with patch('src.app.deps.get_access_controller') as mock_controller:
            mock_controller.return_value.validate_session.return_value = {"username": "test", "role": "employee"}
            mock_controller.return_value.check_permission.return_value = True
            
            response = test_client.post(
                "/api/v1/chat",
                json={"query": very_long_query},
                headers=auth_headers
            )
            
            # Should either process or reject gracefully
            assert response.status_code in [200, 400, 413, 422]

    def test_error_response_format_consistency(self, test_client):
        """Test that error responses have consistent format."""
        # Test various error scenarios
        error_responses = []
        
        # 400 Bad Request
        response = test_client.post("/api/v1/chat", json={})
        if response.status_code >= 400:
            error_responses.append(response)
        
        # 401 Unauthorized
        response = test_client.get("/api/v1/documents")
        if response.status_code >= 400:
            error_responses.append(response)
        
        # 404 Not Found
        response = test_client.get("/api/v1/nonexistent")
        if response.status_code >= 400:
            error_responses.append(response)
        
        # All error responses should have consistent format
        for response in error_responses:
            data = response.json()
            
            # Should have error information
            assert isinstance(data, dict)
            
            # Should have at least one of these fields
            error_fields = ["detail", "error", "message", "errors"]
            present_fields = [field for field in error_fields if field in data]
            assert len(present_fields) > 0

    def test_security_error_information_disclosure(self, test_client):
        """Test that error responses don't disclose sensitive information."""
        # Try to trigger internal errors
        with patch('src.app.deps.get_rag_pipeline') as mock_pipeline:
            mock_pipeline.side_effect = Exception("Internal database password: secret123")
            
            response = test_client.post(
                "/api/v1/chat",
                json={"query": "test"},
                headers={"Authorization": "Bearer test_token"}
            )
            
            if response.status_code >= 500:
                data = response.json()
                error_text = str(data).lower()
                
                # Should not expose sensitive information
                sensitive_keywords = ["password", "secret", "key", "token", "database", "internal"]
                for keyword in sensitive_keywords:
                    assert keyword not in error_text or "internal server error" in error_text
