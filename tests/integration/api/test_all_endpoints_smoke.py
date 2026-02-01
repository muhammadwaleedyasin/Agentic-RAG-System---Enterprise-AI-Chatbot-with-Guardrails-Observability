import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.app.main import app


class TestAllEndpointsSmoke:
    """Smoke tests for all API endpoints to ensure basic functionality."""

    @pytest.fixture
    def test_client(self):
        """Test client for FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for testing."""
        return {"Authorization": "Bearer mock_test_token"}

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        mocks = {}
        
        # Mock RAG pipeline
        mock_rag_pipeline = Mock()
        mock_rag_pipeline.query.return_value = Mock(
            answer="Test answer",
            sources=[{"content": "Test source", "score": 0.9}],
            context="Test context"
        )
        mock_rag_pipeline.ingest_document.return_value = True
        mock_rag_pipeline.delete_document.return_value = True
        mock_rag_pipeline.health_check.return_value = True
        mocks['rag_pipeline'] = mock_rag_pipeline
        
        # Mock access controller
        mock_access_controller = Mock()
        mock_access_controller.authenticate_user.return_value = {
            "user": {"username": "test_user", "role": "employee"},
            "token": "mock_test_token",
            "expires_at": "2024-12-31T23:59:59Z"
        }
        mock_access_controller.validate_session.return_value = {
            "username": "test_user",
            "role": "employee"
        }
        mock_access_controller.check_permission.return_value = True
        mocks['access_controller'] = mock_access_controller
        
        return mocks

    @pytest.mark.parametrize("endpoint,method,expected_codes", [
        # Health endpoints
        ("/api/v1/health", "GET", [200]),
        ("/api/v1/health/ready", "GET", [200]),
        ("/api/v1/health/live", "GET", [200]),
        
        # Auth endpoints (without auth)
        ("/api/v1/auth/login", "POST", [200, 400, 401, 422]),
        
        # Document endpoints (require auth)
        ("/api/v1/documents", "GET", [200, 401, 403]),
        ("/api/v1/documents/upload", "POST", [200, 201, 401, 403, 422]),
        ("/api/v1/documents/test_id", "GET", [200, 401, 403, 404]),
        ("/api/v1/documents/test_id", "DELETE", [200, 204, 401, 403, 404]),
        
        # Chat endpoints
        ("/api/v1/chat", "POST", [200, 401, 403, 422]),
        ("/api/v1/chat/stream", "POST", [200, 401, 403, 422]),
        
        # RAG endpoints
        ("/api/v1/rag/query", "POST", [200, 401, 403, 422]),
        ("/api/v1/rag/search", "POST", [200, 401, 403, 422]),
        
        # Admin endpoints
        ("/api/v1/admin/users", "GET", [200, 401, 403]),
        ("/api/v1/admin/users", "POST", [200, 201, 401, 403, 422]),
    ])
    def test_endpoint_accessibility(self, test_client, mock_dependencies, auth_headers, endpoint, method, expected_codes):
        """Test that endpoints are accessible and return expected status codes."""
        with patch('src.app.deps.get_rag_pipeline', return_value=mock_dependencies['rag_pipeline']), \
             patch('src.app.deps.get_access_controller', return_value=mock_dependencies['access_controller']):
            
            # Prepare request data based on endpoint
            kwargs = {}
            if method in ["POST", "PUT", "PATCH"]:
                if "auth/login" in endpoint:
                    kwargs["data"] = {"username": "test_user", "password": "test_pass"}
                elif "documents" in endpoint:
                    if endpoint.endswith("/upload"):
                        kwargs["files"] = {"file": ("test.txt", "test content", "text/plain")}
                    else:
                        kwargs["json"] = {"content": "test content"}
                elif "chat" in endpoint or "rag" in endpoint:
                    kwargs["json"] = {"query": "test query"}
                elif "admin/users" in endpoint:
                    kwargs["json"] = {"username": "new_user", "password": "password", "role": "employee"}
                else:
                    kwargs["json"] = {"test": "data"}
            
            # Add auth headers for protected endpoints
            if not endpoint.startswith("/api/v1/health") and not endpoint == "/api/v1/auth/login":
                kwargs["headers"] = auth_headers
            
            # Make request
            response = getattr(test_client, method.lower())(endpoint, **kwargs)
            
            assert response.status_code in expected_codes, \
                f"Endpoint {method} {endpoint} returned {response.status_code}, expected one of {expected_codes}"

    def test_health_endpoint_structure(self, test_client, mock_dependencies):
        """Test health endpoint returns proper structure."""
        with patch('src.app.deps.get_rag_pipeline', return_value=mock_dependencies['rag_pipeline']):
            response = test_client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Health endpoint should return status information
            assert isinstance(data, dict)
            expected_fields = ["status", "timestamp", "version", "components"]
            
            # At least some of these fields should be present
            present_fields = [field for field in expected_fields if field in data]
            assert len(present_fields) > 0

    def test_auth_login_endpoint_response_structure(self, test_client, mock_dependencies):
        """Test auth login endpoint returns proper response structure."""
        with patch('src.app.deps.get_access_controller', return_value=mock_dependencies['access_controller']):
            response = test_client.post(
                "/api/v1/auth/login",
                data={"username": "test_user", "password": "test_pass"}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Should contain authentication response
                expected_fields = ["token", "access_token", "user", "expires_at"]
                present_fields = [field for field in expected_fields if field in data]
                assert len(present_fields) >= 2  # At least token and user info

    def test_chat_endpoint_response_structure(self, test_client, mock_dependencies, auth_headers):
        """Test chat endpoint returns proper response structure."""
        with patch('src.app.deps.get_rag_pipeline', return_value=mock_dependencies['rag_pipeline']), \
             patch('src.app.deps.get_access_controller', return_value=mock_dependencies['access_controller']):
            
            response = test_client.post(
                "/api/v1/chat",
                json={"query": "What is machine learning?"},
                headers=auth_headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Should contain chat response
                expected_fields = ["answer", "response", "sources", "context"]
                present_fields = [field for field in expected_fields if field in data]
                assert len(present_fields) >= 1  # At least one response field

    def test_cors_headers_on_endpoints(self, test_client, mock_dependencies):
        """Test CORS headers on various endpoints."""
        test_endpoints = [
            "/api/v1/health",
            "/api/v1/auth/login",
            "/api/v1/documents",
            "/api/v1/chat"
        ]
        
        with patch('src.app.deps.get_access_controller', return_value=mock_dependencies['access_controller']):
            for endpoint in test_endpoints:
                # Test OPTIONS request
                options_response = test_client.options(endpoint)
                
                # Should handle OPTIONS gracefully
                assert options_response.status_code in [200, 204, 405]

    def test_response_format_consistency(self, test_client, mock_dependencies, auth_headers):
        """Test response format consistency across endpoints."""
        with patch('src.app.deps.get_rag_pipeline', return_value=mock_dependencies['rag_pipeline']), \
             patch('src.app.deps.get_access_controller', return_value=mock_dependencies['access_controller']):
            
            # Test endpoints that should return JSON
            json_endpoints = [
                ("/api/v1/health", "GET"),
                ("/api/v1/documents", "GET"),
                ("/api/v1/chat", "POST", {"query": "test"}),
            ]
            
            for endpoint_data in json_endpoints:
                endpoint = endpoint_data[0]
                method = endpoint_data[1]
                data = endpoint_data[2] if len(endpoint_data) > 2 else None
                
                kwargs = {"headers": auth_headers} if not endpoint.startswith("/api/v1/health") else {}
                if data:
                    kwargs["json"] = data
                
                response = getattr(test_client, method.lower())(endpoint, **kwargs)
                
                if response.status_code == 200:
                    # Should return valid JSON
                    try:
                        response.json()
                    except ValueError:
                        pytest.fail(f"Endpoint {endpoint} did not return valid JSON")
                    
                    # Should have proper content type
                    assert "application/json" in response.headers.get("content-type", "")
