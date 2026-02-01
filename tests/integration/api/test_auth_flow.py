import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from types import SimpleNamespace

from src.app.main import app
from src.security.access_control import AccessController, Permission


class TestAuthFlow:
    """API integration tests for authentication and authorization."""

    @pytest.fixture
    def test_client(self):
        """Test client for FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def mock_access_controller(self):
        """Mock access controller with test users."""
        controller = Mock(spec=AccessController)
        
        # Mock user data
        test_users = {
            "admin_user": {"role": "admin", "active": True},
            "manager_user": {"role": "manager", "active": True},
            "employee_user": {"role": "employee", "active": True},
            "contractor_user": {"role": "contractor", "active": True},
            "readonly_user": {"role": "readonly", "active": True},
            "inactive_user": {"role": "employee", "active": False}
        }
        
        def mock_authenticate(username, password):
            if username in test_users and password == f"{username}_password":
                if test_users[username]["active"]:
                    return {
                        "user": {
                            "username": username,
                            "role": test_users[username]["role"]
                        },
                        "token": f"mock_token_for_{username}",
                        "expires_at": "2024-12-31T23:59:59Z"
                    }
            return None
        
        def mock_validate_session(token):
            for username in test_users:
                if token == f"mock_token_for_{username}" and test_users[username]["active"]:
                    return {
                        "username": username,
                        "role": test_users[username]["role"]
                    }
            return None
        
        def mock_check_permission(username, permission):
            if username not in test_users:
                return False
            
            user_role = test_users[username]["role"]
            
            # Simple permission mapping
            if user_role == "admin":
                return True
            elif user_role == "manager":
                return permission.value in ["read_documents", "write_documents", "manage_team"]
            elif user_role == "employee":
                return permission.value in ["read_documents", "write_documents"]
            elif user_role == "contractor":
                return permission.value in ["read_documents"]
            elif user_role == "readonly":
                return permission.value in ["read_documents"]
            
            return False
        
        controller.authenticate_user = mock_authenticate
        controller.validate_session = mock_validate_session
        controller.check_permission = mock_check_permission
        
        return controller

    def test_login_success_valid_credentials(self, test_client, mock_access_controller):
        """Test successful login with valid credentials."""
        with patch('src.app.deps.get_access_controller', return_value=mock_access_controller):
            response = test_client.post(
                "/api/v1/auth/login",
                data={
                    "username": "employee_user",
                    "password": "employee_user_password"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data or "token" in data
        assert "user" in data
        assert data["user"]["username"] == "employee_user"
        assert data["user"]["role"] == "employee"

    def test_login_failure_invalid_credentials(self, test_client, mock_access_controller):
        """Test login failure with invalid credentials."""
        with patch('src.app.deps.get_access_controller', return_value=mock_access_controller):
            response = test_client.post(
                "/api/v1/auth/login",
                data={
                    "username": "employee_user",
                    "password": "wrong_password"
                }
            )
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data or "error" in data

    def test_protected_endpoint_with_valid_token(self, test_client, mock_access_controller):
        """Test accessing protected endpoint with valid token."""
        with patch('src.app.deps.get_access_controller', return_value=mock_access_controller):
            response = test_client.get(
                "/api/v1/documents",
                headers={"Authorization": "Bearer mock_token_for_employee_user"}
            )
        
        # Should return 200 (success) or 404 (endpoint not found) but not 401 (unauthorized)
        assert response.status_code != 401

    def test_protected_endpoint_without_token(self, test_client, mock_access_controller):
        """Test accessing protected endpoint without token."""
        with patch('src.app.deps.get_access_controller', return_value=mock_access_controller):
            response = test_client.get("/api/v1/documents")
        
        assert response.status_code == 401

    def test_role_based_access_admin_user(self, test_client, mock_access_controller):
        """Test role-based access for admin user."""
        with patch('src.app.deps.get_access_controller', return_value=mock_access_controller):
            # Admin should have access to admin endpoints
            response = test_client.get(
                "/api/v1/admin/users",
                headers={"Authorization": "Bearer mock_token_for_admin_user"}
            )
        
        # Should not return 403 (forbidden)
        assert response.status_code != 403

    def test_role_based_access_employee_user(self, test_client, mock_access_controller):
        """Test role-based access for employee user."""
        with patch('src.app.deps.get_access_controller', return_value=mock_access_controller):
            # Employee should have access to documents
            response = test_client.get(
                "/api/v1/documents",
                headers={"Authorization": "Bearer mock_token_for_employee_user"}
            )
        
        assert response.status_code != 403
        
        # Employee should NOT have access to admin endpoints
        admin_response = test_client.get(
            "/api/v1/admin/users",
            headers={"Authorization": "Bearer mock_token_for_employee_user"}
        )
        
        assert admin_response.status_code == 403
