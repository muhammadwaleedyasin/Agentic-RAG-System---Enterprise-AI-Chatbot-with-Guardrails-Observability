"""Unit tests for AccessController from security access control module."""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.security.access_control import (
    AccessController, 
    User, 
    Role, 
    Permission
)


class TestAccessController:
    """Test suite for AccessController class."""
    
    @pytest.fixture
    def access_controller(self):
        """Create AccessController instance for testing."""
        return AccessController(secret_key="test_secret_key")
    
    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for testing."""
        return {
            "user_id": "test_user_1",
            "username": "testuser",
            "email": "test@example.com",
            "role_name": "employee",
            "password": "test_password"
        }
    
    def test_init_with_secret_key(self):
        """Test AccessController initialization with provided secret key."""
        secret_key = "custom_secret"
        controller = AccessController(secret_key=secret_key)
        
        assert controller.secret_key == secret_key
        assert isinstance(controller.users, dict)
        assert isinstance(controller.active_sessions, dict)
        assert len(controller.roles) > 0
    
    def test_init_without_secret_key(self):
        """Test AccessController initialization without secret key."""
        controller = AccessController()
        
        assert controller.secret_key is not None
        assert len(controller.secret_key) > 0
    
    def test_default_roles_setup(self, access_controller):
        """Test that default roles are set up correctly."""
        expected_roles = ["admin", "manager", "employee", "contractor", "readonly"]
        
        for role_name in expected_roles:
            assert role_name in access_controller.roles
            assert isinstance(access_controller.roles[role_name], Role)
    
    def test_admin_role_permissions(self, access_controller):
        """Test that admin role has all permissions."""
        admin_role = access_controller.roles["admin"]
        
        # Admin should have all permissions
        all_permissions = set(Permission)
        assert admin_role.permissions == all_permissions
    
    def test_employee_role_permissions(self, access_controller):
        """Test that employee role has limited permissions."""
        employee_role = access_controller.roles["employee"]
        
        expected_permissions = {Permission.READ_DOCUMENTS, Permission.QUERY_SYSTEM}
        assert employee_role.permissions == expected_permissions
    
    def test_create_user_success(self, access_controller, sample_user_data):
        """Test successful user creation."""
        user = access_controller.create_user(**sample_user_data)
        
        assert isinstance(user, User)
        assert user.user_id == sample_user_data["user_id"]
        assert user.username == sample_user_data["username"]
        assert user.email == sample_user_data["email"]
        assert user.role.name == sample_user_data["role_name"]
        assert user.is_active is True
        
        # User should be stored in controller
        assert sample_user_data["user_id"] in access_controller.users
    
    def test_create_user_duplicate_id(self, access_controller, sample_user_data):
        """Test creating user with duplicate ID raises error."""
        # Create first user
        access_controller.create_user(**sample_user_data)
        
        # Try to create duplicate
        with pytest.raises(ValueError, match="User test_user_1 already exists"):
            access_controller.create_user(**sample_user_data)
    
    def test_create_user_invalid_role(self, access_controller, sample_user_data):
        """Test creating user with invalid role raises error."""
        sample_user_data["role_name"] = "nonexistent_role"
        
        with pytest.raises(ValueError, match="Role nonexistent_role does not exist"):
            access_controller.create_user(**sample_user_data)
    
    def test_authenticate_user_success(self, access_controller, sample_user_data):
        """Test successful user authentication."""
        # Create user first
        access_controller.create_user(**sample_user_data)
        
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Token should be in active sessions
        assert token in access_controller.active_sessions
        
        # User should have updated login info
        user = access_controller.users[sample_user_data["user_id"]]
        assert user.last_login is not None
        assert user.session_token == token
        assert user.failed_login_attempts == 0
    
    def test_authenticate_user_not_found(self, access_controller):
        """Test authentication with nonexistent user."""
        token = access_controller.authenticate_user("nonexistent", "password")
        
        assert token is None
    
    def test_authenticate_user_wrong_password(self, access_controller, sample_user_data):
        """Test authentication with wrong password."""
        # Create user first
        user = access_controller.create_user(**sample_user_data)
        
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            "wrong_password"
        )
        
        assert token is None
        
        # Failed attempts should be incremented
        assert user.failed_login_attempts == 1
    
    def test_authenticate_user_account_locked(self, access_controller, sample_user_data):
        """Test authentication with locked account."""
        # Create user and lock account
        user = access_controller.create_user(**sample_user_data)
        user.failed_login_attempts = access_controller.max_failed_attempts
        
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        assert token is None
    
    def test_validate_session_success(self, access_controller, sample_user_data):
        """Test successful session validation."""
        # Create user and authenticate
        access_controller.create_user(**sample_user_data)
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        user = access_controller.validate_session(token)
        
        assert user is not None
        assert user.user_id == sample_user_data["user_id"]
    
    def test_validate_session_invalid_token(self, access_controller):
        """Test session validation with invalid token."""
        user = access_controller.validate_session("invalid_token")
        
        assert user is None
    
    def test_validate_session_expired(self, access_controller, sample_user_data):
        """Test session validation with expired session."""
        # Create user and authenticate
        access_controller.create_user(**sample_user_data)
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        # Manually expire session
        user = access_controller.users[sample_user_data["user_id"]]
        user.session_expires = datetime.now() - timedelta(minutes=1)
        
        validated_user = access_controller.validate_session(token)
        
        assert validated_user is None
        # Expired session should be cleaned up
        assert token not in access_controller.active_sessions
    
    def test_check_permission_success(self, access_controller, sample_user_data):
        """Test successful permission check."""
        # Create user and authenticate
        access_controller.create_user(**sample_user_data)
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        # Employee should have READ_DOCUMENTS permission
        has_permission = access_controller.check_permission(
            token, 
            Permission.READ_DOCUMENTS
        )
        
        assert has_permission is True
    
    def test_check_permission_denied(self, access_controller, sample_user_data):
        """Test permission check denial."""
        # Create user and authenticate
        access_controller.create_user(**sample_user_data)
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        # Employee should not have DELETE_DOCUMENTS permission
        has_permission = access_controller.check_permission(
            token, 
            Permission.DELETE_DOCUMENTS
        )
        
        assert has_permission is False
    
    def test_check_permission_invalid_session(self, access_controller):
        """Test permission check with invalid session."""
        has_permission = access_controller.check_permission(
            "invalid_token", 
            Permission.READ_DOCUMENTS
        )
        
        assert has_permission is False
    
    def test_check_permission_with_document_metadata(self, access_controller, sample_user_data):
        """Test permission check with document metadata filtering."""
        # Create user and authenticate
        access_controller.create_user(**sample_user_data)
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        # Employee can access public documents
        has_permission = access_controller.check_permission(
            token,
            Permission.READ_DOCUMENTS,
            document_metadata={"access_level": "public"}
        )
        assert has_permission is True
        
        # Employee cannot access confidential documents
        has_permission = access_controller.check_permission(
            token,
            Permission.READ_DOCUMENTS,
            document_metadata={"access_level": "confidential"}
        )
        assert has_permission is False
    
    def test_filter_documents_by_access(self, access_controller, sample_user_data):
        """Test document filtering based on user access."""
        # Create user and authenticate
        access_controller.create_user(**sample_user_data)
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        documents = [
            {"id": "1", "metadata": {"access_level": "public"}},
            {"id": "2", "metadata": {"access_level": "internal"}},
            {"id": "3", "metadata": {"access_level": "confidential"}},
            {"id": "4", "metadata": {}}  # No access level
        ]
        
        filtered_docs = access_controller.filter_documents_by_access(token, documents)
        
        # Employee should only see public and internal documents
        assert len(filtered_docs) == 2
        doc_ids = [doc["id"] for doc in filtered_docs]
        assert "1" in doc_ids  # public
        assert "2" in doc_ids  # internal
        assert "3" not in doc_ids  # confidential
        assert "4" not in doc_ids  # no access level (restricted)
    
    def test_filter_documents_invalid_session(self, access_controller):
        """Test document filtering with invalid session."""
        documents = [{"id": "1", "metadata": {"access_level": "public"}}]
        
        filtered_docs = access_controller.filter_documents_by_access(
            "invalid_token", 
            documents
        )
        
        assert filtered_docs == []
    
    def test_logout_user(self, access_controller, sample_user_data):
        """Test user logout."""
        # Create user and authenticate
        access_controller.create_user(**sample_user_data)
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        # Logout user
        access_controller.logout_user(token)
        
        # Session should be removed
        assert token not in access_controller.active_sessions
        
        # User session info should be cleared
        user = access_controller.users[sample_user_data["user_id"]]
        assert user.session_token is None
        assert user.session_expires is None
    
    def test_logout_invalid_token(self, access_controller):
        """Test logout with invalid token."""
        # Should not raise exception
        access_controller.logout_user("invalid_token")
    
    def test_update_user_role_success(self, access_controller, sample_user_data):
        """Test successful user role update."""
        # Create user
        access_controller.create_user(**sample_user_data)
        
        # Update role
        access_controller.update_user_role(sample_user_data["user_id"], "manager")
        
        user = access_controller.users[sample_user_data["user_id"]]
        assert user.role.name == "manager"
    
    def test_update_user_role_nonexistent_user(self, access_controller):
        """Test updating role for nonexistent user."""
        with pytest.raises(ValueError, match="User nonexistent not found"):
            access_controller.update_user_role("nonexistent", "manager")
    
    def test_update_user_role_invalid_role(self, access_controller, sample_user_data):
        """Test updating to invalid role."""
        access_controller.create_user(**sample_user_data)
        
        with pytest.raises(ValueError, match="Role invalid_role does not exist"):
            access_controller.update_user_role(sample_user_data["user_id"], "invalid_role")
    
    def test_deactivate_user(self, access_controller, sample_user_data):
        """Test user deactivation."""
        # Create user and authenticate
        access_controller.create_user(**sample_user_data)
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        # Deactivate user
        access_controller.deactivate_user(sample_user_data["user_id"])
        
        user = access_controller.users[sample_user_data["user_id"]]
        assert user.is_active is False
        
        # Active sessions should be removed
        assert token not in access_controller.active_sessions
    
    def test_get_user_audit_info(self, access_controller, sample_user_data):
        """Test getting user audit information."""
        # Create user
        access_controller.create_user(**sample_user_data)
        
        audit_info = access_controller.get_user_audit_info(sample_user_data["user_id"])
        
        assert "user_id" in audit_info
        assert "username" in audit_info
        assert "role" in audit_info
        assert "permissions" in audit_info
        assert "is_active" in audit_info
        assert "created_at" in audit_info
        assert "last_login" in audit_info
        assert "session_active" in audit_info
        assert "failed_attempts" in audit_info
        
        assert audit_info["user_id"] == sample_user_data["user_id"]
        assert audit_info["username"] == sample_user_data["username"]
        assert audit_info["role"] == sample_user_data["role_name"]
        assert audit_info["is_active"] is True
    
    def test_get_user_audit_info_nonexistent(self, access_controller):
        """Test getting audit info for nonexistent user."""
        audit_info = access_controller.get_user_audit_info("nonexistent")
        
        assert audit_info == {}
    
    def test_cleanup_expired_sessions(self, access_controller, sample_user_data):
        """Test cleanup of expired sessions."""
        # Create user and authenticate
        access_controller.create_user(**sample_user_data)
        token = access_controller.authenticate_user(
            sample_user_data["username"], 
            sample_user_data["password"]
        )
        
        # Manually expire session
        user = access_controller.users[sample_user_data["user_id"]]
        user.session_expires = datetime.now() - timedelta(minutes=1)
        
        # Run cleanup
        access_controller.cleanup_expired_sessions()
        
        # Expired session should be removed
        assert token not in access_controller.active_sessions
    
    @patch('src.security.access_control.jwt.encode')
    def test_generate_session_token(self, mock_jwt_encode, access_controller, sample_user_data):
        """Test JWT session token generation."""
        mock_jwt_encode.return_value = "mocked_jwt_token"
        
        # Create user
        user = access_controller.create_user(**sample_user_data)
        
        token = access_controller._generate_session_token(user)
        
        assert token == "mocked_jwt_token"
        mock_jwt_encode.assert_called_once()
        
        # Check JWT payload structure
        call_args = mock_jwt_encode.call_args[0]
        payload = call_args[0]
        
        assert "user_id" in payload
        assert "username" in payload
        assert "role" in payload
        assert "iat" in payload
        assert "exp" in payload
    
    def test_hash_password(self, access_controller):
        """Test password hashing."""
        password = "test_password"
        hash1 = access_controller._hash_password(password)
        hash2 = access_controller._hash_password(password)
        
        # Same password should produce same hash
        assert hash1 == hash2
        assert len(hash1) > 0
        assert hash1 != password  # Should be different from plain password
    
    def test_verify_password(self, access_controller):
        """Test password verification."""
        password = "test_password"
        stored_hash = access_controller._hash_password(password)
        
        # Test correct password
        result = access_controller._verify_password(password, stored_hash)
        assert result is True
        
        # Test incorrect password
        result = access_controller._verify_password("wrong_password", stored_hash)
        assert result is False
    
    def test_find_user_by_username(self, access_controller, sample_user_data):
        """Test finding user by username."""
        # Create user
        access_controller.create_user(**sample_user_data)
        
        found_user = access_controller._find_user_by_username(sample_user_data["username"])
        
        assert found_user is not None
        assert found_user.username == sample_user_data["username"]
        assert found_user.user_id == sample_user_data["user_id"]
    
    def test_find_user_by_username_not_found(self, access_controller):
        """Test finding nonexistent user by username."""
        found_user = access_controller._find_user_by_username("nonexistent")
        
        assert found_user is None


class TestRole:
    """Test suite for Role class."""
    
    def test_role_creation(self):
        """Test Role creation with basic properties."""
        permissions = {Permission.READ_DOCUMENTS, Permission.QUERY_SYSTEM}
        role = Role(
            name="test_role",
            permissions=permissions,
            description="Test role",
            max_session_duration=240
        )
        
        assert role.name == "test_role"
        assert role.permissions == permissions
        assert role.description == "Test role"
        assert role.max_session_duration == 240
    
    def test_has_permission_true(self):
        """Test permission check returns True when permission exists."""
        permissions = {Permission.READ_DOCUMENTS, Permission.QUERY_SYSTEM}
        role = Role(name="test", permissions=permissions)
        
        assert role.has_permission(Permission.READ_DOCUMENTS) is True
    
    def test_has_permission_false(self):
        """Test permission check returns False when permission missing."""
        permissions = {Permission.READ_DOCUMENTS}
        role = Role(name="test", permissions=permissions)
        
        assert role.has_permission(Permission.DELETE_DOCUMENTS) is False
    
    def test_can_access_document_no_filters(self):
        """Test document access when no filters are defined."""
        role = Role(name="test", permissions=set())
        
        # Should allow access when no filters
        result = role.can_access_document({"any": "metadata"})
        assert result is True
    
    def test_can_access_document_matching_filter(self):
        """Test document access with matching filters."""
        role = Role(
            name="test",
            permissions=set(),
            document_filters={"department": ["hr", "finance"]}
        )
        
        result = role.can_access_document({"department": "hr"})
        assert result is True
    
    def test_can_access_document_non_matching_filter(self):
        """Test document access with non-matching filters."""
        role = Role(
            name="test",
            permissions=set(),
            document_filters={"department": ["hr", "finance"]}
        )
        
        result = role.can_access_document({"department": "engineering"})
        assert result is False
    
    def test_can_access_document_single_value_filter(self):
        """Test document access with single value filter."""
        role = Role(
            name="test",
            permissions=set(),
            document_filters={"access_level": "public"}
        )
        
        result = role.can_access_document({"access_level": "public"})
        assert result is True
        
        result = role.can_access_document({"access_level": "private"})
        assert result is False
    
    def test_can_access_document_missing_metadata_key(self):
        """Test document access when metadata key is missing."""
        role = Role(
            name="test",
            permissions=set(),
            document_filters={"department": ["hr"]}
        )
        
        # Should allow access when metadata key is missing
        result = role.can_access_document({"other_key": "value"})
        assert result is True


class TestUser:
    """Test suite for User class."""
    
    @pytest.fixture
    def sample_role(self):
        """Create sample role for testing."""
        return Role(
            name="test_role",
            permissions={Permission.READ_DOCUMENTS}
        )
    
    def test_user_creation(self, sample_role):
        """Test User creation with basic properties."""
        user = User(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            role=sample_role
        )
        
        assert user.user_id == "test_user"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == sample_role
        assert user.is_active is True
        assert user.session_token is None
        assert user.failed_login_attempts == 0
    
    def test_is_session_valid_no_token(self, sample_role):
        """Test session validation with no token."""
        user = User("id", "user", "email", sample_role)
        
        assert user.is_session_valid() is False
    
    def test_is_session_valid_expired(self, sample_role):
        """Test session validation with expired session."""
        user = User("id", "user", "email", sample_role)
        user.session_token = "token"
        user.session_expires = datetime.now() - timedelta(minutes=1)
        
        assert user.is_session_valid() is False
    
    def test_is_session_valid_active(self, sample_role):
        """Test session validation with active session."""
        user = User("id", "user", "email", sample_role)
        user.session_token = "token"
        user.session_expires = datetime.now() + timedelta(minutes=30)
        
        assert user.is_session_valid() is True
    
    def test_can_perform_action_inactive_user(self, sample_role):
        """Test action permission for inactive user."""
        user = User("id", "user", "email", sample_role)
        user.is_active = False
        
        assert user.can_perform_action(Permission.READ_DOCUMENTS) is False
    
    def test_can_perform_action_active_user_with_permission(self, sample_role):
        """Test action permission for active user with permission."""
        user = User("id", "user", "email", sample_role)
        
        assert user.can_perform_action(Permission.READ_DOCUMENTS) is True
    
    def test_can_perform_action_active_user_without_permission(self, sample_role):
        """Test action permission for active user without permission."""
        user = User("id", "user", "email", sample_role)
        
        assert user.can_perform_action(Permission.DELETE_DOCUMENTS) is False