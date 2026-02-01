"""
Role-Based Access Control (RBAC) System

Implements enterprise-grade access control for RAG systems with
role-based permissions and document-level security.
"""
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import jwt
import secrets


class Permission(Enum):
    """System permissions"""
    READ_DOCUMENTS = "read_documents"
    WRITE_DOCUMENTS = "write_documents"
    DELETE_DOCUMENTS = "delete_documents"
    ADMIN_ACCESS = "admin_access"
    QUERY_SYSTEM = "query_system"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_USERS = "manage_users"
    CONFIGURE_SYSTEM = "configure_system"
    ACCESS_SENSITIVE = "access_sensitive"
    EXPORT_DATA = "export_data"


@dataclass
class Role:
    """User role with associated permissions"""
    name: str
    permissions: Set[Permission]
    description: str = ""
    document_filters: Dict[str, Any] = field(default_factory=dict)
    max_session_duration: int = 480  # minutes
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission"""
        return permission in self.permissions
    
    def can_access_document(self, document_metadata: Dict[str, Any]) -> bool:
        """Check if role can access specific document based on metadata"""
        if not self.document_filters:
            return True
        
        for filter_key, allowed_values in self.document_filters.items():
            if filter_key in document_metadata:
                doc_value = document_metadata[filter_key]
                if isinstance(allowed_values, list):
                    if doc_value not in allowed_values:
                        return False
                elif doc_value != allowed_values:
                    return False
        
        return True


@dataclass
class User:
    """System user with role and session management"""
    user_id: str
    username: str
    email: str
    role: Role
    password_hash: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    session_token: Optional[str] = None
    session_expires: Optional[datetime] = None
    failed_login_attempts: int = 0
    
    def is_session_valid(self) -> bool:
        """Check if user session is still valid"""
        if not self.session_token or not self.session_expires:
            return False
        return datetime.now() < self.session_expires
    
    def can_perform_action(self, permission: Permission) -> bool:
        """Check if user can perform specific action"""
        if not self.is_active:
            return False
        return self.role.has_permission(permission)


class AccessController:
    """Central access control system"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, str] = {}  # token -> user_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize default roles
        self._setup_default_roles()
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.session_timeout = timedelta(hours=8)
    
    def _setup_default_roles(self):
        """Set up default system roles"""
        self.roles = {
            "admin": Role(
                name="admin",
                permissions={
                    Permission.READ_DOCUMENTS, Permission.WRITE_DOCUMENTS,
                    Permission.DELETE_DOCUMENTS, Permission.ADMIN_ACCESS,
                    Permission.QUERY_SYSTEM, Permission.VIEW_ANALYTICS,
                    Permission.MANAGE_USERS, Permission.CONFIGURE_SYSTEM,
                    Permission.ACCESS_SENSITIVE, Permission.EXPORT_DATA
                },
                description="Full system administrator",
                max_session_duration=480
            ),
            "manager": Role(
                name="manager",
                permissions={
                    Permission.READ_DOCUMENTS, Permission.WRITE_DOCUMENTS,
                    Permission.QUERY_SYSTEM, Permission.VIEW_ANALYTICS,
                    Permission.ACCESS_SENSITIVE
                },
                description="Department manager with elevated access",
                document_filters={"department": ["hr", "finance", "legal"]},
                max_session_duration=360
            ),
            "employee": Role(
                name="employee",
                permissions={
                    Permission.READ_DOCUMENTS, Permission.QUERY_SYSTEM
                },
                description="Standard employee access",
                document_filters={"access_level": ["public", "internal"]},
                max_session_duration=240
            ),
            "contractor": Role(
                name="contractor",
                permissions={Permission.READ_DOCUMENTS, Permission.QUERY_SYSTEM},
                description="External contractor with limited access",
                document_filters={"access_level": ["public"]},
                max_session_duration=120
            ),
            "readonly": Role(
                name="readonly",
                permissions={Permission.READ_DOCUMENTS},
                description="Read-only access for compliance/audit",
                max_session_duration=480
            )
        }
    
    def create_user(self,
                   user_id: str,
                   username: str,
                   email: str,
                   role_name: str,
                   password: str) -> User:
        """Create a new user account"""
        if user_id in self.users:
            raise ValueError(f"User {user_id} already exists")
        
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")
        
        # Hash password
        password_hash = self._hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=self.roles[role_name],
            password_hash=password_hash
        )
        
        self.users[user_id] = user
        self.logger.info(f"Created user {username} with role {role_name}")
        
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token"""
        user = self._find_user_by_username(username)
        if not user:
            self.logger.warning(f"Authentication failed: user {username} not found")
            return None
        
        # Check if account is locked due to failed attempts
        if user.failed_login_attempts >= self.max_failed_attempts:
            self.logger.warning(f"Account locked: {username} has too many failed attempts")
            return None
        
        # Verify password against stored hash
        if not self._verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            self.logger.warning(f"Authentication failed: invalid password for {username}")
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        
        # Generate session token
        session_token = self._generate_session_token(user)
        user.session_token = session_token
        user.session_expires = datetime.now() + timedelta(
            minutes=user.role.max_session_duration
        )
        
        self.active_sessions[session_token] = user.user_id
        
        self.logger.info(f"User {username} authenticated successfully")
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[User]:
        """Validate session token and return user"""
        if session_token not in self.active_sessions:
            return None
        
        user_id = self.active_sessions[session_token]
        user = self.users.get(user_id)
        
        if not user or not user.is_session_valid():
            # Clean up expired session
            if session_token in self.active_sessions:
                del self.active_sessions[session_token]
            return None
        
        return user
    
    def check_permission(self,
                        session_token: str,
                        permission: Permission,
                        document_metadata: Optional[Dict] = None) -> bool:
        """Check if user has permission for specific action"""
        user = self.validate_session(session_token)
        if not user:
            return False
        
        # Check basic permission
        if not user.can_perform_action(permission):
            return False
        
        # Check document-level access if metadata provided
        if document_metadata and not user.role.can_access_document(document_metadata):
            return False
        
        return True
    
    def filter_documents_by_access(self,
                                  session_token: str,
                                  documents: List[Dict]) -> List[Dict]:
        """Filter documents based on user access permissions"""
        user = self.validate_session(session_token)
        if not user:
            return []
        
        accessible_documents = []
        for doc in documents:
            if user.role.can_access_document(doc.get('metadata', {})):
                accessible_documents.append(doc)
        
        return accessible_documents
    
    def logout_user(self, session_token: str):
        """Log out user and invalidate session"""
        if session_token in self.active_sessions:
            user_id = self.active_sessions[session_token]
            user = self.users.get(user_id)
            
            if user:
                user.session_token = None
                user.session_expires = None
            
            del self.active_sessions[session_token]
            self.logger.info(f"User {user_id} logged out")
    
    def update_user_role(self, user_id: str, new_role_name: str):
        """Update user's role"""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        if new_role_name not in self.roles:
            raise ValueError(f"Role {new_role_name} does not exist")
        
        old_role = self.users[user_id].role.name
        self.users[user_id].role = self.roles[new_role_name]
        
        self.logger.info(f"Updated user {user_id} role from {old_role} to {new_role_name}")
    
    def deactivate_user(self, user_id: str):
        """Deactivate user account"""
        if user_id in self.users:
            self.users[user_id].is_active = False
            
            # Invalidate active sessions
            sessions_to_remove = [
                token for token, uid in self.active_sessions.items() 
                if uid == user_id
            ]
            for token in sessions_to_remove:
                del self.active_sessions[token]
            
            self.logger.info(f"Deactivated user {user_id}")
    
    def get_user_audit_info(self, user_id: str) -> Dict[str, Any]:
        """Get audit information for user"""
        user = self.users.get(user_id)
        if not user:
            return {}
        
        return {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.name,
            "permissions": [p.value for p in user.role.permissions],
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "session_active": user.is_session_valid(),
            "failed_attempts": user.failed_login_attempts
        }
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []
        
        for token, user_id in self.active_sessions.items():
            user = self.users.get(user_id)
            if not user or not user.is_session_valid():
                expired_sessions.append(token)
        
        for token in expired_sessions:
            del self.active_sessions[token]
        
        self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def _hash_password(self, password: str) -> str:
        """Hash password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        # Use proper password comparison
        expected_hash = self._hash_password(password)
        return expected_hash == stored_hash
    
    def _generate_session_token(self, user: User) -> str:
        """Generate JWT session token"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.name,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=user.role.max_session_duration)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")


# Example usage and testing
if __name__ == "__main__":
    # Test the access control system
    access_controller = AccessController()
    
    # Create test users
    admin_user = access_controller.create_user(
        "admin1", "admin", "admin@company.com", "admin", "admin_password"
    )
    
    employee_user = access_controller.create_user(
        "emp1", "john_doe", "john@company.com", "employee", "employee_password"
    )
    
    # Test authentication
    admin_token = access_controller.authenticate_user("admin", "admin_password")
    employee_token = access_controller.authenticate_user("john_doe", "employee_password")
    
    print(f"Admin authenticated: {admin_token is not None}")
    print(f"Employee authenticated: {employee_token is not None}")
    
    # Test permissions
    can_admin_delete = access_controller.check_permission(
        admin_token, Permission.DELETE_DOCUMENTS
    )
    can_employee_delete = access_controller.check_permission(
        employee_token, Permission.DELETE_DOCUMENTS
    )
    
    print(f"Admin can delete: {can_admin_delete}")
    print(f"Employee can delete: {can_employee_delete}")
    
    # Test document filtering
    test_documents = [
        {"id": "1", "metadata": {"access_level": "public"}},
        {"id": "2", "metadata": {"access_level": "internal"}},
        {"id": "3", "metadata": {"access_level": "confidential"}}
    ]
    
    employee_docs = access_controller.filter_documents_by_access(
        employee_token, test_documents
    )
    
    print(f"Employee can access {len(employee_docs)} of {len(test_documents)} documents")