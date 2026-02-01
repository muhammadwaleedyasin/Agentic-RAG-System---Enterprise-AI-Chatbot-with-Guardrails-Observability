"""
Document-Level Access Control System

Provides comprehensive role-based access control with document-level permissions,
metadata-based filtering, and hierarchical role management.
"""

from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
import logging
from abc import ABC, abstractmethod

class AccessLevel(Enum):
    """Document access levels"""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class ResourceType(Enum):
    """Types of resources that can be protected"""
    DOCUMENT = "document"
    COLLECTION = "collection"
    EMBEDDING = "embedding"
    METADATA = "metadata"
    SEARCH_RESULT = "search_result"

@dataclass
class Permission:
    """Represents a permission with resource and access level"""
    resource_type: ResourceType
    resource_id: str
    access_level: AccessLevel
    conditions: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    granted_by: Optional[str] = None
    granted_at: datetime = field(default_factory=datetime.utcnow)

    def is_expired(self) -> bool:
        """Check if permission has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def matches_resource(self, resource_type: ResourceType, resource_id: str) -> bool:
        """Check if permission applies to given resource"""
        if self.resource_type != resource_type:
            return False
        
        # Support wildcard patterns
        if "*" in self.resource_id:
            pattern = self.resource_id.replace("*", ".*")
            return bool(re.match(pattern, resource_id))
        
        return self.resource_id == resource_id

@dataclass
class Role:
    """User role with permissions and hierarchy"""
    name: str
    permissions: List[Permission] = field(default_factory=list)
    parent_roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_system_role: bool = False

    def add_permission(self, permission: Permission):
        """Add a permission to this role"""
        self.permissions.append(permission)

    def remove_permission(self, resource_type: ResourceType, resource_id: str):
        """Remove permissions matching resource"""
        self.permissions = [
            p for p in self.permissions 
            if not p.matches_resource(resource_type, resource_id)
        ]

@dataclass
class User:
    """User with roles and direct permissions"""
    user_id: str
    roles: List[str] = field(default_factory=list)
    direct_permissions: List[Permission] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_access: Optional[datetime] = None

    def add_role(self, role_name: str):
        """Add a role to user"""
        if role_name not in self.roles:
            self.roles.append(role_name)

    def remove_role(self, role_name: str):
        """Remove a role from user"""
        if role_name in self.roles:
            self.roles.remove(role_name)

@dataclass
class AccessContext:
    """Context information for access decisions"""
    user_id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    request_time: datetime = field(default_factory=datetime.utcnow)
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessPolicy:
    """Access policy with conditions and rules"""
    name: str
    resource_type: ResourceType
    resource_pattern: str
    required_access_level: AccessLevel
    conditions: Dict[str, Any] = field(default_factory=dict)
    deny_conditions: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    priority: int = 0  # Higher priority policies evaluated first

    def evaluate_conditions(self, context: AccessContext, resource_metadata: Dict[str, Any]) -> bool:
        """Evaluate policy conditions"""
        # Check allow conditions
        for condition_type, condition_value in self.conditions.items():
            if not self._evaluate_condition(condition_type, condition_value, context, resource_metadata):
                return False
        
        # Check deny conditions
        for condition_type, condition_value in self.deny_conditions.items():
            if self._evaluate_condition(condition_type, condition_value, context, resource_metadata):
                return False
        
        return True

    def _evaluate_condition(self, condition_type: str, condition_value: Any, 
                          context: AccessContext, resource_metadata: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        if condition_type == "time_range":
            current_time = context.request_time.time()
            start_time = datetime.strptime(condition_value["start"], "%H:%M").time()
            end_time = datetime.strptime(condition_value["end"], "%H:%M").time()
            return start_time <= current_time <= end_time
        
        elif condition_type == "ip_whitelist":
            return context.ip_address in condition_value
        
        elif condition_type == "metadata_match":
            for key, expected_value in condition_value.items():
                if resource_metadata.get(key) != expected_value:
                    return False
            return True
        
        elif condition_type == "classification_level":
            resource_level = resource_metadata.get("classification", "public")
            allowed_levels = condition_value
            return resource_level in allowed_levels
        
        return True

class AccessDecision(Enum):
    """Access decision results"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"

@dataclass
class AccessResult:
    """Result of access control evaluation"""
    decision: AccessDecision
    access_level: Optional[AccessLevel] = None
    reason: str = ""
    applied_policies: List[str] = field(default_factory=list)
    effective_permissions: List[Permission] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)

class DocumentAccessControl:
    """Main access control system for documents and resources"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.policies: List[AccessPolicy] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize default roles
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize system default roles"""
        # Super Admin role
        super_admin = Role(
            name="super_admin",
            permissions=[
                Permission(
                    resource_type=ResourceType.DOCUMENT,
                    resource_id="*",
                    access_level=AccessLevel.ADMIN
                ),
                Permission(
                    resource_type=ResourceType.COLLECTION,
                    resource_id="*",
                    access_level=AccessLevel.ADMIN
                )
            ],
            is_system_role=True
        )
        self.roles["super_admin"] = super_admin
        
        # Admin role
        admin = Role(
            name="admin",
            permissions=[
                Permission(
                    resource_type=ResourceType.DOCUMENT,
                    resource_id="*",
                    access_level=AccessLevel.WRITE
                ),
                Permission(
                    resource_type=ResourceType.COLLECTION,
                    resource_id="*",
                    access_level=AccessLevel.WRITE
                )
            ],
            parent_roles=["user"],
            is_system_role=True
        )
        self.roles["admin"] = admin
        
        # User role
        user_role = Role(
            name="user",
            permissions=[
                Permission(
                    resource_type=ResourceType.DOCUMENT,
                    resource_id="*",
                    access_level=AccessLevel.READ
                )
            ],
            is_system_role=True
        )
        self.roles["user"] = user_role
        
        # Guest role
        guest = Role(
            name="guest",
            permissions=[
                Permission(
                    resource_type=ResourceType.DOCUMENT,
                    resource_id="public/*",
                    access_level=AccessLevel.READ
                )
            ],
            is_system_role=True
        )
        self.roles["guest"] = guest

    def create_user(self, user_id: str, roles: List[str] = None, 
                   metadata: Dict[str, Any] = None) -> User:
        """Create a new user"""
        if user_id in self.users:
            raise ValueError(f"User {user_id} already exists")
        
        user = User(
            user_id=user_id,
            roles=roles or ["user"],
            metadata=metadata or {}
        )
        self.users[user_id] = user
        
        self.logger.info(f"Created user: {user_id} with roles: {roles}")
        return user

    def create_role(self, name: str, permissions: List[Permission] = None,
                   parent_roles: List[str] = None, metadata: Dict[str, Any] = None) -> Role:
        """Create a new role"""
        if name in self.roles:
            raise ValueError(f"Role {name} already exists")
        
        role = Role(
            name=name,
            permissions=permissions or [],
            parent_roles=parent_roles or [],
            metadata=metadata or {}
        )
        self.roles[name] = role
        
        self.logger.info(f"Created role: {name}")
        return role

    def add_policy(self, policy: AccessPolicy):
        """Add an access policy"""
        self.policies.append(policy)
        # Sort by priority (highest first)
        self.policies.sort(key=lambda p: p.priority, reverse=True)
        
        self.logger.info(f"Added policy: {policy.name}")

    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get all effective permissions for a user"""
        if user_id not in self.users:
            return []
        
        user = self.users[user_id]
        all_permissions = list(user.direct_permissions)
        
        # Get permissions from roles (including inherited)
        for role_name in user.roles:
            role_permissions = self._get_role_permissions(role_name)
            all_permissions.extend(role_permissions)
        
        # Remove expired permissions
        valid_permissions = [p for p in all_permissions if not p.is_expired()]
        
        return valid_permissions

    def _get_role_permissions(self, role_name: str, visited: Set[str] = None) -> List[Permission]:
        """Get permissions for a role including inherited permissions"""
        if visited is None:
            visited = set()
        
        if role_name in visited or role_name not in self.roles:
            return []
        
        visited.add(role_name)
        role = self.roles[role_name]
        permissions = list(role.permissions)
        
        # Add inherited permissions
        for parent_role in role.parent_roles:
            parent_permissions = self._get_role_permissions(parent_role, visited)
            permissions.extend(parent_permissions)
        
        return permissions

    def check_access(self, user_id: str, resource_type: ResourceType, 
                    resource_id: str, access_level: AccessLevel,
                    context: AccessContext = None, 
                    resource_metadata: Dict[str, Any] = None) -> AccessResult:
        """Check if user has access to resource"""
        if context is None:
            context = AccessContext(user_id=user_id)
        
        if resource_metadata is None:
            resource_metadata = {}
        
        # Update user last access
        if user_id in self.users:
            self.users[user_id].last_access = datetime.utcnow()
        
        # Check if user exists and is active
        if user_id not in self.users:
            return AccessResult(
                decision=AccessDecision.DENY,
                reason="User not found"
            )
        
        user = self.users[user_id]
        if not user.is_active:
            return AccessResult(
                decision=AccessDecision.DENY,
                reason="User account is inactive"
            )
        
        # Evaluate policies first
        policy_result = self._evaluate_policies(context, resource_type, resource_id, 
                                              access_level, resource_metadata)
        if policy_result.decision == AccessDecision.DENY:
            return policy_result
        
        # Check user permissions
        user_permissions = self.get_user_permissions(user_id)
        effective_permissions = []
        max_access_level = AccessLevel.NONE
        
        for permission in user_permissions:
            if permission.matches_resource(resource_type, resource_id):
                # Check permission conditions
                if self._check_permission_conditions(permission, context, resource_metadata):
                    effective_permissions.append(permission)
                    if self._access_level_covers(permission.access_level, access_level):
                        max_access_level = max(max_access_level, permission.access_level, key=lambda x: list(AccessLevel).index(x))
        
        # Determine final decision
        if self._access_level_covers(max_access_level, access_level):
            decision = AccessDecision.ALLOW
        else:
            decision = AccessDecision.DENY
        
        return AccessResult(
            decision=decision,
            access_level=max_access_level,
            reason=f"Access level {max_access_level.value} {'covers' if decision == AccessDecision.ALLOW else 'does not cover'} required {access_level.value}",
            effective_permissions=effective_permissions,
            applied_policies=[p.name for p in self.policies if p.is_active]
        )

    def _evaluate_policies(self, context: AccessContext, resource_type: ResourceType,
                          resource_id: str, access_level: AccessLevel,
                          resource_metadata: Dict[str, Any]) -> AccessResult:
        """Evaluate access policies"""
        applied_policies = []
        
        for policy in self.policies:
            if not policy.is_active:
                continue
            
            # Check if policy applies to this resource
            if policy.resource_type != resource_type:
                continue
            
            if not re.match(policy.resource_pattern.replace("*", ".*"), resource_id):
                continue
            
            applied_policies.append(policy.name)
            
            # Evaluate conditions
            if not policy.evaluate_conditions(context, resource_metadata):
                return AccessResult(
                    decision=AccessDecision.DENY,
                    reason=f"Policy {policy.name} conditions not met",
                    applied_policies=applied_policies
                )
        
        return AccessResult(
            decision=AccessDecision.ALLOW,
            reason="No blocking policies",
            applied_policies=applied_policies
        )

    def _check_permission_conditions(self, permission: Permission, 
                                   context: AccessContext, 
                                   resource_metadata: Dict[str, Any]) -> bool:
        """Check if permission conditions are met"""
        for condition_type, condition_value in permission.conditions.items():
            if condition_type == "ip_whitelist":
                if context.ip_address not in condition_value:
                    return False
            elif condition_type == "time_window":
                current_time = context.request_time.time()
                start_time = datetime.strptime(condition_value["start"], "%H:%M").time()
                end_time = datetime.strptime(condition_value["end"], "%H:%M").time()
                if not (start_time <= current_time <= end_time):
                    return False
        
        return True

    def _access_level_covers(self, granted_level: AccessLevel, required_level: AccessLevel) -> bool:
        """Check if granted access level covers required level"""
        access_hierarchy = {
            AccessLevel.NONE: 0,
            AccessLevel.READ: 1,
            AccessLevel.WRITE: 2,
            AccessLevel.DELETE: 3,
            AccessLevel.ADMIN: 4
        }
        
        return access_hierarchy[granted_level] >= access_hierarchy[required_level]

    def filter_documents_by_access(self, user_id: str, documents: List[Dict[str, Any]],
                                 access_level: AccessLevel = AccessLevel.READ,
                                 context: AccessContext = None) -> List[Dict[str, Any]]:
        """Filter documents based on user access permissions"""
        if context is None:
            context = AccessContext(user_id=user_id)
        
        filtered_documents = []
        
        for doc in documents:
            doc_id = doc.get("id", "")
            doc_metadata = doc.get("metadata", {})
            
            access_result = self.check_access(
                user_id=user_id,
                resource_type=ResourceType.DOCUMENT,
                resource_id=doc_id,
                access_level=access_level,
                context=context,
                resource_metadata=doc_metadata
            )
            
            if access_result.decision == AccessDecision.ALLOW:
                filtered_documents.append(doc)
        
        return filtered_documents

    def get_accessible_collections(self, user_id: str, 
                                 access_level: AccessLevel = AccessLevel.READ) -> List[str]:
        """Get list of collections accessible to user"""
        user_permissions = self.get_user_permissions(user_id)
        accessible_collections = set()
        
        for permission in user_permissions:
            if (permission.resource_type == ResourceType.COLLECTION and 
                self._access_level_covers(permission.access_level, access_level)):
                
                if "*" in permission.resource_id:
                    # For wildcard permissions, we'd need to query actual collections
                    # This is a simplified implementation
                    accessible_collections.add("*")
                else:
                    accessible_collections.add(permission.resource_id)
        
        return list(accessible_collections)

    def export_user_permissions(self, user_id: str) -> Dict[str, Any]:
        """Export user permissions for audit or backup"""
        if user_id not in self.users:
            return {}
        
        user = self.users[user_id]
        permissions = self.get_user_permissions(user_id)
        
        return {
            "user_id": user_id,
            "roles": user.roles,
            "direct_permissions": [
                {
                    "resource_type": p.resource_type.value,
                    "resource_id": p.resource_id,
                    "access_level": p.access_level.value,
                    "conditions": p.conditions,
                    "expires_at": p.expires_at.isoformat() if p.expires_at else None
                }
                for p in user.direct_permissions
            ],
            "effective_permissions": [
                {
                    "resource_type": p.resource_type.value,
                    "resource_id": p.resource_id,
                    "access_level": p.access_level.value,
                    "granted_by": p.granted_by
                }
                for p in permissions
            ],
            "metadata": user.metadata,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
            "last_access": user.last_access.isoformat() if user.last_access else None
        }

    def cleanup_expired_permissions(self):
        """Clean up expired permissions"""
        expired_count = 0
        
        for user in self.users.values():
            original_count = len(user.direct_permissions)
            user.direct_permissions = [p for p in user.direct_permissions if not p.is_expired()]
            expired_count += original_count - len(user.direct_permissions)
        
        for role in self.roles.values():
            if not role.is_system_role:  # Don't modify system roles
                original_count = len(role.permissions)
                role.permissions = [p for p in role.permissions if not p.is_expired()]
                expired_count += original_count - len(role.permissions)
        
        self.logger.info(f"Cleaned up {expired_count} expired permissions")
        return expired_count