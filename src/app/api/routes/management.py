"""
Management and administrative endpoints for Enterprise RAG Chatbot.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel

from ....security.access_control import AccessController, Permission, User
from ....core.rag_pipeline import RAGPipeline
from ....config.settings import settings
from ...deps import get_current_user, get_access_controller, get_rag_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


class UserCreateRequest(BaseModel):
    username: str
    email: str
    role_name: str
    password: str


class UserUpdateRequest(BaseModel):
    role_name: Optional[str] = None
    is_active: Optional[bool] = None


class SystemStatsResponse(BaseModel):
    status: str
    message: str
    stats: Dict[str, Any]


class UserListResponse(BaseModel):
    status: str
    message: str
    users: List[Dict[str, Any]]
    total: int


# Admin permission check
async def require_admin(current_user: User = Depends(get_current_user)):
    """Require admin permissions."""
    if not current_user.can_perform_action(Permission.ADMIN_ACCESS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    current_user: User = Depends(require_admin),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    access_controller: AccessController = Depends(get_access_controller)
):
    """Get comprehensive system statistics."""
    try:
        # Get RAG pipeline stats
        rag_stats = await rag_pipeline.get_stats()
        
        # Get user stats
        user_stats = {
            "total_users": len(access_controller.users),
            "active_sessions": len(access_controller.active_sessions),
            "roles_distribution": {}
        }
        
        for user in access_controller.users.values():
            role_name = user.role.name
            user_stats["roles_distribution"][role_name] = user_stats["roles_distribution"].get(role_name, 0) + 1
        
        # System health
        health_status = await rag_pipeline.health_check()
        
        stats = {
            "rag_pipeline": rag_stats,
            "users": user_stats,
            "health": health_status,
            "configuration": {
                "llm_provider": settings.llm_provider.value,
                "debug_mode": settings.debug,
                "api_version": settings.app_version
            }
        }
        
        return SystemStatsResponse(
            status="success",
            message="System statistics retrieved successfully",
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system statistics: {str(e)}"
        )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    current_user: User = Depends(require_admin),
    access_controller: AccessController = Depends(get_access_controller),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """List all users with pagination."""
    try:
        users_list = []
        all_users = list(access_controller.users.values())
        
        # Apply pagination
        paginated_users = all_users[skip:skip + limit]
        
        for user in paginated_users:
            user_info = access_controller.get_user_audit_info(user.user_id)
            users_list.append(user_info)
        
        return UserListResponse(
            status="success",
            message="Users retrieved successfully",
            users=users_list,
            total=len(all_users)
        )
        
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}"
        )


@router.post("/users")
async def create_user(
    user_request: UserCreateRequest,
    current_user: User = Depends(require_admin),
    access_controller: AccessController = Depends(get_access_controller)
):
    """Create a new user account."""
    try:
        if not current_user.can_perform_action(Permission.MANAGE_USERS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User management permission required"
            )
        
        # Check if username already exists
        for user in access_controller.users.values():
            if user.username == user_request.username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists"
                )
        
        # Create user
        new_user = access_controller.create_user(
            user_id=user_request.username,  # Use username as ID for simplicity
            username=user_request.username,
            email=user_request.email,
            role_name=user_request.role_name,
            password=user_request.password
        )
        
        logger.info(f"User {user_request.username} created by {current_user.username}")
        
        return {
            "status": "success",
            "message": "User created successfully",
            "user": {
                "user_id": new_user.user_id,
                "username": new_user.username,
                "email": new_user.email,
                "role": new_user.role.name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )


@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    current_user: User = Depends(require_admin),
    access_controller: AccessController = Depends(get_access_controller)
):
    """Get user details by ID."""
    try:
        if user_id not in access_controller.users:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user_info = access_controller.get_user_audit_info(user_id)
        
        return {
            "status": "success",
            "message": "User details retrieved successfully",
            "user": user_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user details: {str(e)}"
        )


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    user_update: UserUpdateRequest,
    current_user: User = Depends(require_admin),
    access_controller: AccessController = Depends(get_access_controller)
):
    """Update user account."""
    try:
        if not current_user.can_perform_action(Permission.MANAGE_USERS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User management permission required"
            )
        
        if user_id not in access_controller.users:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update role if provided
        if user_update.role_name:
            access_controller.update_user_role(user_id, user_update.role_name)
            logger.info(f"User {user_id} role updated to {user_update.role_name} by {current_user.username}")
        
        # Update active status if provided
        if user_update.is_active is not None:
            if user_update.is_active:
                access_controller.users[user_id].is_active = True
            else:
                access_controller.deactivate_user(user_id)
            logger.info(f"User {user_id} active status updated to {user_update.is_active} by {current_user.username}")
        
        return {
            "status": "success",
            "message": "User updated successfully",
            "user": access_controller.get_user_audit_info(user_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_admin),
    access_controller: AccessController = Depends(get_access_controller)
):
    """Deactivate user account."""
    try:
        if not current_user.can_perform_action(Permission.MANAGE_USERS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User management permission required"
            )
        
        if user_id not in access_controller.users:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent self-deletion
        if user_id == current_user.user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )
        
        access_controller.deactivate_user(user_id)
        logger.info(f"User {user_id} deactivated by {current_user.username}")
        
        return {
            "status": "success",
            "message": "User deactivated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deactivate user: {str(e)}"
        )


@router.post("/sessions/cleanup")
async def cleanup_sessions(
    current_user: User = Depends(require_admin),
    access_controller: AccessController = Depends(get_access_controller)
):
    """Manually trigger session cleanup."""
    try:
        access_controller.cleanup_expired_sessions()
        
        return {
            "status": "success",
            "message": "Session cleanup completed",
            "active_sessions": len(access_controller.active_sessions)
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup sessions: {str(e)}"
        )


@router.get("/sessions")
async def list_active_sessions(
    current_user: User = Depends(require_admin),
    access_controller: AccessController = Depends(get_access_controller)
):
    """List all active sessions."""
    try:
        sessions = []
        for token, user_id in access_controller.active_sessions.items():
            user = access_controller.users.get(user_id)
            if user:
                sessions.append({
                    "user_id": user_id,
                    "username": user.username,
                    "role": user.role.name,
                    "session_expires": user.session_expires.isoformat() if user.session_expires else None,
                    "is_valid": user.is_session_valid()
                })
        
        return {
            "status": "success",
            "message": "Active sessions retrieved successfully",
            "sessions": sessions,
            "total": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}"
        )


@router.post("/system/reinitialize")
async def reinitialize_system(
    current_user: User = Depends(require_admin),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Reinitialize the RAG pipeline system."""
    try:
        if not current_user.can_perform_action(Permission.CONFIGURE_SYSTEM):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System configuration permission required"
            )
        
        # Reinitialize pipeline
        await rag_pipeline.initialize()
        
        logger.info(f"System reinitialized by {current_user.username}")
        
        return {
            "status": "success",
            "message": "System reinitialized successfully"
        }
        
    except Exception as e:
        logger.error(f"Error reinitializing system: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reinitialize system: {str(e)}"
        )
