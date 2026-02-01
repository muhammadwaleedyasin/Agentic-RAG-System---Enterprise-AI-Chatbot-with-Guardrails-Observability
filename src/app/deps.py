"""
Dependency injection module for FastAPI.

This module provides dependency functions that break circular imports between
main.py and API modules by accessing global singletons through FastAPI's
app.state mechanism.
"""

from typing import TYPE_CHECKING, Optional

from fastapi import Request, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..core.rag_pipeline import RAGPipeline
from ..security.access_control import AccessController

if TYPE_CHECKING:
    from ..app.main import ConnectionManager
    from ..security.access_control import User

# Initialize security scheme
security = HTTPBearer(auto_error=False)


def get_rag_pipeline(request: Request) -> RAGPipeline:
    """Get the RAG pipeline instance from app state."""
    return request.app.state.rag_pipeline


def get_access_controller(request: Request) -> AccessController:
    """Get the access controller instance from app state."""
    return request.app.state.access_controller


def get_connection_manager(request: Request) -> "ConnectionManager":
    """Get the WebSocket connection manager from app state."""
    return request.app.state.connection_manager


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> "User":
    """Get current authenticated user using HTTPBearer token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_controller = request.app.state.access_controller
    user = access_controller.validate_session(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user
