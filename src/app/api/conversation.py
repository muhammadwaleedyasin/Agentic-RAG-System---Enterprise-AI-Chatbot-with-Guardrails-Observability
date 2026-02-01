"""
Conversation management API module.
Merged from src/api/conversation_routes.py into the canonical src/app/api/ structure.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...models.chat import (
    ChatMessage,
    MessageRole,
    ConversationSummary,
    ConversationHistory
)
from ...security.access_control import User, Permission
from ...memory.conversation_manager import conversation_manager
from ...core.observability_mixin import ObservabilityMixin
from ..deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


class ConversationObservability(ObservabilityMixin):
    """Observability mixin for conversation operations."""
    
    def __init__(self):
        super().__init__()
        self.component_name = "conversation_api"


observability = ConversationObservability()


# Request/Response Models
class ConversationCreateRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    conversation_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    metadata: Optional[Dict[str, Any]] = None


class MessageCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    metadata: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")


class ConversationSearchResponse(BaseModel):
    conversations: List[Dict[str, Any]]
    total: int
    query: str
    execution_time_ms: float


class ConversationStatsResponse(BaseModel):
    total_conversations: int
    total_messages: int
    avg_messages_per_conversation: float
    active_conversations_last_24h: int


class ConversationHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    conversation_count: int
    memory_usage_mb: float
    average_response_time_ms: float


# API Endpoints
@router.post(
    "/",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
    description="Create a new conversation with optional initial message"
)
async def create_conversation(
    conversation: ConversationCreateRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> ConversationResponse:
    """Create a new conversation."""
    try:
        # Check permissions
        if not current_user.can_perform_action(Permission.WRITE_DOCUMENTS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create conversations"
            )
        
        # Create conversation
        conversation_id = await conversation_manager.create_conversation(
            user_id=current_user.user_id,
            metadata=conversation.metadata
        )
        
        # Return response
        return ConversationResponse(
            conversation_id=conversation_id,
            user_id=current_user.user_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            message_count=0,
            metadata=conversation.metadata
        )
        
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a message to a conversation",
    description="Add a new message to an existing conversation"
)
async def add_message(
    conversation_id: str,
    message: MessageCreateRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> MessageResponse:
    """Add a message to a conversation."""
    try:
        # Check permissions
        if not current_user.can_perform_action(Permission.WRITE_DOCUMENTS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to add messages to this conversation"
            )
        
        # Add message
        chat_message = await conversation_manager.add_message(
            conversation_id=conversation_id,
            message=message.content,
            role=MessageRole.USER,
            user_id=current_user.user_id,
            metadata=message.metadata
        )
        
        return MessageResponse(
            role=chat_message.role,
            content=chat_message.content,
            timestamp=chat_message.timestamp,
            metadata=chat_message.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message to conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add message: {str(e)}"
        )


@router.get(
    "/{conversation_id}",
    response_model=ConversationHistory,
    summary="Get a conversation",
    description="Retrieve a conversation by ID with all messages"
)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    include_summary: bool = Query(True, description="Include summary in response")
) -> ConversationHistory:
    """Get a conversation by ID."""
    try:
        # Check permissions
        if not current_user.can_perform_action(Permission.READ_DOCUMENTS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access this conversation"
            )
        
        conversation_history = await conversation_manager.get_conversation_history(
            conversation_id=conversation_id,
            user_id=current_user.user_id,
            include_summary=include_summary
        )
        
        return conversation_history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


@router.get(
    "/",
    response_model=List[ConversationSummary],
    summary="List conversations",
    description="List all conversations for the current user"
)
async def list_conversations(
    current_user: User = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of conversations"),
    offset: int = Query(0, ge=0, description="Number of conversations to skip")
) -> List[ConversationSummary]:
    """List conversations for the current user."""
    try:
        conversations = await conversation_manager.get_user_conversations(
            user_id=current_user.user_id,
            limit=limit,
            offset=offset
        )
        
        return conversations
        
    except Exception as e:
        logger.error(f"Error listing conversations for user {current_user.user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}"
        )


@router.post(
    "/search",
    response_model=ConversationSearchResponse,
    summary="Search conversations",
    description="Search conversations by content or metadata"
)
async def search_conversations(
    search_request: ConversationSearchRequest,
    current_user: User = Depends(get_current_user)
) -> ConversationSearchResponse:
    """Search conversations."""
    try:
        start_time = datetime.now()
        
        # Check permissions
        if not current_user.can_perform_action(Permission.READ_DOCUMENTS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to search conversations"
            )
        
        # Perform search
        results = await conversation_manager.search_conversations(
            query=search_request.query,
            user_id=current_user.user_id,
            limit=search_request.limit
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ConversationSearchResponse(
            conversations=results,
            total=len(results),
            query=search_request.query,
            execution_time_ms=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search conversations: {str(e)}"
        )


@router.put(
    "/{conversation_id}/metadata",
    summary="Update conversation metadata",
    description="Update conversation title, description, or other metadata"
)
async def update_conversation_metadata(
    conversation_id: str,
    update_data: ConversationUpdateRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Update conversation metadata."""
    try:
        # Check permissions
        if not current_user.can_perform_action(Permission.WRITE_DOCUMENTS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to update this conversation"
            )
        
        # Update conversation
        success = await conversation_manager.update_conversation_metadata(
            conversation_id=conversation_id,
            metadata=update_data.dict(exclude_unset=True),
            user_id=current_user.user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {"success": True, "message": "Conversation updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update conversation: {str(e)}"
        )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
    description="Delete a conversation and all its messages"
)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Delete a conversation."""
    try:
        # Check permissions
        if not current_user.can_perform_action(Permission.DELETE_DOCUMENTS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to delete this conversation"
            )
        
        # Delete conversation
        success = await conversation_manager.delete_conversation(
            conversation_id=conversation_id,
            user_id=current_user.user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


@router.get(
    "/{conversation_id}/summary",
    response_model=Dict[str, Any],
    summary="Get conversation summary",
    description="Get an AI-generated summary of the conversation"
)
async def get_conversation_summary(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    regenerate: bool = Query(False, description="Force regeneration of summary")
) -> Dict[str, Any]:
    """Get conversation summary."""
    try:
        # Check permissions
        if not current_user.can_perform_action(Permission.READ_DOCUMENTS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access this conversation"
            )
        
        # Get conversation history with summary
        conversation_history = await conversation_manager.get_conversation_history(
            conversation_id=conversation_id,
            user_id=current_user.user_id,
            include_summary=True
        )
        
        # Extract summary from metadata
        summary_text = conversation_history.metadata.get("summary", "No summary available")
        
        return {
            "conversation_id": conversation_id,
            "summary": summary_text,
            "message_count": len(conversation_history.messages),
            "created_at": conversation_history.metadata.get("summary_created"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation summary {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation summary: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=ConversationStatsResponse,
    summary="Get conversation statistics",
    description="Get statistics about conversations for the current user"
)
async def get_conversation_stats(
    current_user: User = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365, description="Number of days to include in stats")
) -> ConversationStatsResponse:
    """Get conversation statistics."""
    try:
        # Check permissions
        if not current_user.can_perform_action(Permission.VIEW_ANALYTICS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view conversation statistics"
            )
        
        # Get user conversations
        conversations = await conversation_manager.get_user_conversations(
            user_id=current_user.user_id,
            limit=1000  # Get all for stats
        )
        
        # Calculate stats
        total_conversations = len(conversations)
        total_messages = sum(conv.message_count for conv in conversations)
        avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
        
        # Count active conversations in last 24h
        recent_cutoff = datetime.utcnow() - timedelta(days=1)
        active_recent = sum(1 for conv in conversations if conv.updated_at > recent_cutoff)
        
        return ConversationStatsResponse(
            total_conversations=total_conversations,
            total_messages=total_messages,
            avg_messages_per_conversation=avg_messages,
            active_conversations_last_24h=active_recent
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation statistics: {str(e)}"
        )


@router.get(
    "/health",
    response_model=ConversationHealthResponse,
    summary="Conversation system health check",
    description="Check the health and status of the conversation system"
)
async def conversation_health() -> ConversationHealthResponse:
    """Health check for conversation system."""
    try:
        metrics = conversation_manager.get_metrics()
        
        return ConversationHealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            conversation_count=metrics.get("active_conversations", 0),
            memory_usage_mb=metrics.get("performance_metrics", {}).get("memory_usage_mb", 0.0),
            average_response_time_ms=metrics.get("performance_metrics", {}).get("avg_response_time_ms", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Conversation health check failed: {str(e)}")
        return ConversationHealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            conversation_count=0,
            memory_usage_mb=0.0,
            average_response_time_ms=0.0
        )
