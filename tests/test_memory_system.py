"""
Comprehensive tests for the memory system including Zep integration, conversation management,
and role-based access controls.
"""
import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.memory.zep_integration import ZepIntegration
from src.memory.conversation_manager import ConversationManager
from src.memory.memory_context import MemoryContextManager
from src.memory.role_scoping import RoleBasedMemoryScoping, MemoryScope, MemoryPermission
from src.models.chat import ChatMessage, MessageRole
from src.security.access_control import AccessController, Role
from src.utils.exceptions import MemoryError, AccessDeniedError


@pytest.fixture
def mock_zep_client():
    """Mock Zep client for testing."""
    client = Mock()
    client.memory = Mock()
    client.user = Mock()
    
    # Mock methods
    client.memory.get_sessions = AsyncMock(return_value=[])
    client.memory.add_session = AsyncMock()
    client.memory.get_session = AsyncMock()
    client.memory.add_memory = AsyncMock()
    client.memory.get_memory = AsyncMock()
    client.memory.search_memory = AsyncMock(return_value=[])
    client.memory.delete_session = AsyncMock()
    client.memory.update_session = AsyncMock()
    
    client.user.get = AsyncMock()
    client.user.add = AsyncMock()
    client.user.get_sessions = AsyncMock(return_value=[])
    
    return client


@pytest.fixture
def zep_integration(mock_zep_client):
    """Zep integration instance for testing."""
    integration = ZepIntegration("http://test:8000", "test-key")
    integration._client = mock_zep_client
    integration._connection_healthy = True
    return integration


@pytest.fixture
def conversation_manager():
    """Conversation manager instance for testing."""
    return ConversationManager()


@pytest.fixture
def memory_context_manager():
    """Memory context manager instance for testing."""
    return MemoryContextManager()


@pytest.fixture
def access_controller():
    """Access controller for testing."""
    controller = AccessController("test-secret-key")
    
    # Create test roles
    admin_role = Role("admin", [])
    user_role = Role("user", [])
    
    # Create test users
    controller.create_user("admin1", "admin", "admin@test.com", "admin", "password")
    controller.create_user("user1", "user", "user@test.com", "user", "password")
    
    return controller


@pytest.fixture
def role_scoping(access_controller):
    """Role-based memory scoping instance for testing."""
    return RoleBasedMemoryScoping(access_controller)


class TestZepIntegration:
    """Test Zep integration functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, zep_integration):
        """Test Zep integration initialization."""
        assert zep_integration._client is not None
        assert zep_integration._connection_healthy is True
    
    @pytest.mark.asyncio
    async def test_create_session(self, zep_integration, mock_zep_client):
        """Test creating a new session."""
        session_id = "test-session-123"
        user_id = "test-user-456"
        
        # Mock user creation
        mock_user = Mock()
        mock_user.user_id = user_id
        mock_zep_client.user.get.side_effect = Exception("User not found")
        mock_zep_client.user.add.return_value = mock_user
        
        # Test session creation
        session = await zep_integration.create_session(session_id, user_id)
        
        assert mock_zep_client.user.add.called
        assert mock_zep_client.memory.add_session.called
    
    @pytest.mark.asyncio
    async def test_add_message(self, zep_integration, mock_zep_client):
        """Test adding a message to session."""
        session_id = "test-session-123"
        message = "Hello, this is a test message"
        role = "user"
        
        mock_memory = Mock()
        mock_zep_client.memory.add_memory.return_value = mock_memory
        
        result = await zep_integration.add_message(session_id, message, role)
        
        assert mock_zep_client.memory.add_memory.called
        assert result == mock_memory
    
    @pytest.mark.asyncio
    async def test_get_memory(self, zep_integration, mock_zep_client):
        """Test retrieving conversation memory."""
        session_id = "test-session-123"
        
        mock_memory = Mock()
        mock_memory.messages = []
        mock_zep_client.memory.get_memory.return_value = mock_memory
        
        result = await zep_integration.get_memory(session_id)
        
        assert mock_zep_client.memory.get_memory.called
        assert result == mock_memory
    
    @pytest.mark.asyncio
    async def test_search_memory(self, zep_integration, mock_zep_client):
        """Test searching conversation memory."""
        session_id = "test-session-123"
        query = "test query"
        
        mock_results = [Mock(), Mock()]
        mock_zep_client.memory.search_memory.return_value = mock_results
        
        results = await zep_integration.search_memory(session_id, query)
        
        assert mock_zep_client.memory.search_memory.called
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, zep_integration, mock_zep_client):
        """Test retry mechanism for failed operations."""
        session_id = "test-session-123"
        
        # Mock consecutive failures then success
        mock_zep_client.memory.get_memory.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            Mock()  # Success on third try
        ]
        
        result = await zep_integration.get_memory(session_id)
        
        assert mock_zep_client.memory.get_memory.call_count == 3
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, zep_integration, mock_zep_client):
        """Test cleanup of expired sessions."""
        # Mock old sessions
        old_session = Mock()
        old_session.session_id = "old-session"
        old_session.created_at = datetime.utcnow() - timedelta(days=35)
        
        recent_session = Mock()
        recent_session.session_id = "recent-session"
        recent_session.created_at = datetime.utcnow() - timedelta(days=5)
        
        mock_zep_client.memory.get_sessions.return_value = [old_session, recent_session]
        mock_zep_client.memory.delete_session.return_value = True
        
        cleanup_count = await zep_integration.cleanup_expired_sessions(30)
        
        assert cleanup_count == 1
        mock_zep_client.memory.delete_session.assert_called_once_with("old-session")


class TestConversationManager:
    """Test conversation manager functionality."""
    
    @pytest.mark.asyncio
    async def test_create_conversation(self, conversation_manager):
        """Test creating a new conversation."""
        user_id = "test-user-123"
        
        with patch.object(conversation_manager, 'zep_integration') as mock_zep:
            mock_zep.create_session = AsyncMock()
            
            conv_id = await conversation_manager.create_conversation(user_id)
            
            assert conv_id.startswith("conv_")
            assert len(conv_id) > 10
            mock_zep.create_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_message(self, conversation_manager):
        """Test adding a message to conversation."""
        conv_id = "test-conv-123"
        user_id = "test-user-123"
        message = "Hello world"
        role = MessageRole.USER
        
        with patch('src.memory.conversation_manager.zep_integration') as mock_zep:
            mock_zep.get_memory = AsyncMock(return_value=Mock())
            mock_zep.add_message = AsyncMock()
            mock_zep.update_session_metadata = AsyncMock()
            
            chat_message = await conversation_manager.add_message(
                conv_id, message, role, user_id
            )
            
            assert chat_message.content == message
            assert chat_message.role == role
            mock_zep.add_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, conversation_manager):
        """Test retrieving conversation history."""
        conv_id = "test-conv-123"
        user_id = "test-user-123"
        
        # Mock Zep memory response
        mock_message = Mock()
        mock_message.role = "user"
        mock_message.content = "Test message"
        mock_message.metadata = {}
        
        mock_memory = Mock()
        mock_memory.messages = [mock_message]
        mock_memory.summary = None
        
        with patch('src.memory.conversation_manager.zep_integration') as mock_zep:
            mock_zep.get_memory = AsyncMock(return_value=mock_memory)
            
            history = await conversation_manager.get_conversation_history(conv_id, user_id)
            
            assert history.conversation_id == conv_id
            assert len(history.messages) == 1
            assert history.messages[0].content == "Test message"
    
    @pytest.mark.asyncio
    async def test_search_conversations(self, conversation_manager):
        """Test searching conversations."""
        query = "test search query"
        user_id = "test-user-123"
        
        # Mock search results
        mock_result = Mock()
        mock_result.message = Mock()
        mock_result.message.content = "Found message"
        mock_result.message.metadata = {}
        mock_result.dist = 0.8
        
        with patch('src.memory.conversation_manager.zep_integration') as mock_zep:
            mock_zep.get_session_list = AsyncMock(return_value=[])
            
            results = await conversation_manager.search_conversations(query, user_id)
            
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_delete_conversation(self, conversation_manager):
        """Test deleting a conversation."""
        conv_id = "test-conv-123"
        user_id = "test-user-123"
        
        with patch('src.memory.conversation_manager.zep_integration') as mock_zep:
            mock_zep.delete_session = AsyncMock(return_value=True)
            
            success = await conversation_manager.delete_conversation(conv_id, user_id)
            
            assert success is True
            mock_zep.delete_session.assert_called_once_with(conv_id)


class TestMemoryContextManager:
    """Test memory context manager functionality."""
    
    @pytest.mark.asyncio
    async def test_get_context(self, memory_context_manager):
        """Test getting memory context."""
        conv_id = "test-conv-123"
        user_id = "test-user-123"
        
        with patch('src.memory.memory_context.zep_integration') as mock_zep:
            mock_zep.get_memory = AsyncMock(return_value=None)
            
            context = await memory_context_manager.get_context(conv_id, user_id)
            
            assert context.conversation_id == conv_id
            assert context.user_id == user_id
            assert len(context.current_window.messages) == 0
    
    @pytest.mark.asyncio
    async def test_update_context(self, memory_context_manager):
        """Test updating context with new message."""
        conv_id = "test-conv-123"
        user_id = "test-user-123"
        
        new_message = ChatMessage(
            role=MessageRole.USER,
            content="New test message"
        )
        
        with patch('src.memory.memory_context.zep_integration') as mock_zep:
            mock_zep.get_memory = AsyncMock(return_value=None)
            
            context = await memory_context_manager.update_context(
                conv_id, user_id, new_message
            )
            
            assert len(context.current_window.messages) == 1
            assert context.current_window.messages[0].content == "New test message"
    
    @pytest.mark.asyncio
    async def test_generate_summary(self, memory_context_manager):
        """Test generating conversation summary."""
        conv_id = "test-conv-123"
        user_id = "test-user-123"
        
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
            ChatMessage(role=MessageRole.USER, content="How are you?"),
            ChatMessage(role=MessageRole.ASSISTANT, content="I'm doing well, thanks!")
        ]
        
        summary = await memory_context_manager.generate_summary(
            conv_id, user_id, messages
        )
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @pytest.mark.asyncio
    async def test_get_relevant_context(self, memory_context_manager):
        """Test getting relevant context for a query."""
        conv_id = "test-conv-123"
        user_id = "test-user-123"
        query = "test query"
        
        with patch('src.memory.memory_context.zep_integration') as mock_zep:
            mock_zep.get_memory = AsyncMock(return_value=None)
            mock_zep.search_memory = AsyncMock(return_value=[])
            
            messages, summary = await memory_context_manager.get_relevant_context(
                conv_id, user_id, query
            )
            
            assert isinstance(messages, list)
            assert summary is None or isinstance(summary, str)


class TestRoleBasedMemoryScoping:
    """Test role-based memory scoping functionality."""
    
    def test_get_user_memory_scope(self, role_scoping):
        """Test getting user memory scope based on role."""
        scope = role_scoping.get_user_memory_scope("user1", "admin")
        assert scope == MemoryScope.ADMIN
        
        scope = role_scoping.get_user_memory_scope("user2", "user")
        assert scope == MemoryScope.PRIVATE
    
    def test_get_user_memory_permissions(self, role_scoping):
        """Test getting user memory permissions."""
        permissions = role_scoping.get_user_memory_permissions("user1", "admin")
        assert MemoryPermission.ADMIN_ACCESS in permissions
        assert MemoryPermission.READ_ALL in permissions
        
        permissions = role_scoping.get_user_memory_permissions("user2", "user")
        assert MemoryPermission.READ_OWN in permissions
        assert MemoryPermission.ADMIN_ACCESS not in permissions
    
    @pytest.mark.asyncio
    async def test_check_conversation_access(self, role_scoping):
        """Test checking conversation access permissions."""
        # Test own conversation access
        has_access = await role_scoping.check_conversation_access(
            user_id="user1",
            conversation_id="conv123",
            action=MemoryPermission.READ_OWN,
            conversation_owner="user1"
        )
        assert has_access is True
        
        # Test accessing another user's conversation without permission
        has_access = await role_scoping.check_conversation_access(
            user_id="user1",
            conversation_id="conv123",
            action=MemoryPermission.READ_OWN,
            conversation_owner="user2"
        )
        assert has_access is False
    
    def test_create_access_rule(self, role_scoping):
        """Test creating custom access rules."""
        rule = role_scoping.create_access_rule(
            user_id="user1",
            scope=MemoryScope.TEAM,
            permissions=[MemoryPermission.READ_TEAM, MemoryPermission.WRITE_TEAM]
        )
        
        assert rule.scope == MemoryScope.TEAM
        assert MemoryPermission.READ_TEAM in rule.permissions
        assert MemoryPermission.WRITE_TEAM in rule.permissions
    
    def test_filter_conversations_by_access(self, role_scoping):
        """Test filtering conversations by access permissions."""
        conversations = [
            {"conversation_id": "conv1", "user_id": "user1", "metadata": {}},
            {"conversation_id": "conv2", "user_id": "user2", "metadata": {}},
            {"conversation_id": "conv3", "user_id": "user1", "metadata": {}}
        ]
        
        filtered = role_scoping.filter_conversations_by_access(
            user_id="user1",
            conversations=conversations,
            action=MemoryPermission.READ_OWN
        )
        
        # User should only see their own conversations
        accessible_ids = {conv["conversation_id"] for conv in filtered}
        assert "conv1" in accessible_ids
        assert "conv3" in accessible_ids
    
    def test_cleanup_expired_rules(self, role_scoping):
        """Test cleanup of expired access rules."""
        # Create expired rule
        expired_rule = role_scoping.create_access_rule(
            user_id="user1",
            scope=MemoryScope.TEAM,
            permissions=[MemoryPermission.READ_TEAM],
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        
        # Create valid rule
        valid_rule = role_scoping.create_access_rule(
            user_id="user1",
            scope=MemoryScope.TEAM,
            permissions=[MemoryPermission.WRITE_TEAM],
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        cleanup_count = role_scoping.cleanup_expired_rules()
        
        assert cleanup_count == 1


class TestMemorySystemIntegration:
    """Test integration between memory system components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_conversation_flow(self, conversation_manager, memory_context_manager):
        """Test complete conversation flow from creation to context management."""
        user_id = "test-user-123"
        
        with patch('src.memory.conversation_manager.zep_integration') as mock_zep, \
             patch('src.memory.memory_context.zep_integration') as mock_zep_context:
            
            # Mock Zep responses
            mock_zep.create_session = AsyncMock()
            mock_zep.add_message = AsyncMock()
            mock_zep.get_memory = AsyncMock(return_value=None)
            mock_zep.update_session_metadata = AsyncMock()
            mock_zep_context.get_memory = AsyncMock(return_value=None)
            
            # Create conversation
            conv_id = await conversation_manager.create_conversation(user_id)
            assert conv_id is not None
            
            # Add messages
            message1 = await conversation_manager.add_message(
                conv_id, "Hello", MessageRole.USER, user_id
            )
            message2 = await conversation_manager.add_message(
                conv_id, "Hi there!", MessageRole.ASSISTANT, user_id
            )
            
            assert message1.content == "Hello"
            assert message2.content == "Hi there!"
            
            # Update context
            context = await memory_context_manager.update_context(
                conv_id, user_id, message1
            )
            context = await memory_context_manager.update_context(
                conv_id, user_id, message2
            )
            
            assert len(context.current_window.messages) == 2
    
    @pytest.mark.asyncio
    async def test_memory_access_control_integration(self, conversation_manager, role_scoping):
        """Test integration between conversation manager and access control."""
        admin_user = "admin1"
        regular_user = "user1"
        
        with patch('src.memory.conversation_manager.zep_integration') as mock_zep:
            mock_zep.create_session = AsyncMock()
            mock_zep.delete_session = AsyncMock(return_value=True)
            
            # Create conversation as regular user
            conv_id = await conversation_manager.create_conversation(regular_user)
            
            # Check admin can access conversation
            admin_access = await role_scoping.check_conversation_access(
                user_id=admin_user,
                conversation_id=conv_id,
                action=MemoryPermission.READ_ALL,
                conversation_owner=regular_user
            )
            assert admin_access is True
            
            # Check regular user cannot access admin functions
            user_admin_access = await role_scoping.check_conversation_access(
                user_id=regular_user,
                conversation_id=conv_id,
                action=MemoryPermission.DELETE_ALL,
                conversation_owner=regular_user
            )
            assert user_admin_access is False


class TestMemorySystemPerformance:
    """Test memory system performance and error handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_conversation_creation(self, conversation_manager):
        """Test creating multiple conversations concurrently."""
        user_id = "test-user-123"
        num_conversations = 10
        
        with patch('src.memory.conversation_manager.zep_integration') as mock_zep:
            mock_zep.create_session = AsyncMock()
            
            # Create conversations concurrently
            tasks = [
                conversation_manager.create_conversation(user_id)
                for _ in range(num_conversations)
            ]
            
            conv_ids = await asyncio.gather(*tasks)
            
            assert len(conv_ids) == num_conversations
            assert len(set(conv_ids)) == num_conversations  # All unique
    
    @pytest.mark.asyncio
    async def test_error_handling_zep_failure(self, zep_integration):
        """Test error handling when Zep service fails."""
        session_id = "test-session-123"
        
        # Mock Zep failure
        zep_integration._client.memory.get_memory.side_effect = Exception("Zep service unavailable")
        
        with pytest.raises(MemoryError):
            await zep_integration.get_memory(session_id)
    
    @pytest.mark.asyncio
    async def test_memory_context_optimization(self, memory_context_manager):
        """Test memory context optimization under load."""
        conv_id = "test-conv-123"
        user_id = "test-user-123"
        
        # Create many messages to trigger optimization
        messages = [
            ChatMessage(role=MessageRole.USER, content=f"Message {i}")
            for i in range(100)
        ]
        
        with patch('src.memory.memory_context.zep_integration') as mock_zep:
            mock_zep.get_memory = AsyncMock(return_value=None)
            
            context = await memory_context_manager.get_context(conv_id, user_id)
            
            # Add all messages
            for message in messages:
                context = await memory_context_manager.update_context(
                    conv_id, user_id, message
                )
            
            # Context should be optimized (messages summarized)
            assert len(context.current_window.messages) <= memory_context_manager.memory_window_size * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])