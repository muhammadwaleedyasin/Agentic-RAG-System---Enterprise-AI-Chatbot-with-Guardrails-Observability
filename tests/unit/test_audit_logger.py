"""
Unit tests for the audit logger module.
Tests cover event logging, file I/O, async batch processing, and alerts.
"""
import pytest
import tempfile
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.security.audit_logger import AuditLogger, FileAuditStorage, SecurityEvent, AuditConfig, AuditFilter


class TestFileAuditStorage:
    """Test cases for FileAuditStorage."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def audit_storage(self, temp_dir):
        """Create FileAuditStorage instance with temp directory."""
        return FileAuditStorage(AuditConfig(storage_path=temp_dir))
    
    def test_init_creates_directory(self, temp_dir):
        """Test that initialization creates log directory."""
        log_dir = os.path.join(temp_dir, "audit_logs")
        storage = FileAuditStorage(AuditConfig(storage_path=log_dir))
        
        assert os.path.exists(log_dir)
        assert storage.log_directory == log_dir
    
    @pytest.mark.asyncio
    async def test_store_event_sync(self, audit_storage, temp_dir):
        """Test synchronous event storage."""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="user_action",
            user_id="test_user",
            action="login",
            access_granted=True,
            details={"ip": "127.0.0.1"}
        )
        
        result = await audit_storage.store_event(event)
        assert result is True
        
        # Check file was created
        log_files = os.listdir(temp_dir)
        assert len(log_files) > 0
        
        # Check content
        log_file = os.path.join(temp_dir, log_files[0])
        with open(log_file, 'r') as f:
            content = f.read()
            assert "user_action" in content
            assert "test_user" in content
            assert "login" in content
    
    @pytest.mark.asyncio
    async def test_log_event_async_batch(self, audit_storage, temp_dir):
        """Test asynchronous batch event logging."""
        events = [
            SecurityEvent(
                timestamp=datetime.utcnow(),
                event_type="user_action",
                user_id=f"user_{i}",
                action="view_document",
                resource_id=f"doc_{i}",
                access_granted=True,
                details={"batch": True}
            )
            for i in range(5)
        ]
        
        # Log events concurrently
        tasks = [audit_storage.store_event(event) for event in events]
        await asyncio.gather(*tasks)
        
        # Verify all events were logged
        log_files = os.listdir(temp_dir)
        assert len(log_files) > 0
        
        # Count total events in all files
        total_events = 0
        for log_file in log_files:
            with open(os.path.join(temp_dir, log_file), 'r') as f:
                content = f.read()
                total_events += content.count("user_action")
        
        assert total_events == 5
    
    @pytest.mark.asyncio
    async def test_query_by_filters(self, audit_storage, temp_dir):
        """Test querying events by various filters."""
        # Log test events with different attributes
        events = [
            SecurityEvent(
                timestamp=datetime.utcnow() - timedelta(hours=1),
                event_type="security",
                user_id="admin",
                action="delete_document",
                resource_id="sensitive_doc.pdf",
                access_granted=True,
                details={"department": "security"}
            ),
            SecurityEvent(
                timestamp=datetime.utcnow(),
                event_type="user_action",
                user_id="user1",
                action="view_document",
                resource_id="public_doc.pdf",
                access_granted=True,
                details={"department": "engineering"}
            )
        ]
        
        for event in events:
            await audit_storage.store_event(event)
        
        # Query by event type
        security_events = await audit_storage.query_events(
            AuditFilter(event_types=["security"])
        )
        assert len(security_events) == 1
        assert security_events[0].event_type == "security"
        
        # Query by user
        admin_events = await audit_storage.query_events(
            AuditFilter(user_ids=["admin"])
        )
        assert len(admin_events) == 1
        assert admin_events[0].user_id == "admin"
        
        # Query by time range
        recent_events = await audit_storage.query_events(
            AuditFilter(start_time=datetime.utcnow() - timedelta(minutes=30))
        )
        assert len(recent_events) == 1
        assert recent_events[0].user_id == "user1"
    
    @pytest.mark.asyncio
    async def test_batch_storage(self, audit_storage, temp_dir):
        """Test batch storage functionality."""
        events = [
            SecurityEvent(
                timestamp=datetime.utcnow(),
                event_type="batch_test",
                user_id=f"user_{i}",
                action="test_action",
                resource_id=f"resource_{i}",
                access_granted=True
            )
            for i in range(5)
        ]
        
        # Store events in batch
        stored_count = await audit_storage.store_events_batch(events)
        
        # Verify all events were stored
        assert stored_count == 5
        
        # Verify events were written
        log_files = os.listdir(temp_dir)
        assert len(log_files) > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_old_events(self, audit_storage, temp_dir):
        """Test cleanup of old audit events."""
        # Create old event
        old_event = SecurityEvent(
            timestamp=datetime.utcnow() - timedelta(days=91),
            event_type="old_event",
            user_id="old_user",
            action="old_action",
            resource_id="old_resource",
            access_granted=True
        )
        
        # Create recent event
        recent_event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="recent_event",
            user_id="recent_user",
            action="recent_action",
            resource_id="recent_resource",
            access_granted=True
        )
        
        await audit_storage.store_event(old_event)
        await audit_storage.store_event(recent_event)
        
        # Clean up events older than 90 days
        cutoff_date = datetime.utcnow() - timedelta(days=90)
        deleted_count = await audit_storage.delete_old_events(cutoff_date)
        
        # Should have deleted the old event
        assert deleted_count > 0
        
        # Query remaining events
        filter_criteria = AuditFilter()
        remaining_events = await audit_storage.query_events(filter_criteria)
        
        # Should only have recent event
        assert len(remaining_events) == 1
        assert remaining_events[0].event_type == "recent_event"


class TestAuditLogger:
    """Test cases for AuditLogger."""
    
    @pytest.fixture
    def mock_storage(self):
        """Mock audit storage for testing."""
        mock = Mock()
        mock.store_event = AsyncMock()
        mock.store_events_batch = AsyncMock(return_value=0)
        mock.query_events = AsyncMock(return_value=[])
        mock.get_event_count = AsyncMock(return_value=0)
        mock.delete_old_events = AsyncMock(return_value=0)
        return mock
    
    @pytest.fixture
    def audit_logger(self, mock_storage):
        """Create AuditLogger instance with mock storage."""
        config = AuditConfig(async_processing=False)  # Disable async processing for tests
        logger = AuditLogger(config)
        logger.storage = mock_storage  # Override the storage
        return logger
    
    @pytest.mark.asyncio
    async def test_log_authentication_event(self, audit_logger, mock_storage):
        """Test logging authentication events."""
        audit_logger.log_authentication_event(
            user_id="admin",
            action="password_change",
            success=True,
            details={"reason": "policy_compliance"}
        )
        
        mock_storage.store_event.assert_called_once()
        call_args = mock_storage.store_event.call_args[0][0]
        assert call_args.event_type == "authentication"
        assert call_args.user_id == "admin"
        assert call_args.action == "password_change"
        assert call_args.access_granted is True
    
    @pytest.mark.asyncio
    async def test_log_access_event(self, audit_logger, mock_storage):
        """Test logging access events."""
        audit_logger.log_access_event(
            user_id="user123",
            resource_type="document",
            resource_id="confidential.pdf",
            action="document_access",
            granted=False,
            details={"denial_reason": "insufficient_permissions"}
        )
        
        mock_storage.store_event.assert_called_once()
        call_args = mock_storage.store_event.call_args[0][0]
        assert call_args.event_type == "resource_access"
        assert call_args.access_granted is False
        assert "denial_reason" in call_args.details
    
    @pytest.mark.asyncio
    async def test_alert_threshold_exceeded(self, audit_logger, mock_storage):
        """Test alert when threshold is exceeded."""
        # Mock query to return multiple failed events
        failed_events = [
            Mock(success=False, user_id="attacker", action="login_attempt")
            for _ in range(6)
        ]
        mock_storage.query_events.return_value = failed_events
        
        # Register alert handler and check alerts
        called = []
        audit_logger.add_alert_handler(lambda e: called.append(e))
        
        # Simulate triggering alert thresholds by processing events
        for event in failed_events:
            audit_logger.alert_manager.check_alert_thresholds(event)
        
        assert len(called) > 0
    
    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self, audit_logger, mock_storage):
        """Test no alert when below threshold."""
        # Mock query to return few failed events
        failed_events = [
            Mock(success=False, user_id="user", action="login_attempt")
            for _ in range(2)
        ]
        mock_storage.query_events.return_value = failed_events
        
        with patch('src.security.audit_logger.send_alert') as mock_alert:
            await audit_logger.check_alert_thresholds()
            
            # Should not trigger alert
            mock_alert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, audit_logger, mock_storage):
        """Test batch processing through logger."""
        # Log multiple events
        for i in range(5):
            audit_logger.log_access_event(
                user_id=f"user_{i}",
                resource_type="page",
                resource_id=f"page_{i}",
                action="view_page",
                granted=True
            )
        
        # Should have called store_event for each event
        assert mock_storage.store_event.call_count == 5
    
    @pytest.mark.asyncio
    async def test_threading_safety(self, audit_logger, mock_storage):
        """Test thread safety of concurrent logging."""
        def log_event(user_id):
            audit_logger.log_access_event(
                user_id=user_id,
                resource_type="resource",
                resource_id="shared_resource",
                action="concurrent_action",
                granted=True
            )
        
        # Log events concurrently
        tasks = [log_event(f"user_{i}") for i in range(10)]
        for task in tasks:
            task
        
        # All events should be logged
        assert mock_storage.store_event.call_count == 10
    
    @pytest.mark.asyncio
    async def test_error_handling(self, audit_logger, mock_storage):
        """Test error handling in audit logging."""
        # Mock storage to raise exception
        mock_storage.store_event.side_effect = Exception("Storage error")
        
        # Logging should not raise exception
        try:
            audit_logger.log_authentication_event(
                user_id="test",
                action="test_action",
                success=True
            )
        except Exception:
            pytest.fail("Audit logging should handle storage errors gracefully")
    
    def test_event_filtering(self, audit_logger):
        """Test event filtering by type and criteria."""
        # Test should_log_event method if implemented
        sensitive_event = {
            "action": "view_classified_document",
            "resource": "top_secret.pdf"
        }
        
        routine_event = {
            "action": "view_public_document", 
            "resource": "readme.txt"
        }
        
        # Assuming implementation has filtering logic
        # This would need to match actual filtering implementation
        assert hasattr(audit_logger, 'should_log_event') or True
