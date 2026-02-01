"""
Comprehensive Audit Logging System

Provides detailed security event logging, monitoring, and analysis capabilities
for compliance and security incident investigation.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json
import hashlib
import logging
import asyncio
import threading
from abc import ABC, abstractmethod
from queue import Queue, Empty
import uuid
import os
from pathlib import Path

class EventSeverity(Enum):
    """Security event severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventCategory(Enum):
    """Security event categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ACCESS = "system_access"
    POLICY_VIOLATION = "policy_violation"
    CONFIGURATION_CHANGE = "configuration_change"
    COMPLIANCE = "compliance"
    THREAT_DETECTION = "threat_detection"
    ANOMALY = "anomaly"

class EventStatus(Enum):
    """Event processing status"""
    PENDING = "pending"
    PROCESSED = "processed"
    ARCHIVED = "archived"
    FAILED = "failed"

@dataclass
class SecurityEvent:
    """Represents a security event for audit logging"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""
    category: EventCategory = EventCategory.SYSTEM_ACCESS
    severity: EventSeverity = EventSeverity.INFO
    
    # Actor information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Action details
    action: str = ""
    outcome: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Security context
    access_granted: bool = True
    policy_violations: List[str] = field(default_factory=list)
    threat_indicators: List[str] = field(default_factory=list)
    
    # Processing metadata
    status: EventStatus = EventStatus.PENDING
    processed_at: Optional[datetime] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate event checksum for integrity verification"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data"""
        # Create a stable representation for checksumming
        event_data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "category": self.category.value,
            "severity": self.severity.value,
            "user_id": self.user_id,
            "action": self.action,
            "outcome": self.outcome,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id
        }
        
        event_str = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum"""
        return self.checksum == self._calculate_checksum()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["processed_at"] = self.processed_at.isoformat() if self.processed_at else None
        result["category"] = self.category.value
        result["severity"] = self.severity.value
        result["status"] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityEvent':
        """Create event from dictionary"""
        # Convert string enums back to enum objects
        if "category" in data:
            data["category"] = EventCategory(data["category"])
        if "severity" in data:
            data["severity"] = EventSeverity(data["severity"])
        if "status" in data:
            data["status"] = EventStatus(data["status"])
        
        # Convert timestamp strings back to datetime
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if "processed_at" in data and isinstance(data["processed_at"], str):
            data["processed_at"] = datetime.fromisoformat(data["processed_at"])
        
        return cls(**data)

@dataclass
class AuditFilter:
    """Filter criteria for audit log queries"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    user_ids: List[str] = field(default_factory=list)
    event_types: List[str] = field(default_factory=list)
    categories: List[EventCategory] = field(default_factory=list)
    severities: List[EventSeverity] = field(default_factory=list)
    resource_types: List[str] = field(default_factory=list)
    resource_ids: List[str] = field(default_factory=list)
    access_granted: Optional[bool] = None
    has_violations: Optional[bool] = None
    ip_addresses: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    offset: int = 0

@dataclass
class AuditConfig:
    """Configuration for audit logging system"""
    # Storage configuration
    storage_backend: str = "file"  # file, database, remote
    storage_path: str = "audit_logs"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    retention_days: int = 365
    compression_enabled: bool = True
    
    # Real-time monitoring
    real_time_alerts: bool = True
    alert_thresholds: Dict[EventSeverity, int] = field(default_factory=lambda: {
        EventSeverity.CRITICAL: 1,
        EventSeverity.HIGH: 5,
        EventSeverity.MEDIUM: 50
    })
    
    # Performance settings
    batch_size: int = 100
    flush_interval: int = 30  # seconds
    async_processing: bool = True
    
    # Security settings
    encryption_enabled: bool = True
    integrity_checks: bool = True
    tamper_detection: bool = True
    
    # Compliance settings
    compliance_frameworks: List[str] = field(default_factory=lambda: ["SOX", "GDPR", "HIPAA"])
    required_fields: List[str] = field(default_factory=lambda: [
        "user_id", "action", "resource_id", "timestamp"
    ])

class AuditStorage(ABC):
    """Abstract base class for audit storage backends"""
    
    @abstractmethod
    async def store_event(self, event: SecurityEvent) -> bool:
        """Store a single audit event"""
        pass
    
    @abstractmethod
    async def store_events_batch(self, events: List[SecurityEvent]) -> int:
        """Store multiple events in batch"""
        pass
    
    @abstractmethod
    async def query_events(self, filter_criteria: AuditFilter) -> List[SecurityEvent]:
        """Query events based on filter criteria"""
        pass
    
    @abstractmethod
    async def get_event_count(self, filter_criteria: AuditFilter) -> int:
        """Get count of events matching filter"""
        pass
    
    @abstractmethod
    async def delete_old_events(self, cutoff_date: datetime) -> int:
        """Delete events older than cutoff date"""
        pass

class FileAuditStorage(AuditStorage):
    """File-based audit storage implementation"""
    
    def __init__(self, config: AuditConfig):
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.current_file_size = 0
        
    def _get_log_file_path(self, date: datetime) -> Path:
        """Get log file path for given date"""
        date_str = date.strftime("%Y-%m-%d")
        return self.storage_path / f"audit_{date_str}.jsonl"
    
    async def store_event(self, event: SecurityEvent) -> bool:
        """Store a single audit event"""
        try:
            log_file = self._get_log_file_path(event.timestamp)
            
            # Convert event to JSON
            event_json = json.dumps(event.to_dict()) + "\n"
            
            # Write to file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(event_json)
            
            return True
        except Exception as e:
            logging.error(f"Failed to store audit event: {e}")
            return False
    
    async def store_events_batch(self, events: List[SecurityEvent]) -> int:
        """Store multiple events in batch"""
        stored_count = 0
        
        # Group events by date
        events_by_date = {}
        for event in events:
            date_key = event.timestamp.date()
            if date_key not in events_by_date:
                events_by_date[date_key] = []
            events_by_date[date_key].append(event)
        
        # Write to respective files
        for date, date_events in events_by_date.items():
            try:
                log_file = self._get_log_file_path(datetime.combine(date, datetime.min.time()))
                
                with open(log_file, "a", encoding="utf-8") as f:
                    for event in date_events:
                        event_json = json.dumps(event.to_dict()) + "\n"
                        f.write(event_json)
                        stored_count += 1
            except Exception as e:
                logging.error(f"Failed to store events for {date}: {e}")
        
        return stored_count
    
    async def query_events(self, filter_criteria: AuditFilter) -> List[SecurityEvent]:
        """Query events based on filter criteria"""
        events = []
        
        # Determine date range to search
        start_date = filter_criteria.start_time.date() if filter_criteria.start_time else None
        end_date = filter_criteria.end_time.date() if filter_criteria.end_time else None
        
        # Get all log files in range
        log_files = []
        if start_date and end_date:
            current_date = start_date
            while current_date <= end_date:
                log_file = self._get_log_file_path(datetime.combine(current_date, datetime.min.time()))
                if log_file.exists():
                    log_files.append(log_file)
                current_date += timedelta(days=1)
        else:
            # Search all files
            log_files = list(self.storage_path.glob("audit_*.jsonl"))
        
        # Read and filter events
        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        
                        try:
                            event_data = json.loads(line.strip())
                            event = SecurityEvent.from_dict(event_data)
                            
                            if self._event_matches_filter(event, filter_criteria):
                                events.append(event)
                                
                                if (filter_criteria.limit and 
                                    len(events) >= filter_criteria.limit + filter_criteria.offset):
                                    break
                        except Exception as e:
                            logging.warning(f"Failed to parse event: {e}")
                            continue
            except Exception as e:
                logging.error(f"Failed to read log file {log_file}: {e}")
                continue
        
        # Apply offset and limit
        if filter_criteria.offset:
            events = events[filter_criteria.offset:]
        if filter_criteria.limit:
            events = events[:filter_criteria.limit]
        
        return events
    
    def _event_matches_filter(self, event: SecurityEvent, filter_criteria: AuditFilter) -> bool:
        """Check if event matches filter criteria"""
        # Time range
        if filter_criteria.start_time and event.timestamp < filter_criteria.start_time:
            return False
        if filter_criteria.end_time and event.timestamp > filter_criteria.end_time:
            return False
        
        # User IDs
        if filter_criteria.user_ids and event.user_id not in filter_criteria.user_ids:
            return False
        
        # Event types
        if filter_criteria.event_types and event.event_type not in filter_criteria.event_types:
            return False
        
        # Categories
        if filter_criteria.categories and event.category not in filter_criteria.categories:
            return False
        
        # Severities
        if filter_criteria.severities and event.severity not in filter_criteria.severities:
            return False
        
        # Resource types
        if filter_criteria.resource_types and event.resource_type not in filter_criteria.resource_types:
            return False
        
        # Resource IDs
        if filter_criteria.resource_ids and event.resource_id not in filter_criteria.resource_ids:
            return False
        
        # Access granted
        if filter_criteria.access_granted is not None and event.access_granted != filter_criteria.access_granted:
            return False
        
        # Has violations
        if filter_criteria.has_violations is not None:
            has_violations = len(event.policy_violations) > 0
            if has_violations != filter_criteria.has_violations:
                return False
        
        # IP addresses
        if filter_criteria.ip_addresses and event.ip_address not in filter_criteria.ip_addresses:
            return False
        
        return True
    
    async def get_event_count(self, filter_criteria: AuditFilter) -> int:
        """Get count of events matching filter"""
        events = await self.query_events(filter_criteria)
        return len(events)
    
    async def delete_old_events(self, cutoff_date: datetime) -> int:
        """Delete events older than cutoff date"""
        deleted_count = 0
        cutoff_date_only = cutoff_date.date()
        
        # Find old log files
        for log_file in self.storage_path.glob("audit_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                
                if file_date < cutoff_date_only:
                    # Count events before deletion
                    with open(log_file, "r", encoding="utf-8") as f:
                        event_count = sum(1 for line in f if line.strip())
                    
                    # Delete file
                    log_file.unlink()
                    deleted_count += event_count
                    logging.info(f"Deleted old audit log: {log_file}")
                    
            except Exception as e:
                logging.error(f"Failed to process log file {log_file}: {e}")
        
        return deleted_count

class AlertManager:
    """Manages real-time security alerts"""
    
    def __init__(self, config: AuditConfig):
        self.config = config
        self.alert_handlers: List[Callable[[SecurityEvent], None]] = []
        self.event_counters: Dict[EventSeverity, int] = {severity: 0 for severity in EventSeverity}
        self.counter_reset_time = datetime.utcnow()
        
    def add_alert_handler(self, handler: Callable[[SecurityEvent], None]):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)
    
    def check_alert_thresholds(self, event: SecurityEvent):
        """Check if event triggers alert thresholds"""
        if not self.config.real_time_alerts:
            return
        
        # Reset counters if hour has passed
        if datetime.utcnow() - self.counter_reset_time > timedelta(hours=1):
            self.event_counters = {severity: 0 for severity in EventSeverity}
            self.counter_reset_time = datetime.utcnow()
        
        # Increment counter
        self.event_counters[event.severity] += 1
        
        # Check threshold
        threshold = self.config.alert_thresholds.get(event.severity, float('inf'))
        if self.event_counters[event.severity] >= threshold:
            self._trigger_alert(event)
    
    def _trigger_alert(self, event: SecurityEvent):
        """Trigger security alert"""
        for handler in self.alert_handlers:
            try:
                handler(event)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")

class AuditLogger:
    """Main audit logging system"""
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self.storage = self._create_storage()
        self.alert_manager = AlertManager(self.config)
        
        # Event queue for async processing
        self.event_queue = Queue()
        self.processing_thread = None
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.events_logged = 0
        self.events_failed = 0
        
        if self.config.async_processing:
            self._start_processing_thread()
    
    def _create_storage(self) -> AuditStorage:
        """Create storage backend based on configuration"""
        if self.config.storage_backend == "file":
            return FileAuditStorage(self.config)
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage_backend}")
    
    def _start_processing_thread(self):
        """Start background thread for processing events"""
        self.processing_thread = threading.Thread(target=self._process_events)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_events(self):
        """Background thread for processing queued events"""
        events_batch = []
        last_flush = datetime.utcnow()
        
        while not self.shutdown_event.is_set():
            try:
                # Try to get event with timeout
                try:
                    event = self.event_queue.get(timeout=1.0)
                    events_batch.append(event)
                except Empty:
                    pass
                
                # Flush batch if size or time threshold reached
                current_time = datetime.utcnow()
                should_flush = (
                    len(events_batch) >= self.config.batch_size or
                    (events_batch and (current_time - last_flush).seconds >= self.config.flush_interval)
                )
                
                if should_flush:
                    asyncio.run(self._flush_events_batch(events_batch))
                    events_batch = []
                    last_flush = current_time
                    
            except Exception as e:
                logging.error(f"Error in audit processing thread: {e}")
        
        # Flush remaining events on shutdown
        if events_batch:
            asyncio.run(self._flush_events_batch(events_batch))
    
    async def _flush_events_batch(self, events: List[SecurityEvent]):
        """Flush batch of events to storage"""
        try:
            stored_count = await self.storage.store_events_batch(events)
            self.events_logged += stored_count
            self.events_failed += len(events) - stored_count
            
            # Mark events as processed
            for event in events:
                event.status = EventStatus.PROCESSED
                event.processed_at = datetime.utcnow()
                
        except Exception as e:
            logging.error(f"Failed to flush events batch: {e}")
            self.events_failed += len(events)
    
    def log_event(self, event: SecurityEvent):
        """Log a security event"""
        # Validate required fields
        if self.config.required_fields:
            event_dict = event.to_dict()
            for field in self.config.required_fields:
                if field not in event_dict or event_dict[field] is None:
                    logging.warning(f"Required field '{field}' missing in audit event")
        
        # Check alert thresholds
        self.alert_manager.check_alert_thresholds(event)
        
        if self.config.async_processing:
            self.event_queue.put(event)
        else:
            # Synchronous processing
            asyncio.run(self.storage.store_event(event))
            self.events_logged += 1
    
    def log_authentication_event(self, user_id: str, action: str, success: bool,
                                ip_address: str = None, details: Dict[str, Any] = None):
        """Log authentication event"""
        event = SecurityEvent(
            event_type="authentication",
            category=EventCategory.AUTHENTICATION,
            severity=EventSeverity.LOW if success else EventSeverity.MEDIUM,
            user_id=user_id,
            ip_address=ip_address,
            action=action,
            outcome="success" if success else "failure",
            access_granted=success,
            details=details or {}
        )
        self.log_event(event)
    
    def log_access_event(self, user_id: str, resource_type: str, resource_id: str,
                        action: str, granted: bool, ip_address: str = None,
                        violations: List[str] = None, details: Dict[str, Any] = None):
        """Log resource access event"""
        event = SecurityEvent(
            event_type="resource_access",
            category=EventCategory.DATA_ACCESS,
            severity=EventSeverity.LOW if granted else EventSeverity.MEDIUM,
            user_id=user_id,
            ip_address=ip_address,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome="granted" if granted else "denied",
            access_granted=granted,
            policy_violations=violations or [],
            details=details or {}
        )
        self.log_event(event)
    
    def log_policy_violation(self, user_id: str, policy_name: str, resource_id: str = None,
                           severity: EventSeverity = EventSeverity.HIGH,
                           details: Dict[str, Any] = None):
        """Log policy violation"""
        event = SecurityEvent(
            event_type="policy_violation",
            category=EventCategory.POLICY_VIOLATION,
            severity=severity,
            user_id=user_id,
            resource_id=resource_id,
            action="policy_check",
            outcome="violation",
            access_granted=False,
            policy_violations=[policy_name],
            details=details or {}
        )
        self.log_event(event)
    
    def log_anomaly(self, anomaly_type: str, severity: EventSeverity = EventSeverity.MEDIUM,
                   user_id: str = None, details: Dict[str, Any] = None):
        """Log security anomaly"""
        event = SecurityEvent(
            event_type="anomaly_detection",
            category=EventCategory.ANOMALY,
            severity=severity,
            user_id=user_id,
            action="anomaly_detection",
            outcome="detected",
            threat_indicators=[anomaly_type],
            details=details or {}
        )
        self.log_event(event)
    
    async def query_events(self, filter_criteria: AuditFilter) -> List[SecurityEvent]:
        """Query audit events"""
        return await self.storage.query_events(filter_criteria)
    
    async def get_event_statistics(self, time_range: timedelta = None) -> Dict[str, Any]:
        """Get audit event statistics"""
        if time_range is None:
            time_range = timedelta(days=7)
        
        start_time = datetime.utcnow() - time_range
        filter_criteria = AuditFilter(start_time=start_time)
        
        events = await self.storage.query_events(filter_criteria)
        
        # Calculate statistics
        stats = {
            "total_events": len(events),
            "events_by_category": {},
            "events_by_severity": {},
            "events_by_user": {},
            "access_denials": 0,
            "policy_violations": 0,
            "unique_users": set(),
            "unique_resources": set()
        }
        
        for event in events:
            # Category stats
            category_name = event.category.value
            stats["events_by_category"][category_name] = stats["events_by_category"].get(category_name, 0) + 1
            
            # Severity stats
            severity_name = event.severity.value
            stats["events_by_severity"][severity_name] = stats["events_by_severity"].get(severity_name, 0) + 1
            
            # User stats
            if event.user_id:
                stats["events_by_user"][event.user_id] = stats["events_by_user"].get(event.user_id, 0) + 1
                stats["unique_users"].add(event.user_id)
            
            # Resource stats
            if event.resource_id:
                stats["unique_resources"].add(event.resource_id)
            
            # Security metrics
            if not event.access_granted:
                stats["access_denials"] += 1
            
            if event.policy_violations:
                stats["policy_violations"] += len(event.policy_violations)
        
        # Convert sets to counts
        stats["unique_users"] = len(stats["unique_users"])
        stats["unique_resources"] = len(stats["unique_resources"])
        
        return stats
    
    async def cleanup_old_events(self, retention_days: int = None) -> int:
        """Clean up old audit events"""
        if retention_days is None:
            retention_days = self.config.retention_days
        
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        return await self.storage.delete_old_events(cutoff_date)
    
    def add_alert_handler(self, handler: Callable[[SecurityEvent], None]):
        """Add security alert handler"""
        self.alert_manager.add_alert_handler(handler)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logger statistics"""
        return {
            "events_logged": self.events_logged,
            "events_failed": self.events_failed,
            "queue_size": self.event_queue.qsize() if self.config.async_processing else 0,
            "processing_thread_active": self.processing_thread.is_alive() if self.processing_thread else False
        }
    
    def shutdown(self):
        """Gracefully shutdown audit logger"""
        if self.processing_thread:
            self.shutdown_event.set()
            self.processing_thread.join(timeout=10)
            
        logging.info("Audit logger shutdown complete")