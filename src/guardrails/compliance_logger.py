"""
Compliance Audit Logger

Provides audit-ready logging for all guardrail activities, compliance
violations, and security events for enterprise governance requirements.
"""
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import os
from pathlib import Path

from ..config.compliance_config import get_compliance_config, AuditConfig


class EventType(Enum):
    """Types of compliance events"""
    PII_DETECTION = "pii_detection"
    CITATION_VIOLATION = "citation_violation"
    TOPIC_BLOCK = "topic_block"
    CONTENT_REDACTION = "content_redaction"
    POLICY_VIOLATION = "policy_violation"
    SECURITY_ALERT = "security_alert"
    ACCESS_DENIED = "access_denied"
    COMPLIANCE_CHECK = "compliance_check"
    AUDIT_EXPORT = "audit_export"


class Severity(Enum):
    """Event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents a compliance audit event"""
    event_id: str
    timestamp: datetime
    event_type: EventType
    severity: Severity
    user_id: Optional[str]
    session_id: Optional[str]
    query_hash: Optional[str]
    response_type: str
    compliance_status: str
    violations: List[str]
    confidence_score: Optional[float]
    metadata: Dict[str, Any]
    remediation_actions: List[str]
    data_classification: str = "internal"
    retention_period: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        event_dict = asdict(self)
        event_dict['timestamp'] = self.timestamp.isoformat()
        event_dict['event_type'] = self.event_type.value
        event_dict['severity'] = self.severity.value
        return event_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create event from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = EventType(data['event_type'])
        data['severity'] = Severity(data['severity'])
        return cls(**data)


class ComplianceLogger:
    """Enterprise compliance logger with audit capabilities"""
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or get_compliance_config().audit
        self.logger = logging.getLogger("compliance_audit")
        
        # Set up audit-specific logging
        self._setup_audit_logging()
        
        # Initialize audit storage
        self.audit_storage_path = Path("./audit_logs")
        self.audit_storage_path.mkdir(exist_ok=True)
        
        # Event buffer for batch processing
        self.event_buffer: List[AuditEvent] = []
        self.buffer_size = 100
    
    def _setup_audit_logging(self):
        """Set up specialized audit logging configuration"""
        # Create audit-specific logger
        audit_handler = logging.FileHandler("compliance_audit.log")
        audit_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Audit log format with structured data
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        
        self.logger.addHandler(audit_handler)
        self.logger.setLevel(getattr(logging, self.config.log_level))
    
    def log_pii_detection(self, 
                         entities_found: List[Dict],
                         user_id: Optional[str] = None,
                         session_id: Optional[str] = None,
                         query_text: Optional[str] = None,
                         remediation: str = "redacted") -> str:
        """Log PII detection event"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.PII_DETECTION,
            severity=self._assess_pii_severity(entities_found),
            user_id=user_id,
            session_id=session_id,
            query_hash=self._hash_text(query_text) if query_text else None,
            response_type="pii_detection",
            compliance_status="violation_detected",
            violations=[f"PII detected: {e.get('entity_type', 'unknown')}" for e in entities_found],
            confidence_score=max([e.get('confidence', 0) for e in entities_found], default=0),
            metadata={
                "pii_entities": [
                    {
                        "type": e.get('entity_type'),
                        "confidence": e.get('confidence'),
                        "position": f"{e.get('start_pos')}-{e.get('end_pos')}"
                    } for e in entities_found
                ],
                "remediation_action": remediation
            },
            remediation_actions=[f"Applied {remediation} to {len(entities_found)} PII entities"],
            data_classification="sensitive"
        )
        
        return self._record_event(event)
    
    def log_citation_violation(self,
                              violations: List[str],
                              citations_found: int,
                              citations_required: int,
                              user_id: Optional[str] = None,
                              session_id: Optional[str] = None) -> str:
        """Log citation policy violation"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.CITATION_VIOLATION,
            severity=Severity.MEDIUM if citations_found == 0 else Severity.LOW,
            user_id=user_id,
            session_id=session_id,
            query_hash=None,
            response_type="citation_check",
            compliance_status="policy_violation",
            violations=violations,
            confidence_score=citations_found / max(citations_required, 1),
            metadata={
                "citations_found": citations_found,
                "citations_required": citations_required,
                "citation_deficit": citations_required - citations_found
            },
            remediation_actions=["Citation enforcement applied", "Response modified"]
        )
        
        return self._record_event(event)
    
    def log_topic_filter(self,
                        blocked_topics: List[str],
                        topic_classifications: List[Dict],
                        action_taken: str,
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        query_text: Optional[str] = None) -> str:
        """Log topic filtering event"""
        severity = Severity.HIGH if blocked_topics else Severity.LOW
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.TOPIC_BLOCK,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            query_hash=self._hash_text(query_text) if query_text else None,
            response_type="topic_filter",
            compliance_status="blocked" if blocked_topics else "allowed",
            violations=[f"Blocked topic: {topic}" for topic in blocked_topics],
            confidence_score=max([t.get('confidence', 0) for t in topic_classifications], default=0),
            metadata={
                "blocked_topics": blocked_topics,
                "all_topics": topic_classifications,
                "action_taken": action_taken
            },
            remediation_actions=[f"Applied {action_taken} action for topic filtering"]
        )
        
        return self._record_event(event)
    
    def log_confidence_check(self,
                           confidence_score: float,
                           threshold: float,
                           abstained: bool,
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None) -> str:
        """Log confidence threshold check"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.COMPLIANCE_CHECK,
            severity=Severity.MEDIUM if abstained else Severity.LOW,
            user_id=user_id,
            session_id=session_id,
            query_hash=None,
            response_type="confidence_check",
            compliance_status="abstained" if abstained else "passed",
            violations=["Low confidence response"] if abstained else [],
            confidence_score=confidence_score,
            metadata={
                "confidence_score": confidence_score,
                "threshold": threshold,
                "abstained": abstained
            },
            remediation_actions=["Response abstained"] if abstained else ["Response approved"]
        )
        
        return self._record_event(event)
    
    def log_security_alert(self,
                          alert_type: str,
                          details: Dict[str, Any],
                          severity: Severity = Severity.HIGH,
                          user_id: Optional[str] = None) -> str:
        """Log security alert"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.SECURITY_ALERT,
            severity=severity,
            user_id=user_id,
            session_id=None,
            query_hash=None,
            response_type="security_alert",
            compliance_status="alert_raised",
            violations=[f"Security alert: {alert_type}"],
            confidence_score=None,
            metadata=details,
            remediation_actions=["Security team notified", "Incident response initiated"],
            data_classification="confidential"
        )
        
        return self._record_event(event)
    
    def _assess_pii_severity(self, entities: List[Dict]) -> Severity:
        """Assess severity based on PII entities detected"""
        high_risk_types = {"SSN", "CREDIT_CARD", "US_PASSPORT"}
        medium_risk_types = {"PHONE_NUMBER", "US_DRIVER_LICENSE", "EMAIL_ADDRESS"}
        
        entity_types = {e.get('entity_type', '') for e in entities}
        
        if entity_types & high_risk_types:
            return Severity.CRITICAL
        elif entity_types & medium_risk_types:
            return Severity.HIGH
        elif len(entities) > 3:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _hash_text(self, text: str) -> str:
        """Create privacy-preserving hash of text"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _record_event(self, event: AuditEvent) -> str:
        """Record audit event to storage and logs"""
        if not self.config.enabled:
            return event.event_id
        
        # Log to standard logging system
        self.logger.info(f"AUDIT: {event.event_type.value} - {event.compliance_status}")
        
        # Add to buffer for batch processing
        self.event_buffer.append(event)
        
        # Flush buffer if it's full
        if len(self.event_buffer) >= self.buffer_size:
            self._flush_buffer()
        
        # Store individual event
        self._store_event(event)
        
        return event.event_id
    
    def _store_event(self, event: AuditEvent):
        """Store individual event to persistent storage"""
        # Create daily audit file
        date_str = event.timestamp.strftime("%Y-%m-%d")
        audit_file = self.audit_storage_path / f"audit_{date_str}.jsonl"
        
        # Append event to daily audit log
        with open(audit_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
    
    def _flush_buffer(self):
        """Flush event buffer to storage"""
        if not self.event_buffer:
            return
        
        # Group events by date for efficient storage
        events_by_date = {}
        for event in self.event_buffer:
            date_str = event.timestamp.strftime("%Y-%m-%d")
            if date_str not in events_by_date:
                events_by_date[date_str] = []
            events_by_date[date_str].append(event)
        
        # Write to daily files
        for date_str, events in events_by_date.items():
            audit_file = self.audit_storage_path / f"audit_{date_str}.jsonl"
            with open(audit_file, "a") as f:
                for event in events:
                    f.write(json.dumps(event.to_dict()) + "\n")
        
        # Clear buffer
        self.event_buffer.clear()
    
    def export_audit_report(self, 
                           start_date: datetime,
                           end_date: datetime,
                           event_types: Optional[List[EventType]] = None,
                           severity_filter: Optional[List[Severity]] = None) -> Dict[str, Any]:
        """Export audit report for specified time period"""
        events = self._query_events(start_date, end_date, event_types, severity_filter)
        
        # Generate summary statistics
        summary = self._generate_audit_summary(events)
        
        # Log the export event
        export_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.AUDIT_EXPORT,
            severity=Severity.LOW,
            user_id=None,
            session_id=None,
            query_hash=None,
            response_type="audit_export",
            compliance_status="completed",
            violations=[],
            confidence_score=None,
            metadata={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "events_exported": len(events),
                "export_format": self.config.export_format
            },
            remediation_actions=["Audit report generated"]
        )
        self._record_event(export_event)
        
        return {
            "summary": summary,
            "events": [event.to_dict() for event in events],
            "export_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "period": f"{start_date.date()} to {end_date.date()}",
                "total_events": len(events),
                "format": self.config.export_format
            }
        }
    
    def _query_events(self,
                     start_date: datetime,
                     end_date: datetime,
                     event_types: Optional[List[EventType]] = None,
                     severity_filter: Optional[List[Severity]] = None) -> List[AuditEvent]:
        """Query events from storage"""
        events = []
        
        # Iterate through date range
        current_date = start_date.date()
        end_date_date = end_date.date()
        
        while current_date <= end_date_date:
            date_str = current_date.strftime("%Y-%m-%d")
            audit_file = self.audit_storage_path / f"audit_{date_str}.jsonl"
            
            if audit_file.exists():
                with open(audit_file, "r") as f:
                    for line in f:
                        try:
                            event_data = json.loads(line.strip())
                            event = AuditEvent.from_dict(event_data)
                            
                            # Apply filters
                            if event.timestamp < start_date or event.timestamp > end_date:
                                continue
                            
                            if event_types and event.event_type not in event_types:
                                continue
                            
                            if severity_filter and event.severity not in severity_filter:
                                continue
                            
                            events.append(event)
                        except (json.JSONDecodeError, KeyError) as e:
                            self.logger.warning(f"Error parsing audit event: {e}")
            
            current_date = current_date.replace(day=current_date.day + 1)
        
        return events
    
    def _generate_audit_summary(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate summary statistics for audit events"""
        if not events:
            return {"total_events": 0}
        
        # Count by event type
        event_type_counts = {}
        for event in events:
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for event in events:
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count violations
        total_violations = sum(len(event.violations) for event in events)
        
        # High-risk events
        high_risk_events = [e for e in events if e.severity in [Severity.HIGH, Severity.CRITICAL]]
        
        return {
            "total_events": len(events),
            "event_type_breakdown": event_type_counts,
            "severity_breakdown": severity_counts,
            "total_violations": total_violations,
            "high_risk_events": len(high_risk_events),
            "compliance_rate": (len(events) - total_violations) / len(events) if events else 1.0,
            "period_start": min(event.timestamp for event in events).isoformat(),
            "period_end": max(event.timestamp for event in events).isoformat()
        }
    
    def cleanup_old_logs(self, retention_days: Optional[int] = None):
        """Clean up old audit logs based on retention policy"""
        retention_days = retention_days or self.config.retention_days
        cutoff_date = datetime.now().date().replace(day=datetime.now().day - retention_days)
        
        # Find and remove old log files
        removed_files = 0
        for audit_file in self.audit_storage_path.glob("audit_*.jsonl"):
            try:
                date_str = audit_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                
                if file_date < cutoff_date:
                    audit_file.unlink()
                    removed_files += 1
                    self.logger.info(f"Removed old audit log: {audit_file}")
            except (ValueError, OSError) as e:
                self.logger.warning(f"Error processing audit file {audit_file}: {e}")
        
        self.logger.info(f"Cleanup completed: {removed_files} old audit files removed")


# Example usage and testing
if __name__ == "__main__":
    # Test the compliance logger
    logger = ComplianceLogger()
    
    # Test PII detection logging
    pii_entities = [
        {"entity_type": "SSN", "confidence": 0.95, "start_pos": 10, "end_pos": 21},
        {"entity_type": "EMAIL_ADDRESS", "confidence": 0.98, "start_pos": 30, "end_pos": 50}
    ]
    
    event_id = logger.log_pii_detection(
        entities_found=pii_entities,
        user_id="user123",
        session_id="session456",
        query_text="Please send my SSN to john@example.com"
    )
    print(f"PII detection logged with ID: {event_id}")
    
    # Test audit report export
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime.now()
    
    report = logger.export_audit_report(start_date, end_date)
    print(f"Audit report generated with {report['summary']['total_events']} events")