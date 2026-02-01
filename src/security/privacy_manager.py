"""
Privacy Management System

Manages data classification, privacy controls, and data retention policies
for enterprise RAG systems with GDPR and privacy compliance features.
"""
import logging
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import re

from .data_encryption import EncryptionManager


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class RetentionPolicy(Enum):
    """Data retention policies"""
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 1 year
    LONG_TERM = "long_term"  # 7 years
    PERMANENT = "permanent"
    CUSTOM = "custom"


@dataclass
class DataSubject:
    """Represents a data subject for privacy management"""
    subject_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    data_sources: List[str] = field(default_factory=list)
    consent_status: Dict[str, bool] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PrivacyRecord:
    """Privacy record for data processing activities"""
    record_id: str
    data_subject_id: str
    data_type: str
    classification: DataClassification
    purpose: str
    legal_basis: str
    retention_policy: RetentionPolicy
    retention_period: Optional[timedelta] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrivacyRequest:
    """Privacy request (access, deletion, portability, etc.)"""
    request_id: str
    subject_id: str
    request_type: str
    status: str
    submitted_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)


class PrivacyManager:
    """Manages privacy compliance and data protection"""
    
    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        self.logger = logging.getLogger(__name__)
        self.encryption_manager = encryption_manager or EncryptionManager()
        
        # Privacy storage
        self.data_subjects: Dict[str, DataSubject] = {}
        self.privacy_records: Dict[str, PrivacyRecord] = {}
        self.privacy_requests: Dict[str, PrivacyRequest] = {}
        
        # Retention periods by policy
        self.retention_periods = {
            RetentionPolicy.SHORT_TERM: timedelta(days=30),
            RetentionPolicy.MEDIUM_TERM: timedelta(days=365),
            RetentionPolicy.LONG_TERM: timedelta(days=365 * 7),
            RetentionPolicy.PERMANENT: None
        }
        
        # Data classification rules
        self.classification_rules = {
            "pii": DataClassification.CONFIDENTIAL,
            "financial": DataClassification.CONFIDENTIAL,
            "health": DataClassification.RESTRICTED,
            "legal": DataClassification.CONFIDENTIAL,
            "public": DataClassification.PUBLIC,
            "internal": DataClassification.INTERNAL
        }
        
        # Load existing data
        self._load_privacy_data()
    
    def register_data_subject(self,
                            subject_id: str,
                            email: Optional[str] = None,
                            name: Optional[str] = None) -> DataSubject:
        """Register a new data subject"""
        if subject_id in self.data_subjects:
            return self.data_subjects[subject_id]
        
        data_subject = DataSubject(
            subject_id=subject_id,
            email=email,
            name=name
        )
        
        self.data_subjects[subject_id] = data_subject
        self._save_privacy_data()
        
        self.logger.info(f"Registered data subject: {subject_id}")
        return data_subject
    
    def record_data_processing(self,
                             subject_id: str,
                             data_type: str,
                             purpose: str,
                             legal_basis: str,
                             classification: Optional[DataClassification] = None,
                             retention_policy: RetentionPolicy = RetentionPolicy.MEDIUM_TERM,
                             custom_retention: Optional[timedelta] = None,
                             encrypt_data: bool = False,
                             metadata: Optional[Dict] = None) -> PrivacyRecord:
        """Record data processing activity"""
        record_id = self._generate_record_id(subject_id, data_type)
        
        # Auto-classify data if not provided
        if classification is None:
            classification = self._classify_data(data_type, metadata or {})
        
        # Calculate retention period
        if retention_policy == RetentionPolicy.CUSTOM and custom_retention:
            retention_period = custom_retention
        else:
            retention_period = self.retention_periods.get(retention_policy)
        
        # Calculate expiration date
        expires_at = None
        if retention_period:
            expires_at = datetime.now() + retention_period
        
        privacy_record = PrivacyRecord(
            record_id=record_id,
            data_subject_id=subject_id,
            data_type=data_type,
            classification=classification,
            purpose=purpose,
            legal_basis=legal_basis,
            retention_policy=retention_policy,
            retention_period=retention_period,
            expires_at=expires_at,
            is_encrypted=encrypt_data,
            metadata=metadata or {}
        )
        
        self.privacy_records[record_id] = privacy_record
        self._save_privacy_data()
        
        self.logger.info(f"Recorded data processing: {record_id} for subject {subject_id}")
        return privacy_record
    
    def update_consent(self,
                      subject_id: str,
                      purpose: str,
                      consented: bool) -> bool:
        """Update consent status for data subject"""
        if subject_id not in self.data_subjects:
            self.register_data_subject(subject_id)
        
        subject = self.data_subjects[subject_id]
        subject.consent_status[purpose] = consented
        subject.last_updated = datetime.now()
        
        self._save_privacy_data()
        
        self.logger.info(f"Updated consent for {subject_id}: {purpose} = {consented}")
        return True
    
    def check_consent(self, subject_id: str, purpose: str) -> bool:
        """Check if data subject has consented for specific purpose"""
        if subject_id not in self.data_subjects:
            return False
        
        return self.data_subjects[subject_id].consent_status.get(purpose, False)
    
    def handle_access_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle data subject access request (GDPR Article 15)"""
        request_id = f"access_{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create privacy request record
        privacy_request = PrivacyRequest(
            request_id=request_id,
            subject_id=subject_id,
            request_type="access",
            status="processing"
        )
        
        self.privacy_requests[request_id] = privacy_request
        
        # Collect all data for the subject
        subject_data = {
            "subject_information": self.data_subjects.get(subject_id, {}).__dict__,
            "processing_records": [],
            "consent_status": {}
        }
        
        # Find all processing records
        for record in self.privacy_records.values():
            if record.data_subject_id == subject_id:
                subject_data["processing_records"].append({
                    "data_type": record.data_type,
                    "purpose": record.purpose,
                    "legal_basis": record.legal_basis,
                    "classification": record.classification.value,
                    "retention_policy": record.retention_policy.value,
                    "created_at": record.created_at.isoformat(),
                    "expires_at": record.expires_at.isoformat() if record.expires_at else None
                })
        
        # Include consent status
        if subject_id in self.data_subjects:
            subject_data["consent_status"] = self.data_subjects[subject_id].consent_status
        
        # Mark request as completed
        privacy_request.status = "completed"
        privacy_request.processed_at = datetime.now()
        privacy_request.details = {"data_exported": True}
        
        self._save_privacy_data()
        
        self.logger.info(f"Processed access request: {request_id}")
        return subject_data
    
    def handle_deletion_request(self, subject_id: str, data_types: Optional[List[str]] = None) -> bool:
        """Handle data subject deletion request (GDPR Article 17)"""
        request_id = f"deletion_{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create privacy request record
        privacy_request = PrivacyRequest(
            request_id=request_id,
            subject_id=subject_id,
            request_type="deletion",
            status="processing",
            details={"data_types": data_types or ["all"]}
        )
        
        self.privacy_requests[request_id] = privacy_request
        
        deleted_records = []
        
        # Delete specific data types or all data
        records_to_delete = []
        for record_id, record in self.privacy_records.items():
            if record.data_subject_id == subject_id:
                if data_types is None or record.data_type in data_types:
                    records_to_delete.append(record_id)
                    deleted_records.append(record.data_type)
        
        # Remove records
        for record_id in records_to_delete:
            del self.privacy_records[record_id]
        
        # If deleting all data, remove data subject
        if data_types is None:
            if subject_id in self.data_subjects:
                del self.data_subjects[subject_id]
        
        # Mark request as completed
        privacy_request.status = "completed"
        privacy_request.processed_at = datetime.now()
        privacy_request.details.update({
            "records_deleted": len(deleted_records),
            "data_types_deleted": deleted_records
        })
        
        self._save_privacy_data()
        
        self.logger.info(f"Processed deletion request: {request_id}, deleted {len(deleted_records)} records")
        return True
    
    def handle_portability_request(self, subject_id: str) -> str:
        """Handle data portability request (GDPR Article 20)"""
        request_id = f"portability_{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get all data for the subject
        subject_data = self.handle_access_request(subject_id)
        
        # Create privacy request record
        privacy_request = PrivacyRequest(
            request_id=request_id,
            subject_id=subject_id,
            request_type="portability",
            status="completed",
            processed_at=datetime.now(),
            details={"export_format": "json"}
        )
        
        self.privacy_requests[request_id] = privacy_request
        self._save_privacy_data()
        
        # Return data in portable format
        portable_data = json.dumps(subject_data, indent=2, default=str)
        
        self.logger.info(f"Processed portability request: {request_id}")
        return portable_data
    
    def audit_data_retention(self) -> List[Dict[str, Any]]:
        """Audit data retention and identify expired records"""
        expired_records = []
        now = datetime.now()
        
        for record_id, record in self.privacy_records.items():
            if record.expires_at and now > record.expires_at:
                expired_records.append({
                    "record_id": record_id,
                    "subject_id": record.data_subject_id,
                    "data_type": record.data_type,
                    "expired_days": (now - record.expires_at).days,
                    "classification": record.classification.value
                })
        
        self.logger.info(f"Found {len(expired_records)} expired records in retention audit")
        return expired_records
    
    def cleanup_expired_data(self) -> int:
        """Remove expired data according to retention policies"""
        expired_records = self.audit_data_retention()
        
        for expired in expired_records:
            record_id = expired["record_id"]
            if record_id in self.privacy_records:
                del self.privacy_records[record_id]
                self.logger.info(f"Deleted expired record: {record_id}")
        
        if expired_records:
            self._save_privacy_data()
        
        return len(expired_records)
    
    def anonymize_data(self, subject_id: str, data_types: Optional[List[str]] = None) -> bool:
        """Anonymize data for a subject while preserving utility"""
        anonymized_count = 0
        
        for record in self.privacy_records.values():
            if record.data_subject_id == subject_id:
                if data_types is None or record.data_type in data_types:
                    # Replace subject ID with anonymous hash
                    anonymous_id = self._create_anonymous_id(subject_id)
                    record.data_subject_id = anonymous_id
                    record.metadata["anonymized"] = True
                    record.metadata["anonymized_at"] = datetime.now().isoformat()
                    anonymized_count += 1
        
        self._save_privacy_data()
        
        self.logger.info(f"Anonymized {anonymized_count} records for subject {subject_id}")
        return anonymized_count > 0
    
    def generate_privacy_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate privacy compliance report"""
        # Filter records by date range
        relevant_records = [
            record for record in self.privacy_records.values()
            if start_date <= record.created_at <= end_date
        ]
        
        # Count by classification
        classification_counts = {}
        for record in relevant_records:
            classification = record.classification.value
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        
        # Count by purpose
        purpose_counts = {}
        for record in relevant_records:
            purpose = record.purpose
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
        
        # Privacy requests in period
        relevant_requests = [
            req for req in self.privacy_requests.values()
            if start_date <= req.submitted_at <= end_date
        ]
        
        request_type_counts = {}
        for request in relevant_requests:
            req_type = request.request_type
            request_type_counts[req_type] = request_type_counts.get(req_type, 0) + 1
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "data_processing": {
                "total_records": len(relevant_records),
                "by_classification": classification_counts,
                "by_purpose": purpose_counts
            },
            "privacy_requests": {
                "total_requests": len(relevant_requests),
                "by_type": request_type_counts,
                "completed_requests": len([r for r in relevant_requests if r.status == "completed"])
            },
            "data_subjects": {
                "total_subjects": len(self.data_subjects),
                "with_consent": len([s for s in self.data_subjects.values() if any(s.consent_status.values())])
            },
            "compliance_metrics": {
                "expired_records": len(self.audit_data_retention()),
                "encrypted_records": len([r for r in relevant_records if r.is_encrypted])
            }
        }
    
    def _classify_data(self, data_type: str, metadata: Dict[str, Any]) -> DataClassification:
        """Automatically classify data based on type and metadata"""
        data_type_lower = data_type.lower()
        
        # Check metadata for classification hints
        if "classification" in metadata:
            try:
                return DataClassification(metadata["classification"])
            except ValueError:
                pass
        
        # Apply classification rules
        for pattern, classification in self.classification_rules.items():
            if pattern in data_type_lower:
                return classification
        
        # Default to internal if no specific classification found
        return DataClassification.INTERNAL
    
    def _generate_record_id(self, subject_id: str, data_type: str) -> str:
        """Generate unique record ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{subject_id}_{data_type}_{timestamp}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"record_{hash_value}"
    
    def _create_anonymous_id(self, subject_id: str) -> str:
        """Create anonymous ID for data subject"""
        hash_value = hashlib.sha256(f"anonymous_{subject_id}".encode()).hexdigest()[:16]
        return f"anon_{hash_value}"
    
    def _load_privacy_data(self):
        """Load privacy data from storage"""
        # This would load from persistent storage in production
        self.logger.info("Privacy data loaded")
    
    def _save_privacy_data(self):
        """Save privacy data to storage"""
        # This would save to persistent storage in production
        self.logger.info("Privacy data saved")


# Example usage and testing
if __name__ == "__main__":
    # Test privacy manager
    privacy_manager = PrivacyManager()
    
    # Register data subject
    subject = privacy_manager.register_data_subject(
        subject_id="user123",
        email="user@company.com",
        name="John Doe"
    )
    
    # Record data processing
    record = privacy_manager.record_data_processing(
        subject_id="user123",
        data_type="query_logs",
        purpose="service_improvement",
        legal_basis="legitimate_interest",
        classification=DataClassification.INTERNAL,
        retention_policy=RetentionPolicy.MEDIUM_TERM
    )
    
    # Update consent
    privacy_manager.update_consent("user123", "service_improvement", True)
    
    # Check consent
    has_consent = privacy_manager.check_consent("user123", "service_improvement")
    print(f"User has consent for service improvement: {has_consent}")
    
    # Handle access request
    user_data = privacy_manager.handle_access_request("user123")
    print(f"User data includes {len(user_data['processing_records'])} processing records")
    
    # Generate privacy report
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    report = privacy_manager.generate_privacy_report(start_date, end_date)
    print(f"Privacy report shows {report['data_processing']['total_records']} records processed")