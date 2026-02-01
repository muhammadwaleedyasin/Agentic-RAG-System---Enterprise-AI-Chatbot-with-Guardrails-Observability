"""
Enterprise Compliance Configuration Management

Centralized configuration for all guardrails, compliance policies,
and security settings with environment-specific overrides.
"""
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json


class ComplianceLevel(Enum):
    """Compliance enforcement levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"
    AUDIT_ONLY = "audit_only"


class ActionType(Enum):
    """Actions to take on policy violations"""
    BLOCK = "block"
    REDACT = "redact"
    WARN = "warn"
    LOG_ONLY = "log_only"


@dataclass
class PIIDetectionConfig:
    """PII detection and redaction configuration"""
    enabled: bool = True
    detection_threshold: float = 0.8
    redaction_char: str = "*"
    preserve_format: bool = True
    entities_to_detect: List[str] = field(default_factory=lambda: [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", 
        "CREDIT_CARD", "IP_ADDRESS", "US_PASSPORT", "US_DRIVER_LICENSE"
    ])
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    action_on_detection: ActionType = ActionType.REDACT


@dataclass
class CitationConfig:
    """Citation enforcement configuration"""
    enforce_citations: bool = True
    min_citations_required: int = 1
    max_citations_allowed: int = 5
    citation_format: str = "[{source_id}]"
    require_page_numbers: bool = False
    allow_partial_citations: bool = True
    confidence_threshold: float = 0.7


@dataclass
class TopicFilterConfig:
    """Topic filtering configuration"""
    enabled: bool = True
    blocked_topics: List[str] = field(default_factory=lambda: [
        "personal_finances", "medical_advice", "legal_advice",
        "political_opinions", "confidential_strategy"
    ])
    allowed_topics: List[str] = field(default_factory=list)
    topic_detection_threshold: float = 0.75
    action_on_blocked_topic: ActionType = ActionType.BLOCK


@dataclass
class ConfidenceConfig:
    """Confidence scoring and abstain thresholds"""
    min_confidence_threshold: float = 0.6
    abstain_threshold: float = 0.4
    enable_abstain: bool = True
    abstain_message: str = "I don't have sufficient confidence to answer this question accurately."
    confidence_factors: Dict[str, float] = field(default_factory=lambda: {
        "retrieval_score": 0.3,
        "llm_confidence": 0.4,
        "citation_quality": 0.2,
        "topic_relevance": 0.1
    })


@dataclass
class AuditConfig:
    """Audit logging configuration"""
    enabled: bool = True
    log_level: str = "INFO"
    include_pii_detections: bool = True
    include_blocked_content: bool = True
    include_user_queries: bool = False  # Privacy consideration
    retention_days: int = 90
    export_format: str = "json"
    audit_fields: List[str] = field(default_factory=lambda: [
        "timestamp", "user_id", "query_hash", "response_type",
        "compliance_status", "violations", "confidence_score"
    ])


@dataclass
class ComplianceConfig:
    """Master compliance configuration"""
    compliance_level: ComplianceLevel = ComplianceLevel.STRICT
    pii_detection: PIIDetectionConfig = field(default_factory=PIIDetectionConfig)
    citation: CitationConfig = field(default_factory=CitationConfig)
    topic_filter: TopicFilterConfig = field(default_factory=TopicFilterConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    
    # Global settings
    fail_fast: bool = True
    enable_monitoring: bool = True
    alert_on_violations: bool = True
    
    @classmethod
    def from_env(cls) -> 'ComplianceConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        if os.getenv('COMPLIANCE_LEVEL'):
            config.compliance_level = ComplianceLevel(os.getenv('COMPLIANCE_LEVEL'))
        
        if os.getenv('PII_DETECTION_ENABLED'):
            config.pii_detection.enabled = os.getenv('PII_DETECTION_ENABLED').lower() == 'true'
        
        if os.getenv('CITATION_ENFORCEMENT'):
            config.citation.enforce_citations = os.getenv('CITATION_ENFORCEMENT').lower() == 'true'
        
        if os.getenv('MIN_CONFIDENCE_THRESHOLD'):
            config.confidence.min_confidence_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD'))
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ComplianceConfig':
        """Load configuration from YAML or JSON file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**data)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'compliance_level': self.compliance_level.value,
            'pii_detection': self.pii_detection.__dict__,
            'citation': self.citation.__dict__,
            'topic_filter': self.topic_filter.__dict__,
            'confidence': self.confidence.__dict__,
            'audit': self.audit.__dict__,
            'fail_fast': self.fail_fast,
            'enable_monitoring': self.enable_monitoring,
            'alert_on_violations': self.alert_on_violations
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to file"""
        data = self.to_dict()
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)


# Global configuration instance
_compliance_config: Optional[ComplianceConfig] = None


def get_compliance_config() -> ComplianceConfig:
    """Get the global compliance configuration instance"""
    global _compliance_config
    if _compliance_config is None:
        _compliance_config = ComplianceConfig.from_env()
    return _compliance_config


def set_compliance_config(config: ComplianceConfig):
    """Set the global compliance configuration instance"""
    global _compliance_config
    _compliance_config = config


def load_compliance_config(config_path: Optional[str] = None) -> ComplianceConfig:
    """Load and set compliance configuration from file or environment"""
    if config_path and os.path.exists(config_path):
        config = ComplianceConfig.from_file(config_path)
    else:
        config = ComplianceConfig.from_env()
    
    set_compliance_config(config)
    return config