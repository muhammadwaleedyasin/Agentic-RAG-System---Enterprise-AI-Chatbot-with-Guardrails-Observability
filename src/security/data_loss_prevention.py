"""
Data Loss Prevention (DLP) System

Provides comprehensive data loss prevention with content classification,
sensitive data detection, policy enforcement, and automated response capabilities.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Pattern
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
import json
import hashlib
import logging
from abc import ABC, abstractmethod
import base64
import mimetypes
from pathlib import Path

class SensitivityLevel(Enum):
    """Data sensitivity classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class DataType(Enum):
    """Types of sensitive data"""
    PII = "pii"                    # Personally Identifiable Information
    PHI = "phi"                    # Protected Health Information
    PCI = "pci"                    # Payment Card Information
    FINANCIAL = "financial"        # Financial data
    INTELLECTUAL_PROPERTY = "ip"   # Intellectual Property
    CREDENTIALS = "credentials"    # Authentication credentials
    LEGAL = "legal"               # Legal documents
    TRADE_SECRET = "trade_secret" # Trade secrets

class ActionType(Enum):
    """DLP response actions"""
    ALLOW = "allow"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ENCRYPT = "encrypt"
    REDACT = "redact"
    ALERT = "alert"
    LOG = "log"

class DetectionMethod(Enum):
    """Methods for detecting sensitive data"""
    REGEX = "regex"
    KEYWORD = "keyword"
    PATTERN = "pattern"
    ML_CLASSIFICATION = "ml_classification"
    FINGERPRINT = "fingerprint"
    HASH_MATCH = "hash_match"

@dataclass
class DetectionRule:
    """Rule for detecting sensitive data"""
    rule_id: str
    name: str
    description: str
    data_type: DataType
    sensitivity_level: SensitivityLevel
    detection_method: DetectionMethod
    
    # Detection parameters
    patterns: List[str] = field(default_factory=list)  # Regex patterns
    keywords: List[str] = field(default_factory=list)  # Keyword list
    confidence_threshold: float = 0.8
    
    # Context requirements
    required_context: List[str] = field(default_factory=list)
    excluded_contexts: List[str] = field(default_factory=list)
    
    # Rule metadata
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    false_positive_rate: float = 0.0

@dataclass
class DLPPolicy:
    """Data Loss Prevention policy"""
    policy_id: str
    name: str
    description: str
    data_types: List[DataType] = field(default_factory=list)
    sensitivity_levels: List[SensitivityLevel] = field(default_factory=list)
    
    # Actions by context
    actions: Dict[str, ActionType] = field(default_factory=dict)  # context -> action
    
    # Conditions
    user_groups: List[str] = field(default_factory=list)
    resource_types: List[str] = field(default_factory=list)
    time_restrictions: Dict[str, Any] = field(default_factory=dict)
    location_restrictions: List[str] = field(default_factory=list)
    
    # Enforcement settings
    is_active: bool = True
    enforcement_mode: str = "enforce"  # monitor, enforce
    priority: int = 100
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Detection:
    """Represents a sensitive data detection"""
    detection_id: str
    rule_id: str
    data_type: DataType
    sensitivity_level: SensitivityLevel
    confidence: float
    
    # Location information
    content_hash: str
    start_position: int
    end_position: int
    matched_text: str = ""  # Actual matched content (may be redacted)
    context: str = ""       # Surrounding context
    
    # Metadata
    detection_method: DetectionMethod = DetectionMethod.REGEX
    timestamp: datetime = field(default_factory=datetime.utcnow)
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DLPEvent:
    """DLP policy enforcement event"""
    event_id: str
    policy_id: str
    user_id: Optional[str]
    resource_id: str
    resource_type: str
    action_taken: ActionType
    
    # Detection details
    detections: List[Detection] = field(default_factory=list)
    total_sensitivity_score: float = 0.0
    
    # Context
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    # Response details
    response_details: Dict[str, Any] = field(default_factory=dict)
    is_false_positive: bool = False
    reviewed_by: Optional[str] = None

class SensitiveDataDetector:
    """Core sensitive data detection engine"""
    
    def __init__(self):
        self.rules: Dict[str, DetectionRule] = {}
        self.compiled_patterns: Dict[str, Pattern] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize default detection rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default detection rules for common sensitive data types"""
        
        # Social Security Numbers (US)
        ssn_rule = DetectionRule(
            rule_id="ssn_us",
            name="US Social Security Number",
            description="Detects US Social Security Numbers",
            data_type=DataType.PII,
            sensitivity_level=SensitivityLevel.CONFIDENTIAL,
            detection_method=DetectionMethod.REGEX,
            patterns=[
                r'\b\d{3}-\d{2}-\d{4}\b',  # XXX-XX-XXXX
                r'\b\d{3}\s\d{2}\s\d{4}\b',  # XXX XX XXXX
                r'\b\d{9}\b'  # XXXXXXXXX (context dependent)
            ],
            required_context=["ssn", "social", "security"]
        )
        self.add_rule(ssn_rule)
        
        # Credit Card Numbers
        cc_rule = DetectionRule(
            rule_id="credit_card",
            name="Credit Card Number",
            description="Detects credit card numbers",
            data_type=DataType.PCI,
            sensitivity_level=SensitivityLevel.RESTRICTED,
            detection_method=DetectionMethod.REGEX,
            patterns=[
                r'\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Visa
                r'\b5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # MasterCard
                r'\b3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}\b',  # American Express
                r'\b6(?:011|5\d{2})[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'  # Discover
            ],
            required_context=["card", "credit", "payment", "visa", "mastercard", "amex"]
        )
        self.add_rule(cc_rule)
        
        # Email Addresses
        email_rule = DetectionRule(
            rule_id="email_address",
            name="Email Address",
            description="Detects email addresses",
            data_type=DataType.PII,
            sensitivity_level=SensitivityLevel.INTERNAL,
            detection_method=DetectionMethod.REGEX,
            patterns=[
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ]
        )
        self.add_rule(email_rule)
        
        # Phone Numbers
        phone_rule = DetectionRule(
            rule_id="phone_number",
            name="Phone Number",
            description="Detects phone numbers",
            data_type=DataType.PII,
            sensitivity_level=SensitivityLevel.INTERNAL,
            detection_method=DetectionMethod.REGEX,
            patterns=[
                r'\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                r'\b\([0-9]{3}\)\s[0-9]{3}-[0-9]{4}\b'
            ]
        )
        self.add_rule(phone_rule)
        
        # IP Addresses
        ip_rule = DetectionRule(
            rule_id="ip_address",
            name="IP Address",
            description="Detects IP addresses",
            data_type=DataType.CREDENTIALS,
            sensitivity_level=SensitivityLevel.CONFIDENTIAL,
            detection_method=DetectionMethod.REGEX,
            patterns=[
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ],
            excluded_contexts=["example", "test", "documentation"]
        )
        self.add_rule(ip_rule)
        
        # API Keys and Tokens
        api_key_rule = DetectionRule(
            rule_id="api_keys",
            name="API Keys and Tokens",
            description="Detects API keys and authentication tokens",
            data_type=DataType.CREDENTIALS,
            sensitivity_level=SensitivityLevel.RESTRICTED,
            detection_method=DetectionMethod.REGEX,
            patterns=[
                r'(?i)api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}',
                r'(?i)token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}',
                r'(?i)secret["\']?\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}',
                r'(?i)password["\']?\s*[:=]\s*["\']?[a-zA-Z0-9]{8,}'
            ],
            required_context=["key", "token", "secret", "password", "auth"]
        )
        self.add_rule(api_key_rule)
        
        # Medical Record Numbers
        mrn_rule = DetectionRule(
            rule_id="medical_record_number",
            name="Medical Record Number",
            description="Detects medical record numbers",
            data_type=DataType.PHI,
            sensitivity_level=SensitivityLevel.RESTRICTED,
            detection_method=DetectionMethod.REGEX,
            patterns=[
                r'\b(?:MRN|MR|Medical Record)[\s#:]*\d{6,12}\b'
            ],
            required_context=["medical", "patient", "record", "hospital", "clinic"]
        )
        self.add_rule(mrn_rule)
        
        # Financial Account Numbers
        account_rule = DetectionRule(
            rule_id="bank_account",
            name="Bank Account Number",
            description="Detects bank account numbers",
            data_type=DataType.FINANCIAL,
            sensitivity_level=SensitivityLevel.RESTRICTED,
            detection_method=DetectionMethod.REGEX,
            patterns=[
                r'\b\d{8,17}\b'  # Generic account number pattern
            ],
            required_context=["account", "bank", "routing", "financial", "deposit"]
        )
        self.add_rule(account_rule)
    
    def add_rule(self, rule: DetectionRule):
        """Add a detection rule"""
        self.rules[rule.rule_id] = rule
        
        # Compile regex patterns for better performance
        if rule.detection_method == DetectionMethod.REGEX:
            compiled_patterns = []
            for pattern in rule.patterns:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    self.logger.error(f"Invalid regex pattern in rule {rule.rule_id}: {pattern} - {e}")
            
            self.compiled_patterns[rule.rule_id] = compiled_patterns
        
        self.logger.info(f"Added detection rule: {rule.rule_id}")
    
    def detect_sensitive_data(self, content: str, context: str = "", 
                            metadata: Dict[str, Any] = None) -> List[Detection]:
        """Detect sensitive data in content"""
        if metadata is None:
            metadata = {}
        
        detections = []
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        for rule in self.rules.values():
            if not rule.is_active:
                continue
            
            rule_detections = self._apply_rule(rule, content, context, content_hash, metadata)
            detections.extend(rule_detections)
        
        # Remove overlapping detections (keep highest confidence)
        detections = self._remove_overlapping_detections(detections)
        
        return detections
    
    def _apply_rule(self, rule: DetectionRule, content: str, context: str,
                   content_hash: str, metadata: Dict[str, Any]) -> List[Detection]:
        """Apply a single detection rule"""
        detections = []
        
        # Check context requirements
        if rule.required_context:
            context_lower = (content + " " + context).lower()
            if not any(req_ctx.lower() in context_lower for req_ctx in rule.required_context):
                return detections
        
        # Check excluded contexts
        if rule.excluded_contexts:
            context_lower = (content + " " + context).lower()
            if any(excl_ctx.lower() in context_lower for excl_ctx in rule.excluded_contexts):
                return detections
        
        if rule.detection_method == DetectionMethod.REGEX:
            detections.extend(self._regex_detection(rule, content, content_hash))
        elif rule.detection_method == DetectionMethod.KEYWORD:
            detections.extend(self._keyword_detection(rule, content, content_hash))
        
        return detections
    
    def _regex_detection(self, rule: DetectionRule, content: str, 
                        content_hash: str) -> List[Detection]:
        """Perform regex-based detection"""
        detections = []
        
        if rule.rule_id not in self.compiled_patterns:
            return detections
        
        for pattern in self.compiled_patterns[rule.rule_id]:
            for match in pattern.finditer(content):
                # Additional validation for specific data types
                if self._validate_match(rule, match.group()):
                    detection = Detection(
                        detection_id=f"{rule.rule_id}_{len(detections)}_{datetime.utcnow().timestamp()}",
                        rule_id=rule.rule_id,
                        data_type=rule.data_type,
                        sensitivity_level=rule.sensitivity_level,
                        confidence=rule.confidence_threshold,
                        content_hash=content_hash,
                        start_position=match.start(),
                        end_position=match.end(),
                        matched_text=self._redact_sensitive_text(match.group(), rule.data_type),
                        context=self._extract_context(content, match.start(), match.end()),
                        detection_method=rule.detection_method
                    )
                    detections.append(detection)
        
        return detections
    
    def _keyword_detection(self, rule: DetectionRule, content: str,
                          content_hash: str) -> List[Detection]:
        """Perform keyword-based detection"""
        detections = []
        content_lower = content.lower()
        
        for keyword in rule.keywords:
            keyword_lower = keyword.lower()
            start = 0
            
            while True:
                pos = content_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                
                detection = Detection(
                    detection_id=f"{rule.rule_id}_{len(detections)}_{datetime.utcnow().timestamp()}",
                    rule_id=rule.rule_id,
                    data_type=rule.data_type,
                    sensitivity_level=rule.sensitivity_level,
                    confidence=rule.confidence_threshold,
                    content_hash=content_hash,
                    start_position=pos,
                    end_position=pos + len(keyword),
                    matched_text=self._redact_sensitive_text(keyword, rule.data_type),
                    context=self._extract_context(content, pos, pos + len(keyword)),
                    detection_method=rule.detection_method
                )
                detections.append(detection)
                start = pos + 1
        
        return detections
    
    def _validate_match(self, rule: DetectionRule, match_text: str) -> bool:
        """Validate a regex match using additional criteria"""
        if rule.data_type == DataType.PCI:
            # Validate credit card number using Luhn algorithm
            return self._validate_credit_card(match_text)
        elif rule.data_type == DataType.PII and "ssn" in rule.rule_id:
            # Basic SSN validation
            digits = re.sub(r'[^\d]', '', match_text)
            return len(digits) == 9 and digits != "000000000"
        
        return True
    
    def _validate_credit_card(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm"""
        # Remove non-digit characters
        digits = re.sub(r'[^\d]', '', card_number)
        
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        # Luhn algorithm
        def luhn_checksum(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10
        
        return luhn_checksum(int(digits)) == 0
    
    def _extract_context(self, content: str, start: int, end: int, 
                        context_size: int = 50) -> str:
        """Extract surrounding context for a match"""
        context_start = max(0, start - context_size)
        context_end = min(len(content), end + context_size)
        
        before = content[context_start:start]
        after = content[end:context_end]
        
        return f"{before}[DETECTED]{after}"
    
    def _redact_sensitive_text(self, text: str, data_type: DataType) -> str:
        """Redact sensitive text based on data type"""
        if data_type in [DataType.PCI, DataType.CREDENTIALS]:
            # Show only last 4 characters for highly sensitive data
            if len(text) > 4:
                return "*" * (len(text) - 4) + text[-4:]
            else:
                return "*" * len(text)
        elif data_type == DataType.PII:
            # Partial redaction for PII
            if len(text) > 6:
                return text[:2] + "*" * (len(text) - 4) + text[-2:]
            else:
                return "*" * len(text)
        else:
            # Light redaction for other types
            return text[:3] + "*" * max(0, len(text) - 3)
    
    def _remove_overlapping_detections(self, detections: List[Detection]) -> List[Detection]:
        """Remove overlapping detections, keeping the one with highest confidence"""
        if not detections:
            return detections
        
        # Sort by position
        sorted_detections = sorted(detections, key=lambda d: d.start_position)
        
        filtered_detections = []
        for detection in sorted_detections:
            # Check if this detection overlaps with any in filtered list
            overlaps = False
            for existing in filtered_detections:
                if (detection.start_position < existing.end_position and 
                    detection.end_position > existing.start_position):
                    # Overlap detected
                    if detection.confidence > existing.confidence:
                        # Replace existing with current (higher confidence)
                        filtered_detections.remove(existing)
                        filtered_detections.append(detection)
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_detections.append(detection)
        
        return filtered_detections

class ContentClassifier:
    """Classifies content based on sensitivity and data types"""
    
    def __init__(self, detector: SensitiveDataDetector):
        self.detector = detector
        self.logger = logging.getLogger(__name__)
    
    def classify_content(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Classify content and determine overall sensitivity"""
        if metadata is None:
            metadata = {}
        
        detections = self.detector.detect_sensitive_data(content, metadata=metadata)
        
        # Determine overall classification
        if not detections:
            overall_sensitivity = SensitivityLevel.PUBLIC
            data_types = []
        else:
            # Get highest sensitivity level
            sensitivity_levels = [d.sensitivity_level for d in detections]
            sensitivity_hierarchy = {
                SensitivityLevel.PUBLIC: 0,
                SensitivityLevel.INTERNAL: 1,
                SensitivityLevel.CONFIDENTIAL: 2,
                SensitivityLevel.RESTRICTED: 3,
                SensitivityLevel.TOP_SECRET: 4
            }
            
            overall_sensitivity = max(sensitivity_levels, 
                                    key=lambda x: sensitivity_hierarchy[x])
            data_types = list(set(d.data_type for d in detections))
        
        # Calculate confidence score
        if detections:
            confidence_score = sum(d.confidence for d in detections) / len(detections)
        else:
            confidence_score = 1.0
        
        return {
            "overall_sensitivity": overall_sensitivity,
            "data_types": data_types,
            "detections": detections,
            "confidence_score": confidence_score,
            "detection_count": len(detections),
            "classification_timestamp": datetime.utcnow(),
            "content_hash": hashlib.sha256(content.encode()).hexdigest()
        }

class DataLossPreventionEngine:
    """Main DLP engine for policy enforcement"""
    
    def __init__(self):
        self.detector = SensitiveDataDetector()
        self.classifier = ContentClassifier(self.detector)
        self.policies: Dict[str, DLPPolicy] = {}
        self.events: List[DLPEvent] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default DLP policies"""
        
        # High-sensitivity data blocking policy
        high_sensitivity_policy = DLPPolicy(
            policy_id="block_high_sensitivity",
            name="Block High Sensitivity Data",
            description="Block access to highly sensitive data",
            data_types=[DataType.PCI, DataType.PHI],
            sensitivity_levels=[SensitivityLevel.RESTRICTED, SensitivityLevel.TOP_SECRET],
            actions={
                "external_share": ActionType.BLOCK,
                "download": ActionType.BLOCK,
                "print": ActionType.BLOCK,
                "email": ActionType.BLOCK
            },
            enforcement_mode="enforce",
            priority=100
        )
        self.add_policy(high_sensitivity_policy)
        
        # PII monitoring policy
        pii_monitoring_policy = DLPPolicy(
            policy_id="monitor_pii",
            name="Monitor PII Access",
            description="Monitor and log PII access",
            data_types=[DataType.PII],
            sensitivity_levels=[SensitivityLevel.CONFIDENTIAL],
            actions={
                "read": ActionType.LOG,
                "download": ActionType.ALERT,
                "external_share": ActionType.BLOCK
            },
            enforcement_mode="enforce",
            priority=80
        )
        self.add_policy(pii_monitoring_policy)
        
        # Credential protection policy
        credential_policy = DLPPolicy(
            policy_id="protect_credentials",
            name="Protect Credentials",
            description="Protect authentication credentials and API keys",
            data_types=[DataType.CREDENTIALS],
            actions={
                "read": ActionType.LOG,
                "share": ActionType.BLOCK,
                "external_access": ActionType.BLOCK
            },
            enforcement_mode="enforce",
            priority=90
        )
        self.add_policy(credential_policy)
    
    def add_policy(self, policy: DLPPolicy):
        """Add a DLP policy"""
        self.policies[policy.policy_id] = policy
        self.logger.info(f"Added DLP policy: {policy.policy_id}")
    
    def remove_policy(self, policy_id: str):
        """Remove a DLP policy"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            self.logger.info(f"Removed DLP policy: {policy_id}")
    
    def analyze_content(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze content for sensitive data and policy violations"""
        if context is None:
            context = {}
        
        # Classify content
        classification = self.classifier.classify_content(content, context)
        
        # Check policies
        applicable_policies = self._get_applicable_policies(classification, context)
        policy_violations = []
        recommended_actions = []
        
        for policy in applicable_policies:
            action_context = context.get("action_context", "read")
            if action_context in policy.actions:
                recommended_action = policy.actions[action_context]
                recommended_actions.append({
                    "policy_id": policy.policy_id,
                    "action": recommended_action,
                    "priority": policy.priority
                })
                
                if recommended_action in [ActionType.BLOCK, ActionType.QUARANTINE]:
                    policy_violations.append(policy.policy_id)
        
        # Determine final action (highest priority)
        if recommended_actions:
            final_action = max(recommended_actions, key=lambda x: x["priority"])["action"]
        else:
            final_action = ActionType.ALLOW
        
        return {
            "classification": classification,
            "applicable_policies": [p.policy_id for p in applicable_policies],
            "policy_violations": policy_violations,
            "recommended_action": final_action,
            "analysis_timestamp": datetime.utcnow(),
            "risk_score": self._calculate_risk_score(classification, len(policy_violations))
        }
    
    def enforce_policy(self, content: str, context: Dict[str, Any]) -> DLPEvent:
        """Enforce DLP policies on content"""
        analysis = self.analyze_content(content, context)
        
        # Create DLP event
        event = DLPEvent(
            event_id=f"dlp_{datetime.utcnow().timestamp()}",
            policy_id=analysis["applicable_policies"][0] if analysis["applicable_policies"] else "default",
            user_id=context.get("user_id"),
            resource_id=context.get("resource_id", "unknown"),
            resource_type=context.get("resource_type", "document"),
            action_taken=analysis["recommended_action"],
            detections=analysis["classification"]["detections"],
            total_sensitivity_score=analysis["risk_score"],
            source_ip=context.get("ip_address"),
            user_agent=context.get("user_agent"),
            additional_context=context,
            response_details=analysis
        )
        
        # Execute action
        success = self._execute_action(event.action_taken, content, context)
        event.response_details["action_success"] = success
        
        # Store event
        self.events.append(event)
        
        # Log event
        self.logger.info(f"DLP event: {event.action_taken.value} for {event.resource_type} "
                        f"by {event.user_id} - {len(event.detections)} detections")
        
        return event
    
    def _get_applicable_policies(self, classification: Dict[str, Any], 
                               context: Dict[str, Any]) -> List[DLPPolicy]:
        """Get policies applicable to the given classification and context"""
        applicable_policies = []
        
        for policy in self.policies.values():
            if not policy.is_active:
                continue
            
            # Check data types
            if policy.data_types:
                classification_data_types = classification["data_types"]
                if not any(dt in classification_data_types for dt in policy.data_types):
                    continue
            
            # Check sensitivity levels
            if policy.sensitivity_levels:
                if classification["overall_sensitivity"] not in policy.sensitivity_levels:
                    continue
            
            # Check user groups (if specified)
            if policy.user_groups:
                user_groups = context.get("user_groups", [])
                if not any(group in user_groups for group in policy.user_groups):
                    continue
            
            # Check resource types
            if policy.resource_types:
                resource_type = context.get("resource_type", "")
                if resource_type not in policy.resource_types:
                    continue
            
            applicable_policies.append(policy)
        
        # Sort by priority (highest first)
        applicable_policies.sort(key=lambda p: p.priority, reverse=True)
        
        return applicable_policies
    
    def _calculate_risk_score(self, classification: Dict[str, Any], 
                            violation_count: int) -> float:
        """Calculate risk score for content"""
        base_score = 0.0
        
        # Score based on sensitivity level
        sensitivity_scores = {
            SensitivityLevel.PUBLIC: 0.1,
            SensitivityLevel.INTERNAL: 0.3,
            SensitivityLevel.CONFIDENTIAL: 0.6,
            SensitivityLevel.RESTRICTED: 0.8,
            SensitivityLevel.TOP_SECRET: 1.0
        }
        
        base_score += sensitivity_scores.get(classification["overall_sensitivity"], 0.0)
        
        # Score based on detection count
        detection_count = classification["detection_count"]
        base_score += min(detection_count * 0.1, 0.5)
        
        # Score based on policy violations
        base_score += violation_count * 0.2
        
        # Score based on confidence
        base_score *= classification["confidence_score"]
        
        return min(base_score, 1.0)
    
    def _execute_action(self, action: ActionType, content: str, 
                       context: Dict[str, Any]) -> bool:
        """Execute the determined DLP action"""
        try:
            if action == ActionType.ALLOW:
                return True
            elif action == ActionType.BLOCK:
                # Block access - typically handled at application level
                self.logger.warning(f"Blocked access to {context.get('resource_id', 'unknown')}")
                return True
            elif action == ActionType.QUARANTINE:
                # Move to quarantine - implementation depends on storage system
                self.logger.warning(f"Quarantined {context.get('resource_id', 'unknown')}")
                return True
            elif action == ActionType.ENCRYPT:
                # Encrypt content - simplified implementation
                encrypted_content = base64.b64encode(content.encode()).decode()
                context["encrypted_content"] = encrypted_content
                return True
            elif action == ActionType.REDACT:
                # Redact sensitive content
                redacted_content = self._redact_content(content)
                context["redacted_content"] = redacted_content
                return True
            elif action in [ActionType.ALERT, ActionType.LOG]:
                # These are handled by the event logging system
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to execute DLP action {action}: {e}")
            return False
    
    def _redact_content(self, content: str) -> str:
        """Redact sensitive data from content"""
        detections = self.detector.detect_sensitive_data(content)
        
        # Sort detections by position (reverse order to maintain positions)
        detections.sort(key=lambda d: d.start_position, reverse=True)
        
        redacted_content = content
        for detection in detections:
            redacted_text = "*" * (detection.end_position - detection.start_position)
            redacted_content = (redacted_content[:detection.start_position] + 
                              redacted_text + 
                              redacted_content[detection.end_position:])
        
        return redacted_content
    
    def get_dlp_statistics(self, time_range: timedelta = None) -> Dict[str, Any]:
        """Get DLP statistics"""
        if time_range is None:
            time_range = timedelta(days=7)
        
        cutoff_time = datetime.utcnow() - time_range
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        # Calculate statistics
        stats = {
            "total_events": len(recent_events),
            "events_by_action": {},
            "events_by_data_type": {},
            "events_by_sensitivity": {},
            "blocked_events": 0,
            "high_risk_events": 0,
            "false_positives": 0,
            "avg_risk_score": 0.0
        }
        
        for event in recent_events:
            # Action statistics
            action_name = event.action_taken.value
            stats["events_by_action"][action_name] = stats["events_by_action"].get(action_name, 0) + 1
            
            # Data type statistics
            for detection in event.detections:
                data_type = detection.data_type.value
                stats["events_by_data_type"][data_type] = stats["events_by_data_type"].get(data_type, 0) + 1
                
                # Sensitivity statistics
                sensitivity = detection.sensitivity_level.value
                stats["events_by_sensitivity"][sensitivity] = stats["events_by_sensitivity"].get(sensitivity, 0) + 1
            
            # Risk statistics
            if event.action_taken in [ActionType.BLOCK, ActionType.QUARANTINE]:
                stats["blocked_events"] += 1
            
            if event.total_sensitivity_score >= 0.7:
                stats["high_risk_events"] += 1
            
            if event.is_false_positive:
                stats["false_positives"] += 1
        
        # Calculate average risk score
        if recent_events:
            stats["avg_risk_score"] = sum(e.total_sensitivity_score for e in recent_events) / len(recent_events)
        
        return stats
    
    def mark_false_positive(self, event_id: str, reviewed_by: str):
        """Mark a DLP event as false positive"""
        for event in self.events:
            if event.event_id == event_id:
                event.is_false_positive = True
                event.reviewed_by = reviewed_by
                
                # Update rule false positive rate
                for detection in event.detections:
                    if detection.rule_id in self.detector.rules:
                        rule = self.detector.rules[detection.rule_id]
                        # Simple false positive rate calculation
                        rule.false_positive_rate = min(rule.false_positive_rate + 0.01, 0.5)
                
                self.logger.info(f"Marked DLP event {event_id} as false positive")
                return True
        
        return False
    
    def export_events(self, time_range: timedelta = None, 
                     format_type: str = "json") -> str:
        """Export DLP events"""
        if time_range is None:
            time_range = timedelta(days=30)
        
        cutoff_time = datetime.utcnow() - time_range
        events_to_export = [e for e in self.events if e.timestamp >= cutoff_time]
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "time_range_days": time_range.days,
            "total_events": len(events_to_export),
            "events": [
                {
                    "event_id": e.event_id,
                    "policy_id": e.policy_id,
                    "user_id": e.user_id,
                    "resource_id": e.resource_id,
                    "resource_type": e.resource_type,
                    "action_taken": e.action_taken.value,
                    "timestamp": e.timestamp.isoformat(),
                    "risk_score": e.total_sensitivity_score,
                    "detection_count": len(e.detections),
                    "is_false_positive": e.is_false_positive
                }
                for e in events_to_export
            ]
        }
        
        if format_type == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")