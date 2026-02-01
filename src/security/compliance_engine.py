"""
Compliance Engine and Policy Enforcement System

Provides comprehensive policy enforcement, regulatory compliance monitoring,
and automated compliance reporting for enterprise security requirements.
"""

from typing import Dict, List, Optional, Any, Union, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
import logging
from abc import ABC, abstractmethod
import yaml
from pathlib import Path

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "sox"              # Sarbanes-Oxley Act
    GDPR = "gdpr"            # General Data Protection Regulation
    HIPAA = "hipaa"          # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"      # Payment Card Industry Data Security Standard
    ISO_27001 = "iso_27001"  # ISO/IEC 27001
    NIST = "nist"            # NIST Cybersecurity Framework
    CCPA = "ccpa"            # California Consumer Privacy Act
    FISMA = "fisma"          # Federal Information Security Management Act

class PolicyType(Enum):
    """Types of security policies"""
    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"
    AUTHENTICATION = "authentication"
    ENCRYPTION = "encryption"
    AUDIT_LOGGING = "audit_logging"
    INCIDENT_RESPONSE = "incident_response"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE_MONITORING = "compliance_monitoring"

class ViolationSeverity(Enum):
    """Policy violation severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStatus(Enum):
    """Compliance assessment status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"
    REMEDIATION_REQUIRED = "remediation_required"

@dataclass
class PolicyRule:
    """Individual policy rule definition"""
    rule_id: str
    name: str
    description: str
    policy_type: PolicyType
    frameworks: List[ComplianceFramework] = field(default_factory=list)
    
    # Rule conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    
    # Violation settings
    violation_severity: ViolationSeverity = ViolationSeverity.MEDIUM
    violation_message: str = ""
    
    # Remediation
    remediation_steps: List[str] = field(default_factory=list)
    auto_remediation: bool = False
    
    # Metadata
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

@dataclass
class ComplianceRule:
    """Compliance rule linking policy rules to frameworks"""
    rule_id: str
    framework: ComplianceFramework
    control_id: str  # Framework-specific control identifier
    policy_rules: List[str] = field(default_factory=list)  # List of policy rule IDs
    requirement_text: str = ""
    assessment_frequency: str = "monthly"  # daily, weekly, monthly, quarterly, annually
    last_assessment: Optional[datetime] = None
    next_assessment: Optional[datetime] = None
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED

@dataclass
class PolicyViolation:
    """Represents a policy violation"""
    violation_id: str
    rule_id: str
    resource_type: str
    resource_id: str
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: ViolationSeverity = ViolationSeverity.MEDIUM
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Resolution tracking
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: str = ""
    
    # Impact assessment
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    affected_systems: List[str] = field(default_factory=list)

@dataclass
class ComplianceAssessment:
    """Compliance assessment result"""
    assessment_id: str
    framework: ComplianceFramework
    assessment_date: datetime = field(default_factory=datetime.utcnow)
    overall_status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    score: float = 0.0  # 0-100 compliance score
    
    # Detailed results
    control_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    violations: List[PolicyViolation] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    assessor: Optional[str] = None
    assessment_scope: Dict[str, Any] = field(default_factory=dict)
    next_assessment_due: Optional[datetime] = None

class PolicyEngine:
    """Core policy evaluation engine"""
    
    def __init__(self):
        self.rules: Dict[str, PolicyRule] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_rule(self, rule: PolicyRule):
        """Add a policy rule"""
        self.rules[rule.rule_id] = rule
        self.logger.info(f"Added policy rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str):
        """Remove a policy rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"Removed policy rule: {rule_id}")
    
    def evaluate_rules(self, context: Dict[str, Any], 
                      rule_types: List[PolicyType] = None) -> List[PolicyViolation]:
        """Evaluate policy rules against given context"""
        violations = []
        
        for rule in self.rules.values():
            if not rule.is_active:
                continue
            
            if rule_types and rule.policy_type not in rule_types:
                continue
            
            violation = self._evaluate_rule(rule, context)
            if violation:
                violations.append(violation)
        
        return violations
    
    def _evaluate_rule(self, rule: PolicyRule, context: Dict[str, Any]) -> Optional[PolicyViolation]:
        """Evaluate a single policy rule"""
        try:
            # Check required fields
            for field in rule.required_fields:
                if field not in context:
                    return PolicyViolation(
                        violation_id=f"{rule.rule_id}_{datetime.utcnow().timestamp()}",
                        rule_id=rule.rule_id,
                        resource_type=context.get("resource_type", "unknown"),
                        resource_id=context.get("resource_id", "unknown"),
                        user_id=context.get("user_id"),
                        severity=rule.violation_severity,
                        message=f"Required field '{field}' missing",
                        details={"missing_field": field},
                        context=context
                    )
            
            # Evaluate conditions
            if not self._evaluate_conditions(rule.conditions, context):
                return PolicyViolation(
                    violation_id=f"{rule.rule_id}_{datetime.utcnow().timestamp()}",
                    rule_id=rule.rule_id,
                    resource_type=context.get("resource_type", "unknown"),
                    resource_id=context.get("resource_id", "unknown"),
                    user_id=context.get("user_id"),
                    severity=rule.violation_severity,
                    message=rule.violation_message or f"Policy rule {rule.rule_id} violated",
                    details={"rule_conditions": rule.conditions},
                    context=context
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
            return None
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate rule conditions"""
        for condition_type, condition_value in conditions.items():
            if not self._evaluate_condition(condition_type, condition_value, context):
                return False
        return True
    
    def _evaluate_condition(self, condition_type: str, condition_value: Any, 
                          context: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        if condition_type == "min_password_length":
            password = context.get("password", "")
            return len(password) >= condition_value
        
        elif condition_type == "required_access_level":
            access_level = context.get("access_level", "")
            return access_level in condition_value
        
        elif condition_type == "allowed_ip_ranges":
            ip_address = context.get("ip_address", "")
            return any(self._ip_in_range(ip_address, ip_range) for ip_range in condition_value)
        
        elif condition_type == "max_failed_attempts":
            failed_attempts = context.get("failed_attempts", 0)
            return failed_attempts <= condition_value
        
        elif condition_type == "encryption_required":
            is_encrypted = context.get("is_encrypted", False)
            return is_encrypted == condition_value
        
        elif condition_type == "audit_logging_enabled":
            audit_enabled = context.get("audit_logging_enabled", False)
            return audit_enabled == condition_value
        
        elif condition_type == "data_classification":
            classification = context.get("data_classification", "")
            allowed_classifications = condition_value
            return classification in allowed_classifications
        
        elif condition_type == "regex_match":
            field_name = condition_value.get("field")
            pattern = condition_value.get("pattern")
            field_value = context.get(field_name, "")
            return bool(re.match(pattern, str(field_value)))
        
        elif condition_type == "time_window":
            current_time = datetime.utcnow()
            start_time = datetime.fromisoformat(condition_value["start"])
            end_time = datetime.fromisoformat(condition_value["end"])
            return start_time <= current_time <= end_time
        
        return True
    
    def _ip_in_range(self, ip_address: str, ip_range: str) -> bool:
        """Check if IP address is in given range (simplified implementation)"""
        try:
            import ipaddress
            return ipaddress.ip_address(ip_address) in ipaddress.ip_network(ip_range, strict=False)
        except:
            return False

class ComplianceEngine:
    """Main compliance engine for policy enforcement and monitoring"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.policy_engine = PolicyEngine()
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.violations: List[PolicyViolation] = []
        self.assessments: Dict[str, ComplianceAssessment] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path:
            self.load_configuration(config_path)
        else:
            self._load_default_rules()
    
    def load_configuration(self, config_path: str):
        """Load compliance configuration from file"""
        config_path = Path(config_path)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Load policy rules
        for rule_data in config.get("policy_rules", []):
            rule = PolicyRule(**rule_data)
            self.policy_engine.add_rule(rule)
        
        # Load compliance rules
        for rule_data in config.get("compliance_rules", []):
            rule = ComplianceRule(**rule_data)
            self.compliance_rules[rule.rule_id] = rule
        
        self.logger.info(f"Loaded configuration from {config_path}")
    
    def _load_default_rules(self):
        """Load default compliance rules"""
        # GDPR - Data Protection Rules
        gdpr_rules = [
            PolicyRule(
                rule_id="gdpr_consent_required",
                name="GDPR Consent Required",
                description="Personal data processing requires explicit consent",
                policy_type=PolicyType.DATA_PROTECTION,
                frameworks=[ComplianceFramework.GDPR],
                conditions={
                    "data_classification": ["personal", "sensitive"],
                    "consent_obtained": True
                },
                violation_severity=ViolationSeverity.HIGH,
                violation_message="Personal data processed without consent"
            ),
            PolicyRule(
                rule_id="gdpr_data_retention",
                name="GDPR Data Retention Limits",
                description="Personal data must not be retained longer than necessary",
                policy_type=PolicyType.DATA_PROTECTION,
                frameworks=[ComplianceFramework.GDPR],
                conditions={
                    "data_age_days": {"max": 365},
                    "retention_justified": True
                },
                violation_severity=ViolationSeverity.MEDIUM
            )
        ]
        
        # HIPAA - Healthcare Data Protection
        hipaa_rules = [
            PolicyRule(
                rule_id="hipaa_encryption_required",
                name="HIPAA Encryption Required",
                description="PHI must be encrypted at rest and in transit",
                policy_type=PolicyType.ENCRYPTION,
                frameworks=[ComplianceFramework.HIPAA],
                conditions={
                    "data_classification": ["phi", "health"],
                    "encryption_required": True
                },
                violation_severity=ViolationSeverity.CRITICAL
            ),
            PolicyRule(
                rule_id="hipaa_access_logging",
                name="HIPAA Access Logging",
                description="All PHI access must be logged",
                policy_type=PolicyType.AUDIT_LOGGING,
                frameworks=[ComplianceFramework.HIPAA],
                conditions={
                    "audit_logging_enabled": True,
                    "data_classification": ["phi", "health"]
                },
                violation_severity=ViolationSeverity.HIGH
            )
        ]
        
        # SOX - Financial Controls
        sox_rules = [
            PolicyRule(
                rule_id="sox_segregation_duties",
                name="SOX Segregation of Duties",
                description="Financial data access requires segregation of duties",
                policy_type=PolicyType.ACCESS_CONTROL,
                frameworks=[ComplianceFramework.SOX],
                conditions={
                    "data_classification": ["financial"],
                    "segregation_enforced": True
                },
                violation_severity=ViolationSeverity.HIGH
            ),
            PolicyRule(
                rule_id="sox_audit_trail",
                name="SOX Audit Trail",
                description="All financial data changes must be audited",
                policy_type=PolicyType.AUDIT_LOGGING,
                frameworks=[ComplianceFramework.SOX],
                conditions={
                    "audit_logging_enabled": True,
                    "change_tracking": True
                },
                violation_severity=ViolationSeverity.CRITICAL
            )
        ]
        
        # Add all default rules
        for rule in gdpr_rules + hipaa_rules + sox_rules:
            self.policy_engine.add_rule(rule)
        
        # Create compliance rules
        self.compliance_rules["gdpr_data_protection"] = ComplianceRule(
            rule_id="gdpr_data_protection",
            framework=ComplianceFramework.GDPR,
            control_id="Article 6 & 7",
            policy_rules=["gdpr_consent_required", "gdpr_data_retention"]
        )
        
        self.compliance_rules["hipaa_safeguards"] = ComplianceRule(
            rule_id="hipaa_safeguards",
            framework=ComplianceFramework.HIPAA,
            control_id="164.312",
            policy_rules=["hipaa_encryption_required", "hipaa_access_logging"]
        )
        
        self.compliance_rules["sox_controls"] = ComplianceRule(
            rule_id="sox_controls",
            framework=ComplianceFramework.SOX,
            control_id="Section 404",
            policy_rules=["sox_segregation_duties", "sox_audit_trail"]
        )
    
    def evaluate_compliance(self, context: Dict[str, Any], 
                          frameworks: List[ComplianceFramework] = None) -> List[PolicyViolation]:
        """Evaluate compliance policies against context"""
        if frameworks:
            # Filter rules by frameworks
            rule_types = []
            for rule in self.policy_engine.rules.values():
                if any(fw in rule.frameworks for fw in frameworks):
                    if rule.policy_type not in rule_types:
                        rule_types.append(rule.policy_type)
            violations = self.policy_engine.evaluate_rules(context, rule_types)
        else:
            violations = self.policy_engine.evaluate_rules(context)
        
        # Store violations
        self.violations.extend(violations)
        
        # Log violations
        for violation in violations:
            self.logger.warning(f"Policy violation: {violation.rule_id} - {violation.message}")
        
        return violations
    
    def assess_compliance_framework(self, framework: ComplianceFramework,
                                  assessment_scope: Dict[str, Any] = None) -> ComplianceAssessment:
        """Perform comprehensive compliance assessment for a framework"""
        assessment = ComplianceAssessment(
            assessment_id=f"{framework.value}_{datetime.utcnow().timestamp()}",
            framework=framework,
            assessment_scope=assessment_scope or {}
        )
        
        # Get relevant compliance rules
        framework_rules = [
            rule for rule in self.compliance_rules.values()
            if rule.framework == framework
        ]
        
        total_controls = len(framework_rules)
        compliant_controls = 0
        
        for comp_rule in framework_rules:
            # Assess each control
            control_violations = []
            for policy_rule_id in comp_rule.policy_rules:
                if policy_rule_id in self.policy_engine.rules:
                    # Get recent violations for this rule
                    recent_violations = [
                        v for v in self.violations
                        if v.rule_id == policy_rule_id and
                        v.timestamp > datetime.utcnow() - timedelta(days=30) and
                        not v.is_resolved
                    ]
                    control_violations.extend(recent_violations)
            
            # Determine control status
            if not control_violations:
                control_status = ComplianceStatus.COMPLIANT
                compliant_controls += 1
            elif len(control_violations) == 1:
                control_status = ComplianceStatus.PARTIALLY_COMPLIANT
                compliant_controls += 0.5
            else:
                control_status = ComplianceStatus.NON_COMPLIANT
            
            assessment.control_results[comp_rule.control_id] = {
                "status": control_status,
                "violations": control_violations,
                "last_assessed": datetime.utcnow()
            }
        
        # Calculate overall score and status
        if total_controls > 0:
            assessment.score = (compliant_controls / total_controls) * 100
            
            if assessment.score >= 95:
                assessment.overall_status = ComplianceStatus.COMPLIANT
            elif assessment.score >= 80:
                assessment.overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
            else:
                assessment.overall_status = ComplianceStatus.NON_COMPLIANT
        
        # Generate recommendations
        assessment.recommendations = self._generate_recommendations(framework, assessment)
        
        # Store assessment
        self.assessments[assessment.assessment_id] = assessment
        
        self.logger.info(f"Completed compliance assessment for {framework.value}: {assessment.score:.1f}%")
        
        return assessment
    
    def _generate_recommendations(self, framework: ComplianceFramework, 
                                assessment: ComplianceAssessment) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        for control_id, control_result in assessment.control_results.items():
            if control_result["status"] != ComplianceStatus.COMPLIANT:
                violations = control_result["violations"]
                
                for violation in violations:
                    # Get rule for remediation steps
                    if violation.rule_id in self.policy_engine.rules:
                        rule = self.policy_engine.rules[violation.rule_id]
                        recommendations.extend(rule.remediation_steps)
        
        # Add framework-specific recommendations
        if framework == ComplianceFramework.GDPR:
            recommendations.extend([
                "Implement data subject rights management",
                "Conduct privacy impact assessments",
                "Establish data breach notification procedures"
            ])
        elif framework == ComplianceFramework.HIPAA:
            recommendations.extend([
                "Implement workforce training programs",
                "Establish business associate agreements",
                "Conduct risk assessments"
            ])
        elif framework == ComplianceFramework.SOX:
            recommendations.extend([
                "Implement change management controls",
                "Establish IT general controls",
                "Document financial reporting processes"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def resolve_violation(self, violation_id: str, resolved_by: str, 
                         resolution_notes: str = ""):
        """Mark a violation as resolved"""
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.is_resolved = True
                violation.resolved_at = datetime.utcnow()
                violation.resolved_by = resolved_by
                violation.resolution_notes = resolution_notes
                
                self.logger.info(f"Resolved violation: {violation_id}")
                return True
        
        return False
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        active_violations = [v for v in self.violations if not v.is_resolved]
        
        # Violations by severity
        violations_by_severity = {}
        for violation in active_violations:
            severity = violation.severity.value
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
        
        # Violations by framework
        violations_by_framework = {}
        for violation in active_violations:
            # Find which frameworks this rule belongs to
            if violation.rule_id in self.policy_engine.rules:
                rule = self.policy_engine.rules[violation.rule_id]
                for framework in rule.frameworks:
                    fw_name = framework.value
                    violations_by_framework[fw_name] = violations_by_framework.get(fw_name, 0) + 1
        
        # Recent assessments
        recent_assessments = sorted(
            self.assessments.values(),
            key=lambda a: a.assessment_date,
            reverse=True
        )[:5]
        
        return {
            "total_violations": len(active_violations),
            "critical_violations": len([v for v in active_violations if v.severity == ViolationSeverity.CRITICAL]),
            "violations_by_severity": violations_by_severity,
            "violations_by_framework": violations_by_framework,
            "recent_assessments": [
                {
                    "framework": a.framework.value,
                    "score": a.score,
                    "status": a.overall_status.value,
                    "date": a.assessment_date.isoformat()
                }
                for a in recent_assessments
            ],
            "compliance_frameworks": [fw.value for fw in ComplianceFramework],
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def export_compliance_report(self, framework: ComplianceFramework = None,
                               format_type: str = "json") -> str:
        """Export compliance report"""
        if framework:
            # Framework-specific report
            assessments = [a for a in self.assessments.values() if a.framework == framework]
            violations = [
                v for v in self.violations
                if v.rule_id in self.policy_engine.rules and
                framework in self.policy_engine.rules[v.rule_id].frameworks
            ]
        else:
            # All frameworks report
            assessments = list(self.assessments.values())
            violations = self.violations
        
        report_data = {
            "report_generated": datetime.utcnow().isoformat(),
            "framework": framework.value if framework else "all",
            "summary": {
                "total_assessments": len(assessments),
                "total_violations": len(violations),
                "unresolved_violations": len([v for v in violations if not v.is_resolved])
            },
            "assessments": [
                {
                    "assessment_id": a.assessment_id,
                    "framework": a.framework.value,
                    "date": a.assessment_date.isoformat(),
                    "score": a.score,
                    "status": a.overall_status.value,
                    "recommendations": a.recommendations
                }
                for a in assessments
            ],
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "rule_id": v.rule_id,
                    "severity": v.severity.value,
                    "message": v.message,
                    "timestamp": v.timestamp.isoformat(),
                    "is_resolved": v.is_resolved,
                    "resolved_at": v.resolved_at.isoformat() if v.resolved_at else None
                }
                for v in violations
            ]
        }
        
        if format_type == "json":
            return json.dumps(report_data, indent=2)
        elif format_type == "yaml":
            return yaml.dump(report_data, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def schedule_assessment(self, framework: ComplianceFramework, 
                          frequency: str = "monthly"):
        """Schedule regular compliance assessment"""
        if framework.value not in self.compliance_rules:
            return False
        
        rule = list(self.compliance_rules.values())[0]  # Get first rule for framework
        rule.assessment_frequency = frequency
        
        # Calculate next assessment date
        if frequency == "daily":
            next_date = datetime.utcnow() + timedelta(days=1)
        elif frequency == "weekly":
            next_date = datetime.utcnow() + timedelta(weeks=1)
        elif frequency == "monthly":
            next_date = datetime.utcnow() + timedelta(days=30)
        elif frequency == "quarterly":
            next_date = datetime.utcnow() + timedelta(days=90)
        elif frequency == "annually":
            next_date = datetime.utcnow() + timedelta(days=365)
        else:
            return False
        
        rule.next_assessment = next_date
        
        self.logger.info(f"Scheduled {frequency} assessment for {framework.value}")
        return True
    
    def get_due_assessments(self) -> List[ComplianceRule]:
        """Get compliance rules with due assessments"""
        due_assessments = []
        current_time = datetime.utcnow()
        
        for rule in self.compliance_rules.values():
            if rule.next_assessment and rule.next_assessment <= current_time:
                due_assessments.append(rule)
        
        return due_assessments