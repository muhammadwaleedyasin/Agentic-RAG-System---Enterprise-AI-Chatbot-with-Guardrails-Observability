"""
Guardrails Orchestrator

Central orchestration system that coordinates all guardrail checks and
ensures enterprise compliance policies are enforced consistently.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .citation_enforcer import CitationEnforcer, CitationValidationResult
from .pii_detector import PIIDetector, PIIDetectionResult
from .topic_filter import TopicFilter, TopicFilterResult
from .compliance_logger import ComplianceLogger, EventType, Severity
from ..config.compliance_config import get_compliance_config, ComplianceLevel, ActionType


class GuardrailType(Enum):
    """Types of guardrail checks"""
    PII_DETECTION = "pii_detection"
    CITATION_ENFORCEMENT = "citation_enforcement"
    TOPIC_FILTERING = "topic_filtering"
    CONFIDENCE_CHECK = "confidence_check"
    CONTENT_SAFETY = "content_safety"


@dataclass
class GuardrailResult:
    """Comprehensive result from all guardrail checks"""
    is_compliant: bool
    final_response: str
    confidence_score: float
    violations: List[str]
    warnings: List[str]
    actions_taken: List[str]
    
    # Individual check results
    pii_result: Optional[PIIDetectionResult] = None
    citation_result: Optional[CitationValidationResult] = None
    topic_result: Optional[TopicFilterResult] = None
    
    # Metadata
    processing_time_ms: Optional[float] = None
    guardrails_applied: List[GuardrailType] = None
    compliance_level: Optional[ComplianceLevel] = None
    audit_event_ids: List[str] = None


class GuardrailsOrchestrator:
    """Orchestrates all guardrail checks for enterprise compliance"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_compliance_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize individual guardrail components
        self.pii_detector = PIIDetector(self.config.pii_detection)
        self.citation_enforcer = CitationEnforcer(self.config.citation)
        self.topic_filter = TopicFilter(self.config.topic_filter)
        self.compliance_logger = ComplianceLogger(self.config.audit)
        
        # Track processing statistics
        self.stats = {
            "total_requests": 0,
            "compliant_requests": 0,
            "blocked_requests": 0,
            "violations_by_type": {},
            "avg_processing_time": 0.0
        }
    
    def process_response(self,
                        response_text: str,
                        retrieved_sources: List[Dict],
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        query_text: Optional[str] = None,
                        context: Optional[Dict] = None) -> GuardrailResult:
        """Process response through all enabled guardrails"""
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        
        violations = []
        warnings = []
        actions_taken = []
        audit_event_ids = []
        guardrails_applied = []
        
        # Working copy of response text that may be modified
        processed_response = response_text
        
        # Initialize results
        pii_result = None
        citation_result = None
        topic_result = None
        
        try:
            # 1. PII Detection and Redaction
            if self.config.pii_detection.enabled:
                guardrails_applied.append(GuardrailType.PII_DETECTION)
                processed_response, pii_result = self._check_pii(
                    processed_response, user_id, session_id, query_text
                )
                
                if pii_result.has_pii:
                    violations.extend([f"PII: {e.entity_type}" for e in pii_result.entities_found])
                    actions_taken.append(f"Redacted {len(pii_result.entities_found)} PII entities")
                    
                    # Log PII detection
                    event_id = self.compliance_logger.log_pii_detection(
                        entities_found=[e.__dict__ for e in pii_result.entities_found],
                        user_id=user_id,
                        session_id=session_id,
                        query_text=query_text
                    )
                    audit_event_ids.append(event_id)
            
            # 2. Topic Filtering
            if self.config.topic_filter.enabled:
                guardrails_applied.append(GuardrailType.TOPIC_FILTERING)
                topic_result = self._check_topics(
                    processed_response, user_id, session_id, query_text, context
                )
                
                if not topic_result.is_allowed:
                    violations.extend(topic_result.violations)
                    actions_taken.append(f"Blocked topics: {', '.join(topic_result.blocked_topics)}")
                    
                    # Handle based on action configuration
                    if self.config.topic_filter.action_on_blocked_topic == ActionType.BLOCK:
                        # Block the entire response
                        processed_response = "I cannot provide information on this topic due to content policy restrictions."
                    elif self.config.topic_filter.action_on_blocked_topic == ActionType.REDACT:
                        # Keep response but add warning
                        processed_response += "\n\n[Content filtered for policy compliance]"
                    
                    # Log topic filtering
                    event_id = self.compliance_logger.log_topic_filter(
                        blocked_topics=topic_result.blocked_topics,
                        topic_classifications=[t.__dict__ for t in topic_result.all_topics],
                        action_taken=topic_result.action_taken,
                        user_id=user_id,
                        session_id=session_id,
                        query_text=query_text
                    )
                    audit_event_ids.append(event_id)
            
            # 3. Citation Enforcement
            if self.config.citation.enforce_citations and retrieved_sources:
                guardrails_applied.append(GuardrailType.CITATION_ENFORCEMENT)
                processed_response, citation_result = self._check_citations(
                    processed_response, retrieved_sources, user_id, session_id
                )
                
                if not citation_result.is_valid:
                    violations.extend(citation_result.violations)
                    warnings.extend(citation_result.suggestions)
                    actions_taken.append("Citation enforcement applied")
                    
                    # Log citation violations
                    event_id = self.compliance_logger.log_citation_violation(
                        violations=citation_result.violations,
                        citations_found=len(citation_result.citations_found),
                        citations_required=self.config.citation.min_citations_required,
                        user_id=user_id,
                        session_id=session_id
                    )
                    audit_event_ids.append(event_id)
            
            # 4. Confidence Check
            confidence_score = self._calculate_overall_confidence(
                pii_result, citation_result, topic_result
            )
            
            abstained = False
            if self.config.confidence.enable_abstain:
                guardrails_applied.append(GuardrailType.CONFIDENCE_CHECK)
                if confidence_score < self.config.confidence.abstain_threshold:
                    processed_response = self.config.confidence.abstain_message
                    actions_taken.append("Response abstained due to low confidence")
                    abstained = True
                
                # Log confidence check
                event_id = self.compliance_logger.log_confidence_check(
                    confidence_score=confidence_score,
                    threshold=self.config.confidence.abstain_threshold,
                    abstained=abstained,
                    user_id=user_id,
                    session_id=session_id
                )
                audit_event_ids.append(event_id)
            
            # Determine final compliance status
            is_compliant = self._assess_compliance(violations, confidence_score)
            
            # Handle compliance level enforcement
            if not is_compliant:
                processed_response = self._handle_non_compliance(
                    processed_response, violations, actions_taken
                )
            
            # Update statistics
            if is_compliant:
                self.stats["compliant_requests"] += 1
            else:
                self.stats["blocked_requests"] += 1
            
            # Track violations by type
            for violation in violations:
                violation_type = violation.split(":")[0] if ":" in violation else "general"
                self.stats["violations_by_type"][violation_type] = \
                    self.stats["violations_by_type"].get(violation_type, 0) + 1
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (self.stats["total_requests"] - 1) + processing_time) 
                / self.stats["total_requests"]
            )
            
            return GuardrailResult(
                is_compliant=is_compliant,
                final_response=processed_response,
                confidence_score=confidence_score,
                violations=violations,
                warnings=warnings,
                actions_taken=actions_taken,
                pii_result=pii_result,
                citation_result=citation_result,
                topic_result=topic_result,
                processing_time_ms=processing_time,
                guardrails_applied=guardrails_applied,
                compliance_level=self.config.compliance_level,
                audit_event_ids=audit_event_ids
            )
            
        except Exception as e:
            self.logger.error(f"Error in guardrails processing: {e}")
            
            # Log security alert for processing errors
            event_id = self.compliance_logger.log_security_alert(
                alert_type="guardrails_processing_error",
                details={"error": str(e), "response_length": len(response_text)},
                severity=Severity.HIGH,
                user_id=user_id
            )
            
            return GuardrailResult(
                is_compliant=False,
                final_response="An error occurred while processing your request. Please try again.",
                confidence_score=0.0,
                violations=["Processing error"],
                warnings=["Guardrails processing failed"],
                actions_taken=["Request blocked due to processing error"],
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                guardrails_applied=guardrails_applied,
                compliance_level=self.config.compliance_level,
                audit_event_ids=[event_id]
            )
    
    def _check_pii(self,
                   text: str,
                   user_id: Optional[str],
                   session_id: Optional[str],
                   query_text: Optional[str]) -> Tuple[str, PIIDetectionResult]:
        """Check for PII and apply redaction"""
        pii_result = self.pii_detector.detect_pii(text)
        
        if pii_result.has_pii:
            # Apply action based on configuration
            action = self.config.pii_detection.action_on_detection
            
            if action == ActionType.BLOCK:
                return "Response blocked due to PII detection.", pii_result
            elif action == ActionType.REDACT:
                return pii_result.redacted_text, pii_result
            else:  # WARN or LOG_ONLY
                return text, pii_result
        
        return text, pii_result
    
    def _check_topics(self,
                     text: str,
                     user_id: Optional[str],
                     session_id: Optional[str],
                     query_text: Optional[str],
                     context: Optional[Dict]) -> TopicFilterResult:
        """Check topics and apply filtering"""
        return self.topic_filter.filter_content(text, context)
    
    def _check_citations(self,
                        text: str,
                        retrieved_sources: List[Dict],
                        user_id: Optional[str],
                        session_id: Optional[str]) -> Tuple[str, CitationValidationResult]:
        """Check and enforce citations"""
        return self.citation_enforcer.enforce_citations(text, retrieved_sources)
    
    def _calculate_overall_confidence(self,
                                    pii_result: Optional[PIIDetectionResult],
                                    citation_result: Optional[CitationValidationResult],
                                    topic_result: Optional[TopicFilterResult]) -> float:
        """Calculate overall confidence score"""
        factors = self.config.confidence.confidence_factors
        total_confidence = 0.0
        total_weight = 0.0
        
        # Citation quality factor
        if citation_result and factors.get("citation_quality", 0) > 0:
            citation_confidence = citation_result.confidence_score
            weight = factors["citation_quality"]
            total_confidence += citation_confidence * weight
            total_weight += weight
        
        # Topic relevance factor
        if topic_result and factors.get("topic_relevance", 0) > 0:
            topic_confidence = 1.0 if topic_result.is_allowed else 0.0
            weight = factors["topic_relevance"]
            total_confidence += topic_confidence * weight
            total_weight += weight
        
        # PII safety factor (inverse of PII risk)
        if pii_result and factors.get("pii_safety", 0) > 0:
            pii_confidence = 1.0 - (len(pii_result.entities_found) * 0.1)
            pii_confidence = max(pii_confidence, 0.0)
            weight = factors.get("pii_safety", 0.2)
            total_confidence += pii_confidence * weight
            total_weight += weight
        
        # Default LLM confidence (placeholder)
        if factors.get("llm_confidence", 0) > 0:
            llm_confidence = 0.8  # Would come from actual LLM confidence scoring
            weight = factors["llm_confidence"]
            total_confidence += llm_confidence * weight
            total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.5
    
    def _assess_compliance(self, violations: List[str], confidence_score: float) -> bool:
        """Assess overall compliance based on violations and confidence"""
        # Critical violations always fail compliance
        critical_violations = [v for v in violations if any(
            term in v.lower() for term in ["ssn", "credit_card", "blocked"]
        )]
        
        if critical_violations and self.config.compliance_level == ComplianceLevel.STRICT:
            return False
        
        # Check confidence threshold
        if confidence_score < self.config.confidence.min_confidence_threshold:
            return False
        
        # Moderate compliance allows some violations
        if self.config.compliance_level == ComplianceLevel.MODERATE:
            return len(violations) <= 2
        
        # Permissive compliance allows most content
        if self.config.compliance_level == ComplianceLevel.PERMISSIVE:
            return len(critical_violations) == 0
        
        # Audit-only mode always passes
        if self.config.compliance_level == ComplianceLevel.AUDIT_ONLY:
            return True
        
        # Strict compliance requires no violations
        return len(violations) == 0
    
    def _handle_non_compliance(self,
                              response: str,
                              violations: List[str],
                              actions_taken: List[str]) -> str:
        """Handle non-compliant responses based on configuration"""
        if self.config.fail_fast:
            return (
                "This response does not meet enterprise compliance requirements. "
                "Please rephrase your question or contact your administrator."
            )
        
        # Add compliance footer
        compliance_footer = (
            f"\n\n⚠️ Compliance Notice: This response has been processed through "
            f"enterprise guardrails. {len(violations)} policy considerations were identified."
        )
        
        return response + compliance_footer
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current guardrails system status"""
        return {
            "guardrails_enabled": {
                "pii_detection": self.config.pii_detection.enabled,
                "citation_enforcement": self.config.citation.enforce_citations,
                "topic_filtering": self.config.topic_filter.enabled,
                "confidence_checks": self.config.confidence.enable_abstain
            },
            "compliance_level": self.config.compliance_level.value,
            "statistics": self.stats,
            "thresholds": {
                "pii_detection": self.config.pii_detection.detection_threshold,
                "topic_classification": self.config.topic_filter.topic_detection_threshold,
                "confidence_minimum": self.config.confidence.min_confidence_threshold,
                "abstain_threshold": self.config.confidence.abstain_threshold
            }
        }
    
    def update_configuration(self, config_updates: Dict[str, Any]):
        """Update guardrails configuration dynamically"""
        # Log configuration change
        self.compliance_logger.log_security_alert(
            alert_type="configuration_change",
            details={"updates": config_updates},
            severity=Severity.MEDIUM
        )
        
        # Apply updates (this would need proper validation in production)
        self.logger.info(f"Configuration updated: {config_updates}")


# Example usage and testing
if __name__ == "__main__":
    # Test the guardrails orchestrator
    orchestrator = GuardrailsOrchestrator()
    
    test_response = """
    Based on our internal policy documents, remote work is permitted with manager approval [1].
    For questions, contact john.doe@company.com or call 555-123-4567.
    Your employee ID 123-45-6789 is required for the application.
    """
    
    test_sources = [
        {"id": "1", "title": "Remote Work Policy", "content": "Remote work guidelines..."}
    ]
    
    result = orchestrator.process_response(
        response_text=test_response,
        retrieved_sources=test_sources,
        user_id="test_user",
        session_id="test_session",
        query_text="What is the remote work policy?"
    )
    
    print(f"Compliant: {result.is_compliant}")
    print(f"Final Response: {result.final_response[:100]}...")
    print(f"Violations: {result.violations}")
    print(f"Actions Taken: {result.actions_taken}")
    print(f"Confidence Score: {result.confidence_score}")