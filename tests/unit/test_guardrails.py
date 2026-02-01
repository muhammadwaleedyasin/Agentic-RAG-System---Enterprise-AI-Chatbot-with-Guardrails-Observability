"""Unit tests for guardrails components."""

import pytest
from unittest.mock import MagicMock, patch
import re

from src.guardrails.pii_detector import PIIDetector
from src.guardrails.topic_filter import TopicFilter
from src.guardrails.citation_enforcer import CitationEnforcer
from src.guardrails.compliance_logger import ComplianceLogger
from src.guardrails.guardrails_orchestrator import GuardrailsOrchestrator


class TestPIIDetector:
    """Test cases for PII (Personally Identifiable Information) detector."""

    @pytest.fixture
    def pii_detector(self):
        """Create PIIDetector instance."""
        return PIIDetector()

    @pytest.mark.unit
    def test_detect_email_addresses(self, pii_detector):
        """Test detection of email addresses."""
        text = "Contact John at john.doe@company.com or jane@example.org for more information."
        
        pii_results = pii_detector.detect_pii(text)
        
        assert len(pii_results) == 2
        assert any(result["type"] == "email" and result["value"] == "john.doe@company.com" for result in pii_results)
        assert any(result["type"] == "email" and result["value"] == "jane@example.org" for result in pii_results)

    @pytest.mark.unit
    def test_detect_phone_numbers(self, pii_detector):
        """Test detection of phone numbers."""
        text = "Call me at (555) 123-4567 or 555.987.6543 or +1-800-555-0123."
        
        pii_results = pii_detector.detect_pii(text)
        
        phone_numbers = [result for result in pii_results if result["type"] == "phone"]
        assert len(phone_numbers) >= 2  # Should detect multiple formats

    @pytest.mark.unit
    def test_detect_social_security_numbers(self, pii_detector):
        """Test detection of SSN patterns."""
        text = "SSN: 123-45-6789 or social security number 987654321."
        
        pii_results = pii_detector.detect_pii(text)
        
        ssn_results = [result for result in pii_results if result["type"] == "ssn"]
        assert len(ssn_results) >= 1

    @pytest.mark.unit
    def test_detect_credit_card_numbers(self, pii_detector):
        """Test detection of credit card numbers."""
        text = "Payment with card 4532-1234-5678-9012 or 5555555555554444."
        
        pii_results = pii_detector.detect_pii(text)
        
        cc_results = [result for result in pii_results if result["type"] == "credit_card"]
        assert len(cc_results) >= 1

    @pytest.mark.unit
    def test_redact_pii(self, pii_detector):
        """Test PII redaction functionality."""
        text = "Contact John at john@company.com or call (555) 123-4567."
        
        redacted_text = pii_detector.redact_pii(text)
        
        assert "john@company.com" not in redacted_text
        assert "(555) 123-4567" not in redacted_text
        assert "[REDACTED-EMAIL]" in redacted_text or "[EMAIL]" in redacted_text
        assert "[REDACTED-PHONE]" in redacted_text or "[PHONE]" in redacted_text

    @pytest.mark.unit
    def test_custom_pii_patterns(self, pii_detector):
        """Test custom PII pattern detection."""
        # Add custom pattern for employee IDs
        custom_patterns = {
            "employee_id": r"EMP-\d{4,6}"
        }
        
        pii_detector.add_custom_patterns(custom_patterns)
        
        text = "Employee ID: EMP-12345 has access to the system."
        pii_results = pii_detector.detect_pii(text)
        
        assert any(result["type"] == "employee_id" for result in pii_results)

    @pytest.mark.unit
    def test_pii_confidence_scoring(self, pii_detector):
        """Test PII detection confidence scoring."""
        # Clear email (should have high confidence)
        text1 = "Email: user@domain.com"
        # Potential false positive (should have lower confidence)
        text2 = "Version 1.2.3@beta"
        
        results1 = pii_detector.detect_pii(text1)
        results2 = pii_detector.detect_pii(text2)
        
        if results1:
            assert results1[0]["confidence"] > 0.8
        
        # Should either not detect or have low confidence for false positive
        if results2:
            assert results2[0]["confidence"] < 0.7

    @pytest.mark.unit
    def test_no_pii_in_clean_text(self, pii_detector):
        """Test that clean text returns no PII detections."""
        text = "This is a clean text with no personally identifiable information."
        
        pii_results = pii_detector.detect_pii(text)
        
        assert len(pii_results) == 0


class TestTopicFilter:
    """Test cases for topic filtering."""

    @pytest.fixture
    def topic_filter(self):
        """Create TopicFilter instance."""
        allowed_topics = ["technology", "science", "education"]
        blocked_topics = ["politics", "adult_content", "violence"]
        return TopicFilter(allowed_topics=allowed_topics, blocked_topics=blocked_topics)

    @pytest.mark.unit
    def test_topic_filter_initialization(self, topic_filter):
        """Test topic filter initialization."""
        assert "technology" in topic_filter.allowed_topics
        assert "politics" in topic_filter.blocked_topics

    @pytest.mark.unit
    def test_allow_valid_topics(self, topic_filter):
        """Test allowing valid topics."""
        # Mock topic classification
        with patch.object(topic_filter, 'classify_topic') as mock_classify:
            mock_classify.return_value = {"topic": "technology", "confidence": 0.9}
            
            result = topic_filter.is_topic_allowed("Discuss machine learning algorithms.")
            
            assert result["allowed"] is True
            assert result["topic"] == "technology"

    @pytest.mark.unit
    def test_block_invalid_topics(self, topic_filter):
        """Test blocking invalid topics."""
        with patch.object(topic_filter, 'classify_topic') as mock_classify:
            mock_classify.return_value = {"topic": "politics", "confidence": 0.85}
            
            result = topic_filter.is_topic_allowed("Political discussion content.")
            
            assert result["allowed"] is False
            assert result["topic"] == "politics"

    @pytest.mark.unit
    def test_uncertain_topic_handling(self, topic_filter):
        """Test handling of uncertain topic classification."""
        with patch.object(topic_filter, 'classify_topic') as mock_classify:
            mock_classify.return_value = {"topic": "unknown", "confidence": 0.3}
            
            # Low confidence should result in cautious handling
            result = topic_filter.is_topic_allowed("Ambiguous content.", confidence_threshold=0.7)
            
            # Should either be allowed (default behavior) or handled cautiously
            assert "allowed" in result

    @pytest.mark.unit
    def test_topic_classification_keywords(self, topic_filter):
        """Test keyword-based topic classification."""
        tech_text = "Machine learning and artificial intelligence are transforming software development."
        
        result = topic_filter.classify_topic(tech_text)
        
        assert result["topic"] in ["technology", "science"] or result["confidence"] > 0.5

    @pytest.mark.unit
    def test_update_topic_lists(self, topic_filter):
        """Test updating allowed and blocked topic lists."""
        # Add new allowed topic
        topic_filter.add_allowed_topic("medicine")
        assert "medicine" in topic_filter.allowed_topics
        
        # Add new blocked topic
        topic_filter.add_blocked_topic("gambling")
        assert "gambling" in topic_filter.blocked_topics
        
        # Remove topic
        topic_filter.remove_allowed_topic("education")
        assert "education" not in topic_filter.allowed_topics


class TestCitationEnforcer:
    """Test cases for citation enforcement."""

    @pytest.fixture
    def citation_enforcer(self):
        """Create CitationEnforcer instance."""
        return CitationEnforcer(require_citations=True, min_sources=2)

    @pytest.mark.unit
    def test_citation_enforcer_initialization(self, citation_enforcer):
        """Test citation enforcer initialization."""
        assert citation_enforcer.require_citations is True
        assert citation_enforcer.min_sources == 2

    @pytest.mark.unit
    def test_enforce_citations_present(self, citation_enforcer):
        """Test enforcement when citations are present."""
        response = "Machine learning is a subset of AI [1]. It uses algorithms to learn from data [2]."
        sources = [
            {"id": "source1", "title": "ML Overview"},
            {"id": "source2", "title": "AI Algorithms"}
        ]
        
        result = citation_enforcer.enforce_citations(response, sources)
        
        assert result["compliant"] is True
        assert result["citation_count"] >= 2

    @pytest.mark.unit
    def test_enforce_citations_missing(self, citation_enforcer):
        """Test enforcement when citations are missing."""
        response = "Machine learning is a subset of AI. It uses algorithms to learn from data."
        sources = [
            {"id": "source1", "title": "ML Overview"},
            {"id": "source2", "title": "AI Algorithms"}
        ]
        
        result = citation_enforcer.enforce_citations(response, sources)
        
        assert result["compliant"] is False
        assert "missing_citations" in result["issues"]

    @pytest.mark.unit
    def test_auto_add_citations(self, citation_enforcer):
        """Test automatic citation addition."""
        response = "Machine learning is a subset of AI. It uses algorithms to learn from data."
        sources = [
            {"id": "source1", "title": "ML Overview", "content": "machine learning subset"},
            {"id": "source2", "title": "AI Algorithms", "content": "algorithms learn data"}
        ]
        
        enhanced_response = citation_enforcer.add_citations(response, sources)
        
        assert "[1]" in enhanced_response or "(1)" in enhanced_response
        assert enhanced_response != response  # Should be modified

    @pytest.mark.unit
    def test_citation_format_validation(self, citation_enforcer):
        """Test citation format validation."""
        valid_formats = [
            "Text with citation [1]",
            "Text with citation (Smith, 2023)",
            "Text with citation¹",
        ]
        
        for text in valid_formats:
            citations = citation_enforcer.extract_citations(text)
            assert len(citations) >= 1

    @pytest.mark.unit
    def test_duplicate_citation_handling(self, citation_enforcer):
        """Test handling of duplicate citations."""
        response = "ML is important [1]. AI is related [1]. Deep learning uses neural networks [1]."
        sources = [{"id": "source1", "title": "AI Overview"}]
        
        result = citation_enforcer.enforce_citations(response, sources)
        
        # Should handle duplicate citations appropriately
        assert result["unique_sources"] == 1

    @pytest.mark.unit
    def test_source_quality_validation(self, citation_enforcer):
        """Test source quality validation."""
        high_quality_source = {
            "id": "source1",
            "title": "Peer-reviewed ML Paper",
            "source_type": "academic",
            "confidence": 0.95
        }
        
        low_quality_source = {
            "id": "source2", 
            "title": "Random Blog Post",
            "source_type": "blog",
            "confidence": 0.3
        }
        
        result1 = citation_enforcer.validate_source_quality(high_quality_source)
        result2 = citation_enforcer.validate_source_quality(low_quality_source)
        
        assert result1["quality_score"] > result2["quality_score"]


class TestComplianceLogger:
    """Test cases for compliance logging."""

    @pytest.fixture
    def compliance_logger(self):
        """Create ComplianceLogger instance."""
        return ComplianceLogger(log_level="INFO")

    @pytest.mark.unit
    def test_compliance_logger_initialization(self, compliance_logger):
        """Test compliance logger initialization."""
        assert compliance_logger.log_level == "INFO"

    @pytest.mark.unit
    def test_log_pii_detection(self, compliance_logger):
        """Test logging of PII detection events."""
        pii_event = {
            "event_type": "pii_detected",
            "pii_type": "email",
            "redacted": True,
            "timestamp": "2023-12-01T10:00:00Z"
        }
        
        with patch.object(compliance_logger, '_write_log') as mock_write:
            compliance_logger.log_pii_event(pii_event)
            mock_write.assert_called_once()

    @pytest.mark.unit
    def test_log_topic_filtering(self, compliance_logger):
        """Test logging of topic filtering events."""
        filter_event = {
            "event_type": "topic_filtered",
            "topic": "politics",
            "action": "blocked",
            "confidence": 0.87
        }
        
        with patch.object(compliance_logger, '_write_log') as mock_write:
            compliance_logger.log_topic_filter_event(filter_event)
            mock_write.assert_called_once()

    @pytest.mark.unit
    def test_log_citation_compliance(self, compliance_logger):
        """Test logging of citation compliance events."""
        citation_event = {
            "event_type": "citation_check",
            "compliant": False,
            "issues": ["missing_citations"],
            "source_count": 3
        }
        
        with patch.object(compliance_logger, '_write_log') as mock_write:
            compliance_logger.log_citation_event(citation_event)
            mock_write.assert_called_once()

    @pytest.mark.unit
    def test_audit_trail_generation(self, compliance_logger):
        """Test audit trail generation."""
        events = [
            {"event_type": "pii_detected", "timestamp": "2023-12-01T10:00:00Z"},
            {"event_type": "topic_filtered", "timestamp": "2023-12-01T10:01:00Z"},
            {"event_type": "citation_check", "timestamp": "2023-12-01T10:02:00Z"}
        ]
        
        for event in events:
            compliance_logger.log_event(event)
        
        audit_trail = compliance_logger.get_audit_trail()
        
        assert len(audit_trail) == 3
        assert all(event["timestamp"] for event in audit_trail)

    @pytest.mark.unit
    def test_compliance_metrics(self, compliance_logger):
        """Test compliance metrics calculation."""
        # Simulate various compliance events
        events = [
            {"event_type": "pii_detected", "action": "redacted"},
            {"event_type": "topic_filtered", "action": "allowed"},
            {"event_type": "citation_check", "compliant": True},
            {"event_type": "citation_check", "compliant": False}
        ]
        
        for event in events:
            compliance_logger.log_event(event)
        
        metrics = compliance_logger.get_compliance_metrics()
        
        assert "pii_detection_rate" in metrics
        assert "topic_filter_rate" in metrics
        assert "citation_compliance_rate" in metrics


class TestGuardrailsOrchestrator:
    """Test cases for guardrails orchestrator."""

    @pytest.fixture
    def mock_components(self):
        """Create mock guardrail components."""
        pii_detector = MagicMock()
        topic_filter = MagicMock()
        citation_enforcer = MagicMock()
        compliance_logger = MagicMock()
        
        return {
            "pii_detector": pii_detector,
            "topic_filter": topic_filter,
            "citation_enforcer": citation_enforcer,
            "compliance_logger": compliance_logger
        }

    @pytest.fixture
    def orchestrator(self, mock_components):
        """Create GuardrailsOrchestrator instance."""
        return GuardrailsOrchestrator(
            pii_detector=mock_components["pii_detector"],
            topic_filter=mock_components["topic_filter"],
            citation_enforcer=mock_components["citation_enforcer"],
            compliance_logger=mock_components["compliance_logger"]
        )

    @pytest.mark.unit
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.pii_detector is not None
        assert orchestrator.topic_filter is not None
        assert orchestrator.citation_enforcer is not None
        assert orchestrator.compliance_logger is not None

    @pytest.mark.unit
    def test_process_query_guardrails(self, orchestrator, mock_components):
        """Test query processing through guardrails."""
        query = "What is machine learning?"
        
        # Mock responses
        mock_components["topic_filter"].is_topic_allowed.return_value = {
            "allowed": True,
            "topic": "technology"
        }
        
        result = orchestrator.process_query(query)
        
        assert result["allowed"] is True
        mock_components["topic_filter"].is_topic_allowed.assert_called_once_with(query)

    @pytest.mark.unit
    def test_process_response_guardrails(self, orchestrator, mock_components):
        """Test response processing through guardrails."""
        response = "Machine learning is a technology that learns from data [1]."
        sources = [{"id": "source1", "title": "ML Guide"}]
        
        # Mock responses
        mock_components["pii_detector"].detect_pii.return_value = []
        mock_components["citation_enforcer"].enforce_citations.return_value = {
            "compliant": True,
            "citation_count": 1
        }
        
        result = orchestrator.process_response(response, sources)
        
        assert result["compliant"] is True
        mock_components["pii_detector"].detect_pii.assert_called_once()
        mock_components["citation_enforcer"].enforce_citations.assert_called_once()

    @pytest.mark.unit
    def test_policy_violation_handling(self, orchestrator, mock_components):
        """Test handling of policy violations."""
        query = "Inappropriate content query"
        
        # Mock policy violation
        mock_components["topic_filter"].is_topic_allowed.return_value = {
            "allowed": False,
            "topic": "adult_content",
            "reason": "Content violates policy"
        }
        
        result = orchestrator.process_query(query)
        
        assert result["allowed"] is False
        assert "reason" in result
        mock_components["compliance_logger"].log_event.assert_called()

    @pytest.mark.unit
    def test_pii_redaction_workflow(self, orchestrator, mock_components):
        """Test PII redaction workflow."""
        response = "Contact john@company.com for more info."
        
        # Mock PII detection and redaction
        mock_components["pii_detector"].detect_pii.return_value = [
            {"type": "email", "value": "john@company.com", "start": 8, "end": 25}
        ]
        mock_components["pii_detector"].redact_pii.return_value = "Contact [REDACTED-EMAIL] for more info."
        
        result = orchestrator.process_response(response, [])
        
        assert "john@company.com" not in result["response"]
        mock_components["pii_detector"].redact_pii.assert_called_once()

    @pytest.mark.unit
    def test_guardrails_configuration(self, orchestrator):
        """Test guardrails configuration management."""
        config = {
            "pii_detection": {"enabled": True, "redaction_mode": "mask"},
            "topic_filtering": {"enabled": True, "strict_mode": False},
            "citation_enforcement": {"enabled": True, "min_sources": 2}
        }
        
        orchestrator.update_configuration(config)
        
        # Verify configuration is applied
        current_config = orchestrator.get_configuration()
        assert current_config["pii_detection"]["enabled"] is True

    @pytest.mark.unit
    def test_guardrails_bypass_for_admin(self, orchestrator):
        """Test guardrails bypass for administrative users."""
        query = "Administrative query"
        
        # Test with admin bypass
        result = orchestrator.process_query(query, user_role="admin", bypass_guardrails=True)
        
        assert result["bypassed"] is True
        
        # Test without admin bypass
        result = orchestrator.process_query(query, user_role="user", bypass_guardrails=False)
        
        assert result.get("bypassed", False) is False

    @pytest.mark.unit
    def test_guardrails_performance_monitoring(self, orchestrator):
        """Test guardrails performance monitoring."""
        import time
        
        # Process several requests to generate metrics
        for i in range(5):
            orchestrator.process_query(f"Test query {i}")
        
        performance_metrics = orchestrator.get_performance_metrics()
        
        assert "avg_processing_time" in performance_metrics
        assert "total_requests" in performance_metrics
        assert performance_metrics["total_requests"] == 5

    @pytest.mark.unit
    def test_custom_guardrail_integration(self, orchestrator):
        """Test integration of custom guardrail rules."""
        def custom_content_filter(text):
            # Custom rule: block content with excessive capitalization
            if text.count(text.upper()) > len(text) * 0.5:
                return {"allowed": False, "reason": "Excessive capitalization"}
            return {"allowed": True}
        
        orchestrator.add_custom_guardrail("content_filter", custom_content_filter)
        
        # Test with violating content
        result = orchestrator.process_query("THIS IS ALL CAPS TEXT!!!")
        
        # Should be blocked by custom rule
        assert result["allowed"] is False or "capitalization" in result.get("reason", "")