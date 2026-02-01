"""
Enterprise Guardrails Framework

A comprehensive enterprise-grade guardrails system for RAG applications
that ensures compliance, security, and quality of LLM responses.

Components:
- Citation Enforcer: Validates and enforces source citations
- PII Detector: Detects and redacts personally identifiable information
- Topic Filter: Blocks/allows content based on topic classification
- Compliance Logger: Audit-ready logging for governance
- Guardrails Orchestrator: Coordinates all guardrail checks
"""

from .citation_enforcer import CitationEnforcer, CitationValidator
from .pii_detector import PIIDetector, PIIRedactor
from .topic_filter import TopicFilter, TopicClassifier
from .compliance_logger import ComplianceLogger, AuditEvent
from .guardrails_orchestrator import GuardrailsOrchestrator, GuardrailResult

__all__ = [
    "CitationEnforcer",
    "CitationValidator", 
    "PIIDetector",
    "PIIRedactor",
    "TopicFilter",
    "TopicClassifier",
    "ComplianceLogger",
    "AuditEvent",
    "GuardrailsOrchestrator",
    "GuardrailResult"
]

__version__ = "1.0.0"