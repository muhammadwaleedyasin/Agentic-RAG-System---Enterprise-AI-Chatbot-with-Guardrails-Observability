"""
Enterprise Security Module for RAG System

This module provides comprehensive security features including:
- Role-based access control (RBAC)
- Document-level permissions
- Audit logging and monitoring
- Compliance enforcement
- Data loss prevention (DLP)
"""

from .document_access_control import DocumentAccessControl, Permission, AccessPolicy
from .audit_logger import AuditLogger, SecurityEvent, AuditConfig
from .compliance_engine import ComplianceEngine, PolicyEngine, ComplianceRule
from .data_loss_prevention import DataLossPreventionEngine, ContentClassifier, SensitiveDataDetector

__all__ = [
    'DocumentAccessControl',
    'Permission',
    'AccessPolicy',
    'AuditLogger',
    'SecurityEvent',
    'AuditConfig',
    'ComplianceEngine',
    'PolicyEngine',
    'ComplianceRule',
    'DataLossPreventionEngine',
    'ContentClassifier',
    'SensitiveDataDetector'
]

__version__ = "1.0.0"