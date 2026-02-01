"""
Security Integration Configuration

Centralized configuration for integrating all security components with
the existing RAG system including memory, authentication, and observability.
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import logging

from ..src.security import (
    DocumentAccessControl, AuditLogger, ComplianceEngine, 
    DataLossPreventionEngine, AuditConfig
)

class SecurityIntegrationConfig:
    """Configuration manager for security integration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize security components
        self.access_control = None
        self.audit_logger = None
        self.compliance_engine = None
        self.dlp_engine = None
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        current_dir = Path(__file__).parent
        return str(current_dir / "security_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in config"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            "security": {
                "environment": "development",
                "debug_mode": True,
                "session": {"timeout_minutes": 30}
            },
            "access_control": {"default_roles": {}},
            "audit_logging": {"storage": {"backend": "file"}},
            "compliance": {"frameworks": ["gdpr"]},
            "data_loss_prevention": {"detection_rules": {}},
            "integrations": {}
        }
    
    def initialize_security_components(self):
        """Initialize all security components with configuration"""
        try:
            # Initialize Access Control
            self.access_control = DocumentAccessControl(
                config=self.config.get("access_control", {})
            )
            
            # Initialize Audit Logger
            audit_config = self._create_audit_config()
            self.audit_logger = AuditLogger(config=audit_config)
            
            # Initialize Compliance Engine
            compliance_config_path = self.config.get("compliance", {}).get("config_path")
            if compliance_config_path:
                self.compliance_engine = ComplianceEngine(compliance_config_path)
            else:
                self.compliance_engine = ComplianceEngine()
            
            # Initialize DLP Engine
            self.dlp_engine = DataLossPreventionEngine()
            
            self.logger.info("Security components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security components: {e}")
            raise
    
    def _create_audit_config(self) -> AuditConfig:
        """Create audit configuration from YAML config"""
        audit_config_data = self.config.get("audit_logging", {})
        
        return AuditConfig(
            storage_backend=audit_config_data.get("storage", {}).get("backend", "file"),
            storage_path=audit_config_data.get("storage", {}).get("path", "./audit_logs"),
            max_file_size=audit_config_data.get("storage", {}).get("max_file_size_mb", 100) * 1024 * 1024,
            retention_days=audit_config_data.get("storage", {}).get("retention_days", 365),
            compression_enabled=audit_config_data.get("storage", {}).get("compression", True),
            real_time_alerts=audit_config_data.get("alerts", {}).get("enabled", True),
            batch_size=audit_config_data.get("events", {}).get("batch_size", 100),
            flush_interval=audit_config_data.get("events", {}).get("flush_interval_seconds", 30),
            async_processing=audit_config_data.get("events", {}).get("async_processing", True),
            encryption_enabled=audit_config_data.get("storage", {}).get("encryption", True),
            integrity_checks=True,
            tamper_detection=True
        )
    
    def integrate_with_memory_system(self, memory_manager):
        """Integrate security with existing memory system"""
        try:
            # Add security metadata to memory operations
            original_store = memory_manager.store
            original_retrieve = memory_manager.retrieve
            
            def secure_store(key: str, value: Any, metadata: Dict[str, Any] = None):
                # Add security classification
                if metadata is None:
                    metadata = {}
                
                # Analyze content for sensitive data
                if isinstance(value, str):
                    dlp_analysis = self.dlp_engine.analyze_content(
                        content=value,
                        context={"operation": "memory_store", "key": key}
                    )
                    metadata["security_classification"] = dlp_analysis["classification"]["overall_sensitivity"].value
                    metadata["sensitive_data_detected"] = len(dlp_analysis["classification"]["detections"]) > 0
                
                # Store with security metadata
                result = original_store(key, value, metadata)
                
                # Log the operation
                self.audit_logger.log_access_event(
                    user_id=metadata.get("user_id", "system"),
                    resource_type="memory",
                    resource_id=key,
                    action="store",
                    granted=True,
                    details={"metadata": metadata}
                )
                
                return result
            
            def secure_retrieve(key: str, user_id: str = "system"):
                # Check access permissions
                access_result = self.access_control.check_access(
                    user_id=user_id,
                    resource_type="memory",
                    resource_id=key,
                    access_level="read"
                )
                
                if access_result.decision.value != "allow":
                    self.audit_logger.log_access_event(
                        user_id=user_id,
                        resource_type="memory",
                        resource_id=key,
                        action="retrieve",
                        granted=False,
                        violations=[access_result.reason]
                    )
                    raise PermissionError(f"Access denied to memory key: {key}")
                
                # Retrieve the data
                result = original_retrieve(key)
                
                # Log successful access
                self.audit_logger.log_access_event(
                    user_id=user_id,
                    resource_type="memory",
                    resource_id=key,
                    action="retrieve",
                    granted=True
                )
                
                return result
            
            # Replace methods
            memory_manager.store = secure_store
            memory_manager.retrieve = secure_retrieve
            
            self.logger.info("Memory system security integration completed")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with memory system: {e}")
            raise
    
    def integrate_with_authentication(self, auth_manager):
        """Integrate security with authentication system"""
        try:
            # Add authentication event logging
            original_authenticate = auth_manager.authenticate
            original_authorize = getattr(auth_manager, 'authorize', None)
            
            def secure_authenticate(credentials: Dict[str, Any]):
                start_time = datetime.utcnow()
                user_id = credentials.get("user_id", "unknown")
                
                try:
                    result = original_authenticate(credentials)
                    
                    # Log successful authentication
                    self.audit_logger.log_authentication_event(
                        user_id=user_id,
                        action="login",
                        success=True,
                        ip_address=credentials.get("ip_address"),
                        details={
                            "auth_method": credentials.get("auth_method", "unknown"),
                            "duration_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000)
                        }
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log failed authentication
                    self.audit_logger.log_authentication_event(
                        user_id=user_id,
                        action="login",
                        success=False,
                        ip_address=credentials.get("ip_address"),
                        details={
                            "error": str(e),
                            "auth_method": credentials.get("auth_method", "unknown")
                        }
                    )
                    raise
            
            if original_authorize:
                def secure_authorize(user_id: str, resource: str, action: str, context: Dict[str, Any] = None):
                    if context is None:
                        context = {}
                    
                    # Use our access control system
                    access_result = self.access_control.check_access(
                        user_id=user_id,
                        resource_type="api",
                        resource_id=resource,
                        access_level=action,
                        context=context
                    )
                    
                    granted = access_result.decision.value == "allow"
                    
                    # Log authorization decision
                    self.audit_logger.log_access_event(
                        user_id=user_id,
                        resource_type="api",
                        resource_id=resource,
                        action=action,
                        granted=granted,
                        violations=[] if granted else [access_result.reason]
                    )
                    
                    return granted
                
                auth_manager.authorize = secure_authorize
            
            auth_manager.authenticate = secure_authenticate
            
            self.logger.info("Authentication system security integration completed")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with authentication system: {e}")
            raise
    
    def integrate_with_observability(self, observability_manager):
        """Integrate security with observability system"""
        try:
            # Add security metrics
            def register_security_metrics():
                observability_manager.register_metric(
                    name="security_events_total",
                    metric_type="counter",
                    description="Total number of security events",
                    labels=["event_type", "severity", "outcome"]
                )
                
                observability_manager.register_metric(
                    name="access_violations_total",
                    metric_type="counter",
                    description="Total number of access violations",
                    labels=["user_id", "resource_type", "violation_type"]
                )
                
                observability_manager.register_metric(
                    name="dlp_detections_total",
                    metric_type="counter",
                    description="Total number of DLP detections",
                    labels=["data_type", "sensitivity_level", "action_taken"]
                )
                
                observability_manager.register_metric(
                    name="compliance_score",
                    metric_type="gauge",
                    description="Current compliance score",
                    labels=["framework"]
                )
            
            # Add security event handlers
            def handle_security_event(event_type: str, details: Dict[str, Any]):
                # Update metrics
                observability_manager.increment_counter(
                    "security_events_total",
                    labels={
                        "event_type": event_type,
                        "severity": details.get("severity", "unknown"),
                        "outcome": details.get("outcome", "unknown")
                    }
                )
                
                # Send to observability system
                observability_manager.log_event(
                    event_type="security",
                    details=details
                )
            
            # Register components
            register_security_metrics()
            
            # Add security event handler to audit logger
            self.audit_logger.add_alert_handler(
                lambda event: handle_security_event(
                    event_type=event.event_type,
                    details={
                        "severity": event.severity.value,
                        "outcome": event.outcome,
                        "user_id": event.user_id,
                        "resource_type": event.resource_type
                    }
                )
            )
            
            self.logger.info("Observability system security integration completed")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with observability system: {e}")
            raise
    
    def integrate_with_document_processing(self, document_processor):
        """Integrate security with document processing"""
        try:
            original_process = document_processor.process_document
            
            def secure_process_document(document: Dict[str, Any], user_id: str = "system"):
                # Check document access permissions
                doc_id = document.get("id", "unknown")
                access_result = self.access_control.check_access(
                    user_id=user_id,
                    resource_type="document",
                    resource_id=doc_id,
                    access_level="read"
                )
                
                if access_result.decision.value != "allow":
                    self.audit_logger.log_access_event(
                        user_id=user_id,
                        resource_type="document",
                        resource_id=doc_id,
                        action="process",
                        granted=False,
                        violations=[access_result.reason]
                    )
                    raise PermissionError(f"Access denied to document: {doc_id}")
                
                # Analyze document content for sensitive data
                content = document.get("content", "")
                if content:
                    dlp_event = self.dlp_engine.enforce_policy(
                        content=content,
                        context={
                            "operation": "document_processing",
                            "document_id": doc_id,
                            "user_id": user_id
                        }
                    )
                    
                    # If DLP blocks processing, raise error
                    if dlp_event.action_taken.value == "block":
                        raise PermissionError("Document processing blocked by DLP policy")
                    
                    # Apply content modifications if needed
                    if dlp_event.action_taken.value == "redact":
                        redacted_content = dlp_event.response_details.get("redacted_content")
                        if redacted_content:
                            document = document.copy()
                            document["content"] = redacted_content
                
                # Process the document
                result = original_process(document)
                
                # Log successful processing
                self.audit_logger.log_access_event(
                    user_id=user_id,
                    resource_type="document",
                    resource_id=doc_id,
                    action="process",
                    granted=True
                )
                
                return result
            
            document_processor.process_document = secure_process_document
            
            self.logger.info("Document processing security integration completed")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with document processing: {e}")
            raise
    
    def setup_security_middleware(self, app):
        """Set up security middleware for FastAPI application"""
        from fastapi import Request, HTTPException
        from fastapi.middleware.base import BaseHTTPMiddleware
        
        class SecurityMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, security_config):
                super().__init__(app)
                self.security_config = security_config
            
            async def dispatch(self, request: Request, call_next):
                # Extract user context
                user_id = getattr(request.state, 'user_id', 'anonymous')
                ip_address = request.client.host
                user_agent = request.headers.get('user-agent', '')
                
                # Check rate limiting
                if self._is_rate_limited(user_id, ip_address):
                    self.security_config.audit_logger.log_anomaly(
                        anomaly_type="rate_limit_exceeded",
                        user_id=user_id,
                        details={"ip_address": ip_address}
                    )
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                # Process request
                start_time = datetime.utcnow()
                response = await call_next(request)
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                # Log API access
                self.security_config.audit_logger.log_access_event(
                    user_id=user_id,
                    resource_type="api",
                    resource_id=str(request.url.path),
                    action=request.method,
                    granted=response.status_code < 400,
                    ip_address=ip_address,
                    details={
                        "status_code": response.status_code,
                        "duration_ms": int(duration * 1000),
                        "user_agent": user_agent
                    }
                )
                
                return response
            
            def _is_rate_limited(self, user_id: str, ip_address: str) -> bool:
                # Simplified rate limiting implementation
                # In production, use Redis or similar for distributed rate limiting
                return False
        
        app.add_middleware(SecurityMiddleware, security_config=self)
        self.logger.info("Security middleware configured")
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive security health check"""
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check Access Control
        try:
            user_count = len(self.access_control.users)
            role_count = len(self.access_control.roles)
            policy_count = len(self.access_control.policies)
            
            health_status["components"]["access_control"] = {
                "status": "healthy",
                "users": user_count,
                "roles": role_count,
                "policies": policy_count
            }
        except Exception as e:
            health_status["components"]["access_control"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Check Audit Logger
        try:
            audit_stats = self.audit_logger.get_statistics()
            health_status["components"]["audit_logger"] = {
                "status": "healthy",
                **audit_stats
            }
        except Exception as e:
            health_status["components"]["audit_logger"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Check Compliance Engine
        try:
            active_rules = len([r for r in self.compliance_engine.policy_engine.rules.values() if r.is_active])
            active_violations = len([v for v in self.compliance_engine.violations if not v.is_resolved])
            
            health_status["components"]["compliance_engine"] = {
                "status": "healthy",
                "active_rules": active_rules,
                "active_violations": active_violations
            }
        except Exception as e:
            health_status["components"]["compliance_engine"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Check DLP Engine
        try:
            detection_rules = len(self.dlp_engine.detector.rules)
            dlp_policies = len(self.dlp_engine.policies)
            recent_events = len(self.dlp_engine.events)
            
            health_status["components"]["dlp_engine"] = {
                "status": "healthy",
                "detection_rules": detection_rules,
                "policies": dlp_policies,
                "recent_events": recent_events
            }
        except Exception as e:
            health_status["components"]["dlp_engine"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        return health_status
    
    def shutdown(self):
        """Gracefully shutdown all security components"""
        try:
            if self.audit_logger:
                self.audit_logger.shutdown()
            
            # Cleanup expired permissions
            if self.access_control:
                self.access_control.cleanup_expired_permissions()
            
            self.logger.info("Security system shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during security system shutdown: {e}")

# Global security configuration instance
security_config = SecurityIntegrationConfig()

# Initialization function for easy setup
def initialize_security(
    memory_manager=None,
    auth_manager=None, 
    observability_manager=None,
    document_processor=None,
    app=None
):
    """Initialize and integrate all security components"""
    try:
        # Initialize security components
        security_config.initialize_security_components()
        
        # Perform integrations
        if memory_manager:
            security_config.integrate_with_memory_system(memory_manager)
        
        if auth_manager:
            security_config.integrate_with_authentication(auth_manager)
        
        if observability_manager:
            security_config.integrate_with_observability(observability_manager)
        
        if document_processor:
            security_config.integrate_with_document_processing(document_processor)
        
        if app:
            security_config.setup_security_middleware(app)
        
        logging.info("Security system initialization completed successfully")
        return security_config
        
    except Exception as e:
        logging.error(f"Security system initialization failed: {e}")
        raise

# Export configuration and initialization function
__all__ = ['SecurityIntegrationConfig', 'security_config', 'initialize_security']