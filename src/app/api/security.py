"""
Security management API module.
Merged from src/api/security_routes.py into the canonical src/app/api/ structure.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ...security.access_control import User, Permission, AccessController
from ...security.audit_logger import AuditLogger
from ...security.compliance_engine import ComplianceEngine
from ...security.data_loss_prevention import DataLossPreventionEngine
from ..deps import get_current_user

logger = logging.getLogger(__name__)

# Initialize security components
try:
    audit_logger = AuditLogger()
    compliance_engine = ComplianceEngine()
    dlp_engine = DataLossPreventionEngine()
except ImportError as e:
    logger.warning(f"Some security components not available: {e}")
    audit_logger = None
    compliance_engine = None
    dlp_engine = None

router = APIRouter()


# Request/Response Models
class UserAccessRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    resource_id: str = Field(..., description="Resource ID")
    permission: str = Field(..., description="Permission level (read, write, delete)")


class UserAccessResponse(BaseModel):
    user_id: str
    resource_id: str
    permission: str
    granted: bool
    reason: Optional[str] = None


class RoleCreateRequest(BaseModel):
    role_name: str = Field(..., description="Role name")
    permissions: List[str] = Field(..., description="List of permissions")
    description: Optional[str] = Field(None, description="Role description")


class RoleResponse(BaseModel):
    role_id: str
    role_name: str
    permissions: List[str]
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class PolicyCreateRequest(BaseModel):
    policy_name: str = Field(..., description="Policy name")
    policy_type: str = Field(..., description="Policy type (access, compliance, dlp)")
    rules: Dict[str, Any] = Field(..., description="Policy rules")
    description: Optional[str] = Field(None, description="Policy description")
    enabled: bool = Field(default=True, description="Whether policy is enabled")


class PolicyResponse(BaseModel):
    policy_id: str
    policy_name: str
    policy_type: str
    rules: Dict[str, Any]
    description: Optional[str] = None
    enabled: bool
    created_at: datetime
    updated_at: Optional[datetime] = None


class AuditEventQuery(BaseModel):
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    action: Optional[str] = Field(None, description="Filter by action type")
    resource_id: Optional[str] = Field(None, description="Filter by resource ID")
    start_date: Optional[datetime] = Field(None, description="Start date for events")
    end_date: Optional[datetime] = Field(None, description="End date for events")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


class AuditEventResponse(BaseModel):
    event_id: str
    user_id: str
    action: str
    resource_id: Optional[str] = None
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any]
    risk_level: str


class AuditStatsResponse(BaseModel):
    total_events: int
    events_by_action: Dict[str, int]
    events_by_user: Dict[str, int]
    high_risk_events: int
    date_range: Dict[str, datetime]


class ComplianceAssessmentRequest(BaseModel):
    resource_id: str = Field(..., description="Resource to assess")
    framework: str = Field(..., description="Compliance framework (gdpr, hipaa, sox)")
    scope: Optional[List[str]] = Field(None, description="Specific compliance areas")


class ComplianceAssessmentResponse(BaseModel):
    assessment_id: str
    resource_id: str
    framework: str
    score: float
    status: str
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    assessed_at: datetime


class ComplianceDashboardResponse(BaseModel):
    overall_score: float
    framework_scores: Dict[str, float]
    total_violations: int
    critical_violations: int
    recent_assessments: List[ComplianceAssessmentResponse]
    compliance_trends: Dict[str, Any]


class ViolationResolutionRequest(BaseModel):
    violation_id: str = Field(..., description="Violation ID to resolve")
    resolution_action: str = Field(..., description="Action taken to resolve")
    notes: Optional[str] = Field(None, description="Additional notes")


class DLPPolicyRequest(BaseModel):
    policy_name: str = Field(..., description="DLP policy name")
    content_types: List[str] = Field(..., description="Content types to monitor")
    patterns: List[str] = Field(..., description="Regex patterns to detect")
    actions: List[str] = Field(..., description="Actions to take on detection")
    severity: str = Field(..., description="Severity level (low, medium, high, critical)")
    enabled: bool = Field(default=True, description="Whether policy is enabled")


class DLPPolicyResponse(BaseModel):
    policy_id: str
    policy_name: str
    content_types: List[str]
    patterns: List[str]
    actions: List[str]
    severity: str
    enabled: bool
    created_at: datetime
    detections_count: int


class DLPAnalysisRequest(BaseModel):
    content: str = Field(..., description="Content to analyze")
    content_type: str = Field(..., description="Type of content")
    source: Optional[str] = Field(None, description="Source of content")


class DLPAnalysisResponse(BaseModel):
    analysis_id: str
    content_safe: bool
    violations: List[Dict[str, Any]]
    risk_score: float
    recommended_actions: List[str]
    analyzed_at: datetime


class DLPStatsResponse(BaseModel):
    total_scans: int
    violations_detected: int
    policies_active: int
    top_violations: List[Dict[str, Any]]
    detection_trends: Dict[str, Any]


class SecurityHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    systems_status: Dict[str, str]
    security_score: float
    active_threats: int
    recent_events: int


# Access Control Endpoints
@router.post(
    "/access/check",
    response_model=UserAccessResponse,
    summary="Check user access",
    description="Check if a user has access to a specific resource"
)
async def check_user_access(
    access_request: UserAccessRequest,
    current_user: User = Depends(get_current_user)
) -> UserAccessResponse:
    """Check user access to a resource."""
    try:
        # Verify admin permissions
        if not current_user.can_perform_action(Permission.ADMIN_ACCESS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to check user access"
            )
        
        # For now, return basic response - would integrate with actual access control
        granted = True  # Simplified logic
        
        return UserAccessResponse(
            user_id=access_request.user_id,
            resource_id=access_request.resource_id,
            permission=access_request.permission,
            granted=granted,
            reason="Access granted" if granted else "Access denied"
        )
        
    except Exception as e:
        logger.error(f"Error checking user access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check user access: {str(e)}"
        )


@router.get(
    "/users/{user_id}/permissions",
    response_model=List[str],
    summary="Get user permissions",
    description="Get all permissions for a specific user"
)
async def get_user_permissions(
    user_id: str,
    current_user: User = Depends(get_current_user)
) -> List[str]:
    """Get user permissions."""
    try:
        # Verify admin permissions
        if not current_user.can_perform_action(Permission.ADMIN_ACCESS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view user permissions"
            )
        
        # Return current user's permissions as example
        permissions = [p.value for p in current_user.role.permissions]
        return permissions
        
    except Exception as e:
        logger.error(f"Error getting user permissions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user permissions: {str(e)}"
        )


# Audit Logging Endpoints (require audit logger)
@router.post(
    "/audit/events/query",
    response_model=List[AuditEventResponse],
    summary="Query audit events",
    description="Query audit events with filters"
)
async def query_audit_events(
    query: AuditEventQuery,
    current_user: User = Depends(get_current_user)
) -> List[AuditEventResponse]:
    """Query audit events."""
    if not audit_logger:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Audit logging not available"
        )
    
    try:
        # Verify audit permissions
        if not current_user.can_perform_action(Permission.VIEW_ANALYTICS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view audit events"
            )
        
        events = await audit_logger.query_events(query.dict(exclude_unset=True))
        return [AuditEventResponse(**event) for event in events]
        
    except Exception as e:
        logger.error(f"Error querying audit events: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query audit events: {str(e)}"
        )


@router.get(
    "/audit/stats",
    response_model=AuditStatsResponse,
    summary="Get audit statistics",
    description="Get audit event statistics"
)
async def get_audit_stats(
    current_user: User = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365, description="Number of days for statistics")
) -> AuditStatsResponse:
    """Get audit statistics."""
    if not audit_logger:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Audit logging not available"
        )
    
    try:
        # Verify audit permissions
        if not current_user.can_perform_action(Permission.VIEW_ANALYTICS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view audit statistics"
            )
        
        stats = await audit_logger.get_statistics(days)
        return AuditStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting audit statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit statistics: {str(e)}"
        )


# Compliance Endpoints (require compliance engine)
@router.post(
    "/compliance/assess",
    response_model=ComplianceAssessmentResponse,
    summary="Run compliance assessment",
    description="Run a compliance assessment on a resource"
)
async def run_compliance_assessment(
    assessment_request: ComplianceAssessmentRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> ComplianceAssessmentResponse:
    """Run compliance assessment."""
    if not compliance_engine:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Compliance engine not available"
        )
    
    try:
        # Verify compliance permissions
        if not current_user.can_perform_action(Permission.VIEW_ANALYTICS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to run compliance assessments"
            )
        
        assessment = await compliance_engine.run_assessment(
            assessment_request.resource_id,
            assessment_request.framework,
            assessment_request.scope
        )
        
        return ComplianceAssessmentResponse(**assessment)
        
    except Exception as e:
        logger.error(f"Error running compliance assessment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run compliance assessment: {str(e)}"
        )


# Data Loss Prevention Endpoints
@router.post(
    "/dlp/analyze",
    response_model=DLPAnalysisResponse,
    summary="Analyze content for DLP violations",
    description="Analyze content for potential data loss prevention violations"
)
async def analyze_content(
    analysis_request: DLPAnalysisRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> DLPAnalysisResponse:
    """Analyze content for DLP violations."""
    if not dlp_engine:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DLP engine not available"
        )
    
    try:
        # Check permissions
        if not current_user.can_perform_action(Permission.ACCESS_SENSITIVE):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to analyze content"
            )
        
        # Analyze content
        analysis_context = {
            "user_id": current_user.user_id,
            "content_type": analysis_request.content_type,
            "source": analysis_request.source
        }
        
        analysis_result = dlp_engine.analyze_content(
            analysis_request.content, 
            analysis_context
        )
        
        # Create response
        return DLPAnalysisResponse(
            analysis_id=f"analysis_{datetime.utcnow().timestamp()}",
            content_safe=len(analysis_result["classification"]["detections"]) == 0,
            violations=[
                {
                    "rule_id": d.rule_id,
                    "data_type": d.data_type.value,
                    "sensitivity": d.sensitivity_level.value,
                    "confidence": d.confidence,
                    "context": d.context
                }
                for d in analysis_result["classification"]["detections"]
            ],
            risk_score=analysis_result["risk_score"],
            recommended_actions=[analysis_result["recommended_action"].value],
            analyzed_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze content: {str(e)}"
        )


@router.get(
    "/dlp/stats",
    response_model=DLPStatsResponse,
    summary="Get DLP statistics",
    description="Get data loss prevention statistics"
)
async def get_dlp_stats(
    current_user: User = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365, description="Number of days for statistics")
) -> DLPStatsResponse:
    """Get DLP statistics."""
    if not dlp_engine:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DLP engine not available"
        )
    
    try:
        # Verify DLP permissions
        if not current_user.can_perform_action(Permission.VIEW_ANALYTICS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view DLP statistics"
            )
        
        stats = dlp_engine.get_dlp_statistics(timedelta(days=days))
        
        return DLPStatsResponse(
            total_scans=stats["total_events"],
            violations_detected=stats["blocked_events"],
            policies_active=len(dlp_engine.policies),
            top_violations=[
                {"type": k, "count": v} 
                for k, v in stats["events_by_data_type"].items()
            ],
            detection_trends=stats["events_by_sensitivity"]
        )
        
    except Exception as e:
        logger.error(f"Error getting DLP statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get DLP statistics: {str(e)}"
        )


# Health Check
@router.get(
    "/health",
    response_model=SecurityHealthResponse,
    summary="Security system health check",
    description="Check the health and status of all security systems"
)
async def security_health() -> SecurityHealthResponse:
    """Security system health check."""
    try:
        systems_status = {
            "access_control": "healthy",  # Always available
            "audit_logging": "healthy" if audit_logger else "unavailable",
            "compliance": "healthy" if compliance_engine else "unavailable",
            "dlp": "healthy" if dlp_engine else "unavailable"
        }
        
        # Count healthy systems
        healthy_systems = sum(1 for status in systems_status.values() if status == "healthy")
        total_systems = len(systems_status)
        security_score = (healthy_systems / total_systems) * 100
        
        overall_status = "healthy" if security_score >= 75 else "degraded"
        
        return SecurityHealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            systems_status=systems_status,
            security_score=security_score,
            active_threats=0,  # Placeholder
            recent_events=0   # Placeholder
        )
        
    except Exception as e:
        logger.error(f"Security health check failed: {str(e)}")
        return SecurityHealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            systems_status={
                "access_control": "unknown",
                "audit_logging": "unknown",
                "compliance": "unknown",
                "dlp": "unknown"
            },
            security_score=0.0,
            active_threats=0,
            recent_events=0
        )
