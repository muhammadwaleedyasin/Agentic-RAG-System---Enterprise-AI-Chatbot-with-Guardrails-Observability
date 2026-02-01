"""
Enhanced health check endpoints with comprehensive system monitoring.
"""
import time
import psutil
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Response
from pydantic import BaseModel

from ...core.rag_pipeline import RAGPipeline
from ...models.common import BaseResponse
from ...config.settings import settings
from ..deps import get_rag_pipeline, get_access_controller, get_connection_manager

router = APIRouter()


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    timestamp: str
    uptime: float
    version: str
    components: Dict[str, Any]
    system: Dict[str, Any]


class ComponentHealth(BaseModel):
    """Individual component health."""
    name: str
    status: str
    response_time: float
    details: Dict[str, Any]


# Track application start time
app_start_time = time.time()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint - always returns quickly.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.app_name,
        "version": settings.app_version
    }


@router.get("/health/detailed", response_model=HealthStatus)
async def detailed_health_check(
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    access_controller = Depends(get_access_controller),
    connection_manager = Depends(get_connection_manager)
):
    """
    Detailed health check with component status and system metrics.
    """
    try:
        start_check_time = time.time()
        
        # RAG Pipeline health
        rag_health = await rag_pipeline.health_check()
        
        # System metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Network connectivity (simplified)
        network_status = "healthy"  # Would check external dependencies
        
        # Component statuses
        components = {
            "rag_pipeline": {
                "status": "healthy" if rag_health.get("overall") else "unhealthy",
                "details": rag_health
            },
            "database": {
                "status": "healthy" if rag_health.get("vector_store") else "unhealthy",
                "details": {"type": "vector_store"}
            },
            "embedding_service": {
                "status": "healthy" if rag_health.get("embedding_service") else "unhealthy",
                "details": {"provider": "configured"}
            },
            "llm_provider": {
                "status": "healthy" if rag_health.get("llm_provider") else "unhealthy",
                "details": {"provider": settings.llm_provider.value}
            },
            "authentication": {
                "status": "healthy",
                "details": {"active_sessions": len(access_controller.active_sessions)}
            },
            "websockets": {
                "status": "healthy",
                "details": {"active_connections": connection_manager.get_active_connections_count()}
            }
        }
        
        # System information
        system_info = {
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 2)
            },
            "cpu": {
                "percent_used": cpu_percent,
                "cores": psutil.cpu_count()
            },
            "uptime": time.time() - app_start_time
        }
        
        # Overall health determination
        overall_healthy = all(
            comp["status"] == "healthy" for comp in components.values()
        )
        
        health_check_time = time.time() - start_check_time
        
        return HealthStatus(
            status="healthy" if overall_healthy else "unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            uptime=system_info["uptime"],
            version=settings.app_version,
            components=components,
            system={
                **system_info,
                "health_check_duration": round(health_check_time, 3)
            }
        )
        
    except Exception as e:
        return HealthStatus(
            status="error",
            timestamp=datetime.utcnow().isoformat(),
            uptime=time.time() - app_start_time,
            version=settings.app_version,
            components={"error": {"status": "error", "details": str(e)}},
            system={"error": str(e)}
        )


@router.get("/health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe - checks if application is running.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/ready")
async def readiness_probe(rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Kubernetes readiness probe - checks if application is ready to serve traffic.
    """
    try:
        # Quick checks for critical components
        health_status = await rag_pipeline.health_check()
        
        # Check if critical components are healthy
        if (health_status.get("embedding_service") and 
            health_status.get("vector_store")):
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


@router.get("/health/startup")
async def startup_probe(rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Kubernetes startup probe - checks if application has finished starting up.
    """
    try:
        # Check if RAG pipeline is initialized
        if rag_pipeline.initialized:
            return {
                "status": "started",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime": time.time() - app_start_time
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Application still starting up"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Startup check failed: {str(e)}"
        )


@router.get("/health/components/{component_name}")
async def component_health_check(
    component_name: str,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    access_controller = Depends(get_access_controller),
    connection_manager = Depends(get_connection_manager)
):
    """
    Check health of a specific component.
    """
    try:
        start_time = time.time()
        
        if component_name == "rag_pipeline":
            health = await rag_pipeline.health_check()
            status = "healthy" if health.get("overall") else "unhealthy"
            details = health
            
        elif component_name == "embedding_service":
            health = await rag_pipeline.health_check()
            status = "healthy" if health.get("embedding_service") else "unhealthy"
            details = {"service": "embedding"}
            
        elif component_name == "vector_store":
            health = await rag_pipeline.health_check()
            status = "healthy" if health.get("vector_store") else "unhealthy"
            details = {"store": "vector_db"}
            
        elif component_name == "llm_provider":
            health = await rag_pipeline.health_check()
            status = "healthy" if health.get("llm_provider") else "unhealthy"
            details = {"provider": settings.llm_provider.value}
            
        elif component_name == "authentication":
            status = "healthy"
            details = {
                "total_users": len(access_controller.users),
                "active_sessions": len(access_controller.active_sessions)
            }
            
        elif component_name == "websockets":
            status = "healthy"
            details = {
                "active_connections": connection_manager.get_active_connections_count(),
                "rooms": connection_manager.get_rooms_info()
            }
            
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Component '{component_name}' not found"
            )
        
        response_time = time.time() - start_time
        
        return ComponentHealth(
            name=component_name,
            status=status,
            response_time=round(response_time, 3),
            details=details
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return ComponentHealth(
            name=component_name,
            status="error",
            response_time=time.time() - start_time,
            details={"error": str(e)}
        )


@router.get("/metrics")
async def get_metrics(
    access_controller = Depends(get_access_controller),
    connection_manager = Depends(get_connection_manager)
):
    """
    Prometheus-style metrics endpoint.
    """
    try:
        # Get system metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent()
        
        # Get application metrics
        # access_controller and connection_manager are now injected dependencies
        
        # Format as Prometheus metrics
        metrics = f"""
# HELP app_uptime_seconds Application uptime in seconds
# TYPE app_uptime_seconds counter
app_uptime_seconds {time.time() - app_start_time}

# HELP system_memory_usage_ratio System memory usage ratio
# TYPE system_memory_usage_ratio gauge
system_memory_usage_ratio {memory.percent / 100}

# HELP system_disk_usage_ratio System disk usage ratio
# TYPE system_disk_usage_ratio gauge
system_disk_usage_ratio {(disk.used / disk.total)}

# HELP system_cpu_usage_ratio System CPU usage ratio
# TYPE system_cpu_usage_ratio gauge
system_cpu_usage_ratio {cpu_percent / 100}

# HELP auth_active_sessions Number of active authentication sessions
# TYPE auth_active_sessions gauge
auth_active_sessions {len(access_controller.active_sessions)}

# HELP websocket_active_connections Number of active WebSocket connections
# TYPE websocket_active_connections gauge
websocket_active_connections {connection_manager.get_active_connections_count()}

# HELP app_info Application information
# TYPE app_info gauge
app_info{{version="{settings.app_version}",environment="{settings.debug}"}} 1
        """.strip()
        
        return Response(content=metrics, media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate metrics: {str(e)}"
        )
