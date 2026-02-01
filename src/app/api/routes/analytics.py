"""
Analytics and monitoring endpoints for Enterprise RAG Chatbot.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
import time

from ....security.access_control import User, Permission
from ....core.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


class UsageMetrics(BaseModel):
    """Usage metrics model."""
    time_period: str
    total_queries: int
    total_documents: int
    total_users: int
    avg_response_time: float
    queries_by_hour: Dict[str, int]
    popular_queries: List[Dict[str, Any]]


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    avg_retrieval_time: float
    avg_generation_time: float
    avg_total_time: float
    success_rate: float
    error_rate: float
    throughput_qps: float


class SystemHealth(BaseModel):
    """System health model."""
    overall_status: str
    components: Dict[str, bool]
    uptime: float
    memory_usage: Dict[str, float]
    disk_usage: Dict[str, float]


# In-memory analytics storage (replace with proper analytics DB in production)
query_logs: List[Dict[str, Any]] = []
system_metrics: Dict[str, Any] = {
    "start_time": time.time(),
    "total_queries": 0,
    "total_errors": 0,
    "response_times": []
}


# Dependencies  
from ...deps import get_current_user, get_rag_pipeline, get_access_controller


# Permission check for analytics access
async def require_analytics_permission(current_user: User = Depends(get_current_user)):
    """Require analytics viewing permission."""
    if not current_user.can_perform_action(Permission.VIEW_ANALYTICS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analytics viewing permission required"
        )
    return current_user


@router.get("/usage", response_model=UsageMetrics)
async def get_usage_metrics(
    time_period: str = Query("24h", description="Time period: 1h, 24h, 7d, 30d"),
    current_user: User = Depends(require_analytics_permission),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    access_controller = Depends(get_access_controller)
):
    """Get usage metrics for specified time period."""
    try:
        # Calculate time range
        now = datetime.utcnow()
        if time_period == "1h":
            start_time = now - timedelta(hours=1)
        elif time_period == "24h":
            start_time = now - timedelta(days=1)
        elif time_period == "7d":
            start_time = now - timedelta(days=7)
        elif time_period == "30d":
            start_time = now - timedelta(days=30)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid time period. Use: 1h, 24h, 7d, 30d"
            )
        
        # Filter logs by time period
        filtered_logs = [
            log for log in query_logs
            if datetime.fromisoformat(log.get("timestamp", "1970-01-01")) >= start_time
        ]
        
        # Calculate metrics
        total_queries = len(filtered_logs)
        total_response_time = sum(log.get("response_time", 0) for log in filtered_logs)
        avg_response_time = total_response_time / total_queries if total_queries > 0 else 0
        
        # Queries by hour
        queries_by_hour = {}
        for log in filtered_logs:
            hour = datetime.fromisoformat(log.get("timestamp", "1970-01-01")).strftime("%H:00")
            queries_by_hour[hour] = queries_by_hour.get(hour, 0) + 1
        
        # Popular queries (simplified)
        query_counts = {}
        for log in filtered_logs:
            query = log.get("query", "")
            if len(query) > 10:  # Only count substantial queries
                query_counts[query] = query_counts.get(query, 0) + 1
        
        popular_queries = [
            {"query": query, "count": count}
            for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Get system stats for additional metrics
        stats = await rag_pipeline.get_stats()
        total_documents = stats.get("vector_store", {}).get("total_documents", 0)
        
        # Get user count from dependency
        total_users = len(access_controller.users)
        
        return UsageMetrics(
            time_period=time_period,
            total_queries=total_queries,
            total_documents=total_documents,
            total_users=total_users,
            avg_response_time=avg_response_time,
            queries_by_hour=queries_by_hour,
            popular_queries=popular_queries
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting usage metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage metrics: {str(e)}"
        )


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    time_period: str = Query("24h", description="Time period: 1h, 24h, 7d, 30d"),
    current_user: User = Depends(require_analytics_permission)
):
    """Get performance metrics for specified time period."""
    try:
        # Calculate time range
        now = datetime.utcnow()
        if time_period == "1h":
            start_time = now - timedelta(hours=1)
        elif time_period == "24h":
            start_time = now - timedelta(days=1)
        elif time_period == "7d":
            start_time = now - timedelta(days=7)
        elif time_period == "30d":
            start_time = now - timedelta(days=30)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid time period. Use: 1h, 24h, 7d, 30d"
            )
        
        # Filter logs by time period
        filtered_logs = [
            log for log in query_logs
            if datetime.fromisoformat(log.get("timestamp", "1970-01-01")) >= start_time
        ]
        
        if not filtered_logs:
            return PerformanceMetrics(
                avg_retrieval_time=0,
                avg_generation_time=0,
                avg_total_time=0,
                success_rate=0,
                error_rate=0,
                throughput_qps=0
            )
        
        # Calculate performance metrics
        retrieval_times = [log.get("retrieval_time", 0) for log in filtered_logs if log.get("retrieval_time")]
        generation_times = [log.get("generation_time", 0) for log in filtered_logs if log.get("generation_time")]
        total_times = [log.get("response_time", 0) for log in filtered_logs if log.get("response_time")]
        
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
        avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0
        avg_total_time = sum(total_times) / len(total_times) if total_times else 0
        
        # Success and error rates
        successful_queries = len([log for log in filtered_logs if log.get("status") == "success"])
        error_queries = len(filtered_logs) - successful_queries
        
        success_rate = successful_queries / len(filtered_logs) if filtered_logs else 0
        error_rate = error_queries / len(filtered_logs) if filtered_logs else 0
        
        # Throughput (queries per second)
        time_span_hours = (now - start_time).total_seconds() / 3600
        throughput_qps = len(filtered_logs) / (time_span_hours * 3600) if time_span_hours > 0 else 0
        
        return PerformanceMetrics(
            avg_retrieval_time=avg_retrieval_time,
            avg_generation_time=avg_generation_time,
            avg_total_time=avg_total_time,
            success_rate=success_rate,
            error_rate=error_rate,
            throughput_qps=throughput_qps
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.get("/health", response_model=SystemHealth)
async def get_system_health(
    current_user: User = Depends(require_analytics_permission),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Get comprehensive system health status."""
    try:
        # Get component health from RAG pipeline
        health_status = await rag_pipeline.health_check()
        
        # Calculate uptime
        uptime = time.time() - system_metrics["start_time"]
        
        # Get memory usage (simplified - would use psutil in production)
        import os
        import psutil
        
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        memory_usage = {
            "total_gb": memory_info.total / (1024**3),
            "used_gb": memory_info.used / (1024**3),
            "available_gb": memory_info.available / (1024**3),
            "percent_used": memory_info.percent
        }
        
        disk_usage = {
            "total_gb": disk_info.total / (1024**3),
            "used_gb": disk_info.used / (1024**3),
            "free_gb": disk_info.free / (1024**3),
            "percent_used": (disk_info.used / disk_info.total) * 100
        }
        
        # Overall health status
        overall_healthy = health_status.get("overall", False)
        overall_status = "healthy" if overall_healthy else "unhealthy"
        
        return SystemHealth(
            overall_status=overall_status,
            components=health_status,
            uptime=uptime,
            memory_usage=memory_usage,
            disk_usage=disk_usage
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        # Return degraded status instead of failing
        return SystemHealth(
            overall_status="degraded",
            components={"health_check": False},
            uptime=time.time() - system_metrics["start_time"],
            memory_usage={},
            disk_usage={}
        )


@router.get("/logs")
async def get_query_logs(
    current_user: User = Depends(require_analytics_permission),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    user_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="Filter by status: success, error")
):
    """Get query logs with filtering and pagination."""
    try:
        # Filter logs
        filtered_logs = query_logs
        
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.get("user_id") == user_id]
        
        if status:
            filtered_logs = [log for log in filtered_logs if log.get("status") == status]
        
        # Sort by timestamp (most recent first)
        filtered_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Apply pagination
        paginated_logs = filtered_logs[offset:offset + limit]
        
        return {
            "status": "success",
            "message": "Query logs retrieved successfully",
            "logs": paginated_logs,
            "total": len(filtered_logs),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error getting query logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get query logs: {str(e)}"
        )


@router.get("/users/{user_id}/activity")
async def get_user_activity(
    user_id: str,
    current_user: User = Depends(require_analytics_permission),
    time_period: str = Query("24h", description="Time period: 1h, 24h, 7d, 30d")
):
    """Get activity metrics for a specific user."""
    try:
        # Check if current user can view other users' activity
        if (user_id != current_user.user_id and 
            not current_user.can_perform_action(Permission.ADMIN_ACCESS)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot view other users' activity"
            )
        
        # Calculate time range
        now = datetime.utcnow()
        if time_period == "1h":
            start_time = now - timedelta(hours=1)
        elif time_period == "24h":
            start_time = now - timedelta(days=1)
        elif time_period == "7d":
            start_time = now - timedelta(days=7)
        elif time_period == "30d":
            start_time = now - timedelta(days=30)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid time period. Use: 1h, 24h, 7d, 30d"
            )
        
        # Filter logs for specific user and time period
        user_logs = [
            log for log in query_logs
            if (log.get("user_id") == user_id and
                datetime.fromisoformat(log.get("timestamp", "1970-01-01")) >= start_time)
        ]
        
        # Calculate user metrics
        total_queries = len(user_logs)
        successful_queries = len([log for log in user_logs if log.get("status") == "success"])
        failed_queries = total_queries - successful_queries
        
        avg_response_time = 0
        if user_logs:
            total_time = sum(log.get("response_time", 0) for log in user_logs)
            avg_response_time = total_time / len(user_logs)
        
        # Query distribution by hour
        queries_by_hour = {}
        for log in user_logs:
            hour = datetime.fromisoformat(log.get("timestamp", "1970-01-01")).strftime("%H:00")
            queries_by_hour[hour] = queries_by_hour.get(hour, 0) + 1
        
        # Recent queries
        recent_queries = sorted(user_logs, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]
        
        return {
            "status": "success",
            "message": "User activity retrieved successfully",
            "user_id": user_id,
            "time_period": time_period,
            "metrics": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "avg_response_time": avg_response_time,
                "queries_by_hour": queries_by_hour
            },
            "recent_queries": recent_queries
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user activity for {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user activity: {str(e)}"
        )


# Utility function to log queries (called from other endpoints)
def log_query(user_id: str, query: str, response_time: float, status: str, 
              retrieval_time: float = None, generation_time: float = None,
              error_message: str = None):
    """Log a query for analytics."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "query": query,
        "response_time": response_time,
        "status": status,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "error_message": error_message
    }
    
    query_logs.append(log_entry)
    
    # Update system metrics
    system_metrics["total_queries"] += 1
    if status != "success":
        system_metrics["total_errors"] += 1
    
    system_metrics["response_times"].append(response_time)
    
    # Keep only last 10000 logs to prevent memory issues
    if len(query_logs) > 10000:
        query_logs.pop(0)


@router.delete("/logs")
async def clear_logs(
    current_user: User = Depends(require_analytics_permission)
):
    """Clear all query logs (admin only)."""
    if not current_user.can_perform_action(Permission.ADMIN_ACCESS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required to clear logs"
        )
    
    try:
        global query_logs
        query_logs.clear()
        
        # Reset system metrics
        system_metrics.update({
            "total_queries": 0,
            "total_errors": 0,
            "response_times": []
        })
        
        logger.info(f"Query logs cleared by {current_user.username}")
        
        return {
            "status": "success",
            "message": "Query logs cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear logs: {str(e)}"
        )
