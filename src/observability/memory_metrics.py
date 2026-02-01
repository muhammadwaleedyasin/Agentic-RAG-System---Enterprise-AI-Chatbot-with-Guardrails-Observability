"""
Memory system observability and metrics collection.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from ..config.settings import settings
from ..core.observability_mixin import ObservabilityMixin

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetric:
    """Individual memory metric data point."""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MemoryPerformanceStats:
    """Memory performance statistics."""
    operation_count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    error_count: int = 0
    success_rate: float = 0.0
    last_operation: Optional[datetime] = None
    
    def update(self, duration_ms: float, success: bool = True):
        """Update stats with new operation data."""
        self.operation_count += 1
        self.total_duration_ms += duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.operation_count
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        
        if not success:
            self.error_count += 1
        
        self.success_rate = (self.operation_count - self.error_count) / self.operation_count
        self.last_operation = datetime.utcnow()


class MemoryMetricsCollector(ObservabilityMixin):
    """
    Collects and manages memory system metrics for observability.
    """
    
    def __init__(self, retention_hours: int = 24):
        super().__init__()
        self.retention_hours = retention_hours
        self.retention_cutoff = timedelta(hours=retention_hours)
        
        # Metric storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._performance_stats: Dict[str, MemoryPerformanceStats] = defaultdict(MemoryPerformanceStats)
        
        # System metrics
        self._system_metrics = {
            "zep_connection_status": 0,
            "active_conversations": 0,
            "cached_contexts": 0,
            "memory_operations_total": 0,
            "memory_errors_total": 0,
            "avg_response_time_ms": 0.0
        }
        
        # Alert thresholds
        self._alert_thresholds = {
            "error_rate": 0.05,  # 5%
            "response_time_ms": 2000,  # 2 seconds
            "memory_usage_mb": 500,  # 500MB
            "connection_failures": 5
        }
        
        # Alert state
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        
        # Metrics metadata
        self._metrics_metadata = {
            "collector_started": datetime.utcnow(),
            "metrics_collected": 0,
            "last_cleanup": datetime.utcnow()
        }
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Record a memory metric."""
        try:
            metric = MemoryMetric(
                timestamp=datetime.utcnow(),
                metric_name=metric_name,
                value=value,
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self._metrics[metric_name].append(metric)
            self._metrics_metadata["metrics_collected"] += 1
            
            # Update system metrics
            if metric_name in self._system_metrics:
                self._system_metrics[metric_name] = value
            
            # Check for alerts
            self._check_alert_conditions(metric_name, value, tags or {})
            
            self.track_operation("metric_recorded", {
                "metric_name": metric_name,
                "value": value,
                "tags_count": len(tags or {})
            })
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")
    
    def record_operation_performance(
        self,
        operation_name: str,
        duration_ms: float,
        success: bool = True,
        metadata: Dict[str, Any] = None
    ):
        """Record operation performance metrics."""
        try:
            # Update performance stats
            self._performance_stats[operation_name].update(duration_ms, success)
            
            # Record as metrics
            self.record_metric(
                f"memory.operation.{operation_name}.duration_ms",
                duration_ms,
                tags={"operation": operation_name, "success": str(success)},
                metadata=metadata
            )
            
            if not success:
                self.record_metric(
                    f"memory.operation.{operation_name}.errors",
                    1,
                    tags={"operation": operation_name}
                )
            
        except Exception as e:
            logger.error(f"Failed to record operation performance: {e}")
    
    def record_conversation_metric(
        self,
        conversation_id: str,
        user_id: str,
        metric_type: str,
        value: float,
        metadata: Dict[str, Any] = None
    ):
        """Record conversation-specific metrics."""
        try:
            self.record_metric(
                f"memory.conversation.{metric_type}",
                value,
                tags={
                    "conversation_id": conversation_id[:8],  # Truncated for privacy
                    "user_id": user_id[:8],  # Truncated for privacy
                    "metric_type": metric_type
                },
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to record conversation metric: {e}")
    
    def record_zep_metric(
        self,
        operation: str,
        success: bool,
        duration_ms: float,
        metadata: Dict[str, Any] = None
    ):
        """Record Zep integration metrics."""
        try:
            self.record_metric(
                "memory.zep.operation_duration_ms",
                duration_ms,
                tags={"operation": operation, "success": str(success)}
            )
            
            self.record_metric(
                "memory.zep.operations_total",
                1,
                tags={"operation": operation, "success": str(success)}
            )
            
            if not success:
                self.record_metric(
                    "memory.zep.errors_total",
                    1,
                    tags={"operation": operation}
                )
            
            # Update connection status
            self._system_metrics["zep_connection_status"] = 1 if success else 0
            
        except Exception as e:
            logger.error(f"Failed to record Zep metric: {e}")
    
    def get_metrics_summary(
        self,
        time_window_hours: int = 1
    ) -> Dict[str, Any]:
        """Get metrics summary for the specified time window."""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            summary = {
                "time_window_hours": time_window_hours,
                "system_metrics": self._system_metrics.copy(),
                "performance_stats": {},
                "metric_counts": {},
                "alerts": list(self._active_alerts.values()),
                "metadata": self._metrics_metadata.copy()
            }
            
            # Performance stats
            for operation, stats in self._performance_stats.items():
                summary["performance_stats"][operation] = asdict(stats)
            
            # Metric counts in time window
            for metric_name, metric_queue in self._metrics.items():
                count = sum(
                    1 for metric in metric_queue
                    if metric.timestamp >= cutoff
                )
                summary["metric_counts"][metric_name] = count
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}
    
    def get_metric_history(
        self,
        metric_name: str,
        time_window_hours: int = 1,
        aggregation: str = "raw"
    ) -> List[Dict[str, Any]]:
        """Get metric history for analysis."""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            if metric_name not in self._metrics:
                return []
            
            # Filter by time window
            filtered_metrics = [
                metric for metric in self._metrics[metric_name]
                if metric.timestamp >= cutoff
            ]
            
            if aggregation == "raw":
                return [asdict(metric) for metric in filtered_metrics]
            
            elif aggregation == "hourly":
                return self._aggregate_metrics_hourly(filtered_metrics)
            
            elif aggregation == "summary":
                return self._aggregate_metrics_summary(filtered_metrics)
            
            else:
                return [asdict(metric) for metric in filtered_metrics]
                
        except Exception as e:
            logger.error(f"Failed to get metric history: {e}")
            return []
    
    def _aggregate_metrics_hourly(self, metrics: List[MemoryMetric]) -> List[Dict[str, Any]]:
        """Aggregate metrics by hour."""
        hourly_data = defaultdict(list)
        
        for metric in metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_data[hour_key].append(metric)
        
        aggregated = []
        for hour, hour_metrics in hourly_data.items():
            values = [m.value for m in hour_metrics]
            aggregated.append({
                "timestamp": hour.isoformat(),
                "count": len(values),
                "avg": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "sum": sum(values)
            })
        
        return sorted(aggregated, key=lambda x: x["timestamp"])
    
    def _aggregate_metrics_summary(self, metrics: List[MemoryMetric]) -> List[Dict[str, Any]]:
        """Create summary statistics for metrics."""
        if not metrics:
            return []
        
        values = [m.value for m in metrics]
        return [{
            "total_count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "first_timestamp": min(m.timestamp for m in metrics).isoformat(),
            "last_timestamp": max(m.timestamp for m in metrics).isoformat()
        }]
    
    def _check_alert_conditions(
        self,
        metric_name: str,
        value: float,
        tags: Dict[str, str]
    ):
        """Check if metric value triggers any alerts."""
        try:
            alert_triggered = False
            alert_message = ""
            
            # Error rate alerts
            if "error" in metric_name and value > self._alert_thresholds["error_rate"]:
                alert_triggered = True
                alert_message = f"High error rate detected: {value:.2%}"
            
            # Response time alerts
            elif "duration_ms" in metric_name and value > self._alert_thresholds["response_time_ms"]:
                alert_triggered = True
                alert_message = f"High response time detected: {value:.2f}ms"
            
            # Connection failure alerts
            elif metric_name == "memory.zep.errors_total" and value >= self._alert_thresholds["connection_failures"]:
                alert_triggered = True
                alert_message = f"Multiple connection failures detected: {value}"
            
            if alert_triggered:
                alert_id = f"{metric_name}_{int(time.time())}"
                self._active_alerts[alert_id] = {
                    "id": alert_id,
                    "metric_name": metric_name,
                    "value": value,
                    "message": alert_message,
                    "tags": tags,
                    "timestamp": datetime.utcnow().isoformat(),
                    "resolved": False
                }
                
                logger.warning(f"Memory system alert: {alert_message}")
        
        except Exception as e:
            logger.error(f"Failed to check alert conditions: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id]["resolved"] = True
            self._active_alerts[alert_id]["resolved_at"] = datetime.utcnow().isoformat()
    
    def cleanup_old_metrics(self):
        """Remove old metrics beyond retention period."""
        try:
            cutoff = datetime.utcnow() - self.retention_cutoff
            cleanup_count = 0
            
            for metric_name, metric_queue in self._metrics.items():
                original_size = len(metric_queue)
                
                # Filter out old metrics
                while metric_queue and metric_queue[0].timestamp < cutoff:
                    metric_queue.popleft()
                    cleanup_count += 1
            
            # Cleanup old alerts
            resolved_alerts = [
                alert_id for alert_id, alert in self._active_alerts.items()
                if alert.get("resolved", False) and 
                datetime.fromisoformat(alert["timestamp"]) < cutoff
            ]
            
            for alert_id in resolved_alerts:
                del self._active_alerts[alert_id]
            
            self._metrics_metadata["last_cleanup"] = datetime.utcnow()
            
            if cleanup_count > 0:
                logger.debug(f"Cleaned up {cleanup_count} old metrics and {len(resolved_alerts)} old alerts")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
    
    def export_metrics(
        self,
        format_type: str = "json",
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Export metrics in specified format."""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "time_window_hours": time_window_hours,
                "system_metrics": self._system_metrics.copy(),
                "performance_stats": {
                    name: asdict(stats) for name, stats in self._performance_stats.items()
                },
                "alerts": list(self._active_alerts.values()),
                "metrics": {}
            }
            
            # Export metric data
            for metric_name, metric_queue in self._metrics.items():
                filtered_metrics = [
                    asdict(metric) for metric in metric_queue
                    if metric.timestamp >= cutoff
                ]
                export_data["metrics"][metric_name] = filtered_metrics
            
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return {"error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of memory metrics system."""
        try:
            error_rate = self._calculate_overall_error_rate()
            avg_response_time = self._calculate_average_response_time()
            
            health_status = "healthy"
            if error_rate > self._alert_thresholds["error_rate"]:
                health_status = "degraded"
            if avg_response_time > self._alert_thresholds["response_time_ms"]:
                health_status = "degraded"
            if len(self._active_alerts) > 5:
                health_status = "unhealthy"
            
            return {
                "status": health_status,
                "metrics_collected": self._metrics_metadata["metrics_collected"],
                "active_alerts": len(self._active_alerts),
                "error_rate": error_rate,
                "avg_response_time_ms": avg_response_time,
                "zep_connected": self._system_metrics["zep_connection_status"] == 1,
                "last_cleanup": self._metrics_metadata["last_cleanup"].isoformat(),
                "uptime_hours": (datetime.utcnow() - self._metrics_metadata["collector_started"]).total_seconds() / 3600
            }
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_overall_error_rate(self) -> float:
        """Calculate overall error rate across all operations."""
        try:
            total_ops = sum(stats.operation_count for stats in self._performance_stats.values())
            total_errors = sum(stats.error_count for stats in self._performance_stats.values())
            
            return total_errors / total_ops if total_ops > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time across all operations."""
        try:
            response_times = [
                stats.avg_duration_ms for stats in self._performance_stats.values()
                if stats.operation_count > 0
            ]
            
            return sum(response_times) / len(response_times) if response_times else 0.0
            
        except Exception:
            return 0.0


# Global metrics collector instance
memory_metrics_collector = MemoryMetricsCollector()