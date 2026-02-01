"""
Custom metrics collector for RAG pipeline monitoring.

Provides comprehensive metrics collection for:
- Performance monitoring
- Cost tracking
- Quality assessment
- System health
- User behavior analytics
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: datetime
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points."""
    name: str
    metric_type: MetricType
    points: deque = field(default_factory=deque)
    max_points: int = 1000
    
    def add_point(self, value: Union[int, float], tags: Optional[Dict[str, str]] = None):
        """Add a new metric point."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        self.points.append(point)
        
        # Keep only recent points
        while len(self.points) > self.max_points:
            self.points.popleft()
    
    def get_recent_values(self, duration: timedelta) -> List[float]:
        """Get values from the last duration."""
        cutoff = datetime.now() - duration
        return [
            point.value for point in self.points
            if point.timestamp >= cutoff
        ]
    
    def get_stats(self, duration: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistical summary of recent values."""
        if duration:
            values = self.get_recent_values(duration)
        else:
            values = [point.value for point in self.points]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
        }


class MetricsCollector:
    """Central metrics collection and aggregation system."""
    
    def __init__(self, max_series_points: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_series_points: Maximum points to keep per metric series
        """
        self.metrics: Dict[str, MetricSeries] = {}
        self.max_series_points = max_series_points
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}
        
        # RAG-specific metrics
        self._query_count = 0
        self._total_latency = 0.0
        self._error_count = 0
        self._cost_tracking = {
            "total_cost": 0.0,
            "llm_calls": 0,
            "embedding_calls": 0
        }
    
    def increment(
        self,
        metric_name: str,
        value: Union[int, float] = 1,
        tags: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric."""
        self._ensure_metric(metric_name, MetricType.COUNTER)
        self._counters[metric_name] += value
        self.metrics[metric_name].add_point(self._counters[metric_name], tags)
    
    def gauge(
        self,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric value."""
        self._ensure_metric(metric_name, MetricType.GAUGE)
        self._gauges[metric_name] = value
        self.metrics[metric_name].add_point(value, tags)
    
    def histogram(
        self,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a histogram value."""
        self._ensure_metric(metric_name, MetricType.HISTOGRAM)
        self.metrics[metric_name].add_point(value, tags)
    
    def timer_start(self, timer_name: str) -> str:
        """Start a timer measurement."""
        timer_id = f"{timer_name}_{time.time()}"
        self._start_times[timer_id] = time.time()
        return timer_id
    
    def timer_end(
        self,
        timer_id: str,
        tags: Optional[Dict[str, str]] = None
    ) -> float:
        """End a timer measurement and record the duration."""
        if timer_id not in self._start_times:
            logger.warning(f"Timer {timer_id} not found")
            return 0.0
        
        duration = time.time() - self._start_times[timer_id]
        del self._start_times[timer_id]
        
        # Extract metric name from timer_id
        metric_name = "_".join(timer_id.split("_")[:-1])
        self._ensure_metric(metric_name, MetricType.TIMER)
        self.metrics[metric_name].add_point(duration, tags)
        
        return duration
    
    def time_context(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimerContext(self, metric_name, tags)
    
    def record_query_metrics(
        self,
        latency: float,
        success: bool = True,
        user_id: Optional[str] = None,
        query_type: str = "general"
    ):
        """Record RAG query-specific metrics."""
        self._query_count += 1
        self._total_latency += latency
        
        if not success:
            self._error_count += 1
        
        tags = {"query_type": query_type}
        if user_id:
            tags["user_id"] = user_id
        
        self.increment("rag.queries.total", tags=tags)
        self.histogram("rag.query.latency", latency, tags=tags)
        self.gauge("rag.queries.error_rate", self._error_count / self._query_count)
        self.gauge("rag.queries.avg_latency", self._total_latency / self._query_count)
    
    def record_document_metrics(
        self,
        document_size: int,
        chunks_created: int,
        processing_time: float,
        document_type: str = "unknown"
    ):
        """Record document ingestion metrics."""
        tags = {"document_type": document_type}
        
        self.increment("rag.documents.processed", tags=tags)
        self.histogram("rag.document.size_bytes", document_size, tags=tags)
        self.histogram("rag.document.chunks", chunks_created, tags=tags)
        self.histogram("rag.document.processing_time", processing_time, tags=tags)
    
    def record_embedding_metrics(
        self,
        text_count: int,
        model: str,
        processing_time: float,
        cost: Optional[float] = None
    ):
        """Record embedding generation metrics."""
        tags = {"model": model}
        
        self.increment("rag.embeddings.calls", tags=tags)
        self.histogram("rag.embeddings.batch_size", text_count, tags=tags)
        self.histogram("rag.embeddings.processing_time", processing_time, tags=tags)
        
        if cost:
            self._cost_tracking["total_cost"] += cost
            self._cost_tracking["embedding_calls"] += 1
            self.histogram("rag.embeddings.cost", cost, tags=tags)
    
    def record_retrieval_metrics(
        self,
        query_embedding_time: float,
        search_time: float,
        results_count: int,
        top_k: int
    ):
        """Record vector retrieval metrics."""
        self.histogram("rag.retrieval.embedding_time", query_embedding_time)
        self.histogram("rag.retrieval.search_time", search_time)
        self.histogram("rag.retrieval.results_count", results_count)
        self.gauge("rag.retrieval.recall_rate", results_count / top_k if top_k > 0 else 0)
    
    def record_llm_metrics(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        cost: Optional[float] = None
    ):
        """Record LLM generation metrics."""
        tags = {"model": model}
        
        self.increment("rag.llm.calls", tags=tags)
        self.histogram("rag.llm.prompt_tokens", prompt_tokens, tags=tags)
        self.histogram("rag.llm.completion_tokens", completion_tokens, tags=tags)
        self.histogram("rag.llm.total_tokens", prompt_tokens + completion_tokens, tags=tags)
        self.histogram("rag.llm.latency", latency, tags=tags)
        
        if cost:
            self._cost_tracking["total_cost"] += cost
            self._cost_tracking["llm_calls"] += 1
            self.histogram("rag.llm.cost", cost, tags=tags)
    
    def record_quality_metrics(
        self,
        relevance_score: Optional[float] = None,
        hallucination_score: Optional[float] = None,
        coherence_score: Optional[float] = None,
        user_rating: Optional[int] = None
    ):
        """Record response quality metrics."""
        if relevance_score is not None:
            self.histogram("rag.quality.relevance", relevance_score)
        if hallucination_score is not None:
            self.histogram("rag.quality.hallucination", hallucination_score)
        if coherence_score is not None:
            self.histogram("rag.quality.coherence", coherence_score)
        if user_rating is not None:
            self.histogram("rag.quality.user_rating", user_rating)
    
    def get_metric_stats(
        self,
        metric_name: str,
        duration: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Get statistical summary for a metric."""
        if metric_name not in self.metrics:
            return {}
        
        return self.metrics[metric_name].get_stats(duration)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        recent_hour = timedelta(hours=1)
        
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "query_metrics": {},
            "performance_metrics": {},
            "cost_metrics": self._cost_tracking.copy(),
            "error_metrics": {}
        }
        
        # Query metrics
        if "rag.queries.total" in self.metrics:
            query_stats = self.get_metric_stats("rag.queries.total", recent_hour)
            health_data["query_metrics"]["queries_per_hour"] = query_stats.get("count", 0)
        
        if "rag.query.latency" in self.metrics:
            latency_stats = self.get_metric_stats("rag.query.latency", recent_hour)
            health_data["performance_metrics"]["avg_latency"] = latency_stats.get("mean", 0)
            health_data["performance_metrics"]["p95_latency"] = self._calculate_percentile(
                "rag.query.latency", 95, recent_hour
            )
        
        # Error rate
        health_data["error_metrics"]["error_rate"] = self._error_count / max(self._query_count, 1)
        
        return health_data
    
    def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export all metrics in specified format."""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "cost_tracking": self._cost_tracking.copy()
        }
        
        # Export metric series
        for name, series in self.metrics.items():
            export_data["metrics"][name] = {
                "type": series.metric_type.value,
                "stats": series.get_stats(),
                "recent_points": [
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "value": point.value,
                        "tags": point.tags
                    }
                    for point in list(series.points)[-10:]  # Last 10 points
                ]
            }
        
        if format == "json":
            return json.dumps(export_data, indent=2)
        else:
            return export_data
    
    def _ensure_metric(self, metric_name: str, metric_type: MetricType):
        """Ensure metric series exists."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = MetricSeries(
                name=metric_name,
                metric_type=metric_type,
                max_points=self.max_series_points
            )
    
    def _calculate_percentile(
        self,
        metric_name: str,
        percentile: int,
        duration: Optional[timedelta] = None
    ) -> float:
        """Calculate percentile for metric values."""
        if metric_name not in self.metrics:
            return 0.0
        
        values = (
            self.metrics[metric_name].get_recent_values(duration)
            if duration else [point.value for point in self.metrics[metric_name].points]
        )
        
        if not values:
            return 0.0
        
        values.sort()
        index = int((percentile / 100.0) * len(values))
        return values[min(index, len(values) - 1)]


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        collector: MetricsCollector,
        metric_name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        self.collector = collector
        self.metric_name = metric_name
        self.tags = tags
        self.timer_id = None
        self.start_time = None
    
    def __enter__(self):
        self.timer_id = self.collector.timer_start(self.metric_name)
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer_id:
            duration = self.collector.timer_end(self.timer_id, self.tags)
            return duration


# Global metrics collector
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def configure_metrics_collector(max_series_points: int = 1000) -> MetricsCollector:
    """Configure global metrics collector."""
    global _global_metrics_collector
    _global_metrics_collector = MetricsCollector(max_series_points)
    return _global_metrics_collector