"""
Performance Analyzer and Bottleneck Detection for Production-Scale RAG Systems

This module provides comprehensive performance analysis, bottleneck detection,
and optimization recommendations for large-scale document retrieval systems.
"""

import asyncio
import logging
import time
import json
import threading
import statistics
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, NamedTuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import sqlite3
import pickle
import psutil
import tracemalloc
from pathlib import Path
import weakref

# External dependencies
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import memory_profiler

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BottleneckInfo:
    """Information about detected bottleneck"""
    component: str
    metric: str
    severity: str  # "low", "medium", "high", "critical"
    current_value: float
    threshold_value: float
    impact_score: float
    recommendation: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceProfile:
    """Performance profile for a component"""
    component_name: str
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    error_rate: float
    memory_usage: float
    cpu_usage: float
    bottlenecks: List[BottleneckInfo]
    timestamp: float

class MetricCollector(ABC):
    """Abstract base class for metric collectors"""
    
    @abstractmethod
    async def collect_metrics(self) -> List[PerformanceMetric]:
        """Collect performance metrics"""
        pass
    
    @abstractmethod
    def get_component_name(self) -> str:
        """Get component name"""
        pass

class SystemMetricCollector(MetricCollector):
    """Collector for system-level metrics"""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self._last_cpu_times = None
        self._last_network_io = None
        self._last_disk_io = None
    
    async def collect_metrics(self) -> List[PerformanceMetric]:
        """Collect system metrics"""
        metrics = []
        timestamp = time.time()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            metrics.extend([
                PerformanceMetric("cpu_usage", cpu_percent, "%", timestamp, "system"),
                PerformanceMetric("cpu_count", cpu_count, "cores", timestamp, "system"),
                PerformanceMetric("load_avg_1m", load_avg[0], "load", timestamp, "system"),
                PerformanceMetric("load_avg_5m", load_avg[1], "load", timestamp, "system"),
                PerformanceMetric("load_avg_15m", load_avg[2], "load", timestamp, "system"),
            ])
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.extend([
                PerformanceMetric("memory_usage", memory.percent, "%", timestamp, "system"),
                PerformanceMetric("memory_total", memory.total, "bytes", timestamp, "system"),
                PerformanceMetric("memory_available", memory.available, "bytes", timestamp, "system"),
                PerformanceMetric("swap_usage", swap.percent, "%", timestamp, "system"),
                PerformanceMetric("swap_total", swap.total, "bytes", timestamp, "system"),
            ])
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics.append(
                PerformanceMetric("disk_usage", disk_usage.percent, "%", timestamp, "system")
            )
            
            if disk_io:
                if self._last_disk_io:
                    read_rate = (disk_io.read_bytes - self._last_disk_io.read_bytes) / self.collection_interval
                    write_rate = (disk_io.write_bytes - self._last_disk_io.write_bytes) / self.collection_interval
                    
                    metrics.extend([
                        PerformanceMetric("disk_read_rate", read_rate, "bytes/s", timestamp, "system"),
                        PerformanceMetric("disk_write_rate", write_rate, "bytes/s", timestamp, "system"),
                    ])
                
                self._last_disk_io = disk_io
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            if network_io:
                if self._last_network_io:
                    recv_rate = (network_io.bytes_recv - self._last_network_io.bytes_recv) / self.collection_interval
                    sent_rate = (network_io.bytes_sent - self._last_network_io.bytes_sent) / self.collection_interval
                    
                    metrics.extend([
                        PerformanceMetric("network_recv_rate", recv_rate, "bytes/s", timestamp, "system"),
                        PerformanceMetric("network_sent_rate", sent_rate, "bytes/s", timestamp, "system"),
                    ])
                
                self._last_network_io = network_io
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    def get_component_name(self) -> str:
        return "system"

class ApplicationMetricCollector(MetricCollector):
    """Collector for application-specific metrics"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self._request_times: deque = deque(maxlen=1000)
        self._error_count = 0
        self._success_count = 0
        self._memory_tracker = None
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request for metric calculation"""
        self._request_times.append(response_time)
        if success:
            self._success_count += 1
        else:
            self._error_count += 1
    
    async def collect_metrics(self) -> List[PerformanceMetric]:
        """Collect application metrics"""
        metrics = []
        timestamp = time.time()
        
        try:
            # Response time metrics
            if self._request_times:
                avg_response_time = statistics.mean(self._request_times)
                p95_response_time = np.percentile(list(self._request_times), 95)
                p99_response_time = np.percentile(list(self._request_times), 99)
                
                metrics.extend([
                    PerformanceMetric("avg_response_time", avg_response_time, "seconds", timestamp, self.component_name),
                    PerformanceMetric("p95_response_time", p95_response_time, "seconds", timestamp, self.component_name),
                    PerformanceMetric("p99_response_time", p99_response_time, "seconds", timestamp, self.component_name),
                ])
            
            # Throughput metrics
            total_requests = self._success_count + self._error_count
            if total_requests > 0:
                error_rate = self._error_count / total_requests
                throughput = len(self._request_times) / 60  # Requests per minute (approximation)
                
                metrics.extend([
                    PerformanceMetric("error_rate", error_rate, "ratio", timestamp, self.component_name),
                    PerformanceMetric("throughput", throughput, "req/min", timestamp, self.component_name),
                    PerformanceMetric("total_requests", total_requests, "count", timestamp, self.component_name),
                ])
            
            # Memory metrics (if tracking enabled)
            if self._memory_tracker:
                current_usage = memory_profiler.memory_usage()[0]
                metrics.append(
                    PerformanceMetric("memory_usage", current_usage, "MB", timestamp, self.component_name)
                )
        
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
        
        return metrics
    
    def get_component_name(self) -> str:
        return self.component_name
    
    def enable_memory_tracking(self):
        """Enable memory tracking for this component"""
        self._memory_tracker = True

class BottleneckDetector:
    """Detects performance bottlenecks based on metrics"""
    
    def __init__(self, thresholds: Dict[str, Dict[str, float]] = None):
        self.thresholds = thresholds or self._get_default_thresholds()
        self._historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get default performance thresholds"""
        return {
            "system": {
                "cpu_usage": {"warning": 70.0, "critical": 90.0},
                "memory_usage": {"warning": 80.0, "critical": 95.0},
                "disk_usage": {"warning": 80.0, "critical": 95.0},
                "load_avg_1m": {"warning": 2.0, "critical": 5.0},
            },
            "application": {
                "avg_response_time": {"warning": 2.0, "critical": 5.0},
                "p95_response_time": {"warning": 5.0, "critical": 10.0},
                "p99_response_time": {"warning": 10.0, "critical": 20.0},
                "error_rate": {"warning": 0.05, "critical": 0.10},
            },
            "database": {
                "query_time": {"warning": 1.0, "critical": 3.0},
                "connection_usage": {"warning": 80.0, "critical": 95.0},
                "lock_wait_time": {"warning": 0.5, "critical": 2.0},
            }
        }
    
    def analyze_metrics(self, metrics: List[PerformanceMetric]) -> List[BottleneckInfo]:
        """Analyze metrics and detect bottlenecks"""
        bottlenecks = []
        
        # Group metrics by component
        component_metrics = defaultdict(list)
        for metric in metrics:
            component_metrics[metric.component].append(metric)
        
        # Analyze each component
        for component, component_metrics_list in component_metrics.items():
            component_bottlenecks = self._analyze_component(component, component_metrics_list)
            bottlenecks.extend(component_bottlenecks)
        
        # Sort by severity and impact
        bottlenecks.sort(key=lambda b: (self._severity_score(b.severity), b.impact_score), reverse=True)
        
        return bottlenecks
    
    def _analyze_component(self, component: str, metrics: List[PerformanceMetric]) -> List[BottleneckInfo]:
        """Analyze metrics for a specific component"""
        bottlenecks = []
        
        # Get thresholds for this component
        component_thresholds = self.thresholds.get(component, {})
        
        for metric in metrics:
            # Store historical data
            self._historical_data[f"{component}.{metric.name}"].append(metric.value)
            
            # Check thresholds
            metric_thresholds = component_thresholds.get(metric.name, {})
            
            severity = None
            threshold_value = None
            
            if "critical" in metric_thresholds and metric.value >= metric_thresholds["critical"]:
                severity = "critical"
                threshold_value = metric_thresholds["critical"]
            elif "warning" in metric_thresholds and metric.value >= metric_thresholds["warning"]:
                severity = "warning"
                threshold_value = metric_thresholds["warning"]
            
            if severity:
                # Calculate impact score
                impact_score = self._calculate_impact_score(component, metric.name, metric.value, threshold_value)
                
                # Generate recommendation
                recommendation = self._generate_recommendation(component, metric.name, metric.value, severity)
                
                bottleneck = BottleneckInfo(
                    component=component,
                    metric=metric.name,
                    severity=severity,
                    current_value=metric.value,
                    threshold_value=threshold_value,
                    impact_score=impact_score,
                    recommendation=recommendation,
                    timestamp=metric.timestamp,
                    context=metric.metadata
                )
                
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _calculate_impact_score(self, component: str, metric_name: str, current_value: float, threshold_value: float) -> float:
        """Calculate impact score for a bottleneck"""
        # Base score on how much the value exceeds the threshold
        base_score = (current_value - threshold_value) / threshold_value
        
        # Apply component-specific weighting
        component_weights = {
            "system": {"cpu_usage": 0.9, "memory_usage": 0.8, "disk_usage": 0.6},
            "application": {"avg_response_time": 1.0, "error_rate": 0.9},
            "database": {"query_time": 0.8, "connection_usage": 0.7}
        }
        
        weight = component_weights.get(component, {}).get(metric_name, 0.5)
        
        return min(base_score * weight, 1.0)
    
    def _generate_recommendation(self, component: str, metric_name: str, value: float, severity: str) -> str:
        """Generate optimization recommendation"""
        recommendations = {
            ("system", "cpu_usage"): "Consider scaling horizontally, optimizing algorithms, or upgrading CPU",
            ("system", "memory_usage"): "Increase available memory, optimize memory usage, or implement caching",
            ("system", "disk_usage"): "Free up disk space, implement log rotation, or add storage capacity",
            ("application", "avg_response_time"): "Optimize queries, implement caching, or scale application",
            ("application", "error_rate"): "Fix application bugs, improve error handling, or review recent changes",
            ("database", "query_time"): "Add database indexes, optimize queries, or upgrade database hardware"
        }
        
        return recommendations.get((component, metric_name), "Monitor closely and investigate root cause")
    
    def _severity_score(self, severity: str) -> int:
        """Convert severity to numeric score"""
        return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(severity, 0)
    
    def get_trend_analysis(self, component: str, metric_name: str, window_size: int = 10) -> Dict[str, Any]:
        """Analyze trends for a specific metric"""
        key = f"{component}.{metric_name}"
        data = list(self._historical_data[key])
        
        if len(data) < 2:
            return {"trend": "insufficient_data", "slope": 0.0, "correlation": 0.0}
        
        # Calculate trend
        x = np.arange(len(data))
        y = np.array(data)
        
        if len(data) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trend = "stable"
            if abs(slope) > 0.1:
                trend = "increasing" if slope > 0 else "decreasing"
            
            return {
                "trend": trend,
                "slope": slope,
                "correlation": r_value,
                "p_value": p_value,
                "recent_average": np.mean(data[-window_size:]) if len(data) >= window_size else np.mean(data),
                "historical_average": np.mean(data)
            }
        
        return {"trend": "stable", "slope": 0.0, "correlation": 0.0}

class PerformanceAnalyzer:
    """Main performance analyzer and monitoring system"""
    
    def __init__(self, 
                 collection_interval: float = 30.0,
                 retention_hours: int = 168,  # 1 week
                 db_path: str = "performance_metrics.db"):
        
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.db_path = db_path
        
        # Components
        self.collectors: List[MetricCollector] = []
        self.bottleneck_detector = BottleneckDetector()
        
        # Data storage
        self._metrics_buffer: deque = deque(maxlen=10000)
        self._bottlenecks_buffer: deque = deque(maxlen=1000)
        
        # Background tasks
        self._collection_task = None
        self._analysis_task = None
        self._cleanup_task = None
        self._running = False
        
        # Initialize database
        self._init_database()
        
        # Add default collectors
        self.add_collector(SystemMetricCollector(collection_interval))
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    value REAL,
                    unit TEXT,
                    timestamp REAL,
                    component TEXT,
                    metadata TEXT
                )
            """)
            
            # Create bottlenecks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bottlenecks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT,
                    metric TEXT,
                    severity TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    impact_score REAL,
                    recommendation TEXT,
                    timestamp REAL,
                    context TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_component ON metrics(component)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bottlenecks_timestamp ON bottlenecks(timestamp)")
            
            conn.commit()
            conn.close()
            
            logger.info(f"Initialized performance database at {self.db_path}")
        
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def add_collector(self, collector: MetricCollector):
        """Add a metric collector"""
        self.collectors.append(collector)
        logger.info(f"Added collector for component: {collector.get_component_name()}")
    
    def remove_collector(self, component_name: str):
        """Remove a metric collector"""
        self.collectors = [c for c in self.collectors if c.get_component_name() != component_name]
        logger.info(f"Removed collector for component: {component_name}")
    
    async def start(self):
        """Start performance monitoring"""
        if self._running:
            logger.warning("Performance analyzer already running")
            return
        
        self._running = True
        
        # Start background tasks
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Performance analyzer started")
    
    async def stop(self):
        """Stop performance monitoring"""
        self._running = False
        
        # Cancel background tasks
        for task in [self._collection_task, self._analysis_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
        
        # Flush remaining data to database
        await self._flush_to_database()
        
        logger.info("Performance analyzer stopped")
    
    async def _collection_loop(self):
        """Background metric collection loop"""
        while self._running:
            try:
                # Collect metrics from all collectors
                all_metrics = []
                
                for collector in self.collectors:
                    try:
                        metrics = await collector.collect_metrics()
                        all_metrics.extend(metrics)
                    except Exception as e:
                        logger.error(f"Collector {collector.get_component_name()} failed: {e}")
                
                # Store metrics in buffer
                self._metrics_buffer.extend(all_metrics)
                
                # Flush to database periodically
                if len(self._metrics_buffer) >= 100:
                    await self._flush_to_database()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
    
    async def _analysis_loop(self):
        """Background bottleneck analysis loop"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Analyze every minute
                
                # Get recent metrics
                recent_metrics = list(self._metrics_buffer)[-50:]  # Last 50 metrics
                
                if recent_metrics:
                    # Detect bottlenecks
                    bottlenecks = self.bottleneck_detector.analyze_metrics(recent_metrics)
                    
                    # Store bottlenecks
                    self._bottlenecks_buffer.extend(bottlenecks)
                    
                    # Log critical bottlenecks
                    critical_bottlenecks = [b for b in bottlenecks if b.severity == "critical"]
                    for bottleneck in critical_bottlenecks:
                        logger.critical(f"Critical bottleneck detected: {bottleneck.component}.{bottleneck.metric} "
                                      f"= {bottleneck.current_value} (threshold: {bottleneck.threshold_value})")
                
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old data
                cutoff_time = time.time() - (self.retention_hours * 3600)
                
                conn = sqlite3.connect(self.db_path)
                conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))
                conn.execute("DELETE FROM bottlenecks WHERE timestamp < ?", (cutoff_time,))
                conn.commit()
                conn.close()
                
                logger.debug(f"Cleaned up metrics older than {self.retention_hours} hours")
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _flush_to_database(self):
        """Flush buffered data to database"""
        if not self._metrics_buffer and not self._bottlenecks_buffer:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Flush metrics
            if self._metrics_buffer:
                metrics_data = []
                while self._metrics_buffer:
                    metric = self._metrics_buffer.popleft()
                    metrics_data.append((
                        metric.name, metric.value, metric.unit, metric.timestamp,
                        metric.component, json.dumps(metric.metadata)
                    ))
                
                conn.executemany("""
                    INSERT INTO metrics (name, value, unit, timestamp, component, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, metrics_data)
            
            # Flush bottlenecks
            if self._bottlenecks_buffer:
                bottlenecks_data = []
                while self._bottlenecks_buffer:
                    bottleneck = self._bottlenecks_buffer.popleft()
                    bottlenecks_data.append((
                        bottleneck.component, bottleneck.metric, bottleneck.severity,
                        bottleneck.current_value, bottleneck.threshold_value,
                        bottleneck.impact_score, bottleneck.recommendation,
                        bottleneck.timestamp, json.dumps(bottleneck.context)
                    ))
                
                conn.executemany("""
                    INSERT INTO bottlenecks (component, metric, severity, current_value,
                                           threshold_value, impact_score, recommendation,
                                           timestamp, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, bottlenecks_data)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to flush data to database: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current performance status"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent metrics (last 5 minutes)
            cutoff_time = time.time() - 300
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT component, name, AVG(value) as avg_value, MAX(value) as max_value
                FROM metrics 
                WHERE timestamp > ?
                GROUP BY component, name
            """, (cutoff_time,))
            
            recent_metrics = {}
            for row in cursor.fetchall():
                component, name, avg_value, max_value = row
                if component not in recent_metrics:
                    recent_metrics[component] = {}
                recent_metrics[component][name] = {"avg": avg_value, "max": max_value}
            
            # Get recent bottlenecks
            cursor.execute("""
                SELECT component, metric, severity, current_value, recommendation
                FROM bottlenecks
                WHERE timestamp > ?
                ORDER BY impact_score DESC
                LIMIT 10
            """, (cutoff_time,))
            
            recent_bottlenecks = [
                {
                    "component": row[0],
                    "metric": row[1], 
                    "severity": row[2],
                    "current_value": row[3],
                    "recommendation": row[4]
                }
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return {
                "status": "running" if self._running else "stopped",
                "collectors": [c.get_component_name() for c in self.collectors],
                "recent_metrics": recent_metrics,
                "recent_bottlenecks": recent_bottlenecks,
                "buffer_sizes": {
                    "metrics": len(self._metrics_buffer),
                    "bottlenecks": len(self._bottlenecks_buffer)
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to get current status: {e}")
            return {"error": str(e)}
    
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_time = time.time() - (hours * 3600)
            
            # Component performance profiles
            profiles = {}
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT component, name, AVG(value), MIN(value), MAX(value), COUNT(*)
                FROM metrics
                WHERE timestamp > ?
                GROUP BY component, name
            """, (cutoff_time,))
            
            for row in cursor.fetchall():
                component, metric_name, avg_val, min_val, max_val, count = row
                
                if component not in profiles:
                    profiles[component] = {}
                
                profiles[component][metric_name] = {
                    "average": avg_val,
                    "minimum": min_val,
                    "maximum": max_val,
                    "sample_count": count
                }
            
            # Bottleneck summary
            cursor.execute("""
                SELECT component, metric, severity, COUNT(*), AVG(impact_score)
                FROM bottlenecks
                WHERE timestamp > ?
                GROUP BY component, metric, severity
                ORDER BY AVG(impact_score) DESC
            """, (cutoff_time,))
            
            bottleneck_summary = [
                {
                    "component": row[0],
                    "metric": row[1],
                    "severity": row[2],
                    "occurrence_count": row[3],
                    "avg_impact_score": row[4]
                }
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return {
                "report_period_hours": hours,
                "generated_at": time.time(),
                "component_profiles": profiles,
                "bottleneck_summary": bottleneck_summary,
                "recommendations": self._generate_optimization_recommendations(profiles, bottleneck_summary)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}
    
    def _generate_optimization_recommendations(self, profiles: Dict, bottlenecks: List) -> List[str]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # Analyze system metrics
        if "system" in profiles:
            system_profile = profiles["system"]
            
            if system_profile.get("cpu_usage", {}).get("average", 0) > 80:
                recommendations.append("High CPU usage detected - consider scaling horizontally or optimizing algorithms")
            
            if system_profile.get("memory_usage", {}).get("average", 0) > 80:
                recommendations.append("High memory usage detected - implement memory optimization or increase available memory")
            
            if system_profile.get("disk_usage", {}).get("average", 0) > 80:
                recommendations.append("High disk usage detected - implement data archiving or increase storage capacity")
        
        # Analyze bottlenecks
        critical_bottlenecks = [b for b in bottlenecks if b["severity"] == "critical"]
        if critical_bottlenecks:
            recommendations.append(f"Address {len(critical_bottlenecks)} critical bottlenecks immediately")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System performance appears normal - continue monitoring")
        
        return recommendations

# Factory functions
def create_performance_analyzer(
    collection_interval: float = 30.0,
    retention_hours: int = 168,
    auto_start: bool = True
) -> PerformanceAnalyzer:
    """Factory function to create performance analyzer"""
    analyzer = PerformanceAnalyzer(
        collection_interval=collection_interval,
        retention_hours=retention_hours
    )
    
    if auto_start:
        asyncio.create_task(analyzer.start())
    
    return analyzer

def create_application_collector(component_name: str) -> ApplicationMetricCollector:
    """Factory function to create application metric collector"""
    return ApplicationMetricCollector(component_name)

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create performance analyzer
        analyzer = create_performance_analyzer(collection_interval=5.0)
        
        # Add application collectors
        search_collector = create_application_collector("search_engine")
        rerank_collector = create_application_collector("reranking_engine")
        
        analyzer.add_collector(search_collector)
        analyzer.add_collector(rerank_collector)
        
        # Start monitoring
        await analyzer.start()
        
        # Simulate some requests
        for i in range(20):
            # Simulate search requests
            search_collector.record_request(np.random.normal(0.5, 0.1), True)
            
            # Simulate reranking requests
            rerank_collector.record_request(np.random.normal(1.2, 0.3), np.random.random() > 0.05)
            
            await asyncio.sleep(0.1)
        
        # Wait a bit for collection
        await asyncio.sleep(10)
        
        # Get current status
        status = analyzer.get_current_status()
        print("Current Status:")
        print(json.dumps(status, indent=2, default=str))
        
        # Generate performance report
        report = analyzer.generate_performance_report(hours=1)
        print("\nPerformance Report:")
        print(json.dumps(report, indent=2, default=str))
        
        # Stop analyzer
        await analyzer.stop()
    
    asyncio.run(main())