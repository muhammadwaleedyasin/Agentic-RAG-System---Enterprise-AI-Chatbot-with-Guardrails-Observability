"""
Dashboard configuration templates for RAG system observability.

Provides pre-configured dashboard templates for:
- Langfuse monitoring dashboards
- Phoenix visualization configs
- Grafana dashboard definitions
- Custom monitoring layouts
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DashboardPanel:
    """Individual dashboard panel configuration."""
    title: str
    panel_type: str  # "graph", "table", "stat", "heatmap"
    metrics: List[str]
    description: Optional[str] = None
    width: int = 12
    height: int = 8
    position: Optional[Dict[str, int]] = None
    options: Optional[Dict[str, Any]] = None


@dataclass
class Dashboard:
    """Complete dashboard configuration."""
    title: str
    description: str
    panels: List[DashboardPanel]
    refresh_interval: str = "30s"
    time_range: str = "1h"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class DashboardConfig:
    """Dashboard configuration manager for RAG observability."""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize pre-built dashboard templates."""
        self.templates = {
            "rag_overview": self._create_rag_overview_dashboard(),
            "performance_monitoring": self._create_performance_dashboard(),
            "cost_tracking": self._create_cost_dashboard(),
            "quality_metrics": self._create_quality_dashboard(),
            "system_health": self._create_health_dashboard(),
            "user_analytics": self._create_user_analytics_dashboard()
        }
    
    def _create_rag_overview_dashboard(self) -> Dashboard:
        """Create RAG system overview dashboard."""
        panels = [
            DashboardPanel(
                title="Query Volume",
                panel_type="graph",
                metrics=["rag.queries.total"],
                description="Total queries processed over time",
                width=6,
                height=6,
                options={
                    "graph_type": "line",
                    "aggregation": "rate",
                    "time_window": "1m"
                }
            ),
            DashboardPanel(
                title="Average Response Latency",
                panel_type="stat",
                metrics=["rag.query.latency"],
                description="Average query response time",
                width=6,
                height=6,
                options={
                    "stat_type": "mean",
                    "unit": "seconds",
                    "threshold_warning": 2.0,
                    "threshold_critical": 5.0
                }
            ),
            DashboardPanel(
                title="Error Rate",
                panel_type="stat",
                metrics=["rag.queries.error_rate"],
                description="Percentage of failed queries",
                width=6,
                height=4,
                options={
                    "stat_type": "current",
                    "unit": "percent",
                    "threshold_warning": 5.0,
                    "threshold_critical": 10.0
                }
            ),
            DashboardPanel(
                title="Documents Processed",
                panel_type="stat",
                metrics=["rag.documents.processed"],
                description="Total documents ingested",
                width=6,
                height=4,
                options={
                    "stat_type": "sum",
                    "unit": "count"
                }
            ),
            DashboardPanel(
                title="Query Latency Distribution",
                panel_type="heatmap",
                metrics=["rag.query.latency"],
                description="Distribution of query response times",
                width=12,
                height=6,
                options={
                    "bucket_size": 0.1,
                    "max_buckets": 50
                }
            ),
            DashboardPanel(
                title="Recent Query Activity",
                panel_type="table",
                metrics=["rag.queries.total", "rag.query.latency"],
                description="Recent queries with details",
                width=12,
                height=8,
                options={
                    "columns": ["timestamp", "query_type", "latency", "success"],
                    "page_size": 20,
                    "sort_by": "timestamp"
                }
            )
        ]
        
        return Dashboard(
            title="RAG System Overview",
            description="High-level overview of RAG system performance and health",
            panels=panels,
            tags=["rag", "overview", "performance"]
        )
    
    def _create_performance_dashboard(self) -> Dashboard:
        """Create performance monitoring dashboard."""
        panels = [
            DashboardPanel(
                title="Query Latency Percentiles",
                panel_type="graph",
                metrics=["rag.query.latency"],
                description="P50, P95, P99 latency trends",
                width=12,
                height=6,
                options={
                    "percentiles": [50, 95, 99],
                    "graph_type": "line"
                }
            ),
            DashboardPanel(
                title="Embedding Generation Time",
                panel_type="graph",
                metrics=["rag.embeddings.processing_time"],
                description="Time to generate embeddings",
                width=6,
                height=6,
                options={
                    "aggregation": "mean",
                    "group_by": "model"
                }
            ),
            DashboardPanel(
                title="Vector Search Performance",
                panel_type="graph",
                metrics=["rag.retrieval.search_time"],
                description="Vector similarity search latency",
                width=6,
                height=6,
                options={
                    "aggregation": "mean"
                }
            ),
            DashboardPanel(
                title="LLM Generation Latency",
                panel_type="graph",
                metrics=["rag.llm.latency"],
                description="LLM response generation time",
                width=6,
                height=6,
                options={
                    "aggregation": "mean",
                    "group_by": "model"
                }
            ),
            DashboardPanel(
                title="Token Usage",
                panel_type="graph",
                metrics=["rag.llm.prompt_tokens", "rag.llm.completion_tokens"],
                description="Token consumption over time",
                width=6,
                height=6,
                options={
                    "graph_type": "stacked_area"
                }
            ),
            DashboardPanel(
                title="Throughput",
                panel_type="stat",
                metrics=["rag.queries.total"],
                description="Queries per second",
                width=4,
                height=4,
                options={
                    "stat_type": "rate",
                    "unit": "qps",
                    "time_window": "1m"
                }
            ),
            DashboardPanel(
                title="Cache Hit Rate",
                panel_type="stat",
                metrics=["rag.cache.hit_rate"],
                description="Percentage of cache hits",
                width=4,
                height=4,
                options={
                    "stat_type": "current",
                    "unit": "percent"
                }
            ),
            DashboardPanel(
                title="Memory Usage",
                panel_type="stat",
                metrics=["system.memory.usage"],
                description="System memory utilization",
                width=4,
                height=4,
                options={
                    "stat_type": "current",
                    "unit": "bytes",
                    "threshold_warning": 0.8,
                    "threshold_critical": 0.9
                }
            )
        ]
        
        return Dashboard(
            title="Performance Monitoring",
            description="Detailed performance metrics for RAG pipeline components",
            panels=panels,
            tags=["performance", "latency", "throughput"]
        )
    
    def _create_cost_dashboard(self) -> Dashboard:
        """Create cost tracking dashboard."""
        panels = [
            DashboardPanel(
                title="Total Cost Trend",
                panel_type="graph",
                metrics=["rag.llm.cost", "rag.embeddings.cost"],
                description="Cost accumulation over time",
                width=12,
                height=6,
                options={
                    "graph_type": "stacked_area",
                    "unit": "usd"
                }
            ),
            DashboardPanel(
                title="Cost per Query",
                panel_type="stat",
                metrics=["rag.cost.per_query"],
                description="Average cost per query",
                width=6,
                height=4,
                options={
                    "stat_type": "mean",
                    "unit": "usd"
                }
            ),
            DashboardPanel(
                title="Daily Cost",
                panel_type="stat",
                metrics=["rag.cost.daily"],
                description="Total cost today",
                width=6,
                height=4,
                options={
                    "stat_type": "sum",
                    "unit": "usd",
                    "time_range": "24h"
                }
            ),
            DashboardPanel(
                title="Cost by Model",
                panel_type="graph",
                metrics=["rag.llm.cost"],
                description="Cost breakdown by LLM model",
                width=6,
                height=6,
                options={
                    "graph_type": "pie",
                    "group_by": "model"
                }
            ),
            DashboardPanel(
                title="Token Cost Analysis",
                panel_type="table",
                metrics=["rag.llm.cost", "rag.llm.total_tokens"],
                description="Cost per token by model",
                width=6,
                height=6,
                options={
                    "columns": ["model", "total_tokens", "total_cost", "cost_per_token"],
                    "sort_by": "total_cost"
                }
            )
        ]
        
        return Dashboard(
            title="Cost Tracking",
            description="Monitor and analyze RAG system operational costs",
            panels=panels,
            tags=["cost", "billing", "optimization"]
        )
    
    def _create_quality_dashboard(self) -> Dashboard:
        """Create response quality metrics dashboard."""
        panels = [
            DashboardPanel(
                title="Relevance Score Trend",
                panel_type="graph",
                metrics=["rag.quality.relevance"],
                description="Response relevance over time",
                width=6,
                height=6,
                options={
                    "aggregation": "mean",
                    "y_min": 0,
                    "y_max": 1
                }
            ),
            DashboardPanel(
                title="Hallucination Detection",
                panel_type="graph",
                metrics=["rag.quality.hallucination"],
                description="Hallucination scores",
                width=6,
                height=6,
                options={
                    "aggregation": "mean",
                    "y_min": 0,
                    "y_max": 1,
                    "threshold_warning": 0.3,
                    "threshold_critical": 0.5
                }
            ),
            DashboardPanel(
                title="User Satisfaction",
                panel_type="stat",
                metrics=["rag.quality.user_rating"],
                description="Average user rating",
                width=4,
                height=4,
                options={
                    "stat_type": "mean",
                    "unit": "stars",
                    "y_min": 1,
                    "y_max": 5
                }
            ),
            DashboardPanel(
                title="Response Coherence",
                panel_type="stat",
                metrics=["rag.quality.coherence"],
                description="Average coherence score",
                width=4,
                height=4,
                options={
                    "stat_type": "mean",
                    "y_min": 0,
                    "y_max": 1
                }
            ),
            DashboardPanel(
                title="Quality Score Distribution",
                panel_type="heatmap",
                metrics=["rag.quality.relevance", "rag.quality.coherence"],
                description="Distribution of quality scores",
                width=4,
                height=4
            ),
            DashboardPanel(
                title="Low Quality Alerts",
                panel_type="table",
                metrics=["rag.quality.relevance", "rag.quality.hallucination"],
                description="Recent low quality responses",
                width=12,
                height=6,
                options={
                    "filters": {
                        "relevance": {"operator": "<", "value": 0.7},
                        "hallucination": {"operator": ">", "value": 0.3}
                    },
                    "columns": ["timestamp", "query", "relevance", "hallucination"],
                    "sort_by": "timestamp"
                }
            )
        ]
        
        return Dashboard(
            title="Quality Metrics",
            description="Monitor and analyze response quality and user satisfaction",
            panels=panels,
            tags=["quality", "evaluation", "user_experience"]
        )
    
    def _create_health_dashboard(self) -> Dashboard:
        """Create system health monitoring dashboard."""
        panels = [
            DashboardPanel(
                title="System Status",
                panel_type="stat",
                metrics=["system.health.status"],
                description="Overall system health status",
                width=3,
                height=4,
                options={
                    "stat_type": "current",
                    "color_mapping": {
                        "healthy": "green",
                        "warning": "yellow",
                        "critical": "red"
                    }
                }
            ),
            DashboardPanel(
                title="Error Rate",
                panel_type="stat",
                metrics=["rag.queries.error_rate"],
                description="Current error rate",
                width=3,
                height=4,
                options={
                    "stat_type": "current",
                    "unit": "percent",
                    "threshold_warning": 5.0,
                    "threshold_critical": 10.0
                }
            ),
            DashboardPanel(
                title="Available Memory",
                panel_type="stat",
                metrics=["system.memory.available"],
                description="Available system memory",
                width=3,
                height=4,
                options={
                    "stat_type": "current",
                    "unit": "bytes"
                }
            ),
            DashboardPanel(
                title="Active Connections",
                panel_type="stat",
                metrics=["system.connections.active"],
                description="Number of active connections",
                width=3,
                height=4,
                options={
                    "stat_type": "current"
                }
            ),
            DashboardPanel(
                title="Error Timeline",
                panel_type="graph",
                metrics=["rag.errors.total"],
                description="Error occurrences over time",
                width=12,
                height=6,
                options={
                    "graph_type": "line",
                    "group_by": "error_type"
                }
            ),
            DashboardPanel(
                title="Resource Utilization",
                panel_type="graph",
                metrics=["system.cpu.usage", "system.memory.usage"],
                description="CPU and memory utilization",
                width=12,
                height=6,
                options={
                    "graph_type": "line",
                    "unit": "percent"
                }
            )
        ]
        
        return Dashboard(
            title="System Health",
            description="Monitor system health and resource utilization",
            panels=panels,
            tags=["health", "monitoring", "alerts"]
        )
    
    def _create_user_analytics_dashboard(self) -> Dashboard:
        """Create user behavior analytics dashboard."""
        panels = [
            DashboardPanel(
                title="Active Users",
                panel_type="stat",
                metrics=["users.active"],
                description="Number of active users",
                width=4,
                height=4,
                options={
                    "stat_type": "count_distinct",
                    "group_by": "user_id",
                    "time_range": "1h"
                }
            ),
            DashboardPanel(
                title="Queries per User",
                panel_type="stat",
                metrics=["rag.queries.per_user"],
                description="Average queries per user",
                width=4,
                height=4,
                options={
                    "stat_type": "mean"
                }
            ),
            DashboardPanel(
                title="Session Duration",
                panel_type="stat",
                metrics=["users.session_duration"],
                description="Average session duration",
                width=4,
                height=4,
                options={
                    "stat_type": "mean",
                    "unit": "minutes"
                }
            ),
            DashboardPanel(
                title="Query Types Distribution",
                panel_type="graph",
                metrics=["rag.queries.total"],
                description="Distribution of query types",
                width=6,
                height=6,
                options={
                    "graph_type": "pie",
                    "group_by": "query_type"
                }
            ),
            DashboardPanel(
                title="User Satisfaction by Type",
                panel_type="graph",
                metrics=["rag.quality.user_rating"],
                description="User ratings by query type",
                width=6,
                height=6,
                options={
                    "aggregation": "mean",
                    "group_by": "query_type"
                }
            ),
            DashboardPanel(
                title="Usage Patterns",
                panel_type="heatmap",
                metrics=["rag.queries.total"],
                description="Query patterns by hour and day",
                width=12,
                height=6,
                options={
                    "x_axis": "hour_of_day",
                    "y_axis": "day_of_week"
                }
            )
        ]
        
        return Dashboard(
            title="User Analytics",
            description="Analyze user behavior and usage patterns",
            panels=panels,
            tags=["users", "analytics", "behavior"]
        )
    
    def get_dashboard(self, name: str) -> Optional[Dashboard]:
        """Get dashboard configuration by name."""
        return self.templates.get(name)
    
    def get_all_dashboards(self) -> Dict[str, Dashboard]:
        """Get all available dashboard configurations."""
        return self.templates.copy()
    
    def export_dashboard(
        self,
        name: str,
        format: str = "json"
    ) -> Optional[str]:
        """Export dashboard configuration in specified format."""
        dashboard = self.get_dashboard(name)
        if not dashboard:
            return None
        
        if format == "json":
            return json.dumps(asdict(dashboard), indent=2)
        elif format == "grafana":
            return self._export_grafana_dashboard(dashboard)
        elif format == "langfuse":
            return self._export_langfuse_dashboard(dashboard)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_grafana_dashboard(self, dashboard: Dashboard) -> str:
        """Export dashboard in Grafana format."""
        grafana_config = {
            "dashboard": {
                "id": None,
                "title": dashboard.title,
                "description": dashboard.description,
                "tags": dashboard.tags,
                "timezone": "browser",
                "panels": [],
                "time": {
                    "from": f"now-{dashboard.time_range}",
                    "to": "now"
                },
                "refresh": dashboard.refresh_interval,
                "schemaVersion": 27,
                "version": 1
            }
        }
        
        for i, panel in enumerate(dashboard.panels):
            grafana_panel = {
                "id": i + 1,
                "title": panel.title,
                "type": self._map_panel_type_to_grafana(panel.panel_type),
                "gridPos": {
                    "h": panel.height,
                    "w": panel.width,
                    "x": 0,
                    "y": 0
                },
                "targets": [
                    {
                        "expr": metric,
                        "refId": chr(65 + j)  # A, B, C, ...
                    }
                    for j, metric in enumerate(panel.metrics)
                ],
                "description": panel.description or ""
            }
            
            if panel.options:
                grafana_panel.update(panel.options)
            
            grafana_config["dashboard"]["panels"].append(grafana_panel)
        
        return json.dumps(grafana_config, indent=2)
    
    def _export_langfuse_dashboard(self, dashboard: Dashboard) -> str:
        """Export dashboard in Langfuse format."""
        langfuse_config = {
            "name": dashboard.title,
            "description": dashboard.description,
            "charts": []
        }
        
        for panel in dashboard.panels:
            chart = {
                "name": panel.title,
                "type": panel.panel_type,
                "metrics": panel.metrics,
                "description": panel.description or "",
                "config": panel.options or {}
            }
            langfuse_config["charts"].append(chart)
        
        return json.dumps(langfuse_config, indent=2)
    
    def _map_panel_type_to_grafana(self, panel_type: str) -> str:
        """Map internal panel types to Grafana panel types."""
        mapping = {
            "graph": "graph",
            "table": "table",
            "stat": "stat",
            "heatmap": "heatmap"
        }
        return mapping.get(panel_type, "graph")
    
    def create_custom_dashboard(
        self,
        name: str,
        title: str,
        description: str,
        panels: List[DashboardPanel],
        **kwargs
    ) -> Dashboard:
        """Create a custom dashboard configuration."""
        dashboard = Dashboard(
            title=title,
            description=description,
            panels=panels,
            **kwargs
        )
        
        self.templates[name] = dashboard
        return dashboard