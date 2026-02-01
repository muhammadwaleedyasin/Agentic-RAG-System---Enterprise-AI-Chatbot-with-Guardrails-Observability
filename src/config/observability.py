"""
Observability configuration management for RAG system.

Centralized configuration for:
- Langfuse tracing settings
- Phoenix monitoring setup
- Metrics collection parameters
- Dashboard configurations
- Alert thresholds
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class LangfuseConfig:
    """Langfuse tracing configuration."""
    enabled: bool = True
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: Optional[str] = None
    session_id: Optional[str] = None
    batch_size: int = 100
    flush_interval: float = 2.0
    debug: bool = False
    
    def __post_init__(self):
        # Load from environment if not provided
        if self.public_key is None:
            self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        if self.secret_key is None:
            self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        if self.host is None:
            self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


@dataclass
class PhoenixConfig:
    """Phoenix observability configuration."""
    enabled: bool = True
    endpoint: Optional[str] = None
    launch_app: bool = False
    port: int = 6006
    auto_instrument: bool = True
    collect_embeddings: bool = True
    collect_llm_calls: bool = True
    export_format: str = "otlp"  # "otlp" or "phoenix"
    
    def __post_init__(self):
        if self.endpoint is None:
            self.endpoint = os.getenv("PHOENIX_ENDPOINT")


@dataclass
class MetricsConfig:
    """Metrics collection configuration."""
    enabled: bool = True
    max_series_points: int = 1000
    collection_interval: float = 30.0  # seconds
    export_interval: float = 300.0  # seconds
    retention_days: int = 30
    
    # Metric collection flags
    collect_performance: bool = True
    collect_cost: bool = True
    collect_quality: bool = True
    collect_system: bool = True
    collect_user_analytics: bool = True
    
    # Performance thresholds
    latency_warning_threshold: float = 2.0  # seconds
    latency_critical_threshold: float = 5.0  # seconds
    error_rate_warning_threshold: float = 5.0  # percent
    error_rate_critical_threshold: float = 10.0  # percent
    
    # Cost thresholds
    daily_cost_warning_threshold: float = 50.0  # USD
    daily_cost_critical_threshold: float = 100.0  # USD
    cost_per_query_warning_threshold: float = 0.10  # USD
    cost_per_query_critical_threshold: float = 0.25  # USD


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    enabled: bool = True
    default_refresh_interval: str = "30s"
    default_time_range: str = "1h"
    
    # Dashboard features
    enable_rag_overview: bool = True
    enable_performance_monitoring: bool = True
    enable_cost_tracking: bool = True
    enable_quality_metrics: bool = True
    enable_system_health: bool = True
    enable_user_analytics: bool = True
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["json", "grafana"])
    export_path: str = "./dashboards"


@dataclass
class AlertConfig:
    """Alert configuration."""
    enabled: bool = True
    
    # Alert channels
    webhook_url: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    slack_webhook: Optional[str] = None
    
    # Alert rules
    alert_on_high_latency: bool = True
    alert_on_high_error_rate: bool = True
    alert_on_high_cost: bool = True
    alert_on_low_quality: bool = True
    alert_on_system_issues: bool = True
    
    # Alert thresholds (inherit from metrics config by default)
    custom_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class ObservabilityConfig:
    """Complete observability configuration."""
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    phoenix: PhoenixConfig = field(default_factory=PhoenixConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    dashboards: DashboardConfig = field(default_factory=DashboardConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    
    # Global settings
    environment: str = "development"
    service_name: str = "rag-system"
    service_version: str = "1.0.0"
    
    @classmethod
    def from_file(cls, config_path: str) -> "ObservabilityConfig":
        """Load configuration from file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        
        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
            else:
                logger.error(f"Unsupported config format: {path.suffix}")
                return cls()
            
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservabilityConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Update Langfuse config
        if 'langfuse' in data:
            langfuse_data = data['langfuse']
            config.langfuse = LangfuseConfig(**langfuse_data)
        
        # Update Phoenix config
        if 'phoenix' in data:
            phoenix_data = data['phoenix']
            config.phoenix = PhoenixConfig(**phoenix_data)
        
        # Update Metrics config
        if 'metrics' in data:
            metrics_data = data['metrics']
            config.metrics = MetricsConfig(**metrics_data)
        
        # Update Dashboard config
        if 'dashboards' in data:
            dashboard_data = data['dashboards']
            config.dashboards = DashboardConfig(**dashboard_data)
        
        # Update Alert config
        if 'alerts' in data:
            alert_data = data['alerts']
            config.alerts = AlertConfig(**alert_data)
        
        # Update global settings
        for key in ['environment', 'service_name', 'service_version']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save_to_file(self, config_path: str, format: str = "yaml"):
        """Save configuration to file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        try:
            if format.lower() in ['yaml', 'yml']:
                with open(path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate Langfuse config
        if self.langfuse.enabled:
            if not self.langfuse.public_key:
                issues.append("Langfuse public key is required when enabled")
            if not self.langfuse.secret_key:
                issues.append("Langfuse secret key is required when enabled")
        
        # Validate Phoenix config
        if self.phoenix.enabled and self.phoenix.launch_app:
            if self.phoenix.port < 1024 or self.phoenix.port > 65535:
                issues.append("Phoenix port must be between 1024 and 65535")
        
        # Validate metrics config
        if self.metrics.enabled:
            if self.metrics.max_series_points < 100:
                issues.append("max_series_points should be at least 100")
            if self.metrics.collection_interval < 1.0:
                issues.append("collection_interval should be at least 1 second")
        
        # Validate thresholds
        if self.metrics.latency_warning_threshold >= self.metrics.latency_critical_threshold:
            issues.append("latency_warning_threshold should be less than critical threshold")
        
        if self.metrics.error_rate_warning_threshold >= self.metrics.error_rate_critical_threshold:
            issues.append("error_rate_warning_threshold should be less than critical threshold")
        
        # Validate alert config
        if self.alerts.enabled:
            if not any([
                self.alerts.webhook_url,
                self.alerts.email_recipients,
                self.alerts.slack_webhook
            ]):
                issues.append("At least one alert channel must be configured when alerts are enabled")
        
        return issues
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get required environment variables."""
        env_vars = {}
        
        if self.langfuse.enabled:
            if self.langfuse.public_key:
                env_vars["LANGFUSE_PUBLIC_KEY"] = self.langfuse.public_key
            if self.langfuse.secret_key:
                env_vars["LANGFUSE_SECRET_KEY"] = self.langfuse.secret_key
            if self.langfuse.host:
                env_vars["LANGFUSE_HOST"] = self.langfuse.host
        
        if self.phoenix.enabled and self.phoenix.endpoint:
            env_vars["PHOENIX_ENDPOINT"] = self.phoenix.endpoint
        
        return env_vars


def load_observability_config(
    config_path: Optional[str] = None
) -> ObservabilityConfig:
    """Load observability configuration from file or environment."""
    if config_path:
        return ObservabilityConfig.from_file(config_path)
    
    # Try to find config file in common locations
    possible_paths = [
        "./config/observability.yaml",
        "./config/observability.yml",
        "./config/observability.json",
        "./observability.yaml",
        "./observability.yml",
        "./observability.json"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            logger.info(f"Loading observability config from {path}")
            return ObservabilityConfig.from_file(path)
    
    # Use environment variables and defaults
    logger.info("No config file found, using environment variables and defaults")
    return ObservabilityConfig()


def create_sample_config(output_path: str = "./config/observability.yaml"):
    """Create a sample observability configuration file."""
    config = ObservabilityConfig()
    
    # Set some example values
    config.langfuse.public_key = "pk_your_public_key_here"
    config.langfuse.secret_key = "sk_your_secret_key_here"
    config.phoenix.launch_app = True
    config.alerts.webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    config.alerts.email_recipients = ["admin@yourcompany.com"]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    config.save_to_file(output_path)
    
    print(f"Sample configuration created at {output_path}")
    print("Please update the configuration with your actual API keys and settings.")


# Global configuration instance
_global_config: Optional[ObservabilityConfig] = None


def get_observability_config() -> ObservabilityConfig:
    """Get global observability configuration."""
    global _global_config
    if _global_config is None:
        _global_config = load_observability_config()
    return _global_config


def set_observability_config(config: ObservabilityConfig):
    """Set global observability configuration."""
    global _global_config
    _global_config = config