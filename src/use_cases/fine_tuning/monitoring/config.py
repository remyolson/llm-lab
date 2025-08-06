"""
Monitoring Configuration Utilities

This module provides easy configuration and setup for monitoring systems
in the fine-tuning pipeline.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .alerts import AlertCondition, AlertSeverity, AlertSystem
from .integrations import MonitoringIntegration
from .structured_logger import LogLevel, StructuredLogger


@dataclass
class MonitoringConfig:
    """Configuration for monitoring setup."""

    # Platforms to enable
    platforms: List[str] = field(default_factory=lambda: ["tensorboard"])

    # Project configuration
    project_name: str = "fine_tuning_experiment"
    experiment_name: Optional[str] = None

    # Logging configuration
    log_level: str = "INFO"
    structured_logging: bool = True
    log_file: Optional[str] = None

    # Resource monitoring
    resource_monitoring: bool = True
    resource_interval: int = 30  # seconds

    # Alerting configuration
    enable_alerts: bool = True
    alert_cooldown: int = 300  # seconds

    # Platform-specific configs
    wandb_config: Dict[str, Any] = field(default_factory=dict)
    tensorboard_config: Dict[str, Any] = field(default_factory=dict)
    mlflow_config: Dict[str, Any] = field(default_factory=dict)

    # Custom logging
    custom_metrics: List[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, config_path: str) -> "MonitoringConfig":
        """Load configuration from file."""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            if path.suffix == ".yaml" or path.suffix == ".yml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(**data)

    def to_file(self, config_path: str):
        """Save configuration to file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "platforms": self.platforms,
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "log_level": self.log_level,
            "structured_logging": self.structured_logging,
            "log_file": self.log_file,
            "resource_monitoring": self.resource_monitoring,
            "resource_interval": self.resource_interval,
            "enable_alerts": self.enable_alerts,
            "alert_cooldown": self.alert_cooldown,
            "wandb_config": self.wandb_config,
            "tensorboard_config": self.tensorboard_config,
            "mlflow_config": self.mlflow_config,
            "custom_metrics": self.custom_metrics,
        }

        with open(path, "w") as f:
            if path.suffix == ".yaml" or path.suffix == ".yml":
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)


class MonitoringSetup:
    """Utility class for setting up monitoring from configuration."""

    def __init__(self, config: MonitoringConfig):
        """Initialize monitoring setup."""
        self.config = config
        self.monitoring = None
        self.logger = None
        self.alert_system = None

    def setup(self) -> Dict[str, Any]:
        """Set up all monitoring components."""
        components = {}

        # Setup main monitoring integration
        self.monitoring = MonitoringIntegration(
            platforms=self.config.platforms,
            project_name=self.config.project_name,
            experiment_name=self.config.experiment_name,
            enable_resource_monitoring=self.config.resource_monitoring,
            resource_interval=self.config.resource_interval,
            **self._get_platform_configs(),
        )
        components["monitoring"] = self.monitoring

        # Setup structured logging
        if self.config.structured_logging:
            log_level = getattr(LogLevel, self.config.log_level.upper())
            self.logger = StructuredLogger(
                name=self.config.project_name, level=log_level, output_file=self.config.log_file
            )
            components["logger"] = self.logger

        # Setup alert system
        if self.config.enable_alerts:
            self.alert_system = AlertSystem()
            self._setup_default_alerts()
            components["alerts"] = self.alert_system

        return components

    def _get_platform_configs(self) -> Dict[str, Any]:
        """Get platform-specific configurations."""
        configs = {}

        if "wandb" in self.config.platforms and self.config.wandb_config:
            configs["wandb_config"] = self.config.wandb_config

        if "tensorboard" in self.config.platforms and self.config.tensorboard_config:
            configs["tensorboard_config"] = self.config.tensorboard_config

        if "mlflow" in self.config.platforms and self.config.mlflow_config:
            configs["mlflow_config"] = self.config.mlflow_config

        return configs

    def _setup_default_alerts(self):
        """Set up default alerts for common training issues."""
        from .alerts import AnomalyDetectionAlert, ConsoleNotifier, ThresholdAlert

        # Loss explosion alert
        self.alert_system.add_alert(
            "loss_explosion",
            ThresholdAlert(
                name="loss_explosion",
                metric="loss",
                threshold=10.0,
                condition=AlertCondition.GREATER_THAN,
                severity=AlertSeverity.ERROR,
                message="Training loss has exploded (> 10.0)",
                notifiers=[ConsoleNotifier()],
            ),
        )

        # Loss stagnation alert
        self.alert_system.add_alert(
            "loss_stagnation",
            AnomalyDetectionAlert(
                name="loss_stagnation",
                metric="loss",
                window_size=50,
                threshold=0.001,
                severity=AlertSeverity.WARNING,
                message="Training loss appears to have stagnated",
                notifiers=[ConsoleNotifier()],
            ),
        )

        # GPU memory alert
        self.alert_system.add_alert(
            "gpu_memory_high",
            ThresholdAlert(
                name="gpu_memory_high",
                metric="gpu_memory_percent",
                threshold=90.0,
                condition=AlertCondition.GREATER_THAN,
                severity=AlertSeverity.WARNING,
                message="GPU memory usage is high (> 90%)",
                notifiers=[ConsoleNotifier()],
            ),
        )

        # Learning rate alert
        self.alert_system.add_alert(
            "lr_too_high",
            ThresholdAlert(
                name="lr_too_high",
                metric="learning_rate",
                threshold=1e-2,
                condition=AlertCondition.GREATER_THAN,
                severity=AlertSeverity.WARNING,
                message="Learning rate may be too high (> 0.01)",
                notifiers=[ConsoleNotifier()],
            ),
        )


def create_default_config(
    platforms: List[str] = None,
    project_name: str = "fine_tuning_experiment",
    enable_wandb: bool = False,
    enable_alerts: bool = True,
) -> MonitoringConfig:
    """Create a default monitoring configuration."""
    if platforms is None:
        platforms = ["tensorboard"]
        if enable_wandb:
            platforms.append("wandb")

    config = MonitoringConfig(
        platforms=platforms,
        project_name=project_name,
        enable_alerts=enable_alerts,
        resource_monitoring=True,
    )

    # Configure Weights & Biases if enabled
    if "wandb" in platforms:
        config.wandb_config = {
            "project": project_name,
            "job_type": "fine-tuning",
            "save_code": True,
            "monitor_gym": True,
        }

    # Configure TensorBoard
    if "tensorboard" in platforms:
        config.tensorboard_config = {
            "log_dir": f"./tensorboard_logs/{project_name}",
            "comment": project_name,
        }

    # Configure MLflow if enabled
    if "mlflow" in platforms:
        config.mlflow_config = {"experiment_name": project_name, "tracking_uri": "file:./mlruns"}

    return config


def setup_monitoring_from_env() -> Optional[Dict[str, Any]]:
    """Set up monitoring from environment variables."""
    config_file = os.getenv("LLLM_MONITORING_CONFIG")

    if config_file and Path(config_file).exists():
        config = MonitoringConfig.from_file(config_file)
    else:
        # Create default configuration from environment
        platforms = []

        if os.getenv("WANDB_API_KEY"):
            platforms.append("wandb")

        # Always include TensorBoard as default
        platforms.append("tensorboard")

        if os.getenv("MLFLOW_TRACKING_URI"):
            platforms.append("mlflow")

        project_name = os.getenv("LLLM_PROJECT_NAME", "fine_tuning_experiment")

        config = create_default_config(
            platforms=platforms, project_name=project_name, enable_wandb="wandb" in platforms
        )

    # Setup monitoring
    setup = MonitoringSetup(config)
    return setup.setup()


def get_monitoring_status(monitoring: MonitoringIntegration) -> Dict[str, Any]:
    """Get current monitoring status."""
    status = {
        "platforms": monitoring.platforms,
        "project_name": monitoring.project_name,
        "experiment_name": monitoring.experiment_name,
        "active_platforms": [],
        "resource_monitoring": monitoring.resource_monitor is not None,
        "metrics_logged": 0,
        "alerts_active": 0,
    }

    # Check platform status
    for platform in monitoring.platforms:
        if platform == "wandb" and hasattr(monitoring, "wandb") and monitoring.wandb:
            status["active_platforms"].append("wandb")
        elif (
            platform == "tensorboard"
            and hasattr(monitoring, "tensorboard")
            and monitoring.tensorboard
        ):
            status["active_platforms"].append("tensorboard")
        elif platform == "mlflow" and hasattr(monitoring, "mlflow") and monitoring.mlflow:
            status["active_platforms"].append("mlflow")

    # Get metrics count
    if hasattr(monitoring, "metrics_logged"):
        status["metrics_logged"] = monitoring.metrics_logged

    # Get alerts count
    if hasattr(monitoring, "alert_manager") and monitoring.alert_manager:
        status["alerts_active"] = len(monitoring.alert_manager.alerts)

    return status


# Convenience functions for quick setup
def quick_setup(
    platforms: List[str] = None, project_name: str = "fine_tuning_experiment"
) -> MonitoringIntegration:
    """Quickly set up monitoring with default configuration."""
    config = create_default_config(platforms, project_name)
    setup = MonitoringSetup(config)
    components = setup.setup()
    return components["monitoring"]


def setup_with_recipe(recipe: Dict[str, Any]) -> Optional[MonitoringIntegration]:
    """Set up monitoring based on recipe configuration."""
    monitoring_config = recipe.get("monitoring", {})

    if not monitoring_config:
        return None

    # Extract configuration from recipe
    platforms = monitoring_config.get("platforms", ["tensorboard"])
    project_name = monitoring_config.get("project_name", recipe.get("name", "fine_tuning"))

    config = MonitoringConfig(
        platforms=platforms,
        project_name=project_name,
        experiment_name=recipe.get("name"),
        enable_alerts=monitoring_config.get("enable_alerts", True),
        resource_monitoring=monitoring_config.get("resource_monitoring", True),
    )

    # Add platform-specific configs from recipe
    for platform in platforms:
        platform_config = monitoring_config.get(f"{platform}_config", {})
        setattr(config, f"{platform}_config", platform_config)

    setup = MonitoringSetup(config)
    components = setup.setup()
    return components["monitoring"]
