"""
Monitoring and Logging Integrations for Fine-Tuning

This module provides comprehensive monitoring integration with popular ML
experiment tracking platforms including Weights & Biases, TensorBoard, and MLflow,
along with custom logging and alerting capabilities.

Example:
    # Initialize monitoring
    monitor = MonitoringIntegration(
        platforms=["wandb", "tensorboard"],
        project_name="fine_tuning_experiment"
    )
    
    # Log metrics
    monitor.log_metrics({
        "loss": 0.5,
        "accuracy": 0.95
    }, step=100)
    
    # Set up alerts
    monitor.add_alert(
        metric="loss",
        threshold=2.0,
        condition="greater_than"
    )
"""

import os
import json
import logging
import time
import psutil
import GPUtil
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import warnings
import numpy as np
from collections import defaultdict, deque
import threading
import queue

# Import platform-specific libraries
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import torch for model logging
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert configuration."""
    metric: str
    threshold: float
    condition: str = "greater_than"  # greater_than, less_than, equals
    message: str = ""
    callback: Optional[Callable] = None
    cooldown_seconds: int = 300  # 5 minutes
    last_triggered: Optional[float] = None


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_metrics: List[Dict[str, Any]] = field(default_factory=list)
    disk_usage_percent: float = 0.0
    network_mb_sent: float = 0.0
    network_mb_recv: float = 0.0


class BaseIntegration(ABC):
    """Base class for monitoring integrations."""
    
    @abstractmethod
    def init(self, **kwargs):
        """Initialize the integration."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        pass
    
    @abstractmethod
    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """Log an artifact."""
        pass
    
    @abstractmethod
    def finish(self):
        """Finish logging and cleanup."""
        pass


class WandbIntegration(BaseIntegration):
    """Weights & Biases integration."""
    
    def __init__(self):
        """Initialize W&B integration."""
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        
        self.run = None
        self.initialized = False
    
    def init(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize W&B run.
        
        Args:
            project: Project name
            name: Run name
            config: Configuration dictionary
            tags: List of tags
            **kwargs: Additional W&B init parameters
        """
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            **kwargs
        )
        self.initialized = True
        logger.info(f"Initialized W&B run: {self.run.id}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B."""
        if not self.initialized:
            logger.warning("W&B not initialized. Skipping metric logging.")
            return
        
        wandb.log(metrics, step=step)
    
    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """Log artifact to W&B."""
        if not self.initialized:
            logger.warning("W&B not initialized. Skipping artifact logging.")
            return
        
        artifact = wandb.Artifact(
            name=f"{artifact_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=artifact_type
        )
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)
    
    def log_model(self, model_path: str, model_name: str):
        """Log model to W&B."""
        self.log_artifact(model_path, artifact_type="model")
        
        # Log model architecture if available
        if TORCH_AVAILABLE and model_path.endswith('.pt'):
            try:
                model = torch.load(model_path, map_location='cpu')
                wandb.watch(model)
            except Exception as e:
                logger.warning(f"Failed to log model architecture: {e}")
    
    def finish(self):
        """Finish W&B run."""
        if self.initialized:
            wandb.finish()
            self.initialized = False


class TensorBoardIntegration(BaseIntegration):
    """TensorBoard integration."""
    
    def __init__(self):
        """Initialize TensorBoard integration."""
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("tensorboard is not installed. Install with: pip install tensorboard")
        
        self.writer = None
        self.log_dir = None
    
    def init(
        self,
        log_dir: str = "./tensorboard_logs",
        comment: str = "",
        **kwargs
    ):
        """Initialize TensorBoard writer.
        
        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment suffix for the log directory
            **kwargs: Additional parameters
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory
        run_dir = self.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{comment}"
        
        self.writer = SummaryWriter(str(run_dir))
        logger.info(f"Initialized TensorBoard writer at: {run_dir}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to TensorBoard."""
        if not self.writer:
            logger.warning("TensorBoard not initialized. Skipping metric logging.")
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            elif isinstance(value, dict):
                self.writer.add_scalars(key, value, step)
            elif isinstance(value, np.ndarray):
                if value.ndim == 1:
                    self.writer.add_histogram(key, value, step)
                elif value.ndim == 2:
                    self.writer.add_image(key, value, step)
    
    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """Log artifact reference to TensorBoard."""
        if not self.writer:
            logger.warning("TensorBoard not initialized. Skipping artifact logging.")
            return
        
        # TensorBoard doesn't directly support artifacts, log as text
        self.writer.add_text(
            f"artifact/{artifact_type}",
            f"Saved {artifact_type} to: {file_path}",
            global_step=int(time.time())
        )
    
    def log_graph(self, model, input_sample):
        """Log model graph to TensorBoard."""
        if not self.writer or not TORCH_AVAILABLE:
            return
        
        try:
            self.writer.add_graph(model, input_sample)
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")
    
    def finish(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
            self.writer = None


class MLflowIntegration(BaseIntegration):
    """MLflow integration."""
    
    def __init__(self):
        """Initialize MLflow integration."""
        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow is not installed. Install with: pip install mlflow")
        
        self.run = None
        self.experiment_id = None
    
    def init(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Initialize MLflow run.
        
        Args:
            experiment_name: Experiment name
            run_name: Run name
            tracking_uri: MLflow tracking URI
            tags: Dictionary of tags
            **kwargs: Additional parameters
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        # Start run
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {self.run.info.run_id}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if not self.run:
            logger.warning("MLflow not initialized. Skipping metric logging.")
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        mlflow.log_metric(f"{key}.{subkey}", subvalue, step=step)
    
    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """Log artifact to MLflow."""
        if not self.run:
            logger.warning("MLflow not initialized. Skipping artifact logging.")
            return
        
        mlflow.log_artifact(file_path)
        
        # Log model specifically if it's a PyTorch model
        if artifact_type == "model" and MLFLOW_AVAILABLE and file_path.endswith('.pt'):
            try:
                model = torch.load(file_path, map_location='cpu')
                mlflow.pytorch.log_model(model, "model")
            except Exception as e:
                logger.warning(f"Failed to log model with MLflow PyTorch: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not self.run:
            return
        
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def finish(self):
        """End MLflow run."""
        if self.run:
            mlflow.end_run()
            self.run = None


class CustomLogger:
    """Custom structured logger for fine-tuning pipeline."""
    
    def __init__(
        self,
        log_file: str = "fine_tuning.log",
        log_level: str = "INFO"
    ):
        """Initialize custom logger.
        
        Args:
            log_file: Log file path
            log_level: Logging level
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("fine_tuning")
        self.logger.setLevel(getattr(logging, log_level))
        
        # File handler with JSON formatting
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(getattr(logging, log_level))
        
        # Custom JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "message": "%(message)s"}'
        )
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
    
    def log_event(
        self,
        event_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a structured event.
        
        Args:
            event_type: Type of event
            message: Event message
            metadata: Additional metadata
        """
        log_data = {
            "event_type": event_type,
            "message": message,
            "metadata": metadata or {}
        }
        
        self.logger.info(json.dumps(log_data))
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        self.log_event(
            "metrics",
            f"Step {step}" if step else "Metrics",
            {"metrics": metrics, "step": step}
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with context."""
        self.log_event(
            "error",
            str(error),
            {
                "error_type": type(error).__name__,
                "context": context or {},
                "traceback": str(error.__traceback__)
            }
        )


class ResourceMonitor:
    """Monitors system resources during training."""
    
    def __init__(
        self,
        interval_seconds: int = 10,
        history_size: int = 1000
    ):
        """Initialize resource monitor.
        
        Args:
            interval_seconds: Monitoring interval
            history_size: Size of metrics history
        """
        self.interval = interval_seconds
        self.history_size = history_size
        self.history = deque(maxlen=history_size)
        self.monitoring = False
        self.thread = None
        
        # Network monitoring baseline
        self.net_io = psutil.net_io_counters()
        self.last_net_time = time.time()
    
    def start(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Started resource monitoring")
    
    def stop(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=self.interval + 1)
        logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.history.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            time.sleep(self.interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_metrics = []
        if GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_metrics.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "temperature": gpu.temperature
                    })
            except Exception as e:
                logger.debug(f"Failed to get GPU metrics: {e}")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network I/O
        net_io_new = psutil.net_io_counters()
        current_time = time.time()
        time_delta = current_time - self.last_net_time
        
        bytes_sent_delta = net_io_new.bytes_sent - self.net_io.bytes_sent
        bytes_recv_delta = net_io_new.bytes_recv - self.net_io.bytes_recv
        
        mb_sent = (bytes_sent_delta / (1024 * 1024)) / time_delta
        mb_recv = (bytes_recv_delta / (1024 * 1024)) / time_delta
        
        self.net_io = net_io_new
        self.last_net_time = current_time
        
        return ResourceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            gpu_metrics=gpu_metrics,
            disk_usage_percent=disk.percent,
            network_mb_sent=mb_sent,
            network_mb_recv=mb_recv
        )
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent metrics."""
        if self.history:
            return self.history[-1]
        return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics."""
        if not self.history:
            return {}
        
        # Convert to arrays for statistics
        cpu_values = [m.cpu_percent for m in self.history]
        memory_values = [m.memory_percent for m in self.history]
        
        summary = {
            "cpu": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
                "std": np.std(cpu_values)
            },
            "memory": {
                "mean": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values),
                "std": np.std(memory_values)
            }
        }
        
        # GPU summary if available
        if any(m.gpu_metrics for m in self.history):
            gpu_summaries = defaultdict(lambda: {"load": [], "memory": [], "temp": []})
            
            for metrics in self.history:
                for gpu in metrics.gpu_metrics:
                    gpu_summaries[gpu["id"]]["load"].append(gpu["load"])
                    gpu_summaries[gpu["id"]]["memory"].append(gpu["memory_percent"])
                    gpu_summaries[gpu["id"]]["temp"].append(gpu["temperature"])
            
            summary["gpus"] = {}
            for gpu_id, values in gpu_summaries.items():
                summary["gpus"][f"gpu_{gpu_id}"] = {
                    "load_mean": np.mean(values["load"]),
                    "load_max": np.max(values["load"]),
                    "memory_mean": np.mean(values["memory"]),
                    "memory_max": np.max(values["memory"]),
                    "temp_mean": np.mean(values["temp"]),
                    "temp_max": np.max(values["temp"])
                }
        
        return summary


class AlertManager:
    """Manages training alerts and anomaly detection."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts = []
        self.alert_history = []
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=100))
    
    def add_alert(
        self,
        metric: str,
        threshold: float,
        condition: str = "greater_than",
        message: str = "",
        callback: Optional[Callable] = None,
        cooldown_seconds: int = 300
    ):
        """Add an alert rule.
        
        Args:
            metric: Metric name to monitor
            threshold: Threshold value
            condition: Alert condition (greater_than, less_than, equals)
            message: Alert message
            callback: Optional callback function
            cooldown_seconds: Cooldown period between alerts
        """
        alert = Alert(
            metric=metric,
            threshold=threshold,
            condition=condition,
            message=message or f"Alert: {metric} {condition} {threshold}",
            callback=callback,
            cooldown_seconds=cooldown_seconds
        )
        
        self.alerts.append(alert)
        logger.info(f"Added alert: {alert.message}")
    
    def check_metrics(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules.
        
        Args:
            metrics: Current metrics dictionary
        """
        current_time = time.time()
        
        # Update metrics buffer
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_buffer[key].append(value)
        
        # Check alerts
        for alert in self.alerts:
            if alert.metric not in metrics:
                continue
            
            value = metrics[alert.metric]
            if not isinstance(value, (int, float)):
                continue
            
            # Check if alert should trigger
            triggered = False
            if alert.condition == "greater_than" and value > alert.threshold:
                triggered = True
            elif alert.condition == "less_than" and value < alert.threshold:
                triggered = True
            elif alert.condition == "equals" and value == alert.threshold:
                triggered = True
            
            # Check cooldown
            if triggered and alert.last_triggered:
                if current_time - alert.last_triggered < alert.cooldown_seconds:
                    triggered = False
            
            if triggered:
                self._trigger_alert(alert, value, current_time)
    
    def _trigger_alert(self, alert: Alert, value: float, timestamp: float):
        """Trigger an alert.
        
        Args:
            alert: Alert configuration
            value: Current metric value
            timestamp: Current timestamp
        """
        alert.last_triggered = timestamp
        
        # Log alert
        alert_info = {
            "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "metric": alert.metric,
            "value": value,
            "threshold": alert.threshold,
            "condition": alert.condition,
            "message": alert.message
        }
        
        self.alert_history.append(alert_info)
        logger.warning(f"ALERT: {alert.message} (value: {value})")
        
        # Call callback if provided
        if alert.callback:
            try:
                alert.callback(alert_info)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_anomaly_detection(
        self,
        metric: str,
        window_size: int = 20,
        std_threshold: float = 3.0
    ):
        """Add anomaly detection for a metric.
        
        Args:
            metric: Metric to monitor
            window_size: Window size for statistics
            std_threshold: Standard deviation threshold
        """
        def anomaly_callback(alert_info):
            """Callback for anomaly detection."""
            values = list(self.metrics_buffer[metric])
            if len(values) >= window_size:
                recent_values = values[-window_size:]
                mean = np.mean(recent_values)
                std = np.std(recent_values)
                
                alert_info["statistics"] = {
                    "mean": mean,
                    "std": std,
                    "deviation": (alert_info["value"] - mean) / std if std > 0 else 0
                }
        
        # Add upper bound alert
        self.add_alert(
            metric=metric,
            threshold=float('inf'),  # Will be updated dynamically
            condition="greater_than",
            message=f"Anomaly detected in {metric}",
            callback=anomaly_callback
        )
    
    def get_alert_history(self) -> List[Dict[str, Any]]:
        """Get alert history."""
        return self.alert_history.copy()


class MonitoringIntegration:
    """Main monitoring integration class that combines all platforms."""
    
    def __init__(
        self,
        platforms: List[str] = ["tensorboard"],
        project_name: str = "fine_tuning",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize monitoring integration.
        
        Args:
            platforms: List of platforms to use
            project_name: Project name
            config: Configuration dictionary
        """
        self.platforms = platforms
        self.project_name = project_name
        self.config = config or {}
        
        # Initialize integrations
        self.integrations = {}
        self._init_integrations()
        
        # Initialize components
        self.custom_logger = CustomLogger(
            log_file=self.config.get("log_file", "fine_tuning.log")
        )
        self.resource_monitor = ResourceMonitor(
            interval_seconds=self.config.get("resource_interval", 10)
        )
        self.alert_manager = AlertManager()
        
        # Start resource monitoring
        self.resource_monitor.start()
    
    def _init_integrations(self):
        """Initialize selected platform integrations."""
        for platform in self.platforms:
            try:
                if platform == "wandb" and WANDB_AVAILABLE:
                    integration = WandbIntegration()
                    integration.init(
                        project=self.project_name,
                        config=self.config,
                        **self.config.get("wandb", {})
                    )
                    self.integrations["wandb"] = integration
                    
                elif platform == "tensorboard" and TENSORBOARD_AVAILABLE:
                    integration = TensorBoardIntegration()
                    integration.init(
                        log_dir=self.config.get("tensorboard_dir", "./tensorboard_logs"),
                        **self.config.get("tensorboard", {})
                    )
                    self.integrations["tensorboard"] = integration
                    
                elif platform == "mlflow" and MLFLOW_AVAILABLE:
                    integration = MLflowIntegration()
                    integration.init(
                        experiment_name=self.project_name,
                        **self.config.get("mlflow", {})
                    )
                    self.integrations["mlflow"] = integration
                    
                else:
                    logger.warning(f"Platform {platform} not available or not supported")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {platform}: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        log_resources: bool = True
    ):
        """Log metrics to all platforms.
        
        Args:
            metrics: Metrics dictionary
            step: Training step
            log_resources: Whether to include resource metrics
        """
        # Add resource metrics if requested
        if log_resources:
            resource_metrics = self.resource_monitor.get_current_metrics()
            if resource_metrics:
                metrics["system/cpu_percent"] = resource_metrics.cpu_percent
                metrics["system/memory_percent"] = resource_metrics.memory_percent
                
                for gpu in resource_metrics.gpu_metrics:
                    gpu_prefix = f"system/gpu_{gpu['id']}"
                    metrics[f"{gpu_prefix}/load"] = gpu["load"]
                    metrics[f"{gpu_prefix}/memory_percent"] = gpu["memory_percent"]
                    metrics[f"{gpu_prefix}/temperature"] = gpu["temperature"]
        
        # Log to all platforms
        for name, integration in self.integrations.items():
            try:
                integration.log_metrics(metrics, step)
            except Exception as e:
                logger.error(f"Failed to log metrics to {name}: {e}")
        
        # Log to custom logger
        self.custom_logger.log_metrics(metrics, step)
        
        # Check alerts
        self.alert_manager.check_metrics(metrics)
    
    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """Log artifact to all platforms.
        
        Args:
            file_path: Path to artifact file
            artifact_type: Type of artifact
        """
        for name, integration in self.integrations.items():
            try:
                integration.log_artifact(file_path, artifact_type)
            except Exception as e:
                logger.error(f"Failed to log artifact to {name}: {e}")
        
        # Log event
        self.custom_logger.log_event(
            "artifact_saved",
            f"Saved {artifact_type}",
            {"path": file_path, "type": artifact_type}
        )
    
    def log_model(
        self,
        model_path: str,
        model_name: str,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Log model with metrics.
        
        Args:
            model_path: Path to model file
            model_name: Model name
            metrics: Optional metrics to log with model
        """
        # Log model as artifact
        self.log_artifact(model_path, "model")
        
        # Log associated metrics
        if metrics:
            model_metrics = {f"model/{k}": v for k, v in metrics.items()}
            self.log_metrics(model_metrics)
        
        # Log event
        self.custom_logger.log_event(
            "model_saved",
            f"Saved model: {model_name}",
            {"path": model_path, "metrics": metrics}
        )
    
    def add_alert(self, **kwargs):
        """Add an alert rule."""
        self.alert_manager.add_alert(**kwargs)
    
    def add_anomaly_detection(self, metric: str, **kwargs):
        """Add anomaly detection for a metric."""
        self.alert_manager.add_anomaly_detection(metric, **kwargs)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        return self.resource_monitor.get_metrics_summary()
    
    def get_alert_history(self) -> List[Dict[str, Any]]:
        """Get alert history."""
        return self.alert_manager.get_alert_history()
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters.
        
        Args:
            params: Hyperparameters dictionary
        """
        # Log to MLflow if available
        if "mlflow" in self.integrations:
            self.integrations["mlflow"].log_params(params)
        
        # Log to W&B if available (already logged in init)
        if "wandb" in self.integrations:
            wandb.config.update(params)
        
        # Log event
        self.custom_logger.log_event(
            "hyperparameters",
            "Logged hyperparameters",
            params
        )
    
    def log_training_start(self, config: Dict[str, Any]):
        """Log training start event.
        
        Args:
            config: Training configuration
        """
        self.custom_logger.log_event(
            "training_start",
            "Training started",
            config
        )
        
        # Log initial configuration
        self.log_hyperparameters(config)
    
    def log_training_end(self, final_metrics: Dict[str, Any]):
        """Log training end event.
        
        Args:
            final_metrics: Final training metrics
        """
        # Get resource summary
        resource_summary = self.get_resource_summary()
        
        # Log final metrics
        self.log_metrics(final_metrics, log_resources=False)
        
        # Log event
        self.custom_logger.log_event(
            "training_end",
            "Training completed",
            {
                "final_metrics": final_metrics,
                "resource_summary": resource_summary,
                "alerts": len(self.get_alert_history())
            }
        )
    
    def finish(self):
        """Cleanup and finish all integrations."""
        # Stop resource monitoring
        self.resource_monitor.stop()
        
        # Finish all integrations
        for name, integration in self.integrations.items():
            try:
                integration.finish()
            except Exception as e:
                logger.error(f"Failed to finish {name}: {e}")
        
        # Final log
        self.custom_logger.log_event(
            "monitoring_finished",
            "Monitoring integration closed",
            {"platforms": list(self.integrations.keys())}
        )


# Example usage
if __name__ == "__main__":
    # Initialize monitoring
    monitor = MonitoringIntegration(
        platforms=["tensorboard", "wandb"],
        project_name="test_fine_tuning",
        config={
            "log_file": "test_training.log",
            "tensorboard_dir": "./test_tb_logs",
            "wandb": {
                "tags": ["test", "example"]
            }
        }
    )
    
    # Add alerts
    monitor.add_alert(
        metric="loss",
        threshold=2.0,
        condition="greater_than",
        message="Loss is too high!"
    )
    
    monitor.add_anomaly_detection("loss")
    
    # Simulate training
    import random
    
    for step in range(100):
        # Simulate metrics
        metrics = {
            "loss": 2.5 - (step * 0.02) + random.random() * 0.1,
            "accuracy": min(0.1 + (step * 0.008) + random.random() * 0.05, 1.0),
            "learning_rate": 1e-4 * (0.95 ** (step // 20))
        }
        
        # Log metrics
        monitor.log_metrics(metrics, step=step)
        
        # Log model periodically
        if step % 20 == 0:
            monitor.log_model(
                f"./checkpoints/model_step_{step}.pt",
                f"model_checkpoint_{step}",
                metrics
            )
        
        time.sleep(0.1)
    
    # Finish monitoring
    print("\nResource Summary:")
    print(json.dumps(monitor.get_resource_summary(), indent=2))
    
    print("\nAlert History:")
    for alert in monitor.get_alert_history():
        print(f"- {alert['timestamp']}: {alert['message']}")
    
    monitor.finish()