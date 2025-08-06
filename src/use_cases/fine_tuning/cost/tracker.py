"""
Cost Tracking System for Fine-Tuning

This module tracks actual resource usage and costs during training jobs.
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import GPUtil
import numpy as np
import psutil

from .estimator import CloudProvider, TrainingCostEstimate

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Resource usage snapshot."""

    timestamp: datetime
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_metrics: List[Dict[str, Any]] = field(default_factory=list)
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_received_mb: float = 0.0

    @property
    def memory_percent(self) -> float:
        """Memory usage percentage."""
        if self.memory_total_gb > 0:
            return (self.memory_used_gb / self.memory_total_gb) * 100
        return 0.0

    @property
    def total_gpu_memory_used_gb(self) -> float:
        """Total GPU memory used across all GPUs."""
        return sum(gpu.get("memory_used_gb", 0) for gpu in self.gpu_metrics)

    @property
    def average_gpu_utilization(self) -> float:
        """Average GPU utilization across all GPUs."""
        if not self.gpu_metrics:
            return 0.0
        return np.mean([gpu.get("utilization_percent", 0) for gpu in self.gpu_metrics])


@dataclass
class TrainingSession:
    """Training session metadata."""

    session_id: str
    job_id: Optional[str]
    model_name: str
    dataset_name: str
    recipe_name: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    provider: CloudProvider = CloudProvider.LOCAL
    instance_type: str = "local"
    region: str = "local"

    # Training parameters
    num_epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-5
    use_lora: bool = False
    use_mixed_precision: bool = False

    # Cost tracking
    estimated_cost: Optional[TrainingCostEstimate] = None
    actual_cost: Optional[float] = None

    @property
    def duration_hours(self) -> float:
        """Training duration in hours."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 3600
        else:
            return (datetime.now() - self.start_time).total_seconds() / 3600

    @property
    def is_active(self) -> bool:
        """Whether the training session is still active."""
        return self.end_time is None


@dataclass
class CostReport:
    """Comprehensive cost report."""

    session: TrainingSession
    resource_usage: List[ResourceUsage]
    cost_breakdown: Dict[str, float]
    total_actual_cost: float
    estimated_vs_actual: Dict[str, float]
    optimization_opportunities: List[Dict[str, Any]]

    @property
    def average_cpu_usage(self) -> float:
        """Average CPU usage during training."""
        if not self.resource_usage:
            return 0.0
        return np.mean([usage.cpu_percent for usage in self.resource_usage])

    @property
    def average_memory_usage(self) -> float:
        """Average memory usage during training."""
        if not self.resource_usage:
            return 0.0
        return np.mean([usage.memory_percent for usage in self.resource_usage])

    @property
    def average_gpu_usage(self) -> float:
        """Average GPU usage during training."""
        if not self.resource_usage:
            return 0.0
        return np.mean([usage.average_gpu_utilization for usage in self.resource_usage])

    @property
    def cost_accuracy(self) -> float:
        """Accuracy of cost estimation (0-100%)."""
        if not self.session.estimated_cost:
            return 0.0

        estimated = self.session.estimated_cost.total_cost
        actual = self.total_actual_cost

        if estimated == 0:
            return 0.0

        error = abs(estimated - actual) / estimated
        return max(0, (1 - error) * 100)


class CostTracker:
    """Real-time cost tracking during training."""

    def __init__(
        self,
        session: TrainingSession,
        sampling_interval: int = 30,
        output_dir: Optional[str] = None,
    ):
        """Initialize cost tracker."""
        self.session = session
        self.sampling_interval = sampling_interval
        self.output_dir = Path(output_dir) if output_dir else Path("./cost_tracking")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.resource_usage: List[ResourceUsage] = []
        self.is_tracking = False
        self.tracking_thread: Optional[threading.Thread] = None

        # Cost calculation
        self.hourly_rates = self._get_hourly_rates()
        self.total_cost = 0.0

        # Callbacks
        self.callbacks: List[Callable] = []

    def _get_hourly_rates(self) -> Dict[str, float]:
        """Get hourly rates for cost calculation."""
        # This would typically load from pricing data
        # For now, use simple estimates
        if self.session.provider == CloudProvider.AWS:
            if "p3" in self.session.instance_type:
                return {"compute": 3.06, "storage": 0.001, "network": 0.001}
            elif "g4dn" in self.session.instance_type:
                return {"compute": 0.526, "storage": 0.001, "network": 0.001}
        elif self.session.provider == CloudProvider.LOCAL:
            return {"electricity": 0.12, "depreciation": 0.10}  # per hour

        return {"compute": 1.0, "storage": 0.001, "network": 0.001}

    def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current resource usage."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)

        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024**2) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024**2) if disk_io else 0
        except:
            disk_read_mb = disk_write_mb = 0

        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / (1024**2) if net_io else 0
            net_recv_mb = net_io.bytes_recv / (1024**2) if net_io else 0
        except:
            net_sent_mb = net_recv_mb = 0

        # GPU metrics
        gpu_metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_metrics.append(
                    {
                        "id": i,
                        "name": gpu.name,
                        "utilization_percent": gpu.load * 100,
                        "memory_used_gb": gpu.memoryUsed / 1024,
                        "memory_total_gb": gpu.memoryTotal / 1024,
                        "memory_percent": gpu.memoryUtil * 100,
                        "temperature": gpu.temperature,
                    }
                )
        except Exception as e:
            logger.warning(f"Could not collect GPU metrics: {e}")

        return ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            gpu_metrics=gpu_metrics,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_received_mb=net_recv_mb,
        )

    def _calculate_interval_cost(self, usage: ResourceUsage) -> float:
        """Calculate cost for a sampling interval."""
        interval_hours = self.sampling_interval / 3600  # Convert seconds to hours
        interval_cost = 0.0

        if self.session.provider == CloudProvider.LOCAL:
            # Local cost calculation based on power consumption
            electricity_rate = self.hourly_rates.get("electricity", 0.12)  # per kWh

            # Estimate power consumption
            cpu_watts = 100 * (usage.cpu_percent / 100)  # Scale with usage
            gpu_watts = sum(
                250 * (gpu.get("utilization_percent", 0) / 100) for gpu in usage.gpu_metrics
            )

            total_watts = cpu_watts + gpu_watts
            kwh = (total_watts * interval_hours) / 1000
            electricity_cost = kwh * electricity_rate

            # Add depreciation cost
            depreciation_cost = self.hourly_rates.get("depreciation", 0.10) * interval_hours

            interval_cost = electricity_cost + depreciation_cost

        else:
            # Cloud cost calculation
            compute_rate = self.hourly_rates.get("compute", 1.0)
            interval_cost = compute_rate * interval_hours

        return interval_cost

    def _tracking_loop(self):
        """Main tracking loop."""
        logger.info(f"Started cost tracking for session {self.session.session_id}")

        while self.is_tracking:
            try:
                # Collect resource usage
                usage = self._collect_resource_usage()
                self.resource_usage.append(usage)

                # Calculate interval cost
                interval_cost = self._calculate_interval_cost(usage)
                self.total_cost += interval_cost

                # Save data periodically
                if len(self.resource_usage) % 10 == 0:  # Every 10 samples
                    self._save_data()

                # Execute callbacks
                for callback in self.callbacks:
                    try:
                        callback(usage, interval_cost)
                    except Exception as e:
                        logger.error(f"Error in cost tracking callback: {e}")

                # Sleep until next sample
                time.sleep(self.sampling_interval)

            except Exception as e:
                logger.error(f"Error in cost tracking loop: {e}")
                time.sleep(self.sampling_interval)

        logger.info(f"Stopped cost tracking for session {self.session.session_id}")

    def start_tracking(self):
        """Start cost tracking."""
        if self.is_tracking:
            logger.warning("Cost tracking already started")
            return

        self.is_tracking = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()

        # Save initial session data
        self._save_data()

    def stop_tracking(self):
        """Stop cost tracking."""
        if not self.is_tracking:
            return

        self.is_tracking = False

        if self.tracking_thread:
            self.tracking_thread.join(timeout=5)

        # Update session end time
        self.session.end_time = datetime.now()
        self.session.actual_cost = self.total_cost

        # Save final data
        self._save_data()

        logger.info(f"Cost tracking completed. Total cost: ${self.total_cost:.4f}")

    def _save_data(self):
        """Save tracking data to disk."""
        try:
            # Save session data
            session_file = self.output_dir / f"session_{self.session.session_id}.json"
            with open(session_file, "w") as f:
                session_data = asdict(self.session)
                # Convert datetime objects to strings
                for key, value in session_data.items():
                    if isinstance(value, datetime):
                        session_data[key] = value.isoformat()
                json.dump(session_data, f, indent=2, default=str)

            # Save resource usage data
            usage_file = self.output_dir / f"usage_{self.session.session_id}.json"
            with open(usage_file, "w") as f:
                usage_data = [asdict(usage) for usage in self.resource_usage]
                # Convert datetime objects to strings
                for usage in usage_data:
                    usage["timestamp"] = usage["timestamp"].isoformat()
                json.dump(usage_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving cost tracking data: {e}")

    def add_callback(self, callback: Callable):
        """Add a callback function for real-time updates."""
        self.callbacks.append(callback)

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current tracking statistics."""
        if not self.resource_usage:
            return {}

        latest_usage = self.resource_usage[-1]

        return {
            "session_id": self.session.session_id,
            "duration_hours": self.session.duration_hours,
            "total_cost": self.total_cost,
            "hourly_cost": self.total_cost / max(self.session.duration_hours, 0.01),
            "cpu_usage": latest_usage.cpu_percent,
            "memory_usage": latest_usage.memory_percent,
            "gpu_usage": latest_usage.average_gpu_utilization,
            "samples_collected": len(self.resource_usage),
        }

    def generate_report(self) -> CostReport:
        """Generate comprehensive cost report."""
        # Calculate cost breakdown
        cost_breakdown = {
            "compute": self.total_cost * 0.9,  # Rough estimate
            "storage": self.total_cost * 0.05,
            "network": self.total_cost * 0.05,
        }

        # Compare with estimates
        estimated_vs_actual = {}
        if self.session.estimated_cost:
            estimated_vs_actual = {
                "estimated_total": self.session.estimated_cost.total_cost,
                "actual_total": self.total_cost,
                "difference": self.total_cost - self.session.estimated_cost.total_cost,
                "accuracy_percent": 100
                - abs(self.total_cost - self.session.estimated_cost.total_cost)
                / max(self.session.estimated_cost.total_cost, 0.01)
                * 100,
            }

        # Generate optimization opportunities
        optimization_opportunities = self._identify_optimizations()

        return CostReport(
            session=self.session,
            resource_usage=self.resource_usage,
            cost_breakdown=cost_breakdown,
            total_actual_cost=self.total_cost,
            estimated_vs_actual=estimated_vs_actual,
            optimization_opportunities=optimization_opportunities,
        )

    def _identify_optimizations(self) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities."""
        if not self.resource_usage:
            return []

        optimizations = []

        # Analyze resource usage patterns
        avg_cpu = np.mean([u.cpu_percent for u in self.resource_usage])
        avg_gpu = np.mean([u.average_gpu_utilization for u in self.resource_usage])
        avg_memory = np.mean([u.memory_percent for u in self.resource_usage])

        # Low GPU utilization
        if avg_gpu < 50:
            optimizations.append(
                {
                    "type": "gpu_underutilization",
                    "title": "Low GPU Utilization Detected",
                    "description": f"Average GPU utilization was {avg_gpu:.1f}%. Consider using smaller/cheaper instances.",
                    "potential_savings": self.total_cost * 0.3,
                    "recommendation": "Switch to smaller GPU instance or increase batch size",
                }
            )

        # Low CPU utilization
        if avg_cpu < 30:
            optimizations.append(
                {
                    "type": "cpu_underutilization",
                    "title": "Low CPU Utilization",
                    "description": f"Average CPU utilization was {avg_cpu:.1f}%.",
                    "potential_savings": self.total_cost * 0.1,
                    "recommendation": "Consider instances with fewer CPU cores",
                }
            )

        # Memory over-provisioning
        if avg_memory < 40:
            optimizations.append(
                {
                    "type": "memory_overprovisioned",
                    "title": "Memory Over-provisioned",
                    "description": f"Average memory usage was {avg_memory:.1f}%.",
                    "potential_savings": self.total_cost * 0.15,
                    "recommendation": "Use instances with less memory",
                }
            )

        # Long training time
        if self.session.duration_hours > 8:
            optimizations.append(
                {
                    "type": "long_training",
                    "title": "Long Training Duration",
                    "description": f"Training took {self.session.duration_hours:.1f} hours.",
                    "potential_savings": 0,
                    "recommendation": "Consider multi-GPU training or larger batch sizes",
                }
            )

        return optimizations


def create_tracking_session(
    model_name: str,
    dataset_name: str,
    recipe_name: Optional[str] = None,
    job_id: Optional[str] = None,
    provider: CloudProvider = CloudProvider.LOCAL,
    instance_type: str = "local",
    **training_params,
) -> TrainingSession:
    """Create a new training session for cost tracking."""

    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return TrainingSession(
        session_id=session_id,
        job_id=job_id,
        model_name=model_name,
        dataset_name=dataset_name,
        recipe_name=recipe_name,
        start_time=datetime.now(),
        provider=provider,
        instance_type=instance_type,
        **training_params,
    )


def load_session_data(
    session_id: str, data_dir: str = "./cost_tracking"
) -> Optional[Dict[str, Any]]:
    """Load session data from disk."""
    data_dir = Path(data_dir)
    session_file = data_dir / f"session_{session_id}.json"
    usage_file = data_dir / f"usage_{session_id}.json"

    if not session_file.exists():
        return None

    try:
        with open(session_file) as f:
            session_data = json.load(f)

        usage_data = []
        if usage_file.exists():
            with open(usage_file) as f:
                usage_data = json.load(f)

        return {"session": session_data, "usage": usage_data}

    except Exception as e:
        logger.error(f"Error loading session data: {e}")
        return None
