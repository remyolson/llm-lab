"""
Cost Estimation System for Fine-Tuning

This module provides comprehensive cost estimation for training jobs across
different cloud providers and configurations.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import requests
import yaml
import numpy as np

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


@dataclass
class InstanceType:
    """Instance type configuration."""
    name: str
    provider: CloudProvider
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    gpu_type: str
    gpu_memory_gb: float
    hourly_rate: float
    spot_rate: Optional[float] = None
    availability_zones: List[str] = field(default_factory=list)
    
    @property
    def spot_savings(self) -> float:
        """Calculate potential spot instance savings."""
        if self.spot_rate is None:
            return 0.0
        return ((self.hourly_rate - self.spot_rate) / self.hourly_rate) * 100


@dataclass
class TrainingCostEstimate:
    """Complete cost estimate for a training job."""
    total_cost: float
    hourly_rate: float
    estimated_hours: float
    compute_cost: float
    storage_cost: float
    network_cost: float
    spot_cost: Optional[float] = None
    optimization_savings: float = 0.0
    
    # Breakdown by component
    model_loading_cost: float = 0.0
    data_preprocessing_cost: float = 0.0
    training_cost: float = 0.0
    evaluation_cost: float = 0.0
    
    # Time breakdown
    setup_time_hours: float = 0.0
    training_time_hours: float = 0.0
    evaluation_time_hours: float = 0.0
    
    # Provider and instance info
    provider: CloudProvider = CloudProvider.AWS
    instance_type: str = ""
    region: str = ""
    
    @property
    def cost_per_epoch(self) -> float:
        """Cost per training epoch."""
        if self.training_time_hours > 0:
            return self.training_cost / self.training_time_hours
        return 0.0
    
    @property
    def potential_spot_savings(self) -> float:
        """Potential savings with spot instances."""
        if self.spot_cost is not None:
            return self.total_cost - self.spot_cost
        return 0.0


class CostEstimator:
    """Main cost estimation engine."""
    
    def __init__(self, pricing_data_path: Optional[str] = None):
        """Initialize cost estimator."""
        self.pricing_data = self._load_pricing_data(pricing_data_path)
        self.instance_types = self._load_instance_types()
    
    def _load_pricing_data(self, path: Optional[str]) -> Dict[str, Any]:
        """Load pricing data from file or use defaults."""
        if path and Path(path).exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default pricing data (updated 2024)
        return {
            "aws": {
                "compute": {
                    "p3.2xlarge": {"hourly": 3.06, "spot": 0.92},
                    "p3.8xlarge": {"hourly": 12.24, "spot": 3.67},
                    "p3.16xlarge": {"hourly": 24.48, "spot": 7.34},
                    "p4d.24xlarge": {"hourly": 32.77, "spot": 9.83},
                    "g4dn.xlarge": {"hourly": 0.526, "spot": 0.158},
                    "g4dn.2xlarge": {"hourly": 0.752, "spot": 0.226},
                    "g4dn.4xlarge": {"hourly": 1.204, "spot": 0.361},
                    "g4dn.8xlarge": {"hourly": 2.176, "spot": 0.653},
                    "g4dn.12xlarge": {"hourly": 3.912, "spot": 1.174},
                    "g5.xlarge": {"hourly": 1.006, "spot": 0.302},
                    "g5.2xlarge": {"hourly": 1.212, "spot": 0.364},
                    "g5.4xlarge": {"hourly": 1.624, "spot": 0.487},
                    "g5.8xlarge": {"hourly": 2.448, "spot": 0.734},
                    "g5.12xlarge": {"hourly": 4.32, "spot": 1.296}
                },
                "storage": {"ebs_gp3": 0.08, "ebs_io2": 0.125},  # per GB/month
                "network": {"data_transfer": 0.09}  # per GB
            },
            "gcp": {
                "compute": {
                    "n1-standard-4-t4": {"hourly": 0.35, "preemptible": 0.105},
                    "n1-standard-8-v100": {"hourly": 2.48, "preemptible": 0.744},
                    "a2-highgpu-1g": {"hourly": 2.93, "preemptible": 0.879},
                    "a2-highgpu-2g": {"hourly": 5.85, "preemptible": 1.755},
                    "a2-highgpu-4g": {"hourly": 11.70, "preemptible": 3.51},
                    "a2-highgpu-8g": {"hourly": 23.40, "preemptible": 7.02}
                },
                "storage": {"ssd": 0.17, "standard": 0.04},  # per GB/month
                "network": {"egress": 0.12}  # per GB
            },
            "azure": {
                "compute": {
                    "NC6": {"hourly": 0.90, "spot": 0.27},
                    "NC12": {"hourly": 1.80, "spot": 0.54},
                    "NC24": {"hourly": 3.60, "spot": 1.08},
                    "ND40rs_v2": {"hourly": 22.03, "spot": 6.61},
                    "NC6s_v3": {"hourly": 3.06, "spot": 0.92},
                    "NC12s_v3": {"hourly": 6.12, "spot": 1.84},
                    "NC24s_v3": {"hourly": 12.24, "spot": 3.67}
                },
                "storage": {"premium_ssd": 0.15, "standard": 0.045},  # per GB/month
                "network": {"bandwidth": 0.087}  # per GB
            },
            "local": {
                "electricity": {"rate": 0.12, "gpu_watts": 250, "cpu_watts": 100},  # per kWh, watts
                "hardware_depreciation": {"gpu_daily": 2.0, "cpu_daily": 0.5}  # per day
            }
        }
    
    def _load_instance_types(self) -> Dict[str, InstanceType]:
        """Load available instance types."""
        instances = {}
        
        # AWS instances
        aws_instances = [
            ("p3.2xlarge", 8, 61, 1, "V100", 16, 3.06, 0.92),
            ("p3.8xlarge", 32, 244, 4, "V100", 16, 12.24, 3.67),
            ("p3.16xlarge", 64, 488, 8, "V100", 16, 24.48, 7.34),
            ("g4dn.xlarge", 4, 16, 1, "T4", 16, 0.526, 0.158),
            ("g4dn.2xlarge", 8, 32, 1, "T4", 16, 0.752, 0.226),
            ("g4dn.4xlarge", 16, 64, 1, "T4", 16, 1.204, 0.361),
            ("g5.xlarge", 4, 16, 1, "A10G", 24, 1.006, 0.302),
            ("g5.2xlarge", 8, 32, 1, "A10G", 24, 1.212, 0.364),
        ]
        
        for name, cpu, mem, gpu_count, gpu_type, gpu_mem, hourly, spot in aws_instances:
            instances[f"aws_{name}"] = InstanceType(
                name=name,
                provider=CloudProvider.AWS,
                cpu_cores=cpu,
                memory_gb=mem,
                gpu_count=gpu_count,
                gpu_type=gpu_type,
                gpu_memory_gb=gpu_mem,
                hourly_rate=hourly,
                spot_rate=spot
            )
        
        # GCP instances
        gcp_instances = [
            ("n1-standard-4-t4", 4, 15, 1, "T4", 16, 0.35, 0.105),
            ("n1-standard-8-v100", 8, 30, 1, "V100", 16, 2.48, 0.744),
            ("a2-highgpu-1g", 12, 85, 1, "A100", 40, 2.93, 0.879),
            ("a2-highgpu-2g", 24, 170, 2, "A100", 40, 5.85, 1.755),
            ("a2-highgpu-4g", 48, 340, 4, "A100", 40, 11.70, 3.51),
        ]
        
        for name, cpu, mem, gpu_count, gpu_type, gpu_mem, hourly, preempt in gcp_instances:
            instances[f"gcp_{name}"] = InstanceType(
                name=name,
                provider=CloudProvider.GCP,
                cpu_cores=cpu,
                memory_gb=mem,
                gpu_count=gpu_count,
                gpu_type=gpu_type,
                gpu_memory_gb=gpu_mem,
                hourly_rate=hourly,
                spot_rate=preempt
            )
        
        return instances
    
    def estimate_model_parameters(self, model_name: str) -> int:
        """Estimate model parameters from name."""
        model_name = model_name.lower()
        
        # Extract parameter count from model name
        if '175b' in model_name or 'gpt-3' in model_name:
            return 175_000_000_000
        elif '70b' in model_name:
            return 70_000_000_000
        elif '65b' in model_name:
            return 65_000_000_000
        elif '30b' in model_name:
            return 30_000_000_000
        elif '13b' in model_name:
            return 13_000_000_000
        elif '7b' in model_name:
            return 7_000_000_000
        elif '3b' in model_name:
            return 3_000_000_000
        elif '1b' in model_name:
            return 1_000_000_000
        elif '350m' in model_name:
            return 350_000_000
        elif '125m' in model_name:
            return 125_000_000
        else:
            # Default estimate
            return 7_000_000_000
    
    def estimate_dataset_size(self, dataset_path: str, dataset_format: str = "auto") -> int:
        """Estimate dataset size in number of samples."""
        path = Path(dataset_path)
        
        if not path.exists():
            logger.warning(f"Dataset path {dataset_path} not found, using default estimate")
            return 10000
        
        try:
            if dataset_format == "jsonl" or path.suffix == ".jsonl":
                with open(path, 'r') as f:
                    return sum(1 for _ in f)
            elif dataset_format == "json" or path.suffix == ".json":
                with open(path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    elif isinstance(data, dict) and 'data' in data:
                        return len(data['data'])
            elif dataset_format == "csv" or path.suffix == ".csv":
                import pandas as pd
                df = pd.read_csv(path)
                return len(df)
            else:
                # Estimate based on file size (rough approximation)
                file_size_mb = path.stat().st_size / (1024 * 1024)
                # Assume ~1KB per sample on average
                return int(file_size_mb * 1000)
        except Exception as e:
            logger.error(f"Error estimating dataset size: {e}")
            return 10000
    
    def estimate_training_time(
        self,
        model_params: int,
        dataset_size: int,
        num_epochs: int,
        batch_size: int,
        instance_type: InstanceType,
        use_lora: bool = False,
        gradient_checkpointing: bool = False
    ) -> float:
        """Estimate training time in hours."""
        
        # Base throughput tokens per second per GPU
        if instance_type.gpu_type == "A100":
            base_throughput = 5000
        elif instance_type.gpu_type == "V100":
            base_throughput = 3000
        elif instance_type.gpu_type == "T4":
            base_throughput = 1000
        elif instance_type.gpu_type == "A10G":
            base_throughput = 2000
        else:
            base_throughput = 1000  # Default
        
        # Adjust for model size (larger models are slower per token)
        if model_params > 50_000_000_000:
            throughput_multiplier = 0.3
        elif model_params > 10_000_000_000:
            throughput_multiplier = 0.5
        elif model_params > 1_000_000_000:
            throughput_multiplier = 0.8
        else:
            throughput_multiplier = 1.0
        
        # Adjust for LoRA (faster training)
        if use_lora:
            throughput_multiplier *= 2.0
        
        # Adjust for gradient checkpointing (slower but memory efficient)
        if gradient_checkpointing:
            throughput_multiplier *= 0.7
        
        # Calculate effective throughput
        effective_throughput = base_throughput * throughput_multiplier * instance_type.gpu_count
        
        # Estimate tokens per sample (rough approximation)
        tokens_per_sample = 512  # Default sequence length
        
        # Calculate total training time
        total_tokens = dataset_size * num_epochs * tokens_per_sample
        training_seconds = total_tokens / effective_throughput
        training_hours = training_seconds / 3600
        
        # Add overhead for data loading, checkpointing, etc.
        overhead_multiplier = 1.2
        
        return training_hours * overhead_multiplier
    
    def estimate_costs(
        self,
        provider: CloudProvider,
        instance_type_name: str,
        model_name: str,
        dataset_path: str,
        num_epochs: int,
        batch_size: int,
        use_lora: bool = False,
        use_spot_instances: bool = False,
        storage_gb: int = 100,
        region: str = "us-east-1"
    ) -> TrainingCostEstimate:
        """Estimate complete training costs."""
        
        # Get instance type
        instance_key = f"{provider.value}_{instance_type_name}"
        if instance_key not in self.instance_types:
            raise ValueError(f"Instance type {instance_key} not found")
        
        instance = self.instance_types[instance_key]
        
        # Estimate model parameters and dataset size
        model_params = self.estimate_model_parameters(model_name)
        dataset_size = self.estimate_dataset_size(dataset_path)
        
        # Estimate training time
        training_hours = self.estimate_training_time(
            model_params=model_params,
            dataset_size=dataset_size,
            num_epochs=num_epochs,
            batch_size=batch_size,
            instance_type=instance,
            use_lora=use_lora
        )
        
        # Add setup and evaluation time
        setup_hours = 0.25  # 15 minutes setup
        eval_hours = training_hours * 0.1  # 10% of training time for evaluation
        total_hours = setup_hours + training_hours + eval_hours
        
        # Calculate compute costs
        hourly_rate = instance.spot_rate if use_spot_instances and instance.spot_rate else instance.hourly_rate
        compute_cost = total_hours * hourly_rate
        
        # Calculate storage costs (monthly rate converted to usage period)
        storage_provider_data = self.pricing_data[provider.value]["storage"]
        storage_rate = list(storage_provider_data.values())[0]  # Use first storage type
        storage_days = max(1, total_hours / 24)
        storage_cost = storage_gb * storage_rate * (storage_days / 30)  # Pro-rated monthly cost
        
        # Calculate network costs (minimal for training)
        network_cost = 1.0  # Fixed small cost for model downloads, etc.
        
        # Total cost
        total_cost = compute_cost + storage_cost + network_cost
        
        # Calculate spot instance savings
        spot_cost = None
        if instance.spot_rate and not use_spot_instances:
            spot_compute_cost = total_hours * instance.spot_rate
            spot_cost = spot_compute_cost + storage_cost + network_cost
        
        return TrainingCostEstimate(
            total_cost=total_cost,
            hourly_rate=hourly_rate,
            estimated_hours=total_hours,
            compute_cost=compute_cost,
            storage_cost=storage_cost,
            network_cost=network_cost,
            spot_cost=spot_cost,
            setup_time_hours=setup_hours,
            training_time_hours=training_hours,
            evaluation_time_hours=eval_hours,
            provider=provider,
            instance_type=instance_type_name,
            region=region
        )
    
    def get_cost_optimization_recommendations(
        self,
        estimate: TrainingCostEstimate,
        model_params: int,
        use_lora: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        # Spot instance recommendation
        if estimate.spot_cost and estimate.potential_spot_savings > 5:
            recommendations.append({
                "type": "spot_instances",
                "title": "Use Spot/Preemptible Instances",
                "description": f"Save ${estimate.potential_spot_savings:.2f} ({estimate.potential_spot_savings/estimate.total_cost*100:.1f}%) by using spot instances",
                "savings": estimate.potential_spot_savings,
                "risk": "Medium - training may be interrupted"
            })
        
        # LoRA recommendation
        if not use_lora and model_params > 1_000_000_000:
            lora_savings = estimate.total_cost * 0.4  # Estimate 40% savings
            recommendations.append({
                "type": "lora",
                "title": "Use LoRA Fine-Tuning",
                "description": f"Reduce training time and costs by ~40% with parameter-efficient fine-tuning",
                "savings": lora_savings,
                "risk": "Low - slightly reduced model quality possible"
            })
        
        # Mixed precision recommendation
        if estimate.training_time_hours > 2:
            fp16_savings = estimate.compute_cost * 0.2  # Estimate 20% savings
            recommendations.append({
                "type": "mixed_precision",
                "title": "Enable Mixed Precision (FP16/BF16)",
                "description": f"Speed up training by ~20% with minimal quality impact",
                "savings": fp16_savings,
                "risk": "Very Low - modern best practice"
            })
        
        # Gradient checkpointing recommendation
        if model_params > 7_000_000_000:
            recommendations.append({
                "type": "gradient_checkpointing",
                "title": "Use Gradient Checkpointing",
                "description": "Train larger models on smaller instances (20% slower but 50% cheaper instances)",
                "savings": estimate.compute_cost * 0.3,
                "risk": "Low - 20% increase in training time"
            })
        
        # Multi-GPU recommendation
        if estimate.training_time_hours > 8:
            recommendations.append({
                "type": "multi_gpu",
                "title": "Use Multi-GPU Training",
                "description": "Reduce training time with distributed training",
                "savings": 0,  # No direct cost savings, but time savings
                "risk": "Low - requires distributed training setup"
            })
        
        return recommendations


class CostOptimizer:
    """Cost optimization engine."""
    
    def __init__(self, estimator: CostEstimator):
        """Initialize optimizer."""
        self.estimator = estimator
    
    def find_cheapest_configuration(
        self,
        providers: List[CloudProvider],
        model_name: str,
        dataset_path: str,
        num_epochs: int,
        max_training_hours: Optional[float] = None,
        use_spot_instances: bool = True
    ) -> List[Tuple[TrainingCostEstimate, Dict[str, Any]]]:
        """Find the cheapest training configurations."""
        
        configurations = []
        
        for provider in providers:
            # Get available instance types for provider
            provider_instances = [
                (name.split('_', 1)[1], inst) for name, inst in self.estimator.instance_types.items()
                if inst.provider == provider
            ]
            
            for instance_name, instance_type in provider_instances:
                # Try different configurations
                configs = [
                    {"use_lora": False, "batch_size": 4},
                    {"use_lora": True, "batch_size": 4},
                    {"use_lora": False, "batch_size": 8},
                    {"use_lora": True, "batch_size": 8},
                ]
                
                for config in configs:
                    try:
                        estimate = self.estimator.estimate_costs(
                            provider=provider,
                            instance_type_name=instance_name,
                            model_name=model_name,
                            dataset_path=dataset_path,
                            num_epochs=num_epochs,
                            batch_size=config["batch_size"],
                            use_lora=config["use_lora"],
                            use_spot_instances=use_spot_instances
                        )
                        
                        # Filter by max training hours if specified
                        if max_training_hours and estimate.estimated_hours > max_training_hours:
                            continue
                        
                        configurations.append((estimate, config))
                        
                    except Exception as e:
                        logger.warning(f"Error estimating costs for {provider.value} {instance_name}: {e}")
                        continue
        
        # Sort by total cost
        configurations.sort(key=lambda x: x[0].total_cost)
        
        return configurations[:10]  # Return top 10 cheapest
    
    def optimize_for_budget(
        self,
        budget: float,
        providers: List[CloudProvider],
        model_name: str,
        dataset_path: str,
        num_epochs: int
    ) -> List[Tuple[TrainingCostEstimate, Dict[str, Any]]]:
        """Find configurations within budget."""
        
        all_configs = self.find_cheapest_configuration(
            providers=providers,
            model_name=model_name,
            dataset_path=dataset_path,
            num_epochs=num_epochs
        )
        
        # Filter by budget
        within_budget = [
            (estimate, config) for estimate, config in all_configs
            if estimate.total_cost <= budget
        ]
        
        return within_budget
    
    def optimize_for_time(
        self,
        max_hours: float,
        providers: List[CloudProvider],
        model_name: str,
        dataset_path: str,
        num_epochs: int
    ) -> List[Tuple[TrainingCostEstimate, Dict[str, Any]]]:
        """Find fastest configurations within time limit."""
        
        all_configs = self.find_cheapest_configuration(
            providers=providers,
            model_name=model_name,
            dataset_path=dataset_path,
            num_epochs=num_epochs,
            max_training_hours=max_hours
        )
        
        # Sort by training time
        all_configs.sort(key=lambda x: x[0].estimated_hours)
        
        return all_configs