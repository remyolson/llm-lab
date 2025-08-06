"""
Resource Management for Local Models

This module provides utilities for monitoring and managing system resources
when running local models, including memory usage, GPU utilization, and
intelligent model caching.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
import platform
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """System resource information."""

    total_ram_mb: float
    available_ram_mb: float
    used_ram_mb: float
    ram_percent: float

    total_vram_mb: float = 0.0
    available_vram_mb: float = 0.0
    used_vram_mb: float = 0.0
    vram_percent: float = 0.0

    gpu_type: str = "none"
    gpu_name: str = ""
    cpu_cores: int = 0
    cpu_percent: float = 0.0


@dataclass
class ModelResourceUsage:
    """Resource usage for a specific model."""

    model_id: str
    ram_mb: float
    vram_mb: float
    last_accessed: float
    load_time: float
    inference_count: int = 0
    total_inference_time: float = 0.0


class ResourceManager:
    """
    Manages system resources for local model execution.

    Provides monitoring, memory management, and intelligent caching
    for optimal performance across multiple local models.
    """

    def __init__(self, memory_threshold: float = 0.8, vram_threshold: float = 0.9):
        """
        Initialize resource manager.

        Args:
            memory_threshold: RAM usage threshold (0.0-1.0)
            vram_threshold: VRAM usage threshold (0.0-1.0)
        """
        self.memory_threshold = memory_threshold
        self.vram_threshold = vram_threshold

        # Model tracking
        self._model_usage: Dict[str, ModelResourceUsage] = {}

        # Cache system info
        self._system_info: SystemResources | None = None
        self._last_system_check = 0.0
        self._system_check_interval = 5.0  # 5 seconds

    def get_system_resources(self, force_refresh: bool = False) -> SystemResources:
        """
        Get current system resource information.

        Args:
            force_refresh: Force refresh even if recently cached

        Returns:
            SystemResources object with current resource info
        """
        current_time = time.time()

        if (
            not force_refresh
            and self._system_info
            and current_time - self._last_system_check < self._system_check_interval
        ):
            return self._system_info

        # Get RAM info
        ram_info = self._get_ram_info()

        # Get GPU info
        gpu_info = self._get_gpu_info()

        # Get CPU info
        cpu_info = self._get_cpu_info()

        self._system_info = SystemResources(**ram_info, **gpu_info, **cpu_info)

        self._last_system_check = current_time
        return self._system_info

    def _get_ram_info(self) -> Dict[str, float]:
        """Get RAM usage information."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "total_ram_mb": memory.total / (1024 * 1024),
                "available_ram_mb": memory.available / (1024 * 1024),
                "used_ram_mb": memory.used / (1024 * 1024),
                "ram_percent": memory.percent,
            }

        except ImportError:
            # Fallback estimates
            logger.warning("psutil not available, using RAM estimates")
            return {
                "total_ram_mb": 8 * 1024,  # Assume 8GB
                "available_ram_mb": 4 * 1024,  # Assume 4GB available
                "used_ram_mb": 4 * 1024,
                "ram_percent": 50.0,
            }

    def _get_gpu_info(self) -> Dict[str, any]:
        """Get GPU information and usage."""
        gpu_info = {
            "total_vram_mb": 0.0,
            "available_vram_mb": 0.0,
            "used_vram_mb": 0.0,
            "vram_percent": 0.0,
            "gpu_type": "none",
            "gpu_name": "",
        }

        # Try CUDA first
        try:
            import torch

            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                total_vram = device_props.total_memory / (1024 * 1024)

                # Get current memory usage
                allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                cached = torch.cuda.memory_reserved(0) / (1024 * 1024)
                used_vram = max(allocated, cached)

                gpu_info.update(
                    {
                        "total_vram_mb": total_vram,
                        "available_vram_mb": total_vram - used_vram,
                        "used_vram_mb": used_vram,
                        "vram_percent": (used_vram / total_vram) * 100,
                        "gpu_type": "cuda",
                        "gpu_name": device_props.name,
                    }
                )

                return gpu_info

        except ImportError:
            pass

        # Check for Metal (Apple Silicon)
        if platform.system() == "Darwin" and platform.processor() == "arm":
            gpu_info.update(
                {
                    "gpu_type": "metal",
                    "gpu_name": "Apple Silicon",
                    # Metal memory is shared with system RAM
                    "total_vram_mb": 0.0,  # Will use system RAM
                }
            )

        return gpu_info

    def _get_cpu_info(self) -> Dict[str, any]:
        """Get CPU information."""
        try:
            import psutil

            return {
                "cpu_cores": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
            }
        except ImportError:
            import os

            return {
                "cpu_cores": os.cpu_count() or 4,
                "cpu_percent": 0.0,
            }

    def register_model(self, model_id: str, ram_mb: float, vram_mb: float) -> None:
        """Register a loaded model for resource tracking."""
        self._model_usage[model_id] = ModelResourceUsage(
            model_id=model_id,
            ram_mb=ram_mb,
            vram_mb=vram_mb,
            last_accessed=time.time(),
            load_time=time.time(),
        )

        logger.debug(f"Registered model {model_id} (RAM: {ram_mb:.1f}MB, VRAM: {vram_mb:.1f}MB)")

    def update_model_access(self, model_id: str, inference_time: float = 0.0) -> None:
        """Update model access time and statistics."""
        if model_id in self._model_usage:
            usage = self._model_usage[model_id]
            usage.last_accessed = time.time()
            usage.inference_count += 1
            usage.total_inference_time += inference_time

    def unregister_model(self, model_id: str) -> None:
        """Unregister a model when it's unloaded."""
        if model_id in self._model_usage:
            del self._model_usage[model_id]
            logger.debug(f"Unregistered model {model_id}")

    def get_model_usage(self, model_id: str) -> ModelResourceUsage | None:
        """Get resource usage for a specific model."""
        return self._model_usage.get(model_id)

    def get_total_model_usage(self) -> Tuple[float | float]:
        """Get total RAM and VRAM usage across all models."""
        total_ram = sum(usage.ram_mb for usage in self._model_usage.values())
        total_vram = sum(usage.vram_mb for usage in self._model_usage.values())
        return total_ram, total_vram

    def should_unload_models(self) -> bool:
        """Check if models should be unloaded due to resource pressure."""
        resources = self.get_system_resources()

        # Check RAM pressure
        ram_pressure = resources.ram_percent / 100.0 > self.memory_threshold

        # Check VRAM pressure (if applicable)
        vram_pressure = False
        if resources.total_vram_mb > 0:
            vram_pressure = resources.vram_percent / 100.0 > self.vram_threshold

        return ram_pressure or vram_pressure

    def get_models_to_unload(self, target_mb: float | None = None) -> List[str]:
        """
        Get list of models that should be unloaded to free memory.

        Args:
            target_mb: Target amount of memory to free (optional)

        Returns:
            List of model IDs sorted by unload priority (least important first)
        """
        if not self._model_usage:
            return []

        # Sort models by priority (least recently used + inference frequency)
        def get_unload_priority(usage: ModelResourceUsage) -> float:
            # Lower score = higher unload priority
            recency = time.time() - usage.last_accessed
            frequency = usage.inference_count / max(
                1, (time.time() - usage.load_time) / 3600
            )  # per hour
            return recency - (frequency * 100)  # Bias towards keeping frequently used models

        sorted_models = sorted(
            self._model_usage.values(),
            key=get_unload_priority,
            reverse=True,  # Highest priority (least important) first
        )

        models_to_unload = []
        freed_ram = 0.0
        freed_vram = 0.0

        for usage in sorted_models:
            models_to_unload.append(usage.model_id)
            freed_ram += usage.ram_mb
            freed_vram += usage.vram_mb

            # Check if we've freed enough memory
            if target_mb and (freed_ram + freed_vram) >= target_mb:
                break

        return models_to_unload

    def estimate_model_memory(
        self, model_size_mb: float, model_format: str
    ) -> Tuple[float | float]:
        """
        Estimate RAM and VRAM usage for a model.

        Args:
            model_size_mb: Size of model file in MB
            model_format: Model format (gguf, safetensors, etc.)

        Returns:
            Tuple of (estimated_ram_mb, estimated_vram_mb)
        """
        resources = self.get_system_resources()

        # Base memory usage (model size + overhead)
        overhead_factor = {
            "gguf": 1.2,  # GGUF is efficient
            "ggml": 1.2,
            "safetensors": 1.5,  # Transformers has more overhead
            "pytorch": 1.5,
            "ollama": 1.1,  # Ollama is optimized
        }.get(model_format, 1.4)

        estimated_memory = model_size_mb * overhead_factor

        # Decide RAM vs VRAM split based on available resources
        if resources.gpu_type == "none" or resources.total_vram_mb == 0:
            # CPU only
            return estimated_memory, 0.0

        elif resources.gpu_type == "metal":
            # Apple Silicon - shared memory
            return estimated_memory, 0.0

        else:
            # Dedicated GPU - prefer VRAM if available
            available_vram = resources.available_vram_mb
            if available_vram > estimated_memory:
                return estimated_memory * 0.2, estimated_memory * 0.8  # Small RAM overhead
            else:
                # Split between RAM and VRAM
                vram_portion = min(available_vram * 0.8, estimated_memory * 0.6)
                ram_portion = estimated_memory - vram_portion
                return ram_portion, vram_portion

    def can_load_model(self, model_size_mb: float, model_format: str) -> bool:
        """Check if a model can be loaded given current resource usage."""
        estimated_ram, estimated_vram = self.estimate_model_memory(model_size_mb, model_format)
        resources = self.get_system_resources()

        # Check RAM availability
        ram_needed = estimated_ram
        ram_available = resources.available_ram_mb

        if ram_needed > ram_available:
            logger.warning(f"Insufficient RAM: need {ram_needed:.1f}MB, have {ram_available:.1f}MB")
            return False

        # Check VRAM availability (if needed)
        if estimated_vram > 0:
            vram_needed = estimated_vram
            vram_available = resources.available_vram_mb

            if vram_needed > vram_available:
                logger.warning(
                    f"Insufficient VRAM: need {vram_needed:.1f}MB, have {vram_available:.1f}MB"
                )
                return False

        return True

    def get_resource_summary(self) -> Dict[str, any]:
        """Get a summary of current resource usage."""
        resources = self.get_system_resources()
        model_ram, model_vram = self.get_total_model_usage()

        return {
            "system": {
                "ram_total_mb": resources.total_ram_mb,
                "ram_used_percent": resources.ram_percent,
                "vram_total_mb": resources.total_vram_mb,
                "vram_used_percent": resources.vram_percent,
                "gpu_type": resources.gpu_type,
                "gpu_name": resources.gpu_name,
                "cpu_cores": resources.cpu_cores,
            },
            "models": {
                "count": len(self._model_usage),
                "total_ram_mb": model_ram,
                "total_vram_mb": model_vram,
                "models": {
                    model_id: {
                        "ram_mb": usage.ram_mb,
                        "vram_mb": usage.vram_mb,
                        "inference_count": usage.inference_count,
                        "avg_inference_time": (
                            usage.total_inference_time / usage.inference_count
                            if usage.inference_count > 0
                            else 0.0
                        ),
                    }
                    for model_id, usage in self._model_usage.items()
                },
            },
            "thresholds": {
                "memory_threshold": self.memory_threshold,
                "vram_threshold": self.vram_threshold,
                "should_unload": self.should_unload_models(),
            },
        }
