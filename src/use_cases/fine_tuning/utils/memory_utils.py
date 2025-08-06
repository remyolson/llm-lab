"""Memory utility functions for fine-tuning."""

from typing import Dict, Optional

import psutil
import torch


def get_memory_stats() -> Dict[str | float]:
    """
    Get current memory statistics.

    Returns:
        Dictionary with memory stats in GB
    """
    stats = {}

    # System RAM
    memory = psutil.virtual_memory()
    stats["ram_total_gb"] = memory.total / 1024**3
    stats["ram_used_gb"] = memory.used / 1024**3
    stats["ram_available_gb"] = memory.available / 1024**3
    stats["ram_percent"] = memory.percent

    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            stats[f"gpu_{i}_allocated_gb"] = torch.cuda.memory_allocated(i) / 1024**3
            stats[f"gpu_{i}_reserved_gb"] = torch.cuda.memory_reserved(i) / 1024**3
            stats[f"gpu_{i}_max_allocated_gb"] = torch.cuda.max_memory_allocated(i) / 1024**3

    return stats


def estimate_model_memory(
    model_params: int,
    batch_size: int,
    sequence_length: int,
    precision: str = "fp32",
    include_gradients: bool = True,
    include_optimizer_states: bool = True,
) -> float:
    """
    Estimate memory required for model training.

    Args:
        model_params: Number of model parameters
        batch_size: Training batch size
        sequence_length: Sequence length
        precision: Model precision (fp32, fp16, bf16, int8, int4)
        include_gradients: Include gradient memory
        include_optimizer_states: Include optimizer state memory (for Adam/AdamW)

    Returns:
        Estimated memory in GB
    """
    # Bytes per parameter based on precision
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}

    param_bytes = bytes_per_param.get(precision, 4)

    # Model parameters memory
    model_memory = model_params * param_bytes

    # Gradients (same size as model)
    if include_gradients:
        model_memory += model_params * param_bytes

    # Optimizer states (Adam/AdamW has 2 states per parameter)
    if include_optimizer_states:
        model_memory += model_params * param_bytes * 2

    # Activation memory (rough estimate)
    # Assumes transformer-like architecture
    hidden_size = int((model_params / 12) ** 0.5)  # Rough estimate
    num_layers = 12  # Typical for many models

    activation_memory = batch_size * sequence_length * hidden_size * num_layers * param_bytes * 10

    total_memory_gb = (model_memory + activation_memory) / 1024**3

    return total_memory_gb


def clear_memory_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_available_memory() -> Optional[float]:
    """
    Get available GPU memory.

    Returns:
        Available memory in GB, or None if no GPU
    """
    if not torch.cuda.is_available():
        return None

    # Get total and allocated memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    reserved_memory = torch.cuda.memory_reserved(0)

    # Available is total minus reserved (not just allocated)
    available_memory = total_memory - reserved_memory

    return available_memory / 1024**3
