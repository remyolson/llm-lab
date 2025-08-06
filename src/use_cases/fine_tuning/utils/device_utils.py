"""Device utility functions for fine-tuning."""

from typing import Dict, List, Optional

import torch


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Get the best available device.

    Args:
        device_preference: Device preference ("auto", "cuda", "mps", "cpu")

    Returns:
        torch.device object
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_preference)


def get_device_properties(device: Optional[torch.device] = None) -> Dict[str, any]:
    """
    Get properties of the specified device.

    Args:
        device: Device to query (None for current device)

    Returns:
        Dictionary with device properties
    """
    if device is None:
        device = get_device()

    properties = {"device_type": device.type, "device_index": device.index}

    if device.type == "cuda":
        if device.index is None:
            device_id = torch.cuda.current_device()
        else:
            device_id = device.index

        props = torch.cuda.get_device_properties(device_id)
        properties.update(
            {
                "name": props.name,
                "total_memory_gb": props.total_memory / 1024**3,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
                "is_integrated": props.is_integrated,
                "is_multi_gpu_board": props.is_multi_gpu_board,
                "max_threads_per_block": props.max_threads_per_block,
                "max_threads_per_multi_processor": props.max_threads_per_multi_processor,
                "memory_clock_rate_khz": props.memory_clock_rate,
                "memory_bus_width_bits": props.memory_bus_width,
                "l2_cache_size_bytes": props.l2_cache_size,
            }
        )
    elif device.type == "mps":
        properties.update({"name": "Apple Silicon GPU", "backend": "Metal Performance Shaders"})
    else:
        properties.update({"name": "CPU", "num_cores": torch.get_num_threads()})

    return properties


def get_available_gpus() -> List[Dict[str, any]]:
    """
    Get information about all available GPUs.

    Returns:
        List of dictionaries with GPU information
    """
    gpus = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            props = get_device_properties(device)
            props["device_id"] = i
            gpus.append(props)

    return gpus


def set_device_memory_fraction(fraction: float, device_id: int = 0):
    """
    Set the fraction of GPU memory to use.

    Args:
        fraction: Fraction of memory to use (0.0 to 1.0)
        device_id: GPU device ID
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device_id)


def enable_tf32():
    """Enable TF32 for faster training on Ampere GPUs."""
    if torch.cuda.is_available():
        # Enable TF32 for matmul operations
        torch.backends.cuda.matmul.allow_tf32 = True
        # Enable TF32 for cuDNN operations
        torch.backends.cudnn.allow_tf32 = True


def optimize_backend_flags():
    """Set backend flags for optimal performance."""
    if torch.cuda.is_available():
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        # Use deterministic algorithms when possible
        torch.backends.cudnn.deterministic = False
        # Enable TF32 on Ampere GPUs
        enable_tf32()
