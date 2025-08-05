"""Checkpoint utility functions for fine-tuning."""

import os
import torch
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model_state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: Dict[str, Any],
    scheduler_state_dict: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a training checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch
        model_state_dict: Model state dictionary
        optimizer_state_dict: Optimizer state dictionary
        scheduler_state_dict: Scheduler state dictionary (optional)
        metrics: Training metrics (optional)
        config: Training configuration (optional)
        additional_info: Additional information to save (optional)
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
    }
    
    if scheduler_state_dict is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state_dict
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    if config is not None:
        checkpoint["config"] = config
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    map_location: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map tensors to
        
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return checkpoint


def save_best_checkpoint(
    checkpoint_dir: str,
    model_state_dict: Dict[str, torch.Tensor],
    metrics: Dict[str, float],
    metric_name: str = "loss",
    mode: str = "min"
) -> bool:
    """
    Save checkpoint if it's the best so far.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        model_state_dict: Model state dictionary
        metrics: Current metrics
        metric_name: Metric to compare
        mode: "min" or "max" for metric comparison
        
    Returns:
        True if checkpoint was saved (new best)
    """
    best_metric_file = os.path.join(checkpoint_dir, "best_metric.json")
    current_metric = metrics.get(metric_name)
    
    if current_metric is None:
        logger.warning(f"Metric {metric_name} not found in metrics")
        return False
    
    # Load previous best metric
    is_best = False
    if os.path.exists(best_metric_file):
        with open(best_metric_file, "r") as f:
            best_info = json.load(f)
            best_metric = best_info["metric_value"]
            
        if mode == "min":
            is_best = current_metric < best_metric
        else:
            is_best = current_metric > best_metric
    else:
        is_best = True
    
    if is_best:
        # Save model
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(model_state_dict, best_model_path)
        
        # Save metric info
        best_info = {
            "metric_name": metric_name,
            "metric_value": current_metric,
            "metrics": metrics
        }
        with open(best_metric_file, "w") as f:
            json.dump(best_info, f, indent=2)
        
        logger.info(f"Saved new best model with {metric_name}={current_metric}")
    
    return is_best


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if no checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_epoch_") and f.endswith(".pt")
    ]
    
    if not checkpoint_files:
        return None
    
    # Sort by epoch number
    checkpoint_files.sort(
        key=lambda x: int(x.replace("checkpoint_epoch_", "").replace(".pt", ""))
    )
    
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    return latest_checkpoint


def clean_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 3,
    keep_best: bool = True
):
    """
    Clean old checkpoints, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best model checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_epoch_") and f.endswith(".pt")
    ]
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # Sort by epoch number
    checkpoint_files.sort(
        key=lambda x: int(x.replace("checkpoint_epoch_", "").replace(".pt", ""))
    )
    
    # Files to remove
    files_to_remove = checkpoint_files[:-keep_last_n]
    
    for file in files_to_remove:
        file_path = os.path.join(checkpoint_dir, file)
        os.remove(file_path)
        logger.info(f"Removed old checkpoint: {file}")
    
    logger.info(f"Kept {keep_last_n} most recent checkpoints")