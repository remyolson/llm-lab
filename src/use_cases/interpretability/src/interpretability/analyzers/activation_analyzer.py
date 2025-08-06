"""Activation pattern analysis for neural networks."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hook_manager import HookManager

logger = logging.getLogger(__name__)


@dataclass
class ActivationStats:
    """Statistics about layer activations."""

    layer_name: str
    activations: torch.Tensor
    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float
    dead_neurons: List[int] = field(default_factory=list)
    histogram: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActivationAnalyzer:
    """Analyzes activation patterns in neural networks."""

    def __init__(self, model: nn.Module):
        """
        Initialize the activation analyzer.

        Args:
            model: The model to analyze
        """
        self.model = model
        self.hook_manager = HookManager(model)
        self.activation_stats = {}
        self.activation_history = defaultdict(list)
        self._setup_hooks()

    def _setup_hooks(self):
        """Set up forward hooks for activation collection."""
        # Register hooks for common layer types
        target_types = [
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ReLU,
            nn.GELU,
            nn.SiLU,
            nn.Tanh,
            nn.Sigmoid,
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
        ]
        self.hook_manager.register_hooks_by_type(target_types, hook_type="forward")

    def analyze_activations(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], track_history: bool = False
    ) -> Dict[str, ActivationStats]:
        """
        Analyze activation patterns for given inputs.

        Args:
            inputs: Model inputs
            track_history: Whether to track activation history

        Returns:
            Dictionary mapping layer names to activation statistics
        """
        self.hook_manager.clear_outputs()

        # Forward pass
        with torch.no_grad():
            if isinstance(inputs, dict):
                outputs = self.model(**inputs)
            else:
                outputs = self.model(inputs)

        # Extract activations
        activations = self.hook_manager.extract_activations()

        # Analyze each layer
        self.activation_stats = {}
        for layer_name, activation_tensor in activations.items():
            stats = self._compute_activation_stats(layer_name, activation_tensor)
            self.activation_stats[layer_name] = stats

            if track_history:
                self.activation_history[layer_name].append(stats)

        return self.activation_stats

    def _compute_activation_stats(
        self, layer_name: str, activations: torch.Tensor
    ) -> ActivationStats:
        """
        Compute statistics for activation tensor.

        Args:
            layer_name: Name of the layer
            activations: Activation tensor

        Returns:
            ActivationStats object
        """
        # Flatten for statistics
        flat_activations = activations.flatten()

        # Compute basic statistics
        mean = float(flat_activations.mean())
        std = float(flat_activations.std())
        min_val = float(flat_activations.min())
        max_val = float(flat_activations.max())

        # Compute sparsity (percentage of zeros)
        sparsity = float((flat_activations == 0).sum() / flat_activations.numel())

        # Identify dead neurons (for linear layers)
        dead_neurons = []
        if len(activations.shape) == 2:  # Batch x Features
            neuron_means = activations.abs().mean(dim=0)
            dead_neurons = torch.where(neuron_means < 1e-8)[0].tolist()

        # Compute histogram
        histogram = None
        if flat_activations.numel() > 0:
            hist_data = flat_activations.cpu().numpy()
            histogram, _ = np.histogram(hist_data, bins=50)

        # Create metadata
        metadata = {
            "shape": list(activations.shape),
            "numel": activations.numel(),
            "dtype": str(activations.dtype),
            "device": str(activations.device),
            "requires_grad": activations.requires_grad,
        }

        return ActivationStats(
            layer_name=layer_name,
            activations=activations,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            sparsity=sparsity,
            dead_neurons=dead_neurons,
            histogram=histogram,
            metadata=metadata,
        )

    def find_maximally_activating_neurons(
        self, layer_name: str, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find neurons with highest activation values.

        Args:
            layer_name: Name of the layer
            top_k: Number of top neurons to return

        Returns:
            List of (neuron_index, activation_value) tuples
        """
        if layer_name not in self.activation_stats:
            return []

        activations = self.activation_stats[layer_name].activations

        # Handle different activation shapes
        if len(activations.shape) == 2:  # Batch x Features
            neuron_maxs = activations.max(dim=0)[0]
        elif len(activations.shape) == 4:  # Batch x Channels x H x W
            neuron_maxs = activations.max(dim=(0, 2, 3))[0]
        else:
            return []

        # Get top-k
        values, indices = torch.topk(neuron_maxs, min(top_k, len(neuron_maxs)))

        return [(int(idx), float(val)) for idx, val in zip(indices, values)]

    def compute_activation_correlation(self, layer1: str, layer2: str) -> float:
        """
        Compute correlation between activations of two layers.

        Args:
            layer1: First layer name
            layer2: Second layer name

        Returns:
            Correlation coefficient
        """
        if layer1 not in self.activation_stats or layer2 not in self.activation_stats:
            return 0.0

        act1 = self.activation_stats[layer1].activations.flatten()
        act2 = self.activation_stats[layer2].activations.flatten()

        # Ensure same size
        min_size = min(len(act1), len(act2))
        act1 = act1[:min_size]
        act2 = act2[:min_size]

        # Compute correlation
        if len(act1) > 0:
            correlation = torch.corrcoef(torch.stack([act1, act2]))[0, 1]
            return float(correlation)

        return 0.0

    def detect_saturation(self, threshold: float = 0.99) -> Dict[str, Dict[str, float]]:
        """
        Detect activation saturation in layers.

        Args:
            threshold: Saturation threshold

        Returns:
            Dictionary of saturation metrics per layer
        """
        saturation_info = {}

        for layer_name, stats in self.activation_stats.items():
            activations = stats.activations

            # Check for common activation functions
            if "relu" in layer_name.lower():
                # ReLU saturation (all zeros)
                saturation_ratio = stats.sparsity
                saturation_type = "zero_saturation"
            elif "sigmoid" in layer_name.lower():
                # Sigmoid saturation (near 0 or 1)
                near_zero = (activations < 0.01).float().mean()
                near_one = (activations > 0.99).float().mean()
                saturation_ratio = float(near_zero + near_one)
                saturation_type = "sigmoid_saturation"
            elif "tanh" in layer_name.lower():
                # Tanh saturation (near -1 or 1)
                near_neg_one = (activations < -0.99).float().mean()
                near_one = (activations > 0.99).float().mean()
                saturation_ratio = float(near_neg_one + near_one)
                saturation_type = "tanh_saturation"
            else:
                # Generic saturation check
                abs_max = max(abs(stats.min_val), abs(stats.max_val))
                if abs_max > 0:
                    near_extremes = ((activations.abs() / abs_max) > threshold).float().mean()
                    saturation_ratio = float(near_extremes)
                else:
                    saturation_ratio = 0.0
                saturation_type = "generic_saturation"

            if saturation_ratio > 0.1:  # More than 10% saturated
                saturation_info[layer_name] = {
                    "ratio": saturation_ratio,
                    "type": saturation_type,
                    "severity": "high" if saturation_ratio > 0.5 else "medium",
                }

        return saturation_info

    def compute_receptive_field_size(
        self, layer_name: str, input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Estimate receptive field size for a layer.

        Args:
            layer_name: Name of the layer
            input_shape: Shape of the input

        Returns:
            Estimated receptive field size
        """
        if layer_name not in self.activation_stats:
            return (0,)

        # This is a simplified estimation
        # In practice, would need to trace through the network architecture
        activation_shape = self.activation_stats[layer_name].metadata["shape"]

        if len(activation_shape) == 4 and len(input_shape) >= 3:
            # Convolutional layer
            h_ratio = input_shape[-2] / activation_shape[-2] if activation_shape[-2] > 0 else 1
            w_ratio = input_shape[-1] / activation_shape[-1] if activation_shape[-1] > 0 else 1

            # Rough estimation of receptive field
            rf_h = int(h_ratio * 3)  # Assuming 3x3 kernels as baseline
            rf_w = int(w_ratio * 3)

            return (rf_h, rf_w)

        return (1,)

    def get_activation_distribution(
        self, layer_name: str, bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get activation distribution for a layer.

        Args:
            layer_name: Name of the layer
            bins: Number of histogram bins

        Returns:
            Tuple of (histogram_values, bin_edges)
        """
        if layer_name not in self.activation_stats:
            return np.array([]), np.array([])

        activations = self.activation_stats[layer_name].activations.flatten()
        activations_np = activations.cpu().numpy()

        histogram, bin_edges = np.histogram(activations_np, bins=bins)

        return histogram, bin_edges

    def compare_activation_distributions(
        self, inputs_list: List[Union[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare activation distributions across different inputs.

        Args:
            inputs_list: List of different inputs

        Returns:
            Dictionary of distribution metrics per layer
        """
        all_stats = []

        for inputs in inputs_list:
            stats = self.analyze_activations(inputs)
            all_stats.append(stats)

        comparison = {}

        for layer_name in all_stats[0].keys():
            layer_comparison = {
                "mean_variance": np.var([s[layer_name].mean for s in all_stats]),
                "std_variance": np.var([s[layer_name].std for s in all_stats]),
                "sparsity_variance": np.var([s[layer_name].sparsity for s in all_stats]),
            }

            # Compute KL divergence between distributions
            if len(all_stats) == 2:
                dist1 = all_stats[0][layer_name].histogram
                dist2 = all_stats[1][layer_name].histogram

                if dist1 is not None and dist2 is not None:
                    # Normalize to probabilities
                    p = dist1 / (dist1.sum() + 1e-8)
                    q = dist2 / (dist2.sum() + 1e-8)

                    # KL divergence
                    kl_div = np.sum(p * np.log((p + 1e-8) / (q + 1e-8)))
                    layer_comparison["kl_divergence"] = float(kl_div)

            comparison[layer_name] = layer_comparison

        return comparison

    def export_activation_stats(self, output_path: str) -> None:
        """
        Export activation statistics to file.

        Args:
            output_path: Path to save statistics
        """
        export_data = {
            "activation_stats": {},
            "saturation_info": self.detect_saturation(),
        }

        for layer_name, stats in self.activation_stats.items():
            export_data["activation_stats"][layer_name] = {
                "mean": stats.mean,
                "std": stats.std,
                "min": stats.min_val,
                "max": stats.max_val,
                "sparsity": stats.sparsity,
                "dead_neurons": stats.dead_neurons,
                "metadata": stats.metadata,
            }

        torch.save(export_data, output_path)
        logger.info(f"Exported activation statistics to {output_path}")

    def clear(self):
        """Clear all stored activation data."""
        self.activation_stats = {}
        self.activation_history.clear()
        self.hook_manager.clear_outputs()

    def __del__(self):
        """Clean up hooks on deletion."""
        if hasattr(self, "hook_manager"):
            self.hook_manager.clear_hooks()
