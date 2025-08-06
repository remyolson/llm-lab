"""Gradient-based analysis for model interpretability."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hook_manager import HookManager

logger = logging.getLogger(__name__)


@dataclass
class GradientInfo:
    """Container for gradient analysis results."""

    layer_name: str
    gradients: torch.Tensor
    gradient_norm: float
    saliency_map: Optional[torch.Tensor] = None
    integrated_gradients: Optional[torch.Tensor] = None
    gradient_x_input: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GradientAnalyzer:
    """Analyzes gradients for model interpretability."""

    def __init__(self, model: nn.Module):
        """
        Initialize the gradient analyzer.

        Args:
            model: The model to analyze
        """
        self.model = model
        self.hook_manager = HookManager(model)
        self.gradient_info = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register backward hooks for gradient collection."""
        # Register hooks for all linear and conv layers
        target_types = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]
        self.hook_manager.register_hooks_by_type(target_types, hook_type="backward")

    def compute_gradients(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target_class: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, GradientInfo]:
        """
        Compute gradients with respect to inputs or a target class.

        Args:
            inputs: Model inputs (tensor or dict of tensors)
            target_class: Optional target class for gradient computation
            loss_fn: Optional custom loss function

        Returns:
            Dictionary mapping layer names to gradient information
        """
        self.hook_manager.clear_outputs()
        self.gradient_info = {}

        # Enable gradient computation
        if isinstance(inputs, dict):
            for key, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    tensor.requires_grad_(True)
        else:
            inputs.requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs) if not isinstance(inputs, dict) else self.model(**inputs)

        # Compute loss
        if loss_fn is not None:
            loss = loss_fn(outputs)
        elif target_class is not None:
            if len(outputs.shape) == 2:
                loss = outputs[:, target_class].sum()
            else:
                loss = outputs[target_class].sum()
        else:
            loss = outputs.sum()

        # Backward pass
        loss.backward()

        # Extract gradients
        gradients = self.hook_manager.extract_gradients()

        # Process gradients
        for layer_name, grad_tensor in gradients.items():
            grad_info = self._process_gradients(layer_name, grad_tensor)
            self.gradient_info[layer_name] = grad_info

        return self.gradient_info

    def _process_gradients(self, layer_name: str, gradients: torch.Tensor) -> GradientInfo:
        """
        Process raw gradients into gradient information.

        Args:
            layer_name: Name of the layer
            gradients: Raw gradient tensor

        Returns:
            GradientInfo object
        """
        # Calculate gradient norm
        gradient_norm = torch.norm(gradients, p=2).item()

        # Create metadata
        metadata = {
            "shape": list(gradients.shape),
            "mean": float(gradients.mean()),
            "std": float(gradients.std()),
            "max": float(gradients.max()),
            "min": float(gradients.min()),
            "sparsity": float((gradients == 0).sum() / gradients.numel()),
        }

        return GradientInfo(
            layer_name=layer_name,
            gradients=gradients,
            gradient_norm=gradient_norm,
            metadata=metadata,
        )

    def compute_saliency_map(
        self, inputs: torch.Tensor, target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute saliency map using vanilla gradients.

        Args:
            inputs: Input tensor
            target_class: Optional target class

        Returns:
            Saliency map tensor
        """
        inputs.requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs)

        # Select target
        if target_class is not None:
            if len(outputs.shape) == 2:
                target = outputs[:, target_class]
            else:
                target = outputs[target_class]
        else:
            target = outputs.max(dim=-1)[0] if len(outputs.shape) > 1 else outputs

        # Backward pass
        self.model.zero_grad()
        target.sum().backward()

        # Get gradients
        saliency = inputs.grad.data.abs()

        # Take max across color channels if needed
        if len(saliency.shape) == 4 and saliency.shape[1] > 1:
            saliency = saliency.max(dim=1, keepdim=True)[0]

        return saliency

    def compute_integrated_gradients(
        self,
        inputs: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        steps: int = 50,
    ) -> torch.Tensor:
        """
        Compute integrated gradients for attribution.

        Args:
            inputs: Input tensor
            baseline: Baseline tensor (zeros if None)
            target_class: Optional target class
            steps: Number of integration steps

        Returns:
            Integrated gradients tensor
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(inputs.device)

        # Accumulate gradients
        integrated_grads = torch.zeros_like(inputs)

        for alpha in alphas:
            # Interpolate
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)

            # Forward pass
            outputs = self.model(interpolated)

            # Select target
            if target_class is not None:
                if len(outputs.shape) == 2:
                    target = outputs[:, target_class]
                else:
                    target = outputs[target_class]
            else:
                target = outputs.max(dim=-1)[0] if len(outputs.shape) > 1 else outputs

            # Backward pass
            self.model.zero_grad()
            target.sum().backward()

            # Accumulate gradients
            integrated_grads += interpolated.grad.data / steps

        # Multiply by input difference
        integrated_grads *= inputs - baseline

        return integrated_grads

    def compute_gradient_x_input(
        self, inputs: torch.Tensor, target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute gradient × input for attribution.

        Args:
            inputs: Input tensor
            target_class: Optional target class

        Returns:
            Gradient × input tensor
        """
        inputs.requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs)

        # Select target
        if target_class is not None:
            if len(outputs.shape) == 2:
                target = outputs[:, target_class]
            else:
                target = outputs[target_class]
        else:
            target = outputs.max(dim=-1)[0] if len(outputs.shape) > 1 else outputs

        # Backward pass
        self.model.zero_grad()
        target.sum().backward()

        # Compute gradient × input
        grad_x_input = inputs.grad.data * inputs.data

        return grad_x_input

    def compute_smoothgrad(
        self,
        inputs: torch.Tensor,
        target_class: Optional[int] = None,
        noise_level: float = 0.1,
        n_samples: int = 50,
    ) -> torch.Tensor:
        """
        Compute SmoothGrad for noise-robust saliency.

        Args:
            inputs: Input tensor
            target_class: Optional target class
            noise_level: Standard deviation of noise
            n_samples: Number of samples

        Returns:
            SmoothGrad saliency map
        """
        smoothgrad = torch.zeros_like(inputs)

        for _ in range(n_samples):
            # Add noise
            noise = torch.randn_like(inputs) * noise_level
            noisy_inputs = inputs + noise

            # Compute saliency
            saliency = self.compute_saliency_map(noisy_inputs, target_class)
            smoothgrad += saliency

        # Average
        smoothgrad /= n_samples

        return smoothgrad

    def get_layer_importance_ranking(self) -> List[Tuple[str, float]]:
        """
        Rank layers by gradient magnitude.

        Returns:
            List of (layer_name, importance_score) tuples
        """
        rankings = []

        for layer_name, grad_info in self.gradient_info.items():
            rankings.append((layer_name, grad_info.gradient_norm))

        # Sort by importance (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def identify_dead_neurons(self, threshold: float = 1e-8) -> Dict[str, List[int]]:
        """
        Identify neurons with near-zero gradients.

        Args:
            threshold: Gradient magnitude threshold

        Returns:
            Dictionary mapping layer names to dead neuron indices
        """
        dead_neurons = {}

        for layer_name, grad_info in self.gradient_info.items():
            gradients = grad_info.gradients

            # Check for linear layers
            if len(gradients.shape) == 2:
                # Check output neurons
                neuron_grads = gradients.abs().mean(dim=0)
                dead_indices = torch.where(neuron_grads < threshold)[0].tolist()
                if dead_indices:
                    dead_neurons[layer_name] = dead_indices

        return dead_neurons

    def compute_gradient_flow_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics about gradient flow through the network.

        Returns:
            Dictionary of gradient flow metrics per layer
        """
        metrics = {}

        for layer_name, grad_info in self.gradient_info.items():
            layer_metrics = {
                "norm": grad_info.gradient_norm,
                "mean": grad_info.metadata["mean"],
                "std": grad_info.metadata["std"],
                "max": grad_info.metadata["max"],
                "min": grad_info.metadata["min"],
                "sparsity": grad_info.metadata["sparsity"],
            }

            # Add vanishing/exploding gradient indicators
            if grad_info.gradient_norm < 1e-5:
                layer_metrics["status"] = "vanishing"
            elif grad_info.gradient_norm > 1e5:
                layer_metrics["status"] = "exploding"
            else:
                layer_metrics["status"] = "normal"

            metrics[layer_name] = layer_metrics

        return metrics

    def export_gradients(self, output_path: str) -> None:
        """
        Export gradient information to file.

        Args:
            output_path: Path to save gradients
        """
        export_data = {
            "gradient_info": {},
            "layer_ranking": self.get_layer_importance_ranking(),
            "flow_metrics": self.compute_gradient_flow_metrics(),
        }

        for layer_name, grad_info in self.gradient_info.items():
            export_data["gradient_info"][layer_name] = {
                "gradient_norm": grad_info.gradient_norm,
                "metadata": grad_info.metadata,
                "shape": list(grad_info.gradients.shape),
            }

        torch.save(export_data, output_path)
        logger.info(f"Exported gradient information to {output_path}")

    def clear(self):
        """Clear all stored gradient information."""
        self.gradient_info = {}
        self.hook_manager.clear_outputs()

    def __del__(self):
        """Clean up hooks on deletion."""
        if hasattr(self, "hook_manager"):
            self.hook_manager.clear_hooks()
