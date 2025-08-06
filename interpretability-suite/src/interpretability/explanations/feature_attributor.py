"""Feature attribution methods for model interpretability."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Container for feature attribution results."""

    method: str
    attributions: torch.Tensor
    baseline_score: Optional[float] = None
    target_score: Optional[float] = None
    feature_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureAttributor:
    """Computes feature attributions using various methods."""

    def __init__(self, model: nn.Module, feature_names: Optional[List[str]] = None):
        """
        Initialize the feature attributor.

        Args:
            model: The model to analyze
            feature_names: Optional names for features
        """
        self.model = model
        self.feature_names = feature_names
        self.attribution_results = []

    def compute_attributions(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        method: str = "integrated_gradients",
        baseline: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute feature attributions using specified method.

        Args:
            inputs: Input tensor
            target: Target class for attribution
            method: Attribution method to use
            baseline: Baseline for comparison
            **kwargs: Method-specific parameters

        Returns:
            AttributionResult object
        """
        if method == "integrated_gradients":
            attributions = self.integrated_gradients(inputs, target, baseline, **kwargs)
        elif method == "gradient_shap":
            attributions = self.gradient_shap(inputs, target, baseline, **kwargs)
        elif method == "deeplift":
            attributions = self.deeplift(inputs, target, baseline, **kwargs)
        elif method == "occlusion":
            attributions = self.occlusion_sensitivity(inputs, target, **kwargs)
        elif method == "lime":
            attributions = self.lime_attribution(inputs, target, **kwargs)
        elif method == "attention_rollout":
            attributions = self.attention_rollout(inputs, target, **kwargs)
        else:
            raise ValueError(f"Unknown attribution method: {method}")

        # Compute scores
        baseline_score = None
        if baseline is not None:
            with torch.no_grad():
                baseline_output = self.model(baseline)
                if target is not None and len(baseline_output.shape) > 1:
                    baseline_score = float(baseline_output[:, target].mean())
                else:
                    baseline_score = float(baseline_output.mean())

        target_score = None
        with torch.no_grad():
            target_output = self.model(inputs)
            if target is not None and len(target_output.shape) > 1:
                target_score = float(target_output[:, target].mean())
            else:
                target_score = float(target_output.mean())

        # Create result
        result = AttributionResult(
            method=method,
            attributions=attributions,
            baseline_score=baseline_score,
            target_score=target_score,
            feature_names=self.feature_names,
            metadata={
                "input_shape": list(inputs.shape),
                "has_baseline": baseline is not None,
                "target_class": target,
            },
        )

        self.attribution_results.append(result)
        return result

    def integrated_gradients(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
        internal_batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Compute integrated gradients attribution.

        Args:
            inputs: Input tensor
            target: Target class
            baseline: Baseline tensor
            steps: Number of integration steps
            internal_batch_size: Batch size for computation

        Returns:
            Attribution tensor
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        # Generate alphas for interpolation
        alphas = torch.linspace(0, 1, steps).to(inputs.device)

        # Initialize gradients accumulator
        grads = []

        # Process in batches for memory efficiency
        for i in range(0, steps, internal_batch_size):
            batch_alphas = alphas[i : i + internal_batch_size]
            batch_size = len(batch_alphas)

            # Create batch of interpolated inputs
            batch_interpolated = torch.stack(
                [baseline + alpha * (inputs - baseline) for alpha in batch_alphas]
            )

            # Reshape for batch processing
            if len(inputs.shape) > 1:
                batch_interpolated = batch_interpolated.view(
                    batch_size * inputs.shape[0], *inputs.shape[1:]
                )

            batch_interpolated.requires_grad_(True)

            # Forward pass
            outputs = self.model(batch_interpolated)

            # Select target
            if target is not None and len(outputs.shape) > 1:
                selected = outputs[:, target]
            else:
                selected = outputs.squeeze()

            # Backward pass
            self.model.zero_grad()
            selected.sum().backward()

            # Collect gradients
            batch_grads = batch_interpolated.grad.data

            # Reshape back
            if len(inputs.shape) > 1:
                batch_grads = batch_grads.view(batch_size, *inputs.shape)

            grads.append(batch_grads)

        # Concatenate and average gradients
        grads = torch.cat(grads, dim=0)
        avg_grads = grads.mean(dim=0)

        # Compute integrated gradients
        integrated_grads = (inputs - baseline) * avg_grads

        return integrated_grads

    def gradient_shap(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        n_samples: int = 50,
        stdevs: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute GradientSHAP attributions.

        Args:
            inputs: Input tensor
            target: Target class
            baseline: Baseline tensor
            n_samples: Number of samples
            stdevs: Standard deviation for noise

        Returns:
            Attribution tensor
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        # Generate random samples between baseline and input
        attributions = torch.zeros_like(inputs)

        for _ in range(n_samples):
            # Random interpolation factor
            alpha = torch.rand(1).item()

            # Interpolate
            sample = baseline + alpha * (inputs - baseline)

            # Add noise
            noise = torch.randn_like(sample) * stdevs
            sample = sample + noise

            # Compute gradient
            sample.requires_grad_(True)
            outputs = self.model(sample)

            if target is not None and len(outputs.shape) > 1:
                selected = outputs[:, target]
            else:
                selected = outputs.squeeze()

            self.model.zero_grad()
            selected.sum().backward()

            # Accumulate gradients
            attributions += sample.grad.data * (inputs - baseline)

        # Average
        attributions /= n_samples

        return attributions

    def deeplift(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute DeepLIFT attributions (simplified version).

        Args:
            inputs: Input tensor
            target: Target class
            baseline: Baseline tensor

        Returns:
            Attribution tensor
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        # Forward pass for input
        inputs.requires_grad_(True)
        output = self.model(inputs)

        if target is not None and len(output.shape) > 1:
            target_output = output[:, target]
        else:
            target_output = output.squeeze()

        # Forward pass for baseline
        with torch.no_grad():
            baseline_output = self.model(baseline)
            if target is not None and len(baseline_output.shape) > 1:
                baseline_target = baseline_output[:, target]
            else:
                baseline_target = baseline_output.squeeze()

        # Compute difference
        diff_output = target_output - baseline_target

        # Backward pass
        self.model.zero_grad()
        diff_output.sum().backward()

        # Attribution is gradient times input difference
        attributions = inputs.grad.data * (inputs - baseline)

        return attributions

    def occlusion_sensitivity(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        window_size: Union[int, Tuple[int, ...]] = 3,
        stride: int = 1,
        occlusion_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute occlusion sensitivity map.

        Args:
            inputs: Input tensor
            target: Target class
            window_size: Size of occlusion window
            stride: Stride for sliding window
            occlusion_value: Value to use for occlusion

        Returns:
            Attribution tensor
        """
        # Get baseline score
        with torch.no_grad():
            baseline_output = self.model(inputs)
            if target is not None and len(baseline_output.shape) > 1:
                baseline_score = baseline_output[:, target]
            else:
                baseline_score = baseline_output

        # Initialize attribution map
        attributions = torch.zeros_like(inputs)

        # Handle different input dimensions
        if len(inputs.shape) == 4:  # Image input (batch, channels, height, width)
            b, c, h, w = inputs.shape

            if isinstance(window_size, int):
                window_h = window_w = window_size
            else:
                window_h, window_w = window_size

            # Slide window across image
            for i in range(0, h - window_h + 1, stride):
                for j in range(0, w - window_w + 1, stride):
                    # Create occluded input
                    occluded = inputs.clone()
                    occluded[:, :, i : i + window_h, j : j + window_w] = occlusion_value

                    # Get prediction for occluded input
                    with torch.no_grad():
                        occluded_output = self.model(occluded)
                        if target is not None and len(occluded_output.shape) > 1:
                            occluded_score = occluded_output[:, target]
                        else:
                            occluded_score = occluded_output

                    # Compute importance (drop in score)
                    importance = (baseline_score - occluded_score).abs()

                    # Assign to attribution map
                    attributions[:, :, i : i + window_h, j : j + window_w] += importance.view(
                        b, 1, 1, 1
                    ).expand_as(attributions[:, :, i : i + window_h, j : j + window_w])

        elif len(inputs.shape) == 2:  # Sequence input (batch, sequence)
            b, seq_len = inputs.shape

            if isinstance(window_size, int):
                window = window_size
            else:
                window = window_size[0]

            # Slide window across sequence
            for i in range(0, seq_len - window + 1, stride):
                # Create occluded input
                occluded = inputs.clone()
                occluded[:, i : i + window] = occlusion_value

                # Get prediction
                with torch.no_grad():
                    occluded_output = self.model(occluded)
                    if target is not None and len(occluded_output.shape) > 1:
                        occluded_score = occluded_output[:, target]
                    else:
                        occluded_score = occluded_output

                # Compute importance
                importance = (baseline_score - occluded_score).abs()

                # Assign to attribution map
                attributions[:, i : i + window] += importance.unsqueeze(1)

        return attributions

    def lime_attribution(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        n_samples: int = 1000,
        feature_mask_prob: float = 0.5,
    ) -> torch.Tensor:
        """
        Simplified LIME-style attribution.

        Args:
            inputs: Input tensor
            target: Target class
            n_samples: Number of samples
            feature_mask_prob: Probability of masking each feature

        Returns:
            Attribution tensor
        """
        # Generate perturbed samples
        samples = []
        predictions = []
        weights = []

        for _ in range(n_samples):
            # Create binary mask
            mask = torch.bernoulli(torch.ones_like(inputs) * (1 - feature_mask_prob))

            # Apply mask
            perturbed = inputs * mask

            # Get prediction
            with torch.no_grad():
                output = self.model(perturbed)
                if target is not None and len(output.shape) > 1:
                    pred = output[:, target]
                else:
                    pred = output

            samples.append(mask)
            predictions.append(pred)

            # Weight by similarity to original
            similarity = mask.mean()
            weights.append(similarity)

        # Stack samples and predictions
        samples = torch.stack(samples)
        predictions = torch.stack(predictions)
        weights = torch.tensor(weights)

        # Fit linear model (simplified)
        # Using weighted least squares
        X = samples.view(n_samples, -1)
        y = predictions.view(n_samples, -1)
        w = weights.view(n_samples, 1)

        # Weighted regression
        X_weighted = X * w.sqrt()
        y_weighted = y * w.sqrt()

        # Solve normal equations
        coefficients = torch.linalg.lstsq(X_weighted, y_weighted).solution

        # Reshape coefficients to input shape
        attributions = coefficients.view(inputs.shape)

        return attributions

    def attention_rollout(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        attention_weights_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute attention rollout for transformer models.

        Args:
            inputs: Input tensor
            target: Target class
            attention_weights_list: List of attention weights per layer

        Returns:
            Attribution tensor
        """
        if attention_weights_list is None:
            # Try to extract attention weights from model
            # This is model-specific and may need adaptation
            logger.warning("Attention weights not provided, using uniform attribution")
            return torch.ones_like(inputs) / inputs.numel()

        # Start with identity matrix
        rollout = torch.eye(inputs.shape[-1]).to(inputs.device)

        for attention in attention_weights_list:
            if isinstance(attention, torch.Tensor):
                # Average across heads if needed
                if len(attention.shape) == 4:
                    attention = attention.mean(dim=1)
                elif len(attention.shape) == 3:
                    attention = attention.mean(dim=0)

                # Add residual connection
                attention = 0.5 * attention + 0.5 * torch.eye(attention.shape[-1]).to(
                    attention.device
                )

                # Normalize
                attention = attention / attention.sum(dim=-1, keepdim=True)

                # Apply rollout
                rollout = torch.matmul(attention, rollout)

        # Extract attribution for target token
        if target is not None and target < rollout.shape[0]:
            attributions = rollout[target]
        else:
            attributions = rollout.mean(dim=0)

        # Expand to input shape if needed
        if len(inputs.shape) > 1:
            attributions = attributions.unsqueeze(0).expand_as(inputs)

        return attributions

    def get_top_features(
        self, attribution_result: AttributionResult, top_k: int = 10, absolute: bool = True
    ) -> List[Tuple[int, float, Optional[str]]]:
        """
        Get top-k most important features.

        Args:
            attribution_result: Attribution result
            top_k: Number of top features
            absolute: Use absolute values for ranking

        Returns:
            List of (index, score, name) tuples
        """
        attributions = attribution_result.attributions.flatten()

        if absolute:
            scores = attributions.abs()
        else:
            scores = attributions

        # Get top-k indices
        top_values, top_indices = torch.topk(scores, min(top_k, len(scores)))

        # Create result list
        top_features = []
        for idx, val in zip(top_indices, top_values):
            idx = int(idx)
            score = float(attributions[idx])  # Use original value, not absolute

            name = None
            if attribution_result.feature_names and idx < len(attribution_result.feature_names):
                name = attribution_result.feature_names[idx]

            top_features.append((idx, score, name))

        return top_features

    def compare_methods(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        methods: List[str] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> Dict[str, AttributionResult]:
        """
        Compare multiple attribution methods.

        Args:
            inputs: Input tensor
            target: Target class
            methods: List of methods to compare
            baseline: Baseline tensor

        Returns:
            Dictionary of attribution results
        """
        if methods is None:
            methods = ["integrated_gradients", "gradient_shap", "deeplift", "occlusion"]

        results = {}

        for method in methods:
            try:
                result = self.compute_attributions(inputs, target, method, baseline)
                results[method] = result
            except Exception as e:
                logger.warning(f"Failed to compute {method}: {e}")

        return results

    def export_attributions(self, output_path: str, format: str = "npz") -> None:
        """
        Export attribution results.

        Args:
            output_path: Path to save attributions
            format: Export format (npz, json, pt)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "npz":
            # Export as numpy archive
            arrays = {}
            metadata = {}

            for i, result in enumerate(self.attribution_results):
                key = f"{result.method}_{i}"
                arrays[key] = result.attributions.cpu().numpy()
                metadata[key] = {
                    "method": result.method,
                    "baseline_score": result.baseline_score,
                    "target_score": result.target_score,
                    "metadata": result.metadata,
                }

            np.savez(output_path, **arrays)

            # Save metadata separately
            meta_path = output_path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        elif format == "pt":
            # Export as PyTorch checkpoint
            torch.save({"results": self.attribution_results}, output_path)

        logger.info(
            f"Exported {len(self.attribution_results)} attribution results to {output_path}"
        )
