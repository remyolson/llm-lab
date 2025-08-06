"""Natural language explanation generator for model predictions."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Container for model explanation."""

    prediction: Any
    confidence: float
    top_features: List[Tuple[str, float]]
    attention_summary: Optional[str] = None
    gradient_summary: Optional[str] = None
    activation_summary: Optional[str] = None
    natural_language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExplanationGenerator:
    """Generates human-readable explanations for model predictions."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        explanation_style: str = "technical",
    ):
        """
        Initialize the explanation generator.

        Args:
            model: The model to explain
            tokenizer: Optional tokenizer for text models
            explanation_style: Style of explanation (technical, simple, detailed)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.explanation_style = explanation_style
        self.explanations = []

    def generate_explanation(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        prediction: Any,
        attention_weights: Optional[Dict[str, torch.Tensor]] = None,
        gradient_info: Optional[Dict[str, Any]] = None,
        activation_stats: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Explanation:
        """
        Generate comprehensive explanation for a prediction.

        Args:
            inputs: Model inputs
            prediction: Model prediction
            attention_weights: Optional attention weights
            gradient_info: Optional gradient information
            activation_stats: Optional activation statistics
            feature_importance: Optional feature importance scores

        Returns:
            Explanation object
        """
        # Extract confidence
        confidence = self._extract_confidence(prediction)

        # Get top features
        top_features = self._extract_top_features(inputs, feature_importance, gradient_info)

        # Generate summaries
        attention_summary = None
        if attention_weights:
            attention_summary = self._generate_attention_summary(attention_weights)

        gradient_summary = None
        if gradient_info:
            gradient_summary = self._generate_gradient_summary(gradient_info)

        activation_summary = None
        if activation_stats:
            activation_summary = self._generate_activation_summary(activation_stats)

        # Generate natural language explanation
        natural_language = self._generate_natural_language(
            prediction,
            confidence,
            top_features,
            attention_summary,
            gradient_summary,
            activation_summary,
        )

        # Create metadata
        metadata = {
            "model_type": self.model.__class__.__name__,
            "explanation_style": self.explanation_style,
            "has_attention": attention_weights is not None,
            "has_gradients": gradient_info is not None,
            "has_activations": activation_stats is not None,
        }

        explanation = Explanation(
            prediction=prediction,
            confidence=confidence,
            top_features=top_features,
            attention_summary=attention_summary,
            gradient_summary=gradient_summary,
            activation_summary=activation_summary,
            natural_language=natural_language,
            metadata=metadata,
        )

        self.explanations.append(explanation)
        return explanation

    def _extract_confidence(self, prediction: Any) -> float:
        """Extract confidence score from prediction."""
        if isinstance(prediction, torch.Tensor):
            if len(prediction.shape) == 1:
                # Binary or regression
                return float(torch.sigmoid(prediction).max())
            else:
                # Multi-class
                probs = torch.softmax(prediction, dim=-1)
                return float(probs.max())
        elif isinstance(prediction, (list, np.ndarray)):
            return float(np.max(prediction))
        else:
            return 0.0

    def _extract_top_features(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        feature_importance: Optional[Dict[str, float]],
        gradient_info: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        """Extract top important features."""
        top_features = []

        if feature_importance:
            # Use provided feature importance
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )
            top_features = sorted_features[:10]
        elif gradient_info:
            # Use gradient magnitudes as importance
            for layer_name, info in gradient_info.items():
                if "gradient_norm" in info:
                    top_features.append((layer_name, info["gradient_norm"]))
            top_features.sort(key=lambda x: x[1], reverse=True)
            top_features = top_features[:10]
        elif self.tokenizer and isinstance(inputs, dict) and "input_ids" in inputs:
            # For text models, use token positions
            input_ids = inputs["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids[0] if len(input_ids.shape) > 1 else input_ids
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
                # Simple heuristic: assign importance based on position
                for i, token in enumerate(tokens[:10]):
                    importance = 1.0 / (i + 1)
                    top_features.append((token, importance))

        return top_features

    def _generate_attention_summary(self, attention_weights: Dict[str, torch.Tensor]) -> str:
        """Generate summary of attention patterns."""
        summaries = []

        for layer_name, weights in attention_weights.items():
            if isinstance(weights, torch.Tensor):
                # Analyze attention pattern
                avg_attention = weights.mean().item()
                max_attention = weights.max().item()

                # Identify focus pattern
                if len(weights.shape) >= 2:
                    # Check diagonal attention (self-focus)
                    seq_len = min(weights.shape[-2], weights.shape[-1])
                    if seq_len > 0:
                        diagonal_avg = (
                            torch.diagonal(weights[..., :seq_len, :seq_len], dim1=-2, dim2=-1)
                            .mean()
                            .item()
                        )
                        if diagonal_avg > 0.5:
                            pattern = "self-focused"
                        elif max_attention > 0.8:
                            pattern = "highly concentrated"
                        else:
                            pattern = "distributed"
                    else:
                        pattern = "unknown"
                else:
                    pattern = "simple"

                summaries.append(
                    f"{layer_name}: {pattern} attention (avg={avg_attention:.3f}, max={max_attention:.3f})"
                )

        if self.explanation_style == "simple":
            return (
                f"The model is paying attention to {len(summaries)} different parts of the input."
            )
        else:
            return " | ".join(summaries[:3])  # Show top 3 layers

    def _generate_gradient_summary(self, gradient_info: Dict[str, Any]) -> str:
        """Generate summary of gradient information."""
        if not gradient_info:
            return "No gradient information available."

        # Find layers with highest gradient norms
        high_gradient_layers = []
        low_gradient_layers = []

        for layer_name, info in gradient_info.items():
            if isinstance(info, dict) and "gradient_norm" in info:
                norm = info["gradient_norm"]
                if norm > 1.0:
                    high_gradient_layers.append((layer_name, norm))
                elif norm < 0.01:
                    low_gradient_layers.append((layer_name, norm))

        high_gradient_layers.sort(key=lambda x: x[1], reverse=True)

        if self.explanation_style == "simple":
            if high_gradient_layers:
                return f"Strong signals detected in {len(high_gradient_layers)} layers."
            else:
                return "Gradient flow is stable across all layers."
        else:
            summaries = []
            if high_gradient_layers:
                top_layer = high_gradient_layers[0]
                summaries.append(f"Highest gradient in {top_layer[0]} (norm={top_layer[1]:.3f})")
            if low_gradient_layers:
                summaries.append(
                    f"{len(low_gradient_layers)} layers show potential vanishing gradients"
                )

            return " | ".join(summaries) if summaries else "Normal gradient flow"

    def _generate_activation_summary(self, activation_stats: Dict[str, Any]) -> str:
        """Generate summary of activation patterns."""
        if not activation_stats:
            return "No activation information available."

        total_layers = len(activation_stats)
        dead_neuron_layers = 0
        high_sparsity_layers = 0
        saturated_layers = 0

        for layer_name, stats in activation_stats.items():
            if isinstance(stats, dict):
                if stats.get("dead_neurons"):
                    dead_neuron_layers += 1
                if stats.get("sparsity", 0) > 0.5:
                    high_sparsity_layers += 1
                if stats.get("saturation_ratio", 0) > 0.1:
                    saturated_layers += 1

        if self.explanation_style == "simple":
            issues = []
            if dead_neuron_layers > 0:
                issues.append(f"{dead_neuron_layers} layers have inactive neurons")
            if high_sparsity_layers > 0:
                issues.append(f"{high_sparsity_layers} layers are highly sparse")
            if saturated_layers > 0:
                issues.append(f"{saturated_layers} layers show saturation")

            if issues:
                return f"Potential issues: {', '.join(issues)}"
            else:
                return "All layers show healthy activation patterns."
        else:
            return (
                f"Activations: {total_layers} layers analyzed | "
                f"{dead_neuron_layers} with dead neurons | "
                f"{high_sparsity_layers} sparse | "
                f"{saturated_layers} saturated"
            )

    def _generate_natural_language(
        self,
        prediction: Any,
        confidence: float,
        top_features: List[Tuple[str, float]],
        attention_summary: Optional[str],
        gradient_summary: Optional[str],
        activation_summary: Optional[str],
    ) -> str:
        """Generate natural language explanation."""
        explanation_parts = []

        # Prediction and confidence
        if self.explanation_style == "simple":
            conf_level = (
                "very confident"
                if confidence > 0.9
                else "moderately confident"
                if confidence > 0.7
                else "uncertain"
            )
            explanation_parts.append(
                f"The model is {conf_level} about this prediction (confidence: {confidence:.1%})."
            )
        else:
            explanation_parts.append(f"Prediction confidence: {confidence:.2%}")

        # Top features
        if top_features:
            if self.explanation_style == "simple":
                feature_names = [f[0] for f in top_features[:3]]
                explanation_parts.append(
                    f"The most important factors are: {', '.join(feature_names)}"
                )
            else:
                features_str = ", ".join(
                    [f"{name} ({score:.3f})" for name, score in top_features[:5]]
                )
                explanation_parts.append(f"Top features: {features_str}")

        # Add component summaries
        if attention_summary:
            if self.explanation_style == "simple":
                explanation_parts.append(attention_summary)
            else:
                explanation_parts.append(f"Attention: {attention_summary}")

        if gradient_summary and self.explanation_style != "simple":
            explanation_parts.append(f"Gradients: {gradient_summary}")

        if activation_summary and self.explanation_style != "simple":
            explanation_parts.append(f"Activations: {activation_summary}")

        # Combine parts
        if self.explanation_style == "simple":
            return " ".join(explanation_parts)
        else:
            return " | ".join(explanation_parts)

    def generate_contrastive_explanation(
        self,
        original_inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        modified_inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        original_prediction: Any,
        modified_prediction: Any,
    ) -> str:
        """
        Generate explanation comparing two predictions.

        Args:
            original_inputs: Original model inputs
            modified_inputs: Modified model inputs
            original_prediction: Original prediction
            modified_prediction: Modified prediction

        Returns:
            Contrastive explanation string
        """
        orig_conf = self._extract_confidence(original_prediction)
        mod_conf = self._extract_confidence(modified_prediction)

        conf_change = mod_conf - orig_conf

        if self.explanation_style == "simple":
            if abs(conf_change) < 0.1:
                return "The modification had minimal impact on the prediction."
            elif conf_change > 0:
                return f"The modification increased confidence by {conf_change:.1%}."
            else:
                return f"The modification decreased confidence by {abs(conf_change):.1%}."
        else:
            explanation = (
                f"Confidence change: {orig_conf:.2%} → {mod_conf:.2%} ({conf_change:+.2%})"
            )

            # Add more detailed analysis if available
            if isinstance(original_prediction, torch.Tensor) and isinstance(
                modified_prediction, torch.Tensor
            ):
                if len(original_prediction.shape) > 1:
                    # Multi-class case
                    orig_class = original_prediction.argmax(dim=-1)
                    mod_class = modified_prediction.argmax(dim=-1)
                    if not torch.equal(orig_class, mod_class):
                        explanation += " | Prediction class changed"

            return explanation

    def export_explanations(self, output_path: str, format: str = "json") -> None:
        """
        Export explanations to file.

        Args:
            output_path: Path to save explanations
            format: Export format (json, txt, html)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            export_data = []
            for exp in self.explanations:
                exp_dict = {
                    "confidence": exp.confidence,
                    "top_features": exp.top_features,
                    "natural_language": exp.natural_language,
                    "attention_summary": exp.attention_summary,
                    "gradient_summary": exp.gradient_summary,
                    "activation_summary": exp.activation_summary,
                    "metadata": exp.metadata,
                }
                export_data.append(exp_dict)

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

        elif format == "txt":
            with open(output_path, "w") as f:
                for i, exp in enumerate(self.explanations):
                    f.write(f"=== Explanation {i + 1} ===\n")
                    f.write(f"{exp.natural_language}\n")
                    f.write(f"Confidence: {exp.confidence:.2%}\n")
                    f.write("\n")

        elif format == "html":
            html_content = "<html><body><h1>Model Explanations</h1>"
            for i, exp in enumerate(self.explanations):
                html_content += f"<div class='explanation'>"
                html_content += f"<h2>Explanation {i + 1}</h2>"
                html_content += f"<p>{exp.natural_language}</p>"
                html_content += f"<p><strong>Confidence:</strong> {exp.confidence:.2%}</p>"
                html_content += "</div><hr>"
            html_content += "</body></html>"

            with open(output_path, "w") as f:
                f.write(html_content)

        logger.info(f"Exported {len(self.explanations)} explanations to {output_path}")

    def clear(self):
        """Clear stored explanations."""
        self.explanations = []
