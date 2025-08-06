"""
Model Behavior Analyzer with Embedding Visualization

This module provides analysis tools for model behavior including embedding space
visualization, attention pattern analysis, and neuron activation heatmaps.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Dimensionality reduction
try:
    from sklearn.manifold import TSNE
    from umap import UMAP

    REDUCTION_AVAILABLE = True
except ImportError:
    REDUCTION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BehaviorAnalysis:
    """Container for behavior analysis results."""

    timestamp: datetime
    embeddings: Optional[np.ndarray] = None
    reduced_embeddings: Optional[np.ndarray] = None
    attention_patterns: Optional[Dict[str, Any]] = None
    activation_maps: Optional[Dict[str, np.ndarray]] = None
    sensitivity_scores: Optional[Dict[str, float]] = None


class ModelBehaviorAnalyzer:
    """Analyzer for model behavior visualization."""

    def __init__(self, reduction_method: str = "tsne"):
        """Initialize behavior analyzer.

        Args:
            reduction_method: Method for dimensionality reduction ('tsne', 'umap')
        """
        self.reduction_method = reduction_method
        self.analysis_cache = {}

    def analyze_embeddings(
        self, embeddings: np.ndarray, labels: Optional[List[str]] = None, n_components: int = 2
    ) -> BehaviorAnalysis:
        """Analyze and reduce embedding dimensions.

        Args:
            embeddings: High-dimensional embeddings
            labels: Optional labels for embeddings
            n_components: Number of components for reduction

        Returns:
            BehaviorAnalysis object
        """
        reduced_embeddings = None

        if REDUCTION_AVAILABLE:
            if self.reduction_method == "tsne":
                reducer = TSNE(n_components=n_components, random_state=42)
            elif self.reduction_method == "umap":
                reducer = UMAP(n_components=n_components, random_state=42)
            else:
                raise ValueError(f"Unknown reduction method: {self.reduction_method}")

            reduced_embeddings = reducer.fit_transform(embeddings)

        return BehaviorAnalysis(
            timestamp=datetime.now(), embeddings=embeddings, reduced_embeddings=reduced_embeddings
        )

    def analyze_attention(self, attention_weights: np.ndarray, tokens: List[str]) -> Dict[str, Any]:
        """Analyze attention patterns.

        Args:
            attention_weights: Attention weight matrix
            tokens: Token list

        Returns:
            Attention analysis results
        """
        # Analyze attention patterns
        max_attention = np.max(attention_weights, axis=-1)
        mean_attention = np.mean(attention_weights, axis=-1)

        # Find most attended tokens
        top_k = min(5, len(tokens))
        top_indices = np.argsort(mean_attention)[-top_k:]
        top_tokens = [tokens[i] for i in top_indices]

        return {
            "max_attention": max_attention.tolist(),
            "mean_attention": mean_attention.tolist(),
            "top_tokens": top_tokens,
            "attention_entropy": -np.sum(attention_weights * np.log(attention_weights + 1e-10)),
        }

    def analyze_activations(
        self, layer_activations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Analyze neuron activations.

        Args:
            layer_activations: Activations by layer

        Returns:
            Activation analysis results
        """
        activation_maps = {}

        for layer_name, activations in layer_activations.items():
            # Compute activation statistics
            activation_maps[layer_name] = {
                "mean": np.mean(activations),
                "std": np.std(activations),
                "max": np.max(activations),
                "sparsity": np.mean(activations == 0),
            }

        return activation_maps

    def test_prompt_sensitivity(
        self, model, base_prompt: str, variations: List[str]
    ) -> Dict[str, float]:
        """Test model sensitivity to prompt variations.

        Args:
            model: Model to test
            base_prompt: Base prompt
            variations: Prompt variations

        Returns:
            Sensitivity scores
        """
        # Placeholder for sensitivity testing
        sensitivity_scores = {}

        for variation in variations:
            # Would compute output difference
            sensitivity_scores[variation] = np.random.random()

        return sensitivity_scores
