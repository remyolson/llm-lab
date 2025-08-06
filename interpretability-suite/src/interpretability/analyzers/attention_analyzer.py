"""Attention pattern analysis for transformer models."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .hook_manager import HookManager

logger = logging.getLogger(__name__)


@dataclass
class AttentionPattern:
    """Container for attention pattern analysis results."""

    layer_name: str
    attention_weights: torch.Tensor
    head_importance: Optional[torch.Tensor] = None
    token_importance: Optional[torch.Tensor] = None
    pattern_type: Optional[str] = None
    metadata: Dict[str, Any] = None


class AttentionAnalyzer:
    """Analyzes attention patterns in transformer models."""

    def __init__(self, model: nn.Module):
        """
        Initialize the attention analyzer.

        Args:
            model: The transformer model to analyze
        """
        self.model = model
        self.hook_manager = HookManager(model)
        self.attention_patterns = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Set up hooks for attention layers."""
        self.hook_manager.register_attention_hooks()

    def analyze_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> List[AttentionPattern]:
        """
        Analyze attention patterns for given input.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            token_type_ids: Optional token type IDs

        Returns:
            List of attention patterns for each layer
        """
        self.hook_manager.clear_outputs()
        self.attention_patterns = []

        # Forward pass to collect attention weights
        with torch.no_grad():
            model_inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids

            # Try different model signatures
            try:
                outputs = self.model(**model_inputs, output_attentions=True)
            except TypeError:
                # Fallback for models without output_attentions
                outputs = self.model(**model_inputs)

        # Extract attention weights from hooks
        attention_weights = self.hook_manager.extract_attention_weights()

        # Analyze each layer's attention
        for layer_name, weights in attention_weights.items():
            pattern = self._analyze_layer_attention(layer_name, weights)
            self.attention_patterns.append(pattern)

        return self.attention_patterns

    def _analyze_layer_attention(
        self, layer_name: str, attention_weights: torch.Tensor
    ) -> AttentionPattern:
        """
        Analyze attention patterns for a single layer.

        Args:
            layer_name: Name of the layer
            attention_weights: Attention weight tensor

        Returns:
            AttentionPattern object with analysis results
        """
        # Calculate head importance (entropy-based)
        head_importance = self._calculate_head_importance(attention_weights)

        # Calculate token importance
        token_importance = self._calculate_token_importance(attention_weights)

        # Identify pattern type
        pattern_type = self._identify_pattern_type(attention_weights)

        # Create metadata
        metadata = {
            "num_heads": attention_weights.shape[1] if len(attention_weights.shape) > 1 else 1,
            "sequence_length": attention_weights.shape[-1],
            "mean_attention": float(attention_weights.mean()),
            "max_attention": float(attention_weights.max()),
            "min_attention": float(attention_weights.min()),
        }

        return AttentionPattern(
            layer_name=layer_name,
            attention_weights=attention_weights,
            head_importance=head_importance,
            token_importance=token_importance,
            pattern_type=pattern_type,
            metadata=metadata,
        )

    def _calculate_head_importance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate importance of each attention head using entropy.

        Args:
            attention_weights: Attention weights tensor [batch, heads, seq_len, seq_len]

        Returns:
            Head importance scores
        """
        if len(attention_weights.shape) < 4:
            return None

        # Calculate entropy for each head
        eps = 1e-8
        attention_probs = attention_weights + eps
        entropy = -(attention_probs * torch.log(attention_probs)).sum(dim=-1).mean(dim=(0, 2))

        # Normalize to importance scores (lower entropy = more focused = more important)
        max_entropy = torch.log(torch.tensor(attention_weights.shape[-1], dtype=torch.float32))
        importance = 1.0 - (entropy / max_entropy)

        return importance

    def _calculate_token_importance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate importance of each token based on attention received.

        Args:
            attention_weights: Attention weights tensor

        Returns:
            Token importance scores
        """
        # Sum attention received by each token across all heads and layers
        if len(attention_weights.shape) == 4:
            # [batch, heads, seq_len, seq_len] -> [seq_len]
            token_importance = attention_weights.mean(dim=(0, 1)).sum(dim=0)
        elif len(attention_weights.shape) == 3:
            # [batch, seq_len, seq_len] -> [seq_len]
            token_importance = attention_weights.mean(dim=0).sum(dim=0)
        else:
            # [seq_len, seq_len] -> [seq_len]
            token_importance = attention_weights.sum(dim=0)

        return token_importance

    def _identify_pattern_type(self, attention_weights: torch.Tensor) -> str:
        """
        Identify the type of attention pattern.

        Args:
            attention_weights: Attention weights tensor

        Returns:
            Pattern type string
        """
        # Get the attention matrix for analysis (average across batch and heads if needed)
        if len(attention_weights.shape) == 4:
            attn_matrix = attention_weights.mean(dim=(0, 1))
        elif len(attention_weights.shape) == 3:
            attn_matrix = attention_weights.mean(dim=0)
        else:
            attn_matrix = attention_weights

        seq_len = attn_matrix.shape[0]

        # Check for diagonal pattern (self-attention)
        diagonal_strength = torch.diag(attn_matrix).mean()

        # Check for vertical pattern (attention to specific tokens)
        max_col_attention = attn_matrix.max(dim=0)[0].mean()

        # Check for horizontal pattern (broadcasting from specific tokens)
        max_row_attention = attn_matrix.max(dim=1)[0].mean()

        # Classify pattern
        if diagonal_strength > 0.5:
            return "diagonal"
        elif max_col_attention > 0.6:
            return "vertical"
        elif max_row_attention > 0.6:
            return "horizontal"
        elif attn_matrix[:, 0].mean() > 0.4:
            return "first_token"
        elif attn_matrix[:, -1].mean() > 0.4:
            return "last_token"
        else:
            return "mixed"

    def get_head_importance_ranking(self) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get ranking of attention heads by importance.

        Returns:
            Dictionary mapping layer names to ranked head indices and scores
        """
        rankings = {}

        for pattern in self.attention_patterns:
            if pattern.head_importance is not None:
                # Get indices and scores
                scores = pattern.head_importance.cpu().numpy()
                indices = np.argsort(scores)[::-1]

                # Create ranking list
                rankings[pattern.layer_name] = [(int(idx), float(scores[idx])) for idx in indices]

        return rankings

    def get_token_importance_ranking(self) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get ranking of tokens by importance.

        Returns:
            Dictionary mapping layer names to ranked token indices and scores
        """
        rankings = {}

        for pattern in self.attention_patterns:
            if pattern.token_importance is not None:
                # Get indices and scores
                scores = pattern.token_importance.cpu().numpy()
                indices = np.argsort(scores)[::-1]

                # Create ranking list
                rankings[pattern.layer_name] = [(int(idx), float(scores[idx])) for idx in indices]

        return rankings

    def find_attention_heads_for_tokens(
        self, token_indices: List[int]
    ) -> Dict[str, Dict[int, float]]:
        """
        Find which attention heads focus on specific tokens.

        Args:
            token_indices: Indices of tokens to analyze

        Returns:
            Dictionary mapping layer names to head indices and attention scores
        """
        results = {}

        for pattern in self.attention_patterns:
            if len(pattern.attention_weights.shape) >= 4:
                # Average across batch dimension
                attn = pattern.attention_weights.mean(dim=0)

                # Calculate attention to specified tokens for each head
                head_scores = {}
                for head_idx in range(attn.shape[0]):
                    score = 0.0
                    for token_idx in token_indices:
                        if token_idx < attn.shape[-1]:
                            score += attn[head_idx, :, token_idx].mean().item()
                    head_scores[head_idx] = score / len(token_indices)

                results[pattern.layer_name] = head_scores

        return results

    def export_patterns(self, output_path: Union[str, Path]) -> None:
        """
        Export attention patterns to file.

        Args:
            output_path: Path to save patterns
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for export
        export_data = {"patterns": []}

        for pattern in self.attention_patterns:
            pattern_data = {
                "layer_name": pattern.layer_name,
                "pattern_type": pattern.pattern_type,
                "metadata": pattern.metadata,
                "attention_shape": list(pattern.attention_weights.shape),
            }

            if pattern.head_importance is not None:
                pattern_data["head_importance"] = pattern.head_importance.cpu().numpy().tolist()

            if pattern.token_importance is not None:
                pattern_data["token_importance"] = pattern.token_importance.cpu().numpy().tolist()

            export_data["patterns"].append(pattern_data)

        # Save as PyTorch checkpoint
        torch.save(export_data, output_path)
        logger.info(f"Exported attention patterns to {output_path}")

    def clear(self):
        """Clear all stored patterns and hook outputs."""
        self.attention_patterns = []
        self.hook_manager.clear_outputs()

    def __del__(self):
        """Clean up hooks on deletion."""
        if hasattr(self, "hook_manager"):
            self.hook_manager.clear_hooks()
