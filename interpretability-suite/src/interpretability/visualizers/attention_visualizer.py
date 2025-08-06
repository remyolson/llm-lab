"""Visualization tools for attention patterns."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """Visualizes attention patterns from transformer models."""

    def __init__(self, style: str = "default"):
        """
        Initialize the attention visualizer.

        Args:
            style: Visual style preset ("default", "dark", "paper")
        """
        self.style = style
        self._setup_style()

    def _setup_style(self):
        """Set up visual style configurations."""
        if self.style == "dark":
            plt.style.use("dark_background")
            self.color_scheme = "viridis"
            self.template = "plotly_dark"
        elif self.style == "paper":
            plt.style.use("seaborn-v0_8-paper")
            self.color_scheme = "coolwarm"
            self.template = "plotly_white"
        else:
            plt.style.use("default")
            self.color_scheme = "Blues"
            self.template = "plotly"

        sns.set_palette("husl")

    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        layer_name: Optional[str] = None,
        save_path: Optional[str] = None,
        interactive: bool = False,
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot attention weights as a heatmap.

        Args:
            attention_weights: Attention weight tensor
            tokens: Optional list of token strings
            layer_name: Optional layer name for title
            save_path: Optional path to save figure
            interactive: Use interactive plotly instead of matplotlib

        Returns:
            Figure object (matplotlib or plotly)
        """
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()

        # Handle different dimensions
        if len(attention_weights.shape) == 4:
            # Average across batch and heads
            attention_weights = attention_weights.mean(axis=(0, 1))
        elif len(attention_weights.shape) == 3:
            # Average across batch or heads
            attention_weights = attention_weights.mean(axis=0)

        seq_len = attention_weights.shape[0]

        if interactive:
            # Create interactive plotly heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=attention_weights,
                    x=tokens if tokens else list(range(seq_len)),
                    y=tokens if tokens else list(range(seq_len)),
                    colorscale=self.color_scheme,
                    hoverongaps=False,
                    hovertemplate="From: %{y}<br>To: %{x}<br>Weight: %{z:.3f}<extra></extra>",
                )
            )

            fig.update_layout(
                title=f"Attention Weights - {layer_name}" if layer_name else "Attention Weights",
                xaxis_title="To Token",
                yaxis_title="From Token",
                template=self.template,
                height=600,
                width=700,
            )

            if save_path:
                fig.write_html(save_path)

            return fig
        else:
            # Create matplotlib heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            im = ax.imshow(attention_weights, cmap=self.color_scheme, aspect="auto")

            # Add colorbar
            plt.colorbar(im, ax=ax)

            # Set ticks and labels
            if tokens:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha="right")
                ax.set_yticklabels(tokens)

            ax.set_xlabel("To Token")
            ax.set_ylabel("From Token")

            title = f"Attention Weights - {layer_name}" if layer_name else "Attention Weights"
            ax.set_title(title)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")

            return fig

    def plot_attention_heads_grid(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        layer_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot attention weights for multiple heads in a grid.

        Args:
            attention_weights: Attention weights [batch, heads, seq, seq]
            tokens: Optional list of token strings
            layer_name: Optional layer name
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()

        # Get dimensions
        if len(attention_weights.shape) == 4:
            attention_weights = attention_weights[0]  # Take first batch item

        num_heads = attention_weights.shape[0]
        seq_len = attention_weights.shape[1]

        # Determine grid size
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if num_heads > 1 else [axes]

        for head_idx in range(num_heads):
            ax = axes[head_idx]
            im = ax.imshow(attention_weights[head_idx], cmap=self.color_scheme, aspect="auto")

            ax.set_title(f"Head {head_idx}")

            if tokens and len(tokens) <= 20:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].set_visible(False)

        if layer_name:
            fig.suptitle(f"Attention Heads - {layer_name}", fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_attention_flow(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot attention flow as a Sankey diagram.

        Args:
            attention_weights: Attention weights
            tokens: List of tokens
            layer_indices: Indices to highlight
            save_path: Optional save path

        Returns:
            Plotly figure
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()

        # Average if needed
        if len(attention_weights.shape) > 2:
            attention_weights = attention_weights.reshape(-1, attention_weights.shape[-1])
            attention_weights = attention_weights.mean(axis=0)

        # Create source and target indices
        sources = []
        targets = []
        values = []

        threshold = 0.1  # Only show connections above threshold

        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if attention_weights[i, j] > threshold:
                    sources.append(i)
                    targets.append(j + len(tokens))
                    values.append(float(attention_weights[i, j]))

        # Create node labels
        node_labels = tokens + [f"{t}'" for t in tokens]

        # Create Sankey diagram
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=node_labels,
                        color="blue",
                    ),
                    link=dict(
                        source=sources, target=targets, value=values, color="rgba(0, 0, 255, 0.4)"
                    ),
                )
            ]
        )

        fig.update_layout(title="Attention Flow", font_size=10, template=self.template, height=600)

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_head_importance(
        self, head_importance_scores: Dict[str, torch.Tensor], save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot importance scores for attention heads.

        Args:
            head_importance_scores: Dict mapping layer names to importance scores
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        layer_names = []
        all_scores = []

        for layer_name, scores in head_importance_scores.items():
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            layer_names.extend([layer_name] * len(scores))
            all_scores.extend(scores)

        # Create bar plot
        x_pos = np.arange(len(all_scores))
        colors = plt.cm.Set3(np.linspace(0, 1, len(head_importance_scores)))

        color_indices = []
        for i, layer in enumerate(head_importance_scores.keys()):
            num_heads = len(head_importance_scores[layer])
            color_indices.extend([i] * num_heads)

        bars = ax.bar(x_pos, all_scores, color=[colors[i] for i in color_indices])

        # Customize plot
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Importance Score")
        ax.set_title("Attention Head Importance Across Layers")

        # Add layer boundaries
        boundaries = []
        current_pos = 0
        for layer_name, scores in head_importance_scores.items():
            if current_pos > 0:
                ax.axvline(x=current_pos - 0.5, color="gray", linestyle="--", alpha=0.5)
            boundaries.append((current_pos, current_pos + len(scores) - 1, layer_name))
            current_pos += len(scores)

        # Add layer labels
        for start, end, name in boundaries:
            ax.text(
                (start + end) / 2,
                ax.get_ylim()[1] * 0.95,
                name,
                ha="center",
                va="top",
                fontsize=10,
                style="italic",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_token_importance(
        self, tokens: List[str], importance_scores: torch.Tensor, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot token importance scores.

        Args:
            tokens: List of tokens
            importance_scores: Importance score tensor
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        if isinstance(importance_scores, torch.Tensor):
            importance_scores = importance_scores.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 4))

        # Create bar plot
        x_pos = np.arange(len(tokens))
        bars = ax.bar(x_pos, importance_scores)

        # Color bars by importance
        norm = plt.Normalize(vmin=importance_scores.min(), vmax=importance_scores.max())
        colors = plt.cm.RdYlGn(norm(importance_scores))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Customize plot
        ax.set_xlabel("Token")
        ax.set_ylabel("Importance Score")
        ax.set_title("Token Importance Analysis")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tokens, rotation=45, ha="right")

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, importance_scores)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def create_attention_animation(
        self,
        attention_weights_sequence: List[torch.Tensor],
        tokens: List[str],
        layer_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create animated attention visualization.

        Args:
            attention_weights_sequence: List of attention weights over time
            tokens: List of tokens
            layer_names: Optional layer names
            save_path: Optional save path

        Returns:
            Plotly figure with animation
        """
        frames = []

        for idx, weights in enumerate(attention_weights_sequence):
            if isinstance(weights, torch.Tensor):
                weights = weights.cpu().numpy()

            # Average if needed
            if len(weights.shape) > 2:
                weights = weights.mean(axis=tuple(range(len(weights.shape) - 2)))

            frame_name = layer_names[idx] if layer_names else f"Step {idx}"

            frame = go.Frame(
                data=[go.Heatmap(z=weights, x=tokens, y=tokens, colorscale=self.color_scheme)],
                name=frame_name,
            )
            frames.append(frame)

        # Create initial figure
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=attention_weights_sequence[0].cpu().numpy()
                    if isinstance(attention_weights_sequence[0], torch.Tensor)
                    else attention_weights_sequence[0],
                    x=tokens,
                    y=tokens,
                    colorscale=self.color_scheme,
                )
            ],
            frames=frames,
        )

        # Add animation controls
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True},
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "steps": [
                        {
                            "args": [
                                [frame.name],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                            "label": frame.name,
                            "method": "animate",
                        }
                        for frame in frames
                    ],
                    "x": 0.1,
                    "len": 0.9,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top",
                    "pad": {"b": 10, "t": 50},
                    "transition": {"duration": 300},
                }
            ],
            title="Attention Pattern Animation",
            xaxis_title="To Token",
            yaxis_title="From Token",
            template=self.template,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def export_visualizations(
        self, output_dir: str, attention_data: Dict[str, Any], format: str = "png"
    ) -> None:
        """
        Export multiple visualizations to directory.

        Args:
            output_dir: Output directory path
            attention_data: Dictionary containing attention data
            format: Output format (png, pdf, html)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export different visualizations
        if "attention_weights" in attention_data:
            for layer_name, weights in attention_data["attention_weights"].items():
                # Heatmap
                save_path = output_path / f"{layer_name}_heatmap.{format}"
                self.plot_attention_heatmap(
                    weights,
                    tokens=attention_data.get("tokens"),
                    layer_name=layer_name,
                    save_path=str(save_path),
                )

                # Head grid if applicable
                if len(weights.shape) >= 3:
                    save_path = output_path / f"{layer_name}_heads.{format}"
                    self.plot_attention_heads_grid(
                        weights,
                        tokens=attention_data.get("tokens"),
                        layer_name=layer_name,
                        save_path=str(save_path),
                    )

        logger.info(f"Exported visualizations to {output_dir}")
