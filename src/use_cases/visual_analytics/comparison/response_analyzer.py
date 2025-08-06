"""
Response Comparison View with Diff Highlighting

This module provides side-by-side model output comparison with diff highlighting,
token-level attention visualization, and semantic similarity scoring.

Example:
    analyzer = ResponseComparisonView()
    comparison = analyzer.compare_responses(
        prompt="Translate to French: Hello world",
        model1_response="Bonjour le monde",
        model2_response="Salut le monde"
    )
    analyzer.visualize_comparison(comparison)
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import difflib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# NLP libraries
try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TokenComparison:
    """Token-level comparison data."""

    token1: str
    token2: str
    position: int
    match: bool
    similarity: float
    attention_weight1: Optional[float] = None
    attention_weight2: Optional[float] = None


@dataclass
class ResponseComparison:
    """Complete response comparison."""

    prompt: str
    response1: str
    response2: str
    model1_name: str
    model2_name: str
    timestamp: datetime

    # Metrics
    exact_match: bool
    char_similarity: float
    token_similarity: float
    semantic_similarity: Optional[float] = None

    # Detailed comparison
    token_comparisons: List[TokenComparison] = field(default_factory=list)
    diff_segments: List[Dict[str, Any]] = field(default_factory=list)

    # Performance metrics
    response1_time_ms: Optional[float] = None
    response2_time_ms: Optional[float] = None
    response1_tokens: Optional[int] = None
    response2_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "response1": self.response1,
            "response2": self.response2,
            "model1_name": self.model1_name,
            "model2_name": self.model2_name,
            "timestamp": self.timestamp.isoformat(),
            "exact_match": self.exact_match,
            "char_similarity": self.char_similarity,
            "token_similarity": self.token_similarity,
            "semantic_similarity": self.semantic_similarity,
            "response1_time_ms": self.response1_time_ms,
            "response2_time_ms": self.response2_time_ms,
            "response1_tokens": self.response1_tokens,
            "response2_tokens": self.response2_tokens,
        }


class ResponseComparisonView:
    """View for comparing model responses."""

    def __init__(
        self,
        similarity_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_model: Optional[str] = "bert-base-uncased",
    ):
        """Initialize response comparison view.

        Args:
            similarity_model: Model for semantic similarity
            tokenizer_model: Tokenizer for token-level comparison
        """
        self.similarity_model_name = similarity_model
        self.tokenizer_model_name = tokenizer_model

        # Initialize models if available
        self.similarity_model = None
        self.tokenizer = None

        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
                if similarity_model:
                    self.similarity_model = AutoModel.from_pretrained(similarity_model)
            except Exception as e:
                logger.warning(f"Failed to load models: {e}")

        # Comparison history
        self.comparison_history: List[ResponseComparison] = []

    def compare_responses(
        self,
        prompt: str,
        response1: str,
        response2: str,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        response1_time_ms: Optional[float] = None,
        response2_time_ms: Optional[float] = None,
        include_attention: bool = False,
    ) -> ResponseComparison:
        """Compare two model responses.

        Args:
            prompt: Input prompt
            response1: First model's response
            response2: Second model's response
            model1_name: Name of first model
            model2_name: Name of second model
            response1_time_ms: Response time for model 1
            response2_time_ms: Response time for model 2
            include_attention: Whether to include attention weights

        Returns:
            ResponseComparison object
        """
        # Basic metrics
        exact_match = response1 == response2
        char_similarity = self._calculate_char_similarity(response1, response2)

        # Token-level comparison
        token_comparisons, token_similarity = self._compare_tokens(response1, response2)

        # Semantic similarity
        semantic_similarity = None
        if self.similarity_model:
            semantic_similarity = self._calculate_semantic_similarity(response1, response2)

        # Generate diff segments
        diff_segments = self._generate_diff_segments(response1, response2)

        # Count tokens
        response1_tokens = len(response1.split()) if response1 else 0
        response2_tokens = len(response2.split()) if response2 else 0

        comparison = ResponseComparison(
            prompt=prompt,
            response1=response1,
            response2=response2,
            model1_name=model1_name,
            model2_name=model2_name,
            timestamp=datetime.now(),
            exact_match=exact_match,
            char_similarity=char_similarity,
            token_similarity=token_similarity,
            semantic_similarity=semantic_similarity,
            token_comparisons=token_comparisons,
            diff_segments=diff_segments,
            response1_time_ms=response1_time_ms,
            response2_time_ms=response2_time_ms,
            response1_tokens=response1_tokens,
            response2_tokens=response2_tokens,
        )

        # Add to history
        self.comparison_history.append(comparison)

        return comparison

    def _calculate_char_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        matcher = difflib.SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def _compare_tokens(self, text1: str, text2: str) -> Tuple[List[TokenComparison] | float]:
        """Compare texts at token level.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Tuple of (token comparisons, overall similarity)
        """
        # Simple word tokenization if no tokenizer available
        if self.tokenizer:
            tokens1 = self.tokenizer.tokenize(text1)
            tokens2 = self.tokenizer.tokenize(text2)
        else:
            tokens1 = text1.split()
            tokens2 = text2.split()

        comparisons = []
        matcher = difflib.SequenceMatcher(None, tokens1, tokens2)

        matches = 0
        total = max(len(tokens1), len(tokens2))

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    comparisons.append(
                        TokenComparison(
                            token1=tokens1[i],
                            token2=tokens2[j],
                            position=i,
                            match=True,
                            similarity=1.0,
                        )
                    )
                    matches += 1
            elif tag == "replace":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    similarity = self._calculate_char_similarity(tokens1[i], tokens2[j])
                    comparisons.append(
                        TokenComparison(
                            token1=tokens1[i],
                            token2=tokens2[j],
                            position=i,
                            match=False,
                            similarity=similarity,
                        )
                    )
                    matches += similarity
            elif tag == "delete":
                for i in range(i1, i2):
                    comparisons.append(
                        TokenComparison(
                            token1=tokens1[i], token2="", position=i, match=False, similarity=0.0
                        )
                    )
            elif tag == "insert":
                for j in range(j1, j2):
                    comparisons.append(
                        TokenComparison(
                            token1="", token2=tokens2[j], position=j, match=False, similarity=0.0
                        )
                    )

        overall_similarity = matches / total if total > 0 else 0.0

        return comparisons, overall_similarity

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        if not self.similarity_model or not TRANSFORMERS_AVAILABLE:
            return 0.0

        try:
            # Encode texts
            inputs1 = self.tokenizer(text1, return_tensors="pt", truncation=True, max_length=512)
            inputs2 = self.tokenizer(text2, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs1 = self.similarity_model(**inputs1)
                outputs2 = self.similarity_model(**inputs2)

                # Use mean pooling
                embeddings1 = outputs1.last_hidden_state.mean(dim=1)
                embeddings2 = outputs2.last_hidden_state.mean(dim=1)

                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

                return float(similarity.item())
        except Exception as e:
            logger.warning(f"Failed to calculate semantic similarity: {e}")
            return 0.0

    def _generate_diff_segments(self, text1: str, text2: str) -> List[Dict[str, Any]]:
        """Generate diff segments for visualization.

        Args:
            text1: First text
            text2: Second text

        Returns:
            List of diff segments
        """
        segments = []
        differ = difflib.unified_diff(
            text1.splitlines(keepends=True), text2.splitlines(keepends=True)
        )

        for line in differ:
            if line.startswith("+"):
                segments.append({"type": "addition", "text": line[1:].strip(), "side": "right"})
            elif line.startswith("-"):
                segments.append({"type": "deletion", "text": line[1:].strip(), "side": "left"})
            elif not line.startswith(("@@", "++", "--")):
                segments.append({"type": "unchanged", "text": line.strip(), "side": "both"})

        return segments

    def visualize_comparison(
        self, comparison: ResponseComparison, output_format: str = "plotly"
    ) -> go.Figure | str:
        """Visualize response comparison.

        Args:
            comparison: ResponseComparison object
            output_format: Output format ('plotly', 'html', 'streamlit')

        Returns:
            Visualization object
        """
        if output_format == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_visualization(comparison)
        elif output_format == "html":
            return self._create_html_visualization(comparison)
        elif output_format == "streamlit" and STREAMLIT_AVAILABLE:
            return self._create_streamlit_visualization(comparison)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _create_plotly_visualization(self, comparison: ResponseComparison) -> go.Figure:
        """Create Plotly visualization of comparison.

        Args:
            comparison: ResponseComparison object

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                f"{comparison.model1_name} Response",
                f"{comparison.model2_name} Response",
                "Token-Level Comparison",
                "Similarity Metrics",
                "Performance Comparison",
                "Diff View",
            ),
            specs=[
                [{"type": "table"}, {"type": "table"}],
                [{"type": "bar", "colspan": 2}, None],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        # Response tables
        fig.add_trace(
            go.Table(
                header=dict(values=["Response"]),
                cells=dict(values=[[comparison.response1]], align="left", height=30),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Table(
                header=dict(values=["Response"]),
                cells=dict(values=[[comparison.response2]], align="left", height=30),
            ),
            row=1,
            col=2,
        )

        # Token comparison bar chart
        if comparison.token_comparisons:
            tokens = []
            similarities = []
            colors = []

            for tc in comparison.token_comparisons[:50]:  # Limit to first 50 tokens
                tokens.append(f"{tc.token1}|{tc.token2}")
                similarities.append(tc.similarity)
                colors.append("green" if tc.match else "red")

            fig.add_trace(
                go.Bar(x=tokens, y=similarities, marker_color=colors, name="Token Similarity"),
                row=2,
                col=1,
            )

        # Similarity metrics
        metrics = {
            "Character": comparison.char_similarity,
            "Token": comparison.token_similarity,
            "Semantic": comparison.semantic_similarity or 0,
        }

        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=["blue", "green", "purple"],
                name="Similarity Scores",
            ),
            row=3,
            col=1,
        )

        # Performance comparison
        if comparison.response1_time_ms and comparison.response2_time_ms:
            fig.add_trace(
                go.Scatter(
                    x=[comparison.model1_name, comparison.model2_name],
                    y=[comparison.response1_time_ms, comparison.response2_time_ms],
                    mode="markers+lines",
                    marker=dict(size=10),
                    name="Response Time (ms)",
                ),
                row=3,
                col=2,
            )

        fig.update_layout(
            title=f"Response Comparison: {comparison.prompt[:50]}...", height=800, showlegend=False
        )

        return fig

    def _create_html_visualization(self, comparison: ResponseComparison) -> str:
        """Create HTML visualization of comparison.

        Args:
            comparison: ResponseComparison object

        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Response Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ display: flex; gap: 20px; }}
                .response {{ flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .response h3 {{ margin-top: 0; }}
                .match {{ background-color: #d4edda; }}
                .diff {{ background-color: #f8d7da; }}
                .metrics {{ margin-top: 20px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
                .prompt {{ background: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Response Comparison</h1>

            <div class="prompt">
                <strong>Prompt:</strong> {comparison.prompt}
            </div>

            <div class="container">
                <div class="response">
                    <h3>{comparison.model1_name}</h3>
                    <p>{comparison.response1}</p>
                    <small>Time: {comparison.response1_time_ms:.1f}ms | Tokens: {comparison.response1_tokens}</small>
                </div>

                <div class="response">
                    <h3>{comparison.model2_name}</h3>
                    <p>{comparison.response2}</p>
                    <small>Time: {comparison.response2_time_ms:.1f}ms | Tokens: {comparison.response2_tokens}</small>
                </div>
            </div>

            <div class="metrics">
                <h3>Similarity Metrics</h3>
                <div class="metric">
                    <strong>Exact Match:</strong> {"Yes" if comparison.exact_match else "No"}
                </div>
                <div class="metric">
                    <strong>Character Similarity:</strong> {comparison.char_similarity:.2%}
                </div>
                <div class="metric">
                    <strong>Token Similarity:</strong> {comparison.token_similarity:.2%}
                </div>
                <div class="metric">
                    <strong>Semantic Similarity:</strong> {comparison.semantic_similarity:.2% if comparison.semantic_similarity else 'N/A'}
                </div>
            </div>

            <div class="diff-view">
                <h3>Differences</h3>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px;">
        """

        # Add diff segments
        for segment in comparison.diff_segments[:20]:  # Limit to first 20 segments
            if segment["type"] == "addition":
                html += f'<span class="diff">+ {segment["text"]}</span><br>'
            elif segment["type"] == "deletion":
                html += f'<span class="diff">- {segment["text"]}</span><br>'
            else:
                html += f'<span class="match">{segment["text"]}</span><br>'

        html += """
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _create_streamlit_visualization(self, comparison: ResponseComparison):
        """Create Streamlit visualization of comparison.

        Args:
            comparison: ResponseComparison object
        """
        st.header("Response Comparison")

        # Prompt
        st.info(f"**Prompt:** {comparison.prompt}")

        # Side-by-side responses
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(comparison.model1_name)
            st.text_area(
                "Response", comparison.response1, height=200, disabled=True, key="response1"
            )

            if comparison.response1_time_ms:
                st.metric("Response Time", f"{comparison.response1_time_ms:.1f} ms")
            st.metric("Token Count", comparison.response1_tokens)

        with col2:
            st.subheader(comparison.model2_name)
            st.text_area(
                "Response", comparison.response2, height=200, disabled=True, key="response2"
            )

            if comparison.response2_time_ms:
                st.metric("Response Time", f"{comparison.response2_time_ms:.1f} ms")
            st.metric("Token Count", comparison.response2_tokens)

        # Similarity metrics
        st.subheader("Similarity Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Exact Match", "✅" if comparison.exact_match else "❌")

        with col2:
            st.metric("Character Similarity", f"{comparison.char_similarity:.1%}")

        with col3:
            st.metric("Token Similarity", f"{comparison.token_similarity:.1%}")

        with col4:
            if comparison.semantic_similarity:
                st.metric("Semantic Similarity", f"{comparison.semantic_similarity:.1%}")
            else:
                st.metric("Semantic Similarity", "N/A")

        # Diff view
        st.subheader("Differences")

        if comparison.diff_segments:
            diff_html = "<div style='background: #f8f9fa; padding: 10px; border-radius: 5px;'>"

            for segment in comparison.diff_segments[:20]:
                if segment["type"] == "addition":
                    diff_html += (
                        f"<span style='background: #d4edda;'>+ {segment['text']}</span><br>"
                    )
                elif segment["type"] == "deletion":
                    diff_html += (
                        f"<span style='background: #f8d7da;'>- {segment['text']}</span><br>"
                    )
                else:
                    diff_html += f"<span>{segment['text']}</span><br>"

            diff_html += "</div>"
            st.markdown(diff_html, unsafe_allow_html=True)
        else:
            st.info("No differences found")

        # Token comparison chart
        if comparison.token_comparisons and PLOTLY_AVAILABLE:
            st.subheader("Token-Level Comparison")

            tokens = []
            similarities = []

            for tc in comparison.token_comparisons[:30]:
                tokens.append(f"{tc.token1}|{tc.token2}")
                similarities.append(tc.similarity)

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=tokens,
                        y=similarities,
                        marker_color=[
                            "green" if s == 1.0 else "orange" if s > 0.5 else "red"
                            for s in similarities
                        ],
                    )
                ]
            )

            fig.update_layout(
                title="Token Similarity Scores",
                xaxis_title="Token Pairs",
                yaxis_title="Similarity",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    def export_comparison(
        self, comparison: ResponseComparison, format: str = "json"
    ) -> str | bytes:
        """Export comparison data.

        Args:
            comparison: ResponseComparison object
            format: Export format ('json', 'html')

        Returns:
            Exported data
        """
        if format == "json":
            return json.dumps(comparison.to_dict(), indent=2)
        elif format == "html":
            return self._create_html_visualization(comparison)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = ResponseComparisonView()

    # Compare responses
    comparison = analyzer.compare_responses(
        prompt="What is the capital of France?",
        response1="The capital of France is Paris.",
        response2="Paris is the capital city of France.",
        model1_name="GPT-3.5",
        model2_name="Claude",
        response1_time_ms=150.5,
        response2_time_ms=120.3,
    )

    print(f"Exact match: {comparison.exact_match}")
    print(f"Character similarity: {comparison.char_similarity:.2%}")
    print(f"Token similarity: {comparison.token_similarity:.2%}")

    # Create visualization
    if PLOTLY_AVAILABLE:
        fig = analyzer.visualize_comparison(comparison, output_format="plotly")
        fig.show()

    # Export comparison
    json_export = analyzer.export_comparison(comparison, format="json")
    print("\nExported JSON:")
    print(json_export[:500])

    # Streamlit app
    if STREAMLIT_AVAILABLE:
        analyzer._create_streamlit_visualization(comparison)
