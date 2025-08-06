"""
LLM Interpretability Suite - Use Case 12

This module provides comprehensive interpretability and explainability tools for
Large Language Models, including activation analysis, attention visualization,
and feature attribution.

Key Components:
- Activation pattern analysis and visualization
- Attention mechanism interpretation and visualization
- Gradient-based feature attribution
- Hook management for model introspection
- Interactive dashboard for interpretability exploration
- Explanation generation for model decisions

Usage:
    from src.use_cases.interpretability import ActivationAnalyzer, AttentionAnalyzer

    analyzer = ActivationAnalyzer(model)
    activations = await analyzer.analyze_activations(input_text)
"""

__version__ = "1.0.0"
__author__ = "LLM Lab Team"

# Re-export main components for easy access
try:
    from .src.interpretability.analyzers.activation_analyzer import ActivationAnalyzer
    from .src.interpretability.analyzers.attention_analyzer import AttentionAnalyzer
    from .src.interpretability.analyzers.gradient_analyzer import GradientAnalyzer
    from .src.interpretability.explanations.explanation_generator import ExplanationGenerator

    __all__ = [
        "ActivationAnalyzer",
        "AttentionAnalyzer",
        "GradientAnalyzer",
        "ExplanationGenerator",
    ]
except ImportError:
    # Handle graceful import failures for development
    __all__ = []
