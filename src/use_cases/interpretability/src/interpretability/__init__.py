"""
LLM Interpretability Suite

Comprehensive toolkit for understanding and visualizing Large Language Model behavior.
"""

__version__ = "0.1.0"
__author__ = "LLM Lab Team"
__email__ = "team@llm-lab.io"

from .analyzers import ActivationAnalyzer, AttentionAnalyzer, GradientAnalyzer
from .explanations import ExplanationGenerator, FeatureAttributor
from .visualizers import AttentionVisualizer, DashboardManager

__all__ = [
    "AttentionAnalyzer",
    "GradientAnalyzer",
    "ActivationAnalyzer",
    "AttentionVisualizer",
    "DashboardManager",
    "ExplanationGenerator",
    "FeatureAttributor",
]
