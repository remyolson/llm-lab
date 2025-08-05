"""
Model Comparison Components

This module provides interactive interfaces for comparing model performance
before and after fine-tuning.
"""

from .comparison_view import (
    ComparisonView,
    ComparisonConfig,
    create_streamlit_app,
    create_gradio_interface
)

__all__ = [
    "ComparisonView",
    "ComparisonConfig",
    "create_streamlit_app",
    "create_gradio_interface"
]