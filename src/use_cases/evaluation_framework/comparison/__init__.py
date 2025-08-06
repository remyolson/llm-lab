"""
Model Comparison Components

This module provides interactive interfaces for comparing model performance
before and after fine-tuning.
"""

from .comparison_view import (
    ComparisonConfig,
    ComparisonView,
    create_gradio_interface,
    create_streamlit_app,
)

__all__ = ["ComparisonConfig", "ComparisonView", "create_gradio_interface", "create_streamlit_app"]
