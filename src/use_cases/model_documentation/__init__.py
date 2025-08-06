"""
Automated Model Documentation System - Use Case 11

This module provides automated generation of comprehensive documentation for ML models,
including model cards, compliance reports, and technical specifications.

Key Components:
- Automated model card generation with standardized templates
- Model metadata extraction and analysis
- Compliance report generation for various standards
- Template engine with customizable documentation formats
- Integration with popular ML model formats and registries

Usage:
    from src.use_cases.model_documentation import ModelCardGenerator

    generator = ModelCardGenerator()
    model_card = await generator.generate_model_card(model_path)
"""

__version__ = "1.0.0"
__author__ = "LLM Lab Team"

# Re-export main components for easy access
try:
    from .src.model_docs.analyzers.model_inspector import ModelInspector
    from .src.model_docs.generators.model_card_generator import ModelCardGenerator
    from .src.model_docs.generators.template_engine import TemplateEngine
    from .src.model_docs.models import ModelMetadata

    __all__ = ["ModelCardGenerator", "ModelInspector", "TemplateEngine", "ModelMetadata"]
except ImportError:
    # Handle graceful import failures for development
    __all__ = []
