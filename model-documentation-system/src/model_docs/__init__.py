"""
Model Documentation System

Automated documentation generation for machine learning models.
"""

__version__ = "0.1.0"
__author__ = "LLM Lab Team"
__email__ = "team@llm-lab.io"

from .analyzers.metadata_extractor import MetadataExtractor
from .analyzers.model_inspector import ModelInspector
from .generators.compliance_generator import ComplianceGenerator
from .generators.model_card_generator import ModelCardGenerator

__all__ = [
    "ModelInspector",
    "MetadataExtractor",
    "ModelCardGenerator",
    "ComplianceGenerator",
]
