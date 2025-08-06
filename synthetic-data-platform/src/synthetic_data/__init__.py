"""
Synthetic Data Generation Platform

A comprehensive platform for generating high-quality synthetic data using LLMs.
"""

__version__ = "0.1.0"
__author__ = "LLM Lab Team"
__email__ = "team@llm-lab.io"

from .core.generator import SyntheticDataGenerator
from .core.privacy import PrivacyPreserver
from .core.validator import DataValidator
from .generators import (
    CodeDataGenerator,
    EcommerceDataGenerator,
    EducationalDataGenerator,
    FinancialDataGenerator,
    LegalDataGenerator,
    MedicalDataGenerator,
)

__all__ = [
    "SyntheticDataGenerator",
    "DataValidator",
    "PrivacyPreserver",
    "MedicalDataGenerator",
    "FinancialDataGenerator",
    "LegalDataGenerator",
    "EcommerceDataGenerator",
    "EducationalDataGenerator",
    "CodeDataGenerator",
]
