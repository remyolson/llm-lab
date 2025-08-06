"""
Synthetic Data Generation Platform - Use Case 10

This module provides advanced synthetic data generation capabilities for LLM training,
testing, and evaluation, with privacy preservation and domain-specific customization.

Key Components:
- Multi-domain synthetic data generators (medical, legal, financial, educational, code, ecommerce)
- Privacy-preserving data generation with differential privacy
- Data validation and quality assessment
- Configurable generation parameters and templates
- Integration with popular ML frameworks

Usage:
    from src.use_cases.synthetic_data import SyntheticDataGenerator

    generator = SyntheticDataGenerator()
    data = await generator.generate_medical_data(num_samples=1000)
"""

__version__ = "1.0.0"
__author__ = "LLM Lab Team"

# Re-export main components for easy access
try:
    from .src.synthetic_data.core.config import GeneratorConfig
    from .src.synthetic_data.core.generator import SyntheticDataGenerator
    from .src.synthetic_data.core.privacy import PrivacyEngine
    from .src.synthetic_data.core.validator import DataValidator

    __all__ = ["SyntheticDataGenerator", "GeneratorConfig", "DataValidator", "PrivacyEngine"]
except ImportError:
    # Handle graceful import failures for development
    __all__ = []
