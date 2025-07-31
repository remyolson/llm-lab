"""
Compatibility testing module for LLM providers

This module provides comprehensive compatibility testing to ensure consistent
behavior across different LLM providers and identify provider-specific quirks.

Key Features:
- Cross-provider compatibility testing
- Parameter compatibility validation
- Unicode and special character handling
- Edge case prompt handling
- Concurrent request compatibility
- Feature compatibility testing (streaming, token counting, etc.)
- Automated compatibility reporting

Usage:
    pytest tests/compatibility/test_provider_compatibility.py -v
    
    # Run compatibility tests for specific providers
    pytest tests/compatibility/ -k "openai or anthropic" -v
    
    # Generate compatibility report
    pytest tests/compatibility/test_provider_compatibility.py::test_compatibility_report_generation -v -s
"""

from .test_provider_compatibility import (
    TestProviderCompatibility,
    TestProviderFeatureCompatibility
)

__all__ = [
    'TestProviderCompatibility',
    'TestProviderFeatureCompatibility'
]