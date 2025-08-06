"""
LLM Security Testing Framework - Use Case 9

This module provides comprehensive security testing capabilities for Large Language Models,
including vulnerability detection, attack resistance testing, and security assessment reporting.

Key Components:
- Attack library with 500+ categorized security tests
- Multi-strategy vulnerability detection (rule-based, ML-based, heuristic)
- Response analysis engine with pattern matching and sentiment analysis
- Confidence scoring system with multi-factor analysis
- Parallel scanning with intelligent batching and cancellation support
- Comprehensive reporting and compliance tools

Usage:
    from src.use_cases.security_testing import SecurityScanner

    scanner = SecurityScanner()
    results = await scanner.scan_model(model_interface)
"""

__version__ = "1.0.0"
__author__ = "LLM Lab Team"

# Re-export main components for easy access
try:
    from .src.attack_library.analytics.analytics_engine import AnalyticsEngine
    from .src.attack_library.core.library import AttackLibrary
    from .src.attack_library.security.models import ScanResult, VulnerabilityType
    from .src.attack_library.security.scanner import SecurityScanner

    __all__ = [
        "SecurityScanner",
        "ScanResult",
        "VulnerabilityType",
        "AttackLibrary",
        "AnalyticsEngine",
    ]
except ImportError:
    # Handle graceful import failures for development
    __all__ = []
