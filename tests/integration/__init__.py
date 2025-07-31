"""
Integration tests for LLM providers

This module contains integration tests that make real API calls to test providers
against their actual APIs. These tests are optional and require valid API keys
and network connectivity.

To run integration tests:
    # All providers
    TEST_ALL_PROVIDERS_INTEGRATION=true pytest tests/integration/
    
    # Single provider
    TEST_OPENAI_INTEGRATION=true pytest tests/integration/test_openai_integration.py
    
    # With specific models
    INTEGRATION_MODEL_OPENAI=gpt-3.5-turbo pytest tests/integration/
    
Environment variables required:
    - OPENAI_API_KEY: For OpenAI tests
    - ANTHROPIC_API_KEY: For Anthropic tests  
    - GOOGLE_API_KEY: For Google tests
    - TEST_*_INTEGRATION: To enable specific provider tests
"""

from .integration_runner import IntegrationTestRunner
from .test_config import IntegrationTestConfig

__all__ = ['IntegrationTestRunner', 'IntegrationTestConfig']