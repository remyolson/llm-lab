"""
Configuration for integration tests

This module provides configuration settings for integration tests,
including API endpoints, test models, timeouts, and environment setup.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests."""
    
    # Provider-specific settings
    PROVIDER_CONFIGS = {
        'openai': {
            'env_var': 'OPENAI_API_KEY',
            'test_models': ['gpt-3.5-turbo', 'gpt-4o-mini'],
            'default_model': 'gpt-3.5-turbo',
            'rate_limit': 3500,  # RPM
            'rate_limit_delay': 1.0,  # seconds between requests
            'timeout': 30.0,
            'max_retries': 3
        },
        'anthropic': {
            'env_var': 'ANTHROPIC_API_KEY',
            'test_models': ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229'],
            'default_model': 'claude-3-haiku-20240307',
            'rate_limit': 1000,  # RPM
            'rate_limit_delay': 1.2,
            'timeout': 45.0,
            'max_retries': 3
        },
        'google': {
            'env_var': 'GOOGLE_API_KEY',
            'test_models': ['gemini-1.5-flash', 'gemini-1.5-pro'],
            'default_model': 'gemini-1.5-flash',
            'rate_limit': 1500,  # RPM
            'rate_limit_delay': 1.0,
            'timeout': 30.0,
            'max_retries': 3
        }
    }
    
    # Test environment flags
    INTEGRATION_FLAGS = {
        'openai': 'TEST_OPENAI_INTEGRATION',
        'anthropic': 'TEST_ANTHROPIC_INTEGRATION',
        'google': 'TEST_GOOGLE_INTEGRATION',
        'all': 'TEST_ALL_PROVIDERS_INTEGRATION'
    }
    
    # Test prompts for different categories
    TEST_PROMPTS = {
        'simple': [
            "What is 2 + 2?",
            "What color is the sky?",
            "Say hello in French.",
            "Name three planets.",
            "What day comes after Monday?"
        ],
        'reasoning': [
            "If I have 5 apples and give away 2, how many do I have left?",
            "Which is larger: 15 or 8?",
            "What comes next in this sequence: 2, 4, 6, 8, ?",
            "If it's raining, should I take an umbrella?",
            "What's the capital of France?"
        ],
        'creative': [
            "Write a haiku about technology.",
            "Create a short story about a robot.",
            "Describe a sunset in poetic language.",
            "Invent a new ice cream flavor.",
            "Write a limerick about coding."
        ],
        'factual': [
            "When was the first moon landing?",
            "What is the chemical symbol for gold?",
            "Who wrote Romeo and Juliet?",
            "What is the largest ocean on Earth?",
            "In what year did World War II end?"
        ],
        'technical': [
            "Explain what HTTP stands for.",
            "What is the difference between Python and JavaScript?",
            "How does a computer store data?",
            "What is machine learning?",
            "Explain what an API is."
        ]
    }
    
    # Performance benchmarks
    PERFORMANCE_EXPECTATIONS = {
        'max_response_time': {
            'simple': 5.0,  # seconds
            'reasoning': 8.0,
            'creative': 10.0,
            'factual': 6.0,
            'technical': 8.0
        },
        'min_response_length': {
            'simple': 1,  # characters
            'reasoning': 10,
            'creative': 20,
            'factual': 10,
            'technical': 20
        },
        'max_response_length': {
            'simple': 500,
            'reasoning': 1000,
            'creative': 2000,
            'factual': 800,
            'technical': 1500
        }
    }
    
    # Test execution settings
    EXECUTION_SETTINGS = {
        'max_concurrent_requests': 3,
        'default_timeout': 30.0,
        'retry_delays': [1.0, 2.0, 4.0],  # Progressive backoff
        'rate_limit_buffer': 0.1,  # Extra delay buffer
        'test_output_dir': 'test_results/integration',
        'save_responses': True,
        'save_errors': True
    }
    
    @classmethod
    def is_provider_enabled(cls, provider: str) -> bool:
        """Check if integration tests are enabled for a provider."""
        provider_flag = cls.INTEGRATION_FLAGS.get(provider.lower())
        all_flag = cls.INTEGRATION_FLAGS.get('all')
        
        if provider_flag and os.getenv(provider_flag, '').lower() == 'true':
            return True
        
        if all_flag and os.getenv(all_flag, '').lower() == 'true':
            return True
        
        return False
    
    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        config = cls.PROVIDER_CONFIGS.get(provider.lower())
        if not config:
            return None
        
        env_var = config['env_var']
        return os.getenv(env_var)
    
    @classmethod
    def has_valid_api_key(cls, provider: str) -> bool:
        """Check if provider has a valid API key."""
        api_key = cls.get_api_key(provider)
        
        if not api_key:
            return False
        
        # Basic validation
        if provider.lower() == 'openai':
            return api_key.startswith('sk-')
        elif provider.lower() == 'anthropic':
            return len(api_key) > 10  # Basic length check
        elif provider.lower() == 'google':
            return len(api_key) > 10  # Basic length check
        
        return True
    
    @classmethod
    def get_test_model(cls, provider: str, model_override: Optional[str] = None) -> str:
        """Get the test model for a provider."""
        if model_override:
            return model_override
        
        # Check for environment override
        env_var = f'INTEGRATION_MODEL_{provider.upper()}'
        env_model = os.getenv(env_var)
        if env_model:
            return env_model
        
        # Use default from config
        config = cls.PROVIDER_CONFIGS.get(provider.lower(), {})
        return config.get('default_model', 'unknown-model')
    
    @classmethod
    def get_provider_config(cls, provider: str) -> Dict[str, Any]:
        """Get full configuration for a provider."""
        return cls.PROVIDER_CONFIGS.get(provider.lower(), {})
    
    @classmethod
    def get_test_prompts(cls, category: str = 'simple') -> List[str]:
        """Get test prompts for a category."""
        return cls.TEST_PROMPTS.get(category, cls.TEST_PROMPTS['simple'])
    
    @classmethod
    def should_run_slow_tests(cls) -> bool:
        """Check if slow integration tests should be run."""
        return os.getenv('RUN_SLOW_INTEGRATION_TESTS', '').lower() == 'true'
    
    @classmethod
    def should_run_expensive_tests(cls) -> bool:
        """Check if expensive (high token) tests should be run."""
        return os.getenv('RUN_EXPENSIVE_INTEGRATION_TESTS', '').lower() == 'true'
    
    @classmethod
    def get_output_directory(cls) -> str:
        """Get output directory for test results."""
        base_dir = cls.EXECUTION_SETTINGS['test_output_dir']
        override = os.getenv('INTEGRATION_TEST_OUTPUT_DIR')
        return override or base_dir
    
    @classmethod
    def get_performance_expectation(cls, category: str, metric: str) -> float:
        """Get performance expectation for a category and metric."""
        expectations = cls.PERFORMANCE_EXPECTATIONS.get(metric, {})
        return expectations.get(category, expectations.get('simple', 5.0))


# Environment setup helper
def setup_integration_environment():
    """Setup environment for integration tests."""
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    output_dir = IntegrationTestConfig.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for required environment variables
    missing_keys = []
    enabled_providers = []
    
    for provider in ['openai', 'anthropic', 'google']:
        if IntegrationTestConfig.is_provider_enabled(provider):
            enabled_providers.append(provider)
            if not IntegrationTestConfig.has_valid_api_key(provider):
                config = IntegrationTestConfig.get_provider_config(provider)
                missing_keys.append(config.get('env_var', f'{provider.upper()}_API_KEY'))
    
    if enabled_providers:
        print(f"Integration tests enabled for: {', '.join(enabled_providers)}")
    
    if missing_keys:
        print(f"Warning: Missing API keys for enabled providers: {', '.join(missing_keys)}")
        print("Tests for these providers will be skipped.")
    
    return enabled_providers, missing_keys


# Pytest fixtures for integration tests
def pytest_configure():
    """Configure pytest markers for integration tests."""
    import pytest
    
    # Add custom markers
    pytest.mark.integration_openai = pytest.mark.integration_openai
    pytest.mark.integration_anthropic = pytest.mark.integration_anthropic
    pytest.mark.integration_google = pytest.mark.integration_google
    pytest.mark.integration_all = pytest.mark.integration_all
    pytest.mark.slow_integration = pytest.mark.slow_integration
    pytest.mark.expensive_integration = pytest.mark.expensive_integration