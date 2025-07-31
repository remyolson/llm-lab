"""
Test configuration for provider testing

This module contains configuration settings for running provider tests,
including integration test settings, performance benchmarks, and test data.
"""

import os
from typing import Dict, Any


class TestConfig:
    """Configuration for provider tests."""
    
    # Integration test environment variables
    INTEGRATION_TEST_ENV_VARS = {
        'openai': 'TEST_OPENAI_INTEGRATION',
        'anthropic': 'TEST_ANTHROPIC_INTEGRATION', 
        'google': 'TEST_GOOGLE_INTEGRATION',
        'all': 'TEST_ALL_PROVIDERS_INTEGRATION'
    }
    
    # Models to use for integration testing (cheaper/faster models)
    INTEGRATION_TEST_MODELS = {
        'openai': 'gpt-3.5-turbo',
        'anthropic': 'claude-3-haiku-20240307',
        'google': 'gemini-1.5-flash'
    }
    
    # Performance test settings
    PERFORMANCE_SETTINGS = {
        'num_requests': 5,
        'max_avg_response_time': 3.0,  # seconds
        'max_single_response_time': 10.0,  # seconds
        'parallel_requests': 3
    }
    
    # Rate limit test settings
    RATE_LIMIT_SETTINGS = {
        'requests_per_minute': 60,
        'burst_size': 10,
        'retry_delay': 1.0  # seconds
    }
    
    # Test data for various scenarios
    TEST_PROMPTS = {
        'simple': [
            "What is 2 + 2?",
            "What color is the sky?",
            "Is water wet?",
            "Count to 5.",
            "Say hello."
        ],
        'complex': [
            "Explain quantum entanglement in simple terms.",
            "Write a haiku about machine learning.",
            "What are the pros and cons of renewable energy?",
            "Describe the process of photosynthesis.",
            "Compare and contrast TCP and UDP protocols."
        ],
        'code': [
            "Write a Python function to reverse a string.",
            "Create a JavaScript arrow function that filters even numbers.",
            "Show a SQL query to find duplicate records.",
            "Write a bash script to count files in a directory.",
            "Create a CSS class for a centered div."
        ],
        'edge_cases': [
            "",  # Empty prompt
            " ",  # Whitespace only
            "a" * 1000,  # Long prompt
            "What is 2 + 2? " * 100,  # Repeated prompt
            "🌍 Unicode test 你好",  # Unicode characters
            "Test\nwith\nmultiple\nlines",  # Multiline
            "Test with special chars: @#$%^&*()",  # Special characters
        ]
    }
    
    # Expected behaviors for edge cases
    EDGE_CASE_EXPECTATIONS = {
        'empty_prompt': {
            'should_raise': True,
            'error_type': 'ProviderError',
            'error_contains': ['empty', 'prompt', 'required']
        },
        'whitespace_prompt': {
            'should_raise': True,
            'error_type': 'ProviderError',
            'error_contains': ['empty', 'prompt', 'required']
        },
        'very_long_prompt': {
            'should_raise': False,  # Most providers handle this gracefully
            'truncates': True,
            'max_length': 100000
        },
        'unicode_prompt': {
            'should_raise': False,
            'preserves_unicode': True
        }
    }
    
    # Timeout settings for different test types
    TIMEOUT_SETTINGS = {
        'unit_test': 5.0,  # seconds
        'integration_test': 30.0,  # seconds
        'performance_test': 60.0,  # seconds
        'stress_test': 300.0  # seconds
    }
    
    # Coverage requirements
    COVERAGE_REQUIREMENTS = {
        'minimum_coverage': 80,  # percentage
        'minimum_branch_coverage': 70,  # percentage
        'excluded_files': [
            '*/tests/*',
            '*/test_*',
            '*/__pycache__/*'
        ]
    }
    
    @classmethod
    def is_integration_enabled(cls, provider: str = None) -> bool:
        """Check if integration tests are enabled for a provider."""
        if provider:
            env_var = cls.INTEGRATION_TEST_ENV_VARS.get(provider)
            if env_var:
                return os.getenv(env_var, '').lower() == 'true'
        
        # Check if all integration tests are enabled
        return os.getenv(cls.INTEGRATION_TEST_ENV_VARS['all'], '').lower() == 'true'
    
    @classmethod
    def get_test_model(cls, provider: str) -> str:
        """Get the test model for a provider."""
        return cls.INTEGRATION_TEST_MODELS.get(provider, 'default-model')
    
    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Get performance test configuration."""
        config = cls.PERFORMANCE_SETTINGS.copy()
        
        # Allow overrides from environment
        if os.getenv('PERF_TEST_NUM_REQUESTS'):
            config['num_requests'] = int(os.getenv('PERF_TEST_NUM_REQUESTS'))
        
        if os.getenv('PERF_TEST_MAX_TIME'):
            config['max_avg_response_time'] = float(os.getenv('PERF_TEST_MAX_TIME'))
        
        return config
    
    @classmethod
    def should_run_slow_tests(cls) -> bool:
        """Check if slow tests should be run."""
        return os.getenv('RUN_SLOW_TESTS', '').lower() == 'true'
    
    @classmethod
    def get_test_output_dir(cls) -> str:
        """Get directory for test outputs."""
        return os.getenv('TEST_OUTPUT_DIR', 'test_outputs')

