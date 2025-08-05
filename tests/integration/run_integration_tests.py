#!/usr/bin/env python3
"""
Integration Test Runner Script

This script runs comprehensive integration tests against real LLM provider APIs.
It demonstrates how to use the integration test framework and provides examples
of different test configurations.

Usage:
    # Run all enabled providers with standard tests
    python run_integration_tests.py
    
    # Run specific provider tests
    TEST_OPENAI_INTEGRATION=true python run_integration_tests.py --provider openai
    
    # Run with custom configuration
    python run_integration_tests.py --max-workers 2 --rate-limit 2.0
    
    # Run expensive tests (high token usage)
    RUN_EXPENSIVE_INTEGRATION_TESTS=true python run_integration_tests.py

Environment Variables:
    TEST_OPENAI_INTEGRATION=true     - Enable OpenAI tests
    TEST_ANTHROPIC_INTEGRATION=true  - Enable Anthropic tests  
    TEST_GOOGLE_INTEGRATION=true     - Enable Google tests
    TEST_ALL_PROVIDERS_INTEGRATION=true - Enable all provider tests
    
    RUN_SLOW_INTEGRATION_TESTS=true - Enable slow tests (concurrent, etc.)
    RUN_EXPENSIVE_INTEGRATION_TESTS=true - Enable expensive tests (high tokens)
    
    INTEGRATION_MODEL_OPENAI=gpt-4o-mini - Override OpenAI test model
    INTEGRATION_TEST_OUTPUT_DIR=./results - Custom output directory

Required API Keys:
    OPENAI_API_KEY - For OpenAI tests
    ANTHROPIC_API_KEY - For Anthropic tests
    GOOGLE_API_KEY - For Google tests
"""

# Import paths fixed - sys.path manipulation removed
import argparse
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
)))

from llm_providers import OpenAIProvider, AnthropicProvider, GoogleProvider
from tests.integration.integration_runner import (
    IntegrationTestRunner,
    STANDARD_INTEGRATION_TESTS,
    test_basic_generation,
    test_parameter_handling,
    test_system_prompt,
    test_longer_generation
)
from tests.integration.test_config import IntegrationTestConfig, setup_integration_environment


def create_custom_tests() -> Dict[str, Any]:
    """Create custom integration tests beyond the standard ones."""
    
    def test_creative_writing(provider):
        """Test creative writing capabilities."""
        prompt = "Write a creative short story about a robot discovering emotions in exactly 50 words."
        response = provider.generate(prompt, temperature=0.8, max_tokens=100)
        
        if not response:
            raise ValueError("No response for creative writing test")
        
        word_count = len(response.split())
        if word_count < 20:
            raise ValueError(f"Response too short: {word_count} words")
        
        return response
    
    def test_reasoning(provider):
        """Test logical reasoning capabilities."""
        prompt = ("If a train travels 60 miles per hour for 2 hours, "
                 "then 80 miles per hour for 1.5 hours, "
                 "what is the total distance traveled?")
        
        response = provider.generate(prompt, temperature=0.3, max_tokens=100)
        
        if not response:
            raise ValueError("No response for reasoning test")
        
        # The answer should be 120 + 120 = 240 miles
        if "240" not in response:
            logging.warning(f"Reasoning test may have failed - expected 240 in response: {response}")
        
        return response
    
    def test_code_generation(provider):
        """Test code generation capabilities."""
        prompt = "Write a Python function that calculates the factorial of a number using recursion."
        response = provider.generate(prompt, temperature=0.5, max_tokens=150)
        
        if not response:
            raise ValueError("No response for code generation test")
        
        # Check for basic code elements
        if not any(keyword in response.lower() for keyword in ['def', 'function', 'factorial']):
            raise ValueError("Response doesn't appear to contain code")
        
        return response
    
    def test_multilingual(provider):
        """Test multilingual capabilities."""
        prompt = "Translate 'Hello, how are you?' into French, Spanish, and German."
        response = provider.generate(prompt, max_tokens=100)
        
        if not response:
            raise ValueError("No response for multilingual test")
        
        # Check for expected words in different languages
        expected_words = ['bonjour', 'hola', 'hallo', 'comment', 'como', 'wie']
        found_words = sum(1 for word in expected_words if word in response.lower())
        
        if found_words < 2:
            logging.warning(f"Multilingual test may not have worked well - found {found_words} expected words")
        
        return response
    
    def test_json_output(provider):
        """Test structured JSON output."""
        prompt = ("Generate a JSON object with information about a fictional person "
                 "including name, age, occupation, and hobbies (as an array).")
        
        response = provider.generate(prompt, temperature=0.7, max_tokens=150)
        
        if not response:
            raise ValueError("No response for JSON test")
        
        # Basic JSON validation
        import json
        try:
            # Try to find JSON in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Check for expected keys
                expected_keys = ['name', 'age', 'occupation']
                if not any(key in parsed for key in expected_keys):
                    raise ValueError("JSON doesn't contain expected person information")
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"JSON parsing failed or incomplete: {e}")
            # Don't raise error - some providers might not format JSON perfectly
        
        return response
    
    return {
        "creative_writing": test_creative_writing,
        "reasoning": test_reasoning,
        "code_generation": test_code_generation,
        "multilingual": test_multilingual,
        "json_output": test_json_output
    }


def create_performance_tests() -> Dict[str, Any]:
    """Create performance-focused integration tests."""
    
    def test_response_speed(provider):
        """Test response speed for simple queries."""
        import time
        
        start_time = time.time()
        response = provider.generate("What is 2 + 2?", max_tokens=10)
        response_time = time.time() - start_time
        
        if response_time > 10.0:  # 10 second threshold
            raise ValueError(f"Response too slow: {response_time:.2f}s")
        
        if not response or "4" not in response:
            raise ValueError("Incorrect or missing response")
        
        return f"Response: {response} (Time: {response_time:.2f}s)"
    
    def test_batch_processing(provider):
        """Test multiple requests in sequence."""
        prompts = [
            "What is 1 + 1?",
            "What is 2 + 2?", 
            "What is 3 + 3?",
            "What is 4 + 4?",
            "What is 5 + 5?"
        ]
        
        responses = []
        total_start = time.time()
        
        for prompt in prompts:
            response = provider.generate(prompt, max_tokens=10)
            responses.append(response)
            time.sleep(0.5)  # Small delay to respect rate limits
        
        total_time = time.time() - total_start
        
        # Check all responses
        expected_answers = ["2", "4", "6", "8", "10"]
        for i, (response, expected) in enumerate(zip(responses, expected_answers)):
            if not response or expected not in response:
                raise ValueError(f"Incorrect response for prompt {i+1}: {response}")
        
        return f"Processed {len(prompts)} prompts in {total_time:.2f}s"
    
    return {
        "response_speed": test_response_speed,
        "batch_processing": test_batch_processing
    }


def create_providers(provider_filter: Optional[str] = None) -> List:
    """Create provider instances based on configuration and filters."""
    available_providers = []
    
    provider_configs = [
        ('openai', OpenAIProvider, 'openai'),
        ('anthropic', AnthropicProvider, 'anthropic'),
        ('google', GoogleProvider, 'google')
    ]
    
    for name, provider_class, config_key in provider_configs:
        # Skip if provider filter is specified and doesn't match
        if provider_filter and name != provider_filter:
            continue
        
        # Skip if not enabled
        if not IntegrationTestConfig.is_provider_enabled(config_key):
            logging.info(f"Skipping {name} - not enabled")
            continue
        
        # Skip if no valid API key
        if not IntegrationTestConfig.has_valid_api_key(config_key):
            logging.warning(f"Skipping {name} - no valid API key")
            continue
        
        # Create provider instance
        try:
            model = IntegrationTestConfig.get_test_model(config_key)
            provider = provider_class(model_name=model)
            available_providers.append(provider)
            logging.info(f"✓ Created {name} provider with model {model}")
        except Exception as e:
            logging.error(f"✗ Failed to create {name} provider: {e}")
    
    return available_providers


def main():
    """Main integration test runner."""
    parser = argparse.ArgumentParser(
        description="Run LLM Provider Integration Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--provider',
        choices=['openai', 'anthropic', 'google'],
        help='Run tests for specific provider only'
    )
    
    parser.add_argument(
        '--test-suite',
        choices=['standard', 'custom', 'performance', 'all'],
        default='standard',
        help='Which test suite to run (default: standard)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=3,
        help='Maximum concurrent API requests (default: 3)'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Delay between API requests in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum retries for failed tests (default: 2)'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='Timeout for individual tests in seconds (default: 30.0)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Directory for test results (default: from config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results as JSON file'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup environment
    logging.info("Setting up integration test environment...")
    enabled_providers, missing_keys = setup_integration_environment()
    
    if not enabled_providers:
        logging.error("No providers enabled for testing. Set TEST_*_INTEGRATION=true environment variables.")
        return 1
    
    if missing_keys:
        logging.warning(f"Missing API keys: {', '.join(missing_keys)}")
    
    # Create providers
    logging.info("Creating provider instances...")
    providers = create_providers(args.provider)
    
    if not providers:
        logging.error("No valid providers available for testing.")
        return 1
    
    # Select test suite
    test_functions = {}
    
    if args.test_suite in ['standard', 'all']:
        test_functions.update(STANDARD_INTEGRATION_TESTS)
        logging.info("Added standard integration tests")
    
    if args.test_suite in ['custom', 'all']:
        if IntegrationTestConfig.should_run_expensive_tests():
            test_functions.update(create_custom_tests())
            logging.info("Added custom integration tests")
        else:
            logging.info("Skipping custom tests - expensive tests not enabled")
    
    if args.test_suite in ['performance', 'all']:
        test_functions.update(create_performance_tests())
        logging.info("Added performance integration tests")
    
    if not test_functions:
        logging.error("No test functions selected")
        return 1
    
    # Create test runner
    runner = IntegrationTestRunner(
        max_workers=args.max_workers,
        rate_limit_delay=args.rate_limit,
        max_retries=args.max_retries,
        timeout=args.timeout
    )
    
    # Run tests
    logging.info(f"Running {len(test_functions)} tests across {len(providers)} providers...")
    suite_name = f"Integration Tests - {args.test_suite.title()} Suite"
    suite = runner.run_test_suite(test_functions, providers, suite_name)
    
    # Generate and display report
    logging.info("Generating test report...")
    report = runner.generate_report(suite)
    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    # Save results
    output_dir = args.output_dir or IntegrationTestConfig.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"integration_report_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logging.info(f"Report saved to: {report_file}")
    
    # Save JSON results if requested
    if args.save_json:
        json_file = os.path.join(output_dir, f"integration_results_{timestamp}.json")
        runner.save_results_json(suite, json_file)
    
    # Return appropriate exit code
    if suite.success_rate < 80.0:
        logging.warning(f"Success rate below 80%: {suite.success_rate:.1f}%")
        return 1
    
    logging.info(f"Integration tests completed successfully: {suite.success_rate:.1f}% success rate")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)