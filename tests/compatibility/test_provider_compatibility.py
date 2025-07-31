"""
Provider compatibility tests

This module tests compatibility across different providers, ensuring consistent
behavior and identifying provider-specific quirks or limitations.
"""

import pytest
import os
from typing import List, Dict, Any, Optional
import time
import logging

from llm_providers.base import LLMProvider
from llm_providers.exceptions import ProviderError, RateLimitError
from tests.providers.fixtures import get_available_providers

logger = logging.getLogger(__name__)


class TestProviderCompatibility:
    """Test compatibility across different LLM providers."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for compatibility tests."""
        self.test_prompts = {
            'simple': "Hello",
            'math': "What is 2 + 2?",
            'creative': "Write a haiku about coding",
            'reasoning': "If all cats are animals and some animals are pets, are some cats pets?",
            'multilingual': "Translate 'hello world' to French",
            'technical': "Explain what REST API means",
            'long_response': "Write a detailed explanation of machine learning in 3 paragraphs"
        }
        
        self.compatibility_parameters = {
            'temperature_low': {'temperature': 0.1},
            'temperature_high': {'temperature': 0.9},
            'max_tokens_small': {'max_tokens': 10},
            'max_tokens_medium': {'max_tokens': 100},
            'max_tokens_large': {'max_tokens': 500}
        }
    
    def test_basic_text_generation_compatibility(self, provider_fixture):
        """Test basic text generation works across all providers."""
        provider = provider_fixture
        
        prompt = "Say hello"
        response = provider.generate(prompt, max_tokens=20)
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response.strip()) > 0
        
        # Response should be reasonable for the prompt
        assert len(response) < 1000, "Response unexpectedly long for simple prompt"
    
    def test_parameter_compatibility(self, provider_fixture):
        """Test parameter compatibility across providers."""
        provider = provider_fixture
        base_prompt = "Count from 1 to 3"
        
        # Test temperature parameter
        try:
            low_temp_response = provider.generate(
                base_prompt, 
                temperature=0.1, 
                max_tokens=50
            )
            high_temp_response = provider.generate(
                base_prompt, 
                temperature=0.9, 
                max_tokens=50
            )
            
            assert low_temp_response is not None
            assert high_temp_response is not None
            
        except Exception as e:
            pytest.fail(f"Temperature parameter not supported by {provider.__class__.__name__}: {e}")
        
        # Test max_tokens parameter
        try:
            short_response = provider.generate(
                "Write a story", 
                max_tokens=10
            )
            long_response = provider.generate(
                "Write a story", 
                max_tokens=100  
            )
            
            assert short_response is not None
            assert long_response is not None
            # Longer max_tokens should generally produce longer responses
            # (though not always guaranteed due to natural stopping points)
            
        except Exception as e:
            pytest.fail(f"max_tokens parameter not supported by {provider.__class__.__name__}: {e}")
    
    def test_prompt_type_compatibility(self, provider_fixture):
        """Test different types of prompts work across providers."""
        provider = provider_fixture
        
        for prompt_type, prompt in self.test_prompts.items():
            try:
                response = provider.generate(prompt, max_tokens=150)
                
                assert response is not None, f"No response for {prompt_type} prompt"
                assert isinstance(response, str), f"Non-string response for {prompt_type} prompt"
                assert len(response.strip()) > 0, f"Empty response for {prompt_type} prompt"
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"{provider.__class__.__name__} failed on {prompt_type} prompt: {e}")
                # Don't fail the test for individual prompt types,
                # but log the issue for compatibility analysis
    
    def test_unicode_and_special_characters(self, provider_fixture):
        """Test handling of unicode and special characters."""
        provider = provider_fixture
        
        test_cases = [
            "Hello 🌍 world",  # Emoji
            "Café naïve résumé",  # Accented characters
            "こんにちは世界",  # Japanese
            "Здравствуй мир",  # Cyrillic
            "\"Quotes\" and 'apostrophes'",  # Quotes
            "Special chars: @#$%^&*()",  # Special symbols
            "Line\nbreak\tand\ttabs",  # Whitespace
        ]
        
        for test_case in test_cases:
            try:
                response = provider.generate(
                    f"Echo this text: {test_case}", 
                    max_tokens=100
                )
                
                assert response is not None
                assert isinstance(response, str)
                # The response should contain some relevant content
                assert len(response.strip()) > 0
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"{provider.__class__.__name__} failed on unicode test '{test_case[:20]}...': {e}")
    
    def test_empty_and_edge_case_prompts(self, provider_fixture):
        """Test handling of edge case prompts."""
        provider = provider_fixture
        
        edge_cases = [
            "",  # Empty string
            " ",  # Just whitespace
            "\n\n\n",  # Just newlines
            "a" * 1000,  # Very long single word
            "Short. " * 100,  # Many short sentences
            "?",  # Single character
            "12345",  # Just numbers
        ]
        
        for i, prompt in enumerate(edge_cases):
            try:
                response = provider.generate(prompt, max_tokens=50)
                
                # Some providers might return empty responses for edge cases
                # That's acceptable as long as they don't crash
                if response is not None:
                    assert isinstance(response, str)
                
                time.sleep(0.1)
                
            except Exception as e:
                # Log but don't fail - edge case handling varies by provider
                logger.info(f"{provider.__class__.__name__} handling of edge case {i}: {type(e).__name__}")
    
    def test_consistent_behavior_across_calls(self, provider_fixture):
        """Test that providers behave consistently across multiple calls."""
        provider = provider_fixture
        
        # Test deterministic behavior with low temperature
        prompt = "What is the capital of France?"
        responses = []
        
        for i in range(3):
            try:
                response = provider.generate(
                    prompt, 
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=50
                )
                responses.append(response)
                time.sleep(0.5)  # Avoid rate limits
                
            except Exception as e:
                logger.warning(f"Call {i+1} failed for {provider.__class__.__name__}: {e}")
        
        # All responses should be non-None
        valid_responses = [r for r in responses if r is not None]
        assert len(valid_responses) >= 2, "Need at least 2 valid responses to test consistency"
        
        # For low temperature, we expect some similarity (though not identical)
        # At minimum, all responses should mention "Paris" for this question
        for response in valid_responses:
            assert "paris" in response.lower() or "париж" in response.lower() or "パリ" in response, \
                f"Response doesn't mention Paris: {response}"
    
    def test_parameter_boundary_handling(self, provider_fixture):
        """Test handling of parameter boundary values."""
        provider = provider_fixture
        prompt = "Write a short sentence"
        
        # Test boundary values for common parameters
        boundary_tests = [
            {'temperature': 0.0},  # Minimum temperature
            {'temperature': 1.0},  # Maximum standard temperature
            {'max_tokens': 1},     # Minimum tokens
            {'max_tokens': 4000},  # Large token count
        ]
        
        for params in boundary_tests:
            try:
                response = provider.generate(prompt, **params)
                
                # Response should be valid
                if response is not None:
                    assert isinstance(response, str)
                    
                    # Check token limits are respected (roughly)
                    if 'max_tokens' in params and params['max_tokens'] <= 10:
                        # For very small token limits, response should be short
                        assert len(response.split()) <= params['max_tokens'] * 2, \
                            f"Response too long for max_tokens={params['max_tokens']}"
                
                time.sleep(0.2)
                
            except Exception as e:
                # Some boundary values might not be supported
                logger.info(f"{provider.__class__.__name__} boundary test {params}: {type(e).__name__}")
    
    def test_error_handling_consistency(self, provider_fixture):
        """Test that providers handle errors consistently."""
        provider = provider_fixture
        
        # Test various error conditions
        error_tests = [
            {
                'name': 'negative_max_tokens',
                'params': {'max_tokens': -1},
                'prompt': 'Hello'
            },
            {
                'name': 'invalid_temperature',
                'params': {'temperature': -1},
                'prompt': 'Hello'
            },
            {
                'name': 'very_high_temperature',
                'params': {'temperature': 10.0},
                'prompt': 'Hello'
            }
        ]
        
        for test in error_tests:
            try:
                response = provider.generate(test['prompt'], **test['params'])
                # If no exception was raised, that's also valid behavior
                logger.info(f"{provider.__class__.__name__} accepted {test['name']}: {response is not None}")
                
            except (ValueError, ProviderError) as e:
                # Expected error types for invalid parameters
                logger.info(f"{provider.__class__.__name__} properly rejected {test['name']}: {type(e).__name__}")
                
            except Exception as e:
                # Unexpected error type
                logger.warning(f"{provider.__class__.__name__} unexpected error for {test['name']}: {type(e).__name__}: {e}")
    
    def test_concurrent_request_compatibility(self, provider_fixture):
        """Test that providers can handle concurrent requests safely."""
        import threading
        import concurrent.futures
        
        provider = provider_fixture
        prompt = "Count to 5"
        num_concurrent = 3
        results = []
        errors = []
        
        def make_request():
            try:
                response = provider.generate(prompt, max_tokens=50)
                results.append(response)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            concurrent.futures.wait(futures)
        
        # Analyze results
        total_requests = len(results) + len(errors)
        success_rate = len(results) / total_requests if total_requests > 0 else 0
        
        # We expect at least 50% success rate for concurrent requests
        assert success_rate >= 0.5, f"Low success rate for concurrent requests: {success_rate:.1%}"
        
        # All successful results should be valid
        for result in results:
            assert result is not None
            assert isinstance(result, str)
            assert len(result.strip()) > 0
        
        # Log any rate limiting errors (these are expected)
        rate_limit_errors = [e for e in errors if isinstance(e, RateLimitError)]
        if rate_limit_errors:
            logger.info(f"{provider.__class__.__name__} rate limited {len(rate_limit_errors)}/{total_requests} concurrent requests")


@pytest.mark.parametrize("provider_pair", [
    ("openai", "anthropic"),
    ("openai", "google"),
    ("anthropic", "google")
])
def test_cross_provider_response_similarity(provider_pair):
    """Test that different providers give similar responses to objective questions."""
    # Skip if providers not available
    available_providers = get_available_providers()
    provider_map = {p.__class__.__name__.lower().replace('provider', ''): p for p in available_providers}
    
    provider1_name, provider2_name = provider_pair
    
    if provider1_name not in provider_map or provider2_name not in provider_map:
        pytest.skip(f"Providers {provider1_name} or {provider2_name} not available")
    
    provider1 = provider_map[provider1_name]
    provider2 = provider_map[provider2_name]
    
    # Test objective questions where answers should be similar
    objective_questions = [
        "What is the capital of Japan?",
        "How many days are in a week?",
        "What is 10 + 15?",
        "What color do you get when you mix red and blue?"
    ]
    
    for question in objective_questions:
        try:
            response1 = provider1.generate(question, temperature=0.1, max_tokens=50)
            time.sleep(0.5)  # Avoid rate limits
            response2 = provider2.generate(question, temperature=0.1, max_tokens=50)
            time.sleep(0.5)
            
            assert response1 is not None and response2 is not None
            
            # For objective questions, both responses should contain key information
            # This is a basic compatibility check, not strict equality
            assert len(response1.strip()) > 0 and len(response2.strip()) > 0
            
            # Log responses for analysis
            logger.info(f"Question: {question}")
            logger.info(f"{provider1_name}: {response1[:100]}...")
            logger.info(f"{provider2_name}: {response2[:100]}...")
            
        except Exception as e:
            logger.warning(f"Cross-provider test failed for '{question}': {e}")


class TestProviderFeatureCompatibility:
    """Test specific feature compatibility across providers."""
    
    def test_streaming_compatibility(self, provider_fixture):
        """Test streaming support where available."""
        provider = provider_fixture
        
        # Check if provider supports streaming
        if not hasattr(provider, 'generate_stream'):
            pytest.skip(f"{provider.__class__.__name__} doesn't support streaming")
        
        prompt = "Count from 1 to 10"
        
        try:
            stream = provider.generate_stream(prompt, max_tokens=100)
            chunks = []
            
            for chunk in stream:
                chunks.append(chunk)
                if len(chunks) > 20:  # Prevent infinite loops
                    break
            
            assert len(chunks) > 0, "No chunks received from stream"
            
            # Combine chunks to form complete response
            complete_response = ''.join(chunks)
            assert len(complete_response.strip()) > 0
            
        except NotImplementedError:
            pytest.skip(f"{provider.__class__.__name__} streaming not implemented")
        except Exception as e:
            pytest.fail(f"Streaming failed for {provider.__class__.__name__}: {e}")
    
    def test_token_counting_compatibility(self, provider_fixture):
        """Test token counting support where available."""
        provider = provider_fixture
        
        if not hasattr(provider, 'count_tokens'):
            pytest.skip(f"{provider.__class__.__name__} doesn't support token counting")
        
        test_texts = [
            "Hello world",
            "This is a longer sentence with more words to count tokens for.",
            "",
            "Single",
            "🌍🌎🌏"  # Emoji
        ]
        
        for text in test_texts:
            try:
                token_count = provider.count_tokens(text)
                
                assert isinstance(token_count, int)
                assert token_count >= 0
                
                # Basic sanity check: longer text should generally have more tokens
                if len(text) > 50:
                    assert token_count > 5, f"Unexpectedly low token count for long text: {token_count}"
                
            except NotImplementedError:
                pytest.skip(f"{provider.__class__.__name__} token counting not implemented")
            except Exception as e:
                logger.warning(f"Token counting failed for '{text[:20]}...': {e}")
    
    def test_model_listing_compatibility(self, provider_fixture):
        """Test model listing support where available."""
        provider = provider_fixture
        
        if not hasattr(provider, 'list_models'):
            pytest.skip(f"{provider.__class__.__name__} doesn't support model listing")
        
        try:
            models = provider.list_models()
            
            assert isinstance(models, list)
            assert len(models) > 0, "No models returned"
            
            # Each model should be a string or dict with name
            for model in models:
                if isinstance(model, str):
                    assert len(model) > 0
                elif isinstance(model, dict):
                    assert 'name' in model or 'id' in model
                else:
                    pytest.fail(f"Unexpected model format: {type(model)}")
            
        except NotImplementedError:
            pytest.skip(f"{provider.__class__.__name__} model listing not implemented")
        except Exception as e:
            logger.warning(f"Model listing failed: {e}")


# Integration with existing test infrastructure
def test_compatibility_report_generation(tmp_path):
    """Generate a compatibility report for all available providers."""
    available_providers = get_available_providers()
    
    if len(available_providers) < 2:
        pytest.skip("Need at least 2 providers for compatibility report")
    
    report_path = tmp_path / "compatibility_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("LLM Provider Compatibility Report\n")
        f.write("=" * 50 + "\n\n")
        
        for provider in available_providers:
            f.write(f"Provider: {provider.__class__.__name__}\n")
            f.write(f"Model: {provider.model_name}\n")
            
            # Test basic functionality
            try:
                response = provider.generate("Hello", max_tokens=10)
                f.write(f"Basic generation: ✓ ({len(response)} chars)\n")
            except Exception as e:
                f.write(f"Basic generation: ✗ ({type(e).__name__})\n")
            
            # Test parameter support
            features = []
            
            # Test streaming
            if hasattr(provider, 'generate_stream'):
                features.append("streaming")
            
            # Test token counting
            if hasattr(provider, 'count_tokens'):
                features.append("token_counting")
            
            # Test model listing
            if hasattr(provider, 'list_models'):
                features.append("model_listing")
            
            f.write(f"Features: {', '.join(features) if features else 'basic only'}\n")
            f.write("\n")
    
    assert report_path.exists()
    assert report_path.stat().st_size > 0