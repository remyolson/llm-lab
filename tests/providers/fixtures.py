"""
Shared test fixtures for provider testing

This module provides common fixtures used across all provider tests.
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_successful_response():
    """Standard successful API response."""
    return {
        'id': 'test-response-id',
        'object': 'chat.completion',
        'created': int(datetime.now().timestamp()),
        'model': 'test-model',
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': 'This is a test response'
            },
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 20,
            'total_tokens': 30
        }
    }


@pytest.fixture
def mock_rate_limit_response():
    """Rate limit error response."""
    return {
        'error': {
            'message': 'Rate limit exceeded',
            'type': 'rate_limit_error',
            'code': 'rate_limit_exceeded'
        }
    }


@pytest.fixture
def mock_invalid_auth_response():
    """Invalid authentication response."""
    return {
        'error': {
            'message': 'Invalid API key',
            'type': 'invalid_request_error',
            'code': 'invalid_api_key'
        }
    }


@pytest.fixture
def mock_timeout_response():
    """Timeout error response."""
    return {
        'error': {
            'message': 'Request timeout',
            'type': 'timeout_error',
            'code': 'request_timeout'
        }
    }


@pytest.fixture
def sample_prompts():
    """Collection of sample prompts for testing."""
    return {
        'simple': "What is 2 + 2?",
        'with_context': "Based on the previous discussion about mathematics, what is 2 + 2?",
        'code_generation': "Write a Python function to calculate factorial",
        'creative': "Write a haiku about testing",
        'reasoning': "Explain why the sky is blue in simple terms",
        'empty': "",
        'very_long': "a" * 10000,
        'special_chars': "Test with special characters: @#$%^&*()",
        'unicode': "Test with unicode: 你好世界 🌍",
        'multiline': """This is a
        multiline prompt
        with multiple lines"""
    }


@pytest.fixture
def sample_system_prompts():
    """Collection of system prompts for testing."""
    return {
        'default': "You are a helpful assistant.",
        'concise': "Answer concisely in one sentence.",
        'code_expert': "You are an expert programmer. Provide clean, well-documented code.",
        'creative': "You are a creative writer. Be imaginative and descriptive.",
        'formal': "Respond in a formal, professional manner.",
        'json_mode': "Always respond with valid JSON.",
    }


@pytest.fixture
def mock_streaming_chunks():
    """Mock streaming response chunks."""
    return [
        {
            'id': 'chunk-1',
            'choices': [{
                'delta': {'content': 'This '},
                'index': 0
            }]
        },
        {
            'id': 'chunk-2',
            'choices': [{
                'delta': {'content': 'is '},
                'index': 0
            }]
        },
        {
            'id': 'chunk-3',
            'choices': [{
                'delta': {'content': 'a '},
                'index': 0
            }]
        },
        {
            'id': 'chunk-4',
            'choices': [{
                'delta': {'content': 'test.'},
                'index': 0
            }]
        },
        {
            'id': 'chunk-5',
            'choices': [{
                'delta': {},
                'finish_reason': 'stop',
                'index': 0
            }]
        }
    ]


@pytest.fixture
def performance_test_config():
    """Configuration for performance tests."""
    return {
        'num_requests': 10,
        'max_avg_response_time': 3.0,  # seconds
        'max_single_response_time': 10.0,  # seconds
        'test_prompts': [
            "What is 1 + 1?",
            "Name a primary color.",
            "Is ice cold?",
            "Count to 3.",
            "Say yes or no."
        ]
    }


@pytest.fixture
def error_scenarios():
    """Common error scenarios to test."""
    return [
        {
            'name': 'empty_prompt',
            'prompt': '',
            'expected_error': 'ProviderError',
            'error_contains': 'empty'
        },
        {
            'name': 'null_prompt',
            'prompt': None,
            'expected_error': 'ProviderError',
            'error_contains': 'None'
        },
        {
            'name': 'invalid_temperature',
            'prompt': 'Test',
            'params': {'temperature': 2.5},
            'expected_error': 'ProviderError',
            'error_contains': 'temperature'
        },
        {
            'name': 'invalid_max_tokens',
            'prompt': 'Test',
            'params': {'max_tokens': -1},
            'expected_error': 'ProviderError',
            'error_contains': 'max_tokens'
        },
        {
            'name': 'invalid_top_p',
            'prompt': 'Test',
            'params': {'top_p': 1.5},
            'expected_error': 'ProviderError',
            'error_contains': 'top_p'
        }
    ]


@pytest.fixture
def mock_provider_responses():
    """Mock responses for different providers to ensure compatibility."""
    return {
        'openai': {
            'id': 'chatcmpl-123',
            'object': 'chat.completion',
            'created': 1677652288,
            'model': 'gpt-3.5-turbo',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'Test response from OpenAI'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 9,
                'completion_tokens': 12,
                'total_tokens': 21
            }
        },
        'anthropic': {
            'id': 'msg_123',
            'type': 'message',
            'role': 'assistant',
            'content': [{
                'type': 'text',
                'text': 'Test response from Anthropic'
            }],
            'model': 'claude-3-opus-20240229',
            'stop_reason': 'end_turn',
            'stop_sequence': None,
            'usage': {
                'input_tokens': 10,
                'output_tokens': 15
            }
        },
        'google': {
            'candidates': [{
                'content': {
                    'parts': [{
                        'text': 'Test response from Google'
                    }],
                    'role': 'model'
                },
                'finishReason': 'STOP',
                'index': 0
            }],
            'promptFeedback': {
                'safetyRatings': []
            }
        }
    }


class MockAsyncIterator:
    """Mock async iterator for streaming responses."""
    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


@pytest.fixture
def create_mock_client():
    """Factory fixture for creating mock API clients."""
    def _create_client(provider_type='openai', **kwargs):
        client = MagicMock()
        
        if provider_type == 'openai':
            client.chat.completions.create = MagicMock()
        elif provider_type == 'anthropic':
            client.messages.create = MagicMock()
        elif provider_type == 'google':
            client.generate_content = MagicMock()
        
        for key, value in kwargs.items():
            setattr(client, key, value)
        
        return client
    
    return _create_client