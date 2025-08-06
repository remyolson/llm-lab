"""
Shared test data and fixtures for all tests.
"""

# Sample prompts for various test scenarios
SAMPLE_PROMPTS = {
    "simple": [
        "What is 2+2?",
        "What color is the sky?",
        "Hello, world!",
    ],
    "knowledge": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What year did World War II end?",
    ],
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
        "What comes next in the sequence: 2, 4, 8, 16, ?",
    ],
    "code": [
        "Write a Python function to calculate the factorial of a number.",
        "How do you reverse a string in JavaScript?",
        "Explain the difference between a list and a tuple in Python.",
    ],
    "creative": [
        "Write a haiku about artificial intelligence.",
        "Describe a sunset without using the words 'sun', 'orange', or 'beautiful'.",
        "Create a one-sentence story about a time traveler.",
    ],
}

# Expected responses for evaluation
EXPECTED_RESPONSES = {
    "simple": {
        "What is 2+2?": ["4", "four"],
        "What color is the sky?": ["blue", "azure", "cyan"],
        "Hello, world!": ["Hello", "Hi", "Greetings"],
    },
    "knowledge": {
        "What is the capital of France?": ["Paris"],
        "Who wrote Romeo and Juliet?": ["Shakespeare", "William Shakespeare"],
        "What year did World War II end?": ["1945"],
    },
}

# Sample datasets for benchmarking
BENCHMARK_DATASETS = {
    "tiny": {
        "name": "Tiny Test Dataset",
        "size": 5,
        "prompts": SAMPLE_PROMPTS["simple"][:5],
    },
    "small": {
        "name": "Small Benchmark Dataset",
        "size": 20,
        "prompts": [
            prompt
            for category in ["simple", "knowledge"]
            for prompt in SAMPLE_PROMPTS.get(category, [])
        ][:20],
    },
    "medium": {
        "name": "Medium Benchmark Dataset",
        "size": 100,
        "prompts": [
            prompt for category in SAMPLE_PROMPTS.keys() for prompt in SAMPLE_PROMPTS[category]
        ]
        * 4,  # Repeat to get ~100 prompts
    },
}

# Configuration templates
CONFIG_TEMPLATES = {
    "minimal": {
        "providers": {
            "openai": {
                "api_key": "${OPENAI_API_KEY}",
                "model": "gpt-3.5-turbo",
            }
        }
    },
    "standard": {
        "providers": {
            "openai": {
                "api_key": "${OPENAI_API_KEY}",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            "anthropic": {
                "api_key": "${ANTHROPIC_API_KEY}",
                "model": "claude-3-opus-20240229",
                "temperature": 0.5,
                "max_tokens": 1000,
            },
        },
        "benchmarks": {
            "timeout": 30,
            "retries": 3,
            "batch_size": 10,
        },
    },
    "comprehensive": {
        "providers": {
            "openai": {
                "api_key": "${OPENAI_API_KEY}",
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.95,
            },
            "anthropic": {
                "api_key": "${ANTHROPIC_API_KEY}",
                "models": ["claude-3-haiku-20240307", "claude-3-opus-20240229"],
                "temperature": 0.5,
                "max_tokens": 2000,
            },
            "google": {
                "api_key": "${GOOGLE_API_KEY}",
                "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
                "temperature": 0.7,
                "max_tokens": 2000,
            },
        },
        "benchmarks": {
            "timeout": 60,
            "retries": 5,
            "batch_size": 20,
            "parallel": True,
        },
        "monitoring": {
            "enabled": True,
            "metrics": ["latency", "throughput", "errors", "tokens"],
            "export_format": "prometheus",
        },
    },
}

# Error scenarios for testing
ERROR_SCENARIOS = {
    "rate_limit": {
        "error_type": "RateLimitError",
        "message": "Rate limit exceeded. Please try again later.",
        "retry_after": 60,
    },
    "invalid_api_key": {
        "error_type": "AuthenticationError",
        "message": "Invalid API key provided.",
        "retry_after": None,
    },
    "timeout": {
        "error_type": "TimeoutError",
        "message": "Request timed out after 30 seconds.",
        "retry_after": 0,
    },
    "invalid_model": {
        "error_type": "ValidationError",
        "message": "Invalid model name: unknown-model",
        "retry_after": None,
    },
    "server_error": {
        "error_type": "ServerError",
        "message": "Internal server error. Please try again.",
        "retry_after": 5,
    },
}

# Performance baselines for regression testing
PERFORMANCE_BASELINES = {
    "openai": {
        "gpt-3.5-turbo": {
            "latency_p50": 0.5,  # seconds
            "latency_p95": 1.2,
            "latency_p99": 2.0,
            "throughput": 20,  # requests/second
            "error_rate": 0.01,  # 1%
        },
        "gpt-4": {
            "latency_p50": 1.0,
            "latency_p95": 2.5,
            "latency_p99": 4.0,
            "throughput": 10,
            "error_rate": 0.01,
        },
    },
    "anthropic": {
        "claude-3-haiku-20240307": {
            "latency_p50": 0.3,
            "latency_p95": 0.8,
            "latency_p99": 1.5,
            "throughput": 30,
            "error_rate": 0.005,
        },
        "claude-3-opus-20240229": {
            "latency_p50": 0.8,
            "latency_p95": 2.0,
            "latency_p99": 3.5,
            "throughput": 15,
            "error_rate": 0.01,
        },
    },
    "google": {
        "gemini-1.5-flash": {
            "latency_p50": 0.4,
            "latency_p95": 1.0,
            "latency_p99": 1.8,
            "throughput": 25,
            "error_rate": 0.02,
        },
        "gemini-1.5-pro": {
            "latency_p50": 0.7,
            "latency_p95": 1.8,
            "latency_p99": 3.0,
            "throughput": 12,
            "error_rate": 0.015,
        },
    },
}

# Mock responses for deterministic testing
MOCK_RESPONSES = {
    "default": "This is a mock response from the LLM provider.",
    "code": "```python\ndef hello_world():\n    print('Hello, World!')\n```",
    "json": '{"status": "success", "data": {"result": "mock data"}}',
    "error": "I'm sorry, but I cannot process this request.",
    "streaming": ["This ", "is ", "a ", "streaming ", "response."],
}

# Test user profiles for different scenarios
TEST_USERS = {
    "basic": {
        "id": "user_001",
        "name": "Test User",
        "api_keys": {
            "openai": "test-openai-key",
        },
        "preferences": {
            "default_provider": "openai",
            "default_model": "gpt-3.5-turbo",
        },
    },
    "premium": {
        "id": "user_002",
        "name": "Premium User",
        "api_keys": {
            "openai": "test-openai-key",
            "anthropic": "test-anthropic-key",
            "google": "test-google-key",
        },
        "preferences": {
            "default_provider": "anthropic",
            "default_model": "claude-3-opus-20240229",
            "parallel_requests": True,
            "caching_enabled": True,
        },
    },
    "researcher": {
        "id": "user_003",
        "name": "Research User",
        "api_keys": {
            "openai": "test-openai-key",
            "anthropic": "test-anthropic-key",
            "google": "test-google-key",
        },
        "preferences": {
            "benchmarking": True,
            "detailed_metrics": True,
            "export_formats": ["csv", "json", "parquet"],
            "retention_days": 90,
        },
    },
}
