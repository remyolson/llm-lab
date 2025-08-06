"""
Test Data Fixtures

Provides sample data for testing various components.
"""

import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest


@pytest.fixture
def sample_prompts():
    """Collection of sample prompts for testing."""
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Write a Python function to calculate fibonacci numbers",
        "Translate 'Hello, world!' to Spanish",
        "Summarize the key points of machine learning",
        "What are the benefits of test-driven development?",
        "How does photosynthesis work?",
        "Create a SQL query to find duplicate records",
        "Explain the difference between TCP and UDP",
        "What is the significance of the Turing test?",
    ]


@pytest.fixture
def sample_responses():
    """Collection of sample LLM responses."""
    return [
        "The capital of France is Paris.",
        "Quantum computing uses quantum bits (qubits) that can be in multiple states simultaneously, unlike classical bits that are either 0 or 1.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "¡Hola, mundo!",
        "Machine learning key points: 1) Learning from data, 2) Pattern recognition, 3) Predictive modeling, 4) Continuous improvement",
    ]


@pytest.fixture
def evaluation_data():
    """Data for evaluation testing."""
    return {
        "questions": [
            {"id": 1, "question": "What is 2+2?", "expected": "4"},
            {"id": 2, "question": "What color is the sky?", "expected": "blue"},
            {"id": 3, "question": "What is the speed of light?", "expected": "299,792,458 m/s"},
        ],
        "metrics": {
            "accuracy": 0.85,
            "precision": 0.88,
            "recall": 0.82,
            "f1_score": 0.85,
        },
        "model_responses": {
            1: "The answer is 4",
            2: "The sky is typically blue during clear days",
            3: "The speed of light is approximately 300,000 km/s",
        },
    }


@pytest.fixture
def benchmark_data():
    """Data for benchmark testing."""

    def generate_benchmark_result(model_name: str, test_name: str):
        return {
            "model": model_name,
            "test": test_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "response_time_ms": random.uniform(100, 2000),
                "tokens_generated": random.randint(50, 500),
                "tokens_per_second": random.uniform(10, 100),
                "success_rate": random.uniform(0.8, 1.0),
            },
            "samples": random.randint(100, 1000),
        }

    models = ["gpt-4", "claude-3", "gemini-pro", "llama-2"]
    tests = ["truthfulness", "reasoning", "coding", "translation"]

    return [generate_benchmark_result(model, test) for model in models for test in tests]


@pytest.fixture
def large_dataset():
    """Generate a large dataset for stress testing."""

    def generate_record(idx: int):
        return {
            "id": idx,
            "prompt": f"Test prompt {idx}",
            "response": f"Test response {idx}",
            "metadata": {
                "timestamp": (datetime.now() - timedelta(hours=idx)).isoformat(),
                "model": random.choice(["gpt-4", "claude-3", "gemini-pro"]),
                "tokens": random.randint(10, 1000),
                "latency_ms": random.uniform(50, 5000),
            },
        }

    return [generate_record(i) for i in range(1000)]


@pytest.fixture
def test_configurations():
    """Various configuration scenarios for testing."""
    return {
        "minimal": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
        },
        "standard": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "timeout": 30,
        },
        "advanced": {
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 2000,
            "top_p": 0.95,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "timeout": 60,
            "max_retries": 5,
            "retry_delay": 2.0,
        },
        "conservative": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 500,
            "top_p": 0.5,
            "timeout": 15,
        },
    }


@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing error handling."""
    return [
        {
            "type": "validation_error",
            "message": "Invalid temperature value: 2.5",
            "details": {"field": "temperature", "value": 2.5, "constraint": "0 <= value <= 2"},
        },
        {
            "type": "api_error",
            "message": "API rate limit exceeded",
            "details": {"retry_after": 60, "limit": 1000, "remaining": 0},
        },
        {
            "type": "timeout_error",
            "message": "Request timed out after 30 seconds",
            "details": {"timeout": 30, "elapsed": 30.5},
        },
        {
            "type": "network_error",
            "message": "Failed to connect to API endpoint",
            "details": {"endpoint": "https://api.example.com", "error": "Connection refused"},
        },
    ]


@pytest.fixture
def performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "response_times": [random.uniform(50, 500) for _ in range(100)],
        "throughput": [random.uniform(10, 50) for _ in range(100)],
        "error_rates": [random.uniform(0, 0.05) for _ in range(100)],
        "memory_usage": [random.uniform(100, 500) for _ in range(100)],
        "cpu_usage": [random.uniform(10, 90) for _ in range(100)],
    }


@pytest.fixture
def comparison_data():
    """Data for model comparison testing."""
    return {
        "models": ["gpt-4", "claude-3", "gemini-pro"],
        "prompts": [
            "Explain recursion",
            "Write a haiku about programming",
            "What is the meaning of life?",
        ],
        "responses": {
            "gpt-4": [
                "Recursion is a programming technique where a function calls itself...",
                "Code flows like water\nBugs hide in the shadows deep\nDebugger finds truth",
                "The meaning of life is subjective and varies from person to person...",
            ],
            "claude-3": [
                "Recursion is a method where the solution depends on solutions to smaller instances...",
                "Logic gates open\nAlgorithms dance in loops\nSoftware comes alive",
                "From a philosophical perspective, the meaning of life has been debated...",
            ],
            "gemini-pro": [
                "Recursion is when a function calls itself to solve a problem...",
                "Bits and bytes align\nFunctions call in harmony\nPrograms spring to life",
                "The question of life's meaning has occupied philosophers for millennia...",
            ],
        },
    }
