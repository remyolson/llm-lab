"""
General Test Helper Functions

Utility functions for common testing patterns, including prompt generation,
response mocking, configuration management, and performance measurement.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar
from unittest.mock import MagicMock, patch

from .factories import fake

T = TypeVar("T")


def generate_test_prompt(
    length: str = "medium", topic: Optional[str] = None, style: str = "question"
) -> str:
    """
    Generate a test prompt with specified characteristics.

    Args:
        length: "short", "medium", or "long"
        topic: Optional specific topic
        style: "question", "instruction", "completion", or "conversation"

    Returns:
        Generated test prompt
    """
    topic = topic or fake.word()

    templates = {
        "question": {
            "short": f"What is {topic}?",
            "medium": f"Can you explain how {topic} works and why it's important?",
            "long": f"Please provide a comprehensive explanation of {topic}, including its history, current applications, advantages and disadvantages, and future prospects. Include specific examples where relevant.",
        },
        "instruction": {
            "short": f"List 3 facts about {topic}.",
            "medium": f"Write a brief summary about {topic} in 100 words.",
            "long": f"Create a detailed guide about {topic} that covers the following aspects: 1) Basic concepts, 2) Practical applications, 3) Common misconceptions, 4) Best practices, and 5) Additional resources for learning.",
        },
        "completion": {
            "short": f"The {topic} is",
            "medium": f"When considering {topic}, it's important to understand that",
            "long": f"The field of {topic} has evolved significantly over the years. Initially, {topic} was primarily used for basic applications, but today",
        },
        "conversation": {
            "short": f"User: Tell me about {topic}\nAssistant:",
            "medium": f"User: I'm interested in learning about {topic}. Where should I start?\nAssistant: Great question! Let me help you understand {topic}.\n\nUser: What are the key concepts?\nAssistant:",
            "long": f"User: I've been researching {topic} and have some questions.\nAssistant: I'd be happy to help! What would you like to know?\n\nUser: First, can you explain the fundamental principles?\nAssistant: Certainly! The fundamental principles of {topic} include...\n\nUser: That's helpful. How does this apply in practice?\nAssistant:",
        },
    }

    return templates.get(style, templates["question"]).get(length, templates[style]["medium"])


def generate_test_response(
    prompt: str,
    provider: str = "generic",
    style: str = "informative",
    include_reasoning: bool = False,
) -> str:
    """
    Generate a test response for a given prompt.

    Args:
        prompt: The input prompt
        provider: Provider style to mimic
        style: Response style ("informative", "concise", "creative", "technical")
        include_reasoning: Whether to include reasoning steps

    Returns:
        Generated test response
    """
    # Extract topic from prompt if possible
    words = prompt.lower().split()
    topic = words[-1] if words else "topic"

    base_response = {
        "informative": f"Based on the available information, {topic} is a complex subject that involves multiple aspects. {fake.paragraph()}",
        "concise": f"{topic.capitalize()}: {fake.sentence()}",
        "creative": f"Imagine {topic} as {fake.word()}. {fake.paragraph()} In essence, {fake.sentence()}",
        "technical": f"The technical implementation of {topic} requires understanding of {fake.word()} and {fake.word()}. {fake.paragraph()}",
    }

    response = base_response.get(style, base_response["informative"])

    if include_reasoning:
        response = f"Let me think about this step by step:\n1. {fake.sentence()}\n2. {fake.sentence()}\n3. {fake.sentence()}\n\nTherefore, {response}"

    # Add provider-specific flavor
    if provider == "openai":
        response = f"I'll help you understand this. {response}"
    elif provider == "anthropic":
        response = f"I'd be happy to explain. {response}"
    elif provider == "google":
        response = f"Here's what I found: {response}"

    return response


def create_temp_config(
    providers: Optional[List[str]] = None, settings: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Create a temporary configuration file.

    Args:
        providers: List of provider names to include
        settings: Additional settings to include

    Returns:
        Path to temporary config file
    """
    providers = providers or ["openai"]
    settings = settings or {}

    config = {
        "providers": {
            provider: {
                "api_key": f"test-key-{provider}",
                "model": f"test-model-{provider}",
                "temperature": 0.7,
                "max_tokens": 1000,
            }
            for provider in providers
        },
        **settings,
    }

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config, temp_file, indent=2)
    temp_file.close()

    return Path(temp_file.name)


@contextmanager
def mock_api_call(response: Any, delay: float = 0.0, side_effect: Optional[Exception] = None):
    """
    Context manager for mocking API calls.

    Args:
        response: The response to return
        delay: Delay before returning response
        side_effect: Exception to raise instead of returning

    Yields:
        Mock object
    """
    mock = MagicMock()

    if side_effect:
        mock.side_effect = side_effect
    else:

        def delayed_response(*args, **kwargs):
            if delay > 0:
                time.sleep(delay)
            return response

        mock.return_value = response
        mock.side_effect = delayed_response if delay > 0 else None

    yield mock


def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 10.0,
    interval: float = 0.1,
    message: Optional[str] = None,
) -> bool:
    """
    Wait for a condition to become true.

    Args:
        condition: Function that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds
        message: Optional message for timeout error

    Returns:
        True if condition was met, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        if condition():
            return True
        time.sleep(interval)

    if message:
        raise TimeoutError(message)
    return False


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying failed operations.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        return wrapper

    return decorator


@contextmanager
def capture_logs(logger_name: Optional[str] = None, level: str = "INFO"):
    """
    Context manager for capturing log messages.

    Args:
        logger_name: Name of logger to capture (None for root)
        level: Minimum log level to capture

    Yields:
        List of captured log records
    """
    import logging

    logger = logging.getLogger(logger_name)
    original_level = logger.level
    original_handlers = logger.handlers[:]

    # Create custom handler to capture logs
    captured_logs = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            captured_logs.append(record)

    handler = CaptureHandler()
    handler.setLevel(getattr(logging, level))

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level))

    try:
        yield captured_logs
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


@contextmanager
def measure_performance(name: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for measuring performance metrics.

    Args:
        name: Optional name for the measurement

    Yields:
        Dictionary containing performance metrics
    """
    import os

    import psutil

    metrics = {
        "name": name or "unnamed",
        "start_time": time.time(),
        "start_memory": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,  # MB
    }

    yield metrics

    metrics["end_time"] = time.time()
    metrics["duration"] = metrics["end_time"] - metrics["start_time"]
    metrics["end_memory"] = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    metrics["memory_used"] = metrics["end_memory"] - metrics["start_memory"]


def generate_cache_key(data: Any) -> str:
    """
    Generate a cache key for given data.

    Args:
        data: Data to generate key for

    Returns:
        Cache key string
    """
    if isinstance(data, str):
        content = data
    elif isinstance(data, (dict, list)):
        content = json.dumps(data, sort_keys=True)
    else:
        content = str(data)

    return hashlib.sha256(content.encode()).hexdigest()


def create_mock_environment(
    env_vars: Dict[str, str], temp_files: Optional[List[Tuple[str, str]]] = None
) -> Dict[str, Any]:
    """
    Create a mock environment with specified variables and files.

    Args:
        env_vars: Environment variables to set
        temp_files: List of (filename, content) tuples

    Returns:
        Dictionary with cleanup function and file paths
    """
    temp_dir = tempfile.mkdtemp()
    original_env = dict(os.environ)
    created_files = []

    # Set environment variables
    os.environ.update(env_vars)

    # Create temporary files
    if temp_files:
        for filename, content in temp_files:
            file_path = Path(temp_dir) / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            created_files.append(file_path)

    def cleanup():
        # Restore environment
        os.environ.clear()
        os.environ.update(original_env)

        # Remove temporary files
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"temp_dir": temp_dir, "files": created_files, "cleanup": cleanup}


async def async_retry(
    coro_func: Callable[..., T], max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0
) -> T:
    """
    Retry an async operation with exponential backoff.

    Args:
        coro_func: Async function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier

    Returns:
        Result from successful attempt
    """
    current_delay = delay
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await coro_func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
                current_delay *= backoff

    raise last_exception


def compare_json_structures(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
    ignore_keys: Optional[List[str]] = None,
    fuzzy_match_values: bool = False,
) -> List[str]:
    """
    Compare two JSON structures and return differences.

    Args:
        actual: Actual JSON structure
        expected: Expected JSON structure
        ignore_keys: Keys to ignore in comparison
        fuzzy_match_values: Whether to use fuzzy matching for string values

    Returns:
        List of difference descriptions
    """
    ignore_keys = ignore_keys or []
    differences = []

    def compare_values(path: str, val1: Any, val2: Any):
        if type(val1) != type(val2):
            differences.append(
                f"{path}: Type mismatch - {type(val1).__name__} vs {type(val2).__name__}"
            )
        elif isinstance(val1, dict):
            compare_dicts(path, val1, val2)
        elif isinstance(val1, list):
            compare_lists(path, val1, val2)
        elif isinstance(val1, str) and fuzzy_match_values:
            from difflib import SequenceMatcher

            similarity = SequenceMatcher(None, val1, val2).ratio()
            if similarity < 0.8:
                differences.append(f"{path}: String values differ (similarity: {similarity:.2%})")
        elif val1 != val2:
            differences.append(f"{path}: Value mismatch - {val1} vs {val2}")

    def compare_dicts(path: str, d1: Dict, d2: Dict):
        all_keys = set(d1.keys()) | set(d2.keys())
        for key in all_keys:
            if key in ignore_keys:
                continue

            key_path = f"{path}.{key}" if path else key

            if key not in d1:
                differences.append(f"{key_path}: Missing in actual")
            elif key not in d2:
                differences.append(f"{key_path}: Extra in actual")
            else:
                compare_values(key_path, d1[key], d2[key])

    def compare_lists(path: str, l1: List, l2: List):
        if len(l1) != len(l2):
            differences.append(f"{path}: Length mismatch - {len(l1)} vs {len(l2)}")

        for i, (item1, item2) in enumerate(zip(l1, l2)):
            compare_values(f"{path}[{i}]", item1, item2)

    compare_dicts("", actual, expected)
    return differences


def generate_mock_metrics(count: int = 10, metric_type: str = "mixed") -> List[Dict[str, Any]]:
    """
    Generate mock metrics data for testing.

    Args:
        count: Number of metrics to generate
        metric_type: Type of metrics ("accuracy", "latency", "cost", "mixed")

    Returns:
        List of metric dictionaries
    """
    metrics = []

    for i in range(count):
        if metric_type == "mixed" or metric_type == "accuracy":
            metrics.append(
                {
                    "name": "accuracy",
                    "value": random.uniform(0.7, 1.0),
                    "unit": "percentage",
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                }
            )

        if metric_type == "mixed" or metric_type == "latency":
            metrics.append(
                {
                    "name": "latency",
                    "value": random.uniform(50, 500),
                    "unit": "milliseconds",
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                }
            )

        if metric_type == "mixed" or metric_type == "cost":
            metrics.append(
                {
                    "name": "cost",
                    "value": random.uniform(0.01, 0.10),
                    "unit": "dollars",
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                }
            )

    return metrics
