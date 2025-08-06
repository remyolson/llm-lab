"""
Custom Assertion Helpers

Domain-specific assertion helpers for LLM testing, providing clear failure messages
and support for approximate comparisons, response validation, and metric checking.
"""

from __future__ import annotations

import json
import math
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Union

from src.types.core import APIResponse, ConfigDict, ErrorResponse, ProviderInfo
from src.types.evaluation import EvaluationResult, MetricResult


def assert_provider_response(
    response: Any,
    expected_content: Optional[str] = None,
    expected_model: Optional[str] = None,
    expected_provider: Optional[str] = None,
    check_usage: bool = True,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a provider response is valid and matches expectations.

    Args:
        response: The response to validate
        expected_content: Expected content (if provided)
        expected_model: Expected model name
        expected_provider: Expected provider name
        check_usage: Whether to check usage data
        message: Custom assertion message
    """
    base_msg = message or "Provider response validation failed"

    # Check response is correct type
    assert isinstance(response, (dict, APIResponse)), (
        f"{base_msg}: Response must be dict or APIResponse"
    )

    # Convert to dict if needed
    if isinstance(response, APIResponse):
        response_dict = response
    else:
        response_dict = response

    # Check required fields
    assert "content" in response_dict, f"{base_msg}: Missing 'content' field"
    assert response_dict["content"], f"{base_msg}: Content is empty"

    # Check expected values
    if expected_content is not None:
        assert response_dict["content"] == expected_content, (
            f"{base_msg}: Content mismatch. Expected: {expected_content}, Got: {response_dict['content']}"
        )

    if expected_model is not None:
        assert response_dict.get("model") == expected_model, (
            f"{base_msg}: Model mismatch. Expected: {expected_model}, Got: {response_dict.get('model')}"
        )

    if expected_provider is not None:
        assert response_dict.get("provider") == expected_provider, (
            f"{base_msg}: Provider mismatch. Expected: {expected_provider}, Got: {response_dict.get('provider')}"
        )

    # Check usage data
    if check_usage and "usage" in response_dict:
        usage = response_dict["usage"]
        assert isinstance(usage, dict), f"{base_msg}: Usage must be a dictionary"
        assert "prompt_tokens" in usage, f"{base_msg}: Missing prompt_tokens in usage"
        assert "completion_tokens" in usage, f"{base_msg}: Missing completion_tokens in usage"
        assert usage["prompt_tokens"] >= 0, f"{base_msg}: Invalid prompt_tokens"
        assert usage["completion_tokens"] >= 0, f"{base_msg}: Invalid completion_tokens"


def assert_evaluation_result(
    result: EvaluationResult | Dict[str, Any],
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    expected_methods: Optional[List[str]] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that an evaluation result is valid and meets expectations.

    Args:
        result: The evaluation result to validate
        min_score: Minimum acceptable score
        max_score: Maximum acceptable score
        expected_methods: Expected evaluation methods
        message: Custom assertion message
    """
    base_msg = message or "Evaluation result validation failed"

    # Convert to dict if needed
    if isinstance(result, EvaluationResult):
        result_dict = result
    else:
        result_dict = result

    # Check required fields
    assert "results" in result_dict, f"{base_msg}: Missing 'results' field"
    assert isinstance(result_dict["results"], dict), f"{base_msg}: Results must be a dictionary"
    assert len(result_dict["results"]) > 0, f"{base_msg}: Results cannot be empty"

    # Check each method result
    for method, method_result in result_dict["results"].items():
        assert "score" in method_result, f"{base_msg}: Missing score for method {method}"
        score = method_result["score"]

        # Validate score range
        assert 0.0 <= score <= 1.0, (
            f"{base_msg}: Score {score} out of range [0, 1] for method {method}"
        )

        if min_score is not None:
            assert score >= min_score, (
                f"{base_msg}: Score {score} below minimum {min_score} for method {method}"
            )

        if max_score is not None:
            assert score <= max_score, (
                f"{base_msg}: Score {score} above maximum {max_score} for method {method}"
            )

    # Check expected methods
    if expected_methods:
        actual_methods = set(result_dict["results"].keys())
        expected_set = set(expected_methods)
        assert actual_methods == expected_set, (
            f"{base_msg}: Method mismatch. Expected: {expected_set}, Got: {actual_methods}"
        )


def assert_metric_in_range(
    metric: MetricResult | Dict[str, Any] | float,
    min_value: float,
    max_value: float,
    metric_name: Optional[str] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a metric value is within the expected range.

    Args:
        metric: The metric to validate (can be MetricResult, dict, or float)
        min_value: Minimum acceptable value
        max_value: Maximum acceptable value
        metric_name: Name of the metric (for error messages)
        message: Custom assertion message
    """
    metric_name = metric_name or "metric"
    base_msg = message or f"{metric_name} validation failed"

    # Extract value based on type
    if isinstance(metric, (int, float)):
        value = float(metric)
    elif isinstance(metric, dict):
        assert "value" in metric, f"{base_msg}: Missing 'value' field in metric dict"
        value = float(metric["value"])
    elif isinstance(metric, MetricResult):
        value = float(metric.value)
    else:
        raise TypeError(f"{base_msg}: Unsupported metric type {type(metric)}")

    # Check range
    assert min_value <= value <= max_value, (
        f"{base_msg}: Value {value} not in range [{min_value}, {max_value}]"
    )


def assert_config_valid(
    config: ConfigDict | Dict[str, Any],
    required_providers: Optional[List[str]] = None,
    required_settings: Optional[List[str]] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a configuration is valid and complete.

    Args:
        config: The configuration to validate
        required_providers: List of required provider names
        required_settings: List of required top-level settings
        message: Custom assertion message
    """
    base_msg = message or "Configuration validation failed"

    # Convert to dict if needed
    if isinstance(config, ConfigDict):
        config_dict = config
    else:
        config_dict = config

    assert isinstance(config_dict, dict), f"{base_msg}: Configuration must be a dictionary"

    # Check required providers
    if required_providers:
        assert "providers" in config_dict, f"{base_msg}: Missing 'providers' section"
        providers = config_dict["providers"]

        for provider in required_providers:
            assert provider in providers, f"{base_msg}: Missing required provider '{provider}'"

            # Check provider has required fields
            provider_config = providers[provider]
            assert "api_key" in provider_config or "model" in provider_config, (
                f"{base_msg}: Provider '{provider}' missing required fields"
            )

    # Check required settings
    if required_settings:
        for setting in required_settings:
            assert setting in config_dict, f"{base_msg}: Missing required setting '{setting}'"


def assert_approximately_equal(
    actual: float,
    expected: float,
    tolerance: float = 0.01,
    relative: bool = False,
    message: Optional[str] = None,
) -> None:
    """
    Assert that two floating-point values are approximately equal.

    Args:
        actual: The actual value
        expected: The expected value
        tolerance: Absolute or relative tolerance
        relative: If True, use relative tolerance
        message: Custom assertion message
    """
    base_msg = message or "Approximate equality assertion failed"

    if relative:
        if expected == 0:
            diff = abs(actual)
        else:
            diff = abs((actual - expected) / expected)
        assert diff <= tolerance, (
            f"{base_msg}: Relative difference {diff:.6f} exceeds tolerance {tolerance}. "
            f"Actual: {actual}, Expected: {expected}"
        )
    else:
        diff = abs(actual - expected)
        assert diff <= tolerance, (
            f"{base_msg}: Absolute difference {diff:.6f} exceeds tolerance {tolerance}. "
            f"Actual: {actual}, Expected: {expected}"
        )


def assert_response_format(
    response: Any,
    expected_format: Dict[str, type],
    allow_extra_fields: bool = True,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a response matches the expected format/schema.

    Args:
        response: The response to validate
        expected_format: Dictionary mapping field names to expected types
        allow_extra_fields: Whether to allow fields not in expected_format
        message: Custom assertion message
    """
    base_msg = message or "Response format validation failed"

    assert isinstance(response, dict), f"{base_msg}: Response must be a dictionary"

    # Check all expected fields are present with correct types
    for field, expected_type in expected_format.items():
        assert field in response, f"{base_msg}: Missing required field '{field}'"

        actual_value = response[field]
        if expected_type is not None:  # None means any type is acceptable
            assert isinstance(actual_value, expected_type), (
                f"{base_msg}: Field '{field}' has wrong type. "
                f"Expected: {expected_type.__name__}, Got: {type(actual_value).__name__}"
            )

    # Check for unexpected fields
    if not allow_extra_fields:
        extra_fields = set(response.keys()) - set(expected_format.keys())
        assert not extra_fields, f"{base_msg}: Unexpected fields: {extra_fields}"


def assert_error_handled(
    func: Callable,
    expected_exception: type[Exception],
    expected_message: Optional[str] = None,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a function properly handles errors.

    Args:
        func: The function to test
        expected_exception: Expected exception type
        expected_message: Expected error message (substring match)
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        message: Custom assertion message
    """
    base_msg = message or "Error handling assertion failed"
    kwargs = kwargs or {}

    try:
        func(*args, **kwargs)
        assert False, (
            f"{base_msg}: Expected {expected_exception.__name__} but no exception was raised"
        )
    except expected_exception as e:
        if expected_message:
            assert expected_message in str(e), (
                f"{base_msg}: Error message '{str(e)}' does not contain '{expected_message}'"
            )
    except Exception as e:
        assert False, (
            f"{base_msg}: Expected {expected_exception.__name__} but got {type(e).__name__}: {e}"
        )


def assert_text_similarity(
    actual: str,
    expected: str,
    min_similarity: float = 0.8,
    ignore_case: bool = False,
    ignore_whitespace: bool = False,
    message: Optional[str] = None,
) -> None:
    """
    Assert that two text strings are similar enough.

    Args:
        actual: The actual text
        expected: The expected text
        min_similarity: Minimum similarity score (0-1)
        ignore_case: Whether to ignore case differences
        ignore_whitespace: Whether to normalize whitespace
        message: Custom assertion message
    """
    base_msg = message or "Text similarity assertion failed"

    # Preprocess texts if needed
    if ignore_case:
        actual = actual.lower()
        expected = expected.lower()

    if ignore_whitespace:
        actual = " ".join(actual.split())
        expected = " ".join(expected.split())

    # Calculate similarity
    similarity = SequenceMatcher(None, actual, expected).ratio()

    assert similarity >= min_similarity, (
        f"{base_msg}: Similarity {similarity:.3f} below threshold {min_similarity}. "
        f"Actual: '{actual[:100]}...', Expected: '{expected[:100]}...'"
    )


def assert_json_equal(
    actual: Union[str, Dict[str, Any]],
    expected: Union[str, Dict[str, Any]],
    ignore_order: bool = True,
    ignore_keys: Optional[List[str]] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that two JSON objects are equal.

    Args:
        actual: The actual JSON (string or dict)
        expected: The expected JSON (string or dict)
        ignore_order: Whether to ignore order in lists
        ignore_keys: List of keys to ignore in comparison
        message: Custom assertion message
    """
    base_msg = message or "JSON equality assertion failed"
    ignore_keys = ignore_keys or []

    # Parse JSON strings if needed
    if isinstance(actual, str):
        try:
            actual = json.loads(actual)
        except json.JSONDecodeError as e:
            assert False, f"{base_msg}: Failed to parse actual JSON: {e}"

    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError as e:
            assert False, f"{base_msg}: Failed to parse expected JSON: {e}"

    # Remove ignored keys
    def remove_keys(obj, keys):
        if isinstance(obj, dict):
            return {k: remove_keys(v, keys) for k, v in obj.items() if k not in keys}
        elif isinstance(obj, list):
            return [remove_keys(item, keys) for item in obj]
        else:
            return obj

    actual = remove_keys(actual, ignore_keys)
    expected = remove_keys(expected, ignore_keys)

    # Sort lists if ignoring order
    def sort_lists(obj):
        if isinstance(obj, dict):
            return {k: sort_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list) and ignore_order:
            try:
                return sorted([sort_lists(item) for item in obj], key=str)
            except TypeError:
                return [sort_lists(item) for item in obj]
        else:
            return obj

    actual = sort_lists(actual)
    expected = sort_lists(expected)

    assert actual == expected, f"{base_msg}: JSON objects are not equal"


def assert_performance(
    duration: float,
    max_duration: float,
    operation: Optional[str] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that an operation completed within performance constraints.

    Args:
        duration: Actual duration in seconds
        max_duration: Maximum acceptable duration
        operation: Name of the operation (for error messages)
        message: Custom assertion message
    """
    operation = operation or "Operation"
    base_msg = message or f"{operation} performance assertion failed"

    assert duration <= max_duration, (
        f"{base_msg}: {operation} took {duration:.3f}s, exceeding limit of {max_duration:.3f}s"
    )


def assert_no_duplicates(
    items: List[Any],
    key: Optional[Callable[[Any], Any]] = None,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a list contains no duplicate items.

    Args:
        items: List of items to check
        key: Optional function to extract comparison key from items
        message: Custom assertion message
    """
    base_msg = message or "Duplicate items found"

    if key:
        seen = set()
        duplicates = []
        for item in items:
            item_key = key(item)
            if item_key in seen:
                duplicates.append(item)
            seen.add(item_key)
    else:
        seen = set()
        duplicates = []
        for item in items:
            if item in seen:
                duplicates.append(item)
            seen.add(item)

    assert not duplicates, f"{base_msg}: {duplicates}"
