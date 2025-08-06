"""
TypeGuard Utilities for Runtime Type Checking

This module provides comprehensive TypeGuard functions for runtime type validation
and type narrowing, especially useful for validating data at API boundaries,
configuration loading, and dynamic content processing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeGuard, Union, get_args, get_origin

from ..types.core import APIResponse, ModelParameters, ProviderInfo
from ..types.custom import (
    APIKey,
    BenchmarkName,
    EvaluationMethod,
    JSONSerializable,
    ModelName,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ProviderName,
    ValidatedEmail,
    ValidatedURL,
)
from ..types.evaluation import BenchmarkResult, EvaluationResult, MetricResult


# Basic type guards for primitive types with constraints
def is_non_empty_string(value: Any) -> TypeGuard[str]:
    """Check if value is a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0


def is_positive_int(value: Any) -> TypeGuard[PositiveInt]:
    """Check if value is a positive integer."""
    return isinstance(value, int) and value > 0


def is_non_negative_int(value: Any) -> TypeGuard[NonNegativeInt]:
    """Check if value is a non-negative integer."""
    return isinstance(value, int) and value >= 0


def is_positive_float(value: Any) -> TypeGuard[PositiveFloat]:
    """Check if value is a positive float."""
    return isinstance(value, (int, float)) and float(value) > 0.0


def is_non_negative_float(value: Any) -> TypeGuard[NonNegativeFloat]:
    """Check if value is a non-negative float."""
    return isinstance(value, (int, float)) and float(value) >= 0.0


def is_bounded_float(value: Any, min_val: float = 0.0, max_val: float = 1.0) -> TypeGuard[float]:
    """Check if value is a float within specified bounds."""
    return isinstance(value, (int, float)) and min_val <= float(value) <= max_val


def is_percentage(value: Any) -> TypeGuard[float]:
    """Check if value is a valid percentage (0.0 to 1.0)."""
    return is_bounded_float(value, 0.0, 1.0)


def is_probability(value: Any) -> TypeGuard[float]:
    """Check if value is a valid probability (0.0 to 1.0)."""
    return is_bounded_float(value, 0.0, 1.0)


def is_temperature(value: Any) -> TypeGuard[float]:
    """Check if value is a valid model temperature (0.0 to 2.0)."""
    return is_bounded_float(value, 0.0, 2.0)


# String pattern type guards
def is_valid_email(value: Any) -> TypeGuard[ValidatedEmail]:
    """Check if value is a valid email address."""
    if not isinstance(value, str):
        return False

    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(email_pattern, value))


def is_valid_url(value: Any) -> TypeGuard[ValidatedURL]:
    """Check if value is a valid URL."""
    if not isinstance(value, str):
        return False

    url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(url_pattern, value, re.IGNORECASE))


def is_valid_uuid(value: Any) -> TypeGuard[str]:
    """Check if value is a valid UUID string."""
    if not isinstance(value, str):
        return False

    uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    return bool(re.match(uuid_pattern, value, re.IGNORECASE))


def is_semantic_version(value: Any) -> TypeGuard[str]:
    """Check if value is a valid semantic version string."""
    if not isinstance(value, str):
        return False

    semver_pattern = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    return bool(re.match(semver_pattern, value))


def is_hex_color(value: Any) -> TypeGuard[str]:
    """Check if value is a valid hex color code."""
    if not isinstance(value, str):
        return False

    hex_pattern = r"^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$"
    return bool(re.match(hex_pattern, value))


def is_ip_address(value: Any) -> TypeGuard[str]:
    """Check if value is a valid IP address (IPv4 or IPv6)."""
    if not isinstance(value, str):
        return False

    # IPv4 pattern
    ipv4_pattern = r"^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}$"
    if re.match(ipv4_pattern, value):
        return True

    # IPv6 pattern (simplified)
    ipv6_pattern = r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$|^::1$|^::$"
    return bool(re.match(ipv6_pattern, value))


# Domain-specific type guards
def is_model_name(value: Any) -> TypeGuard[ModelName]:
    """Check if value is a valid model name."""
    if not isinstance(value, str):
        return False

    model_name = value.strip()
    if not model_name or len(model_name) > 100:
        return False

    # Valid characters: alphanumeric, hyphens, underscores, dots
    return bool(re.match(r"^[a-zA-Z0-9._-]+$", model_name))


def is_api_key(value: Any, provider: str = "unknown") -> TypeGuard[APIKey]:
    """Check if value is a valid API key format."""
    if not isinstance(value, str):
        return False

    key = value.strip()
    if len(key) < 10 or len(key) > 200:
        return False

    # Provider-specific validation
    if provider.lower() == "openai":
        return key.startswith("sk-")
    elif provider.lower() == "anthropic":
        return key.startswith("sk-ant-")

    return True


def is_provider_name(value: Any) -> TypeGuard[ProviderName]:
    """Check if value is a valid provider name."""
    valid_providers = {"openai", "anthropic", "google", "local", "custom"}
    return isinstance(value, str) and value.lower() in valid_providers


def is_evaluation_method(value: Any) -> TypeGuard[EvaluationMethod]:
    """Check if value is a valid evaluation method."""
    valid_methods = {
        "semantic_similarity",
        "exact_match",
        "fuzzy_match",
        "bleu",
        "rouge",
        "bert_score",
        "custom",
    }
    return isinstance(value, str) and value in valid_methods


def is_benchmark_name(value: Any) -> TypeGuard[BenchmarkName]:
    """Check if value is a valid benchmark name."""
    valid_benchmarks = {"glue", "superglue", "hellaswag", "arc", "mmlu", "custom"}
    return isinstance(value, str) and value in valid_benchmarks


# File system type guards
def is_existing_file_path(value: Any) -> TypeGuard[Path]:
    """Check if value is a path to an existing file."""
    try:
        if isinstance(value, (str, Path)):
            path = Path(value)
            return path.exists() and path.is_file()
    except Exception:
        pass
    return False


def is_existing_directory_path(value: Any) -> TypeGuard[Path]:
    """Check if value is a path to an existing directory."""
    try:
        if isinstance(value, (str, Path)):
            path = Path(value)
            return path.exists() and path.is_dir()
    except Exception:
        pass
    return False


def is_valid_file_path(value: Any) -> TypeGuard[Path]:
    """Check if value is a valid file path (may not exist)."""
    try:
        if isinstance(value, (str, Path)):
            path = Path(value)
            # Check if path is valid and parent directory exists or can be created
            return path.parent.exists() or path.is_absolute()
    except Exception:
        pass
    return False


# Collection type guards with element validation
def is_non_empty_list(value: Any) -> TypeGuard[list[Any]]:
    """Check if value is a non-empty list."""
    return isinstance(value, list) and len(value) > 0


def is_string_list(value: Any) -> TypeGuard[list[str]]:
    """Check if value is a list of strings."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def is_non_empty_string_list(value: Any) -> TypeGuard[list[str]]:
    """Check if value is a list of non-empty strings."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, str) and len(item.strip()) > 0 for item in value)
    )


def is_number_list(value: Any) -> TypeGuard[list[float | int]]:
    """Check if value is a list of numbers."""
    return isinstance(value, list) and all(isinstance(item, (int, float)) for item in value)


def is_dict_with_string_keys(value: Any) -> TypeGuard[dict[str, Any]]:
    """Check if value is a dictionary with string keys."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value.keys())


def is_json_serializable_dict(value: Any) -> TypeGuard[dict[str, JSONSerializable]]:
    """Check if value is a dictionary that can be JSON serialized."""
    if not isinstance(value, dict):
        return False

    def is_json_serializable_value(v: Any) -> bool:
        if v is None or isinstance(v, (bool, int, float, str)):
            return True
        if isinstance(v, list):
            return all(is_json_serializable_value(item) for item in v)
        if isinstance(v, dict):
            return all(isinstance(k, str) for k in v.keys()) and all(
                is_json_serializable_value(val) for val in v.values()
            )
        return False

    return all(isinstance(k, str) and is_json_serializable_value(v) for k, v in value.items())


# Complex type validation
def is_model_parameters(value: Any) -> TypeGuard[ModelParameters]:
    """Check if value is a valid ModelParameters dict."""
    if not isinstance(value, dict):
        return False

    # Check required fields exist and have correct types
    temperature = value.get("temperature")
    if temperature is not None and not is_temperature(temperature):
        return False

    max_tokens = value.get("max_tokens")
    if max_tokens is not None and not is_positive_int(max_tokens):
        return False

    top_p = value.get("top_p")
    if top_p is not None and not is_bounded_float(top_p, 0.0, 1.0):
        return False

    return True


def is_provider_info(value: Any) -> TypeGuard[ProviderInfo]:
    """Check if value matches ProviderInfo structure."""
    if not isinstance(value, dict):
        return False

    required_fields = ["model_name", "provider", "max_tokens"]
    if not all(field in value for field in required_fields):
        return False

    return (
        is_model_name(value["model_name"])
        and is_provider_name(value["provider"])
        and is_positive_int(value["max_tokens"])
    )


def is_evaluation_result(value: Any) -> TypeGuard[EvaluationResult]:
    """Check if value matches EvaluationResult structure."""
    if not isinstance(value, dict):
        return False

    required_fields = ["score", "method"]
    if not all(field in value for field in required_fields):
        return False

    return isinstance(value["score"], (int, float)) and isinstance(value["method"], str)


# Generic type validation with runtime type checking
def is_instance_of_generic(value: Any, generic_type: type) -> bool:
    """Check if value is instance of a generic type at runtime.

    Note: This provides best-effort checking for generic types.
    """
    origin = get_origin(generic_type)
    if origin is None:
        return isinstance(value, generic_type)

    # Handle common generic types
    if origin is list:
        if not isinstance(value, list):
            return False
        args = get_args(generic_type)
        if args:
            return all(isinstance(item, args[0]) for item in value)
        return True

    if origin is dict:
        if not isinstance(value, dict):
            return False
        args = get_args(generic_type)
        if len(args) >= 2:
            key_type, value_type = args[0], args[1]
            return all(isinstance(k, key_type) for k in value.keys()) and all(
                isinstance(v, value_type) for v in value.values()
            )
        return True

    if origin is Union:
        args = get_args(generic_type)
        return any(isinstance(value, arg) for arg in args if arg is not type(None))

    # Fallback to basic isinstance check
    return isinstance(value, origin)


def validate_against_literal(value: Any, literal_type: type) -> bool:
    """Validate value against a Literal type."""
    args = get_args(literal_type)
    return value in args if args else False


# Composite validation functions
def validate_configuration_section(value: Any, section_name: str) -> bool:
    """Validate a configuration section based on its name."""
    if not isinstance(value, dict):
        return False

    if section_name == "providers":
        return all(is_provider_name(key) for key in value.keys())
    elif section_name == "model_parameters":
        return is_model_parameters(value)
    elif section_name == "network":
        timeout = value.get("timeout")
        return timeout is None or is_positive_int(timeout)

    return True  # Default: allow any dict for unknown sections


def validate_batch_data(values: list[Any], validator: TypeGuard) -> bool:
    """Validate a batch of values using a TypeGuard function."""
    return all(validator(value) for value in values)


# Context-aware type guards
def is_valid_in_context(value: Any, context: dict[str, Any]) -> bool:
    """Validate a value based on contextual information."""
    value_type = context.get("expected_type")
    constraints = context.get("constraints", {})

    if value_type == "model_name":
        return is_model_name(value)
    elif value_type == "api_key":
        provider = constraints.get("provider", "unknown")
        return is_api_key(value, provider)
    elif value_type == "positive_number":
        return is_positive_int(value) or is_positive_float(value)
    elif value_type == "bounded_float":
        min_val = constraints.get("min", 0.0)
        max_val = constraints.get("max", 1.0)
        return is_bounded_float(value, min_val, max_val)

    return True  # Default: allow anything for unknown contexts


# Performance-optimized type guards for high-frequency checks
def is_simple_string(value: Any) -> TypeGuard[str]:
    """Fast string check without additional validation."""
    return type(value) is str


def is_simple_int(value: Any) -> TypeGuard[int]:
    """Fast integer check without range validation."""
    return type(value) is int


def is_simple_float(value: Any) -> TypeGuard[float]:
    """Fast float check without range validation."""
    return type(value) is float


def is_simple_bool(value: Any) -> TypeGuard[bool]:
    """Fast boolean check."""
    return type(value) is bool


def is_simple_list(value: Any) -> TypeGuard[list[Any]]:
    """Fast list check without element validation."""
    return type(value) is list


def is_simple_dict(value: Any) -> TypeGuard[dict[str, Any]]:
    """Fast dict check without key/value validation."""
    return type(value) is dict
