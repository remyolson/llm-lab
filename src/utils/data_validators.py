"""
Data Validation Utilities

This module provides common validation patterns used throughout the LLM Lab.
It centralizes input validation, type checking, and parameter validation logic.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import os
import re
from typing import Any, Dict, List, Optional

from .validation import ValidationError


def validate_api_key(
    api_key: str | None,
    provider_name: str,
    key_prefix: str | None = None,
    env_var_name: str | None = None,
) -> str:
    """
    Validate API key with common patterns across providers.

    Args:
        api_key: The API key to validate (can be None to check env var)
        provider_name: Name of the provider for error messages
        key_prefix: Expected prefix for the API key (e.g., 'sk-' for OpenAI)
        env_var_name: Environment variable name to check if api_key is None

    Returns:
        Valid API key

    Raises:
        ValidationError: If API key is invalid or missing
    """
    # Get from environment if not provided
    if api_key is None and env_var_name:
        api_key = os.getenv(env_var_name)

    if not api_key:
        env_msg = f" (environment variable: {env_var_name})" if env_var_name else ""
        raise ValidationError(f"{provider_name} API key not provided{env_msg}")

    if not isinstance(api_key, str):
        raise ValidationError(f"{provider_name} API key must be a string")

    api_key = api_key.strip()
    if not api_key:
        raise ValidationError(f"{provider_name} API key cannot be empty")

    # Check prefix if specified
    if key_prefix and not api_key.startswith(key_prefix):
        raise ValidationError(
            f"{provider_name} API key should start with '{key_prefix}', "
            f"got key starting with '{api_key[:10]}...'"
        )

    return api_key


def validate_text_input(
    text: Any,
    field_name: str = "text",
    min_length: int = 1,
    max_length: int | None = None,
    allow_empty: bool = False,
) -> str:
    """
    Validate text input with common patterns.

    Args:
        text: Text to validate
        field_name: Name of the field for error messages
        min_length: Minimum required length
        max_length: Maximum allowed length
        allow_empty: Whether to allow empty strings

    Returns:
        Validated and cleaned text

    Raises:
        ValidationError: If text is invalid
    """
    if text is None:
        if allow_empty:
            return ""
        raise ValidationError(f"{field_name} cannot be None")

    if not isinstance(text, str):
        raise ValidationError(f"{field_name} must be a string, got {type(text).__name__}")

    text = text.strip()

    if not text and not allow_empty:
        raise ValidationError(f"{field_name} cannot be empty")

    if len(text) < min_length:
        raise ValidationError(f"{field_name} must be at least {min_length} characters long")

    if max_length and len(text) > max_length:
        raise ValidationError(f"{field_name} must be no more than {max_length} characters long")

    return text


def validate_numeric_range(
    value: Any,
    field_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
    allow_none: bool = False,
) -> float | None:
    """
    Validate numeric values within specified ranges.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        allow_none: Whether None values are allowed

    Returns:
        Validated numeric value

    Raises:
        ValidationError: If value is invalid or out of range
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(f"{field_name} cannot be None")

    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name} must be a number, got {type(value).__name__}")

    if min_value is not None and value < min_value:
        raise ValidationError(f"{field_name} must be >= {min_value}, got {value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{field_name} must be <= {max_value}, got {value}")

    return value


def validate_model_name(model_name: Any, supported_models: List[str | None] = None) -> str:
    """
    Validate model name with common patterns.

    Args:
        model_name: Model name to validate
        supported_models: List of supported model names

    Returns:
        Validated model name

    Raises:
        ValidationError: If model name is invalid or unsupported
    """
    if not isinstance(model_name, str):
        raise ValidationError(f"Model name must be a string, got {type(model_name).__name__}")

    model_name = model_name.strip()
    if not model_name:
        raise ValidationError("Model name cannot be empty")

    # Basic format validation
    if len(model_name) > 100:
        raise ValidationError("Model name too long (max 100 characters)")

    # Check against supported models if provided
    if supported_models and model_name not in supported_models:
        raise ValidationError(
            f"Model '{model_name}' not supported. Supported models: {', '.join(supported_models)}"
        )

    return model_name


def validate_generation_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate common generation parameters across providers.

    Args:
        params: Dictionary of generation parameters

    Returns:
        Validated parameters dictionary

    Raises:
        ValidationError: If any parameter is invalid
    """
    validated = {}

    # Temperature validation
    if "temperature" in params:
        validated["temperature"] = validate_numeric_range(
            params["temperature"], "temperature", 0.0, 2.0, allow_none=True
        )

    # Max tokens validation
    if "max_tokens" in params:
        max_tokens = params["max_tokens"]
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ValidationError("max_tokens must be a positive integer")
            if max_tokens > 100000:  # Reasonable upper limit
                raise ValidationError("max_tokens too large (max 100,000)")
        validated["max_tokens"] = max_tokens

    # Top-p validation
    if "top_p" in params:
        validated["top_p"] = validate_numeric_range(
            params["top_p"], "top_p", 0.0, 1.0, allow_none=True
        )

    # Top-k validation
    if "top_k" in params:
        top_k = params["top_k"]
        if top_k is not None:
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValidationError("top_k must be a positive integer")
        validated["top_k"] = top_k

    # Copy other parameters as-is
    for key, value in params.items():
        if key not in validated:
            validated[key] = value

    return validated


def validate_list_of_strings(
    items: Any,
    field_name: str,
    min_items: int = 0,
    max_items: int | None = None,
    allow_empty_strings: bool = False,
) -> List[str]:
    """
    Validate a list of strings with common patterns.

    Args:
        items: Items to validate
        field_name: Name of the field for error messages
        min_items: Minimum number of items required
        max_items: Maximum number of items allowed
        allow_empty_strings: Whether empty strings are allowed in the list

    Returns:
        Validated list of strings

    Raises:
        ValidationError: If the list or its items are invalid
    """
    if not isinstance(items, list):
        raise ValidationError(f"{field_name} must be a list, got {type(items).__name__}")

    if len(items) < min_items:
        raise ValidationError(f"{field_name} must contain at least {min_items} items")

    if max_items and len(items) > max_items:
        raise ValidationError(f"{field_name} must contain no more than {max_items} items")

    validated_items = []
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise ValidationError(f"{field_name}[{i}] must be a string, got {type(item).__name__}")

        item = item.strip()
        if not item and not allow_empty_strings:
            raise ValidationError(f"{field_name}[{i}] cannot be empty")

        validated_items.append(item)

    return validated_items


def validate_file_path(
    file_path: Any,
    field_name: str = "file_path",
    must_exist: bool = False,
    must_be_file: bool = False,
    allowed_extensions: List[str | None] = None,
) -> str:
    """
    Validate file path with common patterns.

    Args:
        file_path: File path to validate
        field_name: Name of the field for error messages
        must_exist: Whether the file must exist
        must_be_file: Whether the path must be a file (not directory)
        allowed_extensions: List of allowed file extensions (with dots)

    Returns:
        Validated file path

    Raises:
        ValidationError: If file path is invalid
    """
    if not isinstance(file_path, str):
        raise ValidationError(f"{field_name} must be a string, got {type(file_path).__name__}")

    file_path = file_path.strip()
    if not file_path:
        raise ValidationError(f"{field_name} cannot be empty")

    # Check if file exists
    if must_exist and not os.path.exists(file_path):
        raise ValidationError(f"{field_name} does not exist: {file_path}")

    # Check if it's a file
    if must_be_file and os.path.exists(file_path) and not os.path.isfile(file_path):
        raise ValidationError(f"{field_name} must be a file, not a directory: {file_path}")

    # Check file extension
    if allowed_extensions:
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in [e.lower() for e in allowed_extensions]:
            raise ValidationError(
                f"{field_name} must have one of these extensions: {', '.join(allowed_extensions)}"
            )

    return file_path


def validate_email(email: Any, field_name: str = "email") -> str:
    """
    Validate email address format.

    Args:
        email: Email address to validate
        field_name: Name of the field for error messages

    Returns:
        Validated email address

    Raises:
        ValidationError: If email is invalid
    """
    if not isinstance(email, str):
        raise ValidationError(f"{field_name} must be a string, got {type(email).__name__}")

    email = email.strip().lower()
    if not email:
        raise ValidationError(f"{field_name} cannot be empty")

    # Basic email pattern validation
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, email):
        raise ValidationError(f"Invalid {field_name} format: {email}")

    return email
