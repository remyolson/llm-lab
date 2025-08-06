"""
Input validation utilities for LLM Lab.

This module provides common validation functions for user inputs,
file paths, API keys, and other parameters throughout the system.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeGuard, TypeVar, Union, overload

from typing_extensions import Literal, ParamSpec

from ..types.custom import (
    APIKey,
    ModelName,
    NormalizedString,
    PositiveFloat,
    PositiveInt,
    ProviderName,
    ValidatedDirectoryPath,
    ValidatedFilePath,
)


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def validate_model_name(model_name: str) -> ModelName:
    """
    Validate and normalize model name.

    Args:
        model_name: The model name to validate

    Returns:
        Normalized model name

    Raises:
        ValidationError: If model name is invalid
    """
    if not isinstance(model_name, str):
        raise ValidationError(f"Model name must be a string, got {type(model_name)}")

    model_name = model_name.strip()
    if not model_name:
        raise ValidationError("Model name cannot be empty")

    if len(model_name) > 100:
        raise ValidationError("Model name too long (max 100 characters)")

    # Check for valid characters (alphanumeric, hyphens, underscores, dots)
    if not re.match(r"^[a-zA-Z0-9._-]+$", model_name):
        raise ValidationError(
            "Model name contains invalid characters (only alphanumeric, ., -, _ allowed)"
        )

    return model_name


def validate_file_path(file_path: str | Path, must_exist: bool = True) -> ValidatedFilePath:
    """
    Validate file path.

    Args:
        file_path: Path to validate
        must_exist: Whether the file must already exist

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path is invalid
    """
    if not file_path:
        raise ValidationError("File path cannot be empty")

    try:
        path = Path(file_path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid file path: {e}")

    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {path}")

    if must_exist and not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")

    return path


def validate_directory_path(
    dir_path: str | Path, create_if_missing: bool = False
) -> ValidatedDirectoryPath:
    """
    Validate directory path.

    Args:
        dir_path: Directory path to validate
        create_if_missing: Whether to create directory if it doesn't exist

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path is invalid
    """
    if not dir_path:
        raise ValidationError("Directory path cannot be empty")

    try:
        path = Path(dir_path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid directory path: {e}")

    if not path.exists():
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Cannot create directory {path}: {e}")
        else:
            raise ValidationError(f"Directory does not exist: {path}")

    if path.exists() and not path.is_dir():
        raise ValidationError(f"Path is not a directory: {path}")

    return path


def validate_api_key(api_key: str, provider: ProviderName | str = "unknown") -> APIKey:
    """
    Validate API key format.

    Args:
        api_key: API key to validate
        provider: Provider name for specific validation rules

    Returns:
        Validated API key

    Raises:
        ValidationError: If API key is invalid
    """
    if not isinstance(api_key, str):
        raise ValidationError(f"API key must be a string, got {type(api_key)}")

    api_key = api_key.strip()
    if not api_key:
        raise ValidationError("API key cannot be empty")

    if len(api_key) < 10:
        raise ValidationError("API key too short (minimum 10 characters)")

    if len(api_key) > 200:
        raise ValidationError("API key too long (maximum 200 characters)")

    # Provider-specific validation
    if provider.lower() == "openai":
        if not api_key.startswith("sk-"):
            raise ValidationError("OpenAI API key must start with 'sk-'")
    elif provider.lower() == "anthropic":
        if not api_key.startswith("sk-ant-"):
            raise ValidationError("Anthropic API key must start with 'sk-ant-'")

    return api_key


@overload
def validate_positive_number(value: int, name: str = "value") -> PositiveInt: ...


@overload
def validate_positive_number(value: float, name: str = "value") -> PositiveFloat: ...


def validate_positive_number(
    value: int | float, name: str = "value"
) -> PositiveInt | PositiveFloat:
    """
    Validate that a number is positive.

    Args:
        value: Number to validate
        name: Name of the parameter for error messages

    Returns:
        Validated number

    Raises:
        ValidationError: If number is not positive
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")

    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")

    return value


def validate_range(
    value: int | float,
    min_val: int | float,
    max_val: int | float,
    name: str = "value",
) -> int | float:
    """
    Validate that a number is within a specific range.

    Args:
        value: Number to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the parameter for error messages

    Returns:
        Validated number

    Raises:
        ValidationError: If number is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")

    if value < min_val or value > max_val:
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")

    return value


def validate_prompt(prompt: str, max_length: int = 10000) -> NormalizedString:
    """
    Validate user prompt input.

    Args:
        prompt: Prompt text to validate
        max_length: Maximum allowed length

    Returns:
        Validated prompt

    Raises:
        ValidationError: If prompt is invalid
    """
    if not isinstance(prompt, str):
        raise ValidationError(f"Prompt must be a string, got {type(prompt)}")

    prompt = prompt.strip()
    if not prompt:
        raise ValidationError("Prompt cannot be empty")

    if len(prompt) > max_length:
        raise ValidationError(f"Prompt too long (max {max_length} characters)")

    # Check for potentially harmful content
    dangerous_patterns = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
        r"eval\s*\(",
        r"exec\s*\(",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            raise ValidationError("Prompt contains potentially unsafe content")

    return prompt


def validate_config_dict(
    config: Dict[str, Any], required_keys: List[str], optional_keys: List[str | None] = None
) -> Dict[str | Any]:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys
        optional_keys: List of optional keys (if None, allows any additional keys)

    Returns:
        Validated configuration

    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Configuration must be a dictionary, got {type(config)}")

    # Check required keys
    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        raise ValidationError(f"Missing required configuration keys: {sorted(missing_keys)}")

    # Check for unexpected keys if optional_keys is specified
    if optional_keys is not None:
        allowed_keys = set(required_keys + optional_keys)
        unexpected_keys = set(config.keys()) - allowed_keys
        if unexpected_keys:
            raise ValidationError(f"Unexpected configuration keys: {sorted(unexpected_keys)}")

    return config


# TypeGuard functions for runtime type checking


def is_valid_model_name(value: Any) -> TypeGuard[ModelName]:
    """Type guard to check if a value is a valid model name at runtime.

    Args:
        value: Value to check

    Returns:
        True if value is a valid model name, False otherwise
    """
    try:
        if isinstance(value, str):
            validate_model_name(value)
            return True
    except ValidationError:
        pass
    return False


def is_valid_api_key(value: Any, provider: ProviderName | str = "unknown") -> TypeGuard[APIKey]:
    """Type guard to check if a value is a valid API key at runtime.

    Args:
        value: Value to check
        provider: Provider name for specific validation

    Returns:
        True if value is a valid API key, False otherwise
    """
    try:
        if isinstance(value, str):
            validate_api_key(value, provider)
            return True
    except ValidationError:
        pass
    return False


def is_positive_number(value: Any) -> TypeGuard[PositiveInt | PositiveFloat]:
    """Type guard to check if a value is a positive number at runtime.

    Args:
        value: Value to check

    Returns:
        True if value is a positive number, False otherwise
    """
    try:
        if isinstance(value, (int, float)):
            validate_positive_number(value)
            return True
    except ValidationError:
        pass
    return False


def is_valid_file_path(value: Any, must_exist: bool = True) -> TypeGuard[ValidatedFilePath]:
    """Type guard to check if a value is a valid file path at runtime.

    Args:
        value: Value to check
        must_exist: Whether file must exist

    Returns:
        True if value is a valid file path, False otherwise
    """
    try:
        if isinstance(value, (str, Path)):
            validate_file_path(value, must_exist)
            return True
    except ValidationError:
        pass
    return False


def is_valid_directory_path(
    value: Any, create_if_missing: bool = False
) -> TypeGuard[ValidatedDirectoryPath]:
    """Type guard to check if a value is a valid directory path at runtime.

    Args:
        value: Value to check
        create_if_missing: Whether to create if missing

    Returns:
        True if value is a valid directory path, False otherwise
    """
    try:
        if isinstance(value, (str, Path)):
            validate_directory_path(value, create_if_missing)
            return True
    except ValidationError:
        pass
    return False


def is_valid_range(
    value: Any, min_val: int | float, max_val: int | float
) -> TypeGuard[int | float]:
    """Type guard to check if a value is within a valid range at runtime.

    Args:
        value: Value to check
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        True if value is in range, False otherwise
    """
    try:
        if isinstance(value, (int, float)):
            validate_range(value, min_val, max_val)
            return True
    except ValidationError:
        pass
    return False


def is_non_empty_string(value: Any) -> TypeGuard[str]:
    """Type guard to check if a value is a non-empty string.

    Args:
        value: Value to check

    Returns:
        True if value is a non-empty string, False otherwise
    """
    return isinstance(value, str) and len(value.strip()) > 0


def is_valid_config_dict(value: Any, required_keys: List[str]) -> TypeGuard[Dict[str, Any]]:
    """Type guard to check if a value is a valid configuration dictionary.

    Args:
        value: Value to check
        required_keys: Required keys that must be present

    Returns:
        True if value is a valid config dict, False otherwise
    """
    try:
        if isinstance(value, dict):
            validate_config_dict(value, required_keys)
            return True
    except ValidationError:
        pass
    return False


# Generic type-safe validation functions with ParamSpec
P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


def create_validator(validation_func: Callable[[T], R]) -> Callable[[T], R]:
    """Create a type-safe validator wrapper.

    Args:
        validation_func: Function that validates and returns a value

    Returns:
        Wrapped validation function with better error handling
    """

    def wrapper(value: T) -> R:
        try:
            return validation_func(value)
        except ValidationError as e:
            # Re-raise with more context
            raise ValidationError(f"Validation failed for {type(value).__name__}: {e}")
        except Exception as e:
            # Convert unexpected errors to ValidationError
            raise ValidationError(f"Unexpected validation error for {type(value).__name__}: {e}")

    return wrapper


def validate_with_callback(
    value: T,
    validator: Callable[[T], R],
    on_success: Callable[[R], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> R:
    """Validate a value with optional success and error callbacks.

    Args:
        value: Value to validate
        validator: Validation function
        on_success: Callback for successful validation
        on_error: Callback for validation errors

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails
    """
    try:
        result = validator(value)
        if on_success:
            on_success(result)
        return result
    except Exception as e:
        if on_error:
            on_error(e)
        raise


def batch_validate(
    values: List[T], validator: Callable[[T], R], stop_on_first_error: bool = False
) -> List[R] | Dict[int, Exception]:
    """Validate a batch of values with a single validator.

    Args:
        values: List of values to validate
        validator: Validation function
        stop_on_first_error: Whether to stop on first error

    Returns:
        List of validated values if all succeed, or dict of errors by index
    """
    validated = []
    errors = {}

    for i, value in enumerate(values):
        try:
            validated.append(validator(value))
        except Exception as e:
            errors[i] = e
            if stop_on_first_error:
                break

    if errors:
        return errors
    return validated


def chain_validators(*validators: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Chain multiple validators together.

    Args:
        *validators: Validation functions to chain

    Returns:
        Combined validation function
    """

    def chained_validator(value: Any) -> Any:
        result = value
        for validator in validators:
            result = validator(result)
        return result

    return chained_validator


# Conditional validation functions
def validate_if(
    value: T, condition: Callable[[T], bool], validator: Callable[[T], R], default: R | None = None
) -> R | T:
    """Validate a value only if a condition is met.

    Args:
        value: Value to potentially validate
        condition: Condition function
        validator: Validation function
        default: Default value if condition is not met

    Returns:
        Validated value if condition is met, otherwise original value or default
    """
    if condition(value):
        return validator(value)
    return default if default is not None else value


def optional_validate(value: T | None, validator: Callable[[T], R]) -> R | None:
    """Validate an optional value.

    Args:
        value: Optional value to validate
        validator: Validation function

    Returns:
        Validated value if not None, otherwise None
    """
    if value is None:
        return None
    return validator(value)


# Validation context manager
class ValidationContext:
    """Context manager for collecting validation errors."""

    def __init__(self, raise_on_exit: bool = True):
        self.errors: List[str] = []
        self.raise_on_exit = raise_on_exit

    def __enter__(self) -> "ValidationContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.errors and self.raise_on_exit:
            error_msg = "Validation failed with errors: " + "; ".join(self.errors)
            raise ValidationError(error_msg)
        return False

    def validate(
        self, value: Any, validator: Callable[[Any], Any], field_name: str = "field"
    ) -> Any:
        """Validate a value and collect any errors.

        Args:
            value: Value to validate
            validator: Validation function
            field_name: Name of the field being validated

        Returns:
            Validated value or original value if validation failed
        """
        try:
            return validator(value)
        except Exception as e:
            self.errors.append(f"{field_name}: {e}")
            return value

    def has_errors(self) -> bool:
        """Check if any validation errors occurred."""
        return len(self.errors) > 0


# Type-specific validation decorators
def ensure_type(expected_type: type) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to ensure function arguments are of expected type.

    Args:
        expected_type: Expected type for the first argument

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(value: Any, *args, **kwargs) -> Any:
            if not isinstance(value, expected_type):
                raise ValidationError(
                    f"Expected {expected_type.__name__}, got {type(value).__name__}"
                )
            return func(value, *args, **kwargs)

        return wrapper

    return decorator


def validate_args(
    **type_validators: Callable[[Any], Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to validate function arguments by name.

    Args:
        **type_validators: Mapping of argument names to validators

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs) -> Any:
            # Get function signature for argument mapping
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate specified arguments
            for arg_name, validator in type_validators.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    try:
                        bound_args.arguments[arg_name] = validator(value)
                    except Exception as e:
                        raise ValidationError(f"Validation failed for argument '{arg_name}': {e}")

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator
