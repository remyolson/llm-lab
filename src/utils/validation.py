"""
Input validation utilities for LLM Lab.

This module provides common validation functions for user inputs,
file paths, API keys, and other parameters throughout the system.
"""

import os
import re
from typing import Any, List, Dict, Optional, Union
from pathlib import Path


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_model_name(model_name: str) -> str:
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
    if not re.match(r'^[a-zA-Z0-9._-]+$', model_name):
        raise ValidationError("Model name contains invalid characters (only alphanumeric, ., -, _ allowed)")
    
    return model_name


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
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


def validate_directory_path(dir_path: Union[str, Path], create_if_missing: bool = False) -> Path:
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


def validate_api_key(api_key: str, provider: str = "unknown") -> str:
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


def validate_positive_number(value: Union[int, float], name: str = "value") -> Union[int, float]:
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


def validate_range(value: Union[int, float], min_val: Union[int, float], 
                   max_val: Union[int, float], name: str = "value") -> Union[int, float]:
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


def validate_prompt(prompt: str, max_length: int = 10000) -> str:
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
        r'<script[^>]*>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'exec\s*\(',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            raise ValidationError("Prompt contains potentially unsafe content")
    
    return prompt


def validate_config_dict(config: Dict[str, Any], required_keys: List[str], 
                        optional_keys: List[str] = None) -> Dict[str, Any]:
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