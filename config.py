"""
Configuration Management Module

This module handles all configuration aspects of the LLM Lab framework, including:
- Loading environment variables from .env files
- Retrieving and validating API keys
- Managing model configuration parameters
- Providing default configurations
- Custom exception handling for configuration errors

The module uses python-dotenv to load environment variables and provides
a centralized way to manage all configuration settings.
"""

import os

from dotenv import load_dotenv


# Custom exceptions for configuration errors
class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class MissingAPIKeyError(ConfigurationError):
    """Raised when a required API key is not found."""
    def __init__(self, key_name: str):
        self.key_name = key_name
        super().__init__(f"Required API key '{key_name}' not found in environment variables. "
                         f"Please set {key_name} in your .env file or environment.")


class InvalidConfigValueError(ConfigurationError):
    """Raised when a configuration value is invalid."""
    def __init__(self, key: str, value: any, reason: str):
        self.key = key
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid configuration value for '{key}': {value}. {reason}")


class ConfigFileError(ConfigurationError):
    """Raised when there's an error reading or parsing a configuration file."""
    def __init__(self, filepath: str, error: str):
        self.filepath = filepath
        self.error = error
        super().__init__(f"Error reading configuration file '{filepath}': {error}")


# Load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    # Don't fail if .env file doesn't exist, but log the issue
    print(f"Warning: Could not load .env file: {e}")

# Constants
DEFAULT_MODEL = 'gemini-1.5-flash'
OUTPUT_DIR = './results/'
BENCHMARK_NAME = 'truthfulness'

# Model configuration defaults
MODEL_DEFAULTS = {
    'temperature': 0.7,
    'max_tokens': 1000,
    'top_p': 1.0,
    'top_k': 40,
    'timeout_seconds': 30,
    'max_retries': 3,
    'retry_delay': 1.0
}


def _convert_env_value(value: str, expected_type: type):
    """Convert environment variable string to expected type."""
    try:
        if expected_type == float:
            return float(value)
        elif expected_type == int:
            return int(value)
        elif expected_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        else:
            return value
    except ValueError:
        return None


def get_model_config():
    """
    Get model configuration parameters with environment variable overrides.
    
    Environment variables should be prefixed with MODEL_ (e.g., MODEL_TEMPERATURE).
    
    Returns:
        dict: Model configuration parameters
    """
    config = MODEL_DEFAULTS.copy()

    # Check for environment variable overrides
    for param, default_value in MODEL_DEFAULTS.items():
        env_key = f'MODEL_{param.upper()}'
        env_value = os.getenv(env_key)

        if env_value is not None:
            converted_value = _convert_env_value(env_value, type(default_value))
            if converted_value is not None:
                config[param] = converted_value
            else:
                raise InvalidConfigValueError(
                    env_key,
                    env_value,
                    f"Could not convert to {type(default_value).__name__}"
                )

    return config


def get_provider_config():
    """
    Get provider configuration with API key from environment variables.
    
    Returns:
        dict: Configuration dictionary with 'api_key' field
        
    Raises:
        MissingAPIKeyError: If GOOGLE_API_KEY environment variable is not set
    """
    api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key:
        raise MissingAPIKeyError('GOOGLE_API_KEY')

    return {
        'api_key': api_key
    }


def get_full_config():
    """
    Get complete configuration including provider and model settings.
    
    Returns:
        dict: Complete configuration with 'provider' and 'model' sections
        
    Raises:
        MissingAPIKeyError: If required API key is not set
        InvalidConfigValueError: If any configuration value is invalid
    """
    return {
        'provider': get_provider_config(),
        'model': get_model_config(),
        'output_dir': OUTPUT_DIR,
        'benchmark_name': BENCHMARK_NAME,
        'default_model': DEFAULT_MODEL
    }


def validate_config(config=None):
    """
    Validate configuration values and collect all errors.
    
    Args:
        config: Configuration dict to validate (if None, loads current config)
        
    Returns:
        list: List of validation error messages (empty if valid)
    """
    errors = []

    if config is None:
        try:
            config = get_full_config()
        except ConfigurationError as e:
            return [str(e)]

    # Validate model parameters
    model_config = config.get('model', {})

    # Temperature validation
    temp = model_config.get('temperature', 0.7)
    if not 0 <= temp <= 2:
        errors.append(f"Temperature {temp} is out of valid range [0, 2]")

    # Max tokens validation
    max_tokens = model_config.get('max_tokens', 1000)
    if not 1 <= max_tokens <= 100000:
        errors.append(f"Max tokens {max_tokens} is out of valid range [1, 100000]")

    # Top-p validation
    top_p = model_config.get('top_p', 1.0)
    if not 0 <= top_p <= 1:
        errors.append(f"Top-p {top_p} is out of valid range [0, 1]")

    # Top-k validation
    top_k = model_config.get('top_k', 40)
    if not 1 <= top_k <= 100:
        errors.append(f"Top-k {top_k} is out of valid range [1, 100]")

    # Timeout validation
    timeout = model_config.get('timeout_seconds', 30)
    if timeout <= 0:
        errors.append(f"Timeout {timeout} must be positive")

    # Retry validation
    max_retries = model_config.get('max_retries', 3)
    if max_retries < 0:
        errors.append(f"Max retries {max_retries} cannot be negative")

    retry_delay = model_config.get('retry_delay', 1.0)
    if retry_delay < 0:
        errors.append(f"Retry delay {retry_delay} cannot be negative")

    # Validate output directory
    output_dir = config.get('output_dir', './results/')
    if not output_dir:
        errors.append("Output directory cannot be empty")

    # Check if output directory can be created
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory '{output_dir}': {e}")

    return errors


def get_api_key(provider_name: str) -> str:
    """
    Get API key for a specific provider.
    
    Args:
        provider_name: Provider name (e.g., 'GOOGLE', 'OPENAI')
        
    Returns:
        str: API key for the provider
        
    Raises:
        MissingAPIKeyError: If API key is not found
    """
    # Map provider names to environment variable names
    provider_key_map = {
        'GOOGLE': 'GOOGLE_API_KEY',
        'OPENAI': 'OPENAI_API_KEY',
        'ANTHROPIC': 'ANTHROPIC_API_KEY',
        # Add more providers as needed
    }

    env_key = provider_key_map.get(provider_name.upper())
    if not env_key:
        raise ConfigurationError(f"Unknown provider: {provider_name}")

    api_key = os.getenv(env_key)
    if not api_key:
        raise MissingAPIKeyError(env_key)

    return api_key
