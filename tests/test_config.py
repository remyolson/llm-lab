"""Tests for config module."""

import os
from unittest.mock import patch

import pytest

from config import (
    MODEL_DEFAULTS,
    ConfigurationError,
    MissingAPIKeyError,
    get_api_key,
    get_model_config,
    validate_config,
)


class TestGetAPIKey:
    """Test get_api_key function."""

    def test_get_api_key_success(self):
        """Test successful API key retrieval."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key-123"}):
            key = get_api_key("GOOGLE")
            assert key == "test-key-123"

    def test_get_api_key_missing(self):
        """Test missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MissingAPIKeyError) as exc_info:
                get_api_key("GOOGLE")
            assert "GOOGLE_API_KEY" in str(exc_info.value)

    def test_get_api_key_unknown_provider(self):
        """Test unknown provider raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_api_key("UNKNOWN_PROVIDER")
        assert "Unknown provider" in str(exc_info.value)


class TestGetModelConfig:
    """Test get_model_config function."""

    def test_get_model_config_defaults(self):
        """Test default model configuration."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_model_config()
            assert config == MODEL_DEFAULTS

    def test_get_model_config_env_override(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {"MODEL_TEMPERATURE": "0.5", "MODEL_MAX_TOKENS": "2000"}):
            config = get_model_config()
            assert config["temperature"] == 0.5
            assert config["max_tokens"] == 2000
            # Other values should remain default
            assert config["top_p"] == MODEL_DEFAULTS["top_p"]

    def test_get_model_config_invalid_env_value(self):
        """Test invalid environment value handling."""
        with patch.dict(os.environ, {"MODEL_TEMPERATURE": "invalid"}):
            with pytest.raises(Exception):  # Should raise on invalid conversion
                get_model_config()


class TestValidateConfig:
    """Test validate_config function."""

    def test_validate_config_valid(self):
        """Test validation with valid config."""
        valid_config = {
            "model": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 0.95,
                "top_k": 40,
                "timeout_seconds": 30,
                "max_retries": 3,
                "retry_delay": 1.0,
            },
            "output_dir": "./results/",
        }
        errors = validate_config(valid_config)
        assert errors == []

    def test_validate_config_invalid_temperature(self):
        """Test validation with invalid temperature."""
        invalid_config = {
            "model": {
                "temperature": 3.0,  # Out of range [0, 2]
                "max_tokens": 1000,
            }
        }
        errors = validate_config(invalid_config)
        assert len(errors) > 0
        assert any("Temperature" in error for error in errors)

    def test_validate_config_invalid_max_tokens(self):
        """Test validation with invalid max_tokens."""
        invalid_config = {
            "model": {
                "temperature": 0.7,
                "max_tokens": 200000,  # Too high
            }
        }
        errors = validate_config(invalid_config)
        assert len(errors) > 0
        assert any("Max tokens" in error for error in errors)

    @patch("os.makedirs")
    def test_validate_config_output_dir_error(self, mock_makedirs):
        """Test validation with output directory creation error."""
        mock_makedirs.side_effect = PermissionError("Permission denied")

        config = {"model": MODEL_DEFAULTS, "output_dir": "/invalid/path"}
        errors = validate_config(config)
        assert len(errors) > 0
        assert any("output directory" in error.lower() for error in errors)
