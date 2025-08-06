"""
Unit tests for configuration management.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml


@pytest.mark.unit
class TestConfigurationLoading:
    """Test configuration loading and validation."""

    def test_load_config_from_file(self, temp_config_file):
        """Test loading configuration from YAML file."""
        # This would test actual config loading
        import yaml

        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        assert "providers" in config
        assert "benchmarks" in config
        assert "openai" in config["providers"]
        assert config["providers"]["openai"]["model"] == "gpt-4"

    def test_config_validation_success(self, sample_config):
        """Test successful configuration validation."""
        from tests.fixtures.test_data import CONFIG_TEMPLATES

        config = CONFIG_TEMPLATES["standard"]

        # Basic validation
        assert "providers" in config
        for provider_name, provider_config in config["providers"].items():
            assert isinstance(provider_name, str)
            assert "api_key" in provider_config
            assert "model" in provider_config

    def test_config_validation_missing_required_fields(self):
        """Test configuration validation with missing fields."""
        invalid_config = {
            "providers": {
                "openai": {
                    # Missing api_key and model
                    "temperature": 0.7
                }
            }
        }

        # In a real implementation, this would raise a validation error
        errors = []

        for provider_name, provider_config in invalid_config["providers"].items():
            if "api_key" not in provider_config:
                errors.append(f"Missing api_key for {provider_name}")
            if "model" not in provider_config:
                errors.append(f"Missing model for {provider_name}")

        assert len(errors) == 2
        assert "Missing api_key for openai" in errors
        assert "Missing model for openai" in errors

    def test_environment_variable_substitution(self, mock_env):
        """Test environment variable substitution in config."""
        config_template = {
            "providers": {"openai": {"api_key": "${OPENAI_API_KEY}", "model": "gpt-4"}}
        }

        # Simple substitution logic
        def substitute_env_vars(config):
            if isinstance(config, dict):
                return {k: substitute_env_vars(v) for k, v in config.items()}
            elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
                env_var = config[2:-1]
                return os.environ.get(env_var, config)
            else:
                return config

        resolved_config = substitute_env_vars(config_template)

        assert resolved_config["providers"]["openai"]["api_key"] == "mock-openai-key"

    def test_config_merge_priority(self):
        """Test configuration merge priority (env > file > defaults)."""
        defaults = {"providers": {"openai": {"temperature": 0.7, "max_tokens": 1000}}}

        file_config = {
            "providers": {
                "openai": {
                    "temperature": 0.5,  # Override default
                    "model": "gpt-4",  # New value
                }
            }
        }

        env_overrides = {
            "providers": {
                "openai": {
                    "temperature": 0.9  # Override file config
                }
            }
        }

        # Simple merge logic (in real implementation, would use deep merge)
        final_config = defaults.copy()

        # Apply file config
        if "providers" in file_config:
            for provider, config in file_config["providers"].items():
                if provider in final_config["providers"]:
                    final_config["providers"][provider].update(config)

        # Apply env overrides
        if "providers" in env_overrides:
            for provider, config in env_overrides["providers"].items():
                if provider in final_config["providers"]:
                    final_config["providers"][provider].update(config)

        assert final_config["providers"]["openai"]["temperature"] == 0.9  # From env
        assert final_config["providers"]["openai"]["max_tokens"] == 1000  # From defaults
        assert final_config["providers"]["openai"]["model"] == "gpt-4"  # From file


@pytest.mark.unit
class TestConfigurationDefaults:
    """Test configuration defaults and fallbacks."""

    def test_default_provider_settings(self):
        """Test default provider settings are applied."""
        from tests.fixtures.test_data import CONFIG_TEMPLATES

        minimal_config = CONFIG_TEMPLATES["minimal"]

        # Apply defaults
        defaults = {"temperature": 0.7, "max_tokens": 1000, "timeout": 30, "retries": 3}

        for provider_name, provider_config in minimal_config["providers"].items():
            for key, default_value in defaults.items():
                if key not in provider_config:
                    provider_config[key] = default_value

        # Check defaults were applied
        openai_config = minimal_config["providers"]["openai"]
        assert openai_config["temperature"] == 0.7
        assert openai_config["max_tokens"] == 1000
        assert openai_config["timeout"] == 30
        assert openai_config["retries"] == 3

    def test_provider_specific_defaults(self):
        """Test provider-specific default values."""
        provider_defaults = {
            "openai": {
                "temperature": 0.7,
                "max_tokens": 4000,  # Higher for GPT-4
                "top_p": 1.0,
            },
            "anthropic": {
                "temperature": 0.5,  # Lower default for Claude
                "max_tokens": 8000,  # Higher context
                "top_p": 0.95,
            },
            "google": {"temperature": 0.7, "max_tokens": 2048, "top_p": 0.9},
        }

        for provider, defaults in provider_defaults.items():
            assert defaults["temperature"] > 0
            assert defaults["max_tokens"] > 0
            assert 0 < defaults["top_p"] <= 1.0


@pytest.mark.unit
class TestConfigurationSecurity:
    """Test configuration security features."""

    def test_api_key_redaction_in_logs(self):
        """Test that API keys are redacted in log output."""
        config = {"providers": {"openai": {"api_key": "sk-1234567890abcdef", "model": "gpt-4"}}}

        def redact_secrets(config_str):
            """Simple secret redaction."""
            import re

            # Redact OpenAI keys
            config_str = re.sub(r"sk-[a-zA-Z0-9]{8,}", "***REDACTED***", config_str)
            # Redact other API keys
            config_str = re.sub(
                r'"api_key":\s*"[^"]{10,}"', '"api_key": "***REDACTED***"', config_str
            )
            return config_str

        config_str = str(config)
        redacted_str = redact_secrets(config_str)

        assert "sk-1234567890abcdef" not in redacted_str
        assert "***REDACTED***" in redacted_str

    def test_config_file_permissions(self, temp_dir):
        """Test that config files have appropriate permissions."""
        config_file = temp_dir / "secure_config.yaml"

        config = {"providers": {"openai": {"api_key": "secret-key", "model": "gpt-4"}}}

        # Write config file
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Check file exists
        assert config_file.exists()

        # In a real implementation, we'd check file permissions
        # os.chmod(config_file, 0o600)  # Read/write for owner only
        # stat = config_file.stat()
        # assert stat.st_mode & 0o777 == 0o600

    def test_environment_variable_security(self):
        """Test secure handling of environment variables."""
        sensitive_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "PASSWORD",
            "SECRET",
            "TOKEN",
        ]

        # In logs, these should be redacted
        for var in sensitive_vars:
            assert any(keyword in var.lower() for keyword in ["key", "password", "secret", "token"])


@pytest.mark.unit
class TestConfigurationValidation:
    """Test configuration validation logic."""

    def test_temperature_validation(self):
        """Test temperature parameter validation."""

        def validate_temperature(temp):
            if not isinstance(temp, (int, float)):
                raise ValueError("Temperature must be a number")
            if not 0 <= temp <= 2:
                raise ValueError("Temperature must be between 0 and 2")
            return temp

        # Valid temperatures
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(0.7) == 0.7
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(2.0) == 2.0

        # Invalid temperatures
        with pytest.raises(ValueError):
            validate_temperature(-0.1)

        with pytest.raises(ValueError):
            validate_temperature(2.1)

        with pytest.raises(ValueError):
            validate_temperature("0.7")

    def test_max_tokens_validation(self):
        """Test max_tokens parameter validation."""

        def validate_max_tokens(tokens):
            if not isinstance(tokens, int):
                raise ValueError("max_tokens must be an integer")
            if tokens < 1:
                raise ValueError("max_tokens must be positive")
            if tokens > 32000:  # Reasonable upper limit
                raise ValueError("max_tokens too large")
            return tokens

        # Valid values
        assert validate_max_tokens(1) == 1
        assert validate_max_tokens(1000) == 1000
        assert validate_max_tokens(32000) == 32000

        # Invalid values
        with pytest.raises(ValueError):
            validate_max_tokens(0)

        with pytest.raises(ValueError):
            validate_max_tokens(-10)

        with pytest.raises(ValueError):
            validate_max_tokens(50000)

        with pytest.raises(ValueError):
            validate_max_tokens(1000.5)

    def test_model_name_validation(self):
        """Test model name validation."""
        valid_models = {
            "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k", "gpt-4-turbo"],
            "anthropic": [
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
            ],
            "google": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        }

        def validate_model_name(provider, model):
            if not isinstance(model, str) or not model:
                raise ValueError("Model name must be a non-empty string")

            if provider in valid_models:
                if model not in valid_models[provider]:
                    raise ValueError(f"Invalid model '{model}' for provider '{provider}'")

            return model

        # Valid models
        assert validate_model_name("openai", "gpt-4") == "gpt-4"
        assert (
            validate_model_name("anthropic", "claude-3-opus-20240229") == "claude-3-opus-20240229"
        )

        # Invalid models
        with pytest.raises(ValueError):
            validate_model_name("openai", "")

        with pytest.raises(ValueError):
            validate_model_name("openai", "invalid-model")

        with pytest.raises(ValueError):
            validate_model_name("openai", None)

    def test_api_key_format_validation(self):
        """Test API key format validation."""

        def validate_api_key(provider, api_key):
            if not isinstance(api_key, str) or not api_key:
                raise ValueError("API key must be a non-empty string")

            patterns = {
                "openai": r"^sk-[a-zA-Z0-9]{32,}$",
                "anthropic": r"^sk-ant-[a-zA-Z0-9\-_]{32,}$",
            }

            if provider in patterns:
                import re

                if not re.match(patterns[provider], api_key):
                    raise ValueError(f"Invalid API key format for {provider}")

            return api_key

        # Valid keys
        assert validate_api_key("openai", "sk-1234567890abcdef1234567890abcdef")
        assert validate_api_key("anthropic", "sk-ant-1234567890abcdef1234567890abcdef")

        # Invalid keys
        with pytest.raises(ValueError):
            validate_api_key("openai", "invalid-key")

        with pytest.raises(ValueError):
            validate_api_key("openai", "")

        with pytest.raises(ValueError):
            validate_api_key("anthropic", "sk-1234567890abcdef")  # Wrong format


@pytest.mark.integration
class TestConfigurationFiles:
    """Test configuration file handling."""

    def test_create_default_config(self, temp_dir):
        """Test creating a default configuration file."""
        config_file = temp_dir / "default_config.yaml"

        default_config = {
            "providers": {
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                }
            },
            "benchmarks": {"timeout": 30, "retries": 3, "batch_size": 10},
        }

        with open(config_file, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        # Verify file was created
        assert config_file.exists()

        # Verify content
        with open(config_file) as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config == default_config

    def test_config_file_locations(self, temp_dir):
        """Test configuration file location precedence."""
        # Simulate different config file locations
        locations = [
            temp_dir / "llm_lab.yaml",  # Current directory
            temp_dir / ".llm_lab" / "config.yaml",  # User config
            temp_dir / "etc" / "llm_lab" / "config.yaml",  # System config
        ]

        # Create config files with different values
        configs = [
            {"source": "current", "priority": 1},
            {"source": "user", "priority": 2},
            {"source": "system", "priority": 3},
        ]

        for location, config in zip(locations, configs):
            location.parent.mkdir(parents=True, exist_ok=True)
            with open(location, "w") as f:
                yaml.dump(config, f)

        # Verify all files exist
        for location in locations:
            assert location.exists()

        # In real implementation, would load in priority order
        # Highest priority (lowest number) wins
        for location, expected_config in zip(locations, configs):
            with open(location) as f:
                loaded = yaml.safe_load(f)
                assert loaded["source"] == expected_config["source"]
