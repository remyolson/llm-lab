"""Tests for the provider configuration system."""

import os
import json
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from src.config.provider_config import (
    ProviderConfigManager,
    ModelConfig,
    ProviderDefaults,
    get_config_manager,
    reset_config_manager
)
from src.providers.exceptions import ProviderConfigurationError


class TestProviderConfigManager:
    """Test the provider configuration manager."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration data."""
        return {
            "providers": {
                "openai": {
                    "env_vars": ["OPENAI_API_KEY"],
                    "default_parameters": {
                        "temperature": 0.5,
                        "max_tokens": 2000
                    },
                    "timeout": 60,
                    "models": {
                        "gpt-4": {
                            "aliases": ["gpt4", "gpt-4-latest"],
                            "parameters": {
                                "temperature": 0.8
                            }
                        }
                    }
                },
                "anthropic": {
                    "env_vars": ["ANTHROPIC_API_KEY"],
                    "default_parameters": {
                        "temperature": 0.7
                    },
                    "models": {
                        "claude-3-opus": {
                            "aliases": ["claude3", "opus"]
                        }
                    }
                }
            },
            "aliases": {
                "best": "gpt-4",
                "fast": "gpt-3.5-turbo"
            }
        }
    
    @pytest.fixture
    def config_manager_with_file(self, temp_config_dir, sample_config):
        """Create a config manager with a test config file."""
        config_path = Path(temp_config_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Reset global manager
        reset_config_manager()
        
        return ProviderConfigManager([config_path])
    
    def test_default_configuration_loading(self):
        """Test that default configurations are loaded correctly."""
        manager = ProviderConfigManager(config_paths=[])
        
        # Check that default providers are loaded
        assert "openai" in manager.provider_defaults
        assert "anthropic" in manager.provider_defaults
        assert "google" in manager.provider_defaults
        
        # Check OpenAI defaults
        openai_defaults = manager.provider_defaults["openai"]
        assert openai_defaults.name == "openai"
        assert "OPENAI_API_KEY" in openai_defaults.env_vars
        assert openai_defaults.default_parameters["temperature"] == 0.7
        assert "gpt-4" in openai_defaults.models
        assert "gpt-3.5-turbo" in openai_defaults.models
    
    def test_config_file_loading(self, config_manager_with_file):
        """Test loading configuration from YAML file."""
        manager = config_manager_with_file
        
        # Check that config was merged
        openai_defaults = manager.provider_defaults["openai"]
        assert openai_defaults.default_parameters["temperature"] == 0.5  # Overridden
        assert openai_defaults.default_parameters["max_tokens"] == 2000  # Overridden
        assert openai_defaults.timeout == 60  # Overridden
        
        # Check model-specific config
        gpt4_config = openai_defaults.models["gpt-4"]
        assert "gpt4" in gpt4_config.aliases
        assert "gpt-4-latest" in gpt4_config.aliases
        assert gpt4_config.parameters["temperature"] == 0.8
    
    def test_alias_resolution(self, config_manager_with_file):
        """Test model alias resolution."""
        manager = config_manager_with_file
        
        # Test direct model name
        assert manager.resolve_model_alias("gpt-4") == "gpt-4"
        
        # Test configured aliases
        assert manager.resolve_model_alias("gpt4") == "gpt-4"
        assert manager.resolve_model_alias("gpt-4-latest") == "gpt-4"
        assert manager.resolve_model_alias("claude3") == "claude-3-opus"
        assert manager.resolve_model_alias("opus") == "claude-3-opus"
        
        # Test global aliases
        assert manager.resolve_model_alias("best") == "gpt-4"
        assert manager.resolve_model_alias("fast") == "gpt-3.5-turbo"
        
        # Test case insensitivity
        assert manager.resolve_model_alias("GPT4") == "gpt-4"
        assert manager.resolve_model_alias("CLAUDE3") == "claude-3-opus"
        
        # Test unknown alias (returns original)
        assert manager.resolve_model_alias("unknown-model") == "unknown-model"
    
    def test_get_provider_for_model(self, config_manager_with_file):
        """Test getting provider name for a model."""
        manager = config_manager_with_file
        
        # Direct model names
        assert manager.get_provider_for_model("gpt-4") == "openai"
        assert manager.get_provider_for_model("claude-3-opus") == "anthropic"
        
        # Via aliases
        assert manager.get_provider_for_model("gpt4") == "openai"
        assert manager.get_provider_for_model("claude3") == "anthropic"
        
        # Unknown model
        assert manager.get_provider_for_model("unknown-model") is None
    
    def test_get_model_config(self, config_manager_with_file):
        """Test getting model configuration."""
        manager = config_manager_with_file
        
        # Get GPT-4 config
        config = manager.get_model_config("gpt-4")
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4"
        assert config["parameters"]["temperature"] == 0.8  # Model-specific override
        assert config["parameters"]["max_tokens"] == 2000  # Provider default
        
        # Get config via alias
        config_via_alias = manager.get_model_config("gpt4")
        assert config_via_alias["model"] == "gpt-4"
        assert config_via_alias["parameters"]["temperature"] == 0.8
        
        # Get Claude config (no model-specific params)
        claude_config = manager.get_model_config("claude-3-opus")
        assert claude_config["provider"] == "anthropic"
        assert claude_config["parameters"]["temperature"] == 0.7  # Provider default
    
    def test_environment_variable_overrides(self, config_manager_with_file, monkeypatch):
        """Test that environment variables override config values."""
        manager = config_manager_with_file
        
        # Set environment variables
        monkeypatch.setenv("GPT_4_TEMPERATURE", "0.3")
        monkeypatch.setenv("GPT_4_MAX_TOKENS", "3000")
        monkeypatch.setenv("GPT_4_TIMEOUT", "120")
        
        config = manager.get_model_config("gpt-4")
        assert config["parameters"]["temperature"] == 0.3
        assert config["parameters"]["max_tokens"] == 3000
        assert config["parameters"]["timeout"] == 120
        
        # Test with dashes in model name
        monkeypatch.setenv("CLAUDE_3_OPUS_TEMPERATURE", "0.9")
        claude_config = manager.get_model_config("claude-3-opus")
        assert claude_config["parameters"]["temperature"] == 0.9
    
    def test_validate_credentials(self, config_manager_with_file, monkeypatch):
        """Test credential validation."""
        manager = config_manager_with_file
        
        # No credentials set
        results = manager.validate_credentials()
        assert results["openai"] is False
        assert results["anthropic"] is False
        
        # Set OpenAI credential
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        results = manager.validate_credentials()
        assert results["openai"] is True
        assert results["anthropic"] is False
        
        # Validate specific provider
        result = manager.validate_credentials("openai")
        assert result["openai"] is True
        
        # Unknown provider
        result = manager.validate_credentials("unknown")
        assert result["unknown"] is False
    
    def test_get_available_models(self, config_manager_with_file):
        """Test getting available models."""
        manager = config_manager_with_file
        
        # Without aliases
        models = manager.get_available_models(include_aliases=False)
        assert "gpt-4" in models["openai"]
        assert "gpt-3.5-turbo" in models["openai"]
        assert "claude-3-opus" in models["anthropic"]
        
        # With aliases
        models_with_aliases = manager.get_available_models(include_aliases=True)
        assert "gpt4" in models_with_aliases["openai"]
        assert "gpt-4-latest" in models_with_aliases["openai"]
        assert "claude3" in models_with_aliases["anthropic"]
        assert "opus" in models_with_aliases["anthropic"]
    
    def test_save_config(self, temp_config_dir):
        """Test saving configuration."""
        manager = ProviderConfigManager(config_paths=[])
        
        # Modify some settings
        manager.provider_defaults["openai"].default_parameters["temperature"] = 0.9
        manager.provider_defaults["openai"].models["gpt-4"].aliases.append("my-gpt4")
        
        # Save config
        save_path = Path(temp_config_dir) / "saved_config.yaml"
        manager.save_config(save_path)
        
        # Load saved config
        with open(save_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config["providers"]["openai"]["default_parameters"]["temperature"] == 0.9
        assert "my-gpt4" in saved_config["providers"]["openai"]["models"]["gpt-4"]["aliases"]
    
    def test_config_inheritance(self, temp_config_dir):
        """Test configuration inheritance from multiple sources."""
        # Create project config
        project_config = {
            "providers": {
                "openai": {
                    "default_parameters": {
                        "temperature": 0.5
                    }
                }
            }
        }
        project_path = Path(temp_config_dir) / "project.yaml"
        with open(project_path, 'w') as f:
            yaml.dump(project_config, f)
        
        # Create user config
        user_config = {
            "providers": {
                "openai": {
                    "default_parameters": {
                        "temperature": 0.3,  # Override project
                        "max_tokens": 1500
                    },
                    "models": {
                        "gpt-4": {
                            "parameters": {
                                "temperature": 0.8  # Model-specific
                            }
                        }
                    }
                }
            }
        }
        user_path = Path(temp_config_dir) / "user.yaml"
        with open(user_path, 'w') as f:
            yaml.dump(user_config, f)
        
        # Load with both configs (user should override project)
        manager = ProviderConfigManager([project_path, user_path])
        
        config = manager.get_model_config("gpt-4")
        assert config["parameters"]["temperature"] == 0.8  # Model-specific wins
        assert config["parameters"]["max_tokens"] == 1500  # From user config
        
        # Test model without specific config
        gpt35_config = manager.get_model_config("gpt-3.5-turbo")
        assert gpt35_config["parameters"]["temperature"] == 0.3  # User default
    
    def test_invalid_config_handling(self, temp_config_dir):
        """Test handling of invalid configuration files."""
        # Create invalid YAML
        invalid_path = Path(temp_config_dir) / "invalid.yaml"
        with open(invalid_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        # Should not raise, just log warning
        manager = ProviderConfigManager([invalid_path])
        assert len(manager.provider_defaults) > 0  # Still has defaults
    
    def test_json_config_support(self, temp_config_dir, sample_config):
        """Test that JSON config files are supported."""
        json_path = Path(temp_config_dir) / "config.json"
        with open(json_path, 'w') as f:
            json.dump(sample_config, f)
        
        manager = ProviderConfigManager([json_path])
        
        # Check that JSON config was loaded
        assert manager.provider_defaults["openai"].default_parameters["temperature"] == 0.5
        assert manager.resolve_model_alias("gpt4") == "gpt-4"
    
    def test_alias_conflict_handling(self, temp_config_dir):
        """Test handling of conflicting aliases."""
        # Create config with conflicting aliases
        config = {
            "providers": {
                "openai": {
                    "models": {
                        "gpt-4": {
                            "aliases": ["best-model"]
                        },
                        "gpt-3.5-turbo": {
                            "aliases": ["best-model"]  # Conflict!
                        }
                    }
                }
            }
        }
        config_path = Path(temp_config_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with patch('config.provider_config.logger') as mock_logger:
            manager = ProviderConfigManager([config_path])
            
            # Should log warning about conflict
            mock_logger.warning.assert_called()
            
            # First model should win
            assert manager.resolve_model_alias("best-model") == "gpt-4"
    
    def test_empty_config_file(self, temp_config_dir):
        """Test handling of empty config files."""
        empty_path = Path(temp_config_dir) / "empty.yaml"
        empty_path.touch()  # Create empty file
        
        # Should not raise
        manager = ProviderConfigManager([empty_path])
        assert len(manager.provider_defaults) > 0  # Still has defaults
    
    def test_global_config_manager(self):
        """Test the global configuration manager singleton."""
        reset_config_manager()
        
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        # Should be the same instance
        assert manager1 is manager2
        
        # Test reset
        reset_config_manager()
        manager3 = get_config_manager()
        assert manager3 is not manager1


class TestModelConfig:
    """Test the ModelConfig dataclass."""
    
    def test_model_config_creation(self):
        """Test creating a ModelConfig instance."""
        config = ModelConfig(
            provider="openai",
            base_model="gpt-4",
            parameters={"temperature": 0.8},
            env_vars=["OPENAI_API_KEY"],
            aliases=["gpt4", "gpt-4-latest"]
        )
        
        assert config.provider == "openai"
        assert config.base_model == "gpt-4"
        assert config.parameters["temperature"] == 0.8
        assert "OPENAI_API_KEY" in config.env_vars
        assert "gpt4" in config.aliases
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig(provider="anthropic")
        
        assert config.provider == "anthropic"
        assert config.base_model is None
        assert config.parameters == {}
        assert config.env_vars == []
        assert config.aliases == []


class TestProviderDefaults:
    """Test the ProviderDefaults dataclass."""
    
    def test_provider_defaults_creation(self):
        """Test creating a ProviderDefaults instance."""
        defaults = ProviderDefaults(
            name="openai",
            env_vars=["OPENAI_API_KEY"],
            default_parameters={"temperature": 0.7},
            timeout=60,
            max_retries=5
        )
        
        assert defaults.name == "openai"
        assert "OPENAI_API_KEY" in defaults.env_vars
        assert defaults.default_parameters["temperature"] == 0.7
        assert defaults.timeout == 60
        assert defaults.max_retries == 5
    
    def test_provider_defaults_with_models(self):
        """Test ProviderDefaults with model configurations."""
        model = ModelConfig(provider="openai", aliases=["gpt4"])
        defaults = ProviderDefaults(
            name="openai",
            env_vars=["OPENAI_API_KEY"],
            models={"gpt-4": model}
        )
        
        assert "gpt-4" in defaults.models
        assert defaults.models["gpt-4"].provider == "openai"
        assert "gpt4" in defaults.models["gpt-4"].aliases


class TestIntegrationWithProviders:
    """Test integration between config manager and providers."""
    
    @pytest.fixture
    def mock_registry(self):
        """Mock the provider registry."""
        with patch('llm_providers.registry.registry') as mock:
            mock._providers = {
                'openai': MagicMock(),
                'anthropic': MagicMock()
            }
            yield mock
    
    def test_provider_uses_config_manager(self, mock_registry):
        """Test that providers use configuration from config manager."""
        from llm_providers.base import LLMProvider
        
        # Create a test provider class
        class TestProvider(LLMProvider):
            SUPPORTED_MODELS = ['test-model']
            
            def generate(self, prompt: str, **kwargs) -> str:
                return "test"
            
            def get_model_info(self) -> dict:
                return {}
            
            def validate_credentials(self) -> bool:
                return True
        
        # Create provider instance
        with patch('config.provider_config.get_config_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_model_config.return_value = {
                'parameters': {
                    'temperature': 0.5,
                    'max_tokens': 2000
                }
            }
            mock_get_manager.return_value = mock_manager
            
            provider = TestProvider('test-model')
            
            # Check that config was loaded
            assert provider.config.temperature == 0.5
            assert provider.config.max_tokens == 2000
    
    def test_registry_alias_resolution(self, mock_registry):
        """Test that registry resolves aliases through config manager."""
        reset_config_manager()
        
        with patch('config.provider_config.ProviderConfigManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.resolve_model_alias.return_value = "gpt-4"
            mock_manager.get_provider_for_model.return_value = "openai"
            mock_manager_class.return_value = mock_manager
            
            from llm_providers.registry import get_provider_for_model
            
            # Mock registry get_provider
            mock_registry.get_provider.return_value = MagicMock()
            
            # Should resolve alias
            provider = get_provider_for_model("my-custom-gpt")
            mock_manager.resolve_model_alias.assert_called_with("my-custom-gpt")