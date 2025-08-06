"""
Provider Configuration System

This module manages provider-specific configurations, model aliases,
and credential validation for the LLM Lab framework.

Features:
- YAML/JSON configuration file support
- Environment variable overrides
- Model name aliasing
- Configuration inheritance
- Credential validation
- Default parameter management
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: str
    base_model: str | None = None  # For aliases
    parameters: Dict[str, Any] = field(default_factory=dict)
    env_vars: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)


@dataclass
class ProviderDefaults:
    """Default configuration for a provider."""

    name: str
    env_vars: List[str]
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class ProviderConfigManager:
    """
    Manages provider configurations from multiple sources.

    Configuration priority (highest to lowest):
    1. Runtime parameters
    2. Environment variables
    3. User config file (~/.llm-lab/config.yaml)
    4. Project config file (./providers.yaml)
    5. Default config
    """

    def __init__(self, config_paths: List[Path | None] = None):
        """
        Initialize the configuration manager.

        Args:
            config_paths: Optional list of config file paths to load
        """
        self.configs: Dict[str, Any] = {}
        self.model_aliases: Dict[str, str] = {}
        self.provider_defaults: Dict[str, ProviderDefaults] = {}

        # Default config paths
        if config_paths is None:
            config_paths = [Path("providers.yaml"), Path.home() / ".llm-lab" / "config.yaml"]

        self.config_paths = config_paths
        self._load_default_config()
        self._load_config_files()
        self._build_alias_map()

    def _load_default_config(self):
        """Load default provider configurations."""
        self.provider_defaults = {
            "openai": ProviderDefaults(
                name="openai",
                env_vars=["OPENAI_API_KEY"],
                default_parameters={"temperature": 0.7, "max_tokens": 1000, "top_p": 1.0},
                models={
                    "gpt-4": ModelConfig(provider="openai", aliases=["gpt4"]),
                    "gpt-4-turbo": ModelConfig(
                        provider="openai", aliases=["gpt4-turbo", "gpt-4-turbo-preview"]
                    ),
                    "gpt-3.5-turbo": ModelConfig(
                        provider="openai", aliases=["gpt35", "gpt-35-turbo", "chatgpt"]
                    ),
                    "gpt-4o": ModelConfig(provider="openai", aliases=["gpt4o"]),
                    "gpt-4o-mini": ModelConfig(provider="openai", aliases=["gpt4o-mini"]),
                },
            ),
            "anthropic": ProviderDefaults(
                name="anthropic",
                env_vars=["ANTHROPIC_API_KEY"],
                default_parameters={"temperature": 0.7, "max_tokens": 1000, "top_p": 1.0},
                models={
                    "claude-3-opus": ModelConfig(
                        provider="anthropic", aliases=["claude3-opus", "opus"]
                    ),
                    "claude-3-sonnet": ModelConfig(
                        provider="anthropic", aliases=["claude3-sonnet", "sonnet"]
                    ),
                    "claude-3-haiku": ModelConfig(
                        provider="anthropic", aliases=["claude3-haiku", "haiku"]
                    ),
                    "claude-3.5-sonnet": ModelConfig(
                        provider="anthropic", aliases=["claude-35-sonnet"]
                    ),
                },
            ),
            "google": ProviderDefaults(
                name="google",
                env_vars=["GOOGLE_API_KEY"],
                default_parameters={"temperature": 0.7, "max_tokens": 1000, "top_p": 1.0},
                models={
                    "gemini-1.5-flash": ModelConfig(
                        provider="google", aliases=["gemini-flash", "gemini15-flash"]
                    ),
                    "gemini-1.5-pro": ModelConfig(
                        provider="google", aliases=["gemini-pro", "gemini15-pro"]
                    ),
                    "gemini-1.0-pro": ModelConfig(
                        provider="google", aliases=["gemini", "gemini10-pro"]
                    ),
                },
            ),
        }

    def _load_config_files(self):
        """Load configuration from YAML/JSON files."""
        for config_path in self.config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        if config_path.suffix in [".yaml", ".yml"]:
                            config = yaml.safe_load(f)
                        else:
                            config = json.load(f)

                    if config:
                        self._merge_config(config)
                        logger.info(f"Loaded configuration from {config_path}")

                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")

    def _merge_config(self, config: Dict[str, Any]):
        """Merge a configuration dictionary into the current config."""
        # Merge provider configurations
        if "providers" in config:
            for provider_name, provider_config in config["providers"].items():
                if provider_name not in self.provider_defaults:
                    self.provider_defaults[provider_name] = ProviderDefaults(
                        name=provider_name, env_vars=provider_config.get("env_vars", [])
                    )

                defaults = self.provider_defaults[provider_name]

                # Update provider-level settings
                if "default_parameters" in provider_config:
                    defaults.default_parameters.update(provider_config["default_parameters"])

                if "timeout" in provider_config:
                    defaults.timeout = provider_config["timeout"]

                if "max_retries" in provider_config:
                    defaults.max_retries = provider_config["max_retries"]

                # Update model configurations
                if "models" in provider_config:
                    for model_name, model_config in provider_config["models"].items():
                        if model_name not in defaults.models:
                            defaults.models[model_name] = ModelConfig(provider=provider_name)

                        model = defaults.models[model_name]

                        if "aliases" in model_config:
                            model.aliases.extend(model_config["aliases"])

                        if "parameters" in model_config:
                            model.parameters.update(model_config["parameters"])

                        if "base_model" in model_config:
                            model.base_model = model_config["base_model"]

        # Merge global aliases
        if "aliases" in config:
            self.model_aliases.update(config["aliases"])

    def _build_alias_map(self):
        """Build a map of model aliases to canonical model names."""
        for provider_name, provider in self.provider_defaults.items():
            for model_name, model_config in provider.models.items():
                # Add the model name itself as an alias
                self.model_aliases[model_name] = model_name

                # Add all configured aliases
                for alias in model_config.aliases:
                    if alias in self.model_aliases:
                        logger.debug(
                            f"Alias '{alias}' already mapped to '{self.model_aliases[alias]}', "
                            f"skipping mapping to '{model_name}'"
                        )
                    else:
                        self.model_aliases[alias.lower()] = model_name

    def resolve_model_alias(self, model_name: str) -> str:
        """
        Resolve a model alias to its canonical name.

        Args:
            model_name: The model name or alias

        Returns:
            The canonical model name
        """
        # Check exact match first
        if model_name in self.model_aliases:
            return self.model_aliases[model_name]

        # Check lowercase match
        lower_name = model_name.lower()
        if lower_name in self.model_aliases:
            return self.model_aliases[lower_name]

        # Return original if no alias found
        return model_name

    def get_provider_for_model(self, model_name: str) -> str | None:
        """
        Get the provider name for a given model.

        Args:
            model_name: The model name (can be an alias)

        Returns:
            The provider name or None if not found
        """
        # Resolve alias first
        canonical_name = self.resolve_model_alias(model_name)

        # Search through providers
        for provider_name, provider in self.provider_defaults.items():
            if canonical_name in provider.models:
                return provider_name

        return None

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific model.

        Args:
            model_name: The model name (can be an alias)

        Returns:
            Configuration dictionary
        """
        canonical_name = self.resolve_model_alias(model_name)
        provider_name = self.get_provider_for_model(canonical_name)

        if not provider_name:
            return {}

        provider = self.provider_defaults[provider_name]
        model_config = provider.models.get(canonical_name, ModelConfig(provider=provider_name))

        # Build configuration with inheritance
        config = {"provider": provider_name, "model": canonical_name, "parameters": {}}

        # Start with provider defaults
        config["parameters"].update(provider.default_parameters)

        # Apply model-specific parameters
        config["parameters"].update(model_config.parameters)

        # Apply environment variable overrides
        env_overrides = self._get_env_overrides(model_name)
        config["parameters"].update(env_overrides)

        return config

    def _get_env_overrides(self, model_name: str) -> Dict[str, Any]:
        """Get parameter overrides from environment variables."""
        overrides = {}

        # Check for model-specific env vars (e.g., GPT4_TEMPERATURE)
        model_prefix = model_name.replace("-", "_").replace(".", "_").upper()

        param_mappings = {
            f"{model_prefix}_TEMPERATURE": "temperature",
            f"{model_prefix}_MAX_TOKENS": "max_tokens",
            f"{model_prefix}_TOP_P": "top_p",
            f"{model_prefix}_TIMEOUT": "timeout",
        }

        for env_var, param_name in param_mappings.items():
            value = os.getenv(env_var)
            if value:
                try:
                    if param_name in ["temperature", "top_p"]:
                        overrides[param_name] = float(value)
                    else:
                        overrides[param_name] = int(value)
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {value}")

        return overrides

    def validate_credentials(self, provider_name: str | None = None) -> Dict[str, bool]:
        """
        Validate that required credentials are present.

        Args:
            provider_name: Optional specific provider to validate

        Returns:
            Dictionary mapping provider names to validation status
        """
        results = {}

        providers_to_check = (
            [provider_name] if provider_name else list(self.provider_defaults.keys())
        )

        for pname in providers_to_check:
            if pname not in self.provider_defaults:
                results[pname] = False
                continue

            provider = self.provider_defaults[pname]

            # Check all required environment variables
            all_present = all(os.getenv(env_var) for env_var in provider.env_vars)
            results[pname] = all_present

            if not all_present:
                missing = [var for var in provider.env_vars if not os.getenv(var)]
                logger.warning(f"Missing credentials for {pname}: {', '.join(missing)}")

        return results

    def get_available_models(self, include_aliases: bool = False) -> Dict[str | List[str]]:
        """
        Get all available models grouped by provider.

        Args:
            include_aliases: Whether to include model aliases

        Returns:
            Dictionary mapping provider names to lists of model names
        """
        result = {}

        for provider_name, provider in self.provider_defaults.items():
            models = list(provider.models.keys())

            if include_aliases:
                # Add aliases for each model
                for model_name, model_config in provider.models.items():
                    models.extend(model_config.aliases)

            result[provider_name] = sorted(models)

        return result

    def save_config(self, path: Path | None = None):
        """
        Save the current configuration to a YAML file.

        Args:
            path: Optional path to save to (defaults to user config)
        """
        if path is None:
            path = Path.home() / ".llm-lab" / "config.yaml"

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build configuration dictionary
        config = {"providers": {}}

        for provider_name, provider in self.provider_defaults.items():
            provider_config = {
                "env_vars": provider.env_vars,
                "default_parameters": provider.default_parameters,
                "timeout": provider.timeout,
                "max_retries": provider.max_retries,
                "models": {},
            }

            for model_name, model_config in provider.models.items():
                model_dict = {}
                if model_config.aliases:
                    model_dict["aliases"] = model_config.aliases
                if model_config.parameters:
                    model_dict["parameters"] = model_config.parameters
                if model_config.base_model:
                    model_dict["base_model"] = model_config.base_model

                if model_dict:
                    provider_config["models"][model_name] = model_dict

            config["providers"][provider_name] = provider_config

        # Save to file
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {path}")


# Global configuration manager instance
_config_manager: ProviderConfigManager | None = None


def get_config_manager() -> ProviderConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ProviderConfigManager()
    return _config_manager


def reset_config_manager():
    """Reset the global configuration manager (mainly for testing)."""
    global _config_manager
    _config_manager = None
