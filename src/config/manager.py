"""
Configuration Manager with Multi-Source Loading

This module provides advanced configuration management with support for:
- CLI arguments (highest priority)
- Environment variables
- YAML/TOML configuration files
- Default values (lowest priority)
- Configuration profiles (dev, staging, production)
- Dynamic reloading and validation
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .settings import Settings


class ConfigurationError(Exception):
    """Configuration-related errors."""

    pass


class ConfigurationManager:
    """
    Advanced configuration manager supporting multiple sources with proper precedence.

    Source priority (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. Configuration files (YAML/TOML)
    4. Default values
    """

    def __init__(
        self,
        config_file: str | Path | None = None,
        profile: str | None = None,
        auto_reload: bool = False,
        strict_mode: bool = True,
    ):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to configuration file
            profile: Configuration profile (dev, staging, prod)
            auto_reload: Enable automatic configuration reloading
            strict_mode: Raise errors on configuration issues
        """
        self.config_file = Path(config_file) if config_file else None
        self.profile = profile or os.getenv("LLMLAB_PROFILE", "development")
        self.auto_reload = auto_reload
        self.strict_mode = strict_mode

        self._settings: Settings | None = None
        self._file_modified_time: float | None = None

        # Configuration file search paths
        self.config_search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd() / "config.yml",
            Path.cwd() / "config.toml",
            Path.cwd() / "llmlab.yaml",
            Path.cwd() / "llmlab.yml",
            Path.cwd() / "llmlab.toml",
            Path.cwd() / ".llmlab.yaml",
            Path.cwd() / ".llmlab.yml",
            Path.cwd() / ".llmlab.toml",
            Path.home() / ".config" / "llmlab" / "config.yaml",
            Path.home() / ".llmlab" / "config.yaml",
        ]

    def load_configuration(
        self, cli_args: Dict[str, Any | None] = None, env_override: Dict[str, str | None] = None
    ) -> Settings:
        """
        Load configuration from all sources with proper precedence.

        Args:
            cli_args: CLI arguments dictionary
            env_override: Environment variable overrides

        Returns:
            Settings instance with merged configuration
        """
        config_data = {}

        # 1. Load defaults (already handled by Pydantic defaults)

        # 2. Load from configuration files
        file_config = self._load_from_file()
        if file_config:
            config_data.update(file_config)

        # 3. Load from environment variables (handled by Pydantic)
        env_config = self._load_from_environment(env_override)
        if env_config:
            self._deep_update(config_data, env_config)

        # 4. Apply CLI arguments (highest priority)
        if cli_args:
            self._deep_update(config_data, cli_args)

        # 5. Apply profile-specific overrides
        profile_config = self._load_profile_config(self.profile)
        if profile_config:
            self._deep_update(config_data, profile_config)

        try:
            settings = Settings(**config_data)
            self._settings = settings

            if self.config_file and self.config_file.exists():
                self._file_modified_time = self.config_file.stat().st_mtime

            return settings

        except ValidationError as e:
            error_msg = f"Configuration validation failed: {e}"
            if self.strict_mode:
                raise ConfigurationError(error_msg) from e
            else:
                warnings.warn(error_msg)
                return Settings()  # Return default settings

    def _load_from_file(self) -> Dict[str | Any | None]:
        """Load configuration from file."""
        config_file = self._find_config_file()
        if not config_file:
            return None

        try:
            return self._parse_config_file(config_file)
        except Exception as e:
            error_msg = f"Failed to load config file {config_file}: {e}"
            if self.strict_mode:
                raise ConfigurationError(error_msg) from e
            else:
                warnings.warn(error_msg)
                return None

    def _find_config_file(self) -> Path | None:
        """Find configuration file from search paths."""
        # Use explicit file if provided
        if self.config_file:
            if self.config_file.exists():
                return self.config_file
            elif self.strict_mode:
                raise ConfigurationError(f"Config file not found: {self.config_file}")
            else:
                warnings.warn(f"Config file not found: {self.config_file}")
                return None

        # Search in standard locations
        for config_path in self.config_search_paths:
            if config_path.exists():
                return config_path

        return None

    def _parse_config_file(self, config_path: Path) -> Dict[str | Any]:
        """Parse configuration file based on extension."""
        suffix = config_path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            return self._parse_yaml(config_path)
        elif suffix == ".toml":
            return self._parse_toml(config_path)
        elif suffix == ".json":
            return self._parse_json(config_path)
        else:
            raise ConfigurationError(f"Unsupported config file format: {suffix}")

    def _parse_yaml(self, config_path: Path) -> Dict[str | Any]:
        """Parse YAML configuration file."""
        try:
            import yaml
        except ImportError:
            raise ConfigurationError(
                "PyYAML is required for YAML config files. Install with: pip install PyYAML"
            )

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        return data if isinstance(data, dict) else {}

    def _parse_toml(self, config_path: Path) -> Dict[str | Any]:
        """Parse TOML configuration file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise ConfigurationError(
                    "tomli is required for TOML config files. Install with: pip install tomli"
                )

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        return data if isinstance(data, dict) else {}

    def _parse_json(self, config_path: Path) -> Dict[str | Any]:
        """Parse JSON configuration file."""
        import json

        with open(config_path, "r") as f:
            data = json.load(f)

        return data if isinstance(data, dict) else {}

    def _load_from_environment(self, env_override: Dict[str, str | None] = None) -> Dict[str | Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Use provided environment or system environment
        env_vars = env_override or os.environ

        # Map environment variables to configuration structure
        env_mapping = {
            # Network configuration
            "NETWORK_DEFAULT_TIMEOUT": ("network", "default_timeout"),
            "NETWORK_GENERATION_TIMEOUT": ("network", "generation_timeout"),
            "NETWORK_MODEL_PULL_TIMEOUT": ("network", "model_pull_timeout"),
            "NETWORK_OLLAMA_BASE_URL": ("network", "ollama_base_url"),
            # System configuration
            "SYSTEM_DEFAULT_BATCH_SIZE": ("system", "default_batch_size"),
            "SYSTEM_MAX_WORKERS": ("system", "max_workers"),
            "SYSTEM_MEMORY_THRESHOLD": ("system", "memory_threshold"),
            # Server configuration
            "SERVER_API_PORT": ("server", "api_port"),
            "SERVER_API_HOST": ("server", "api_host"),
            "SERVER_WEBSOCKET_PORT": ("server", "websocket_port"),
            "SERVER_DASHBOARD_PORT": ("server", "dashboard_port"),
            # Model parameters
            "MODEL_TEMPERATURE": ("providers", "default", "model_parameters", "temperature"),
            "MODEL_MAX_TOKENS": ("providers", "default", "model_parameters", "max_tokens"),
            # Global settings
            "LLMLAB_DEBUG": ("debug",),
            "LLMLAB_ENVIRONMENT": ("environment",),
            "LLMLAB_LOG_LEVEL": ("monitoring", "log_level"),
        }

        for env_var, config_path in env_mapping.items():
            if env_var in env_vars:
                value = env_vars[env_var]
                self._set_nested_value(env_config, config_path, self._convert_env_value(value))

        return env_config

    def _convert_env_value(self, value: str) -> str | int | float | bool:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False

        # Numeric conversion
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _set_nested_value(self, config_dict: Dict[str, Any], path: tuple, value: Any):
        """Set nested value in configuration dictionary."""
        for key in path[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]

        config_dict[path[-1]] = value

    def _load_profile_config(self, profile: str) -> Dict[str | Any | None]:
        """Load profile-specific configuration."""
        profile_configs = {
            "development": {
                "debug": True,
                "monitoring": {"log_level": "DEBUG", "performance_logging": True},
                "network": {
                    "default_timeout": 60,  # Longer timeouts for development
                },
            },
            "testing": {
                "debug": True,
                "monitoring": {
                    "log_level": "WARNING",  # Reduce noise in tests
                    "enable_telemetry": False,
                },
                "network": {
                    "default_timeout": 10,  # Shorter timeouts for tests
                },
            },
            "staging": {
                "debug": False,
                "monitoring": {"log_level": "INFO", "enable_telemetry": True},
            },
            "production": {
                "debug": False,
                "monitoring": {
                    "log_level": "WARNING",
                    "enable_telemetry": True,
                    "performance_tracking": True,
                },
            },
        }

        return profile_configs.get(profile)

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep update dictionary with another dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def get_settings(self) -> Settings:
        """Get current settings, loading if necessary."""
        if self._settings is None or self._should_reload():
            return self.load_configuration()
        return self._settings

    def _should_reload(self) -> bool:
        """Check if configuration should be reloaded."""
        if not self.auto_reload:
            return False

        if not self.config_file or not self.config_file.exists():
            return False

        current_mtime = self.config_file.stat().st_mtime
        return current_mtime != self._file_modified_time

    def reload_configuration(self) -> Settings:
        """Force reload configuration from all sources."""
        self._settings = None
        return self.load_configuration()

    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return list of issues."""
        settings = self.get_settings()
        return settings.validate_all()

    def save_configuration(
        self,
        config_path: str | Path | None = None,
        format: str = "yaml",
        include_defaults: bool = False,
    ):
        """
        Save current configuration to file.

        Args:
            config_path: Path to save configuration
            format: File format (yaml, json, toml)
            include_defaults: Include default values in output
        """
        settings = self.get_settings()
        save_path = Path(config_path) if config_path else Path(f"llmlab_config.{format}")

        if format.lower() == "yaml":
            settings.to_yaml(save_path)
        elif format.lower() == "json":
            settings.to_json(save_path)
        elif format.lower() == "toml":
            self._save_toml(settings, save_path, include_defaults)
        else:
            raise ConfigurationError(f"Unsupported save format: {format}")

    def _save_toml(self, settings: Settings, config_path: Path, include_defaults: bool):
        """Save configuration as TOML."""
        try:
            import tomli_w
        except ImportError:
            raise ConfigurationError(
                "tomli_w is required to save TOML files. Install with: pip install tomli_w"
            )

        config_data = settings.model_dump(exclude_none=not include_defaults)

        with open(config_path, "wb") as f:
            tomli_w.dump(config_data, f)

    def create_cli_parser(self) -> argparse.ArgumentParser:
        """Create CLI argument parser with configuration options."""
        parser = argparse.ArgumentParser(
            description="LLM Lab Configuration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Configuration file options
        parser.add_argument("--config", "-c", type=str, help="Configuration file path")

        parser.add_argument(
            "--profile",
            "-p",
            choices=["development", "testing", "staging", "production"],
            default=self.profile,
            help="Configuration profile",
        )

        # Global options
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")

        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level",
        )

        # Network options
        parser.add_argument("--timeout", type=int, help="Default network timeout in seconds")

        parser.add_argument("--ollama-url", type=str, help="Ollama base URL")

        # Server options
        parser.add_argument("--api-port", type=int, help="API server port")

        parser.add_argument("--dashboard-port", type=int, help="Dashboard port")

        # Model options
        parser.add_argument("--temperature", type=float, help="Default model temperature")

        parser.add_argument("--max-tokens", type=int, help="Default max tokens")

        # System options
        parser.add_argument("--batch-size", type=int, help="Default batch size")

        return parser

    def parse_cli_args(self, args: List[str | None] = None) -> Dict[str | Any]:
        """Parse CLI arguments into configuration dictionary."""
        parser = self.create_cli_parser()
        parsed_args = parser.parse_args(args)

        # Convert namespace to dict and filter None values
        cli_config = {}
        args_dict = vars(parsed_args)

        # Map CLI args to configuration structure
        cli_mapping = {
            "debug": ("debug",),
            "log_level": ("monitoring", "log_level"),
            "timeout": ("network", "default_timeout"),
            "ollama_url": ("network", "ollama_base_url"),
            "api_port": ("server", "api_port"),
            "dashboard_port": ("server", "dashboard_port"),
            "temperature": ("providers", "default", "model_parameters", "temperature"),
            "max_tokens": ("providers", "default", "model_parameters", "max_tokens"),
            "batch_size": ("system", "default_batch_size"),
        }

        for cli_arg, config_path in cli_mapping.items():
            if cli_arg in args_dict and args_dict[cli_arg] is not None:
                self._set_nested_value(cli_config, config_path, args_dict[cli_arg])

        # Handle special arguments
        if args_dict.get("config"):
            self.config_file = Path(args_dict["config"])

        if args_dict.get("profile"):
            self.profile = args_dict["profile"]

        return cli_config

    def get_config_summary(self) -> Dict[str | str]:
        """Get a summary of configuration sources and key values."""
        settings = self.get_settings()

        config_file_status = "Not found"
        if self.config_file and self.config_file.exists():
            config_file_status = str(self.config_file)
        elif self._find_config_file():
            config_file_status = str(self._find_config_file())

        return {
            "Configuration File": config_file_status,
            "Profile": self.profile,
            "Environment": settings.environment,
            "Debug Mode": str(settings.debug),
            "API Port": str(settings.server.api_port),
            "Log Level": settings.monitoring.log_level.value,
            "Auto Reload": str(self.auto_reload),
            "Strict Mode": str(self.strict_mode),
        }


# Global configuration manager instance
_config_manager: ConfigurationManager | None = None


def get_config_manager(**kwargs) -> ConfigurationManager:
    """
    Get the global configuration manager instance.

    Args:
        **kwargs: Arguments passed to ConfigurationManager constructor

    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(**kwargs)
    return _config_manager


def load_configuration(**kwargs) -> Settings:
    """
    Load configuration using the global manager.

    Args:
        **kwargs: Arguments passed to load_configuration

    Returns:
        Settings instance
    """
    manager = get_config_manager()
    return manager.load_configuration(**kwargs)


def reset_config_manager():
    """Reset the global configuration manager (mainly for testing)."""
    global _config_manager
    _config_manager = None
