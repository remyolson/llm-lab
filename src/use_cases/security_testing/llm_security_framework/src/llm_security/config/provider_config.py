"""Configuration management for LLM providers."""

import base64
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types."""

    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"
    TEST = "test"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    cooldown_seconds: int = 60


@dataclass
class RetryConfig:
    """Retry policy configuration."""

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""

    # Required fields
    provider_type: str
    api_key: Optional[str] = None

    # Optional fields
    api_base_url: Optional[str] = None
    api_version: Optional[str] = None
    organization_id: Optional[str] = None
    deployment_name: Optional[str] = None  # For Azure
    model_name: Optional[str] = None

    # Connection settings
    timeout_seconds: int = 30
    max_connections: int = 10
    verify_ssl: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)

    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Retry policy
    retry_policy: RetryConfig = field(default_factory=RetryConfig)

    # Feature flags
    supports_streaming: bool = False
    supports_functions: bool = False
    supports_async: bool = False

    # Metadata
    environment: Environment = Environment.DEVELOPMENT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["environment"] = self.environment.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        """Create from dictionary."""
        # Handle environment enum
        if "environment" in data and isinstance(data["environment"], str):
            data["environment"] = Environment(data["environment"])

        # Handle nested configs
        if "rate_limit" in data and isinstance(data["rate_limit"], dict):
            data["rate_limit"] = RateLimitConfig(**data["rate_limit"])

        if "retry_policy" in data and isinstance(data["retry_policy"], dict):
            data["retry_policy"] = RetryConfig(**data["retry_policy"])

        return cls(**data)

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors
        """
        errors = []

        if not self.provider_type:
            errors.append("provider_type is required")

        if not self.api_key and self.environment != Environment.TEST:
            errors.append("api_key is required for non-test environments")

        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")

        if self.max_connections <= 0:
            errors.append("max_connections must be positive")

        return errors


class SecureStorage:
    """Secure storage for sensitive configuration data."""

    def __init__(self, password: Optional[str] = None):
        """
        Initialize secure storage.

        Args:
            password: Password for encryption (uses env var if not provided)
        """
        self.password = password or os.getenv("CONFIG_ENCRYPTION_KEY", "default-dev-key")
        self._cipher_suite = self._create_cipher()

    def _create_cipher(self) -> Fernet:
        """Create cipher for encryption/decryption."""
        # Derive key from password
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"llm-security-salt",  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
        return Fernet(key)

    def encrypt(self, data: str) -> str:
        """
        Encrypt sensitive data.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted string
        """
        return self._cipher_suite.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted string
        """
        return self._cipher_suite.decrypt(encrypted_data.encode()).decode()

    def encrypt_config(self, config: ProviderConfig) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in configuration.

        Args:
            config: Provider configuration

        Returns:
            Config dict with encrypted fields
        """
        data = config.to_dict()

        # Encrypt sensitive fields
        sensitive_fields = ["api_key", "organization_id"]
        for field in sensitive_fields:
            if field in data and data[field]:
                data[field] = self.encrypt(data[field])
                data[f"{field}_encrypted"] = True

        return data

    def decrypt_config(self, data: Dict[str, Any]) -> ProviderConfig:
        """
        Decrypt configuration.

        Args:
            data: Encrypted config data

        Returns:
            Decrypted ProviderConfig
        """
        # Decrypt sensitive fields
        sensitive_fields = ["api_key", "organization_id"]
        for field in sensitive_fields:
            if f"{field}_encrypted" in data and data[f"{field}_encrypted"]:
                if field in data and data[field]:
                    data[field] = self.decrypt(data[field])
                del data[f"{field}_encrypted"]

        return ProviderConfig.from_dict(data)


class ConfigLoader:
    """Load and manage provider configurations."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        environment: Optional[Environment] = None,
        secure_storage: Optional[SecureStorage] = None,
    ):
        """
        Initialize config loader.

        Args:
            config_path: Path to config file
            environment: Current environment
            secure_storage: Secure storage instance
        """
        self.config_path = config_path or Path("config/providers.yaml")
        self.environment = environment or self._detect_environment()
        self.secure_storage = secure_storage or SecureStorage()
        self._configs: Dict[str, ProviderConfig] = {}
        self._load_configs()

    def _detect_environment(self) -> Environment:
        """Detect current environment from env vars."""
        env_str = os.getenv("ENVIRONMENT", "dev").lower()

        env_map = {
            "dev": Environment.DEVELOPMENT,
            "development": Environment.DEVELOPMENT,
            "staging": Environment.STAGING,
            "prod": Environment.PRODUCTION,
            "production": Environment.PRODUCTION,
            "test": Environment.TEST,
        }

        return env_map.get(env_str, Environment.DEVELOPMENT)

    def _load_configs(self) -> None:
        """Load configurations from various sources."""
        # Load from environment variables
        self._load_from_env()

        # Load from config file
        if self.config_path.exists():
            self._load_from_file()

        # Load from .env file
        self._load_from_dotenv()

        logger.info(f"Loaded {len(self._configs)} provider configurations")

    def _load_from_env(self) -> None:
        """Load configurations from environment variables."""
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            self._configs["openai"] = ProviderConfig(
                provider_type="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base_url=os.getenv("OPENAI_API_BASE"),
                organization_id=os.getenv("OPENAI_ORGANIZATION"),
                environment=self.environment,
            )

        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            self._configs["anthropic"] = ProviderConfig(
                provider_type="anthropic",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                api_base_url=os.getenv("ANTHROPIC_API_BASE"),
                environment=self.environment,
            )

        # Azure OpenAI
        if os.getenv("AZURE_OPENAI_API_KEY"):
            self._configs["azure_openai"] = ProviderConfig(
                provider_type="azure_openai",
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                environment=self.environment,
            )

    def _load_from_file(self) -> None:
        """Load configurations from YAML/JSON file."""
        try:
            with open(self.config_path, "r") as f:
                if self.config_path.suffix in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                elif self.config_path.suffix == ".json":
                    data = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {self.config_path}")
                    return

            # Load providers section
            if "providers" in data:
                for provider_name, provider_data in data["providers"].items():
                    # Check for environment-specific config
                    if self.environment.value in provider_data:
                        provider_data = provider_data[self.environment.value]

                    # Decrypt if needed
                    if provider_data.get("encrypted", False):
                        config = self.secure_storage.decrypt_config(provider_data)
                    else:
                        config = ProviderConfig.from_dict(provider_data)

                    self._configs[provider_name] = config

            # Load defaults
            if "defaults" in data:
                self._apply_defaults(data["defaults"])

        except Exception as e:
            logger.error(f"Error loading config file: {e}")

    def _load_from_dotenv(self) -> None:
        """Load configurations from .env file."""
        dotenv_path = Path(".env")
        if not dotenv_path.exists():
            return

        try:
            from dotenv import load_dotenv

            load_dotenv()
            # Re-load from environment after dotenv
            self._load_from_env()
        except ImportError:
            logger.debug("python-dotenv not installed, skipping .env file")

    def _apply_defaults(self, defaults: Dict[str, Any]) -> None:
        """Apply default values to all configurations."""
        for config in self._configs.values():
            for key, value in defaults.items():
                if hasattr(config, key) and getattr(config, key) is None:
                    setattr(config, key, value)

    def get_config(self, provider: str) -> Optional[ProviderConfig]:
        """
        Get configuration for a specific provider.

        Args:
            provider: Provider name

        Returns:
            ProviderConfig or None
        """
        return self._configs.get(provider)

    def list_providers(self) -> List[str]:
        """Get list of configured providers."""
        return list(self._configs.keys())

    def add_config(self, name: str, config: ProviderConfig) -> None:
        """
        Add or update a provider configuration.

        Args:
            name: Provider name
            config: Provider configuration
        """
        # Validate config
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")

        self._configs[name] = config
        logger.info(f"Added configuration for provider: {name}")

    def save_configs(self, path: Optional[Path] = None, encrypt: bool = True) -> None:
        """
        Save configurations to file.

        Args:
            path: Output path (uses default if not provided)
            encrypt: Whether to encrypt sensitive data
        """
        path = path or self.config_path
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"providers": {}}

        for name, config in self._configs.items():
            if encrypt:
                config_data = self.secure_storage.encrypt_config(config)
                config_data["encrypted"] = True
            else:
                config_data = config.to_dict()

            data["providers"][name] = config_data

        # Add metadata
        data["metadata"] = {
            "environment": self.environment.value,
            "encrypted": encrypt,
            "version": "1.0",
        }

        with open(path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                yaml.safe_dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)

        logger.info(f"Saved configurations to {path}")

    def validate_all(self) -> Dict[str, List[str]]:
        """
        Validate all configurations.

        Returns:
            Dictionary of provider names to error lists
        """
        errors = {}

        for name, config in self._configs.items():
            config_errors = config.validate()
            if config_errors:
                errors[name] = config_errors

        return errors
