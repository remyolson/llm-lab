"""
Centralized Configuration Management using Pydantic Settings

This module provides a unified configuration system for the LLM Lab framework
using Pydantic Settings for type validation, environment variable loading,
and configuration management.

Features:
- Type-safe configuration with validation
- Automatic environment variable loading
- YAML/JSON file support
- Configuration inheritance and overrides
- Comprehensive validation with helpful error messages
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Supported log levels.

    Examples:
        >>> LogLevel.DEBUG.value
        'DEBUG'
        >>> LogLevel.INFO in LogLevel
        True
        >>> list(LogLevel)
        [<LogLevel.DEBUG: 'DEBUG'>, <LogLevel.INFO: 'INFO'>, <LogLevel.WARNING: 'WARNING'>, <LogLevel.ERROR: 'ERROR'>, <LogLevel.CRITICAL: 'CRITICAL'>]
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelParameters(BaseModel):
    """Model generation parameters (enhanced with audit findings).

    Examples:
        >>> # Create with defaults
        >>> params = ModelParameters()
        >>> params.temperature
        0.7
        >>> params.max_tokens
        1000

        >>> # Create with custom values
        >>> params = ModelParameters(
        ...     temperature=0.5,
        ...     max_tokens=500,
        ...     top_p=0.9
        ... )
        >>> params.temperature
        0.5

        >>> # Validation example
        >>> try:
        ...     bad_params = ModelParameters(temperature=3.0)
        ... except ValueError as e:
        ...     print("Validation failed")
        Validation failed
    """

    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Controls randomness in generation (0-2)"
    )
    max_tokens: int = Field(default=1000, ge=1, le=100000, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int | None = Field(  # Made optional to match audit findings
        default=None, ge=1, description="Top-k sampling parameter (optional)"
    )
    frequency_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Frequency penalty for token repetition"
    )
    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Presence penalty for token repetition"
    )
    stop_sequences: List[str | None] = Field(
        default=None, description="List of sequences where generation stops"
    )

    # Context and sequence limits (from audit)
    max_prompt_length: int = Field(
        default=50000, ge=100, le=1000000, description="Maximum allowed prompt length in characters"
    )

    default_max_length: int = Field(
        default=512, ge=64, le=32768, description="Default tokenizer max_length for most operations"
    )

    extended_max_length: int = Field(
        default=1024,
        ge=128,
        le=32768,
        description="Extended max_length for longer context operations",
    )

    # Evaluation-specific parameters (from audit findings)
    eval_temperature_conservative: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Conservative temperature for deterministic evaluation",
    )

    eval_temperature_standard: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Standard temperature for balanced evaluation"
    )

    eval_max_new_tokens_short: int = Field(
        default=50, ge=10, le=500, description="Max new tokens for short evaluation responses"
    )

    eval_max_new_tokens_medium: int = Field(
        default=100, ge=20, le=1000, description="Max new tokens for medium evaluation responses"
    )

    eval_max_new_tokens_long: int = Field(
        default=150, ge=50, le=2000, description="Max new tokens for long evaluation responses"
    )

    @field_validator("temperature")
    def validate_temperature(cls, v):
        """Validate temperature is in acceptable range."""
        if not 0 <= v <= 2:
            raise ValueError(f"Temperature {v} must be between 0 and 2")
        return v


class RetryConfig(BaseModel):
    """Configuration for retry behavior (enhanced with audit findings)."""

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum number of retry attempts")
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=30.0, description="Initial delay between retries in seconds"
    )
    exponential_backoff: bool = Field(
        default=True, description="Use exponential backoff for retries"
    )
    backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff factor for retry delays"
    )
    max_retry_delay: float = Field(
        default=60.0, ge=1.0, le=300.0, description="Maximum delay between retries in seconds"
    )
    timeout_seconds: int = Field(default=30, ge=1, le=600, description="Request timeout in seconds")

    # Jitter for distributed systems
    retry_jitter: bool = Field(
        default=True, description="Add random jitter to retry delays to avoid thundering herd"
    )

    # Timeout settings for retries
    retry_timeout: int = Field(
        default=300, ge=30, le=1800, description="Total timeout for all retry attempts in seconds"
    )


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    type: ProviderType
    api_key: str | None = Field(default=None, description="API key for the provider")
    api_base: str | None = Field(default=None, description="Base URL for API endpoint")
    organization: str | None = Field(
        default=None, description="Organization ID (for providers that support it)"
    )
    default_model: str | None = Field(
        default=None, description="Default model to use for this provider"
    )
    model_parameters: ModelParameters = Field(
        default_factory=ModelParameters, description="Default model parameters"
    )
    retry_config: RetryConfig = Field(
        default_factory=RetryConfig, description="Retry configuration"
    )
    custom_headers: Dict[str, str | None] = Field(
        default=None, description="Custom headers to include in requests"
    )

    @model_validator(mode="after")
    def validate_api_key(self):
        """Ensure API key is provided for non-local providers."""
        if self.type != ProviderType.LOCAL and not self.api_key:
            raise ValueError(f"API key required for provider type {self.type}")
        return self


class DatasetConfig(BaseModel):
    """Configuration for dataset handling."""

    base_path: Path = Field(default=Path("./datasets"), description="Base path for datasets")
    cache_dir: Path = Field(
        default=Path("./cache/datasets"), description="Cache directory for processed datasets"
    )
    max_cache_size_gb: float = Field(
        default=10.0, ge=0.1, le=1000.0, description="Maximum cache size in GB"
    )
    auto_download: bool = Field(
        default=True, description="Automatically download datasets when needed"
    )
    validation_split: float = Field(
        default=0.2, ge=0.0, le=0.5, description="Validation split ratio"
    )


class BenchmarkConfig(BaseModel):
    """Configuration for benchmarking."""

    output_dir: Path = Field(
        default=Path("./results"), description="Directory for benchmark results"
    )
    default_benchmark: str = Field(default="truthfulness", description="Default benchmark to run")
    batch_size: int = Field(
        default=8, ge=1, le=128, description="Batch size for benchmark processing"
    )
    parallel_requests: int = Field(
        default=4, ge=1, le=32, description="Number of parallel API requests"
    )
    save_raw_outputs: bool = Field(default=True, description="Save raw model outputs")
    enable_caching: bool = Field(default=True, description="Enable result caching")


class NetworkConfig(BaseModel):
    """Network and connectivity configuration settings (from audit findings)."""

    # Core timeout settings
    default_timeout: int = Field(
        default=30, ge=1, le=600, description="Default timeout for network requests in seconds"
    )

    generation_timeout: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Timeout for LLM generation requests (5 minutes default)",
    )

    model_pull_timeout: int = Field(
        default=1800,
        ge=300,
        le=3600,
        description="Timeout for model download/pull operations (30 minutes default)",
    )

    api_request_timeout: int = Field(
        default=30, ge=5, le=300, description="Timeout for general API requests in seconds"
    )

    # Connection settings
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Base URL for Ollama local model server"
    )

    # Polling intervals
    monitoring_poll_interval: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Monitoring dashboard polling interval in seconds",
    )

    buffer_poll_interval: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Buffer polling interval in seconds"
    )

    @field_validator("ollama_base_url")
    def validate_ollama_url(cls, v):
        """Validate Ollama URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Ollama base URL must start with http:// or https://")
        return v


class SystemConfig(BaseModel):
    """System performance and resource configuration settings (from audit findings)."""

    # Batch processing settings
    default_batch_size: int = Field(
        default=8, ge=1, le=128, description="Default batch size for processing operations"
    )

    small_batch_size: int = Field(
        default=4, ge=1, le=32, description="Small batch size for memory-constrained operations"
    )

    large_batch_size: int = Field(
        default=16, ge=4, le=256, description="Large batch size for throughput-optimized operations"
    )

    max_batch_size: int = Field(
        default=32, ge=8, le=512, description="Maximum batch size for high-throughput operations"
    )

    # Training-specific batch sizes
    train_batch_size: int = Field(
        default=8, ge=1, le=64, description="Training batch size per device"
    )

    eval_batch_size: int = Field(
        default=16, ge=1, le=128, description="Evaluation batch size per device"
    )

    # Buffer and chunk sizes
    download_chunk_size: int = Field(
        default=8192, ge=1024, le=1048576, description="Chunk size for file downloads in bytes"
    )

    buffer_size: int = Field(
        default=10000, ge=100, le=100000, description="Internal buffer size for data processing"
    )

    # Performance settings
    max_workers: int = Field(
        default=4, ge=1, le=32, description="Maximum number of worker threads/processes"
    )

    dataloader_workers: int = Field(
        default=4, ge=0, le=16, description="Number of workers for data loading"
    )

    # Memory management
    memory_threshold: float = Field(
        default=0.8, ge=0.1, le=0.95, description="Memory usage threshold before cleanup (0.0-1.0)"
    )

    vram_threshold: float = Field(
        default=0.9, ge=0.1, le=0.95, description="VRAM usage threshold before cleanup (0.0-1.0)"
    )


class ServerConfig(BaseModel):
    """Server and API configuration settings (from audit findings)."""

    # Core API server
    api_port: int = Field(
        default=8000, ge=1024, le=65535, description="Port for the main API server"
    )

    api_host: str = Field(default="0.0.0.0", description="Host address for the API server")

    # WebSocket server
    websocket_port: int = Field(
        default=8001, ge=1024, le=65535, description="Port for the WebSocket server"
    )

    # Dashboard and monitoring
    dashboard_port: int = Field(
        default=7860, ge=1024, le=65535, description="Port for the Gradio dashboard"
    )

    monitoring_port: int = Field(
        default=8002, ge=1024, le=65535, description="Port for monitoring services"
    )

    # CORS configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="Allowed CORS origins for API access",
    )

    cors_allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS requests"
    )

    # API configuration
    api_prefix: str = Field(default="/api/v1", description="API path prefix")

    docs_url: str = Field(default="/docs", description="OpenAPI documentation URL path")

    @field_validator("cors_origins")
    def validate_cors_origins(cls, v):
        """Validate CORS origins are valid URLs."""
        for origin in v:
            if not origin.startswith(("http://", "https://")):
                raise ValueError(
                    f"Invalid CORS origin: {origin} must start with http:// or https://"
                )
        return v


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and logging (enhanced with audit findings)."""

    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_dir: Path = Field(default=Path("./logs"), description="Directory for log files")

    # Enhanced logging configuration (from audit)
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format string",
    )

    log_max_size: int = Field(
        default=10485760,  # 10MB
        ge=1048576,
        le=1073741824,
        description="Maximum log file size in bytes before rotation",
    )

    log_backup_count: int = Field(
        default=5, ge=1, le=50, description="Number of log backup files to keep"
    )

    # Module-specific log levels
    provider_log_level: LogLevel = Field(
        default=LogLevel.INFO, description="Log level for provider modules"
    )

    evaluation_log_level: LogLevel = Field(
        default=LogLevel.INFO, description="Log level for evaluation modules"
    )

    training_log_level: LogLevel = Field(
        default=LogLevel.INFO, description="Log level for training modules"
    )

    # Performance and structured logging
    performance_logging: bool = Field(
        default=False, description="Enable performance/timing logging"
    )

    structured_logging: bool = Field(default=True, description="Enable structured JSON logging")

    enable_telemetry: bool = Field(default=False, description="Enable telemetry collection")
    metrics_port: int = Field(
        default=8080, ge=1024, le=65535, description="Port for metrics endpoint"
    )
    alert_webhook_url: str | None = Field(default=None, description="Webhook URL for alerts")
    performance_tracking: bool = Field(default=True, description="Track performance metrics")


class ValidationConfig(BaseModel):
    """Validation rules and constraint configuration (from audit findings)."""

    # Text validation
    min_prompt_length: int = Field(
        default=1, ge=0, description="Minimum allowed prompt length in characters"
    )

    max_prompt_length: int = Field(
        default=50000, ge=100, description="Maximum allowed prompt length in characters"
    )

    # Model parameter validation ranges
    temperature_min: float = Field(
        default=0.0, ge=0.0, description="Minimum allowed temperature value"
    )

    temperature_max: float = Field(
        default=2.0, le=5.0, description="Maximum allowed temperature value"
    )

    top_p_min: float = Field(default=0.0, ge=0.0, description="Minimum allowed top_p value")

    top_p_max: float = Field(default=1.0, le=1.0, description="Maximum allowed top_p value")

    # Token validation
    min_max_tokens: int = Field(default=1, ge=1, description="Minimum allowed max_tokens value")

    max_max_tokens: int = Field(
        default=100000, ge=1000, description="Maximum allowed max_tokens value"
    )

    # Validation behavior
    strict_validation: bool = Field(default=True, description="Enable strict validation mode")

    validation_warnings: bool = Field(default=True, description="Show validation warnings")


class FineTuningConfig(BaseModel):
    """Configuration for fine-tuning."""

    models_dir: Path = Field(
        default=Path("./models"), description="Directory for fine-tuned models"
    )
    checkpoints_dir: Path = Field(
        default=Path("./checkpoints"), description="Directory for training checkpoints"
    )
    default_batch_size: int = Field(
        default=4, ge=1, le=128, description="Default training batch size"
    )
    gradient_accumulation_steps: int = Field(
        default=4, ge=1, le=32, description="Gradient accumulation steps"
    )
    learning_rate: float = Field(
        default=2e-5, ge=1e-8, le=1e-2, description="Default learning rate"
    )
    num_train_epochs: int = Field(default=3, ge=1, le=100, description="Number of training epochs")
    warmup_ratio: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Warmup ratio for learning rate schedule"
    )
    use_mixed_precision: bool = Field(default=True, description="Use mixed precision training")


class Settings(BaseSettings):
    """
    Main configuration class for LLM Lab.

    Configuration is loaded from (in order of priority):
    1. Environment variables
    2. .env file
    3. Configuration files (YAML/JSON)
    4. Default values
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="allow",
    )

    # Project settings
    project_name: str = Field(default="LLM Lab", description="Project name")
    environment: str = Field(
        default="development", description="Environment (development/staging/production)"
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # Provider configurations
    providers: Dict[str, ProviderConfig] = Field(
        default_factory=dict, description="Provider configurations"
    )
    default_provider: ProviderType = Field(
        default=ProviderType.OPENAI, description="Default provider to use"
    )

    # Component configurations
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig, description="Dataset configuration"
    )
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig, description="Benchmark configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )
    fine_tuning: FineTuningConfig = Field(
        default_factory=FineTuningConfig, description="Fine-tuning configuration"
    )

    # Enhanced configuration domains (from audit findings)
    network: NetworkConfig = Field(
        default_factory=NetworkConfig, description="Network and connectivity configuration"
    )
    system: SystemConfig = Field(
        default_factory=SystemConfig, description="System performance and resource configuration"
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig, description="Server and API configuration"
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig, description="Validation rules and constraints"
    )

    # API Keys (loaded from environment)
    openai_api_key: str | None = Field(
        default=None, alias="OPENAI_API_KEY", description="OpenAI API key"
    )
    anthropic_api_key: str | None = Field(
        default=None, alias="ANTHROPIC_API_KEY", description="Anthropic API key"
    )
    google_api_key: str | None = Field(
        default=None, alias="GOOGLE_API_KEY", description="Google API key"
    )

    @model_validator(mode="after")
    def setup_providers(self):
        """Initialize provider configurations from API keys."""
        # Auto-configure providers based on available API keys
        if self.openai_api_key and "openai" not in self.providers:
            self.providers["openai"] = ProviderConfig(
                type=ProviderType.OPENAI, api_key=self.openai_api_key, default_model="gpt-4o-mini"
            )

        if self.anthropic_api_key and "anthropic" not in self.providers:
            self.providers["anthropic"] = ProviderConfig(
                type=ProviderType.ANTHROPIC,
                api_key=self.anthropic_api_key,
                default_model="claude-3-haiku-20240307",
            )

        if self.google_api_key and "google" not in self.providers:
            self.providers["google"] = ProviderConfig(
                type=ProviderType.GOOGLE,
                api_key=self.google_api_key,
                default_model="gemini-1.5-flash",
            )

        return self

    @model_validator(mode="after")
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.dataset.base_path,
            self.dataset.cache_dir,
            self.benchmark.output_dir,
            self.monitoring.log_dir,
            self.fine_tuning.models_dir,
            self.fine_tuning.checkpoints_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        return self

    def get_provider_config(self, provider_name: str | None = None) -> ProviderConfig:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Provider name (uses default if None)

        Returns:
            Provider configuration

        Raises:
            KeyError: If provider not found
        """
        if provider_name is None:
            provider_name = self.default_provider.value

        if provider_name not in self.providers:
            raise KeyError(f"Provider '{provider_name}' not configured")

        return self.providers[provider_name]

    def validate_all(self) -> List[str]:
        """
        Validate all configuration settings.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check at least one provider is configured
        if not self.providers:
            errors.append("No providers configured")

        # Validate each provider has necessary credentials
        for name, provider in self.providers.items():
            if provider.type != ProviderType.LOCAL and not provider.api_key:
                errors.append(f"Provider '{name}' missing API key")

        # Check default provider exists
        if self.default_provider.value not in self.providers:
            errors.append(f"Default provider '{self.default_provider}' not configured")

        # Validate paths are writable
        for path_name, path in [
            ("output_dir", self.benchmark.output_dir),
            ("log_dir", self.monitoring.log_dir),
            ("models_dir", self.fine_tuning.models_dir),
        ]:
            try:
                path.mkdir(parents=True, exist_ok=True)
                test_file = path / ".write_test"
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                errors.append(f"Cannot write to {path_name} ({path}): {e}")

        return errors

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """
        Load settings from a YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Settings instance
        """
        import yaml

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "Settings":
        """
        Load settings from a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Settings instance
        """
        import json

        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls(**data)

    def to_yaml(self, path: str | Path):
        """
        Save settings to a YAML file.

        Args:
            path: Path to save to
        """
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(exclude_none=True), f, default_flow_style=False, sort_keys=False
            )

    def to_json(self, path: str | Path, indent: int = 2):
        """
        Save settings to a JSON file.

        Args:
            path: Path to save to
            indent: JSON indentation
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.model_dump(exclude_none=True), f, indent=indent, default=str)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset the global settings (mainly for testing)."""
    global _settings
    _settings = None


def load_settings(path: str | Path | None = None) -> Settings:
    """
    Load settings from a file or environment.

    Args:
        path: Optional path to configuration file

    Returns:
        Settings instance
    """
    global _settings

    if path:
        path = Path(path)
        if path.suffix in [".yaml", ".yml"]:
            _settings = Settings.from_yaml(path)
        elif path.suffix == ".json":
            _settings = Settings.from_json(path)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    else:
        _settings = Settings()

    return _settings
