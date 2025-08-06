"""
Configuration TypedDict definitions for LLM Lab

This module provides TypedDict classes for configuration validation,
nested configuration structures, and settings management.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

from typing import Annotated, Dict, List, Literal

from typing_extensions import Doc, NotRequired, TypeAlias, TypedDict

# Type alias for configuration values
ConfigValue: TypeAlias = str | int | float | bool | List[str]


class ValidationConfig(TypedDict):
    """Type-safe validation configuration structure.

    Configuration for validating other configuration objects.
    """

    required_keys: List[str]
    optional_keys: NotRequired[List[str]]
    validation_rules: NotRequired[Dict[str, str]]
    strict_mode: NotRequired[bool]
    allow_extra_keys: NotRequired[bool]


class NestedConfig(TypedDict):
    """Type-safe nested configuration structure.

    For deeply nested configuration hierarchies with recursive nesting.
    """

    section: str
    values: Dict[str, ConfigValue]
    nested: NotRequired[Dict[str, "NestedConfig"]]
    metadata: NotRequired[Dict[str, str | int | float | bool]]


class ConfigurationSource(TypedDict):
    """Type-safe configuration source information.

    Tracks where configuration values came from for debugging.
    """

    source_type: Literal["file", "env", "cli", "default", "runtime"]
    source_path: NotRequired[Annotated[str, Doc("Path to source file or identifier")]]
    timestamp: NotRequired[Annotated[str, Doc("ISO timestamp when config was loaded")]]
    priority: NotRequired[Annotated[int, Doc("Priority level (higher overrides lower)")]]
    profile: NotRequired[Annotated[str, Doc("Configuration profile name")]]


class ConfigurationError(TypedDict):
    """Type-safe configuration error structure.

    Detailed error information for configuration validation failures.
    """

    error_type: Literal["validation", "missing", "type", "range", "format", "dependency"]
    message: Annotated[str, Doc("Human-readable error description")]
    field_path: NotRequired[
        Annotated[str, Doc("Dot-separated path to the field (e.g., 'providers.openai.api_key')")]
    ]
    invalid_value: NotRequired[str | int | float | bool | None]
    expected_type: NotRequired[Annotated[str, Doc("Expected type or format description")]]
    suggestion: NotRequired[Annotated[str, Doc("Suggestion for fixing the error")]]
    source: NotRequired[ConfigurationSource]


class ValidationResult(TypedDict):
    """Type-safe validation result structure.

    Results from configuration validation operations.
    """

    is_valid: bool
    errors: List[ConfigurationError]
    warnings: NotRequired[List[str]]
    validated_fields: NotRequired[List[str]]
    skipped_fields: NotRequired[List[str]]
    validation_time: NotRequired[float]


class ConfigurationSummary(TypedDict):
    """Type-safe configuration summary structure.

    High-level summary of current configuration state.
    """

    profile: str
    environment: str
    debug_mode: bool
    config_file_path: NotRequired[str]
    loaded_sources: List[str]
    total_settings: int
    overridden_settings: NotRequired[int]
    last_updated: NotRequired[str]


class MigrationInfo(TypedDict):
    """Type-safe configuration migration information.

    Information about migrating from old to new configuration formats.
    """

    from_version: str
    to_version: str
    migration_path: str
    required_changes: List[str]
    optional_changes: NotRequired[List[str]]
    breaking_changes: NotRequired[List[str]]
    migration_script: NotRequired[str]


class SettingsDict(TypedDict):
    """Type-safe general settings dictionary structure.

    Generic settings structure for various configuration contexts.
    """

    project_name: Annotated[str, Doc("Name of the project or application")]
    environment: Literal["development", "testing", "staging", "production"]
    debug: Annotated[bool, Doc("Enable debug mode with verbose logging")]
    providers: NotRequired[Dict[str, Dict[str, ConfigValue]]]
    network: NotRequired[Dict[str, ConfigValue]]
    system: NotRequired[Dict[str, ConfigValue]]
    server: NotRequired[Dict[str, ConfigValue]]
    monitoring: NotRequired[Dict[str, ConfigValue]]
    validation: NotRequired[Dict[str, ConfigValue]]


class ProfileConfig(TypedDict):
    """Type-safe profile configuration structure.

    Configuration overrides for specific deployment profiles.
    """

    profile_name: str
    description: NotRequired[str]
    base_profile: NotRequired[str]
    overrides: Dict[str, ConfigValue]
    environment_variables: NotRequired[Dict[str, str]]
    feature_flags: NotRequired[Dict[str, bool]]


class ConfigurationTemplate(TypedDict):
    """Type-safe configuration template structure.

    Template for generating configuration files.
    """

    template_name: str
    template_version: str
    description: str
    target_profile: str
    required_sections: List[str]
    optional_sections: NotRequired[List[str]]
    default_values: Dict[str, ConfigValue]
    placeholders: NotRequired[Dict[str, str]]
    example_values: NotRequired[Dict[str, ConfigValue]]


# Additional literal types for configuration
ConfigFormat: TypeAlias = Literal["json", "yaml", "toml", "ini", "env"]
LogFormat: TypeAlias = Literal["json", "text", "structured", "simple"]
CompressionType: TypeAlias = Literal["none", "gzip", "bzip2", "lzma"]
EncryptionType: TypeAlias = Literal["none", "aes256", "rsa", "fernet"]
CacheStrategy: TypeAlias = Literal["lru", "lfu", "fifo", "none", "redis", "memory"]
DatabaseType: TypeAlias = Literal["sqlite", "postgresql", "mysql", "mongodb", "redis"]
AuthMethod: TypeAlias = Literal["api_key", "oauth", "jwt", "basic", "bearer", "none"]
RateLimitStrategy: TypeAlias = Literal["token_bucket", "sliding_window", "fixed_window", "none"]
LoadBalancer: TypeAlias = Literal["round_robin", "least_connections", "ip_hash", "random"]
HealthCheckType: TypeAlias = Literal["http", "tcp", "ping", "custom", "none"]


class EnhancedProviderConfig(TypedDict):
    """Enhanced provider configuration with literal constraints."""

    provider_name: Literal["openai", "anthropic", "google", "azure", "huggingface", "local"]
    api_key: NotRequired[Annotated[str, Doc("API key for authentication")]]
    base_url: NotRequired[Annotated[str, Doc("Base URL for API requests")]]
    timeout: NotRequired[Annotated[int, Doc("Request timeout in seconds")]]
    max_retries: NotRequired[Annotated[int, Doc("Maximum number of retry attempts")]]
    rate_limit: NotRequired[Annotated[int, Doc("Maximum requests per minute")]]
    rate_limit_strategy: NotRequired[RateLimitStrategy]
    auth_method: NotRequired[AuthMethod]
    verify_ssl: NotRequired[Annotated[bool, Doc("Verify SSL certificates")]]
    proxy_url: NotRequired[Annotated[str, Doc("HTTP proxy URL")]]
    user_agent: NotRequired[Annotated[str, Doc("Custom user agent string")]]
    headers: NotRequired[Dict[str, str]]
    model_parameters: NotRequired[Dict[str, float | int | bool | str]]
    supported_models: NotRequired[List[str]]
    capabilities: NotRequired[
        List[Literal["text_generation", "embeddings", "chat", "function_calling", "fine_tuning"]]
    ]


class NetworkConfig(TypedDict):
    """Enhanced network configuration with literal constraints."""

    default_timeout: Annotated[int, Doc("Default timeout for network requests in seconds")]
    max_retries: Annotated[int, Doc("Maximum number of retry attempts")]
    retry_delay: Annotated[float, Doc("Delay between retries in seconds")]
    connection_pool_size: NotRequired[Annotated[int, Doc("Size of HTTP connection pool")]]
    dns_timeout: NotRequired[Annotated[int, Doc("DNS resolution timeout in seconds")]]
    keep_alive: NotRequired[Annotated[bool, Doc("Enable HTTP keep-alive")]]
    compression: NotRequired[CompressionType]
    load_balancer: NotRequired[LoadBalancer]
    circuit_breaker_threshold: NotRequired[
        Annotated[int, Doc("Failure threshold for circuit breaker")]
    ]
    health_check: NotRequired[HealthCheckType]
    bandwidth_limit: NotRequired[Annotated[int, Doc("Bandwidth limit in bytes per second")]]


class LoggingConfig(TypedDict):
    """Enhanced logging configuration with literal constraints."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: LogFormat
    output: NotRequired[Literal["console", "file", "syslog", "json_file", "remote"]]
    file_path: NotRequired[Annotated[str, Doc("Path to log file")]]
    max_file_size: NotRequired[Annotated[int, Doc("Maximum log file size in bytes")]]
    backup_count: NotRequired[Annotated[int, Doc("Number of backup log files to keep")]]
    compression: NotRequired[CompressionType]
    rotation: NotRequired[Literal["time", "size", "none"]]
    structured_data: NotRequired[Annotated[bool, Doc("Include structured data in logs")]]
    include_traceback: NotRequired[Annotated[bool, Doc("Include full tracebacks in error logs")]]
    filter_sensitive: NotRequired[Annotated[bool, Doc("Filter sensitive information from logs")]]
    remote_endpoint: NotRequired[Annotated[str, Doc("Remote logging endpoint URL")]]


class CacheConfig(TypedDict):
    """Enhanced cache configuration with literal constraints."""

    strategy: CacheStrategy
    max_size: NotRequired[Annotated[int, Doc("Maximum cache size in items")]]
    max_memory: NotRequired[Annotated[int, Doc("Maximum memory usage in bytes")]]
    ttl: NotRequired[Annotated[int, Doc("Time to live in seconds")]]
    cleanup_interval: NotRequired[Annotated[int, Doc("Cleanup interval in seconds")]]
    persistence: NotRequired[Annotated[bool, Doc("Enable cache persistence to disk")]]
    encryption: NotRequired[EncryptionType]
    compression: NotRequired[CompressionType]
    redis_url: NotRequired[Annotated[str, Doc("Redis connection URL for Redis cache")]]
    key_prefix: NotRequired[Annotated[str, Doc("Prefix for cache keys")]]
    serialization: NotRequired[Literal["pickle", "json", "msgpack", "protobuf"]]


class SecurityConfig(TypedDict):
    """Enhanced security configuration with literal constraints."""

    encryption: EncryptionType
    hash_algorithm: NotRequired[Literal["sha256", "sha512", "blake2b", "argon2"]]
    api_key_rotation: NotRequired[Annotated[bool, Doc("Enable automatic API key rotation")]]
    rate_limiting: NotRequired[Annotated[bool, Doc("Enable rate limiting")]]
    cors_enabled: NotRequired[Annotated[bool, Doc("Enable CORS (Cross-Origin Resource Sharing)")]]
    cors_origins: NotRequired[List[str]]
    allowed_hosts: NotRequired[List[str]]
    max_request_size: NotRequired[Annotated[int, Doc("Maximum request size in bytes")]]
    session_timeout: NotRequired[Annotated[int, Doc("Session timeout in seconds")]]
    password_policy: NotRequired[Dict[str, int | bool]]
    two_factor_auth: NotRequired[Annotated[bool, Doc("Enable two-factor authentication")]]
    audit_logging: NotRequired[Annotated[bool, Doc("Enable security audit logging")]]


class PerformanceConfig(TypedDict):
    """Enhanced performance configuration with literal constraints."""

    max_workers: Annotated[int, Doc("Maximum number of worker threads/processes")]
    batch_size: Annotated[int, Doc("Default batch size for processing")]
    queue_size: NotRequired[Annotated[int, Doc("Maximum queue size")]]
    processing_timeout: NotRequired[Annotated[int, Doc("Processing timeout in seconds")]]
    memory_limit: NotRequired[Annotated[int, Doc("Memory limit in bytes")]]
    cpu_limit: NotRequired[Annotated[float, Doc("CPU usage limit as percentage (0.0-1.0)")]]
    gc_threshold: NotRequired[Annotated[int, Doc("Garbage collection threshold")]]
    profiling_enabled: NotRequired[Annotated[bool, Doc("Enable performance profiling")]]
    metrics_collection: NotRequired[Annotated[bool, Doc("Enable metrics collection")]]
    optimization_level: NotRequired[Literal["none", "basic", "aggressive"]]


class MonitoringConfig(TypedDict):
    """Enhanced monitoring configuration with literal constraints."""

    enabled: Annotated[bool, Doc("Enable monitoring")]
    metrics_endpoint: NotRequired[Annotated[str, Doc("Metrics collection endpoint")]]
    health_check_endpoint: NotRequired[Annotated[str, Doc("Health check endpoint")]]
    alert_threshold: NotRequired[Annotated[float, Doc("Alert threshold value")]]
    notification_channels: NotRequired[List[Literal["email", "slack", "webhook", "sms"]]]
    sample_rate: NotRequired[Annotated[float, Doc("Sampling rate for metrics (0.0-1.0)")]]
    retention_days: NotRequired[Annotated[int, Doc("Metrics retention period in days")]]
    custom_metrics: NotRequired[List[str]]
    dashboard_url: NotRequired[Annotated[str, Doc("Monitoring dashboard URL")]]
    integration: NotRequired[Literal["prometheus", "datadog", "newrelic", "custom"]]
