# Configuration API Reference

## Overview

The configuration module provides centralized configuration management for all LLM Lab components.

## Configuration Manager

### `ConfigManager`

Central configuration management class.

```python
from src.config import ConfigManager

# Load configuration
config = ConfigManager()

# Get provider configuration
openai_config = config.get_provider_config("openai")

# Get global settings
settings = config.get_settings()
```

### Configuration Sources

Configuration is loaded from multiple sources in order of precedence:

1. Environment variables (highest priority)
2. `.env` file
3. `config.yaml` file
4. Default values (lowest priority)

## Provider Configuration

### `ProviderConfig`

Provider-specific configuration.

```python
from src.config import ProviderConfig
from pydantic import BaseModel

class ProviderConfig(BaseModel):
    """Provider configuration schema."""

    api_key: Optional[str] = None
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0

    class Config:
        extra = "allow"  # Allow provider-specific fields
```

### Loading Provider Configuration

```python
# From environment
os.environ['OPENAI_API_KEY'] = 'sk-...'
config = ProviderConfig(model="gpt-4")

# From YAML
with open("providers.yaml") as f:
    provider_configs = yaml.safe_load(f)

config = ProviderConfig(**provider_configs["openai"])

# From code
config = ProviderConfig(
    api_key="sk-...",
    model="gpt-4",
    temperature=0.8,
    max_tokens=2000
)
```

## Global Settings

### `Settings`

Application-wide settings.

```python
from src.config import Settings

class Settings(BaseModel):
    """Global application settings."""

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None

    # Performance
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_concurrent_requests: int = 10

    # Security
    encrypt_api_keys: bool = True
    mask_pii: bool = True

    # Storage
    results_dir: str = "./results"
    cache_dir: str = "./.cache"

    # Monitoring
    metrics_enabled: bool = True
    metrics_port: int = 9090
```

### Environment Variables

All settings can be overridden via environment variables:

```bash
# Set log level
export LLLM_LOG_LEVEL=DEBUG

# Enable/disable cache
export LLLM_CACHE_ENABLED=false

# Set results directory
export LLLM_RESULTS_DIR=/data/results
```

## Configuration Files

### `providers.yaml`

Provider configurations:

```yaml
providers:
  openai:
    model: gpt-4
    temperature: 0.7
    max_tokens: 2000
    timeout: 60

  anthropic:
    model: claude-3-5-sonnet-20241022
    temperature: 0.8
    max_tokens: 4000

  google:
    model: gemini-1.5-pro
    temperature: 0.9
    max_tokens: 8192

  azure:
    endpoint: https://myinstance.openai.azure.com
    deployment: gpt-4-deployment
    api_version: "2024-02-15-preview"
```

### `benchmarks.yaml`

Benchmark configurations:

```yaml
benchmarks:
  truthfulness:
    dataset: truthfulqa
    metrics:
      - accuracy
      - f1_score
    sample_size: 100

  performance:
    iterations: 10
    warmup: 2
    metrics:
      - latency
      - throughput
      - tokens_per_second
```

## Configuration Validation

### Schema Validation

```python
from src.config import validate_config

# Validate provider config
try:
    validate_config(provider_config, schema="provider")
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")

# Validate benchmark config
validate_config(benchmark_config, schema="benchmark")
```

### Custom Validators

```python
from pydantic import validator

class CustomProviderConfig(ProviderConfig):
    rate_limit: int = 100

    @validator('rate_limit')
    def validate_rate_limit(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('Rate limit must be between 1 and 1000')
        return v
```

## Configuration Profiles

### Using Profiles

```python
# Load specific profile
config = ConfigManager(profile="production")

# Profile-specific settings
if config.profile == "development":
    config.set("log_level", "DEBUG")
elif config.profile == "production":
    config.set("log_level", "WARNING")
```

### Profile Files

- `config.dev.yaml` - Development settings
- `config.prod.yaml` - Production settings
- `config.test.yaml` - Test settings

## Dynamic Configuration

### Runtime Updates

```python
# Update configuration at runtime
config = ConfigManager()

# Update single value
config.set("providers.openai.temperature", 0.9)

# Update multiple values
config.update({
    "log_level": "DEBUG",
    "cache_enabled": False
})

# Reload from files
config.reload()
```

### Configuration Observers

```python
# Register configuration change listener
def on_config_change(key: str, old_value: Any, new_value: Any):
    print(f"Config {key} changed from {old_value} to {new_value}")

config.add_observer("providers.openai.model", on_config_change)
```

## Best Practices

1. **Use Environment Variables for Secrets**
   ```python
   # Never hardcode API keys
   api_key = os.getenv("OPENAI_API_KEY")
   if not api_key:
       raise ValueError("OPENAI_API_KEY not set")
   ```

2. **Validate Early**
   ```python
   # Validate configuration on startup
   try:
       config = ConfigManager()
       config.validate()
   except ConfigurationError as e:
       logger.error(f"Configuration error: {e}")
       sys.exit(1)
   ```

3. **Use Type Hints**
   ```python
   # Type-safe configuration access
   def get_provider_config(name: str) -> ProviderConfig:
       config = ConfigManager()
       return ProviderConfig(**config.get(f"providers.{name}"))
   ```

4. **Separate Concerns**
   ```python
   # Keep provider configs separate from app configs
   provider_config = config.get_provider_config("openai")
   app_settings = config.get_settings()
   ```

## Configuration Examples

### Multi-Environment Setup

```python
# config.py
import os

ENV = os.getenv("LLLM_ENV", "development")

CONFIGS = {
    "development": {
        "log_level": "DEBUG",
        "cache_enabled": True,
        "providers": {
            "openai": {"model": "gpt-3.5-turbo"}
        }
    },
    "production": {
        "log_level": "INFO",
        "cache_enabled": True,
        "providers": {
            "openai": {"model": "gpt-4"}
        }
    }
}

config = CONFIGS[ENV]
```

### Secure Configuration

```python
# Encrypted configuration
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)

    def get_api_key(self, provider: str) -> str:
        encrypted = os.getenv(f"{provider.upper()}_API_KEY_ENCRYPTED")
        if encrypted:
            return self.cipher.decrypt(encrypted.encode()).decode()
        return os.getenv(f"{provider.upper()}_API_KEY")
```

## See Also

- [Providers API](providers.md) - Provider configuration usage
- [Environment Setup](../guides/PREREQUISITES.md) - Setting up configuration
- [Security Best Practices](../guides/security.md) - Secure configuration
