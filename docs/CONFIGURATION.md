# LLM Lab Configuration Guide

This document provides comprehensive documentation for all configuration options in the LLM Lab framework.

## Table of Contents

- [Overview](#overview)
- [Configuration Sources](#configuration-sources)
- [Quick Start](#quick-start)
- [Configuration Options](#configuration-options)
  - [Project Settings](#project-settings)
  - [Provider Configuration](#provider-configuration)
  - [Dataset Configuration](#dataset-configuration)
  - [Benchmark Configuration](#benchmark-configuration)
  - [Monitoring Configuration](#monitoring-configuration)
  - [Fine-tuning Configuration](#fine-tuning-configuration)
- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Configuration Wizard](#configuration-wizard)
- [Validation](#validation)
- [Examples](#examples)

## Overview

LLM Lab uses a centralized configuration system built on Pydantic Settings, providing:

- **Type-safe configuration** with automatic validation
- **Multiple configuration sources** (environment variables, files, defaults)
- **Configuration inheritance** and overrides
- **Interactive setup wizard** for first-time users
- **Comprehensive validation** with helpful error messages

## Configuration Sources

Configuration is loaded from multiple sources in the following priority order (highest to lowest):

1. **Environment variables** - Override any other settings
2. **`.env` file** - Local environment configuration
3. **Configuration files** (YAML/JSON) - Project or user settings
4. **Default values** - Built-in defaults

## Quick Start

### Using the Configuration Wizard

The easiest way to get started is using the interactive configuration wizard:

```bash
python -m src.config.wizard
```

This will guide you through:
- Setting up API keys
- Configuring providers
- Setting default parameters
- Creating configuration files

### Manual Configuration

1. Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

2. Create a `config.yaml` file (optional):

```yaml
project_name: "My LLM Project"
environment: "development"
debug: false

default_provider: openai

providers:
  openai:
    default_model: "gpt-4o-mini"
    model_parameters:
      temperature: 0.7
      max_tokens: 1000
```

3. Use the configuration in your code:

```python
from src.config.settings import get_settings

settings = get_settings()
provider_config = settings.get_provider_config("openai")
```

## Configuration Options

### Project Settings

| Setting | Type | Default | Description | Environment Variable |
|---------|------|---------|-------------|---------------------|
| `project_name` | string | "LLM Lab" | Name of your project | `PROJECT_NAME` |
| `environment` | string | "development" | Environment (development/staging/production) | `ENVIRONMENT` |
| `debug` | boolean | false | Enable debug mode | `DEBUG` |
| `default_provider` | string | "openai" | Default LLM provider to use | `DEFAULT_PROVIDER` |

### Provider Configuration

Each provider can be configured with the following settings:

| Setting | Type | Required | Description |
|---------|------|----------|-------------|
| `type` | enum | Yes | Provider type (openai/anthropic/google/local) |
| `api_key` | string | Yes* | API key for the provider (*not required for local) |
| `api_base` | string | No | Custom API endpoint URL |
| `organization` | string | No | Organization ID (if applicable) |
| `default_model` | string | No | Default model to use |
| `model_parameters` | object | No | Model generation parameters |
| `retry_config` | object | No | Retry configuration |
| `custom_headers` | dict | No | Custom HTTP headers |

#### Model Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `temperature` | float | 0.0-2.0 | 0.7 | Controls randomness in generation |
| `max_tokens` | integer | 1-100000 | 1000 | Maximum tokens to generate |
| `top_p` | float | 0.0-1.0 | 1.0 | Nucleus sampling parameter |
| `top_k` | integer | 1-100 | 40 | Top-k sampling parameter |
| `frequency_penalty` | float | -2.0-2.0 | null | Frequency penalty for repetition |
| `presence_penalty` | float | -2.0-2.0 | null | Presence penalty for repetition |
| `stop_sequences` | array | - | null | Sequences where generation stops |

#### Retry Configuration

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `max_retries` | integer | 0-10 | 3 | Maximum retry attempts |
| `retry_delay` | float | 0-60 | 1.0 | Initial delay between retries (seconds) |
| `exponential_backoff` | boolean | - | true | Use exponential backoff |
| `timeout_seconds` | integer | 1-600 | 30 | Request timeout (seconds) |

### Dataset Configuration

| Setting | Type | Default | Description | Environment Variable |
|---------|------|---------|-------------|---------------------|
| `base_path` | path | "./datasets" | Base directory for datasets | `DATASET__BASE_PATH` |
| `cache_dir` | path | "./cache/datasets" | Cache directory | `DATASET__CACHE_DIR` |
| `max_cache_size_gb` | float | 10.0 | Maximum cache size in GB | `DATASET__MAX_CACHE_SIZE_GB` |
| `auto_download` | boolean | true | Auto-download datasets | `DATASET__AUTO_DOWNLOAD` |
| `validation_split` | float | 0.2 | Validation split ratio | `DATASET__VALIDATION_SPLIT` |

### Benchmark Configuration

| Setting | Type | Default | Description | Environment Variable |
|---------|------|---------|-------------|---------------------|
| `output_dir` | path | "./results" | Results directory | `BENCHMARK__OUTPUT_DIR` |
| `default_benchmark` | string | "truthfulness" | Default benchmark | `BENCHMARK__DEFAULT_BENCHMARK` |
| `batch_size` | integer | 8 | Batch size for processing | `BENCHMARK__BATCH_SIZE` |
| `parallel_requests` | integer | 4 | Parallel API requests | `BENCHMARK__PARALLEL_REQUESTS` |
| `save_raw_outputs` | boolean | true | Save raw model outputs | `BENCHMARK__SAVE_RAW_OUTPUTS` |
| `enable_caching` | boolean | true | Enable result caching | `BENCHMARK__ENABLE_CACHING` |

### Monitoring Configuration

| Setting | Type | Default | Description | Environment Variable |
|---------|------|---------|-------------|---------------------|
| `log_level` | enum | "INFO" | Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) | `MONITORING__LOG_LEVEL` |
| `log_dir` | path | "./logs" | Log directory | `MONITORING__LOG_DIR` |
| `enable_telemetry` | boolean | false | Enable telemetry | `MONITORING__ENABLE_TELEMETRY` |
| `metrics_port` | integer | 8080 | Metrics endpoint port | `MONITORING__METRICS_PORT` |
| `alert_webhook_url` | string | null | Webhook URL for alerts | `MONITORING__ALERT_WEBHOOK_URL` |
| `performance_tracking` | boolean | true | Track performance metrics | `MONITORING__PERFORMANCE_TRACKING` |

### Fine-tuning Configuration

| Setting | Type | Default | Description | Environment Variable |
|---------|------|---------|-------------|---------------------|
| `models_dir` | path | "./models" | Directory for models | `FINE_TUNING__MODELS_DIR` |
| `checkpoints_dir` | path | "./checkpoints" | Checkpoints directory | `FINE_TUNING__CHECKPOINTS_DIR` |
| `default_batch_size` | integer | 4 | Training batch size | `FINE_TUNING__DEFAULT_BATCH_SIZE` |
| `gradient_accumulation_steps` | integer | 4 | Gradient accumulation | `FINE_TUNING__GRADIENT_ACCUMULATION_STEPS` |
| `learning_rate` | float | 2e-5 | Learning rate | `FINE_TUNING__LEARNING_RATE` |
| `num_train_epochs` | integer | 3 | Training epochs | `FINE_TUNING__NUM_TRAIN_EPOCHS` |
| `warmup_ratio` | float | 0.1 | Warmup ratio | `FINE_TUNING__WARMUP_RATIO` |
| `use_mixed_precision` | boolean | true | Use mixed precision | `FINE_TUNING__USE_MIXED_PRECISION` |

## Environment Variables

### API Keys

API keys are loaded from environment variables:

```bash
# Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Optional: Custom API endpoints
OPENAI_API_BASE=https://api.openai.com/v1
ANTHROPIC_API_BASE=https://api.anthropic.com/v1
```

### Nested Configuration

Use double underscores (`__`) to set nested configuration values:

```bash
# Dataset configuration
DATASET__BASE_PATH=/data/datasets
DATASET__CACHE_DIR=/data/cache
DATASET__MAX_CACHE_SIZE_GB=50.0

# Benchmark configuration
BENCHMARK__OUTPUT_DIR=/results
BENCHMARK__BATCH_SIZE=16
BENCHMARK__PARALLEL_REQUESTS=8

# Monitoring configuration
MONITORING__LOG_LEVEL=DEBUG
MONITORING__LOG_DIR=/var/log/llm-lab
MONITORING__ENABLE_TELEMETRY=true
```

### Model-specific Overrides

Set model-specific parameters using environment variables:

```bash
# OpenAI GPT-4 specific settings
GPT4_TEMPERATURE=0.5
GPT4_MAX_TOKENS=2000
GPT4_TOP_P=0.9
GPT4_TIMEOUT=60

# Anthropic Claude specific settings
CLAUDE_3_OPUS_TEMPERATURE=0.8
CLAUDE_3_OPUS_MAX_TOKENS=4000
```

## Configuration Files

### YAML Configuration

Create a `config.yaml` file:

```yaml
project_name: "Advanced LLM Lab"
environment: "production"
debug: false

default_provider: openai

providers:
  openai:
    type: openai
    default_model: "gpt-4o"
    model_parameters:
      temperature: 0.7
      max_tokens: 2000
      top_p: 0.95
    retry_config:
      max_retries: 5
      retry_delay: 2.0
      exponential_backoff: true
      timeout_seconds: 60

  anthropic:
    type: anthropic
    default_model: "claude-3-opus-20240229"
    model_parameters:
      temperature: 0.8
      max_tokens: 4000
    retry_config:
      timeout_seconds: 90

dataset:
  base_path: "/data/datasets"
  cache_dir: "/data/cache"
  max_cache_size_gb: 50.0
  auto_download: true
  validation_split: 0.2

benchmark:
  output_dir: "/results"
  default_benchmark: "mmlu"
  batch_size: 16
  parallel_requests: 8
  save_raw_outputs: true
  enable_caching: true

monitoring:
  log_level: "INFO"
  log_dir: "/var/log/llm-lab"
  enable_telemetry: true
  metrics_port: 9090
  performance_tracking: true

fine_tuning:
  models_dir: "/models"
  checkpoints_dir: "/checkpoints"
  default_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  num_train_epochs: 5
  warmup_ratio: 0.1
  use_mixed_precision: true
```

### JSON Configuration

Alternatively, use a `config.json` file:

```json
{
  "project_name": "LLM Lab",
  "environment": "development",
  "debug": true,
  "default_provider": "openai",
  "providers": {
    "openai": {
      "type": "openai",
      "default_model": "gpt-4o-mini",
      "model_parameters": {
        "temperature": 0.7,
        "max_tokens": 1000
      }
    }
  },
  "dataset": {
    "base_path": "./datasets",
    "auto_download": true
  },
  "benchmark": {
    "output_dir": "./results",
    "batch_size": 8
  }
}
```

### Loading Configuration Files

```python
from src.config.settings import Settings

# Load from YAML
settings = Settings.from_yaml("config.yaml")

# Load from JSON
settings = Settings.from_json("config.json")

# Load with environment variable overrides
settings = Settings()  # Loads .env and applies overrides
```

## Configuration Wizard

The configuration wizard provides an interactive setup experience:

### Running the Wizard

```bash
# First-time setup
python -m src.config.wizard

# Reconfigure existing setup
python -m src.config.wizard --reconfigure

# Validate configuration
python -m src.config.wizard --validate

# Save to custom location
python -m src.config.wizard --config-path /path/to/config.yaml
```

### Wizard Features

- **API Key Setup**: Securely configure API keys with password input
- **Provider Configuration**: Set up each provider with custom settings
- **Default Selection**: Choose default provider and models
- **Advanced Options**: Configure datasets, benchmarks, monitoring, and fine-tuning
- **Validation**: Verify configuration before saving
- **Summary**: Review all settings before completion

## Validation

### Automatic Validation

The settings system automatically validates:

- **Type checking**: Ensures correct data types
- **Range validation**: Checks numeric values are within valid ranges
- **Required fields**: Verifies required fields are present
- **Path validation**: Checks directories can be created/accessed
- **API key presence**: Ensures API keys exist for non-local providers

### Manual Validation

```python
from src.config.settings import get_settings

settings = get_settings()
errors = settings.validate_all()

if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid!")
```

### Validation in CLI

```bash
# Validate current configuration
python -m src.config.wizard --validate

# Validate specific config file
python -c "from src.config.settings import Settings; s = Settings.from_yaml('config.yaml'); print(s.validate_all())"
```

## Examples

### Example 1: Minimal Configuration

`.env` file:
```bash
OPENAI_API_KEY=sk-...
```

Python code:
```python
from src.config.settings import get_settings

settings = get_settings()
# Uses all defaults with OpenAI provider
```

### Example 2: Multi-Provider Setup

`config.yaml`:
```yaml
default_provider: anthropic

providers:
  openai:
    type: openai
    default_model: "gpt-4o"
    model_parameters:
      temperature: 0.5

  anthropic:
    type: anthropic
    default_model: "claude-3-opus-20240229"
    model_parameters:
      temperature: 0.7
      max_tokens: 4000

  google:
    type: google
    default_model: "gemini-1.5-pro"
    model_parameters:
      temperature: 0.8
```

### Example 3: Production Configuration

`.env`:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
ENVIRONMENT=production
DEBUG=false
MONITORING__LOG_LEVEL=WARNING
MONITORING__ENABLE_TELEMETRY=true
```

`config.yaml`:
```yaml
project_name: "Production LLM Service"
environment: "production"

providers:
  openai:
    type: openai
    default_model: "gpt-4o"
    retry_config:
      max_retries: 5
      timeout_seconds: 60
      exponential_backoff: true

benchmark:
  output_dir: "/var/lib/llm-lab/results"
  parallel_requests: 16
  enable_caching: true

monitoring:
  log_dir: "/var/log/llm-lab"
  alert_webhook_url: "https://hooks.slack.com/services/..."
  performance_tracking: true
```

### Example 4: Development with Local Models

```yaml
project_name: "Local LLM Development"
environment: "development"
debug: true

default_provider: local

providers:
  local:
    type: local
    default_model: "llama-2-7b"
    model_parameters:
      temperature: 0.7
      max_tokens: 512

dataset:
  base_path: "./dev-datasets"
  cache_dir: "./dev-cache"

benchmark:
  output_dir: "./dev-results"
  batch_size: 1
  parallel_requests: 1
```

### Example 5: Using Configuration in Code

```python
from src.config.settings import get_settings, load_settings
from pathlib import Path

# Load default configuration
settings = get_settings()

# Or load from specific file
settings = load_settings("config.yaml")

# Access configuration values
print(f"Project: {settings.project_name}")
print(f"Environment: {settings.environment}")
print(f"Debug mode: {settings.debug}")

# Get provider configuration
openai_config = settings.get_provider_config("openai")
print(f"OpenAI model: {openai_config.default_model}")
print(f"Temperature: {openai_config.model_parameters.temperature}")

# Check validation
errors = settings.validate_all()
if errors:
    for error in errors:
        print(f"Error: {error}")

# Save configuration
settings.to_yaml("backup-config.yaml")
settings.to_json("backup-config.json")
```

## Best Practices

1. **Use environment variables for secrets**: Never commit API keys to version control
2. **Create environment-specific configs**: Use different configs for dev/staging/prod
3. **Validate configuration on startup**: Check for errors before running
4. **Use the wizard for initial setup**: Ensures all required fields are configured
5. **Document custom settings**: Add comments in YAML files for clarity
6. **Version control config templates**: Commit example configs, not actual ones
7. **Use consistent naming**: Follow the naming conventions in this guide
8. **Set appropriate defaults**: Choose sensible defaults for your use case
9. **Monitor configuration changes**: Log when configuration is loaded/changed
10. **Test configuration thoroughly**: Validate all settings work as expected

## Troubleshooting

### Common Issues

**Issue**: API key not found
```bash
Error: API key 'OPENAI_API_KEY' not found
```
**Solution**: Set the API key in `.env` file or environment variable

**Issue**: Invalid configuration value
```bash
Error: Temperature 3.0 must be between 0 and 2
```
**Solution**: Check value ranges in the configuration documentation

**Issue**: Cannot write to directory
```bash
Error: Cannot write to output_dir (/results): Permission denied
```
**Solution**: Ensure the user has write permissions or change the directory

**Issue**: Configuration file not found
```bash
Error: Configuration file 'config.yaml' not found
```
**Solution**: Create the file or specify correct path

### Debug Mode

Enable debug mode for detailed configuration information:

```python
settings = get_settings()
settings.debug = True

# Or via environment
DEBUG=true python your_script.py
```

## Migration Guide

### From Old Configuration System

If migrating from the old configuration system:

1. **Export existing settings**:
```python
# Old system
from src.config.config import get_full_config
old_config = get_full_config()

# Convert to new system
from src.config.settings import Settings
new_settings = Settings()
# Map old config values to new settings
```

2. **Update API key references**:
```python
# Old
api_key = os.getenv("GOOGLE_API_KEY")

# New
settings = get_settings()
api_key = settings.google_api_key
```

3. **Update provider configuration**:
```python
# Old
from src.config.provider_config import get_config_manager
config_manager = get_config_manager()

# New
from src.config.settings import get_settings
settings = get_settings()
provider_config = settings.get_provider_config("openai")
```

## Support

For configuration issues or questions:

1. Check this documentation
2. Run the configuration wizard validator
3. Review example configurations
4. Open an issue on GitHub
5. Check the FAQ section in the main README
