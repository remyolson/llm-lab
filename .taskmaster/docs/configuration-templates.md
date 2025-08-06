# Configuration Templates - LLM Lab

This document provides configuration templates for different environments and use cases.

## Quick Start Templates

### Development Configuration (YAML)

```yaml
# llmlab-dev.yaml
project_name: "LLM Lab - Development"
environment: "development"
debug: true

# Provider Configuration
providers:
  openai:
    type: "openai"
    api_key: "${OPENAI_API_KEY}"
    default_model: "gpt-4o-mini"
    model_parameters:
      temperature: 0.7
      max_tokens: 1000
      top_p: 1.0

# Network Settings
network:
  default_timeout: 60  # Longer timeouts for development
  generation_timeout: 600
  model_pull_timeout: 3600
  ollama_base_url: "http://localhost:11434"

# System Settings
system:
  default_batch_size: 4  # Smaller batches for development
  max_workers: 2
  memory_threshold: 0.7

# Server Configuration
server:
  api_port: 8000
  websocket_port: 8001
  dashboard_port: 7860
  api_host: "127.0.0.1"  # Localhost only for dev
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:3001"

# Monitoring
monitoring:
  log_level: "DEBUG"
  performance_logging: true
  enable_telemetry: false
```

### Production Configuration (YAML)

```yaml
# llmlab-prod.yaml
project_name: "LLM Lab - Production"
environment: "production"
debug: false

# Provider Configuration
providers:
  openai:
    type: "openai"
    api_key: "${OPENAI_API_KEY}"
    default_model: "gpt-4o"
    model_parameters:
      temperature: 0.3  # More conservative for production
      max_tokens: 2000
      top_p: 0.9

# Network Settings (Optimized for production)
network:
  default_timeout: 30
  generation_timeout: 300
  model_pull_timeout: 1800
  api_request_timeout: 15

# System Settings (High performance)
system:
  default_batch_size: 16
  max_batch_size: 64
  max_workers: 8
  memory_threshold: 0.85
  vram_threshold: 0.9

# Server Configuration
server:
  api_port: 8000
  websocket_port: 8001
  dashboard_port: 7860
  api_host: "0.0.0.0"
  cors_origins:
    - "https://your-frontend.com"
    - "https://api.your-domain.com"

# Monitoring (Production settings)
monitoring:
  log_level: "WARNING"
  performance_logging: true
  enable_telemetry: true
  structured_logging: true
  performance_tracking: true
```

### Local Model Configuration (YAML)

```yaml
# llmlab-local.yaml - For Ollama/Local models only
project_name: "LLM Lab - Local Models"
environment: "development"
debug: true

# Network Settings (Optimized for local)
network:
  default_timeout: 120  # Local models may be slower
  generation_timeout: 900  # 15 minutes for local generation
  model_pull_timeout: 3600  # 1 hour for model downloads
  ollama_base_url: "http://localhost:11434"

# System Settings (Conservative for local hardware)
system:
  default_batch_size: 2
  small_batch_size: 1
  max_workers: 2
  memory_threshold: 0.6  # Leave more memory for models
  vram_threshold: 0.8

# Server Configuration
server:
  api_port: 8000
  dashboard_port: 7860
  api_host: "127.0.0.1"

# Monitoring
monitoring:
  log_level: "INFO"
  performance_logging: true
```

## Environment Variable Configuration

### .env Template

```bash
# .env - Environment variables for LLM Lab

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Environment
LLMLAB_PROFILE=development
LLMLAB_DEBUG=true
LLMLAB_ENVIRONMENT=development

# Network Configuration
NETWORK_DEFAULT_TIMEOUT=30
NETWORK_GENERATION_TIMEOUT=300
NETWORK_OLLAMA_BASE_URL=http://localhost:11434

# System Configuration
SYSTEM_DEFAULT_BATCH_SIZE=8
SYSTEM_MAX_WORKERS=4

# Server Configuration
SERVER_API_PORT=8000
SERVER_WEBSOCKET_PORT=8001
SERVER_DASHBOARD_PORT=7860
SERVER_API_HOST=0.0.0.0

# Monitoring
LLMLAB_LOG_LEVEL=INFO
```

## JSON Configuration Templates

### Minimal JSON Configuration

```json
{
  "project_name": "LLM Lab",
  "environment": "development",
  "providers": {
    "openai": {
      "type": "openai",
      "api_key": "${OPENAI_API_KEY}",
      "default_model": "gpt-4o-mini"
    }
  },
  "network": {
    "default_timeout": 30,
    "generation_timeout": 300
  },
  "server": {
    "api_port": 8000,
    "cors_origins": ["http://localhost:3000"]
  }
}
```

## Configuration Migration Examples

### Legacy to New Format Migration

```python
# migrate_config.py - Example migration script

from src.config.cli import migrate_config
from pathlib import Path

# Migrate old configuration
success = migrate_config(
    old_config_path=Path("config/old_config.json"),
    new_config_path=Path("config/llmlab.yaml")
)

if success:
    print("✅ Configuration migrated successfully!")
else:
    print("❌ Migration failed")
```

## CLI Usage Examples

### Validate Configuration

```bash
# Validate a configuration file
python -m src.config.cli validate llmlab-prod.yaml

# Strict validation mode
python -m src.config.cli validate --strict llmlab-prod.yaml
```

### Generate Templates

```bash
# Generate development template
python -m src.config.cli generate-template --format yaml --profile development

# Generate production template and save to file
python -m src.config.cli generate-template --format yaml --profile production --output config/prod-template.yaml

# Generate JSON template
python -m src.config.cli generate-template --format json --profile staging
```

### Export JSON Schema

```bash
# Export schema to stdout
python -m src.config.cli export-schema

# Save schema to file
python -m src.config.cli export-schema --output config-schema.json
```

### Migrate Configurations

```bash
# Migrate old JSON to new YAML
python -m src.config.cli migrate old_config.json new_config.yaml

# Migrate YAML to JSON
python -m src.config.cli migrate config.yaml config.json
```

## Best Practices

### 1. Environment-Specific Configurations

- Use separate config files for each environment
- Store sensitive data in environment variables
- Use configuration profiles for environment-specific settings

### 2. Configuration Validation

- Always validate configurations before deployment
- Use the CLI validation tool in CI/CD pipelines
- Test configurations in staging environments

### 3. Security Considerations

- Never commit API keys to version control
- Use environment variable substitution for secrets
- Restrict file permissions on production config files

### 4. Configuration Management

- Use version control for configuration templates
- Document configuration changes
- Implement configuration review processes

## Configuration Hierarchy

The configuration system loads values in this order (highest to lowest priority):

1. **CLI Arguments** - Direct command-line overrides
2. **Environment Variables** - System environment variables
3. **Configuration Files** - YAML/TOML/JSON files
4. **Profile Settings** - Environment-specific defaults
5. **Default Values** - Built-in fallback values

This allows for flexible configuration management across different deployment scenarios.
