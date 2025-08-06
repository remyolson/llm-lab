# Configuration Reference - LLM Lab

This document provides a comprehensive reference for all configuration parameters in the LLM Lab system.

## Configuration Domains

### NetworkConfig

Network and connectivity settings for timeouts and API endpoints.

| Parameter | Type | Default | Description | Environment Variable |
|-----------|------|---------|-------------|---------------------|
| `default_timeout` | int | 30 | Default timeout for network requests (seconds) | `NETWORK_DEFAULT_TIMEOUT` |
| `generation_timeout` | int | 300 | Timeout for LLM generation requests (seconds) | `NETWORK_GENERATION_TIMEOUT` |
| `model_pull_timeout` | int | 1800 | Timeout for model downloads (seconds) | `NETWORK_MODEL_PULL_TIMEOUT` |
| `api_request_timeout` | int | 30 | Timeout for API requests (seconds) | `NETWORK_API_REQUEST_TIMEOUT` |
| `ollama_base_url` | str | "http://localhost:11434" | Base URL for Ollama server | `NETWORK_OLLAMA_BASE_URL` |
| `monitoring_poll_interval` | float | 60.0 | Monitoring dashboard polling interval | `NETWORK_MONITORING_POLL_INTERVAL` |
| `buffer_poll_interval` | float | 0.1 | Buffer polling interval (seconds) | `NETWORK_BUFFER_POLL_INTERVAL` |

### SystemConfig

System performance and resource management settings.

| Parameter | Type | Default | Description | Environment Variable |
|-----------|------|---------|-------------|---------------------|
| `default_batch_size` | int | 8 | Default batch size for processing | `SYSTEM_DEFAULT_BATCH_SIZE` |
| `small_batch_size` | int | 4 | Small batch size for memory-constrained ops | `SYSTEM_SMALL_BATCH_SIZE` |
| `large_batch_size` | int | 16 | Large batch size for throughput optimization | `SYSTEM_LARGE_BATCH_SIZE` |
| `max_batch_size` | int | 32 | Maximum batch size | `SYSTEM_MAX_BATCH_SIZE` |
| `train_batch_size` | int | 8 | Training batch size per device | `SYSTEM_TRAIN_BATCH_SIZE` |
| `eval_batch_size` | int | 16 | Evaluation batch size per device | `SYSTEM_EVAL_BATCH_SIZE` |
| `download_chunk_size` | int | 8192 | File download chunk size (bytes) | `SYSTEM_DOWNLOAD_CHUNK_SIZE` |
| `buffer_size` | int | 10000 | Internal buffer size | `SYSTEM_BUFFER_SIZE` |
| `max_workers` | int | 4 | Maximum worker threads/processes | `SYSTEM_MAX_WORKERS` |
| `dataloader_workers` | int | 4 | Data loading workers | `SYSTEM_DATALOADER_WORKERS` |
| `memory_threshold` | float | 0.8 | Memory usage threshold (0.0-1.0) | `SYSTEM_MEMORY_THRESHOLD` |
| `vram_threshold` | float | 0.9 | VRAM usage threshold (0.0-1.0) | `SYSTEM_VRAM_THRESHOLD` |

### ServerConfig

Server and API configuration settings.

| Parameter | Type | Default | Description | Environment Variable |
|-----------|------|---------|-------------|---------------------|
| `api_port` | int | 8000 | Main API server port | `SERVER_API_PORT` |
| `api_host` | str | "0.0.0.0" | API server host address | `SERVER_API_HOST` |
| `websocket_port` | int | 8001 | WebSocket server port | `SERVER_WEBSOCKET_PORT` |
| `dashboard_port` | int | 7860 | Gradio dashboard port | `SERVER_DASHBOARD_PORT` |
| `monitoring_port` | int | 8002 | Monitoring services port | `SERVER_MONITORING_PORT` |
| `cors_origins` | List[str] | ["http://localhost:3000", "http://localhost:3001"] | Allowed CORS origins | `SERVER_CORS_ORIGINS` |
| `cors_allow_credentials` | bool | True | Allow credentials in CORS requests | `SERVER_CORS_ALLOW_CREDENTIALS` |
| `api_prefix` | str | "/api/v1" | API path prefix | `SERVER_API_PREFIX` |
| `docs_url` | str | "/docs" | OpenAPI documentation URL | `SERVER_DOCS_URL` |

### ModelParameters

Default model generation parameters.

| Parameter | Type | Default | Description | Environment Variable |
|-----------|------|---------|-------------|---------------------|
| `temperature` | float | 0.7 | Controls randomness (0-2) | `MODEL_TEMPERATURE` |
| `max_tokens` | int | 1000 | Maximum tokens to generate | `MODEL_MAX_TOKENS` |
| `top_p` | float | 1.0 | Nucleus sampling parameter (0-1) | `MODEL_TOP_P` |
| `top_k` | int | None | Top-k sampling parameter | `MODEL_TOP_K` |
| `frequency_penalty` | float | None | Frequency penalty (-2 to 2) | `MODEL_FREQUENCY_PENALTY` |
| `presence_penalty` | float | None | Presence penalty (-2 to 2) | `MODEL_PRESENCE_PENALTY` |
| `max_prompt_length` | int | 50000 | Maximum prompt length (characters) | `MODEL_MAX_PROMPT_LENGTH` |
| `default_max_length` | int | 512 | Default tokenizer max_length | `MODEL_DEFAULT_MAX_LENGTH` |
| `extended_max_length` | int | 1024 | Extended max_length for longer context | `MODEL_EXTENDED_MAX_LENGTH` |

#### Evaluation-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_temperature_conservative` | float | 0.1 | Conservative temperature for deterministic evaluation |
| `eval_temperature_standard` | float | 0.7 | Standard temperature for balanced evaluation |
| `eval_max_new_tokens_short` | int | 50 | Max new tokens for short responses |
| `eval_max_new_tokens_medium` | int | 100 | Max new tokens for medium responses |
| `eval_max_new_tokens_long` | int | 150 | Max new tokens for long responses |

### RetryConfig

Retry and error handling configuration.

| Parameter | Type | Default | Description | Environment Variable |
|-----------|------|---------|-------------|---------------------|
| `max_retries` | int | 3 | Maximum retry attempts | `RETRY_MAX_RETRIES` |
| `retry_delay` | float | 1.0 | Initial retry delay (seconds) | `RETRY_DELAY` |
| `exponential_backoff` | bool | True | Use exponential backoff | `RETRY_EXPONENTIAL_BACKOFF` |
| `backoff_factor` | float | 2.0 | Exponential backoff factor | `RETRY_BACKOFF_FACTOR` |
| `max_retry_delay` | float | 60.0 | Maximum retry delay (seconds) | `RETRY_MAX_DELAY` |
| `timeout_seconds` | int | 30 | Request timeout (seconds) | `RETRY_TIMEOUT` |
| `retry_jitter` | bool | True | Add random jitter to delays | `RETRY_JITTER` |
| `retry_timeout` | int | 300 | Total timeout for all retries | `RETRY_TOTAL_TIMEOUT` |

### MonitoringConfig

Logging and monitoring configuration.

| Parameter | Type | Default | Description | Environment Variable |
|-----------|------|---------|-------------|---------------------|
| `log_level` | LogLevel | INFO | Logging level | `LLMLAB_LOG_LEVEL` |
| `log_dir` | Path | "./logs" | Log files directory | `MONITORING_LOG_DIR` |
| `log_format` | str | "%(asctime)s - %(name)s - %(levelname)s - %(message)s" | Log format | `MONITORING_LOG_FORMAT` |
| `log_max_size` | int | 10485760 | Max log file size (bytes) | `MONITORING_LOG_MAX_SIZE` |
| `log_backup_count` | int | 5 | Number of backup log files | `MONITORING_LOG_BACKUP_COUNT` |
| `provider_log_level` | LogLevel | INFO | Provider modules log level | `MONITORING_PROVIDER_LOG_LEVEL` |
| `evaluation_log_level` | LogLevel | INFO | Evaluation modules log level | `MONITORING_EVALUATION_LOG_LEVEL` |
| `training_log_level` | LogLevel | INFO | Training modules log level | `MONITORING_TRAINING_LOG_LEVEL` |
| `performance_logging` | bool | False | Enable performance logging | `MONITORING_PERFORMANCE_LOGGING` |
| `structured_logging` | bool | True | Enable structured JSON logging | `MONITORING_STRUCTURED_LOGGING` |
| `enable_telemetry` | bool | False | Enable telemetry collection | `MONITORING_ENABLE_TELEMETRY` |
| `metrics_port` | int | 8080 | Metrics endpoint port | `MONITORING_METRICS_PORT` |
| `performance_tracking` | bool | True | Track performance metrics | `MONITORING_PERFORMANCE_TRACKING` |

### ValidationConfig

Validation rules and constraints.

| Parameter | Type | Default | Description | Environment Variable |
|-----------|------|---------|-------------|---------------------|
| `min_prompt_length` | int | 1 | Minimum prompt length (characters) | `VALIDATION_MIN_PROMPT_LENGTH` |
| `max_prompt_length` | int | 50000 | Maximum prompt length (characters) | `VALIDATION_MAX_PROMPT_LENGTH` |
| `temperature_min` | float | 0.0 | Minimum temperature value | `VALIDATION_TEMPERATURE_MIN` |
| `temperature_max` | float | 2.0 | Maximum temperature value | `VALIDATION_TEMPERATURE_MAX` |
| `top_p_min` | float | 0.0 | Minimum top_p value | `VALIDATION_TOP_P_MIN` |
| `top_p_max` | float | 1.0 | Maximum top_p value | `VALIDATION_TOP_P_MAX` |
| `min_max_tokens` | int | 1 | Minimum max_tokens value | `VALIDATION_MIN_MAX_TOKENS` |
| `max_max_tokens` | int | 100000 | Maximum max_tokens value | `VALIDATION_MAX_MAX_TOKENS` |
| `strict_validation` | bool | True | Enable strict validation mode | `VALIDATION_STRICT` |
| `validation_warnings` | bool | True | Show validation warnings | `VALIDATION_WARNINGS` |

## Provider Configuration

### ProviderConfig Structure

```yaml
providers:
  openai:
    type: "openai"
    api_key: "${OPENAI_API_KEY}"
    default_model: "gpt-4o-mini"
    model_parameters:
      temperature: 0.7
      max_tokens: 1000
      top_p: 1.0
    retry_config:
      max_retries: 3
      retry_delay: 1.0
    custom_headers:
      User-Agent: "LLMLabv1.0"
```

### Supported Provider Types

- **openai** - OpenAI API (GPT models)
- **anthropic** - Anthropic API (Claude models)
- **google** - Google API (Gemini models)
- **local** - Local models via Ollama
- **custom** - Custom provider implementations

## Configuration Profiles

### Available Profiles

- **development** - Development environment settings
- **testing** - Test environment settings
- **staging** - Staging environment settings
- **production** - Production environment settings

### Profile-Specific Overrides

```yaml
# Development profile automatically applies:
debug: true
monitoring:
  log_level: "DEBUG"
  performance_logging: true
network:
  default_timeout: 60

# Production profile automatically applies:
debug: false
monitoring:
  log_level: "WARNING"
  enable_telemetry: true
```

## Environment Variable Mapping

Environment variables use the pattern: `{DOMAIN}_{PARAMETER}` (uppercase).

Examples:
- `NETWORK_DEFAULT_TIMEOUT=30`
- `SYSTEM_DEFAULT_BATCH_SIZE=8`
- `SERVER_API_PORT=8000`
- `MODEL_TEMPERATURE=0.7`

## Configuration Loading Priority

1. **CLI Arguments** (highest priority)
2. **Environment Variables**
3. **Configuration Files** (YAML/TOML/JSON)
4. **Profile Defaults**
5. **Built-in Defaults** (lowest priority)

## Validation Rules

- All numeric parameters have min/max constraints
- URL parameters are validated for proper format
- File paths are checked for writability
- Provider configurations require valid API keys
- Timeout values must be positive integers
- Percentage values (thresholds) must be between 0.0 and 1.0
