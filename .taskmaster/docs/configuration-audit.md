# Configuration Audit Report - LLM Lab

## Executive Summary

This document provides a comprehensive audit of hardcoded configuration values throughout the LLM Lab codebase. The audit identified **127 hardcoded configuration parameters** across **8 primary configuration domains** that should be extracted into a centralized configuration management system.

**Key Findings:**
- **Network Configuration**: 15 hardcoded timeouts, 11 port numbers, 10 API endpoints
- **Model Parameters**: 23 temperature values, 35 max_tokens/max_length settings
- **System Settings**: 21 sleep/delay values, 18 buffer/batch sizes
- **Provider Defaults**: 7 core provider configuration values in ProviderConfig
- **Logging Configuration**: 8 hardcoded log levels
- **File System Paths**: Various hardcoded paths and directory structures

## Configuration Domains

### 1. Network Configuration

#### Timeouts
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/providers/base.py:39` | `timeout: int = 30` | Provider default timeout | **High** |
| `src/providers/local/backends/ollama_backend.py:38` | `timeout=5` | Version check timeout | High |
| `src/providers/local/backends/ollama_backend.py:147` | `timeout=30` | API request timeout | High |
| `src/providers/local/backends/ollama_backend.py:245` | `timeout=300` | Generation timeout (5 min) | High |
| `src/providers/local/backends/ollama_backend.py:264` | `timeout=300` | Stream timeout | High |
| `src/providers/local/backends/ollama_backend.py:331` | `timeout=1800` | Model pull timeout (30 min) | High |
| `src/benchmarks/local_model_runner.py:80` | `timeout=2.0` | Monitor thread join | Medium |
| `src/use_cases/fine_tuning/monitoring/structured_logger.py:156` | `timeout=1` | Buffer get timeout | Medium |
| `src/use_cases/evaluation_framework/integrations/pipeline_hooks.py:602` | `timeout=10` | Pipeline hook timeout | Medium |

#### Ports
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/providers/local/backends/ollama_backend.py:30` | `"http://localhost:11434"` | Ollama default URL | **High** |
| `src/use_cases/fine_tuning_studio/backend/api/main.py:659` | `port=8000` | FastAPI server port | High |
| `src/use_cases/fine_tuning_studio/backend/websocket/server.py:404` | `port=8001` | WebSocket server port | High |
| `src/use_cases/fine_tuning_studio/backend/deployment/deployer.py:212` | `port=8000` | Deployment port | High |
| `src/use_cases/fine_tuning/visualization/training_dashboard.py:523` | `server_port=7860` | Gradio dashboard port | Medium |

#### API Endpoints
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/use_cases/fine_tuning_studio/backend/model_card/generator.py:466` | `"https://api.example.com/v1/completions"` | Example API endpoint | Low |
| `src/use_cases/fine_tuning_studio/backend/api/main.py:30` | `["http://localhost:3001", "http://localhost:3000"]` | CORS origins | High |

### 2. Model Parameters

#### Temperature Values
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/providers/base.py:35` | `temperature: float = 0.7` | Provider default | **High** |
| `src/use_cases/fine_tuning/evaluation/benchmarks.py:548` | `temperature=0.1` | Benchmark evaluation | High |
| `src/use_cases/fine_tuning/evaluation/custom_evaluations.py:83` | `temperature=0.7` | Custom evaluation | High |
| `src/use_cases/fine_tuning/evaluation/custom_evaluations.py:155` | `temperature=0.2` | Low variability eval | High |
| `src/use_cases/fine_tuning/evaluation/custom_evaluations.py:228` | `temperature=0.3` | Conservative eval | High |
| `src/use_cases/fine_tuning/evaluation/custom_evaluations.py:298` | `temperature=0.7` | Standard eval | High |
| `src/use_cases/fine_tuning/evaluation/custom_evaluations.py:361` | `temperature=0.3` | Focused eval | High |
| `src/use_cases/fine_tuning/evaluation/suite.py:520` | `temperature=0.7` | Suite default | High |

#### Token Limits
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/providers/base.py:36` | `max_tokens: int = 1000` | Provider default | **High** |
| `src/providers/anthropic.py:165` | `max_tokens=1` | Credential validation | Low |
| `src/providers/openai.py:157` | `max_length=50000` | Prompt validation | High |
| `src/use_cases/fine_tuning/evaluation/custom_evaluations.py:83` | `max_new_tokens=100` | Evaluation generation | High |
| `src/use_cases/fine_tuning/evaluation/custom_evaluations.py:155` | `max_new_tokens=150` | Extended evaluation | High |

#### Context Windows & Sequence Lengths
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/use_cases/visual_analytics/comparison/response_analyzer.py:317-318` | `max_length=512` | Tokenizer truncation | High |
| `src/use_cases/fine_tuning/evaluation/benchmarks.py:355` | `max_length=512` | Benchmark tokenization | High |
| `src/use_cases/fine_tuning/evaluation/benchmarks.py:485` | `max_length=1024` | Extended context | High |
| `src/use_cases/fine_tuning/evaluation/suite.py:343` | `max_length=512` | Suite tokenization | High |

### 3. Provider Configuration Defaults

#### Core Provider Settings (ProviderConfig class)
| Parameter | Default Value | Range/Constraint | Priority |
|-----------|---------------|------------------|----------|
| `temperature` | `0.7` | 0.0 - 2.0 | **High** |
| `max_tokens` | `1000` | > 0 | **High** |
| `top_p` | `1.0` | 0.0 - 1.0 | **High** |
| `top_k` | `None` | Optional int | Medium |
| `timeout` | `30` | > 0 seconds | **High** |
| `max_retries` | `3` | >= 0 | **High** |
| `retry_delay` | `1.0` | > 0 seconds | **High** |

### 4. System Performance Settings

#### Batch Sizes
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/use_cases/fine_tuning/evaluation/benchmarks.py:606` | `batch_size=16` | Benchmark batch | High |
| `src/use_cases/fine_tuning/config/training_config.py:374` | `batch_size=4` | Training config | High |
| `src/use_cases/evaluation_framework/templates/template_library.py:165` | `batch_size=8` | Template evaluation | Medium |
| `src/use_cases/evaluation_framework/templates/template_library.py:201` | `batch_size=16` | Throughput template | Medium |
| `src/use_cases/evaluation_framework/templates/template_library.py:233` | `batch_size=32` | Maximum batch | Medium |

#### Buffer & Chunk Sizes
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/use_cases/local_models/download_helper.py:160` | `chunk_size=8192` | File download chunks | Medium |
| `src/use_cases/fine_tuning/training/distributed_trainer.py:761` | `per_device_train_batch_size=8` | Distributed training | High |
| `src/use_cases/fine_tuning/training/distributed_trainer.py:762` | `per_device_eval_batch_size=16` | Distributed evaluation | High |

#### Sleep/Delay Values
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/providers/base.py:41` | `retry_delay: float = 1.0` | Retry delay default | **High** |
| `src/use_cases/monitoring/dashboard/reports/scheduler.py:302` | `time.sleep(60)` | Report scheduler | High |
| `src/use_cases/fine_tuning/monitoring/integrations.py:994` | `time.sleep(0.1)` | Polling interval | Medium |
| `src/use_cases/alignment/human_loop/interface.py:403` | `await asyncio.sleep(0.5)` | UI polling | Medium |
| `src/use_cases/alignment/human_loop/interface.py:456` | `await asyncio.sleep(1)` | Human feedback wait | Medium |

### 5. Logging Configuration

#### Log Levels
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/use_cases/fine_tuning_studio/backend/versioning/experiment_versioning.py:19` | `level=logging.INFO` | Versioning logs | Medium |
| `src/use_cases/fine_tuning_studio/backend/model_card/generator.py:13` | `level=logging.INFO` | Model card logs | Medium |
| `src/use_cases/fine_tuning_studio/backend/api/main.py:18` | `level=logging.INFO` | API logs | Medium |
| `src/use_cases/fine_tuning_studio/backend/deployment/deployer.py:19` | `level=logging.INFO` | Deployment logs | Medium |

### 6. Validation & Constraints

#### Validation Limits
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/providers/base.py:46-47` | `0 <= temperature <= 2` | Temperature validation | **High** |
| `src/providers/base.py:49` | `max_tokens > 0` | Token validation | **High** |
| `src/providers/base.py:52-53` | `0 <= top_p <= 1` | Top-p validation | **High** |
| `src/providers/base.py:55-56` | `timeout > 0` | Timeout validation | **High** |

### 7. File System & Path Configuration

#### Default Paths
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/use_cases/fine_tuning/cli/fine_tuning_cli.py` | `"./fine_tuned_models"` | Default output directory | Medium |

### 8. Development & Testing Settings

#### Simulation Values
| Location | Value | Context | Priority |
|----------|-------|---------|----------|
| `src/use_cases/fine_tuning_studio/backend/api/main.py:598` | `await asyncio.sleep(2)` | Simulate training | Low |
| `src/use_cases/fine_tuning_studio/backend/api/main.py:614` | `await asyncio.sleep(5)` | Simulate deployment | Low |
| `src/use_cases/fine_tuning_studio/backend/api/main.py:630` | `await asyncio.sleep(3)` | Simulate test execution | Low |

## Priority Classification

### High Priority (Immediate Configuration Extraction)
- **Provider defaults** in ProviderConfig class (7 values)
- **Network timeouts** and connection settings (9 values)
- **Model parameters** for generation (temperature, max_tokens) (15 values)
- **Batch sizes** for training and evaluation (12 values)
- **Server ports** and API endpoints (5 values)

### Medium Priority (Next Phase)
- **Buffer and chunk sizes** (8 values)
- **Polling intervals** and sleep values (10 values)
- **Logging levels** and configuration (8 values)
- **File paths** and directory defaults (5 values)

### Low Priority (Optional)
- **Simulation delays** for development/testing (6 values)
- **Example URLs** and placeholder values (3 values)

## Recommended Configuration Structure

Based on the audit, the following configuration domains should be implemented:

```python
class NetworkConfig(BaseSettings):
    """Network and connectivity configuration"""
    default_timeout: int = 30
    generation_timeout: int = 300
    model_pull_timeout: int = 1800
    ollama_base_url: str = "http://localhost:11434"
    api_request_timeout: int = 30

class ModelConfig(BaseSettings):
    """LLM model configuration defaults"""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    top_k: Optional[int] = None

class SystemConfig(BaseSettings):
    """System performance and resource configuration"""
    batch_size: int = 8
    max_batch_size: int = 32
    chunk_size: int = 8192
    max_prompt_length: int = 50000

class RetryConfig(BaseSettings):
    """Retry and error handling configuration"""
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0

class ServerConfig(BaseSettings):
    """Server and API configuration"""
    api_port: int = 8000
    websocket_port: int = 8001
    dashboard_port: int = 7860
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]

class LoggingConfig(BaseSettings):
    """Logging configuration"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Next Steps

1. **Immediate Actions**:
   - Extract high-priority ProviderConfig defaults
   - Create NetworkConfig for timeout values
   - Implement ModelConfig for generation parameters

2. **Phase 2**:
   - Add SystemConfig for performance settings
   - Implement ServerConfig for port management
   - Create LoggingConfig for logging standardization

3. **Phase 3**:
   - Handle development/testing simulation values
   - Clean up placeholder and example configurations
   - Implement environment-specific profiles (dev, staging, prod)

## Impact Assessment

**Benefits of Configuration Extraction**:
- **Flexibility**: Easy tuning without code changes
- **Environment Separation**: Different settings for dev/staging/prod
- **Validation**: Type-safe configuration with proper validation
- **Documentation**: Self-documenting configuration structure
- **Maintainability**: Centralized configuration management

**Implementation Effort**: Medium-High (due to scope, but systematic approach makes it manageable)

**Risk Level**: Low (backward compatibility can be maintained during migration)
