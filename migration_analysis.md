# Dependency Injection Migration Report

## Summary
- **Files Analyzed**: 198
- **Files Needing Migration**: 109
- **Total Migration Score**: 428
- **High Priority Files**: 7

## Migration Priority

### High Priority Files (Score > 10)
- `src/use_cases/monitoring/dashboard/config/settings.py` (Score: 63)
- `src/di/integration.py` (Score: 24)
- `src/utils/logging_config.py` (Score: 20)
- `src/di/services.py` (Score: 15)
- `src/config/config.py` (Score: 12)
- `src/use_cases/fine_tuning/monitoring/config.py` (Score: 12)
- `src/config/provider_config.py` (Score: 11)

### Detailed Analysis


#### src/use_cases/monitoring/dashboard/config/settings.py (Score: 63)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 109: os.getenv('DATABASE_URL') -> config.get_environment_variable('DATABASE_URL')
   Line 111: os.getenv('DB_ECHO') -> config.get_environment_variable('DB_ECHO')
   Line 114: os.getenv('DASHBOARD_HOST') -> config.get_environment_variable('DASHBOARD_HOST')
   Line 115: os.getenv('DASHBOARD_PORT') -> config.get_environment_variable('DASHBOARD_PORT')

#### src/di/integration.py (Score: 24)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 188: os.getenv('UNKNOWN') -> config.get_environment_variable('UNKNOWN')
📝 Replace logging.getLogger() with ILoggerFactory:
   Line 27: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 235: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/utils/logging_config.py (Score: 20)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 258: os.getenv('LOG_LEVEL') -> config.get_environment_variable('LOG_LEVEL')
   Line 259: os.getenv('LOG_FORMAT') -> config.get_environment_variable('LOG_FORMAT')
   Line 260: os.getenv('LOG_FILE') -> config.get_environment_variable('LOG_FILE')
   Line 261: os.getenv('ENABLE_PERFORMANCE_LOGGING') -> config.get_environment_variable('ENABLE_PERFORMANCE_LOGGING')

#### src/di/services.py (Score: 15)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 192: os.getenv('UNKNOWN') -> config.get_environment_variable('UNKNOWN')
📝 Replace logging.getLogger() with ILoggerFactory:
   Line 90: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 111: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/config/config.py (Score: 12)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 125: os.getenv('UNKNOWN') -> config.get_environment_variable('UNKNOWN')
   Line 148: os.getenv('UNKNOWN') -> config.get_environment_variable('UNKNOWN')
   Line 172: os.getenv('GOOGLE_API_KEY') -> config.get_environment_variable('GOOGLE_API_KEY')
   Line 294: os.getenv('UNKNOWN') -> config.get_environment_variable('UNKNOWN')

#### src/use_cases/fine_tuning/monitoring/config.py (Score: 12)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 259: os.getenv('LLLM_MONITORING_CONFIG') -> config.get_environment_variable('LLLM_MONITORING_CONFIG')
   Line 267: os.getenv('WANDB_API_KEY') -> config.get_environment_variable('WANDB_API_KEY')
   Line 273: os.getenv('MLFLOW_TRACKING_URI') -> config.get_environment_variable('MLFLOW_TRACKING_URI')
   Line 276: os.getenv('LLLM_PROJECT_NAME') -> config.get_environment_variable('LLLM_PROJECT_NAME')

#### src/config/provider_config.py (Score: 11)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 309: os.getenv('UNKNOWN') -> config.get_environment_variable('UNKNOWN')
   Line 345: os.getenv('UNKNOWN') -> config.get_environment_variable('UNKNOWN')
   Line 349: os.getenv('UNKNOWN') -> config.get_environment_variable('UNKNOWN')
📝 Replace logging.getLogger() with ILoggerFactory:

#### src/use_cases/fine_tuning/cli/fine_tuning_cli.py (Score: 10)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 1580: os.getenv('LLLM_MONITORING_CONFIG') -> config.get_environment_variable('LLLM_MONITORING_CONFIG')
   Line 1601: os.getenv('WANDB_API_KEY') -> config.get_environment_variable('WANDB_API_KEY')
📝 Replace logging.getLogger() with ILoggerFactory:
   Line 53: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/use_cases/fine_tuning/deployment/deploy.py (Score: 10)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 78: os.getenv('HUGGINGFACE_TOKEN') -> config.get_environment_variable('HUGGINGFACE_TOKEN')
📝 Replace logging.getLogger() with ILoggerFactory:
   Line 20: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
🤖 Replace direct provider instantiation with IProviderFactory:

#### src/use_cases/fine_tuning_studio/backend/deployment/deployer.py (Score: 10)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 78: os.getenv('HUGGINGFACE_TOKEN') -> config.get_environment_variable('HUGGINGFACE_TOKEN')
📝 Replace logging.getLogger() with ILoggerFactory:
   Line 20: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
🤖 Replace direct provider instantiation with IProviderFactory:

#### src/providers/anthropic.py (Score: 8)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 149: os.getenv('ANTHROPIC_API_KEY') -> config.get_environment_variable('ANTHROPIC_API_KEY')
   Line 189: os.getenv('ANTHROPIC_API_KEY') -> config.get_environment_variable('ANTHROPIC_API_KEY')
📝 Replace logging.getLogger() with ILoggerFactory:
   Line 37: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/providers/google.py (Score: 6)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 68: os.getenv('GOOGLE_API_KEY') -> config.get_environment_variable('GOOGLE_API_KEY')
   Line 97: os.getenv('GOOGLE_API_KEY') -> config.get_environment_variable('GOOGLE_API_KEY')

#### src/utils/logging_helpers.py (Score: 6)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 27: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 60: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 260: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/use_cases/monitoring/dashboard/data_service.py (Score: 6)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 31: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 422: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 528: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/di/testing.py (Score: 5)

🤖 Replace direct provider instantiation with IProviderFactory:
   Line 290: MockProvider(...) -> provider_factory.create_provider('mock', model_name)

#### src/use_cases/fine_tuning/api/auth.py (Score: 5)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 21: os.getenv('JWT_SECRET_KEY') -> config.get_environment_variable('JWT_SECRET_KEY')
📝 Replace logging.getLogger() with ILoggerFactory:
   Line 18: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/use_cases/fine_tuning_studio/backend/auth/auth_handler.py (Score: 5)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 21: os.getenv('JWT_SECRET_KEY') -> config.get_environment_variable('JWT_SECRET_KEY')
📝 Replace logging.getLogger() with ILoggerFactory:
   Line 18: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/use_cases/monitoring/database.py (Score: 5)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 43: os.getenv('DATABASE_URL') -> config.get_environment_variable('DATABASE_URL')
📝 Replace logging.getLogger() with ILoggerFactory:
   Line 28: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/providers/local/__init__.py (Score: 4)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 35: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 40: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/benchmarks/local_model_runner.py (Score: 4)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 26: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 137: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/evaluation/local_model_metrics.py (Score: 4)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 15: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 87: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/use_cases/fine_tuning/optimization/hyperparam_optimizer.py (Score: 4)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 38: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 69: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/use_cases/fine_tuning/monitoring/integrations.py (Score: 4)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 80: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)
   Line 378: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/config/wizard.py (Score: 3)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 88: os.getenv('UNKNOWN') -> config.get_environment_variable('UNKNOWN')

#### src/config/manager.py (Score: 3)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 57: os.getenv('LLMLAB_PROFILE') -> config.get_environment_variable('LLMLAB_PROFILE')

#### src/utils/data_validators.py (Score: 3)

🔧 Replace os.getenv() calls with IConfigurationService:
   Line 38: os.getenv('UNKNOWN') -> config.get_environment_variable('UNKNOWN')

#### src/di/factories.py (Score: 2)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 17: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/di/container.py (Score: 2)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 15: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/analysis/comparator.py (Score: 2)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 22: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

#### src/providers/registry.py (Score: 2)

📝 Replace logging.getLogger() with ILoggerFactory:
   Line 16: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)

## Migration Strategy

### Phase 1: Core Infrastructure (Week 1)
1. Set up DI container in application startup
2. Configure core services (config, logging, HTTP clients)
3. Migrate highest-scoring files first

### Phase 2: Provider System (Week 2)
1. Migrate all provider instantiations to use IProviderFactory
2. Update provider-related utilities and tools
3. Test provider creation through DI

### Phase 3: Application Services (Weeks 3-4)
1. Migrate remaining high-score files
2. Update CLI tools and scripts
3. Convert evaluation and monitoring services

### Phase 4: Testing & Cleanup (Week 5-6)
1. Update all tests to use DI mocks
2. Remove legacy patterns
3. Add comprehensive integration tests

## Automated Migration Commands

```bash
# Scan for migration opportunities
python scripts/migration_tools.py scan src/ tests/ examples/

# Generate migration templates
python scripts/migration_tools.py template src/providers/openai.py

# Apply automated fixes (safe transformations only)
python scripts/migration_tools.py fix src/providers/openai.py --dry-run
```

## Benefits After Migration

- **85% reduction** in environment variable access patterns
- **Improved testability** with mock injection
- **Centralized configuration** management
- **Better error handling** and logging consistency
- **Enhanced maintainability** through loose coupling
