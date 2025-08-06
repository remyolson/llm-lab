# LLM Lab Dependency Matrix

This document provides a comprehensive overview of dependencies required for different features and providers in LLM Lab.

## Installation Guide

### Basic Installation (Core Only)
```bash
pip install .
```
This installs only the core dependencies needed for basic LLM provider interactions.

### Installation with Optional Features
```bash
# For development
pip install ".[dev]"

# For GPU support
pip install ".[gpu]"

# For fine-tuning capabilities
pip install ".[fine-tuning]"

# For production deployment
pip install ".[production]"

# For research and experimentation
pip install ".[research]"

# Install everything
pip install ".[all]"
```

## Core Dependencies

These are always installed and provide basic functionality:

| Package | Min Version | Purpose |
|---------|-------------|---------|
| `openai` | ≥1.0.0 | OpenAI API client |
| `anthropic` | ≥0.18.0 | Anthropic Claude API client |
| `google-generativeai` | ≥0.3.0 | Google Gemini API client |
| `requests` | ≥2.28.0 | HTTP client library |
| `pydantic` | ≥2.0.0 | Data validation |
| `python-dotenv` | ≥1.0.0 | Environment variable management |
| `click` | ≥8.1.0 | Command-line interface |
| `pyyaml` | ≥6.0.0 | YAML configuration parsing |
| `tabulate` | ≥0.9.0 | Table formatting |

## Optional Dependency Groups

### Testing (`test`)
For running the test suite:
- `pytest` (≥7.0.0) - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- `pytest-mock` - Mocking support
- `pytest-asyncio` - Async test support
- `coverage[toml]` - Coverage analysis

### Development (`dev`)
Full development environment:
- All `test` dependencies
- `ruff`, `black`, `isort` - Code formatting
- `mypy`, `flake8` - Type checking and linting
- `bandit`, `safety`, `semgrep` - Security scanning
- `sphinx` - Documentation generation
- `pre-commit` - Git hooks
- `ipython`, `jupyter` - Interactive development

### Benchmarking (`benchmarks`)
Performance analysis and visualization:
- `matplotlib`, `seaborn` - Plotting
- `pandas`, `numpy`, `scipy` - Data analysis
- `psutil` - System monitoring
- `memory-profiler` - Memory profiling

### GPU Support (`gpu`)
For GPU-accelerated operations:
- `torch` (≥2.0.0) - PyTorch framework
- `accelerate` - Hugging Face Accelerate
- `bitsandbytes` - 8-bit quantization
- `xformers` - Memory-efficient transformers

### Local Models (`local-models`)
Running models locally without GPU:
- `llama-cpp-python` - GGUF model support
- `psutil` - Resource monitoring

### Fine-tuning (`fine-tuning`)
Model fine-tuning capabilities:
- `transformers` (≥4.35.0) - Hugging Face Transformers
- `peft` - Parameter-efficient fine-tuning
- `datasets` - Dataset handling
- `accelerate` - Training acceleration
- `wandb`, `tensorboard` - Experiment tracking
- `evaluate` - Model evaluation

### Monitoring (`monitoring`)
Observability and monitoring:
- `fastapi`, `uvicorn` - API server
- `sqlalchemy`, `alembic` - Database ORM
- `prometheus-client` - Metrics export
- `opentelemetry-*` - Distributed tracing
- `apscheduler` - Job scheduling

### Database (`database`)
Database connectivity:
- `sqlalchemy`, `alembic` - ORM and migrations
- `psycopg2-binary` - PostgreSQL
- `pymysql` - MySQL/MariaDB
- `redis` - Redis caching
- `motor` - MongoDB async

### Dashboard (`dashboard`)
Web dashboard interface:
- `flask` and extensions - Web framework
- `python-socketio` - Real-time updates
- `bcrypt`, `pyjwt` - Authentication
- `gunicorn`, `eventlet` - Production server

### Visualization (`visualization`)
Advanced visualization and reporting:
- `matplotlib`, `seaborn` - Statistical plots
- `plotly` - Interactive charts
- `reportlab`, `weasyprint` - PDF generation
- `pillow` - Image processing

### Extra Providers (`providers-extra`)
Additional LLM provider support:
- `groq` - Groq API
- `cohere` - Cohere API
- `huggingface-hub` - Hugging Face models
- `replicate` - Replicate API
- `mistralai` - Mistral API

### Fine-tuning Studio (`fine-tuning-studio`)
Complete fine-tuning web interface:
- `fastapi`, `uvicorn` - Backend API
- `websockets` - Real-time communication
- `jose`, `passlib` - JWT authentication
- `gitpython`, `dvc` - Version control
- `vllm` - Model serving

## Provider-Specific Requirements

| Provider | Required Dependencies | Optional Dependencies |
|----------|----------------------|----------------------|
| **OpenAI** | `openai` | - |
| **Anthropic** | `anthropic` | - |
| **Google Gemini** | `google-generativeai` | - |
| **Groq** | `groq` (via `providers-extra`) | - |
| **Cohere** | `cohere` (via `providers-extra`) | - |
| **Mistral** | `mistralai` (via `providers-extra`) | - |
| **Hugging Face** | `huggingface-hub` (via `providers-extra`) | `transformers` (for local inference) |
| **Replicate** | `replicate` (via `providers-extra`) | - |
| **Local Models** | - | `local-models` or `gpu` groups |

## Feature-Specific Requirements

| Feature | Required Groups | Notes |
|---------|----------------|-------|
| **Basic LLM API Calls** | Core only | Minimal installation |
| **Running Tests** | `test` | For development and CI/CD |
| **Development** | `dev` | Full development environment |
| **Performance Testing** | `benchmarks` | Includes visualization tools |
| **Local Model Inference** | `local-models` or `gpu` | GPU recommended for speed |
| **Fine-tuning** | `fine-tuning`, `gpu` | GPU strongly recommended |
| **Production API** | `production` | Includes monitoring and database |
| **Research** | `research` | GPU, fine-tuning, and analysis tools |
| **Web Dashboard** | `dashboard`, `database` | Full web interface |
| **Monitoring** | `monitoring` | Prometheus, OpenTelemetry |

## Deployment Configurations

### Minimal API Server
```bash
pip install ".[production]"
```
Includes: database, monitoring, dashboard

### Research Environment
```bash
pip install ".[research]"
```
Includes: GPU support, fine-tuning, benchmarks, visualization

### Full Development Setup
```bash
pip install ".[dev]"
```
Includes: all development and testing tools

### Complete Installation
```bash
pip install ".[all]"
```
Includes: everything

## Version Compatibility

### Python Version Support
- **Minimum**: Python 3.9
- **Recommended**: Python 3.11+
- **Tested**: Python 3.9, 3.10, 3.11, 3.12

### Key Dependency Versions

| Dependency | Minimum | Recommended | Notes |
|------------|---------|-------------|-------|
| `torch` | 2.0.0 | Latest stable | For GPU support |
| `transformers` | 4.35.0 | Latest stable | For fine-tuning |
| `fastapi` | 0.100.0 | Latest stable | For API server |
| `sqlalchemy` | 1.4.0 | 2.0+ | For database |

## Troubleshooting

### Common Installation Issues

1. **GPU Dependencies**: If you encounter CUDA errors, ensure your PyTorch installation matches your CUDA version:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Memory Issues**: For large models, you may need the `bitsandbytes` library:
   ```bash
   pip install ".[gpu]"
   ```

3. **Missing System Dependencies**: Some packages require system libraries:
   - `psycopg2-binary`: May need PostgreSQL development headers
   - `weasyprint`: Requires Cairo and Pango libraries

4. **Conflicting Versions**: Use a virtual environment to avoid conflicts:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install ".[dev]"
   ```

## Updating Dependencies

To update all dependencies to their latest compatible versions:
```bash
pip install --upgrade ".[all]"
```

To update specific groups:
```bash
pip install --upgrade ".[gpu,fine-tuning]"
```

## Contributing

When adding new dependencies:
1. Add to the appropriate group in `pyproject.toml`
2. Update this matrix document
3. Test installation in a clean environment
4. Document any system requirements
