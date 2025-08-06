# 🧬 LLM Lab - Advanced Large Language Model Research & Development Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://codecov.io)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue)](docs/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/charliermarsh/ruff)

A comprehensive, production-grade platform for benchmarking, fine-tuning, monitoring, and deploying Large Language Models (LLMs) with enterprise features and local model support.

## 🌟 What's New (v2.0)

### Major Enhancements
- **🎯 Small LLMs Hub**: Optimized support for running models from 70M to 20B parameters locally on MacBook Pro
- **🏗️ Fine-Tuning Studio**: Complete web-based platform for model fine-tuning with real-time monitoring
- **📊 Visual Analytics Suite**: Advanced model comparison and response analysis tools
- **💉 Dependency Injection**: Enterprise-grade DI system for modular, testable code
- **📚 Documentation System**: Full Sphinx documentation with CI/CD and interactive examples
- **🎭 Type System**: Comprehensive type annotations with protocols and generics
- **🔄 Restructured Architecture**: Flattened directory structure for better maintainability

## 🚀 Key Features

### Core Capabilities
- **🌐 Multi-Provider Support**: Unified interface for 15+ providers including OpenAI, Anthropic, Google, Azure, Cohere, Mistral, Perplexity, and local models
- **📈 Comprehensive Benchmarking**: Multi-dataset evaluation with statistical analysis, confidence intervals, and performance tracking
- **🎓 Fine-Tuning Platform**: Web-based studio for cloud and local model fine-tuning (LoRA, QLoRA, full fine-tuning)
- **🛡️ Advanced Alignment**: Constitutional AI, multi-layer safety filters, and preference learning (RLHF/DPO)
- **📊 Real-Time Monitoring**: Production dashboards with Grafana, alerts, and SLA tracking
- **💰 Cost Management**: Detailed tracking, forecasting, and optimization recommendations
- **🏢 Enterprise Features**: Multi-tenancy, A/B testing, compliance reporting, and audit logging

### New Features
- **🖥️ Local Model Excellence**: Optimized for Apple Silicon with Metal acceleration
- **🔬 Visual Analytics**: Interactive model response comparison and analysis
- **🏗️ Modern Architecture**: Clean dependency injection and modular design
- **📝 Rich Documentation**: Comprehensive guides, API docs, and interactive notebooks
- **🧪 95%+ Test Coverage**: Extensive unit, integration, and end-to-end testing

## 📋 Prerequisites

- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum (16GB+ recommended for local models)
- **Storage**: 10GB free space for models and datasets
- **API Keys**: For cloud providers (OpenAI, Anthropic, Google, etc.)
- **Optional**:
  - CUDA GPU for accelerated local training
  - Apple Silicon Mac for Metal acceleration
  - Docker for containerized deployments

## 🛠️ Installation

### Quick Start (Recommended)

```bash
# Clone the repository
git clone https://github.com/remyolson/llm-lab.git
cd llm-lab

# Run the automated setup script
./scripts/setup.sh

# Or use the configuration wizard
python -m src.config.wizard
```

### Manual Installation

1. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install as editable package
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Download models and datasets**
   ```bash
   # Download everything (models + datasets)
   python download_assets.py --all

   # Or selectively:
   python models/small-llms/download_small_models.py  # Small local models
   python download_assets.py --datasets               # Benchmark datasets
   ```

## 🎯 Quick Start Guide

### 1. Running Small Models Locally

```bash
# Quick setup for local models (macOS with Ollama)
./models/small-llms/quick_setup.sh

# Run interactive demo
python models/small-llms/run_small_model_demo.py

# Benchmark local models
python models/small-llms/system_assessment.py
```

### 2. Multi-Provider Benchmarking

```bash
# Compare models across providers
python src/use_cases/multi_provider_benchmarking.py \
    --providers openai anthropic google local \
    --models gpt-4o claude-3-5-sonnet gemini-1.5-pro llama3.2:1b \
    --datasets truthfulness gsm8k \
    --output results/benchmark_comparison.html
```

### 3. Fine-Tuning Studio

```bash
# Launch the web-based fine-tuning studio
cd src/use_cases/fine_tuning
python -m api.main  # Start backend API

# In another terminal
cd web && npm install && npm run dev  # Start frontend
# Open http://localhost:3000
```

### 4. Visual Analytics Dashboard

```bash
# Launch the monitoring dashboard
python src/use_cases/monitoring/dashboard.py --port 8080

# Or use Streamlit interface
streamlit run src/use_cases/visual_analytics/app.py
```

## 🏗️ Architecture

### Project Structure

```
llm-lab/
├── src/                        # Core source code
│   ├── providers/              # LLM provider implementations
│   │   ├── base.py            # Base provider interface
│   │   ├── openai.py          # OpenAI implementation
│   │   ├── anthropic.py       # Anthropic implementation
│   │   ├── google.py          # Google implementation
│   │   └── local/             # Local model support
│   │       ├── unified_provider.py
│   │       └── backends/      # Ollama, Transformers, llama.cpp
│   ├── benchmarks/            # Benchmarking system
│   │   ├── integrated_runner.py
│   │   └── local_model_runner.py
│   ├── evaluation/            # Evaluation framework
│   │   ├── improved_evaluation.py
│   │   └── local_model_metrics.py
│   ├── config/                # Configuration management
│   │   ├── settings.py        # Pydantic settings
│   │   ├── wizard.py          # Interactive setup
│   │   └── manager.py         # Config management
│   ├── di/                    # Dependency injection
│   │   ├── container.py       # DI container
│   │   ├── factories.py       # Object factories
│   │   └── services.py        # Service layer
│   ├── types/                 # Type system
│   │   ├── protocols.py       # Protocol definitions
│   │   ├── generics.py        # Generic types
│   │   └── core.py            # Core types
│   └── use_cases/             # Production use cases
│       ├── fine_tuning/       # Fine-tuning platform
│       │   ├── api/           # FastAPI backend
│       │   ├── web/           # Next.js frontend
│       │   └── deployment/    # Deployment configs
│       ├── visual_analytics/  # Visual analysis tools
│       ├── monitoring/        # Monitoring system
│       └── alignment/         # Safety alignment
├── models/                    # Model storage
│   └── small-llms/           # Local model zoo
│       ├── pythia-70m/       # Tiny models (70-160M)
│       ├── smollm-135m/      # Small models (135-360M)
│       ├── qwen-0.5b/        # Medium models (0.5-1B)
│       └── assessments/      # Performance reports
├── datasets/                  # Dataset storage
│   ├── benchmarking/         # Evaluation datasets
│   └── fine-tuning/          # Training datasets
├── docs/                      # Documentation
│   ├── conf.py               # Sphinx configuration
│   ├── api/                  # API documentation
│   ├── guides/               # User guides
│   └── notebooks/            # Jupyter tutorials
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── e2e/                  # End-to-end tests
└── scripts/                   # Utility scripts
    ├── setup.sh              # Automated setup
    └── verification/         # Verification tools
```

### Technology Stack

- **Backend**: FastAPI, Pydantic, SQLAlchemy
- **Frontend**: Next.js, React, TypeScript, TailwindCSS
- **ML/AI**: Transformers, PyTorch, Ollama, llama.cpp
- **Testing**: Pytest, Coverage.py, Mypy
- **Documentation**: Sphinx, MkDocs, Jupyter
- **Monitoring**: Grafana, Prometheus, OpenTelemetry
- **Deployment**: Docker, Kubernetes, GitHub Actions

## 🎮 Use Cases

### 1. Local Model Development

Perfect for development and experimentation without API costs:

```python
from src.providers.local import UnifiedLocalProvider

# Initialize local provider
provider = UnifiedLocalProvider(
    backend="ollama",  # or "transformers", "llamacpp"
    model_name="llama3.2:1b"
)

# Generate text
response = provider.generate("Explain quantum computing in simple terms")
print(response)
```

### 2. Fine-Tuning Studio

Web-based interface for model fine-tuning:

```python
# Use the API programmatically
from src.use_cases.fine_tuning.api import FineTuningClient

client = FineTuningClient()
job = client.create_fine_tuning_job(
    base_model="gpt-3.5-turbo",
    training_data="data/training.jsonl",
    hyperparameters={"epochs": 3, "learning_rate": 2e-5}
)
```

### 3. Visual Model Comparison

Compare responses across models visually:

```python
from src.use_cases.visual_analytics import ResponseAnalyzer

analyzer = ResponseAnalyzer()
comparison = analyzer.compare_models(
    prompt="What is consciousness?",
    models=["gpt-4o", "claude-3.5", "llama3.2:1b"],
    visualize=True
)
```

### 4. Cost-Optimized Deployment

Find the best model for your budget:

```python
from src.use_cases.cost_optimization import CostOptimizer

optimizer = CostOptimizer()
recommendation = optimizer.recommend(
    required_accuracy=0.9,
    max_latency_ms=500,
    monthly_budget=1000,
    expected_requests=100000
)
```

### 5. A/B Testing

Test models in production:

```python
from src.use_cases.ab_testing import ABTestRunner

runner = ABTestRunner()
results = runner.run_test(
    control_model="gpt-3.5-turbo",
    treatment_model="llama3.2:1b",
    test_duration_hours=168,  # 1 week
    traffic_split=0.5
)
```

### 6. Safety Alignment

Implement constitutional AI and safety filters:

```python
from src.use_cases.alignment import SafetyPipeline

pipeline = SafetyPipeline()
safe_response = pipeline.generate(
    prompt=user_input,
    safety_level="strict",
    constitutional_rules="rules.yaml"
)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit -v           # Unit tests only
pytest tests/integration -v    # Integration tests
pytest tests/e2e -v            # End-to-end tests

# Run with markers
pytest -m "not slow"           # Skip slow tests
pytest -m "local"              # Local model tests only

# Type checking
mypy src/

# Code quality
ruff check .
ruff format .
```

## 📊 Benchmarking Results

### Small Model Performance (M3 MacBook Pro)

| Model | Size | Speed (tok/s) | Quality | Memory |
|-------|------|---------------|---------|--------|
| Pythia-70M | 280MB | 50-70 | Basic | 0.5GB |
| SmolLM-135M | 270MB | 30-50 | Good | 0.8GB |
| Qwen-0.5B | 1GB | 15-30 | Better | 1.5GB |
| Llama3.2-1B | 700MB | 10-20 | Excellent | 2GB |

### Provider Comparison

| Provider | Models | Cost | Latency | Rate Limits |
|----------|--------|------|---------|-------------|
| OpenAI | GPT-4, GPT-3.5 | $$$ | Low | 10K RPM |
| Anthropic | Claude 3.5 | $$$ | Low | 4K RPM |
| Google | Gemini 1.5 | $$ | Medium | 60 RPM |
| Local | Various | Free | Varies | Unlimited |

## 🚀 Deployment

### Docker

```bash
# Build and run with Docker
docker build -t llm-lab .
docker run -p 8080:8080 --env-file .env llm-lab

# Using docker-compose
docker-compose up -d
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/

# Scale deployments
kubectl scale deployment llm-lab --replicas=3
```

### GitHub Actions CI/CD

The project includes comprehensive CI/CD pipelines:
- Automated testing on all PRs
- Documentation building and deployment
- Docker image building and pushing
- Performance regression detection

## 📚 Documentation

- **[Quick Start Guide](docs/getting_started/README.md)** - Get up and running quickly
- **[API Reference](docs/api/)** - Complete API documentation
- **[Architecture Guide](docs/architecture/)** - System design and patterns
- **[Use Case Guides](docs/guides/)** - Detailed tutorials for each use case
- **[Configuration Reference](docs/CONFIGURATION.md)** - All configuration options
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### Interactive Documentation

```bash
# Launch documentation server
cd docs && make livehtml
# Open http://localhost:8000

# Run Jupyter tutorials
jupyter notebook docs/notebooks/
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run code quality checks
make lint
make format
make type-check
```

### Areas for Contribution

- 🌟 New provider integrations
- 📊 Additional benchmark datasets
- 🔧 Performance optimizations
- 📚 Documentation improvements
- 🌍 Internationalization
- 🧪 Test coverage expansion
- 🎨 UI/UX improvements

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- OpenAI, Anthropic, Google, and all LLM providers for their APIs
- The Ollama team for local model runtime
- Hugging Face for model hosting and tools
- The open-source community for invaluable contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/remyolson/llm-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/remyolson/llm-lab/discussions)
- **Documentation**: [Full Docs](https://remyolson.github.io/llm-lab)

## 🏆 Project Status

- **Version**: 2.0.0
- **Status**: Active Development
- **Coverage**: 95%+
- **Python**: 3.9+
- **Last Updated**: December 2024

---

<div align="center">

**Built with ❤️ by the LLM Lab Team**

[Website](https://llm-lab.dev) • [Documentation](https://docs.llm-lab.dev) • [Blog](https://blog.llm-lab.dev)

</div>
