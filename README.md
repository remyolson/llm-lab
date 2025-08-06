# 🧬 LLM Lab - Advanced Large Language Model Research & Development Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://codecov.io)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue)](docs/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/charliermarsh/ruff)

A comprehensive, production-grade platform for benchmarking, fine-tuning, monitoring, and deploying Large Language Models (LLMs) with enterprise features and local model support.

## 🌟 What's New (v2.0)

### Major Enhancements
- **🛡️ Security Testing Framework**: 500+ attack patterns with red team simulation capabilities
- **🏗️ Fine-Tuning Studio**: Complete web-based platform for model fine-tuning with real-time monitoring
- **📊 Visual Analytics Suite**: Advanced model comparison and response analysis tools
- **🧬 Synthetic Data Platform**: Privacy-preserving data generation across 6 specialized domains
- **📋 Benchmark Creation Tool**: Git-based versioning system for custom benchmark development
- **🔍 Interpretability Suite**: Deep model introspection with attention visualization
- **🎯 Small LLMs Hub**: Optimized support for running models from 70M to 20B parameters locally
- **📚 Model Documentation System**: Automated compliance reporting (EU AI Act, ISO 26000)

## 🚀 Key Features

### Core Platform Capabilities
- **🌐 Multi-Provider Support**: Unified interface for 15+ providers including OpenAI, Anthropic, Google, Azure, Cohere, Mistral, Perplexity, and local models
- **📈 Comprehensive Benchmarking**: Multi-dataset evaluation with statistical analysis, confidence intervals, and performance tracking
- **🎓 Dual Fine-Tuning Systems**: CLI-based pipeline and web-based studio for cloud and local model fine-tuning (LoRA, QLoRA, full fine-tuning)
- **🛡️ Advanced Security**: 500+ attack patterns, red team simulation, and comprehensive vulnerability assessment
- **📊 Real-Time Monitoring**: Production dashboards with Grafana, alerts, regression detection, and SLA tracking
- **💰 Cost Management**: Detailed tracking, forecasting, and optimization recommendations
- **🏢 Enterprise Features**: Multi-tenancy, A/B testing, compliance reporting (EU AI Act), and audit logging

### 13 Specialized Use Cases
1. **Security Testing**: Comprehensive LLM vulnerability assessment with red team capabilities
2. **Synthetic Data**: Privacy-preserving data generation across 6 domains
3. **Model Documentation**: Automated compliance and model card generation
4. **Interpretability**: Deep model introspection and explainability tools
5. **Benchmark Creation**: Custom benchmark development with version control
6. **Fine-Tuning**: Recipe-based training pipeline with cost optimization
7. **Monitoring**: Continuous performance tracking and alerting
8. **Custom Prompts**: Advanced prompt engineering and optimization
9. **Local Models**: Optimized local model management and deployment
10. **Alignment**: Constitutional AI and safety filter implementation
11. **Evaluation Framework**: Extensible evaluation system with plugins
12. **Visual Analytics**: Interactive dashboards for model analysis
13. **Fine-Tuning Studio**: Full-stack web UI with real-time collaboration

### Technical Excellence
- **🖥️ Local Model Optimization**: Support for 70M to 20B parameters, optimized for Apple Silicon
- **🔬 Visual Analytics**: Interactive model response comparison and behavior analysis
- **🏗️ Modern Architecture**: Clean dependency injection, modular design, and 95%+ test coverage
- **📝 Rich Documentation**: Comprehensive guides, API docs, and interactive Jupyter notebooks
- **🔄 CI/CD Pipeline**: Automated testing, documentation building, and performance regression detection

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
│   └── use_cases/             # 13 Production use cases
│       ├── security_testing/  # LLM vulnerability assessment (500+ attacks)
│       ├── synthetic_data/    # Privacy-preserving data generation
│       ├── model_documentation/ # Automated docs & compliance
│       ├── interpretability/  # Model introspection & explainability
│       ├── benchmark_creation/ # Custom benchmark development
│       ├── fine_tuning/       # Recipe-based training pipeline
│       ├── monitoring/        # Continuous performance tracking
│       ├── custom_prompts/    # Prompt engineering tools
│       ├── local_models/      # Local model management
│       ├── alignment/         # Constitutional AI & safety
│       ├── evaluation_framework/ # Extensible evaluation system
│       ├── visual_analytics/  # Interactive analysis dashboard
│       └── fine_tuning_studio/ # Web-based fine-tuning UI
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

## 🎮 13 Production-Ready Use Cases

### 1. 🛡️ Security Testing Framework
**500+ attack patterns for comprehensive LLM vulnerability assessment**

```python
from src.use_cases.security_testing import SecurityScanner

scanner = SecurityScanner()
results = scanner.scan(
    model="gpt-4o-mini",
    attack_categories=["jailbreak", "injection", "extraction"],
    severity_threshold="medium"
)
# Features: Red team simulation, evasion techniques, domain-specific scenarios
```

### 2. 🧬 Synthetic Data Generation
**Privacy-preserving data generation across 6 specialized domains**

```python
from src.use_cases.synthetic_data import DataGenerator

generator = DataGenerator(domain="medical")
synthetic_data = generator.generate(
    num_samples=1000,
    privacy_level="differential",  # PII detection & anonymization
    format="parquet"
)
# Domains: medical, financial, legal, e-commerce, educational, code
```

### 3. 📚 Model Documentation System
**Automated documentation with compliance reporting**

```python
from src.use_cases.model_documentation import ModelCardGenerator

doc_gen = ModelCardGenerator()
model_card = doc_gen.generate(
    model_path="models/custom_model.pt",
    compliance_standards=["EU_AI_Act", "ISO_26000"],
    include_bias_analysis=True
)
# Outputs: Model cards, compliance reports, ethical assessments
```

### 4. 🔍 Interpretability Suite
**Deep model introspection and explainability**

```python
from src.use_cases.interpretability import InterpretabilityAnalyzer

analyzer = InterpretabilityAnalyzer()
explanations = analyzer.analyze(
    model="llama3.2:1b",
    input_text="Why is the sky blue?",
    methods=["attention", "gradients", "activations"]
)
# Features: Attention visualization, feature attribution, interactive dashboards
```

### 5. 📋 Benchmark Creation Tool
**Build and version custom LLM benchmarks**

```python
from src.use_cases.benchmark_creation import BenchmarkBuilder

builder = BenchmarkBuilder()
benchmark = builder.create(
    name="domain_specific_qa",
    generation_method="model_assisted",
    validation_enabled=True,
    version_control=True  # Git-based versioning
)
```

### 6. 🎓 Fine-Tuning Platform
**Complete fine-tuning pipeline with recipe management**

```python
from src.use_cases.fine_tuning import FineTuningPipeline

pipeline = FineTuningPipeline()
model = pipeline.train(
    recipe="recipes/chat_finetuning.yaml",  # YAML-based configs
    cost_estimation=True,
    monitoring=["wandb", "tensorboard"],
    optimization="qlora"  # Memory optimization
)
```

### 7. 📊 Continuous Monitoring
**Production monitoring with regression detection**

```python
from src.use_cases.monitoring import MonitoringDashboard

monitor = MonitoringDashboard()
monitor.track(
    models=["production_v1", "production_v2"],
    metrics=["latency", "accuracy", "cost"],
    alerts={"regression_threshold": 0.05},
    channels=["slack", "email"]
)
```

### 8. 🎯 Custom Prompt Engineering
**Advanced prompt management and optimization**

```python
from src.use_cases.custom_prompts import PromptEngine

engine = PromptEngine()
optimized_prompt = engine.optimize(
    base_prompt="Summarize this text",
    optimization_goal="accuracy",
    test_dataset="validation_set.jsonl"
)
```

### 9. 🖥️ Local Model Management
**Run models locally with automatic optimization**

```python
from src.use_cases.local_models import LocalModelProvider

provider = LocalModelProvider()
provider.setup(
    model="llama3.2:1b",
    quantization="Q4_K_M",  # Automatic quantization
    gpu_layers="auto"  # Automatic GPU detection
)
```

### 10. 🤝 AI Alignment Tools
**Constitutional AI and safety measures**

```python
from src.use_cases.alignment import AlignmentPipeline

pipeline = AlignmentPipeline()
safe_response = pipeline.generate(
    prompt=user_input,
    constitutional_rules="rules.yaml",
    human_feedback_enabled=True,
    safety_filters=["toxicity", "bias", "pii"]
)
```

### 11. 📈 Evaluation Framework
**Comprehensive model evaluation system**

```python
from src.use_cases.evaluation_framework import EvaluationSuite

suite = EvaluationSuite()
results = suite.evaluate(
    models=["gpt-4o", "claude-3.5"],
    benchmarks=["truthfulness", "reasoning", "safety"],
    include_cost_analysis=True,
    export_format="html"
)
```

### 12. 📊 Visual Analytics Dashboard
**Interactive visualization for model analysis**

```python
from src.use_cases.visual_analytics import AnalyticsDashboard

dashboard = AnalyticsDashboard()
dashboard.launch(
    port=8080,
    features=["response_comparison", "behavior_analysis",
              "regression_detection", "task_evaluation"]
)
```

### 13. 🏗️ Fine-Tuning Studio (Web UI)
**Full-stack web application for interactive fine-tuning**

```bash
# Launch the complete web-based fine-tuning environment
cd src/use_cases/fine_tuning_studio
python backend/api/main.py  # FastAPI backend

# In another terminal
cd frontend && npm install && npm run dev  # Next.js frontend
# Features: Real-time collaboration, A/B testing, dataset explorer, live preview
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

### Quick Links
- **[Quick Start Guide](docs/getting_started/README.md)** - Get up and running quickly
- **[API Reference](docs/api/)** - Complete API documentation
- **[Architecture Guide](docs/architecture/)** - System design and patterns
- **[Configuration Reference](docs/CONFIGURATION.md)** - All configuration options
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### Use Case Guides
- **[Security Testing Guide](docs/guides/USE_CASE_1_HOW_TO.md)** - LLM vulnerability assessment
- **[Synthetic Data Guide](docs/guides/USE_CASE_2_HOW_TO.md)** - Privacy-preserving data generation
- **[Model Documentation Guide](docs/guides/USE_CASE_3_HOW_TO.md)** - Automated documentation & compliance
- **[Interpretability Guide](docs/guides/USE_CASE_4_HOW_TO.md)** - Model introspection tools
- **[Benchmark Creation Guide](docs/guides/USE_CASE_5_HOW_TO.md)** - Custom benchmark development
- **[Fine-Tuning Guide](docs/guides/USE_CASE_6_HOW_TO.md)** - Recipe-based training pipeline
- **[Monitoring Guide](docs/guides/USE_CASE_7_HOW_TO.md)** - Performance tracking & alerts
- **[Custom Prompts Guide](docs/guides/USE_CASE_8_HOW_TO.md)** - Prompt engineering tools
- **[Local Models Guide](docs/guides/USE_CASE_9_HOW_TO.md)** - Running models locally
- **[Alignment Guide](docs/guides/USE_CASE_10_HOW_TO.md)** - Safety and alignment tools
- **[Evaluation Framework Guide](docs/guides/USE_CASE_11_HOW_TO.md)** - Model evaluation system
- **[Visual Analytics Guide](docs/guides/USE_CASE_12_HOW_TO.md)** - Interactive analysis tools
- **[Fine-Tuning Studio Guide](docs/guides/USE_CASE_13_HOW_TO.md)** - Web-based fine-tuning UI

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

**R3MY 0LS0N**

[Website](https://llm-lab.dev) • [Documentation](https://docs.llm-lab.dev) • [Blog](https://blog.llm-lab.dev)

</div>
