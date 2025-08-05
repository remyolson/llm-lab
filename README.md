# LLM Lab - Comprehensive Large Language Model Research Platform

A professional-grade framework for benchmarking, fine-tuning, monitoring, and aligning Large Language Models (LLMs) across multiple providers with enterprise features.

## 🚀 Overview

LLM Lab is a comprehensive platform designed for organizations and researchers working with Large Language Models. It provides production-ready tools for evaluation, optimization, safety alignment, and continuous monitoring of LLM deployments.

### Key Features

- **Multi-Provider Support**: Unified interface for OpenAI, Anthropic, Google, Azure, Cohere, Mistral, and local models
- **Comprehensive Benchmarking**: Multi-dataset evaluation with statistical analysis and confidence intervals
- **Fine-Tuning Pipelines**: Support for both cloud-based and local model fine-tuning (LoRA, QLoRA, full)
- **Advanced Alignment**: Constitutional AI, multi-layer safety filters, and preference learning (RLHF)
- **Real-Time Monitoring**: Production monitoring with dashboards, alerts, and SLA tracking
- **Cost Management**: Detailed cost tracking, forecasting, and optimization recommendations
- **Enterprise Features**: Multi-tenancy, custom evaluators, A/B testing, and compliance reporting
- **Production Ready**: 95%+ test coverage, comprehensive error handling, and extensive documentation

## 📋 Prerequisites

- Python 3.9 or higher
- API keys for desired LLM providers (OpenAI, Anthropic, Google, etc.)
- pip package manager
- (Optional) CUDA-capable GPU for local model fine-tuning
- (Optional) Docker for containerized deployments

## 🛠️ Installation

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lllm-lab.git
   cd lllm-lab
   ```

2. **Create and activate a virtual environment**
   
   On macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   On Windows:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models and datasets**
   ```bash
   # Download everything (recommended for first-time setup)
   python download_assets.py --all

   # Or download selectively:
   python download_assets.py --models      # Only models
   python download_assets.py --datasets    # Only datasets
   python download_assets.py --list        # See what's available
   ```

### 📥 Asset Management

This repository uses a download-based approach for large files (models and datasets) instead of Git LFS to ensure:
- No bandwidth costs for contributors
- Easy setup for forks and clones
- Better performance for large files

**Available download options:**
- `--all`: Download everything (~2.5GB)
- `--models`: Download all models (~2.4GB)
- `--datasets`: Download all datasets (~10MB)
- `--model <name>`: Download specific model (e.g., `qwen-0.5b`)
- `--dataset <name>`: Download specific dataset (e.g., `truthfulqa-full`)
- `--verify`: Check that all downloaded files are valid

**Note:** The repository includes small sample files for development/testing. Large model files and complete datasets are downloaded separately.

## ⚙️ Configuration

1. **Copy the example environment file**
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys**
   
   Edit `.env` and add the API keys for providers you want to use:
   ```
   # Core Providers
   OPENAI_API_KEY=your-openai-key-here
   ANTHROPIC_API_KEY=your-anthropic-key-here
   GOOGLE_API_KEY=your-google-key-here
   
   # Additional Providers (optional)
   AZURE_OPENAI_ENDPOINT=your-azure-endpoint
   AZURE_OPENAI_API_KEY=your-azure-key
   COHERE_API_KEY=your-cohere-key
   MISTRAL_API_KEY=your-mistral-key
   PERPLEXITY_API_KEY=your-perplexity-key
   
   # Monitoring (optional)
   SLACK_WEBHOOK_URL=your-slack-webhook
   SMTP_SERVER=smtp.gmail.com
   SMTP_USER=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   ```

   API Key Resources:
   - OpenAI: [platform.openai.com](https://platform.openai.com/api-keys)
   - Anthropic: [console.anthropic.com](https://console.anthropic.com/)
   - Google: [makersuite.google.com](https://makersuite.google.com/app/apikey)
   - Azure: [portal.azure.com](https://portal.azure.com/)
   - Cohere: [dashboard.cohere.ai](https://dashboard.cohere.ai/)
   - Mistral: [console.mistral.ai](https://console.mistral.ai/)

## 🎮 Usage

### Use Case 1: Multi-Provider Benchmarking

Run comprehensive benchmarks across multiple providers:

```bash
# Basic benchmark
python src/use_cases/multi_provider_benchmarking.py

# With specific providers and models
python src/use_cases/multi_provider_benchmarking.py \
    --providers openai anthropic google \
    --models gpt-4 claude-3-5-sonnet-20241022 gemini-1.5-pro \
    --datasets truthfulness gsm8k humaneval \
    --output-format html pdf csv
```

### Use Case 2: Cost-Optimized Model Selection

Analyze cost vs performance trade-offs:

```bash
python src/use_cases/cost_optimization.py \
    --budget 1000 \
    --min-accuracy 0.9 \
    --workload-profile production
```

### Use Case 3: A/B Testing

Compare models in production:

```bash
python src/use_cases/ab_testing.py \
    --model-a gpt-4 \
    --model-b claude-3-5-sonnet-20241022 \
    --traffic-split 50/50 \
    --duration 7d
```

### Use Case 4: Custom Evaluation Pipelines

Create domain-specific evaluations:

```bash
python src/use_cases/custom_evaluation.py \
    --evaluator medical \
    --test-suite clinical-notes \
    --compliance hipaa
```

### Use Case 5: Production Monitoring

Monitor deployed models:

```bash
python src/use_cases/monitoring/dashboard.py \
    --models production-models.yaml \
    --port 8080 \
    --alerts slack email
```

### Use Case 6: Model Fine-Tuning

Fine-tune models on custom data:

```bash
# Cloud fine-tuning
python src/use_cases/fine_tuning.py \
    --provider openai \
    --base-model gpt-3.5-turbo \
    --dataset custom-data.jsonl \
    --epochs 3

# Local fine-tuning
python src/use_cases/fine_tuning.py \
    --local \
    --model llama-2-7b \
    --method lora \
    --dataset custom-data.json
```

### Use Case 7: Alignment Research

Implement safety measures:

```bash
python src/use_cases/alignment.py \
    --rules constitutional-ai.yaml \
    --safety-filters strict \
    --consensus-models 3
```

### Use Case 8: Automated Reporting

Generate comprehensive reports:

```bash
python src/use_cases/reporting.py \
    --schedule daily \
    --recipients team@company.com \
    --include costs performance safety compliance

### Example Output

```
🔬 LLM Lab Benchmark Runner
==================================================

1. Loading configuration...
   ✓ Model configuration loaded
   ✓ API key loaded for google

2. Initializing provider...
   ✓ google provider initialized

3. Loading dataset...
   ✓ Dataset validated
   ✓ Loaded 1 prompts from truthfulness

4. Running evaluations...
   Prompt 1/1: Who wrote 'Don Quixote'?...
   Response: Miguel de Cervantes wrote Don Quixote...
   ✓ Evaluation passed (matched: Cervantes, Miguel de Cervantes)

==================================================
📊 Benchmark Results Summary
==================================================
Provider: google
Dataset: truthfulness
Total Prompts: 1
Successful: 1
Failed: 0
Overall Score: 100.00%
```

## 📊 Results and Reporting

### Enhanced Results Logging

Results are automatically saved with organized structure and enhanced file naming:

```
results/
├── truthfulness/                    # Dataset-based organization
│   └── 2024-12/                    # Year-month organization
│       ├── benchmark_google_gemini-1-5-flash_truthfulness_20241231_143022.csv
│       ├── benchmark_openai_gpt-4o-mini_truthfulness_20241231_143022.csv
│       ├── benchmark_anthropic_claude-3-haiku_truthfulness_20241231_143022.csv
│       ├── results_index.json       # Metadata index
│       └── performance_analysis_20241231_143022.html
└── performance_benchmarks/
    └── 2024-12/
        ├── performance_report_20241231_143022.json
        └── benchmark_charts_20241231_143022.png
```

### CSV Results Format

Results are saved in CSV format with enhanced columns:

| Column | Description |
|--------|-------------|
| timestamp | ISO format timestamp of the evaluation |
| provider | Provider name (google, openai, anthropic) |
| model_name | Full model identifier (e.g., google/gemini-1.5-flash) |
| model_version | Model version/variant |
| benchmark_name | Name of the benchmark dataset |
| prompt_id | Unique identifier for the prompt |
| prompt_text | The input prompt text |
| model_response | The model's generated response |
| expected_keywords | JSON array of expected keywords |
| matched_keywords | JSON array of matched keywords |
| score | Evaluation score (0.0 to 1.0) |
| success | Pass/fail status |
| evaluation_method | Method used for evaluation |
| response_time_seconds | Time taken to generate response |
| tokens_used | Number of tokens consumed (if available) |
| cost_estimate | Estimated cost in USD (if available) |
| error | Any error message (if applicable) |
| session_id | Unique session identifier for grouping |
| run_metadata | Additional run configuration (JSON) |

### Performance Analysis Reports

When `--analyze-results` is used, additional reports are generated:

- **HTML Performance Dashboard**: Interactive charts and analysis
- **Statistical Summary**: Response time distributions, confidence intervals
- **Comparison Matrix**: Head-to-head model comparisons
- **Cost Analysis**: Token usage and cost estimates by provider
- **Error Analysis**: Failure patterns and retry statistics

## 📁 Project Structure

```
lllm-lab/
├── src/                     # Source code
│   ├── llm_providers/       # LLM provider implementations
│   │   ├── base.py          # Base provider interface
│   │   ├── openai.py        # OpenAI GPT models
│   │   ├── anthropic.py     # Anthropic Claude models
│   │   ├── google.py        # Google Gemini models
│   │   ├── azure.py         # Azure OpenAI service
│   │   ├── cohere.py        # Cohere models
│   │   ├── mistral.py       # Mistral models
│   │   └── local.py         # Local model support
│   ├── evaluation/          # Evaluation methods
│   │   ├── metrics.py       # Core metrics
│   │   ├── custom_evaluators.py # Domain evaluators
│   │   └── statistical_analysis.py # Stats tools
│   ├── use_cases/           # Production use cases
│   │   ├── multi_provider_benchmarking.py
│   │   ├── cost_optimization.py
│   │   ├── ab_testing.py
│   │   ├── custom_evaluation.py
│   │   ├── fine_tuning.py
│   │   ├── alignment.py
│   │   ├── monitoring/      # Monitoring subsystem
│   │   │   ├── dashboard.py
│   │   │   ├── metrics_collector.py
│   │   │   └── alert_manager.py
│   │   └── reporting.py
│   └── utils/               # Shared utilities
│       ├── cost_tracker.py
│       ├── results_logger.py
│       └── config_manager.py
├── examples/                # Example implementations
│   ├── use_cases/           # Use case examples
│   │   ├── alignment_demo.py
│   │   ├── fine_tuning_complete_demo.py
│   │   ├── monitoring_complete_demo.py
│   │   └── integrated_workflow_demo.py
│   ├── custom_prompts/      # Prompt templates
│   │   ├── alignment_rules.yaml
│   │   ├── safety_filters.yaml
│   │   └── monitoring_config.yaml
│   └── notebooks/           # Jupyter notebooks
│       └── integrated_workflows.ipynb
├── benchmarks/              # Benchmark datasets
│   ├── truthfulness/
│   ├── gsm8k/
│   ├── humaneval/
│   └── custom/
├── datasets/                # Training datasets
│   ├── benchmarking/
│   └── fine-tuning/
├── models/                  # Model storage
│   ├── cloud-configs/       # Cloud model configs
│   └── local-models/        # Local model files
├── tests/                   # Comprehensive test suite
│   ├── test_alignment.py
│   ├── test_fine_tuning.py
│   ├── test_monitoring.py
│   ├── test_integrated_workflows.py
│   └── ... (40+ test files)
├── docs/                    # Documentation
│   ├── guides/              # How-to guides
│   │   ├── USE_CASE_*.md   # Use case guides
│   │   └── README.md        # Guide index
│   ├── api/                 # API documentation
│   └── architecture/        # System design docs
├── templates/               # Report templates
│   ├── email/
│   └── pdf/
├── .github/                 # GitHub configuration
│   └── workflows/
├── docker/                  # Docker configurations
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt         # Core dependencies
├── requirements-dev.txt     # Development deps
├── requirements-gpu.txt     # GPU/ML deps
├── pyproject.toml          # Project config
├── .env.example            # Environment template
└── README.md               # This file
```

### 🗂️ Asset Organization

- **Sample files**: Small example files are included in the repository for testing
- **Large files**: Models (GB-sized) and full datasets are downloaded via `download_assets.py`
- **Ignore patterns**: Large files are gitignored to keep the repository lightweight

## 🧪 Testing

The project includes comprehensive test coverage for all components:

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_alignment.py      # Alignment tests
pytest tests/test_fine_tuning.py    # Fine-tuning tests
pytest tests/test_monitoring.py     # Monitoring tests
pytest tests/test_integrated_workflows.py  # Integration tests

# Run with coverage report
pytest --cov=. --cov-report=html --cov-report=term

# Run only fast tests (skip integration)
pytest -m "not integration"

# Run with specific provider
pytest --provider=openai
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component workflows
- **Performance Tests**: Latency and throughput benchmarks
- **Safety Tests**: Alignment and filter validation
- **Mock Tests**: No API calls required

### Coverage Report

```bash
# Generate and view coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

Current coverage: 95%+ across all modules

## 🔧 Development

### Quick Start Development Guide

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run code quality checks
make lint
make format
make type-check

# Run all checks before committing
make check-all
```

### Adding New Features

#### New LLM Provider

1. Extend `BaseProvider` in `src/llm_providers/base.py`
2. Implement required methods:
   - `generate()` - Text generation
   - `get_model_info()` - Model metadata
   - `estimate_cost()` - Cost calculation
3. Add provider configuration to `config.yaml`
4. Write tests in `tests/test_providers/`

#### New Evaluation Method

1. Create evaluator in `src/evaluation/custom_evaluators.py`
2. Implement `evaluate()` method returning scores
3. Register in `EVALUATOR_REGISTRY`
4. Add example usage to documentation

#### New Use Case

1. Create script in `src/use_cases/`
2. Add corresponding guide in `docs/guides/`
3. Include example in `examples/use_cases/`
4. Write comprehensive tests

### Code Quality Standards

- **Type Hints**: All functions must have type annotations
- **Docstrings**: Google-style docstrings required
- **Testing**: Minimum 90% coverage for new code
- **Linting**: Pass all ruff and mypy checks
- **Documentation**: Update relevant docs with changes

## 🚀 Advanced Features

### Production Deployment

#### Docker Deployment

```bash
# Build and run with Docker
docker build -t llm-lab .
docker run -p 8080:8080 --env-file .env llm-lab

# Using docker-compose for full stack
docker-compose up -d
```

#### Kubernetes Deployment

```yaml
# See kubernetes/ directory for manifests
kubectl apply -f kubernetes/
```

#### Monitoring Stack

- **Grafana Dashboards**: Pre-built dashboards for all metrics
- **Prometheus Integration**: Metrics export in Prometheus format
- **Custom Alerts**: Slack, Email, PagerDuty, SMS support
- **SLA Tracking**: Automated compliance reporting

### Enterprise Features

#### Multi-Tenancy

```python
# Configure tenant isolation
config = {
    "tenants": {
        "team-a": {"models": ["gpt-4"], "budget": 1000},
        "team-b": {"models": ["claude-3-5"], "budget": 2000}
    }
}
```

#### Compliance & Security

- **Data Encryption**: At-rest and in-transit
- **Audit Logging**: Complete activity tracking
- **PII Detection**: Automatic redaction
- **HIPAA/GDPR**: Compliance templates included

#### High Availability

- **Load Balancing**: Automatic request distribution
- **Failover**: Multi-region support
- **Caching**: Response caching with TTL
- **Rate Limiting**: Per-tenant and global limits

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following our code standards
4. Write/update tests (maintain 90%+ coverage)
5. Update documentation
6. Run `make check-all` to ensure quality
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request with a detailed description

### Areas for Contribution

- 🌟 New LLM provider integrations
- 📊 Additional benchmark datasets
- 🔧 Performance optimizations
- 📚 Documentation improvements
- 🌍 Internationalization
- 🧪 Test coverage expansion

## 📚 Resources

### Documentation

- [Full Documentation](https://llm-lab.readthedocs.io)
- [API Reference](docs/api/README.md)
- [Architecture Guide](docs/architecture/README.md)
- [Use Case Guides](docs/guides/README.md)

### Community

- [Discord Server](https://discord.gg/llm-lab)
- [GitHub Discussions](https://github.com/yourusername/llm-lab/discussions)
- [Twitter Updates](https://twitter.com/llm_lab)

### Related Projects

- [LangChain](https://github.com/hwchase17/langchain) - LLM application framework
- [LlamaIndex](https://github.com/jerryjliu/llama_index) - Data framework for LLMs
- [Haystack](https://github.com/deepset-ai/haystack) - NLP framework

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI, Anthropic, Google, and other LLM providers for their APIs
- The open-source community for invaluable tools and libraries
- Contributors and users who help improve LLM Lab
- Research papers and benchmarks that inform our evaluations

## 📊 Project Status

![GitHub Stars](https://img.shields.io/github/stars/yourusername/llm-lab)
![GitHub Forks](https://img.shields.io/github/forks/yourusername/llm-lab)
![License](https://img.shields.io/github/license/yourusername/llm-lab)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)

---

Built with ❤️ by the LLM Lab Team