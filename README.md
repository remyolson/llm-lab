# LLM Lab - Large Language Model Benchmark Framework

A modular framework for benchmarking Large Language Model (LLM) performance across various evaluation tasks.

## 🚀 Overview

LLM Lab provides a flexible and extensible platform for evaluating LLM capabilities through standardized benchmarks. The framework currently supports Google's Gemini models and includes a truthfulness benchmark with keyword-based evaluation.

### Key Features

- **Modular Architecture**: Easily add new LLM providers and evaluation methods
- **Google Gemini Integration**: Built-in support for Gemini 1.5 Flash
- **Automated Evaluation**: Keyword matching evaluation for response validation
- **CSV Result Logging**: Detailed results exported to CSV for analysis
- **Comprehensive Testing**: 83%+ test coverage with pytest
- **Code Quality**: Enforced with ruff linter and GitHub Actions CI

## 📋 Prerequisites

- Python 3.8 or higher
- Google Cloud API key with Gemini API access
- pip package manager

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

2. **Add your Google API key**
   
   Edit `.env` and add your API key:
   ```
   GOOGLE_API_KEY=your-api-key-here
   ```

   To obtain a Google API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create or select a project
   - Generate an API key

## 🎮 Usage

Run a benchmark with the following command:

```bash
python run_benchmarks.py --provider google --dataset truthfulness
```

### Command Options

- `--provider`: LLM provider to use (currently only `google` is supported)
- `--dataset`: Benchmark dataset to run (currently only `truthfulness` is available)
- `--no-csv`: Disable CSV output
- `--output-dir`: Directory for saving results (default: `./results`)

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

## 📊 Results Format

Results are saved in CSV format with the following columns:

| Column | Description |
|--------|-------------|
| timestamp | ISO format timestamp of the evaluation |
| model_name | Model identifier (e.g., google/gemini-1.5-flash) |
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
| error | Any error message (if applicable) |

## 📁 Project Structure

```
lllm-lab/
├── benchmarks/              # Benchmark datasets
│   └── truthfulness/        # Truthfulness benchmark
│       ├── __init__.py      # Dataset validation
│       └── dataset.jsonl    # Sample benchmark prompts
├── datasets/                # Dataset management
│   ├── benchmarking/        # Benchmark datasets
│   │   ├── raw/             # Raw dataset files
│   │   └── processed/       # Processed dataset files
│   ├── fine-tuning/         # Fine-tuning datasets
│   └── download_datasets.py # Dataset utilities
├── models/                  # Model files (downloaded separately)
│   └── small-llms/          # Small language models
│       ├── qwen-0.5b/       # Qwen 0.5B model files
│       ├── qwen-0.5b-gguf/  # Qwen 0.5B GGUF format
│       ├── smollm-135m/     # SmolLM 135M model files
│       └── smollm-360m/     # SmolLM 360M model files
├── evaluation/              # Evaluation methods
│   ├── __init__.py
│   └── keyword_match.py     # Keyword matching evaluator
├── llm_providers/           # LLM provider implementations
│   ├── __init__.py
│   ├── base.py              # Base provider interface
│   ├── google.py            # Google Gemini provider
│   ├── openai.py            # OpenAI provider
│   └── anthropic.py         # Anthropic provider
├── tests/                   # Test suite
│   ├── test_config.py       # Configuration tests
│   ├── test_evaluation.py   # Evaluation tests
│   ├── test_llm_providers.py # Provider tests
│   └── test_results_logger.py # Logger tests
├── .github/                 # GitHub configuration
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline
├── download_assets.py       # 📥 Asset download script
├── config.py                # Configuration management
├── results_logger.py        # CSV result logging
├── run_benchmarks.py        # Main entry point
├── requirements.txt         # Python dependencies
├── pytest.ini              # Pytest configuration
├── ruff.toml              # Linter configuration
├── .env.example           # Environment template
└── README.md              # This file
```

### 🗂️ Asset Organization

- **Sample files**: Small example files are included in the repository for testing
- **Large files**: Models (GB-sized) and full datasets are downloaded via `download_assets.py`
- **Ignore patterns**: Large files are gitignored to keep the repository lightweight

## 🧪 Testing

Run the test suite:

```bash
pytest
```

Run tests with coverage report:

```bash
pytest --cov=. --cov-report=html
```

View the HTML coverage report:

```bash
open htmlcov/index.html  # macOS
# or
start htmlcov/index.html  # Windows
```

## 🔧 Development

### Adding a New LLM Provider

1. Create a new file in `llm_providers/`
2. Implement the provider interface with a `generate(prompt: str) -> str` method
3. Add the provider to the `PROVIDERS` dict in `run_benchmarks.py`

### Adding a New Evaluation Method

1. Create a new file in `evaluation/`
2. Implement an evaluation function that returns a results dictionary
3. Update the benchmark dataset to use your evaluation method

### Code Quality

Run linting:

```bash
ruff check .
```

Auto-fix linting issues:

```bash
ruff check . --fix
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Generative AI team for the Gemini API
- The Python community for excellent tools and libraries