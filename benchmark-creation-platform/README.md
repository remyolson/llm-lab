# Benchmark Creation Platform

A comprehensive platform for creating, validating, and managing benchmarks for Large Language Model evaluation.

## Features

- **Test Case Generation**: Intelligent generation of diverse test cases across multiple domains
- **Quality Validation**: Comprehensive validation system ensuring benchmark quality
- **Storage & Versioning**: Git-based versioning system for benchmark evolution tracking
- **Multiple Export Formats**: Support for JSON, CSV, HuggingFace datasets, and custom formats
- **CLI Interface**: Command-line tools for benchmark management

## Installation

```bash
# Clone the repository
git clone https://github.com/llm-lab/benchmark-creation-platform.git
cd benchmark-creation-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```bash
# Create a new benchmark
benchmark-builder create --name "reasoning_benchmark" --domain "logical_reasoning" --size 1000

# Generate test cases
benchmark-builder generate --benchmark "reasoning_benchmark" --strategy "template"

# Validate benchmark quality
benchmark-builder validate --benchmark "reasoning_benchmark"

# Export benchmark
benchmark-builder export --benchmark "reasoning_benchmark" --format "json" --output "./exports/"
```

## Project Structure

```
benchmark-creation-platform/
├── src/
│   └── benchmark_builder/
│       ├── generators/      # Test case generation logic
│       ├── validators/      # Quality validation modules
│       ├── templates/       # Benchmark templates
│       └── storage/        # Storage and versioning
├── tests/                  # Unit tests
├── config/                 # Configuration files
├── benchmarks/            # Generated benchmarks
└── docs/                  # Documentation
```

## Core Components

### Test Case Generator
- Template-based generation
- Rule-based generation
- Model-assisted generation
- Domain-specific generators

### Quality Validator
- Completeness checking
- Answer correctness validation
- Difficulty distribution analysis
- Duplicate detection
- Format consistency validation

### Storage System
- CRUD operations for benchmarks
- Git-based version control
- Multiple export formats
- Metadata management

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
generation:
  default_strategy: "template"
  batch_size: 100

validation:
  min_quality_score: 0.8
  check_duplicates: true

storage:
  versioning_enabled: true
  default_format: "json"
```

## API Usage

```python
from benchmark_builder import BenchmarkBuilder
from benchmark_builder.generators import TextGenerator
from benchmark_builder.validators import QualityValidator

# Create builder
builder = BenchmarkBuilder()

# Generate test cases
generator = TextGenerator(config={
    "domain": "math",
    "difficulty": "medium",
    "count": 100
})
test_cases = generator.generate()

# Validate quality
validator = QualityValidator()
results = validator.validate(test_cases)

# Save benchmark
builder.save_benchmark(
    name="math_benchmark",
    test_cases=test_cases,
    metadata={"version": "1.0"}
)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=benchmark_builder

# Run specific test module
pytest tests/test_generators.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions, please use the [GitHub Issues](https://github.com/llm-lab/benchmark-creation-platform/issues) page.

## Acknowledgments

- Built as part of the LLM Lab project
- Inspired by leading benchmark datasets like MMLU, BigBench, and HellaSwag
