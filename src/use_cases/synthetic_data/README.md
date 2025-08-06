# Synthetic Data Generation Platform

A comprehensive platform for generating high-quality synthetic data using LLMs with built-in privacy preservation and quality validation.

## Features

- **Multi-Domain Support**: Generate synthetic data for medical, financial, legal, e-commerce, educational, and code domains
- **Privacy Preservation**: Built-in PII detection, anonymization, and differential privacy
- **Quality Validation**: Automated quality scoring, diversity metrics, and consistency checks
- **Flexible Generation**: Batch processing, streaming, and customizable generation parameters
- **Multiple Export Formats**: JSON, CSV, Parquet, JSONL
- **LLM Provider Support**: OpenAI, Anthropic, Google AI, and more

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/llm-lab/synthetic-data-platform.git
cd synthetic-data-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Using pip

```bash
pip install synthetic-data-platform
```

## Quick Start

### Basic Usage

```python
from synthetic_data import MedicalDataGenerator

# Initialize generator
generator = MedicalDataGenerator()

# Generate a single record
patient = generator.generate_single(record_type="patient")
print(patient)

# Generate a dataset
result = generator.generate_dataset(
    count=100,
    record_type="patient",
    validate=True,
    preserve_privacy=True
)

# Export data
generator.export_data(result, "patients.json", format="json")
```

### Domain-Specific Generators

#### Medical Data
```python
from synthetic_data import MedicalDataGenerator

generator = MedicalDataGenerator()

# Generate different record types
patient = generator.generate_single(record_type="patient")
encounter = generator.generate_single(record_type="encounter")
lab_result = generator.generate_single(record_type="lab_result")
prescription = generator.generate_single(record_type="prescription")
```

#### Financial Data
```python
from synthetic_data import FinancialDataGenerator

generator = FinancialDataGenerator()

# Generate financial records
transaction = generator.generate_single(record_type="transaction")
account = generator.generate_single(record_type="account")
loan = generator.generate_single(record_type="loan")
investment = generator.generate_single(record_type="investment")
```

### Privacy Preservation

```python
from synthetic_data import DataValidator, PrivacyPreserver

# Configure privacy settings
privacy = PrivacyPreserver({
    "anonymization_level": "high",
    "pii_detection": True,
    "pii_removal": True,
    "differential_privacy": True,
    "differential_privacy_epsilon": 1.0
})

# Apply privacy to existing data
protected_data = privacy.apply_privacy(your_data)

# Validate privacy
validation = privacy.validate_privacy(protected_data)
print(f"PII found: {validation['pii_found']}")
print(f"Anonymization coverage: {validation['anonymization_coverage']}")
```

### Quality Validation

```python
from synthetic_data import DataValidator

validator = DataValidator()

# Validate data quality
quality_result = validator.calculate_quality_score(generated_data)
print(f"Quality score: {quality_result.score}")
print(f"Metrics: {quality_result.metrics}")
print(f"Recommendations: {quality_result.recommendations}")

# Validate batch
validation = validator.validate_batch(generated_data)
if validation.is_valid:
    print("Data validation passed")
else:
    print(f"Errors: {validation.errors}")
    print(f"Warnings: {validation.warnings}")
```

## Command Line Interface

```bash
# Generate synthetic data
synth-generate --domain medical --type patient --count 1000 --output patients.json

# Validate existing data
synth-validate --input data.json --output validation-report.json

# Apply privacy preservation
synth-data privacy --input raw-data.json --output private-data.json --level high
```

## Configuration

Create a `config/config.yaml` file:

```yaml
generation:
  default_model: "gpt-4"
  temperature: 0.7
  batch_size: 10

validation:
  min_quality_score: 0.8
  check_diversity: true
  check_consistency: true

privacy:
  anonymization_level: "medium"
  pii_detection: true
  differential_privacy_epsilon: 1.0

domains:
  medical:
    privacy_level: "high"
    validation_rules: "strict"

  financial:
    privacy_level: "high"
    validation_rules: "strict"
```

## Supported Domains

### Medical/Healthcare
- Patient records
- Medical encounters
- Lab results
- Prescriptions
- Vital signs

### Financial/Banking
- Transactions
- Account information
- Customer profiles
- Loans
- Investments

### Legal
- Case records
- Legal documents
- Court proceedings

### E-commerce
- Orders
- Customer data
- Product information
- Reviews

### Educational
- Student records
- Course information
- Grades and assessments

### Code/Programming
- Repository metadata
- Code snippets
- Documentation

## Advanced Features

### Batch Processing

```python
# Generate large datasets efficiently
result = generator.generate_dataset(
    count=10000,
    batch_size=100,
    validate=True
)
```

### Custom Templates

```python
# Use custom generation templates
generator = MedicalDataGenerator()
generator.config.templates_path = "custom_templates/"
```

### Export Options

```python
# Export in different formats
generator.export_data(data, "output.csv", format="csv")
generator.export_data(data, "output.parquet", format="parquet")
generator.export_data(data, "output.jsonl", format="jsonl")
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=synthetic_data

# Run specific test file
pytest tests/test_generators.py
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/
```

## API Reference

### SyntheticDataGenerator

Base class for all generators.

**Methods:**
- `generate_single(**kwargs)`: Generate a single record
- `generate_dataset(count, batch_size, validate, preserve_privacy)`: Generate multiple records
- `validate_quality(data)`: Validate data quality
- `ensure_privacy(data)`: Apply privacy preservation
- `export_data(data, filepath, format)`: Export data to file

### DataValidator

Validates synthetic data quality.

**Methods:**
- `validate_batch(data)`: Validate a batch of data
- `calculate_quality_score(data)`: Calculate quality metrics
- `validate_quality(data)`: Comprehensive quality validation

### PrivacyPreserver

Handles privacy preservation.

**Methods:**
- `apply_privacy(data)`: Apply privacy preservation
- `detect_pii(text)`: Detect PII in text
- `validate_privacy(data)`: Validate privacy preservation

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, email team@llm-lab.io or open an issue on GitHub.

## Acknowledgments

- Faker library for realistic data generation
- scikit-learn for diversity metrics
- Privacy preservation techniques from differential privacy research
