# Model Documentation System

Automated documentation generation system for machine learning models with compliance reporting and ethical assessment capabilities.

## Features

- **Automatic Model Analysis**: Extract metadata, architecture, and parameters from models
- **Model Card Generation**: Create comprehensive model cards following industry standards
- **Multi-Framework Support**: PyTorch, TensorFlow, ONNX, scikit-learn, Hugging Face Transformers
- **Compliance Reporting**: EU AI Act, ISO 26000, Model Cards standard
- **Ethical Assessment**: Bias detection, fairness metrics, privacy considerations
- **Multiple Output Formats**: Markdown, JSON, PDF

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/llm-lab/model-documentation-system.git
cd model-documentation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Using pip

```bash
pip install model-documentation-system
```

## Quick Start

### Command Line Interface

```bash
# Initialize a new project
model-docs init --path my-project

# Generate documentation for a model
model-docs generate models/my_model.pt -o docs/model_card.md

# Generate PDF documentation
model-docs generate models/my_model.h5 -o docs/model_card.pdf -f pdf

# Validate compliance
model-docs validate docs/model_card.json --framework eu_ai_act
```

### Python API

```python
from model_docs import ModelCardGenerator, ComplianceGenerator
from model_docs.analyzers import ModelLoader

# Load a model
model = ModelLoader.load_model("path/to/model.pt")

# Generate model card
generator = ModelCardGenerator()
model_card = generator.generate_model_card(
    model,
    usage_guidelines="This model is intended for research purposes only.",
    license="MIT"
)

# Save documentation
generator.save_model_card(model_card, "model_card.md", format="markdown")

# Check compliance
compliance = ComplianceGenerator()
report = compliance.generate_compliance_report(model_card, "eu_ai_act")
print(f"Compliant: {report.compliant}")
print(f"Score: {report.score:.1%}")
```

## Model Card Structure

The generated model cards include:

### Model Details
- Version and framework information
- Architecture specifications
- Parameter counts
- Input/output shapes

### Training Configuration
- Dataset information
- Hyperparameters
- Hardware used
- Training time and carbon footprint

### Performance Metrics
- Standard metrics (accuracy, precision, recall, F1)
- Custom metrics
- Evaluation dataset details

### Ethical Considerations
- Intended use cases
- Known limitations
- Potential biases
- Privacy implications
- Environmental impact

## Supported Frameworks

- **PyTorch**: `.pt`, `.pth` files
- **TensorFlow/Keras**: `.h5`, `.keras`, SavedModel format
- **ONNX**: `.onnx` files
- **scikit-learn**: `.pkl`, `.joblib` files
- **Hugging Face Transformers**: Model directories with `config.json`

## Compliance Frameworks

### EU AI Act
- Risk assessment documentation
- Transparency requirements
- Human oversight documentation
- Data governance

### ISO 26000
- Social responsibility assessment
- Stakeholder engagement
- Ethical considerations
- Environmental impact

### Model Cards Standard
- Model details
- Training data documentation
- Evaluation data
- Ethical considerations

## Advanced Usage

### Custom Templates

Create custom Jinja2 templates in the `templates/` directory:

```jinja2
# Custom Model Documentation

## Model: {{ metadata.name }}

Version: {{ metadata.version }}
Created: {{ metadata.created_date }}

{{ custom_content }}
```

### Batch Processing

```python
from pathlib import Path
from model_docs import ModelCardGenerator

generator = ModelCardGenerator()

# Process multiple models
model_dir = Path("models/")
for model_path in model_dir.glob("*.pt"):
    model = ModelLoader.load_model(model_path)
    model_card = generator.generate_model_card(model)
    output_path = Path("docs") / f"{model_path.stem}_card.md"
    generator.save_model_card(model_card, output_path)
```

### Integration with MLOps

```python
# Integration with MLflow
import mlflow
from model_docs import ModelCardGenerator

with mlflow.start_run():
    # Train model
    model = train_model()

    # Log model
    mlflow.pytorch.log_model(model, "model")

    # Generate and log documentation
    generator = ModelCardGenerator()
    model_card = generator.generate_model_card(model)

    # Save as artifact
    generator.save_model_card(model_card, "model_card.md")
    mlflow.log_artifact("model_card.md")
```

## Configuration

Create a `config.yaml` file:

```yaml
defaults:
  format: markdown
  frameworks:
    - eu_ai_act
    - model_cards

templates:
  model_card: templates/model_card.md.j2
  compliance_report: templates/compliance.md.j2

compliance:
  strict_mode: true
  auto_generate_recommendations: true

output:
  include_timestamp: true
  include_changelog: true
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=model_docs

# Run specific test
pytest tests/test_generators.py
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
mypy src/
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{model_documentation_system,
  title = {Model Documentation System},
  author = {LLM Lab Team},
  year = {2024},
  url = {https://github.com/llm-lab/model-documentation-system}
}
```

## Support

For support, email team@llm-lab.io or open an issue on GitHub.

## Acknowledgments

- Model Cards paper by Mitchell et al.
- EU AI Act documentation requirements
- ISO 26000 social responsibility guidelines
