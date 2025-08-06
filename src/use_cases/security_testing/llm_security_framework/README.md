# LLM Security Testing Framework

A comprehensive framework for testing and evaluating the security of Large Language Models (LLMs), including vulnerability scanning, red team simulations, and compliance reporting.

## Features

- **Security Scanning**: Automated detection of common LLM vulnerabilities
- **Red Team Simulation**: Adversarial testing with customizable attack scenarios
- **Compliance Reporting**: Generate reports for OWASP Top 10, NIST, and custom frameworks
- **Prompt Analysis**: Deep analysis of prompt injection and manipulation attempts
- **Attack Generation**: Automated generation of test cases for security evaluation

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/llm-lab/llm-security-framework.git
cd llm-security-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Using pip

```bash
pip install llm-security-framework
```

## Quick Start

### Command Line Interface

```bash
# Run a basic security scan
llm-scan --model gpt-3.5-turbo --tests all

# Perform red team simulation
llm-redteam --scenario jailbreak --target http://localhost:8000/api

# Generate compliance report
llm-security report --framework owasp --output report.pdf
```

### Python API

```python
from llm_security import SecurityScanner, RedTeamSimulator

# Initialize scanner
scanner = SecurityScanner(
    model="gpt-3.5-turbo",
    api_key="your-api-key"
)

# Run security tests
results = scanner.run_tests([
    "prompt_injection",
    "data_extraction",
    "jailbreak"
])

# Red team simulation
redteam = RedTeamSimulator()
attacks = redteam.generate_attacks(
    target_behavior="reveal_training_data",
    num_attempts=100
)
```

## Project Structure

```
llm-security-framework/
├── src/
│   └── llm_security/
│       ├── core/           # Core scanning and reporting
│       ├── scanner/        # Vulnerability scanning
│       ├── redteam/        # Red team simulations
│       ├── compliance/     # Compliance checking
│       └── utils/          # Utilities and helpers
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
├── config/                 # Configuration files
└── examples/              # Example scripts and configs
```

## Development

### Setting up development environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_security

# Run specific test file
pytest tests/test_scanner.py
```

## Configuration

Create a `config/config.yaml` file:

```yaml
scanner:
  timeout: 30
  max_retries: 3

redteam:
  scenarios:
    - jailbreak
    - prompt_injection
    - data_extraction

compliance:
  frameworks:
    - owasp
    - nist
    - custom
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, email team@llm-lab.io or open an issue on GitHub.

## Acknowledgments

- OWASP for security testing guidelines
- The AI safety research community
- Contributors and maintainers
