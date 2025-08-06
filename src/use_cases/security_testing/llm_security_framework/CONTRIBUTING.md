# Contributing to LLM Security Framework

We welcome contributions to the LLM Security Framework! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/llm-security-framework.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest`
6. Commit your changes: `git commit -m "feat: add new feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use the following tools to maintain code quality:
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all formatters and linters:
```bash
make format
make lint
```

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting PR
- Aim for >80% test coverage

Run tests:
```bash
make test
```

## Commit Messages

We follow conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/tool changes

## Pull Request Process

1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure all tests pass
4. Update README.md if needed
5. Request review from maintainers

## Code of Conduct

Please be respectful and inclusive in all interactions. We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

## Questions?

Open an issue or contact the maintainers at team@llm-lab.io
