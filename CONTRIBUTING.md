# Contributing to LLM Lab

Thank you for your interest in contributing to LLM Lab! We welcome contributions and are grateful for any help you can provide.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Getting Help](#getting-help)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the project
- Show empathy towards other contributors

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lllm-lab.git
   cd lllm-lab
   ```
3. **Set up the development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
4. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/lllm-lab.git
   ```

## How to Contribute

### Reporting Issues

- **Check existing issues** first to avoid duplicates
- Use the issue templates when available
- Provide clear, detailed information:
  - Steps to reproduce
  - Expected behavior
  - Actual behavior
  - Environment details (OS, Python version, etc.)
  - Error messages and stack traces

### Suggesting Features

- Open a discussion first for major features
- Clearly describe the problem your feature solves
- Provide use cases and examples
- Consider the maintenance burden

### Contributing Code

Areas where we especially welcome contributions:

- **New LLM Provider Integrations**: Add support for new LLM providers
- **Benchmark Datasets**: Contribute new evaluation datasets
- **Performance Optimizations**: Improve speed and efficiency
- **Documentation**: Improve guides, API docs, and examples
- **Test Coverage**: Expand test suites
- **Bug Fixes**: Fix reported issues

## Development Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add/update tests as needed
- Update documentation

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### 4. Lint and Format

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all checks
make check-all
```

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all tests** and ensure they pass:
   ```bash
   make test
   make check-all
   ```

3. **Update documentation** if needed

4. **Write a clear commit message**:
   ```
   feat: Add support for Cohere API integration
   
   - Implement CohereProvider class
   - Add configuration options
   - Include comprehensive tests
   - Update documentation
   
   Closes #123
   ```

### Submitting the PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create the Pull Request** on GitHub

3. **Fill out the PR template** completely:
   - Clear description of changes
   - Link related issues
   - List any breaking changes
   - Include test results

4. **Address review feedback** promptly

### PR Requirements

- All tests must pass
- Code coverage must not decrease
- Documentation must be updated
- Commit messages must be clear
- PR must be up to date with main branch

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html)
- Maximum line length: 100 characters
- Use type hints for all functions

### Example Code Style

```python
from typing import List, Dict, Optional

class ExampleProvider(BaseProvider):
    """Example LLM provider implementation.
    
    This provider demonstrates the expected code style and documentation
    standards for the project.
    
    Args:
        api_key: The API key for authentication
        timeout: Request timeout in seconds
        
    Attributes:
        model_name: The name of the model being used
        max_tokens: Maximum tokens for generation
    """
    
    def __init__(self, api_key: str, timeout: int = 30) -> None:
        """Initialize the provider with configuration."""
        super().__init__()
        self.api_key = api_key
        self.timeout = timeout
        
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text from the model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing the generated text and metadata
            
        Raises:
            ProviderError: If the API request fails
        """
        # Implementation here
        pass
```

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import List, Dict

# Third-party imports
import numpy as np
import pandas as pd
from loguru import logger

# Local imports
from src.providers.base import BaseProvider
from src.config import Config
```

## Testing Requirements

### Test Structure

- Write tests for all new functionality
- Place tests in appropriate directories:
  - `tests/unit/` - Unit tests
  - `tests/integration/` - Integration tests
  - `tests/performance/` - Performance tests

### Test Style

```python
import pytest
from unittest.mock import Mock, patch

class TestExampleProvider:
    """Test suite for ExampleProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing."""
        return ExampleProvider(api_key="test-key")
    
    def test_generate_success(self, provider):
        """Test successful text generation."""
        # Arrange
        prompt = "Test prompt"
        expected = {"text": "Generated text"}
        
        # Act
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = expected
            result = provider.generate(prompt)
        
        # Assert
        assert result == expected
        mock_post.assert_called_once()
```

### Coverage Requirements

- New code must have at least 90% test coverage
- Run coverage reports:
  ```bash
  pytest --cov=src --cov-report=term-missing
  ```

## Documentation

### Code Documentation

- All public functions must have docstrings
- Use clear, descriptive variable names
- Add inline comments for complex logic
- Update relevant documentation files

### Documentation Types

1. **API Documentation**: Update docstrings
2. **User Guides**: Update guides in `docs/guides/`
3. **Examples**: Add examples in `examples/`
4. **README**: Update if adding major features

### Documentation Checklist

- [ ] All new functions have docstrings
- [ ] Complex algorithms are explained
- [ ] Examples are provided for new features
- [ ] README is updated if needed
- [ ] CHANGELOG is updated

## Getting Help

- Check the documentation first
- Search existing issues and discussions
- Ask in GitHub Discussions
- Open an issue if you need assistance

### Reviewing PRs

Contributors are encouraged to review PRs:

- Be constructive and respectful
- Focus on the code, not the person
- Suggest improvements, don't demand them
- Approve PRs you've reviewed thoroughly

### Becoming a Maintainer

Active contributors may be invited to become maintainers. Maintainers:

- Review and merge PRs
- Triage issues
- Guide project direction
- Help other contributors

## Recognition

We value all contributions, not just code:

- **Code contributions**: Features, bug fixes, optimizations
- **Documentation**: Guides, examples, API docs
- **Testing**: New tests, test improvements
- **Reviews**: PR reviews, issue triage
- **Support**: Helping others, discussions

Contributors are recognized in:
- The project README
- Release notes
- Our contributors page

## Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search GitHub issues and discussions
3. Ask in GitHub Discussions
4. Contact the maintainers

Thank you for contributing to LLM Lab! Your efforts help make this project better for everyone.