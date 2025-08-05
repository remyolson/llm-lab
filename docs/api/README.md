# LLM Lab API Reference

This directory contains comprehensive API documentation for all LLM Lab modules.

## Table of Contents

### Core Modules

- [Providers API](providers.md) - LLM provider interfaces and implementations
- [Configuration API](configuration.md) - Configuration management
- [Evaluation API](evaluation.md) - Evaluation metrics and methods
- [Analysis API](analysis.md) - Results analysis and comparison

### Use Case APIs

- [Benchmarking API](benchmarking.md) - Multi-provider benchmarking
- [Fine-tuning API](fine-tuning.md) - Model fine-tuning interfaces
- [Alignment API](alignment.md) - Alignment and safety features
- [Monitoring API](monitoring.md) - Production monitoring
- [Custom Prompts API](custom-prompts.md) - Custom prompt evaluation

### Utilities

- [Logging API](logging.md) - Results logging and storage
- [Validation API](validation.md) - Input validation utilities

## Quick Start

```python
from src.providers import OpenAIProvider, AnthropicProvider
from src.evaluation import TruthfulnessEvaluator
from src.analysis import ResultsComparator

# Initialize providers
openai = OpenAIProvider(model="gpt-4")
anthropic = AnthropicProvider(model="claude-3-5-sonnet-20241022")

# Run evaluation
evaluator = TruthfulnessEvaluator()
results = evaluator.evaluate_providers([openai, anthropic], dataset="truthfulqa")

# Compare results
comparator = ResultsComparator()
comparison = comparator.compare(results)
print(comparison.summary())
```

## API Conventions

### Return Types

All API methods follow consistent return type patterns:

- **Providers**: Return `ProviderResponse` objects
- **Evaluators**: Return `EvaluationResult` objects
- **Analyzers**: Return `AnalysisReport` objects

### Error Handling

All modules use typed exceptions:

```python
from src.providers.exceptions import (
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError
)
```

### Async Support

Most APIs support both sync and async operations:

```python
# Synchronous
response = provider.generate(prompt)

# Asynchronous
response = await provider.agenerate(prompt)
```

## Module Documentation

Each module documentation includes:

1. **Overview** - Module purpose and capabilities
2. **Classes** - Detailed class documentation
3. **Functions** - Standalone function documentation
4. **Examples** - Usage examples
5. **Best Practices** - Recommendations for optimal use

## Contributing to Documentation

When adding new APIs:

1. Create a new `.md` file in the appropriate directory
2. Follow the documentation template
3. Include code examples
4. Add to this index
5. Update type hints in code

For more details on contributing, see [Contributing Guide](../../CONTRIBUTING.md).