# Providers API Reference

## Overview

The providers module provides a unified interface for interacting with various LLM providers including OpenAI, Anthropic, Google, and more.

## Base Provider

### `BaseProvider`

Abstract base class for all LLM providers.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseProvider(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from the model.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            **kwargs: Provider-specific parameters

        Returns:
            Dictionary containing:
                - text: Generated text
                - usage: Token usage statistics
                - model: Model identifier
                - finish_reason: Why generation stopped
        """
        pass
```

## Provider Implementations

### `OpenAIProvider`

OpenAI GPT models provider.

```python
from src.providers import OpenAIProvider

# Initialize
provider = OpenAIProvider(
    api_key="your-api-key",  # Optional if in env
    model="gpt-4",
    timeout=30
)

# Generate text
response = provider.generate(
    prompt="Explain quantum computing",
    max_tokens=200,
    temperature=0.8
)
```

**Supported Models:**
- `gpt-4` - Most capable model
- `gpt-4-turbo` - Faster, more affordable GPT-4
- `gpt-3.5-turbo` - Fast and inexpensive
- `gpt-4o-mini` - Small, efficient model

**Additional Parameters:**
- `top_p`: Nucleus sampling parameter
- `frequency_penalty`: Reduce repetition (-2 to 2)
- `presence_penalty`: Encourage new topics (-2 to 2)
- `response_format`: Specify JSON output format

### `AnthropicProvider`

Anthropic Claude models provider.

```python
from src.providers import AnthropicProvider

provider = AnthropicProvider(
    api_key="your-api-key",
    model="claude-3-5-sonnet-20241022"
)

response = provider.generate(
    prompt="Write a haiku about programming",
    max_tokens=50
)
```

**Supported Models:**
- `claude-3-5-sonnet-20241022` - Most intelligent model
- `claude-3-opus-20240229` - Powerful model for complex tasks
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-haiku-20240307` - Fast and compact

### `GoogleProvider`

Google Gemini models provider.

```python
from src.providers import GoogleProvider

provider = GoogleProvider(
    api_key="your-api-key",
    model="gemini-1.5-pro"
)

response = provider.generate(
    prompt="Analyze this image: [image description]",
    max_tokens=300
)
```

**Supported Models:**
- `gemini-1.5-pro` - Most capable multimodal model
- `gemini-1.5-flash` - Fast multimodal model
- `gemini-1.0-pro` - Text-only model

## Provider Registry

### `ProviderRegistry`

Central registry for managing providers.

```python
from src.providers import ProviderRegistry

# Get registry instance
registry = ProviderRegistry()

# Register custom provider
registry.register("custom", CustomProvider)

# List available providers
providers = registry.list_providers()
# ['openai', 'anthropic', 'google', 'azure', 'custom']

# Create provider instance
provider = registry.create_provider(
    "openai",
    model="gpt-4",
    **config
)
```

## Error Handling

### Provider Exceptions

```python
from src.providers.exceptions import (
    ProviderError,          # Base exception
    AuthenticationError,    # Invalid API key
    RateLimitError,        # Rate limit exceeded
    ModelNotFoundError,    # Invalid model name
    ProviderTimeoutError,  # Request timeout
    InvalidRequestError    # Invalid parameters
)

try:
    response = provider.generate(prompt)
except RateLimitError as e:
    print(f"Rate limit hit: {e.retry_after} seconds")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Advanced Features

### Streaming Responses

```python
# Stream tokens as they're generated
for chunk in provider.stream_generate(prompt):
    print(chunk.text, end="", flush=True)
```

### Batch Processing

```python
prompts = ["Question 1", "Question 2", "Question 3"]
responses = provider.batch_generate(prompts, max_tokens=100)
```

### Cost Estimation

```python
# Estimate cost before making request
cost = provider.estimate_cost(
    prompt_tokens=150,
    completion_tokens=200
)
print(f"Estimated cost: ${cost:.4f}")

# Get actual cost after generation
response = provider.generate(prompt)
print(f"Actual cost: ${response['cost']:.4f}")
```

### Model Information

```python
# Get model details
info = provider.get_model_info()
print(f"Context window: {info['context_window']}")
print(f"Max output tokens: {info['max_tokens']}")
print(f"Cost per 1K tokens: ${info['cost_per_1k_tokens']}")
```

## Best Practices

1. **API Key Management**
   ```python
   # Use environment variables
   os.environ['OPENAI_API_KEY'] = 'your-key'
   provider = OpenAIProvider()  # Auto-loads from env
   ```

2. **Error Handling**
   ```python
   # Always handle provider errors
   try:
       response = provider.generate(prompt)
   except RateLimitError:
       # Implement exponential backoff
       time.sleep(retry_after)
   except ProviderError:
       # Use fallback provider
       response = fallback_provider.generate(prompt)
   ```

3. **Resource Management**
   ```python
   # Use context managers for cleanup
   with OpenAIProvider() as provider:
       response = provider.generate(prompt)
   ```

4. **Performance Optimization**
   ```python
   # Cache frequently used responses
   from functools import lru_cache

   @lru_cache(maxsize=100)
   def cached_generate(prompt: str) -> str:
       return provider.generate(prompt)['text']
   ```

## See Also

- [Configuration API](configuration.md) - Provider configuration
- [Evaluation API](evaluation.md) - Evaluating provider responses
- [Examples](../../examples/providers/) - Provider usage examples
