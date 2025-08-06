# Google Gemini Provider Documentation

## Overview

The Google provider integrates with Google's Gemini models through the Google Generative AI API. It supports various Gemini model variants with comprehensive configuration options.

## Supported Models

| Model Name | Description | Context Window | Use Case |
|------------|-------------|----------------|----------|
| `gemini-1.5-flash` | Fast, efficient model | 1M tokens | Quick responses, high throughput |
| `gemini-1.5-pro` | Advanced reasoning model | 2M tokens | Complex reasoning, analysis |
| `gemini-1.0-pro` | Balanced performance | 32K tokens | General purpose tasks |

## Setup

### 1. API Key Configuration

Obtain your API key from [Google AI Studio](https://makersuite.google.com/app/apikey):

1. Visit Google AI Studio
2. Create or select a project
3. Generate an API key
4. Add to your `.env` file:

```bash
GOOGLE_API_KEY=your-api-key-here
```

### 2. Provider Initialization

```python
from llm_providers import GoogleProvider

# Basic usage
provider = GoogleProvider(model_name="gemini-1.5-flash")

# With custom configuration
provider = GoogleProvider(
    model_name="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    top_k=40
)
```

## Configuration Parameters

### Required Parameters

- `model_name` (str): The Gemini model to use

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.7 | Controls randomness (0.0-1.0) |
| `max_tokens` | int | 1000 | Maximum tokens to generate |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `top_k` | int | 40 | Top-k sampling limit |
| `stop_sequences` | list | [] | Sequences to stop generation |

### Example Configuration

```python
config = {
    "model_name": "gemini-1.5-flash",
    "temperature": 0.2,  # More deterministic
    "max_tokens": 2000,
    "top_p": 0.95,
    "top_k": 50,
    "stop_sequences": ["\n\n", "END"]
}

provider = GoogleProvider(**config)
```

## Usage Examples

### Basic Text Generation

```python
from llm_providers import GoogleProvider

provider = GoogleProvider(model_name="gemini-1.5-flash")

response = provider.generate("Explain quantum computing in simple terms.")
print(response)
```

### Multi-Turn Conversation

```python
provider = GoogleProvider(model_name="gemini-1.5-pro")

# First message
response1 = provider.generate("What are the benefits of renewable energy?")

# Follow-up (note: Gemini handles context internally)
response2 = provider.generate("Can you elaborate on solar power specifically?")
```

### Batch Processing

```python
from llm_providers import GoogleProvider
import asyncio

async def process_batch():
    provider = GoogleProvider(model_name="gemini-1.5-flash")

    prompts = [
        "Summarize the benefits of AI in healthcare",
        "Explain blockchain technology",
        "Describe the impact of climate change"
    ]

    # Process concurrently
    tasks = [provider.generate_async(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)

    return responses
```

## Rate Limits and Quotas

### Free Tier Limits

- **15 requests per minute** (RPM)
- **1 million tokens per minute** (TPM)
- **1,500 requests per day** (RPD)

### Paid API Limits

- **300 requests per minute** (RPM)
- **2 million tokens per minute** (TPM)
- Higher daily quotas based on billing

### Best Practices

1. **Implement retry logic** with exponential backoff
2. **Monitor quota usage** through Google Cloud Console
3. **Use connection pools** for high-throughput applications
4. **Cache responses** when appropriate

```python
from llm_providers import GoogleProvider
import time
import random

provider = GoogleProvider(model_name="gemini-1.5-flash")

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return provider.generate(prompt)
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise e
```

## Error Handling

### Common Error Types

| Error Code | Description | Solution |
|------------|-------------|----------|
| `429` | Rate limit exceeded | Implement backoff, reduce request rate |
| `400` | Invalid request | Check parameters and prompt format |
| `401` | Authentication failed | Verify API key is correct |
| `403` | Quota exceeded | Upgrade plan or wait for quota reset |
| `500` | Server error | Retry with exponential backoff |

### Error Handling Example

```python
from llm_providers import GoogleProvider
from llm_providers.exceptions import RateLimitError, AuthenticationError

provider = GoogleProvider(model_name="gemini-1.5-flash")

try:
    response = provider.generate("Your prompt here")
except RateLimitError as e:
    print(f"Rate limit hit, retry after: {e.retry_after} seconds")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization

### Model Selection Guidelines

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| High-volume, simple tasks | `gemini-1.5-flash` | Fastest response, lowest cost |
| Complex reasoning | `gemini-1.5-pro` | Best performance for analysis |
| Balanced workloads | `gemini-1.0-pro` | Good performance/cost ratio |

### Optimization Tips

1. **Choose appropriate model** for your use case
2. **Optimize prompt length** - shorter prompts process faster
3. **Use appropriate temperature** - lower for deterministic tasks
4. **Implement caching** for repeated queries
5. **Batch similar requests** when possible

```python
# Optimized configuration for high-throughput scenarios
provider = GoogleProvider(
    model_name="gemini-1.5-flash",
    temperature=0.1,  # More deterministic
    max_tokens=500,   # Shorter responses
    top_p=0.8        # Focused sampling
)
```

## Integration Examples

### With LLM Lab Benchmarking

```python
from llm_providers import GoogleProvider
import json

# Initialize provider
provider = GoogleProvider(model_name="gemini-1.5-flash")

# Run benchmark
results = []
for prompt_data in benchmark_dataset:
    response = provider.generate(prompt_data['prompt'])

    result = {
        'prompt_id': prompt_data['id'],
        'model_name': 'google/gemini-1.5-flash',
        'response': response,
        'timestamp': datetime.now().isoformat()
    }
    results.append(result)

# Save results
with open('benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### With Custom Evaluation

```python
from llm_providers import GoogleProvider
from evaluation import keyword_match

provider = GoogleProvider(model_name="gemini-1.5-pro")

def evaluate_truthfulness(questions):
    results = []

    for question in questions:
        response = provider.generate(question['prompt'])

        evaluation = keyword_match.evaluate(
            prompt=question['prompt'],
            response=response,
            expected_keywords=question['expected_keywords']
        )

        results.append({
            'question_id': question['id'],
            'response': response,
            'score': evaluation['score'],
            'matched_keywords': evaluation['matched_keywords']
        })

    return results
```

## Troubleshooting

### Common Issues

#### 1. Authentication Failures

**Problem**: `401 Unauthorized` errors

**Solutions**:
- Verify API key is correctly set in environment
- Check key hasn't expired or been revoked
- Ensure key has necessary permissions

```bash
# Check environment variable
echo $GOOGLE_API_KEY

# Test API key
curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
     "https://generativelanguage.googleapis.com/v1/models"
```

#### 2. Rate Limiting

**Problem**: `429 Too Many Requests` errors

**Solutions**:
- Implement exponential backoff
- Reduce request rate
- Consider upgrading to paid tier

```python
import time
from functools import wraps

def rate_limit_retry(max_retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
                        continue
                    raise
            return None
        return wrapper
    return decorator

@rate_limit_retry()
def generate_text(prompt):
    return provider.generate(prompt)
```

#### 3. Response Quality Issues

**Problem**: Poor quality or inconsistent responses

**Solutions**:
- Adjust temperature (lower for consistency)
- Improve prompt engineering
- Use appropriate model for task complexity
- Add examples in prompts (few-shot learning)

```python
# Better prompt engineering
def create_structured_prompt(question, context=None):
    prompt = f"""
Answer the following question accurately and concisely.

Question: {question}
"""

    if context:
        prompt += f"\nContext: {context}\n"

    prompt += "\nAnswer:"
    return prompt

# Usage
structured_prompt = create_structured_prompt(
    "What is machine learning?",
    context="Focus on practical applications"
)
response = provider.generate(structured_prompt)
```

#### 4. Quota Exceeded

**Problem**: Daily or monthly quota limits reached

**Solutions**:
- Monitor usage through Google Cloud Console
- Implement usage tracking in your application
- Consider upgrading billing plan
- Optimize requests to reduce token usage

```python
class QuotaTracker:
    def __init__(self, daily_limit=1500):
        self.daily_limit = daily_limit
        self.requests_today = 0
        self.last_reset = datetime.now().date()

    def can_make_request(self):
        today = datetime.now().date()
        if today > self.last_reset:
            self.requests_today = 0
            self.last_reset = today

        return self.requests_today < self.daily_limit

    def record_request(self):
        self.requests_today += 1

tracker = QuotaTracker()

def safe_generate(prompt):
    if not tracker.can_make_request():
        raise Exception("Daily quota exceeded")

    response = provider.generate(prompt)
    tracker.record_request()
    return response
```

## Monitoring and Logging

### Performance Metrics

Track these key metrics for optimal performance:

- **Response Time**: Monitor latency trends
- **Success Rate**: Track failed vs successful requests
- **Token Usage**: Monitor costs and quotas
- **Error Patterns**: Identify common failure modes

```python
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleProviderWithLogging(GoogleProvider):
    def generate(self, prompt):
        start_time = time.time()
        request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        logger.info(f"Request {request_id} started - Model: {self.model_name}")

        try:
            response = super().generate(prompt)
            duration = time.time() - start_time

            logger.info(f"Request {request_id} completed - Duration: {duration:.2f}s")
            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request {request_id} failed - Duration: {duration:.2f}s - Error: {e}")
            raise
```

## Cost Optimization

### Pricing Information

Google Gemini API pricing (as of latest update):

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|---------------------|----------------------|
| Gemini 1.5 Flash | $0.00015 | $0.0006 |
| Gemini 1.5 Pro | $0.00125 | $0.005 |
| Gemini 1.0 Pro | $0.0005 | $0.0015 |

### Cost Reduction Strategies

1. **Choose the right model** for your use case
2. **Optimize prompts** to reduce token usage
3. **Implement caching** for repeated queries
4. **Use shorter max_tokens** when appropriate
5. **Batch similar requests** to reduce overhead

```python
# Cost-optimized configuration
class CostOptimizedGoogleProvider(GoogleProvider):
    def __init__(self, model_name="gemini-1.5-flash", **kwargs):
        # Use most cost-effective model by default
        super().__init__(
            model_name=model_name,
            max_tokens=kwargs.get('max_tokens', 500),  # Limit response length
            temperature=kwargs.get('temperature', 0.1),  # More deterministic
            **kwargs
        )

    def generate_with_cost_tracking(self, prompt):
        # Estimate input tokens (rough approximation)
        input_tokens = len(prompt.split()) * 1.3  # Account for tokenization

        response = self.generate(prompt)

        # Estimate output tokens
        output_tokens = len(response.split()) * 1.3

        # Calculate estimated cost (for gemini-1.5-flash)
        cost = (input_tokens * 0.00015 / 1000) + (output_tokens * 0.0006 / 1000)

        return {
            'response': response,
            'estimated_cost': cost,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }
```

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for configuration
3. **Implement proper access controls**
4. **Rotate API keys** regularly
5. **Monitor usage** for unusual patterns

```python
import os
from pathlib import Path

def load_api_key():
    # Try environment variable first
    api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key:
        # Try .env file
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith('GOOGLE_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        break

    if not api_key:
        raise ValueError("Google API key not found in environment or .env file")

    return api_key

# Usage
try:
    api_key = load_api_key()
    provider = GoogleProvider(model_name="gemini-1.5-flash")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Advanced Features

### Custom Safety Settings

```python
provider = GoogleProvider(
    model_name="gemini-1.5-pro",
    safety_settings={
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE'
    }
)
```

### Function Calling (Tool Use)

```python
# Define functions that the model can call
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
]

provider = GoogleProvider(
    model_name="gemini-1.5-pro",
    functions=functions
)

response = provider.generate("What's the weather like in New York?")
```

This documentation provides comprehensive guidance for using the Google Gemini provider effectively within the LLM Lab framework.
