# OpenAI Provider Documentation

## Overview

The OpenAI provider integrates with OpenAI's GPT models through the OpenAI API. It supports the full range of GPT models with comprehensive configuration options for various use cases.

## Supported Models

| Model Name | Description | Context Window | Use Case |
|------------|-------------|----------------|----------|
| `gpt-4o` | Latest GPT-4 Omni model | 128K tokens | Most capable, multimodal |
| `gpt-4o-mini` | Efficient GPT-4 variant | 128K tokens | Fast, cost-effective |
| `gpt-4-turbo` | Enhanced GPT-4 | 128K tokens | Advanced reasoning |
| `gpt-4` | Standard GPT-4 | 8K tokens | High-quality responses |
| `gpt-3.5-turbo` | Fast and efficient | 16K tokens | General purpose, cost-effective |

## Setup

### 1. API Key Configuration

Obtain your API key from [OpenAI API Keys](https://platform.openai.com/api-keys):

1. Visit OpenAI Platform
2. Create an account or sign in
3. Navigate to API Keys section
4. Generate a new API key
5. Add to your `.env` file:

```bash
OPENAI_API_KEY=your-api-key-here
```

### 2. Provider Initialization

```python
from llm_providers import OpenAIProvider

# Basic usage
provider = OpenAIProvider(model_name="gpt-4o-mini")

# With custom configuration
provider = OpenAIProvider(
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1
)
```

## Configuration Parameters

### Required Parameters

- `model_name` (str): The GPT model to use

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.7 | Controls randomness (0.0-2.0) |
| `max_tokens` | int | 1000 | Maximum tokens to generate |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `frequency_penalty` | float | 0.0 | Reduce repetition (-2.0 to 2.0) |
| `presence_penalty` | float | 0.0 | Encourage topic diversity (-2.0 to 2.0) |
| `stop` | list | [] | Stop sequences |
| `stream` | bool | False | Enable streaming responses |

### Example Configuration

```python
config = {
    "model_name": "gpt-4o-mini",
    "temperature": 0.2,  # More deterministic
    "max_tokens": 2000,
    "top_p": 0.95,
    "frequency_penalty": 0.1,  # Reduce repetition
    "presence_penalty": 0.1,   # Encourage diversity
    "stop": ["\n\n", "###"]
}

provider = OpenAIProvider(**config)
```

## Usage Examples

### Basic Text Generation

```python
from llm_providers import OpenAIProvider

provider = OpenAIProvider(model_name="gpt-4o-mini")

response = provider.generate("Explain the concept of machine learning in simple terms.")
print(response)
```

### System Messages and Chat Format

```python
provider = OpenAIProvider(model_name="gpt-4o")

# Using system message for context
response = provider.generate(
    prompt="What are the benefits of renewable energy?",
    system_message="You are an environmental science expert. Provide accurate, well-researched information."
)
```

### Multi-Turn Conversation

```python
provider = OpenAIProvider(model_name="gpt-4o-mini")

# Conversation history
conversation = [
    {"role": "system", "content": "You are a helpful programming assistant."},
    {"role": "user", "content": "How do I create a Python function?"},
    {"role": "assistant", "content": "Here's how to create a Python function..."},
    {"role": "user", "content": "Can you show me an example with parameters?"}
]

response = provider.generate_chat(conversation)
```

### Streaming Responses

```python
provider = OpenAIProvider(
    model_name="gpt-4o-mini",
    stream=True
)

def stream_response(prompt):
    for chunk in provider.generate_stream(prompt):
        print(chunk, end='', flush=True)
    print()  # New line at end

stream_response("Write a short story about AI.")
```

### Batch Processing with Async

```python
import asyncio
from llm_providers import OpenAIProvider

async def process_batch():
    provider = OpenAIProvider(model_name="gpt-4o-mini")
    
    prompts = [
        "Summarize the benefits of cloud computing",
        "Explain blockchain in simple terms",
        "Describe the future of renewable energy"
    ]
    
    # Process concurrently
    tasks = [provider.generate_async(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    
    return responses

# Run batch processing
responses = asyncio.run(process_batch())
```

## Rate Limits and Quotas

### Rate Limits by Model Tier

#### Free Tier
- **3 RPM** (requests per minute)
- **40,000 TPM** (tokens per minute)
- **200 RPD** (requests per day)

#### Tier 1 ($5+ in usage)
- **3,500 RPM**
- **90,000 TPM**
- **10,000 RPD**

#### Tier 2 ($50+ in usage)
- **5,000 RPM**
- **450,000 TPM**
- **Unlimited RPD**

### Best Practices for Rate Limiting

```python
from llm_providers import OpenAIProvider
from llm_providers.exceptions import RateLimitError
import time
import random

provider = OpenAIProvider(model_name="gpt-4o-mini")

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return provider.generate(prompt)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                # Extract retry-after from headers if available
                retry_after = getattr(e, 'retry_after', None)
                if retry_after:
                    time.sleep(retry_after)
                else:
                    # Exponential backoff with jitter
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                continue
            raise e

# Usage
response = generate_with_retry("Your prompt here")
```

### Concurrent Request Management

```python
import asyncio
from asyncio import Semaphore

class RateLimitedOpenAIProvider(OpenAIProvider):
    def __init__(self, max_concurrent=5, **kwargs):
        super().__init__(**kwargs)
        self.semaphore = Semaphore(max_concurrent)
    
    async def generate_async_limited(self, prompt):
        async with self.semaphore:
            return await self.generate_async(prompt)

# Usage
provider = RateLimitedOpenAIProvider(
    model_name="gpt-4o-mini",
    max_concurrent=3
)
```

## Error Handling

### Common Error Types

| Error Code | Description | Solution |
|------------|-------------|----------|
| `429` | Rate limit exceeded | Implement backoff, reduce request rate |
| `400` | Invalid request | Check parameters and prompt format |
| `401` | Invalid API key | Verify API key is correct |
| `403` | Forbidden/Quota exceeded | Check billing, upgrade plan |
| `500` | Server error | Retry with exponential backoff |
| `503` | Service unavailable | Wait and retry |

### Comprehensive Error Handling

```python
from llm_providers import OpenAIProvider
from llm_providers.exceptions import (
    RateLimitError, 
    AuthenticationError, 
    InvalidRequestError,
    ServiceUnavailableError
)

provider = OpenAIProvider(model_name="gpt-4o-mini")

def robust_generate(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return provider.generate(prompt)
            
        except RateLimitError as e:
            print(f"Rate limit hit, retry after: {e.retry_after} seconds")
            if e.retry_after:
                time.sleep(e.retry_after)
            else:
                time.sleep(2 ** attempt)
                
        except AuthenticationError as e:
            print(f"Authentication failed: {e}")
            raise  # Don't retry auth errors
            
        except InvalidRequestError as e:
            print(f"Invalid request: {e}")
            raise  # Don't retry invalid requests
            
        except ServiceUnavailableError as e:
            print(f"Service unavailable, retrying in {2 ** attempt} seconds")
            time.sleep(2 ** attempt)
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    raise Exception("Max retries exceeded")
```

## Performance Optimization

### Model Selection Guidelines

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| Development/Testing | `gpt-3.5-turbo` | Fast, cost-effective |
| General Production | `gpt-4o-mini` | Balance of capability and cost |
| Complex Reasoning | `gpt-4o` | Highest capability |
| High Volume/Simple | `gpt-3.5-turbo` | Lowest cost per token |

### Optimization Strategies

```python
# Performance-optimized configuration
provider = OpenAIProvider(
    model_name="gpt-4o-mini",
    temperature=0.1,    # More deterministic = faster
    max_tokens=500,     # Shorter responses = faster
    top_p=0.8,         # Focused sampling
    frequency_penalty=0.2,  # Reduce repetition
)

# Prompt optimization for speed
def optimize_prompt(base_prompt):
    """Create concise, clear prompts for faster processing"""
    return f"""Answer concisely:

{base_prompt}

Response:"""

# Token usage optimization
def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 0.75 words)"""
    return int(len(text.split()) * 1.33)

def optimize_for_tokens(prompt, max_input_tokens=3000):
    """Truncate prompt if too long"""
    estimated_tokens = estimate_tokens(prompt)
    if estimated_tokens > max_input_tokens:
        # Truncate to fit within limits
        words = prompt.split()
        target_words = int(max_input_tokens * 0.75)
        prompt = ' '.join(words[:target_words])
    return prompt
```

## Integration Examples

### With LLM Lab Benchmarking

```python
from llm_providers import OpenAIProvider
import json
from datetime import datetime

# Initialize provider
provider = OpenAIProvider(model_name="gpt-4o-mini")

def run_benchmark(dataset, output_file):
    results = []
    
    for i, prompt_data in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}")
        
        try:
            start_time = time.time()
            response = provider.generate(prompt_data['prompt'])
            end_time = time.time()
            
            result = {
                'prompt_id': prompt_data['id'],
                'model_name': 'openai/gpt-4o-mini',
                'prompt': prompt_data['prompt'],
                'response': response,
                'response_time': end_time - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            result = {
                'prompt_id': prompt_data['id'],
                'model_name': 'openai/gpt-4o-mini',
                'prompt': prompt_data['prompt'],
                'response': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
        
        results.append(result)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
```

### Function Calling (Tool Use)

```python
provider = OpenAIProvider(model_name="gpt-4o")

# Define functions the model can call
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Example with function calling
response = provider.generate_with_functions(
    prompt="What's the weather like in London?",
    functions=functions
)

# Handle function calls
if response.get("function_call"):
    function_name = response["function_call"]["name"]
    function_args = json.loads(response["function_call"]["arguments"])
    
    if function_name == "get_weather":
        # Call your actual weather function
        weather_data = get_weather(function_args["location"])
        
        # Send function result back to model
        final_response = provider.generate_with_function_result(
            function_call=response["function_call"],
            function_result=weather_data
        )
```

## Cost Optimization

### Pricing Information (as of latest update)

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|---------------------|----------------------|
| GPT-4o | $0.005 | $0.015 |
| GPT-4o-mini | $0.00015 | $0.0006 |
| GPT-4-turbo | $0.01 | $0.03 |
| GPT-3.5-turbo | $0.001 | $0.002 |

### Cost Reduction Strategies

```python
class CostOptimizedOpenAIProvider(OpenAIProvider):
    def __init__(self, budget_per_day=10.0, **kwargs):
        # Default to most cost-effective model
        if 'model_name' not in kwargs:
            kwargs['model_name'] = 'gpt-4o-mini'
        
        super().__init__(**kwargs)
        self.budget_per_day = budget_per_day
        self.daily_spend = 0.0
        self.last_reset = datetime.now().date()
    
    def generate_with_cost_tracking(self, prompt):
        # Reset daily spend if new day
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_spend = 0.0
            self.last_reset = today
        
        # Check budget
        if self.daily_spend >= self.budget_per_day:
            raise Exception("Daily budget exceeded")
        
        # Estimate cost before making request
        input_tokens = self.estimate_tokens(prompt)
        estimated_input_cost = input_tokens * self.get_input_price() / 1000
        
        if self.daily_spend + estimated_input_cost > self.budget_per_day:
            raise Exception("Request would exceed daily budget")
        
        # Make request
        response = self.generate(prompt)
        
        # Calculate actual cost (approximation)
        output_tokens = self.estimate_tokens(response)
        actual_cost = (
            input_tokens * self.get_input_price() / 1000 +
            output_tokens * self.get_output_price() / 1000
        )
        
        self.daily_spend += actual_cost
        
        return {
            'response': response,
            'cost': actual_cost,
            'daily_spend': self.daily_spend,
            'budget_remaining': self.budget_per_day - self.daily_spend
        }
    
    def get_input_price(self):
        prices = {
            'gpt-4o': 0.005,
            'gpt-4o-mini': 0.00015,
            'gpt-4-turbo': 0.01,
            'gpt-3.5-turbo': 0.001
        }
        return prices.get(self.model_name, 0.005)
    
    def get_output_price(self):
        prices = {
            'gpt-4o': 0.015,
            'gpt-4o-mini': 0.0006,
            'gpt-4-turbo': 0.03,
            'gpt-3.5-turbo': 0.002
        }
        return prices.get(self.model_name, 0.015)
```

## Advanced Features

### Custom Fine-Tuned Models

```python
# Using a fine-tuned model
provider = OpenAIProvider(model_name="ft:gpt-3.5-turbo-1106:your-org:custom-model:abc123")

response = provider.generate("Your prompt here")
```

### Image Input (GPT-4 Vision)

```python
provider = OpenAIProvider(model_name="gpt-4o")

def analyze_image(image_path, question):
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }
    ]
    
    return provider.generate_chat(messages)

# Usage
response = analyze_image("chart.png", "What trends do you see in this chart?")
```

### JSON Mode

```python
provider = OpenAIProvider(
    model_name="gpt-4o-mini",
    response_format="json_object"
)

response = provider.generate("""
Extract the following information from this text and return as JSON:
- Name
- Age  
- Occupation

Text: "John Smith is a 35-year-old software engineer working at Tech Corp."
""")

# Response will be valid JSON
import json
data = json.loads(response)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. API Key Issues

```bash
# Test your API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Check environment variable
echo $OPENAI_API_KEY
```

#### 2. Token Limit Exceeded

```python
def handle_token_limits(prompt, max_tokens=4000):
    """Handle prompts that exceed token limits"""
    estimated_tokens = len(prompt.split()) * 1.33
    
    if estimated_tokens > max_tokens:
        # Truncate or split the prompt
        words = prompt.split()
        max_words = int(max_tokens * 0.75)
        
        if len(words) > max_words:
            truncated = ' '.join(words[:max_words])
            return truncated + "\n[Content truncated...]"
    
    return prompt
```

#### 3. Model Access Issues

```python
def check_model_access():
    """Verify you have access to specific models"""
    provider = OpenAIProvider(model_name="gpt-4o-mini")
    
    try:
        response = provider.generate("Test message")
        return True
    except Exception as e:
        if "model" in str(e).lower():
            print(f"Model access issue: {e}")
            return False
        raise e
```

#### 4. Quota and Billing Issues

```python
import requests

def check_usage():
    """Check current usage (requires API key with appropriate permissions)"""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    
    response = requests.get(
        "https://api.openai.com/v1/usage",
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error checking usage: {response.status_code}")
        return None
```

## Security and Best Practices

### API Key Security

```python
import os
from pathlib import Path
import keyring  # Optional: for secure key storage

class SecureOpenAIProvider(OpenAIProvider):
    def __init__(self, **kwargs):
        api_key = self.load_secure_api_key()
        super().__init__(api_key=api_key, **kwargs)
    
    def load_secure_api_key(self):
        """Load API key securely"""
        # Try environment variable first
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            return api_key
        
        # Try keyring (macOS Keychain, Windows Credential Store, etc.)
        try:
            api_key = keyring.get_password("openai", "api_key")
            if api_key:
                return api_key
        except:
            pass
        
        # Try .env file as last resort
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        return line.split('=', 1)[1].strip()
        
        raise ValueError("OpenAI API key not found")
```

### Content Filtering

```python
def safe_generate(provider, prompt):
    """Generate with content safety checks"""
    
    # Pre-filter prompts
    if any(word in prompt.lower() for word in ['harmful', 'illegal', 'violence']):
        return "I can't help with that request."
    
    try:
        response = provider.generate(prompt)
        
        # Post-filter responses if needed
        if any(word in response.lower() for word in ['dangerous', 'harmful']):
            return "I cannot provide that information."
        
        return response
        
    except Exception as e:
        # Log security-related errors
        if "content_policy" in str(e).lower():
            print("Content policy violation detected")
        raise e
```

### Usage Monitoring

```python
import logging
from datetime import datetime

class MonitoredOpenAIProvider(OpenAIProvider):
    def __init__(self, log_file='openai_usage.log', **kwargs):
        super().__init__(**kwargs)
        
        # Setup logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def generate(self, prompt):
        start_time = datetime.now()
        prompt_length = len(prompt)
        
        try:
            response = super().generate(prompt)
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"SUCCESS - Model: {self.model_name}, "
                           f"Prompt length: {prompt_length}, "
                           f"Response length: {len(response)}, "
                           f"Duration: {duration:.2f}s")
            
            return response
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.error(f"ERROR - Model: {self.model_name}, "
                            f"Prompt length: {prompt_length}, "
                            f"Duration: {duration:.2f}s, "
                            f"Error: {str(e)}")
            raise e
```

This comprehensive documentation covers all aspects of using the OpenAI provider effectively within the LLM Lab framework.