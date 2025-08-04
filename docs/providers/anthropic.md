# Anthropic Claude Provider Documentation

## Overview

The Anthropic provider integrates with Anthropic's Claude models through the Anthropic API. Claude models are known for their helpful, harmless, and honest responses with strong reasoning capabilities and large context windows.

## Supported Models

| Model Name | Description | Context Window | Use Case |
|------------|-------------|----------------|----------|
| `claude-3-5-sonnet-20241022` | Latest Claude 3.5 Sonnet | 200K tokens | Most capable, balanced performance |
| `claude-3-5-haiku-20241022` | Fast Claude 3.5 variant | 200K tokens | Quick responses, cost-effective |
| `claude-3-opus-20240229` | Highest capability Claude 3 | 200K tokens | Complex reasoning, analysis |
| `claude-3-sonnet-20240229` | Balanced Claude 3 | 200K tokens | General purpose tasks |
| `claude-3-haiku-20240307` | Fastest Claude 3 | 200K tokens | Simple tasks, high throughput |

## Setup

### 1. API Key Configuration

Obtain your API key from [Anthropic Console](https://console.anthropic.com/):

1. Visit Anthropic Console
2. Create an account or sign in
3. Navigate to API Keys section
4. Generate a new API key
5. Add to your `.env` file:

```bash
ANTHROPIC_API_KEY=your-api-key-here
```

### 2. Provider Initialization

```python
from llm_providers import AnthropicProvider

# Basic usage
provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022")

# With custom configuration
provider = AnthropicProvider(
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    top_k=40
)
```

## Configuration Parameters

### Required Parameters

- `model_name` (str): The Claude model to use

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.7 | Controls randomness (0.0-1.0) |
| `max_tokens` | int | 1000 | Maximum tokens to generate |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `top_k` | int | 40 | Top-k sampling limit |
| `stop_sequences` | list | [] | Sequences to stop generation |
| `stream` | bool | False | Enable streaming responses |

### Example Configuration

```python
config = {
    "model_name": "claude-3-5-sonnet-20241022",
    "temperature": 0.3,  # More focused responses
    "max_tokens": 2000,
    "top_p": 0.95,
    "top_k": 50,
    "stop_sequences": ["\n\nHuman:", "\n\nAssistant:"]
}

provider = AnthropicProvider(**config)
```

## Usage Examples

### Basic Text Generation

```python
from llm_providers import AnthropicProvider

provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022")

response = provider.generate("Explain the principles of sustainable development.")
print(response)
```

### System Messages and Structured Prompts

```python
provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022")

# Using system message for context
response = provider.generate(
    prompt="What are the key considerations for database design?",
    system_message="You are a database architecture expert with 15 years of experience. Provide practical, actionable advice."
)
```

### Multi-Turn Conversation

```python
provider = AnthropicProvider(model_name="claude-3-5-haiku-20241022")

# Conversation with proper format
messages = [
    {"role": "user", "content": "I'm learning Python. Can you help me understand functions?"},
    {"role": "assistant", "content": "I'd be happy to help you understand Python functions! Functions are reusable blocks of code..."},
    {"role": "user", "content": "Can you show me an example with parameters?"}
]

response = provider.generate_chat(messages)
```

### Streaming Responses

```python
provider = AnthropicProvider(
    model_name="claude-3-5-haiku-20241022",
    stream=True
)

def stream_response(prompt):
    for chunk in provider.generate_stream(prompt):
        print(chunk, end='', flush=True)
    print()  # New line at end

stream_response("Write a detailed explanation of machine learning.")
```

### Long Context Processing

```python
# Claude excels at processing long documents
provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022")

def analyze_long_document(document_path, analysis_prompt):
    with open(document_path, 'r') as f:
        document_content = f.read()
    
    # Claude can handle up to 200K tokens
    full_prompt = f"""
Please analyze the following document:

<document>
{document_content}
</document>

Analysis request: {analysis_prompt}
"""
    
    return provider.generate(full_prompt)

# Usage
analysis = analyze_long_document(
    "research_paper.txt", 
    "Summarize the key findings and methodology"
)
```

## Rate Limits and Quotas

### Rate Limits by Plan

#### Free Tier
- **5 RPM** (requests per minute)
- **25,000 TPM** (tokens per minute)
- **$5 monthly credit**

#### Pro Plan ($20/month)
- **1,000 RPM**
- **100,000 TPM**
- **$25 monthly credit + usage**

#### Team Plan ($25/month per member)
- **1,000 RPM**
- **100,000 TPM**
- **$30 monthly credit + usage**

### Best Practices for Rate Limiting

```python
from llm_providers import AnthropicProvider
from llm_providers.exceptions import RateLimitError
import time
import random

provider = AnthropicProvider(model_name="claude-3-5-haiku-20241022")

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return provider.generate(prompt)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                # Claude rate limits include retry-after header
                retry_after = getattr(e, 'retry_after', None)
                if retry_after:
                    time.sleep(retry_after)
                else:
                    # Exponential backoff
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                continue
            raise e

# Usage with automatic retry
response = generate_with_retry("Your prompt here")
```

## Error Handling

### Common Error Types

| Error Code | Description | Solution |
|------------|-------------|----------|
| `429` | Rate limit exceeded | Implement backoff, check retry-after header |
| `400` | Invalid request | Check parameters and message format |
| `401` | Invalid API key | Verify API key is correct |
| `403` | Forbidden | Check billing status, model access |
| `500` | Server error | Retry with exponential backoff |
| `529` | Overloaded | Wait longer, reduce request rate |

### Comprehensive Error Handling

```python
from llm_providers import AnthropicProvider
from llm_providers.exceptions import (
    RateLimitError, 
    AuthenticationError, 
    InvalidRequestError,
    ServerError
)

provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022")

def robust_generate(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return provider.generate(prompt)
            
        except RateLimitError as e:
            if e.retry_after:
                print(f"Rate limited, waiting {e.retry_after} seconds")
                time.sleep(e.retry_after)
            else:
                time.sleep(2 ** attempt)
                
        except AuthenticationError as e:
            print(f"Authentication failed: {e}")
            raise  # Don't retry auth errors
            
        except InvalidRequestError as e:
            print(f"Invalid request: {e}")
            # Check if it's a fixable issue
            if "max_tokens" in str(e).lower():
                print("Reducing max_tokens and retrying...")
                provider.max_tokens = min(provider.max_tokens, 1000)
                continue
            raise
            
        except ServerError as e:
            print(f"Server error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
            
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
| Quick Q&A, simple tasks | `claude-3-haiku-20240307` | Fastest, most cost-effective |
| General production use | `claude-3-5-haiku-20241022` | Good balance of speed and capability |
| Complex analysis | `claude-3-5-sonnet-20241022` | Best overall performance |
| Highest quality needs | `claude-3-opus-20240229` | Maximum capability |

### Optimization Strategies

```python
# Performance-optimized configuration
provider = AnthropicProvider(
    model_name="claude-3-5-haiku-20241022",
    temperature=0.1,    # More deterministic
    max_tokens=1000,    # Appropriate limit
    top_p=0.8          # Focused sampling
)

# Efficient prompt structure for Claude
def create_efficient_prompt(task, context=None):
    """Create well-structured prompts for Claude"""
    prompt = f"<task>\n{task}\n</task>\n"
    
    if context:
        prompt += f"\n<context>\n{context}\n</context>\n"
    
    prompt += "\nPlease provide a clear, concise response:"
    return prompt

# Token usage optimization
def optimize_for_long_context(text, max_context_tokens=150000):
    """Optimize text for Claude's long context window"""
    # Rough token estimation (1 token ≈ 0.75 words for Claude)
    estimated_tokens = len(text.split()) * 1.33
    
    if estimated_tokens > max_context_tokens:
        # Truncate but preserve structure
        words = text.split()
        target_words = int(max_context_tokens * 0.75)
        
        # Try to keep beginning and end
        keep_start = target_words // 2
        keep_end = target_words - keep_start
        
        if len(words) > target_words:
            truncated = (
                ' '.join(words[:keep_start]) + 
                '\n\n[... content truncated ...]\n\n' +
                ' '.join(words[-keep_end:])
            )
            return truncated
    
    return text
```

## Integration Examples

### With LLM Lab Benchmarking

```python
from llm_providers import AnthropicProvider
import json
from datetime import datetime
import time

def run_claude_benchmark(dataset, model_name="claude-3-5-haiku-20241022"):
    provider = AnthropicProvider(model_name=model_name)
    results = []
    
    for i, prompt_data in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)} with {model_name}")
        
        try:
            start_time = time.time()
            response = provider.generate(prompt_data['prompt'])
            end_time = time.time()
            
            result = {
                'prompt_id': prompt_data['id'],
                'model_name': f'anthropic/{model_name}',
                'prompt': prompt_data['prompt'],
                'response': response,
                'response_time': end_time - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'provider': 'anthropic'
            }
            
        except Exception as e:
            result = {
                'prompt_id': prompt_data['id'],
                'model_name': f'anthropic/{model_name}',
                'prompt': prompt_data['prompt'],
                'response': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'provider': 'anthropic'
            }
        
        results.append(result)
        
        # Rate limiting - Claude free tier is 5 RPM
        time.sleep(12)  # Wait 12 seconds between requests
    
    return results

# Compare multiple Claude models
models = [
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022", 
    "claude-3-5-sonnet-20241022"
]

all_results = []
for model in models:
    model_results = run_claude_benchmark(dataset, model)
    all_results.extend(model_results)

# Save comprehensive results
with open('claude_benchmark_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
```

### Tool Use (Function Calling)

```python
provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022")

# Define tools Claude can use
tools = [
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for current information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
]

# Generate with tool use
def generate_with_tools(prompt):
    response = provider.generate_with_tools(
        prompt=prompt,
        tools=tools,
        max_tokens=2000
    )
    
    # Handle tool use in response
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_input = tool_call['input']
            
            if tool_name == "calculator":
                # Execute calculation
                result = eval(tool_input['expression'])  # In production, use safer evaluation
                print(f"Calculator: {tool_input['expression']} = {result}")
                
            elif tool_name == "web_search":
                # Execute web search
                print(f"Searching for: {tool_input['query']}")
                # result = perform_web_search(tool_input['query'])
    
    return response.content

# Usage
response = generate_with_tools("What's 15% of 2,450, and can you find recent news about AI developments?")
```

## Cost Optimization

### Pricing Information (as of latest update)

| Model | Input (per MTok) | Output (per MTok) |
|-------|------------------|-------------------|
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3.5 Haiku | $0.25 | $1.25 |
| Claude 3 Opus | $15.00 | $75.00 |
| Claude 3 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |

### Cost Reduction Strategies

```python
class CostOptimizedAnthropicProvider(AnthropicProvider):
    def __init__(self, budget_per_day=5.0, **kwargs):
        # Default to most cost-effective model
        if 'model_name' not in kwargs:
            kwargs['model_name'] = 'claude-3-5-haiku-20241022'
        
        super().__init__(**kwargs)
        self.budget_per_day = budget_per_day
        self.daily_spend = 0.0
        self.last_reset = datetime.now().date()
    
    def estimate_cost(self, prompt, response_length_estimate=500):
        """Estimate cost for a request"""
        input_tokens = len(prompt.split()) * 1.33
        output_tokens = response_length_estimate
        
        # Cost per million tokens
        costs = {
            'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
            'claude-3-5-haiku-20241022': {'input': 0.25, 'output': 1.25},
            'claude-3-opus-20240229': {'input': 15.00, 'output': 75.00},
            'claude-3-sonnet-20240229': {'input': 3.00, 'output': 15.00},
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25}
        }
        
        model_costs = costs.get(self.model_name, costs['claude-3-5-haiku-20241022'])
        
        input_cost = (input_tokens / 1_000_000) * model_costs['input']
        output_cost = (output_tokens / 1_000_000) * model_costs['output']
        
        return input_cost + output_cost
    
    def generate_with_budget_control(self, prompt, max_response_tokens=1000):
        # Reset daily spend if new day
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_spend = 0.0
            self.last_reset = today
        
        # Estimate cost
        estimated_cost = self.estimate_cost(prompt, max_response_tokens)
        
        if self.daily_spend + estimated_cost > self.budget_per_day:
            raise Exception(f"Request would exceed daily budget. Estimated cost: ${estimated_cost:.4f}")
        
        # Set max_tokens to control output cost
        original_max_tokens = self.max_tokens
        self.max_tokens = min(self.max_tokens, max_response_tokens)
        
        try:
            response = self.generate(prompt)
            
            # Calculate actual cost (approximation)
            actual_cost = self.estimate_cost(prompt, len(response.split()) * 1.33)
            self.daily_spend += actual_cost
            
            return {
                'response': response,
                'cost': actual_cost,
                'daily_spend': self.daily_spend,
                'budget_remaining': self.budget_per_day - self.daily_spend
            }
            
        finally:
            # Restore original max_tokens
            self.max_tokens = original_max_tokens
```

## Advanced Features

### Document Analysis

```python
provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022")

def analyze_document(file_path, analysis_type="summary"):
    """Analyze documents using Claude's long context window"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    analysis_prompts = {
        "summary": "Provide a comprehensive summary of this document, highlighting key points and main arguments.",
        "questions": "Generate 10 thoughtful questions that this document answers or raises.",
        "critique": "Provide a balanced critique of the arguments and evidence presented in this document.",
        "extract": "Extract all important facts, figures, dates, and names from this document."
    }
    
    prompt = f"""
<document>
{content}
</document>

{analysis_prompts.get(analysis_type, analysis_prompts["summary"])}
"""
    
    return provider.generate(prompt)

# Usage
summary = analyze_document("research_paper.pdf", "summary")
questions = analyze_document("research_paper.pdf", "questions")
```

### Structured Output

```python
provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022")

def extract_structured_data(text):
    """Extract structured information using Claude"""
    prompt = f"""
Please extract structured information from the following text and format it as JSON:

<text>
{text}
</text>

Extract:
- People (names and roles)
- Organizations
- Dates
- Locations
- Key facts

Format as JSON with these keys: people, organizations, dates, locations, facts.
"""
    
    response = provider.generate(prompt)
    
    # Claude often provides JSON wrapped in markdown code blocks
    if "```json" in response:
        json_str = response.split("```json")[1].split("```")[0].strip()
    else:
        json_str = response
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"raw_response": response, "parsed": False}

# Usage
data = extract_structured_data("John Smith, CEO of TechCorp, announced on March 15, 2024, that the company would expand to New York.")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. API Key Issues

```bash
# Test your API key
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-haiku-20240307",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

#### 2. Message Format Issues

```python
def validate_claude_messages(messages):
    """Validate message format for Claude API"""
    valid_roles = {'user', 'assistant'}
    
    for i, message in enumerate(messages):
        if 'role' not in message:
            raise ValueError(f"Message {i} missing 'role' field")
        
        if message['role'] not in valid_roles:
            raise ValueError(f"Invalid role in message {i}: {message['role']}")
        
        if 'content' not in message:
            raise ValueError(f"Message {i} missing 'content' field")
        
        if not isinstance(message['content'], str):
            raise ValueError(f"Message {i} content must be string")
    
    # Check alternating pattern
    for i in range(1, len(messages)):
        if messages[i]['role'] == messages[i-1]['role']:
            print(f"Warning: Non-alternating roles at messages {i-1} and {i}")
    
    return True

# Usage
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]

validate_claude_messages(messages)
```

#### 3. Context Length Management

```python
def manage_context_length(messages, max_tokens=180000):
    """Manage conversation length for Claude's context window"""
    
    def estimate_tokens(text):
        return len(text.split()) * 1.33
    
    total_tokens = sum(estimate_tokens(msg['content']) for msg in messages)
    
    if total_tokens <= max_tokens:
        return messages
    
    # Keep system message and recent messages
    result = []
    if messages and messages[0].get('role') == 'system':
        result.append(messages[0])
        messages = messages[1:]
    
    # Keep most recent messages that fit
    current_tokens = sum(estimate_tokens(msg['content']) for msg in result)
    
    for message in reversed(messages):
        msg_tokens = estimate_tokens(message['content'])
        if current_tokens + msg_tokens <= max_tokens:
            result.insert(-len([m for m in result if m.get('role') != 'system']), message)
            current_tokens += msg_tokens
        else:
            break
    
    return result
```

#### 4. Rate Limit Optimization

```python
import asyncio
from datetime import datetime, timedelta

class ClaudeRateLimiter:
    def __init__(self, requests_per_minute=5):
        self.requests_per_minute = requests_per_minute
        self.requests = []
    
    async def wait_if_needed(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < timedelta(minutes=1)]
        
        if len(self.requests) >= self.requests_per_minute:
            # Wait until the oldest request is more than 1 minute old
            oldest_request = min(self.requests)
            wait_time = 60 - (now - oldest_request).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.requests.append(now)

# Usage
limiter = ClaudeRateLimiter(requests_per_minute=4)  # Conservative limit

async def rate_limited_generate(provider, prompt):
    await limiter.wait_if_needed()
    return provider.generate(prompt)
```

## Security and Best Practices

### Content Safety

```python
def safe_prompt_processing(prompt):
    """Process prompts safely for Claude"""
    
    # Claude has built-in safety measures, but you can add your own
    sensitive_patterns = [
        r'\b(?:password|api_key|secret|token)\b',
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card pattern
        r'\b\d{3}-\d{2}-\d{4}\b'  # SSN pattern
    ]
    
    import re
    for pattern in sensitive_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            print("Warning: Potential sensitive information detected in prompt")
            break
    
    return prompt

def monitor_responses(response):
    """Monitor responses for quality and safety"""
    
    # Check for refusals or safety concerns
    refusal_indicators = [
        "I can't help with that",
        "I'm not able to provide",
        "I cannot assist with",
        "I'm not comfortable"
    ]
    
    for indicator in refusal_indicators:
        if indicator.lower() in response.lower():
            print(f"Response indicates safety refusal: {indicator}")
            break
    
    return response
```

### Usage Monitoring

```python
import logging
from datetime import datetime

class MonitoredAnthropicProvider(AnthropicProvider):
    def __init__(self, log_file='claude_usage.log', **kwargs):
        super().__init__(**kwargs)
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
    
    def generate(self, prompt, **kwargs):
        self.request_count += 1
        start_time = datetime.now()
        
        # Log request details
        self.logger.info(f"Request #{self.request_count} - Model: {self.model_name}")
        self.logger.info(f"Prompt length: {len(prompt)} characters")
        
        try:
            response = super().generate(prompt, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Request #{self.request_count} completed in {duration:.2f}s")
            self.logger.info(f"Response length: {len(response)} characters")
            
            return response
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Request #{self.request_count} failed after {duration:.2f}s: {str(e)}")
            raise
```

This comprehensive documentation provides detailed guidance for effectively using the Anthropic Claude provider within the LLM Lab framework, covering all aspects from basic usage to advanced optimization and troubleshooting.