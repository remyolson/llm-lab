# Troubleshooting Guide for LLM Lab

This comprehensive guide helps you diagnose and resolve common issues when using the LLM Lab multi-model framework.

## 🚨 Quick Diagnostic Checklist

Before diving into specific issues, run this quick diagnostic:

```bash
# 1. Check environment setup
make dev-setup

# 2. Verify API keys
echo "Google: $([ -n "$GOOGLE_API_KEY" ] && echo "✓ Set" || echo "✗ Missing")"
echo "OpenAI: $([ -n "$OPENAI_API_KEY" ] && echo "✓ Set" || echo "✗ Missing")"
echo "Anthropic: $([ -n "$ANTHROPIC_API_KEY" ] && echo "✓ Set" || echo "✗ Missing")"

# 3. Run basic tests
make test-unit

# 4. Test provider connectivity
python -c "
from llm_providers import GoogleProvider
try:
    provider = GoogleProvider(model_name='gemini-1.5-flash')
    response = provider.generate('Hello')
    print('✓ Google provider working')
except Exception as e:
    print(f'✗ Google provider error: {e}')
"
```

## 🔧 Installation and Setup Issues

### Issue: Package Installation Failures

**Symptoms**:
- `pip install` fails with dependency conflicts
- Missing required packages
- Version compatibility errors

**Solutions**:

1. **Use a fresh virtual environment**:
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate  # On Windows: fresh_env\Scripts\activate
   pip install --upgrade pip
   pip install -e ".[all]"
   ```

2. **Install dependencies step by step**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -r requirements-test.txt
   ```

3. **Check Python version compatibility**:
   ```bash
   python --version  # Should be 3.8 or higher
   ```

4. **Clear pip cache if issues persist**:
   ```bash
   pip cache purge
   pip install --no-cache-dir -e ".[all]"
   ```

### Issue: Environment Configuration Problems

**Symptoms**:
- `.env` file not being loaded
- API keys not recognized
- Configuration parameters ignored

**Solutions**:

1. **Verify .env file location and format**:
   ```bash
   # .env should be in project root
   ls -la .env

   # Check format (no spaces around =)
   cat .env
   # Correct: GOOGLE_API_KEY=your-key-here
   # Wrong: GOOGLE_API_KEY = your-key-here
   ```

2. **Test environment loading**:
   ```python
   from dotenv import load_dotenv
   import os

   load_dotenv()
   print("Google API Key:", "✓ Set" if os.getenv('GOOGLE_API_KEY') else "✗ Missing")
   ```

3. **Use absolute paths for .env**:
   ```python
   from pathlib import Path
   from dotenv import load_dotenv

   env_path = Path(__file__).parent / '.env'
   load_dotenv(dotenv_path=env_path)
   ```

## 🔑 API Key and Authentication Issues

### Issue: Invalid or Missing API Keys

**Symptoms**:
- `InvalidCredentialsError: Invalid or missing credentials for OpenAI`
- `401 Unauthorized` errors
- `403 Forbidden` responses
- `Invalid API key` messages

**Enhanced Error Messages**:
Our improved error handling now provides specific troubleshooting steps:
```
InvalidCredentialsError: Invalid or missing credentials for OpenAI

Troubleshooting:
1. Check that OPENAI_API_KEY is set in your environment or .env file
2. Verify your API key at https://platform.openai.com/api-keys
3. Ensure your API key starts with 'sk-'
4. Check if your account has available credits
```

**Diagnosis**:
```bash
# Test each API key
curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
     "https://generativelanguage.googleapis.com/v1/models"

curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     "https://api.openai.com/v1/models"

curl -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "Content-Type: application/json" \
     -H "anthropic-version: 2023-06-01" \
     "https://api.anthropic.com/v1/messages" \
     -d '{"model": "claude-3-haiku-20240307", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}'
```

**Solutions**:

1. **Verify API keys are valid and active**:
   - Check expiration dates in provider consoles
   - Ensure keys have necessary permissions
   - Regenerate keys if needed

2. **Check API key format**:
   ```bash
   # Google: Should start with 'AIza'
   # OpenAI: Should start with 'sk-'
   # Anthropic: Should start with 'sk-ant-'
   ```

3. **Ensure no extra whitespace**:
   ```bash
   # Remove any trailing spaces/newlines
   export GOOGLE_API_KEY=$(echo "$GOOGLE_API_KEY" | tr -d ' \n\r')
   ```

### Issue: API Quota and Billing Problems

**Symptoms**:
- `quota exceeded` errors
- `billing not enabled` messages
- Sudden service interruptions

**Solutions**:

1. **Check quota usage**:
   - Google: [Google Cloud Console](https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas)
   - OpenAI: [OpenAI Usage Dashboard](https://platform.openai.com/usage)
   - Anthropic: [Anthropic Console](https://console.anthropic.com/usage)

2. **Implement quota monitoring**:
   ```python
   import time
   from datetime import datetime, timedelta

   class QuotaTracker:
       def __init__(self, requests_per_minute=10):
           self.rpm_limit = requests_per_minute
           self.requests = []

       def can_make_request(self):
           now = datetime.now()
           # Remove requests older than 1 minute
           self.requests = [req for req in self.requests
                           if now - req < timedelta(minutes=1)]
           return len(self.requests) < self.rpm_limit

       def record_request(self):
           self.requests.append(datetime.now())
   ```

3. **Set up billing alerts** in each provider's console

## 🌐 Network and Connectivity Issues

### Issue: Network Timeouts and Connection Errors

**Symptoms**:
- `Connection timeout` errors
- `DNS resolution failed`
- Intermittent connection issues

**Solutions**:

1. **Increase timeout settings**:
   ```python
   from llm_providers import OpenAIProvider

   provider = OpenAIProvider(
       model_name="gpt-4o-mini",
       timeout=60  # Increase from default 30 seconds
   )
   ```

2. **Implement retry logic with exponential backoff**:
   ```python
   import time
   import random

   def retry_with_backoff(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except Exception as e:
               if attempt == max_retries - 1:
                   raise e

               wait_time = (2 ** attempt) + random.uniform(0, 1)
               print(f"Retry {attempt + 1} in {wait_time:.1f}s...")
               time.sleep(wait_time)
   ```

3. **Check network connectivity**:
   ```bash
   # Test basic connectivity
   ping google.com
   ping api.openai.com
   ping api.anthropic.com

   # Test DNS resolution
   nslookup generativelanguage.googleapis.com
   ```

### Issue: Corporate Firewall or Proxy Problems

**Symptoms**:
- Connections work outside corporate network but fail inside
- SSL certificate errors
- Proxy authentication required

**Solutions**:

1. **Configure proxy settings**:
   ```python
   import os
   import requests

   # Set proxy environment variables
   os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
   os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'

   # Or configure in requests session
   session = requests.Session()
   session.proxies = {
       'http': 'http://proxy.company.com:8080',
       'https': 'http://proxy.company.com:8080'
   }
   ```

2. **Handle SSL certificate verification**:
   ```python
   import ssl
   import urllib3

   # Disable SSL warnings (not recommended for production)
   urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

   # Or provide custom certificate bundle
   import certifi
   ssl_context = ssl.create_default_context(cafile=certifi.where())
   ```

3. **Work with IT team to whitelist domains**:
   - `*.googleapis.com` (Google)
   - `*.openai.com` (OpenAI)
   - `*.anthropic.com` (Anthropic)

## 🤖 Provider-Specific Issues

### Google Provider Issues

#### Issue: Model Not Found Errors

**Symptoms**:
- `Model not found` errors
- Invalid model name messages

**Solutions**:
```python
# List available models
from google.generativeai import list_models

for model in list_models():
    print(f"Model: {model.name}")

# Use correct model names
provider = GoogleProvider(model_name="gemini-1.5-flash")  # Correct
# Not: GoogleProvider(model_name="gemini-flash")  # Wrong
```

#### Issue: Safety Filter Rejections

**Symptoms**:
- Responses blocked by safety filters
- Empty responses for certain prompts

**Solutions**:
```python
from llm_providers import GoogleProvider

provider = GoogleProvider(
    model_name="gemini-1.5-flash",
    safety_settings={
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_ONLY_HIGH',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_ONLY_HIGH',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_ONLY_HIGH',
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_ONLY_HIGH'
    }
)
```

### OpenAI Provider Issues

#### Issue: Model Access Denied

**Symptoms**:
- `model not found` for GPT-4 models
- Access denied to specific models

**Solutions**:
1. **Check model availability**:
   ```python
   import openai

   client = openai.OpenAI()
   models = client.models.list()
   for model in models.data:
       print(model.id)
   ```

2. **Use available models**:
   ```python
   # If GPT-4 not available, use GPT-3.5-turbo
   provider = OpenAIProvider(model_name="gpt-3.5-turbo")
   ```

#### Issue: Token Limit Exceeded

**Symptoms**:
- `maximum context length exceeded` errors
- Requests failing with long prompts

**Solutions**:
```python
def truncate_prompt(prompt, max_tokens=3000):
    """Rough token estimation and truncation."""
    # Approximate: 1 token ≈ 0.75 words ≈ 4 characters
    max_chars = max_tokens * 4
    if len(prompt) > max_chars:
        return prompt[:max_chars] + "..."
    return prompt

provider = OpenAIProvider(
    model_name="gpt-4o-mini",
    max_tokens=1000  # Limit response length
)
```

### Anthropic Provider Issues

#### Issue: Message Format Errors

**Symptoms**:
- `invalid message format` errors
- API rejecting message structure

**Solutions**:
```python
# Correct message format
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]

# Validate message format
def validate_messages(messages):
    for i, msg in enumerate(messages):
        assert "role" in msg, f"Message {i} missing 'role'"
        assert "content" in msg, f"Message {i} missing 'content'"
        assert msg["role"] in ["user", "assistant"], f"Invalid role in message {i}"
```

#### Issue: Rate Limiting Errors

**Symptoms**:
- `RateLimitError: Rate limit exceeded for OpenAI`
- `429 Too Many Requests` errors
- Requests failing frequently

**Enhanced Error Messages**:
Our improved error handling now provides specific suggestions:
```
RateLimitError: Rate limit exceeded for OpenAI. Retry after 60 seconds

Suggestions:
1. Implement exponential backoff with retry logic
2. Consider using a different model with higher rate limits
3. Batch requests to reduce API calls
4. Check your rate limits at https://platform.openai.com/account/limits
```

**Solutions**:
```python
import asyncio
from datetime import datetime, timedelta

class AnthropicRateLimiter:
    def __init__(self, requests_per_minute=5):  # Conservative for free tier
        self.rpm = requests_per_minute
        self.requests = []

    async def wait_if_needed(self):
        now = datetime.now()
        # Remove old requests
        self.requests = [req for req in self.requests
                        if now - req < timedelta(minutes=1)]

        if len(self.requests) >= self.rpm:
            oldest = min(self.requests)
            wait_time = 60 - (now - oldest).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)
```

## 📊 Performance and Resource Issues

### Issue: Slow Response Times

**Symptoms**:
- Responses taking much longer than expected
- Timeouts on reasonable requests
- Performance degradation over time

**Diagnosis**:
```python
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@measure_time
def test_provider_speed():
    provider = GoogleProvider(model_name="gemini-1.5-flash")
    return provider.generate("Quick test")
```

**Solutions**:

1. **Optimize provider configuration**:
   ```python
   # Use faster models
   google_fast = GoogleProvider(model_name="gemini-1.5-flash")  # Fastest Gemini
   openai_fast = OpenAIProvider(model_name="gpt-4o-mini")      # Fastest GPT-4 class
   anthropic_fast = AnthropicProvider(model_name="claude-3-5-haiku-20241022")  # Fastest Claude

   # Optimize parameters
   provider = GoogleProvider(
       model_name="gemini-1.5-flash",
       temperature=0.1,  # Lower = faster
       max_tokens=500,   # Shorter responses = faster
   )
   ```

2. **Implement caching**:
   ```python
   from functools import lru_cache
   import hashlib

   class CachedProvider:
       def __init__(self, provider):
           self.provider = provider
           self.cache = {}

       def generate(self, prompt):
           # Create cache key
           key = hashlib.md5(prompt.encode()).hexdigest()

           if key in self.cache:
               return self.cache[key]

           response = self.provider.generate(prompt)
           self.cache[key] = response
           return response
   ```

3. **Use async/await for concurrent requests**:
   ```python
   import asyncio

   async def process_multiple_prompts(provider, prompts):
       async def process_one(prompt):
           return await provider.generate_async(prompt)

       tasks = [process_one(prompt) for prompt in prompts]
       return await asyncio.gather(*tasks)
   ```

### Issue: Memory Usage Problems

**Symptoms**:
- Memory usage growing over time
- Out of memory errors
- System becoming unresponsive

**Solutions**:

1. **Monitor memory usage**:
   ```python
   import psutil
   import gc

   def check_memory():
       process = psutil.Process()
       memory_mb = process.memory_info().rss / 1024 / 1024
       print(f"Memory usage: {memory_mb:.1f} MB")
       return memory_mb

   # Force garbage collection
   def cleanup_memory():
       gc.collect()
       check_memory()
   ```

2. **Process data in batches**:
   ```python
   def process_in_batches(data, batch_size=10):
       for i in range(0, len(data), batch_size):
           batch = data[i:i + batch_size]
           # Process batch
           yield process_batch(batch)
           # Cleanup between batches
           gc.collect()
   ```

3. **Limit response caching**:
   ```python
   from collections import OrderedDict

   class LimitedCache:
       def __init__(self, max_size=100):
           self.cache = OrderedDict()
           self.max_size = max_size

       def get(self, key):
           if key in self.cache:
               # Move to end (most recently used)
               self.cache.move_to_end(key)
               return self.cache[key]
           return None

       def set(self, key, value):
           if key in self.cache:
               self.cache.move_to_end(key)
           self.cache[key] = value

           # Remove oldest if over limit
           while len(self.cache) > self.max_size:
               self.cache.popitem(last=False)
   ```

## 🧪 Testing and Development Issues

### Issue: Tests Failing

**Symptoms**:
- Unit tests failing unexpectedly
- Integration tests timing out
- Inconsistent test results

**Diagnosis**:
```bash
# Run specific test categories
make test-unit          # Fast tests only
make test-integration   # API-dependent tests
make test-performance   # Benchmark tests

# Run with verbose output
pytest tests/ -v --tb=long

# Run specific test files
pytest tests/test_providers.py -v
```

**Solutions**:

1. **Check test configuration**:
   ```python
   # pytest.ini or pyproject.toml
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = ["test_*.py"]
   markers = [
       "slow: marks tests as slow (deselect with '-m \"not slow\"')",
       "integration: marks tests as integration tests"
   ]
   ```

2. **Skip tests conditionally**:
   ```python
   import pytest
   import os

   @pytest.mark.skipif(not os.getenv('GOOGLE_API_KEY'),
                      reason="Google API key not available")
   def test_google_provider():
       # Test implementation
       pass
   ```

3. **Use proper test isolation**:
   ```python
   import pytest
   from unittest.mock import patch

   @pytest.fixture
   def mock_provider():
       with patch('llm_providers.GoogleProvider') as mock:
           mock.return_value.generate.return_value = "Mocked response"
           yield mock

   def test_with_mock(mock_provider):
       # Test uses mocked provider
       pass
   ```

### Issue: Import Errors in Tests

**Symptoms**:
- `ModuleNotFoundError` in tests
- Import paths not working
- Tests can't find project modules

**Solutions**:

1. **Install in development mode**:
   ```bash
   pip install -e .
   ```

2. **Fix Python path in tests**:
   ```python
   import sys
   from pathlib import Path

   # Add project root to path
   project_root = Path(__file__).parent.parent
   sys.path.insert(0, str(project_root))
   ```

3. **Use proper package structure**:
   ```
   project/
   ├── llm_providers/
   │   ├── __init__.py
   │   └── google.py
   └── tests/
       ├── __init__.py
       └── test_providers.py
   ```

## 📈 Performance Interpretation Guide

### Understanding Response Time Metrics

**Response Time Categories**:
- **Excellent**: < 1 second
- **Good**: 1-3 seconds
- **Acceptable**: 3-10 seconds
- **Slow**: > 10 seconds

**Factors Affecting Response Time**:
1. **Model size**: Larger models are generally slower
2. **Prompt length**: Longer prompts take more time to process
3. **Response length**: Longer responses take more time to generate
4. **Network latency**: Distance to API servers affects speed
5. **Server load**: High demand can slow responses

### Interpreting Benchmark Scores

**Score Ranges**:
- **0.9-1.0**: Excellent performance
- **0.7-0.9**: Good performance
- **0.5-0.7**: Acceptable performance
- **< 0.5**: Poor performance

**Score Factors**:
1. **Evaluation method**: Different methods produce different score ranges
2. **Task difficulty**: Complex tasks naturally have lower scores
3. **Model specialization**: Models perform better on tasks they're trained for
4. **Prompt quality**: Well-crafted prompts improve scores

### Cost Analysis Interpretation

**Cost Efficiency Metrics**:
- **Cost per request**: Total cost divided by number of requests
- **Cost per token**: Useful for comparing across different response lengths
- **Requests per dollar**: How many requests you can make with $1
- **Cost efficiency score**: Performance divided by cost

**Optimization Strategies**:
1. **Use appropriate models**: Don't use expensive models for simple tasks
2. **Limit response length**: Set reasonable `max_tokens` values
3. **Implement caching**: Avoid repeated requests for same content
4. **Batch requests**: Some providers offer batch discounts

## 🔍 Advanced Debugging Techniques

### Logging and Monitoring

1. **Enable detailed logging**:
   ```python
   import logging

   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('llm_lab.log'),
           logging.StreamHandler()
       ]
   )

   logger = logging.getLogger(__name__)
   ```

2. **Monitor API calls**:
   ```python
   import time
   from functools import wraps

   def monitor_api_calls(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           start_time = time.time()
           try:
               result = func(*args, **kwargs)
               end_time = time.time()
               logger.info(f"API call successful: {end_time - start_time:.2f}s")
               return result
           except Exception as e:
               end_time = time.time()
               logger.error(f"API call failed after {end_time - start_time:.2f}s: {e}")
               raise
       return wrapper
   ```

3. **Track resource usage**:
   ```python
   import psutil
   import threading
   import time

   class ResourceMonitor:
       def __init__(self, interval=5):
           self.interval = interval
           self.monitoring = False
           self.stats = []

       def start_monitoring(self):
           self.monitoring = True
           thread = threading.Thread(target=self._monitor)
           thread.daemon = True
           thread.start()

       def _monitor(self):
           while self.monitoring:
               stats = {
                   'cpu_percent': psutil.cpu_percent(),
                   'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                   'timestamp': time.time()
               }
               self.stats.append(stats)
               time.sleep(self.interval)

       def stop_monitoring(self):
           self.monitoring = False
           return self.stats
   ```

### Network Debugging

1. **Test connectivity**:
   ```python
   import requests
   import time

   def test_endpoint_connectivity(url, timeout=10):
       try:
           start = time.time()
           response = requests.get(url, timeout=timeout)
           end = time.time()

           return {
               'success': response.status_code == 200,
               'status_code': response.status_code,
               'response_time': end - start,
               'headers': dict(response.headers)
           }
       except Exception as e:
           return {
               'success': False,
               'error': str(e),
               'response_time': None
           }

   # Test provider endpoints
   endpoints = {
       'google': 'https://generativelanguage.googleapis.com/v1/models',
       'openai': 'https://api.openai.com/v1/models',
       'anthropic': 'https://api.anthropic.com/v1/messages'
   }

   for name, url in endpoints.items():
       result = test_endpoint_connectivity(url)
       print(f"{name}: {'✓' if result['success'] else '✗'} {result.get('response_time', 'N/A')}")
   ```

### Error Pattern Analysis

1. **Categorize errors**:
   ```python
   from collections import defaultdict
   import re

   class ErrorAnalyzer:
       def __init__(self):
           self.errors = defaultdict(list)

       def record_error(self, error, context=None):
           error_type = self._categorize_error(str(error))
           self.errors[error_type].append({
               'error': str(error),
               'context': context,
               'timestamp': time.time()
           })

       def _categorize_error(self, error_str):
           patterns = {
               'rate_limit': r'rate limit|429|too many requests',
               'auth': r'unauthorized|401|invalid.*key',
               'network': r'connection|timeout|dns',
               'model': r'model.*not found|invalid model',
               'quota': r'quota|billing|exceeded'
           }

           for category, pattern in patterns.items():
               if re.search(pattern, error_str.lower()):
                   return category
           return 'unknown'

       def get_error_summary(self):
           return {
               category: {
                   'count': len(errors),
                   'recent': errors[-5:] if errors else []
               }
               for category, errors in self.errors.items()
           }
   ```

## 📞 Getting Additional Help

### Self-Help Resources

1. **Check the documentation**:
   - Provider-specific guides: `docs/providers/`
   - Migration guide: `docs/MIGRATION_GUIDE.md`
   - CI/CD guide: `CI_CD_GUIDE.md`

2. **Review test cases**:
   - Unit tests: `tests/unit_providers/`
   - Integration tests: `tests/integration/`
   - Example usage: `examples/`

3. **Run diagnostic commands**:
   ```bash
   # Full diagnostic suite
   make dev-setup
   make test-unit
   make lint
   make type-check

   # Check specific components
   python -c "from llm_providers import GoogleProvider; print('✓ Import OK')"
   python examples/notebooks/01_basic_multi_model_comparison.py
   ```

### Community Support

1. **Search existing issues**:
   - Check GitHub issues for similar problems
   - Review closed issues for solutions
   - Look for relevant labels (bug, help-wanted, etc.)

2. **Create a detailed issue**:
   ```markdown
   ## Problem Description
   Brief description of the issue

   ## Environment
   - OS: macOS/Linux/Windows
   - Python version: 3.x
   - LLM Lab version: x.x.x
   - Provider: Google/OpenAI/Anthropic

   ## Steps to Reproduce
   1. Step 1
   2. Step 2
   3. Step 3

   ## Expected Behavior
   What you expected to happen

   ## Actual Behavior
   What actually happened

   ## Error Messages
   ```
   Full error traceback
   ```

   ## Additional Context
   Any other relevant information
   ```

3. **Include diagnostic information**:
   ```bash
   # Gather system info
   python --version
   pip list | grep -E "(llm|openai|anthropic|google)"
   echo "OS: $(uname -a)"

   # Test basic functionality
   python -c "
   import sys
   print('Python path:', sys.path)
   try:
       from llm_providers import GoogleProvider
       print('✓ Imports working')
   except Exception as e:
       print('✗ Import error:', e)
   "
   ```

### Professional Support

For enterprise users or complex integration needs:

1. **Consider professional consulting** for:
   - Large-scale deployments
   - Custom integration requirements
   - Performance optimization
   - Security compliance

2. **Contact provider support** for:
   - API-specific issues
   - Billing and quota problems
   - Service availability concerns

---

This troubleshooting guide covers the most common issues encountered when using LLM Lab. If you're still experiencing problems after following these steps, please create a detailed issue on the project's GitHub repository with the diagnostic information requested above.
