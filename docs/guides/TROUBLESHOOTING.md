# LLM Lab Troubleshooting Guide

## Overview

This comprehensive guide helps you diagnose and resolve common issues when using LLM Lab. Each section includes symptoms, causes, and step-by-step solutions.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [API Connection Problems](#api-connection-problems)
3. [Performance Issues](#performance-issues)
4. [Fine-Tuning Problems](#fine-tuning-problems)
5. [Monitoring & Alerting Issues](#monitoring--alerting-issues)
6. [Memory and Resource Issues](#memory-and-resource-issues)
7. [Docker and Deployment Issues](#docker-and-deployment-issues)
8. [Common Error Messages](#common-error-messages)

## Installation Issues

### Problem: Package Installation Fails

**Symptoms:**
- `pip install` returns errors
- Missing dependencies
- Version conflicts

**Solutions:**

1. **Update pip and setuptools:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Use virtual environment:**
   ```bash
   # Create fresh environment
   python -m venv venv_new
   source venv_new/bin/activate  # or venv_new\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Install with specific versions:**
   ```bash
   # If conflicts occur
   pip install --force-reinstall -r requirements.txt
   ```

4. **For M1/M2 Macs:**
   ```bash
   # Install with platform-specific flags
   pip install --no-binary :all: -r requirements.txt
   ```

### Problem: CUDA/GPU Not Detected

**Symptoms:**
- `torch.cuda.is_available()` returns False
- Local fine-tuning fails

**Solutions:**

1. **Check CUDA installation:**
   ```bash
   nvidia-smi  # Should show GPU info
   nvcc --version  # Should show CUDA version
   ```

2. **Install correct PyTorch version:**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify installation:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

## API Connection Problems

### Problem: Authentication Failures

**Symptoms:**
- 401/403 errors
- "Invalid API key" messages
- "Unauthorized" responses

**Solutions:**

1. **Verify API keys:**
   ```bash
   # Check .env file
   cat .env | grep API_KEY
   
   # Test individual keys
   python -c "import os; print(os.getenv('OPENAI_API_KEY')[:10] + '...')"
   ```

2. **Check key permissions:**
   ```python
   # Test OpenAI key
   from openai import OpenAI
   client = OpenAI()
   try:
       client.models.list()
       print("OpenAI key is valid")
   except Exception as e:
       print(f"OpenAI key error: {e}")
   ```

3. **Validate all providers:**
   ```bash
   python scripts/validate_api_keys.py
   ```

### Problem: Rate Limit Errors

**Symptoms:**
- 429 errors
- "Rate limit exceeded" messages
- Intermittent failures

**Solutions:**

1. **Implement exponential backoff:**
   ```python
   from tenacity import retry, wait_exponential, stop_after_attempt
   
   @retry(
       wait=wait_exponential(multiplier=1, min=4, max=60),
       stop=stop_after_attempt(5)
   )
   def call_api(prompt):
       return llm.generate(prompt)
   ```

2. **Configure rate limits:**
   ```yaml
   # config.yaml
   rate_limits:
     openai:
       rpm: 3500  # Requests per minute
       tpm: 90000  # Tokens per minute
     anthropic:
       rpm: 1000
       tpm: 100000
   ```

3. **Use request queuing:**
   ```python
   from llm_lab.utils import RateLimiter
   
   limiter = RateLimiter(requests_per_minute=60)
   
   for prompt in prompts:
       with limiter:
           response = llm.generate(prompt)
   ```

### Problem: Network Timeouts

**Symptoms:**
- Connection timeouts
- "Network unreachable" errors
- Slow responses

**Solutions:**

1. **Increase timeout settings:**
   ```python
   # In config.py
   API_TIMEOUTS = {
       "default": 60,  # seconds
       "gpt-4": 120,   # Longer for complex models
       "fine-tuning": 600  # Much longer for training
   }
   ```

2. **Use connection pooling:**
   ```python
   import httpx
   
   # Create persistent client
   client = httpx.Client(
       timeout=30.0,
       limits=httpx.Limits(max_keepalive_connections=10)
   )
   ```

3. **Configure proxy if needed:**
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

## Performance Issues

### Problem: Slow Response Times

**Symptoms:**
- High latency (>5s for simple queries)
- Timeouts on complex prompts
- Degraded performance over time

**Solutions:**

1. **Profile performance:**
   ```python
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Your code here
   result = benchmark.run()
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)
   ```

2. **Optimize prompts:**
   ```python
   # Before: Long system prompt
   system_prompt = """You are an AI assistant. You should be helpful, 
   harmless, and honest. You should provide accurate information..."""  # 500+ tokens
   
   # After: Concise prompt
   system_prompt = "You are a helpful AI assistant."  # 7 tokens
   ```

3. **Enable caching:**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_generate(prompt_hash):
       return llm.generate(prompt)
   ```

### Problem: High Memory Usage

**Symptoms:**
- Out of memory errors
- System slowdown
- Process killed by OS

**Solutions:**

1. **Monitor memory usage:**
   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

2. **Implement batch processing:**
   ```python
   def process_in_batches(items, batch_size=100):
       for i in range(0, len(items), batch_size):
           batch = items[i:i + batch_size]
           yield process_batch(batch)
           
           # Force garbage collection
           import gc
           gc.collect()
   ```

3. **Use streaming for large datasets:**
   ```python
   def stream_dataset(filepath):
       with open(filepath, 'r') as f:
           for line in f:
               yield json.loads(line)
   ```

## Fine-Tuning Problems

### Problem: Fine-Tuning Job Fails

**Symptoms:**
- Job status shows "failed"
- Training interrupted
- Invalid dataset errors

**Solutions:**

1. **Validate dataset format:**
   ```python
   from llm_lab.utils import DatasetValidator
   
   validator = DatasetValidator()
   issues = validator.validate("training_data.jsonl")
   
   if issues:
       print("Dataset issues found:")
       for issue in issues:
           print(f"  - Line {issue['line']}: {issue['error']}")
   ```

2. **Check dataset statistics:**
   ```python
   def analyze_dataset(filepath):
       stats = {
           "total_examples": 0,
           "avg_length": 0,
           "max_length": 0,
           "min_length": float('inf')
       }
       
       with open(filepath, 'r') as f:
           for line in f:
               data = json.loads(line)
               length = len(str(data))
               stats["total_examples"] += 1
               stats["avg_length"] += length
               stats["max_length"] = max(stats["max_length"], length)
               stats["min_length"] = min(stats["min_length"], length)
       
       stats["avg_length"] /= stats["total_examples"]
       return stats
   ```

3. **Use smaller learning rate:**
   ```python
   # If model is not converging
   fine_tune_config = {
       "learning_rate_multiplier": 0.5,  # Reduce from default
       "n_epochs": 5,  # Increase epochs
       "batch_size": 4  # Smaller batches
   }
   ```

### Problem: Poor Fine-Tuned Model Performance

**Symptoms:**
- Lower quality than base model
- Overfitting or underfitting
- Inconsistent outputs

**Solutions:**

1. **Evaluate with proper metrics:**
   ```python
   evaluator = ModelEvaluator()
   
   # Compare base vs fine-tuned
   base_metrics = evaluator.evaluate("gpt-3.5-turbo", test_set)
   ft_metrics = evaluator.evaluate("ft:gpt-3.5-turbo:xxx", test_set)
   
   print(f"Base accuracy: {base_metrics['accuracy']:.2%}")
   print(f"Fine-tuned accuracy: {ft_metrics['accuracy']:.2%}")
   ```

2. **Increase dataset quality:**
   ```python
   # Add diversity to training data
   augmented_data = []
   for example in original_data:
       # Original
       augmented_data.append(example)
       
       # Paraphrased version
       augmented_data.append(paraphrase(example))
       
       # Different format
       augmented_data.append(reformat(example))
   ```

3. **Use validation set:**
   ```python
   from sklearn.model_selection import train_test_split
   
   train_data, val_data = train_test_split(
       dataset, 
       test_size=0.2, 
       random_state=42
   )
   ```

## Monitoring & Alerting Issues

### Problem: Alerts Not Triggering

**Symptoms:**
- No notifications despite issues
- Alerts delayed or missing
- Dashboard shows problems but no alerts

**Solutions:**

1. **Verify alert configuration:**
   ```yaml
   # monitoring_config.yaml
   alerts:
     rules:
       - name: "High Latency"
         condition: "avg_latency > 2.0"
         severity: "critical"
         enabled: true  # Make sure enabled
     
     channels:
       - type: email
         enabled: true  # Check this too
         recipients: ["team@company.com"]
   ```

2. **Test alert channels:**
   ```python
   from llm_lab.monitoring import AlertManager
   
   alert_manager = AlertManager()
   
   # Send test alert
   alert_manager.send_test_alert(
       channel="email",
       message="Test alert from LLM Lab"
   )
   ```

3. **Check alert cooldowns:**
   ```python
   # Clear cooldown cache if needed
   alert_manager.clear_cooldowns()
   ```

### Problem: Inaccurate Metrics

**Symptoms:**
- Dashboard shows wrong values
- Metrics don't match logs
- Missing data points

**Solutions:**

1. **Verify metric collection:**
   ```python
   # Add debug logging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Check raw metrics
   metrics = collector.get_raw_metrics(hours=1)
   print(f"Raw metrics count: {len(metrics)}")
   ```

2. **Check time synchronization:**
   ```bash
   # Ensure system time is correct
   timedatectl status
   
   # Sync if needed
   sudo ntpdate -s time.nist.gov
   ```

3. **Validate aggregations:**
   ```python
   # Compare different aggregation methods
   from statistics import mean, median
   
   latencies = [m['latency'] for m in metrics]
   print(f"Mean: {mean(latencies):.3f}")
   print(f"Median: {median(latencies):.3f}")
   print(f"P95: {np.percentile(latencies, 95):.3f}")
   ```

## Memory and Resource Issues

### Problem: Memory Leaks

**Symptoms:**
- Gradual memory increase
- Eventually runs out of memory
- Need to restart frequently

**Solutions:**

1. **Track object references:**
   ```python
   import objgraph
   
   # Before operation
   objgraph.show_growth()
   
   # Run operation
   results = run_benchmark()
   
   # After operation
   objgraph.show_growth()
   objgraph.show_most_common_types(limit=10)
   ```

2. **Fix common leaks:**
   ```python
   # Clear caches periodically
   import gc
   
   def periodic_cleanup():
       # Clear LRU caches
       for func in [cached_func1, cached_func2]:
           func.cache_clear()
       
       # Force garbage collection
       gc.collect()
   ```

3. **Use context managers:**
   ```python
   # Ensure resources are freed
   class LLMClient:
       def __enter__(self):
           self.client = create_client()
           return self.client
       
       def __exit__(self, exc_type, exc_val, exc_tb):
           self.client.close()
           del self.client
   ```

## Docker and Deployment Issues

### Problem: Container Won't Start

**Symptoms:**
- Docker container exits immediately
- "Container exited with code 1"
- No logs available

**Solutions:**

1. **Check container logs:**
   ```bash
   # Get container ID
   docker ps -a
   
   # View logs
   docker logs <container_id>
   
   # Interactive debug
   docker run -it --entrypoint /bin/bash llm-lab
   ```

2. **Verify environment variables:**
   ```bash
   # Check what's passed to container
   docker run --rm llm-lab env | grep API_KEY
   
   # Use env file
   docker run --env-file .env llm-lab
   ```

3. **Fix permission issues:**
   ```dockerfile
   # In Dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   
   # Ensure writable directories
   RUN mkdir -p /app/logs /app/data && \
       chown -R appuser:appuser /app
   ```

### Problem: Kubernetes Pod Crashes

**Symptoms:**
- Pod in CrashLoopBackOff
- Readiness probe failing
- Resource limits hit

**Solutions:**

1. **Check pod details:**
   ```bash
   # Get pod info
   kubectl describe pod <pod-name>
   
   # View logs
   kubectl logs <pod-name> --previous
   
   # Check events
   kubectl get events --sort-by='.lastTimestamp'
   ```

2. **Adjust resource limits:**
   ```yaml
   # deployment.yaml
   resources:
     requests:
       memory: "2Gi"
       cpu: "1000m"
     limits:
       memory: "4Gi"
       cpu: "2000m"
   ```

3. **Fix probe configuration:**
   ```yaml
   livenessProbe:
     httpGet:
       path: /health
       port: 8080
     initialDelaySeconds: 30
     periodSeconds: 10
     timeoutSeconds: 5
     failureThreshold: 3
   ```

## Common Error Messages

### "Model not found"

**Cause:** Trying to use a model that doesn't exist or you don't have access to.

**Solution:**
```python
# List available models
for provider in ["openai", "anthropic", "google"]:
    models = llm_lab.list_models(provider)
    print(f"{provider}: {models}")
```

### "Context length exceeded"

**Cause:** Input exceeds model's maximum context window.

**Solution:**
```python
# Implement chunking
def chunk_text(text, max_tokens=4000):
    # Simple word-based chunking
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_tokens * 4:  # Approximate
            chunks.append(' '.join(current_chunk[:-1]))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

### "SSL Certificate Error"

**Cause:** Corporate proxy or firewall blocking SSL verification.

**Solution:**
```python
# Temporary fix (not for production!)
import ssl
import certifi

# Use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Or disable verification (INSECURE!)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

### "Quota exceeded"

**Cause:** Hit API usage limits or budget limits.

**Solution:**
```python
# Check usage
usage = llm_lab.get_usage_stats()
print(f"Tokens used: {usage['tokens_used']:,}")
print(f"Cost so far: ${usage['total_cost']:.2f}")

# Implement budget controls
if usage['total_cost'] > DAILY_BUDGET * 0.8:
    send_alert("Approaching daily budget limit")
    enable_strict_mode()
```

## Debug Mode

Enable comprehensive debugging:

```python
# In your script
import logging

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Enable HTTP debugging
import httpx
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.DEBUG)

# Run with debug mode
llm_lab.run(debug=True, verbose=True)
```

## Getting Help

If you're still experiencing issues:

1. **Check the logs:**
   ```bash
   tail -f logs/llm-lab.log
   ```

2. **Run diagnostics:**
   ```bash
   python -m llm_lab.diagnostics
   ```

3. **Join our community:**
   - Discord: [discord.gg/llm-lab](https://discord.gg/llm-lab)
   - GitHub Issues: [github.com/remyolson/llm-lab/issues](https://github.com/remyolson/llm-lab/issues)

4. **Contact support:**
   - Email: support@llm-lab.io
   - Include diagnostic output and logs

Remember to sanitize any sensitive information (API keys, private data) before sharing logs or asking for help!