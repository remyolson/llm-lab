# Use Case 5: Local Model Development and Testing

*Run and evaluate small LLMs on your MacBook Pro for cost-free development and experimentation.*

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:
- **Run pre-downloaded small LLMs** (SmolLM, Qwen) on your MacBook Pro
- **Compare local vs cloud models** for performance, cost, and quality trade-offs
- **Optimize models for Apple Silicon** with Metal acceleration
- **Develop and test offline** without API costs or internet dependency
- **Benchmark local models** using the same framework as cloud providers
- **Create local API endpoints** for seamless integration with existing applications
- **Work efficiently** with models optimized for standard laptops

## 📋 Before You Begin

- Complete all [Prerequisites](./PREREQUISITES.md)
- Ensure you have Python 3.9+ installed
- Time required: ~30 minutes (models already downloaded)
- Estimated ongoing cost: $0.00 (local compute only)

### 💻 Hardware Requirements for MacBook Pro

This guide is optimized for standard MacBook Pro configurations:

**💡 Already Downloaded:** The repository includes 6 small models in `/models/small-llms/`

- **MacBook Pro M1/M2/M3 (8GB RAM)**:
  - ✅ Pythia 70M: ~50-70 tokens/sec (smallest)
  - ✅ Pythia 160M: ~40-60 tokens/sec
  - ✅ SmolLM 135M: ~30-50 tokens/sec
  - ✅ SmolLM 360M: ~20-40 tokens/sec
  - ✅ Qwen 0.5B: ~15-30 tokens/sec
  - ⚠️ Qwen 0.5B GGUF: ~25-45 tokens/sec (quantized)

- **MacBook Pro M1/M2/M3 (16GB+ RAM)**:
  - All models run smoothly with headroom for other applications
  - Metal acceleration provides significant speed boost
  - Can run multiple models simultaneously

- **Intel MacBook Pro**:
  - All models work but at reduced speed (~50% of Apple Silicon)
  - Recommend using quantized GGUF version for better performance

*Note: These small models are specifically chosen to run efficiently on standard laptops without requiring high-end hardware.*

## 📊 Pre-Downloaded Local Models

These models are already available in `/models/small-llms/`:

| Model | Size | What It's Best For | Speed on M2 | Example Use Case |
|-------|------|-------------------|-------------|------------------|
| **pythia-70m** | 70M params (~280MB) | Fastest inference, minimal tasks | 50-70 tokens/sec | Ultra-quick testing, demos |
| **pythia-160m** | 160M params (~640MB) | Very fast, basic quality | 40-60 tokens/sec | Simple generation tasks |
| **smollm-135m** | 135M params (~500MB) | Fast inference, basic tasks | 30-50 tokens/sec | Quick testing, simple Q&A |
| **smollm-360m** | 360M params (~1.4GB) | Better quality, still fast | 20-40 tokens/sec | General chat, explanations |
| **qwen-0.5b** | 500M params (~2GB) | Best quality of small models | 15-30 tokens/sec | More complex reasoning |
| **qwen-0.5b-gguf** | 500M params (~300MB) | Quantized for efficiency | 25-45 tokens/sec | Production use, API serving |

### 🎯 **Model Selection Guide:**

- **⚡ For ultra-speed:** Use `pythia-70m` (fastest possible, 50-70 tokens/sec)
- **🚀 For speed:** Use `pythia-160m` or `smollm-135m` (very fast responses)
- **⚖️ For balance:** Use `smollm-360m` (good quality and speed)
- **🎯 For quality:** Use `qwen-0.5b` (best responses from small models)
- **🔧 For production:** Use `qwen-0.5b-gguf` (quantized for efficiency)
- **📊 For testing:** Start with pythia-70m, work up to larger models

## 🚀 Step-by-Step Guide

### Step 1: Install Required Dependencies

Install the necessary Python packages for running local models:

```bash
# Install transformers for standard model format
pip install transformers torch accelerate

# For Apple Silicon optimization (recommended for Mac users)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GGUF model support (quantized models)
pip install llama-cpp-python

# On Apple Silicon Macs, install with Metal support:
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Step 2: Verify Pre-Downloaded Models

The repository already includes 4 small models. Let's verify they're available:

```bash
# List available models
ls -la models/small-llms/

# You should see:
# - pythia-70m/
# - pythia-160m/
# - smollm-135m/
# - smollm-360m/
# - qwen-0.5b/
# - qwen-0.5b-gguf/

# Quick test to verify models are accessible
python models/small-llms/quick_demo.py
```

**Expected Output:**
```
🤖 Small LLM Quick Demo
==================================================
✓ Found 6 pre-downloaded models:
  - pythia-70m (70M parameters)
  - pythia-160m (160M parameters)
  - smollm-135m (135M parameters)
  - smollm-360m (360M parameters)
  - qwen-0.5b (500M parameters)
  - qwen-0.5b-gguf (500M parameters, quantized)
✓ Models ready for use!
```

**Note:** If you need to download additional models, use:
```bash
# Download script saves to the same location
python models/small-llms/download_small_models.py
```

### Step 3: Run Your First Local Inference

Test the pre-downloaded models with simple prompts:

```bash
# Quick test with the smallest model (recommended first test)
python models/small-llms/inference.py --model pythia-70m --prompt "What is machine learning?"

# Test progressively larger models
python models/small-llms/inference.py --model pythia-160m --prompt "Explain quantum computing"
python models/small-llms/inference.py --model smollm-135m --prompt "What is Python?"
python models/small-llms/inference.py --model smollm-360m --prompt "Explain databases"
python models/small-llms/inference.py --model qwen-0.5b --prompt "Write a Python function to sort a list"

# Test the quantized GGUF model (best efficiency)
python models/small-llms/inference.py --model qwen-0.5b-gguf --prompt "What is AI?"

# Interactive chat mode
python models/small-llms/inference.py --model smollm-360m --interactive
```

**Expected Output:**
```
🖥️  Loading local model: pythia-70m
✓ Model loaded successfully (0.4s)
🧠 Generating response...

Machine learning is a type of artificial intelligence that allows computers
to learn from data without being explicitly programmed...
✓ Response generated (1.5s, ~55 tokens/sec)
```

**MacBook Pro Specific Tips:**
- On Apple Silicon, models automatically use Metal acceleration
- First run may be slower as Metal shaders compile
- Subsequent runs will be significantly faster

### Step 4: Compare Local vs Cloud Performance

Run side-by-side comparisons with cloud providers:

```bash
# Compare smallest local model with cloud models
python run_benchmarks.py \
  --provider local \
  --model models/small-llms/smollm-135m \
  --dataset truthfulness \
  --limit 5

# Compare all local models
python scripts/compare_local_models.py \
  --models smollm-135m,smollm-360m,qwen-0.5b,qwen-0.5b-gguf \
  --prompts "What is AI?" "Explain machine learning" "Write a haiku"

# Compare local vs cloud for cost analysis
python scripts/local_vs_cloud_comparison.py \
  --local-model qwen-0.5b \
  --cloud-model gpt-4o-mini \
  --num-requests 100

# Batch comparison across multiple prompts
python models/small-llms/inference.py \
  --model qwen-0.5b \
  --batch-file test_prompts.txt \
  --compare-with openai:gpt-4o-mini
```

**Expected Comparison Output:**
```
📊 Local vs Cloud Model Comparison
==================================================
Task: "Explain machine learning in simple terms"

Model               Response Time   Cost      Quality   Tokens/Sec
----------------------------------------------------------------
qwen-0.5b (local)   3.2s           $0.000    7.0/10    22.5
gpt-4o-mini (cloud) 1.8s           $0.003    8.5/10    N/A
----------------------------------------------------------------

💡 Analysis: Local model provides 70% quality at 0% cost
   Break-even: After 334 requests, local hardware pays for itself
```

### Step 5: MacBook Pro Optimization

Optimize performance specifically for MacBook Pro:

```bash
# Apple Silicon optimization (M1/M2/M3)
python models/small-llms/inference.py \
  --model qwen-0.5b \
  --use-metal \
  --prompt "Test Metal acceleration"

# CPU optimization for Intel Macs
python models/small-llms/inference.py \
  --model smollm-135m \
  --cpu-threads 8 \
  --prompt "Test CPU performance"

# Memory-efficient settings for 8GB Macs
python models/small-llms/inference.py \
  --model qwen-0.5b-gguf \
  --max-tokens 256 \
  --batch-size 1 \
  --prompt "Test memory efficiency"

# Benchmark different optimization settings
python models/small-llms/benchmark_optimizations.py \
  --model smollm-360m \
  --test-configs "default,metal,cpu-optimized"
```

**MacBook Pro Performance Tips:**
```bash
# Check if Metal is being used (Apple Silicon only)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Monitor GPU usage on Apple Silicon
sudo powermetrics --samplers gpu_power -i 1000 -n 10

# For best performance on MacBook Pro:
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Fallback for unsupported ops
export OMP_NUM_THREADS=8              # Optimize CPU threads
```

### Step 6: Benchmark Local Model Performance

Run comprehensive benchmarks to measure performance on MacBook Pro:

```bash
# Quick performance test across all models
python models/small-llms/benchmark_all.py

# Detailed benchmark with specific metrics
python run_benchmarks.py \
  --provider local \
  --model models/small-llms/qwen-0.5b \
  --dataset truthfulness \
  --limit 10 \
  --output-format json

# Compare model sizes vs performance
python models/small-llms/size_vs_performance.py \
  --models smollm-135m,smollm-360m,qwen-0.5b \
  --test-suite "speed,quality,memory"

# Batch processing efficiency test
python models/small-llms/inference.py \
  --model smollm-360m \
  --batch-prompts \
  --prompts "What is AI?" "Define ML" "Explain neural networks"
```

**Expected Benchmark Results (MacBook Pro M2):**
```
📊 Local Model Benchmark Results
==================================================
Model: qwen-0.5b
Hardware: MacBook Pro M2 (16GB)

Performance Metrics:
- Load Time: 1.2s
- First Token: 0.15s
- Tokens/Second: 28.3
- Memory Usage: 2.1GB
- Power Draw: ~15W

Quality Metrics (vs GPT-4):
- Truthfulness: 68%
- Coherence: 72%
- Helpfulness: 70%
```

## 📊 Understanding the Results

### Key Metrics Explained

1. **Tokens/Second**: Inference speed (higher is better, 10-50+ is good)
2. **Memory Usage**: RAM/VRAM consumption during inference
3. **Load Time**: Time to initialize model in memory (one-time cost)
4. **Quality Score**: Response accuracy and coherence (compared to reference)
5. **Power Consumption**: Energy usage (important for cost calculations)

### Interpreting Local vs Cloud Results

For small models on MacBook Pro, here's what to expect:

**📊 Typical Performance Patterns:**
- **Small local models**: 15-50 tokens/sec, $0 per call, 0.5-2s load time, offline capable
- **Cloud models**: 1-3s total latency, $0.001-0.01 per call, internet required, higher quality
- **Quality comparison**:
  - SmolLM 135M ≈ 60% of GPT-3.5 quality
  - SmolLM 360M ≈ 65% of GPT-3.5 quality
  - Qwen 0.5B ≈ 70% of GPT-3.5 quality
- **Cost breakeven**: Immediate - no API costs ever
- **Best use cases**: Development, testing, learning, privacy-sensitive tasks

**🎯 When to Use Small Local Models:**
- **Perfect for**: Quick prototyping, offline development, learning LLM concepts
- **Good for**: Simple Q&A, basic text generation, cost-free experimentation
- **Not ideal for**: Complex reasoning, professional writing, production applications
- **Hybrid approach**: Small local models for dev/test, cloud APIs for production

### Example Results

```
📊 Small Model Performance on MacBook Pro M2
==================================================
Task: "Explain machine learning in simple terms"
Hardware: MacBook Pro M2 16GB

📈 Model Comparison:
--------------------------------------------------------------------------------
Model            Load Time  Response Time  Tokens/Sec  Memory   Quality
--------------------------------------------------------------------------------
smollm-135m      0.5s      1.8s          42.3        0.5GB    6/10
smollm-360m      0.8s      2.4s          31.2        1.4GB    6.5/10
qwen-0.5b        1.2s      3.1s          23.8        2.0GB    7/10
qwen-0.5b-gguf   0.3s      2.2s          35.6        0.3GB    6.8/10
--------------------------------------------------------------------------------

💡 Best Overall: qwen-0.5b-gguf (fast, efficient, good quality)
💡 Best Quality: qwen-0.5b (highest accuracy of small models)
💡 Best Speed: smollm-135m (fastest generation)
```

### Results Organization

Local model results are saved alongside cloud results for easy comparison:

```bash
# List all results including local models
ls -la results/*/YYYY-MM/

# View local model specific results
ls -la results/local/YYYY-MM/       # Local model results
ls -la results/comparison/YYYY-MM/  # Local vs cloud comparisons

# View detailed benchmark report
cat results/local/2025-01/local_phi-2_benchmark_20250115_143022.json
```

## 🎨 Advanced Usage

### Using GGUF Quantized Models

The GGUF model offers the best performance/efficiency trade-off:

```bash
# Use the quantized model for production workloads
python models/small-llms/inference.py \
  --model qwen-0.5b-gguf \
  --use-metal \
  --context-length 2048 \
  --prompt "Your prompt here"

# Compare quantized vs full precision
python models/small-llms/compare_quantization.py \
  --models qwen-0.5b,qwen-0.5b-gguf \
  --test-prompts "Explain AI" "Write code" "Summarize text"

# Optimize for streaming responses
python models/small-llms/inference.py \
  --model qwen-0.5b-gguf \
  --stream \
  --prompt "Tell me a story"
```

**Quantization Benefits on MacBook Pro:**
- 70% smaller model size
- 30-50% faster inference
- Minimal quality loss (2-5%)
- Lower memory usage
- Better battery life

### Creating Local API Endpoints

Set up a local API server compatible with OpenAI's API:

```python
# models/small-llms/api_server.py
from flask import Flask, request, jsonify
from inference import load_model, generate_response

app = Flask(__name__)

# Load model once at startup
model, tokenizer = load_model("qwen-0.5b-gguf")

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    prompt = data["messages"][-1]["content"]

    response = generate_response(model, tokenizer, prompt)
    return jsonify({
        "model": "qwen-0.5b-local",
        "choices": [{
            "message": {"role": "assistant", "content": response},
            "finish_reason": "stop"
        }]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

```bash
# Start the server
python models/small-llms/api_server.py

# Test with curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

# Use with OpenAI client
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="local"
```

### Batch Processing for Efficiency

Process multiple prompts efficiently:

```bash
# Create batch input file
cat > batch_prompts.txt << EOF
What is machine learning?
Explain quantum computing
Define artificial intelligence
How do neural networks work?
What is deep learning?
EOF

# Process batch with local model
python examples/use_cases/local_model_demo.py \
  --model mistral-7b \
  --batch-file batch_prompts.txt \
  --output batch_results.json

# Compare batch vs individual processing
python examples/use_cases/local_model_demo.py \
  --model phi-2 \
  --benchmark-batch-processing
```

### Optimizing for MacBook Pro Battery Life

When running on battery, optimize for efficiency:

```bash
# Low power mode settings
python models/small-llms/inference.py \
  --model smollm-135m \
  --low-power \
  --max-tokens 128 \
  --prompt "Quick response needed"

# Monitor power usage
python models/small-llms/power_monitor.py \
  --model qwen-0.5b-gguf \
  --duration 60  # Monitor for 60 seconds

# Batch processing to reduce overhead
python models/small-llms/batch_efficient.py \
  --model smollm-360m \
  --input-file prompts.txt \
  --power-efficient
```

**Battery Optimization Tips:**
- Use quantized models (GGUF)
- Limit max tokens
- Process in batches
- Use CPU-only mode when on battery
- Close other applications

## 🎯 Pro Tips for MacBook Pro Users

💡 **Start with SmolLM-135M**: Fastest model to verify your setup works correctly

💡 **Use GGUF for Production**: The quantized Qwen model offers the best efficiency

💡 **Metal Acceleration**: Always enabled on Apple Silicon - no configuration needed

💡 **Memory Management**: These small models leave plenty of RAM for other apps

💡 **Quick Model Selection**:
  - Testing/Learning: `smollm-135m` (instant responses)
  - Development: `smollm-360m` (good balance)
  - Best Quality: `qwen-0.5b` (most capable)
  - Production: `qwen-0.5b-gguf` (optimized)

💡 **MacBook Pro Specific**:
  - M1/M2/M3 chips excel at running these models
  - Use Activity Monitor to track GPU usage
  - Models automatically use Neural Engine when available
  - Battery life impact is minimal with small models

## 🐛 Troubleshooting

### Common MacBook Pro Issues and Solutions

#### Issue: "MPS backend out of memory"
**Solution**:
```bash
# Use smaller batch size
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python models/small-llms/inference.py \
  --model qwen-0.5b \
  --batch-size 1 \
  --prompt "Your prompt"

# Or fallback to CPU
python models/small-llms/inference.py \
  --model smollm-135m \
  --device cpu \
  --prompt "Your prompt"
```

#### Issue: Slow first run on Apple Silicon
**Solution**: Metal shaders compile on first use
```bash
# Warm up the model
python models/small-llms/warmup.py --model qwen-0.5b-gguf

# Subsequent runs will be much faster
```

#### Issue: Models not found
**Solution**: Ensure you're in the correct directory
```bash
# Run from repository root
cd /path/to/llm-lab
python models/small-llms/inference.py --model smollm-135m

# Or use absolute paths
python /full/path/to/inference.py \
  --model-path /Users/ro/Documents/GitHub/llm-lab/models/small-llms/smollm-135m
```

#### Issue: High memory usage warning
**Solution**: These models are tiny - ignore if < 3GB
```bash
# Check actual memory usage
python models/small-llms/memory_check.py

# These models use:
# smollm-135m: ~500MB
# smollm-360m: ~1.4GB
# qwen-0.5b: ~2GB
# qwen-0.5b-gguf: ~300MB
```

### Debugging Commands

```bash
# Test model loading
python -c "
import sys
sys.path.append('models/small-llms')
from inference import load_model
model, tokenizer = load_model('smollm-135m')
print('✓ Model loaded successfully')
"

# Check Metal/MPS availability (Apple Silicon)
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"

# Monitor system resources on macOS
# In Terminal 1:
top -o cpu  # CPU usage

# In Terminal 2:
sudo powermetrics --samplers gpu_power -i 1000  # GPU usage (requires admin)

# Check model files
ls -lah models/small-llms/*/
du -sh models/small-llms/*  # Check sizes
```

## 📈 Next Steps

Now that you've mastered small local models:

1. **Build a Local Chatbot**: Create an interactive assistant
   ```bash
   python models/small-llms/chatbot_demo.py --model qwen-0.5b-gguf
   ```

2. **Set up API Server**: Replace OpenAI API with local endpoint
   ```bash
   python models/small-llms/api_server.py --model qwen-0.5b-gguf --port 8000
   ```

3. **Compare All Models**: Run comprehensive comparison
   ```bash
   python models/small-llms/compare_all_models.py --output-report comparison.html
   ```

4. **Explore Larger Models**: When ready for more capability
   ```bash
   # Download slightly larger but still efficient models
   python models/small-llms/download_small_models.py --model tinyllama
   ```

### Related Use Cases
- [Use Case 1: Benchmarking](./USE_CASE_1_HOW_TO.md) - Include local models in benchmarks
- [Use Case 2: Cost Analysis](./USE_CASE_2_HOW_TO.md) - Compare $0 local vs cloud costs
- [Use Case 3: Custom Prompts](./USE_CASE_3_HOW_TO.md) - Test prompts locally first

## 📚 Understanding Your Small Model Collection

The pre-downloaded models each serve different purposes:

### SmolLM-135M (135M Parameters)
HuggingFace's ultra-efficient model:
- **Best for**: Instant responses, basic Q&A, learning/testing
- **Speed**: 40+ tokens/sec on MacBook Pro
- **Quality**: Basic but functional responses
- **Example Use**: "What is Python?" → Simple, direct answers

### SmolLM-360M (360M Parameters)
Larger SmolLM with better quality:
- **Best for**: General chat, explanations, development
- **Speed**: 30+ tokens/sec on MacBook Pro
- **Quality**: Noticeable improvement over 135M
- **Example Use**: "Explain how databases work" → Clear explanations

### Qwen-0.5B (500M Parameters)
Alibaba's capable small model:
- **Best for**: More complex tasks, better reasoning
- **Speed**: 20+ tokens/sec on MacBook Pro
- **Quality**: Best of the small models
- **Example Use**: "Write a function to sort a list" → Decent code generation

### Qwen-0.5B-GGUF (500M Parameters, Quantized)
Optimized version of Qwen:
- **Best for**: Production use, API serving, efficiency
- **Speed**: 35+ tokens/sec with Metal acceleration
- **Quality**: 95% of full model quality at 70% size
- **Example Use**: Production API endpoint for cost-free inference

## 🔄 Why Small Models on MacBook Pro?

Small models are perfect for MacBook Pro users because:
- **Zero Cost**: No API fees, unlimited usage
- **Always Available**: Work offline, no internet required
- **Fast Iteration**: Test ideas instantly without rate limits
- **Privacy First**: Your data never leaves your machine
- **Learn by Doing**: Understand LLMs without cloud complexity
- **Energy Efficient**: Minimal battery impact

## 📚 Additional Resources

### Model Information
- **SmolLM Models**: [HuggingFace SmolLM Collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)
- **Qwen Models**: [Qwen Official Page](https://huggingface.co/Qwen)
- **GGUF Format**: [Understanding Quantization](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)

### MacBook Optimization
- **Apple Metal**: [PyTorch Metal Acceleration](https://pytorch.org/docs/stable/notes/mps.html)
- **Power Management**: [macOS Energy Guide](https://support.apple.com/guide/mac-help/mchl11a6d54f/mac)

### Example Code
- **Inference Script**: `models/small-llms/inference.py`
- **Quick Demo**: `models/small-llms/quick_demo.py`
- **API Server**: `models/small-llms/api_server.py`

## 💭 Feedback

Help us improve this guide:
- Found an error? [Open an issue](https://github.com/remyolson/llm-lab/issues/new)
- Have suggestions? See our [Contribution Guide](../../CONTRIBUTING.md)

---

*Last updated: January 2025*
