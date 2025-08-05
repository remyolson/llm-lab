# Use Case 5: Local Model Development and Testing

*Run and evaluate local LLMs on your own hardware for cost-free development and experimentation.*

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:
- **Download and run local LLMs** (Llama, Mistral, Phi) on your hardware
- **Compare local vs cloud models** for performance, cost, and quality trade-offs
- **Optimize models for your hardware** with GPU acceleration and quantization
- **Develop and test offline** without API costs or internet dependency
- **Benchmark local models** using the same framework as cloud providers
- **Fine-tune and customize** local models for your specific use cases
- **Create local API endpoints** for seamless integration with existing applications

## 📋 Before You Begin

- Complete all [Prerequisites](./PREREQUISITES.md)
- Ensure you have sufficient disk space (5-50GB depending on models)
- Time required: ~60 minutes (including model download)
- Estimated ongoing cost: $0.00 (local compute only)

### 💰 Hardware Requirements and Costs

Running local models requires upfront hardware investment but $0 ongoing costs:

**💡 Pro Tip:** Start with quantized 7B models to test on modest hardware, then scale up as needed

- **CPU-Only Setup**:
  - **7B models**: 16GB RAM, ~2-5 tokens/sec (budget-friendly)
  - **13B models**: 32GB RAM, ~1-3 tokens/sec (moderate performance)
  - **Free ongoing costs**: $0 per inference vs $0.001-0.01 per API call

- **GPU-Accelerated Setup**:
  - **RTX 4060 Ti (16GB)**: 7B-13B models, ~15-50 tokens/sec
  - **RTX 4090 (24GB)**: Up to 30B models, ~25-80 tokens/sec
  - **Free ongoing costs**: Electricity only (~$0.0001 per inference)

- **Apple Silicon**:
  - **M1/M2 Mac (16GB)**: 7B models, ~10-25 tokens/sec (excellent efficiency)
  - **M1/M2 Mac (32GB+)**: Up to 13B models, ~8-20 tokens/sec
  - **Free ongoing costs**: Very low power consumption

*Note: Performance varies by model complexity and quantization level. Initial hardware investment pays off after ~1000-10000 API calls.*

## 📊 Available Local Models

Choose the right model based on your hardware capabilities and use case:

| Model | Size | What It's Best For | Hardware Needs | Example Use Case |
|-------|------|-------------------|----------------|------------------|
| **phi-2** | 2.7B (1.6GB) | Fast inference, simple tasks | 8GB RAM, any GPU | Quick testing, simple Q&A |
| **mistral-7b** | 7B (4.1GB) | Balanced performance/efficiency | 16GB RAM, 8GB VRAM | General purpose, coding help |
| **llama-2-7b** | 7B (3.8GB) | Instruction following, chat | 16GB RAM, 8GB VRAM | Conversational AI, analysis |
| **llama-2-13b** | 13B (7.3GB) | Higher quality responses | 32GB RAM, 16GB VRAM | Complex reasoning, writing |
| **codellama-7b** | 7B (3.8GB) | Code generation and review | 16GB RAM, 8GB VRAM | Programming assistance, debugging |

### 🎯 **Model Selection Guide:**

- **🔍 For testing/learning:** Start with `phi-2` (fastest download and setup)
- **🧮 For general development:** Use `mistral-7b` (best balance of size/quality)
- **🎓 For production applications:** Choose `llama-2-13b` (highest quality)
- **🌍 For coding tasks:** Select `codellama-7b` (specialized for programming)
- **📊 For comprehensive testing:** Download multiple models and compare

## 🚀 Step-by-Step Guide

### Step 1: Install Local Model Dependencies

First, install the required dependencies for local model support:

```bash
# Install llama-cpp-python for local inference
pip install llama-cpp-python

# For GPU acceleration (NVIDIA)
pip install llama-cpp-python[cuda]

# For Apple Silicon optimization
pip install llama-cpp-python[metal]
```

### Step 2: Download Your First Local Model

Start with a small, fast model to verify setup:

```bash
# Quick test with Phi-2 model (recommended for first run)
python -m src.use_cases.local_models.download_helper --model phi-2

# Download more capable models
python -m src.use_cases.local_models.download_helper --model mistral-7b    # Balanced performance
python -m src.use_cases.local_models.download_helper --model llama-2-7b    # Instruction following
python -m src.use_cases.local_models.download_helper --model codellama-7b  # Code generation

# Check available models and download status
python -m src.use_cases.local_models.download_helper --list
```

**What Happens:**
- Models are downloaded to `~/.cache/lllm-lab/models/` directory
- Progress is displayed during download (can take 5-30 minutes depending on model size)
- Models are validated after download to ensure integrity
- GGUF format models are optimized for efficient inference

**Expected Output:**
```
📥 Downloading phi-2 model...
✓ Model downloaded: ~/.cache/lllm-lab/models/phi-2.gguf (1.6GB)
✓ Model validation successful
✓ Ready for inference
```

### Step 3: Run Your First Local Inference

Test the downloaded model with a simple prompt:

```bash
# Quick test with Phi-2 (recommended first test)
python examples/use_cases/local_model_demo.py --model phi-2 --prompt "What is machine learning?"

# Test different models with same prompt
python examples/use_cases/local_model_demo.py --model mistral-7b --prompt "Explain quantum computing"
python examples/use_cases/local_model_demo.py --model llama-2-7b --prompt "Write a Python function to sort a list"

# Interactive chat mode
python examples/use_cases/local_model_demo.py --model phi-2 --interactive
```

**Expected Output:**
```
🖥️  Loading local model: phi-2
✓ Model loaded successfully (2.1s)
🧠 Generating response...

Machine learning is a subset of artificial intelligence that enables
computers to learn and improve from data without being explicitly programmed...
✓ Response generated (3.4s, ~12 tokens/sec)
```

### Step 4: Compare Local vs Cloud Performance

Run side-by-side comparisons with cloud providers:

```bash
# Compare local Phi-2 with cloud models (recommended comparison)
python scripts/run_benchmarks.py \
  --providers local,openai \
  --models phi-2,gpt-4o-mini \
  --custom-prompt "Explain the benefits of renewable energy" \
  --limit 3

# Compare multiple local models with cloud alternatives
python scripts/run_benchmarks.py \
  --providers local,anthropic \
  --models mistral-7b,claude-3-5-haiku-20241022 \
  --custom-prompt "Write a professional email about a project delay" \
  --limit 2

# Comprehensive local vs cloud comparison
declare -a prompts=(
  "What is artificial intelligence?"
  "Write a short story about space exploration"
  "Explain photosynthesis in simple terms"
)

for prompt in "${prompts[@]}"; do
  echo "Comparing: $prompt"
  python scripts/run_benchmarks.py \
    --providers local,openai \
    --models phi-2,gpt-4o-mini \
    --custom-prompt "$prompt" \
    --limit 1
done
```

### Step 5: Hardware Optimization

Optimize performance for your specific hardware:

```bash
# CPU-only inference (recommended for testing)
python examples/use_cases/local_model_demo.py \
  --model phi-2 \
  --cpu-only \
  --threads 4 \
  --prompt "Test CPU performance"

# GPU acceleration (if available)
python examples/use_cases/local_model_demo.py \
  --model mistral-7b \
  --gpu-layers 32 \
  --prompt "Test GPU acceleration"

# Memory optimization for larger models
python examples/use_cases/local_model_demo.py \
  --model llama-2-13b \
  --gpu-layers 20 \
  --batch-size 256 \
  --context-size 1024 \
  --prompt "Test with optimized settings"
```

### Step 6: Benchmark Local Model Performance

Run comprehensive benchmarks to measure performance:

```bash
# Performance benchmark on reasoning tasks
python scripts/run_benchmarks.py \
  --providers local \
  --models phi-2,mistral-7b \
  --datasets arc \
  --limit 10

# Quality comparison across model sizes
python scripts/run_benchmarks.py \
  --providers local \
  --models phi-2,mistral-7b,llama-2-7b \
  --custom-prompt "Solve this math problem: If a train travels 60 mph for 2 hours, how far does it go?" \
  --limit 5

# Batch processing efficiency test
python examples/use_cases/local_model_demo.py \
  --model mistral-7b \
  --batch-prompts \
  --prompts "What is AI?" "Define machine learning" "Explain neural networks"
```

**💡 Pro Tip:** Start with `--limit 10` on reasoning benchmarks to test performance quickly, then scale up for comprehensive evaluation.

## 📊 Understanding the Results

### Key Metrics Explained

1. **Tokens/Second**: Inference speed (higher is better, 10-50+ is good)
2. **Memory Usage**: RAM/VRAM consumption during inference
3. **Load Time**: Time to initialize model in memory (one-time cost)
4. **Quality Score**: Response accuracy and coherence (compared to reference)
5. **Power Consumption**: Energy usage (important for cost calculations)

### Interpreting Local vs Cloud Results

Different aspects reveal trade-offs between local and cloud deployment:

**📊 Typical Performance Patterns:**
- **Local models**: 5-50 tokens/sec, $0 per call, 2-10s load time, offline capable
- **Cloud models**: Instant response, $0.001-0.01 per call, internet required, consistent performance
- **Quality comparison**: 7B local ≈ GPT-3.5 quality, 13B+ local ≈ GPT-4 mini quality
- **Cost breakeven**: ~1000-10000 calls depending on model and hardware
- **Latency**: Local can be faster for simple tasks, cloud better for complex reasoning

**🎯 When to Use Local vs Cloud:**
- **Use local for**: High-volume tasks, sensitive data, offline work, cost optimization
- **Use cloud for**: Occasional use, highest quality needs, specialized tasks, no hardware constraints
- **Hybrid approach**: Local for development/testing, cloud for production quality

### Example Results

```
📊 Local vs Cloud Model Comparison
==================================================
Task: "Explain machine learning in simple terms"
Models: phi-2 (local) vs gpt-4o-mini (cloud)

📈 Performance Comparison:
--------------------------------------------------------------------------------
Model                          Response Time  Cost       Quality    Tokens/Sec
--------------------------------------------------------------------------------
phi-2 (local)                  4.2s          $0.000     7.5/10     15.3
gpt-4o-mini (cloud)           1.8s          $0.003     8.2/10     N/A
--------------------------------------------------------------------------------

💡 Analysis: Local model offers 100% cost savings with 90% of the quality
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

### Model Quantization for Better Performance

Use different quantization levels to balance quality and speed:

```bash
# Download different quantization levels
python -m src.use_cases.local_models.download_helper --model llama-2-7b --quantization q4_0  # 4-bit (fastest)
python -m src.use_cases.local_models.download_helper --model llama-2-7b --quantization q5_1  # 5-bit (balanced)
python -m src.use_cases.local_models.download_helper --model llama-2-7b --quantization f16   # 16-bit (highest quality)

# Compare quantization performance
python examples/use_cases/local_model_demo.py \
  --compare-quantization \
  --model llama-2-7b \
  --prompt "Write a technical explanation of neural networks"
```

### Creating Local API Endpoints

Set up a local API server for integration with existing applications:

```python
# local_api_server.py
from src.use_cases.local_models import LocalModelProvider
from flask import Flask, request, jsonify

app = Flask(__name__)
provider = LocalModelProvider(model_name="mistral-7b")

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    prompt = data["messages"][-1]["content"]
    
    response = provider.complete(prompt)
    return jsonify({
        "choices": [{"message": {"content": response["content"]}}]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

```bash
# Start local API server
python local_api_server.py

# Test the API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello, how are you?"}]}'
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

### Fine-tuning Local Models

Customize models for your specific use case:

```bash
# Prepare training data
python examples/use_cases/fine_tuning_demo.py \
  --prepare-data \
  --input custom_dataset.jsonl \
  --output training_data.json

# Fine-tune local model
python examples/use_cases/fine_tuning_demo.py \
  --model llama-2-7b \
  --train training_data.json \
  --epochs 3 \
  --output fine_tuned_model

# Test fine-tuned model
python examples/use_cases/local_model_demo.py \
  --model ./fine_tuned_model \
  --prompt "Test fine-tuned capabilities"
```

## 🎯 Pro Tips

💡 **Start Small**: Begin with `phi-2` to test your setup before downloading larger models

💡 **Use Quantization**: 4-bit quantized models are 75% smaller with minimal quality loss

💡 **GPU Memory Management**: Monitor VRAM usage and adjust `--gpu-layers` accordingly

💡 **Batch Processing**: Process multiple prompts together for 2-5x better throughput

💡 **Model Caching**: Keep frequently used models loaded in memory to avoid reload time

💡 **Choose Models Wisely**:
  - For speed: `phi-2` (2.7B params)
  - For balance: `mistral-7b` (7B params)
  - For quality: `llama-2-13b` (13B params)
  - For coding: `codellama-7b` (7B params, code-specialized)

💡 **Hardware Optimization**: Use GPU acceleration when available, but CPU-only is perfectly viable for development

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue: Out of memory errors
**Solution**: 
```bash
# Reduce context size and batch size
python examples/use_cases/local_model_demo.py \
  --model mistral-7b \
  --context-size 512 \
  --batch-size 128 \
  --prompt "Your prompt here"

# Use CPU-only mode
python examples/use_cases/local_model_demo.py \
  --model phi-2 \
  --cpu-only \
  --prompt "Your prompt here"
```

#### Issue: Slow inference speed
**Solution**: Enable GPU acceleration and optimize settings
```bash
# Optimize for speed
python examples/use_cases/local_model_demo.py \
  --model phi-2 \
  --gpu-layers -1 \
  --threads 8 \
  --batch-size 512
```

#### Issue: Model download failures
**Solution**: Check network connection and disk space
```bash
# Check available disk space
df -h ~/.cache/lllm-lab/

# Retry download with verbose output
python -m src.use_cases.local_models.download_helper \
  --model phi-2 \
  --verbose \
  --retry 3
```

#### Issue: llama-cpp-python installation problems
**Solution**: Install from source or use conda
```bash
# Install from source with GPU support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Or use conda-forge
conda install -c conda-forge llama-cpp-python
```

### Debugging Commands

```bash
# Test model loading
python -c "
from src.use_cases.local_models import LocalModelProvider
provider = LocalModelProvider('phi-2')
print('Model loaded successfully')
"

# Check GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# Monitor system resources
htop  # or top on macOS
nvidia-smi  # for NVIDIA GPUs
```

## 📈 Next Steps

Now that you've mastered local model development:

1. **Fine-tune Models**: Customize local models for your specific use case
   ```bash
   python examples/use_cases/fine_tuning_demo.py
   ```

2. **Set up Production Serving**: Create robust local API endpoints
   ```bash
   python examples/use_cases/local_api_server.py
   ```

3. **Cost Analysis**: Calculate ROI compared to cloud APIs
   ```bash
   python examples/use_cases/cost_scenarios.py --include-local
   ```

4. **Hybrid Workflows**: Combine local and cloud models strategically

### Related Use Cases
- [Use Case 6: Fine-tuning](./USE_CASE_6_HOW_TO.md) - Customize local models for your domain
- [Use Case 4: Cross-LLM Testing](./USE_CASE_4_HOW_TO.md) - Include local models in testing suites
- [Use Case 2: Cost Analysis](./USE_CASE_2_HOW_TO.md) - Compare local vs cloud economics

## 📚 Understanding Local Model Ecosystem

Each model family has different strengths and characteristics:

### Phi-2 (2.7B Parameters)
Microsoft's efficient small model:
- **Best for**: Quick testing, simple Q&A, educational use
- **Strengths**: Fast inference, low memory usage, good reasoning for size
- **Example**: "Explain a concept in simple terms"

### Mistral 7B (7B Parameters)
Balanced performance and efficiency:
- **Best for**: General-purpose applications, balanced quality/speed
- **Strengths**: Strong instruction following, good multilingual support
- **Example**: "Write a professional email or analyze a document"

### Llama 2 (7B/13B Parameters)
Meta's foundation models:
- **Best for**: Conversational AI, reasoning tasks, content generation
- **Strengths**: Strong chat capabilities, good safety training
- **Example**: "Have a natural conversation or solve complex problems"

### CodeLlama (7B Parameters)
Specialized for programming tasks:
- **Best for**: Code generation, debugging, technical documentation
- **Strengths**: Understanding code context, multiple programming languages
- **Example**: "Generate Python functions or explain code snippets"

## 🔄 Continuous Improvement

This local model framework provides a foundation for:
- **Cost-free development**: Unlimited experimentation without API costs
- **Privacy-preserving AI**: Keep sensitive data on your hardware
- **Offline capabilities**: Work without internet connectivity
- **Custom model development**: Fine-tune models for specific domains
- **Hybrid deployments**: Strategic combination of local and cloud resources

## 📚 Additional Resources

- **Model Sources**: 
  - [Hugging Face Models](https://huggingface.co/models?library=gguf)
  - [Ollama Model Library](https://ollama.ai/library)
  - [LLaMA.cpp Models](https://github.com/ggerganov/llama.cpp#description)
- **Documentation**: [Local Models Guide](../../examples/use_cases/local_model_demo.py)
- **Hardware Guides**: [GPU Optimization](../HARDWARE_OPTIMIZATION.md)

## 💭 Feedback

Help us improve this guide:
- Found an error? [Open an issue](https://github.com/yourusername/lllm-lab/issues/new)
- Have suggestions? See our [Contribution Guide](../../CONTRIBUTING.md)

---

*Last updated: January 2025*