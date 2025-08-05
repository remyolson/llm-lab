# Small LLMs for MacBook Pro

This directory contains very small language models that can run efficiently on MacBook Pro using CPU or Apple Metal acceleration. Models are available in two formats: **Transformers** (Python library) and **Ollama** (optimized local runtime).

## 🎯 Default Model Set

The repository includes 8 models (6 small + 2 large):
- **Pythia-70M & Pythia-160M**: Fastest inference speeds (50-70 tokens/sec)
- **SmolLM-135M & SmolLM-360M**: Great balance of speed and quality
- **Qwen-0.5B**: Best quality among tiny models
- **Qwen-0.5B-GGUF**: Quantized version for production efficiency
- **Llama3.2-1B** ⭐: High-quality 1B model via Ollama
- **GPT-OSS-20B**: OpenAI's powerful 20B model via Ollama (⚠️ LARGE)

## Downloaded Models

### Pythia-70M
- **Size**: 70M parameters (~280MB)
- **Speed**: ~50-70 tokens/sec on M-series
- **Use case**: Extremely fast inference, basic experiments, embedding tasks
- **Quality**: Very limited but fastest possible generation

### Pythia-160M
- **Size**: 160M parameters (~640MB)
- **Speed**: ~40-60 tokens/sec on M-series
- **Use case**: Very fast responses, simple generation tasks
- **Quality**: Basic but better than 70M variant

### SmolLM2-135M-Instruct
- **Size**: 135M parameters (~270MB)
- **Speed**: ~30-50 tokens/sec on M-series
- **Use case**: Ultra-fast responses, basic Q&A, simple tasks
- **Quality**: Limited but functional for basic queries

### SmolLM2-360M-Instruct  
- **Size**: 360M parameters (~720MB)
- **Speed**: ~20-40 tokens/sec on M-series
- **Use case**: Better quality than 135M, still very fast
- **Quality**: Reasonable for simple conversations and tasks

### Qwen2.5-0.5B-Instruct
- **Size**: 500M parameters (~1GB)
- **Speed**: ~15-30 tokens/sec on M-series
- **Use case**: Best quality among the tiny models
- **Quality**: Good for instruction following, coding, and general chat

### Qwen2.5-0.5B-GGUF (Quantized)
- **Size**: ~300MB (4-bit quantized)
- **Speed**: ~25-45 tokens/sec on M-series
- **Format**: GGUF for llama.cpp
- **Use case**: Even faster inference with slight quality trade-off

### Llama3.2-1B ⭐ (Ollama)
- **Size**: 1B parameters (~700MB via Ollama)
- **Speed**: ~10-20 tokens/sec on M-series
- **Use case**: High-quality chat, code generation, reasoning
- **Quality**: Excellent balance of size and capability
- **Format**: Ollama optimized (automatically quantized)

### GPT-OSS-20B ⚠️ (Large Model via Ollama)
- **Size**: 20B parameters (~13GB via Ollama)
- **Speed**: ~2-5 tokens/sec on M-series (varies by RAM)
- **Use case**: High-quality generation, complex reasoning
- **Quality**: Comparable to GPT-3.5 level performance
- **Format**: Ollama optimized (automatically quantized)

## ⚡ Super Quick Start

**New to this? Run our one-command setup:**
```bash
# One-click setup for macOS (installs Ollama + downloads models)
./models/small-llms/quick_setup.sh
```

This will:
- ✅ Install Ollama (if needed)
- ✅ Download Llama3.2-1B (highly recommended)
- ⚠️ Optionally download GPT-OSS-20B (large model)
- 🧪 Provide test commands

Then you're ready to go!

## Manual Setup

### 0. Setup Ollama (Recommended for best performance)
```bash
# Install Ollama for optimized model performance
brew install ollama

# Start Ollama service
brew services start ollama

# Download high-quality models via Ollama
ollama pull llama3.2:1b    # Excellent 1B model
ollama pull gpt-oss:20b    # Large 20B model (requires more RAM)
```

### 1. Interactive Demo
```bash
# Activate virtual environment
source venv/bin/activate

# Run the interactive demo
python models/small-llms/quick_demo.py

# Test Ollama models directly
python models/small-llms/run_small_model_demo.py --model llama3.2-1b "Hello world!"
python models/small-llms/run_small_model_demo.py --model gpt-oss-20b "Explain quantum computing"
```

### 2. Programmatic Usage
```python
from inference import SmallLLMInference

# Load fastest model (Pythia-70M)
llm = SmallLLMInference("models/small-llms/pythia-70m")
llm.load_transformers_model()

# Generate text
result = llm.chat("What is Python?", max_new_tokens=100)
print(result['response'])
print(f"Speed: {result['tokens_per_second']:.1f} tokens/sec")

# Or use higher quality model
llm = SmallLLMInference("models/small-llms/qwen-0.5b")
llm.load_transformers_model()
result = llm.chat("Explain machine learning", max_new_tokens=200)
```

### 3. Download More Models
```bash
# Download default set (all small models for transformers)
python models/small-llms/download_small_models.py

# Setup Ollama models (recommended)
python models/small-llms/download_small_models.py --setup-ollama

# Download only small models (skip large ones)
python models/small-llms/download_small_models.py --small-only

# Download specific models only
python models/small-llms/download_small_models.py --models pythia-70m,smollm-135m

# List available models without downloading
python models/small-llms/download_small_models.py --list

# Download only GGUF quantized models
python models/small-llms/download_small_models.py --gguf-only

# Download all available models (including larger ones)
python models/small-llms/download_small_models.py --all

# Download ANY custom model from HuggingFace
python models/small-llms/download_small_models.py --custom microsoft/phi-1_5 --custom-name phi-1.5

# Download a custom GGUF quantized model
python models/small-llms/download_small_models.py \
  --custom TheBloke/Llama-2-7B-GGUF \
  --custom-gguf \
  --gguf-file llama-2-7b.Q4_K_M.gguf \
  --custom-name llama2-7b-gguf
```

## 🎯 Download Custom Models from HuggingFace

You can download ANY model from HuggingFace using the `--custom` flag:

### Examples:
```bash
# Download Microsoft's Phi-1.5 (1.3B params)
python models/small-llms/download_small_models.py \
  --custom microsoft/phi-1_5 \
  --custom-name phi-1.5

# Download Facebook's OPT-125M
python models/small-llms/download_small_models.py \
  --custom facebook/opt-125m \
  --custom-name opt-125m

# Download a specific GGUF file from a quantized model repo
python models/small-llms/download_small_models.py \
  --custom TheBloke/CodeLlama-7B-Instruct-GGUF \
  --custom-gguf \
  --gguf-file codellama-7b-instruct.Q4_K_M.gguf \
  --custom-name codellama-7b

# Download Cerebras GPT models
python models/small-llms/download_small_models.py \
  --custom cerebras/Cerebras-GPT-111M \
  --custom-name cerebras-111m
```

### Custom Model Tips:
- **Model ID**: Use the full HuggingFace model ID (e.g., `meta-llama/Llama-2-7b-hf`)
- **Custom Name**: Specify a local folder name with `--custom-name` (defaults to model name)
- **GGUF Models**: For quantized models, use `--custom-gguf` and specify the exact file with `--gguf-file`
- **Size Warning**: Be mindful of model sizes - some models are very large!

## Additional Available Models (Not Downloaded by Default)

### Small Models (1-3B params)
- **TinyLlama-1.1B**: 1.1B params - Small but quite capable
- **phi-2**: 2.7B params - Microsoft's efficient model

## Performance Tips

1. **Use Metal (MPS)**: Models automatically use Apple Metal when available
2. **Batch Processing**: Process multiple inputs together for efficiency
3. **Token Limits**: Keep max_new_tokens low (50-200) for faster responses
4. **Model Selection**:
   - Use 135M for ultra-fast, simple responses
   - Use 360M for better quality with good speed
   - Use 500M+ for best quality among tiny models

## GGUF Models with llama.cpp

For even better performance, use GGUF quantized models with llama.cpp:

```bash
# Install llama-cpp-python (requires cmake)
brew install cmake
pip install llama-cpp-python

# Use GGUF model
from llama_cpp import Llama

llm = Llama(
    model_path="models/small-llms/qwen-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    n_gpu_layers=-1,  # Use all layers on GPU
    n_ctx=2048,
)

response = llm("Q: What is AI? A:", max_tokens=100)
print(response['choices'][0]['text'])
```

## Model Comparison

| Model | Size | Speed (tok/s) | Quality | Best For | Format |
|-------|------|---------------|---------|----------|--------|
| Pythia-70M | 280MB | 50-70 | Very Limited | Fastest possible inference | Transformers |
| Pythia-160M | 640MB | 40-60 | Limited | Very fast generation | Transformers |
| SmolLM-135M | 270MB | 30-50 | Basic | Ultra-fast responses | Transformers |
| SmolLM-360M | 720MB | 20-40 | Good | Balanced speed/quality | Transformers |
| Qwen-0.5B | 1GB | 15-30 | Better | Best tiny model quality | Transformers |
| Qwen-0.5B-GGUF | 300MB | 25-45 | Good | Efficient production use | GGUF |
| **Llama3.2-1B** ⭐ | 700MB | 10-20 | **Excellent** | **High-quality chat & code** | **Ollama** |
| **GPT-OSS-20B** ⚠️ | 13GB | 2-5 | **Outstanding** | **Complex reasoning** | **Ollama** |
| TinyLlama-1.1B* | 2.2GB | 8-15 | Good | Longer conversations | Transformers |
| Phi-2* | 2.7GB | 5-10 | Very Good | Complex reasoning | Transformers |

*Not downloaded by default, use `--all` flag to include
⭐ **Recommended**: Ollama models offer the best quality-to-size ratio
⚠️ Large model: May require 16GB+ RAM depending on system

## Use Cases

### Good For:
- Quick prototyping and experiments
- Local development without internet
- Privacy-sensitive applications  
- Learning about LLMs
- Simple chatbots and assistants
- Basic code completion
- Text classification

### Limitations:
- Complex reasoning tasks
- Long-form content generation
- Advanced coding problems
- Multilingual support (varies by model)
- Factual accuracy (smaller = less accurate)

## Troubleshooting

### General Issues
1. **Out of Memory**: Use smaller models or reduce batch size
2. **Slow on CPU**: Ensure MPS/Metal is being used (check device in output)
3. **Import Errors**: Make sure virtual environment is activated
4. **Model Not Found**: Run download script first

### Ollama Issues
5. **Ollama not found**: Install with `brew install ollama` and start with `brew services start ollama`
6. **Model not available**: Download with `ollama pull llama3.2:1b` or `ollama pull gpt-oss:20b`
7. **Ollama slow to respond**: First run may be slower while model loads into memory
8. **Connection refused**: Restart Ollama service with `brew services restart ollama`

### Legacy GPT-OSS-20B Issues (Transformers)
9. **GPT-OSS-20B via transformers**:
   - **Won't load**: Use Ollama instead (`ollama pull gpt-oss:20b`)
   - **MXFP4 errors**: Ollama handles quantization automatically
   - **Memory issues**: Ollama is much more memory efficient