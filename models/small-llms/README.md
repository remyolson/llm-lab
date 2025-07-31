# Small LLMs for MacBook Pro

This directory contains very small language models that can run efficiently on MacBook Pro using CPU or Apple Metal acceleration.

## Downloaded Models

### SmolLM2-135M-Instruct
- **Size**: 135M parameters (~270MB)
- **Speed**: ~17-20 tokens/sec on M-series
- **Use case**: Ultra-fast responses, basic Q&A, simple tasks
- **Quality**: Limited but functional for basic queries

### SmolLM2-360M-Instruct  
- **Size**: 360M parameters (~720MB)
- **Speed**: ~10-15 tokens/sec on M-series
- **Use case**: Better quality than 135M, still very fast
- **Quality**: Reasonable for simple conversations and tasks

### Qwen2.5-0.5B-Instruct
- **Size**: 500M parameters (~1GB)
- **Speed**: ~8-12 tokens/sec on M-series
- **Use case**: Best quality among the tiny models
- **Quality**: Good for instruction following, coding, and general chat

### Qwen2.5-0.5B-GGUF (Quantized)
- **Size**: ~300MB (4-bit quantized)
- **Format**: GGUF for llama.cpp
- **Use case**: Even faster inference with slight quality trade-off

## Quick Start

### 1. Interactive Demo
```bash
# Activate virtual environment
source venv/bin/activate

# Run the interactive demo
python models/small-llms/quick_demo.py
```

### 2. Programmatic Usage
```python
from inference import SmallLLMInference

# Load model
llm = SmallLLMInference("models/small-llms/qwen-0.5b")
llm.load_transformers_model()

# Generate text
result = llm.chat("What is Python?", max_new_tokens=100)
print(result['response'])
print(f"Speed: {result['tokens_per_second']:.1f} tokens/sec")
```

### 3. Download More Models
```bash
# Edit download_small_models.py to select different models
python models/small-llms/download_small_models.py
```

## Available Models (Not Downloaded)

### Tiny Models (<200M params)
- **pythia-70m**: 70M params - Extremely small GPT-NeoX
- **pythia-160m**: 160M params - Slightly larger GPT-NeoX

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

| Model | Size | Speed (tok/s) | Quality | Best For |
|-------|------|---------------|---------|----------|
| SmolLM-135M | 270MB | 17-20 | Basic | Ultra-fast responses |
| SmolLM-360M | 720MB | 10-15 | Good | Balanced speed/quality |
| Qwen-0.5B | 1GB | 8-12 | Better | Best tiny model quality |
| Pythia-70M | 140MB | 20-25 | Limited | Experiments, embedding |
| TinyLlama-1.1B | 2.2GB | 4-8 | Good | Longer conversations |

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

1. **Out of Memory**: Use smaller models or reduce batch size
2. **Slow on CPU**: Ensure MPS/Metal is being used (check device in output)
3. **Import Errors**: Make sure virtual environment is activated
4. **Model Not Found**: Run download script first