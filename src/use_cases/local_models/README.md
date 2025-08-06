# Local Model Integration for LLM Lab

This module provides integration for running local models within the LLM Lab benchmark framework, enabling cost-free, private, and offline LLM inference.

## Features

- **GGUF Model Support**: Compatible with quantized models in GGUF format
- **Automatic GPU Detection**: Intelligent GPU layer offloading for optimal performance
- **Memory Management**: Built-in memory monitoring and management
- **Streaming Responses**: Token-by-token streaming for better UX
- **Model Download Helper**: Automated downloading from Hugging Face
- **Benchmark Integration**: Fully compatible with LLM Lab's benchmark framework

## Supported Models

| Model | Size | Context | Description |
|-------|------|---------|-------------|
| Llama 2 7B | 3.83 GB | 4096 | Meta's versatile chat model |
| Llama 2 13B | 7.37 GB | 4096 | Larger Llama 2 variant |
| Mistral 7B | 4.07 GB | 32768 | Efficient model with long context |
| Phi-2 | 1.45 GB | 2048 | Microsoft's compact model |

## Installation

### 1. Install Dependencies

```bash
# Required: llama-cpp-python
pip install llama-cpp-python

# Optional: For GPU detection and memory monitoring
pip install torch psutil
```

### 2. Download Models

Use the built-in download helper:

```bash
# Download a specific model
python -m src.use_cases.local_models.download_helper --model phi-2

# List available models
python -m src.use_cases.local_models.download_helper --list

# Download all models
python -m src.use_cases.local_models.download_helper --all
```

## Usage

### Basic Example

```python
from src.use_cases.local_models.provider import LocalModelProvider
from src.use_cases.local_models.download_helper import ModelDownloader

# Check if model is downloaded
downloader = ModelDownloader()
model_path = downloader.get_model_path("phi-2")

# Initialize provider
provider = LocalModelProvider(
    model_name="phi-2",
    model_path=str(model_path),
    n_ctx=2048,  # Context window
    n_gpu_layers=-1,  # Auto-detect GPU
    temperature=0.7,
    max_tokens=100
)

# Generate text
response = provider.generate("Explain quantum computing in simple terms.")
print(response)
```

### Using with Benchmarks

```python
# Register the provider
from src.use_cases.local_models.register import register_local_provider
register_local_provider()

# Run benchmarks
from benchmarks.runners.multi_runner import MultiModelRunner

runner = MultiModelRunner(
    models=["phi-2", "gpt-4o-mini", "claude-3-haiku"],
    dataset="truthfulness"
)

results = runner.run()
```

### Streaming Responses

```python
# Enable streaming
response = provider.generate(
    "Write a story about AI",
    stream=True,
    max_tokens=200
)
```

### Custom Models

```python
# Use any GGUF model
provider = LocalModelProvider(
    model_name="custom",
    model_path="/path/to/your/model.gguf",
    n_ctx=4096,
    n_gpu_layers=35
)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_ctx` | 2048 | Context window size |
| `n_batch` | 512 | Batch size for prompt processing |
| `n_threads` | Auto | CPU threads to use |
| `n_gpu_layers` | -1 | GPU layers (-1 for auto, 0 for CPU) |
| `use_mmap` | True | Use memory-mapped files |
| `use_mlock` | False | Lock model in RAM |
| `temperature` | 0.7 | Generation temperature |
| `max_tokens` | 1000 | Maximum tokens to generate |

## Performance Optimization

### GPU Acceleration

The provider automatically detects and uses:
- **CUDA**: For NVIDIA GPUs
- **Metal**: For Apple Silicon Macs
- **CPU**: Fallback for systems without GPU

### Memory Management

```python
# Check memory usage
memory = provider.get_memory_usage()
print(f"RAM: {memory['ram_used_mb']:.1f} MB")
print(f"VRAM: {memory['vram_used_mb']:.1f} MB")

# Unload model when done
provider.unload_model()
```

### Quantization Levels

Models are pre-quantized in GGUF format:
- **Q4_K_M**: 4-bit quantization (recommended)
- **Q8_0**: 8-bit quantization (higher quality)
- **F16**: 16-bit (full precision, larger)

## Troubleshooting

### Common Issues

1. **Import Error: llama-cpp-python not found**
   ```bash
   pip install llama-cpp-python
   ```

2. **Model file not found**
   ```bash
   python -m src.use_cases.local_models.download_helper --model <model-name>
   ```

3. **Out of memory**
   - Reduce `n_gpu_layers` to offload less to GPU
   - Use smaller models (e.g., Phi-2)
   - Enable `use_mmap=True`

4. **Slow generation**
   - Increase `n_threads` for CPU inference
   - Enable GPU with `n_gpu_layers > 0`
   - Use smaller context window (`n_ctx`)

### Debug Mode

Enable verbose logging:

```python
provider = LocalModelProvider(
    model_name="phi-2",
    model_path=model_path,
    verbose=True
)
```

## Integration Tests

Run the test suite:

```bash
# Run all local model tests
pytest tests/test_local_model_provider.py -v

# Run specific test
pytest tests/test_local_model_provider.py::TestLocalModelProvider::test_generate_text -v
```

## Examples

See `examples/use_cases/local_model_demo.py` for:
- Model downloading
- Inference examples
- Streaming responses
- Performance comparison
- Benchmark integration

## Cost Comparison

| Model | Cost per 1K tokens | Latency | Privacy |
|-------|-------------------|---------|---------|
| Local Models | $0.00 | Low | Full |
| GPT-4o-mini | $0.15 | Medium | Limited |
| Claude 3 Haiku | $0.25 | Medium | Limited |
| Gemini 1.5 Flash | $0.075 | Low | Limited |

## Next Steps

1. **Download Models**: Start with Phi-2 for testing
2. **Run Benchmarks**: Compare local vs API models
3. **Optimize Performance**: Tune GPU layers and batch size
4. **Production Use**: Deploy for privacy-sensitive applications

## Contributing

To add support for new models:

1. Add model info to `model_configs.py`
2. Test with the provider
3. Update documentation
4. Submit a pull request

## License

This module follows the LLM Lab project license. Individual models have their own licenses - please check before commercial use.
