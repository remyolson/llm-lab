# Models Directory

This directory contains local model files and utilities for working with small language models.

## Structure

### small-llms/
Contains small, locally-runnable language models for testing and development:
- **qwen-0.5b/** - Qwen 0.5B model files
- **qwen-0.5b-gguf/** - Quantized GGUF format for efficient inference
- **smollm-135m/** - SmolLM 135M parameter model
- **smollm-360m/** - SmolLM 360M parameter model

### Utilities
- **download_small_models.py** - Script to download model files
- **inference.py** - Local inference utilities
- **quick_demo.py** - Quick demonstration of local models

## Usage

To download models:
```bash
cd models/small-llms
python download_small_models.py
```

To run a quick demo:
```bash
python quick_demo.py
```

## Note

These models are stored with Git LFS (Large File Storage) due to their size. Make sure you have Git LFS installed and configured before cloning or pulling updates.