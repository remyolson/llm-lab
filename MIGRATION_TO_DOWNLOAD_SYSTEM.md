# Migration to Download-Based Asset Management

This document explains how to migrate from Git LFS to a download-based system for managing large model files and datasets.

## 🎯 Why This Change?

**Problem**: GitHub rejected the push because `qwen2.5-0.5b-instruct-q4_k_m.gguf` (468.64 MB) exceeds the 100MB limit, and Git LFS has bandwidth/storage costs that can become expensive for open source projects.

**Solution**: Move to a download-based system where:
- ✅ Large files are downloaded from original sources (Hugging Face, etc.)
- ✅ No bandwidth costs for contributors or maintainers
- ✅ Easy setup for anyone who forks the repository
- ✅ Better performance for large files
- ✅ Automatic verification of file integrity

## 🚀 Quick Migration Steps

### 1. Update .gitignore (✅ Done)
```bash
# Added to .gitignore:
models/small-llms/*/model.safetensors
models/small-llms/*/onnx/
models/small-llms/*/*.gguf
models/small-llms/*/runs/
datasets/benchmarking/raw/*/data/
datasets/fine-tuning/raw/*/data/
downloads/
*.tmp
```

### 2. Clean up Git tracking
```bash
# Run the automated cleanup script:
./cleanup_git_assets.sh

# Then commit the changes:
git add .
git commit -m "feat: Switch to download-based asset management

- Add download_assets.py script for automated asset management
- Update .gitignore to exclude large model files and datasets  
- Remove large files from Git LFS tracking
- Update README with new setup instructions

Resolves GitHub LFS file size limits and bandwidth costs.
Assets are now downloaded from original sources (Hugging Face, etc.)."
```

### 3. Force push to fix the remote
```bash
# This overwrites the problematic commit on the remote
git push --force-with-lease
```

### 4. Verify everything works
```bash
# Test the download system
python download_assets.py --list
python download_assets.py --verify

# Test a selective download
python download_assets.py --model qwen-0.5b-gguf
```

## 📋 New Contributor Workflow

For anyone cloning the repository:

```bash
# 1. Clone the repo (fast, no large files)
git clone https://github.com/yourusername/lllm-lab.git
cd lllm-lab

# 2. Set up Python environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Download assets (one-time setup)
python download_assets.py --all  # Download everything (~2.4GB)
# or selectively:
python download_assets.py --models     # Only models
python download_assets.py --datasets   # Only datasets

# 4. Ready to go!
python run_benchmarks.py --provider google --dataset truthfulness
```

## 🔍 Asset Management Commands

```bash
# List available assets
python download_assets.py --list

# Download everything
python download_assets.py --all

# Download by category
python download_assets.py --models
python download_assets.py --datasets

# Download specific assets
python download_assets.py --model qwen-0.5b
python download_assets.py --dataset truthfulqa-full

# Verify downloaded files
python download_assets.py --verify
```

## 🏗️ System Architecture

### Before (Git LFS):
```
Repository Size: ~3GB+ with LFS pointers
Clone Time: Long (downloads all LFS files)
Bandwidth: Costs money for maintainers
Fork-friendly: ❌ (LFS setup required)
```

### After (Download System):
```
Repository Size: <50MB (no large files)
Clone Time: Fast (no large files)
Bandwidth: Free (uses Hugging Face, etc.)
Fork-friendly: ✅ (just run download script)
```

## 🔧 Customization

To add new models or datasets, edit `download_assets.py`:

```python
# Add to the assets dictionary:
"new-model": {
    "description": "Description of the model",
    "files": {
        "model.safetensors": {
            "url": "https://huggingface.co/.../model.safetensors",
            "size_mb": 500,
            "sha256": "optional_checksum"
        }
    },
    "destination": "models/small-llms/new-model/"
}
```

## 🎉 Benefits

1. **No more LFS costs**: Free hosting for large files
2. **Faster clones**: Repository is now <50MB instead of 3GB+
3. **Fork-friendly**: No special Git LFS setup required
4. **Better UX**: Clear progress indicators and error handling
5. **Verification**: Automatic size and checksum validation
6. **Selective downloads**: Download only what you need
7. **Open source friendly**: No costs or limits for contributors

## 🚨 Important Notes

- **Local files preserved**: The cleanup script keeps your local model files
- **One-time migration**: You only need to run the cleanup script once
- **Force push required**: The problematic commit needs to be overwritten
- **Update CI/CD**: Any automated workflows should use the download script

## 📞 Support

If you encounter issues during migration:

1. Check that all local changes are committed before running cleanup
2. Verify the download script works: `python download_assets.py --list`
3. Test selective downloads before downloading everything
4. Use `--verify` to check file integrity after download

This migration solves the immediate Git LFS issue and sets up a much better system for managing large files in an open source ML project! 🎯