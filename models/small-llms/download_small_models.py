#!/usr/bin/env python3
"""
Download very small LLMs suitable for running on MacBook Pro
These models are optimized for CPU/Metal performance
"""

import os
import json
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Very small models perfect for MacBook Pro
SMALL_MODELS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "size": "1.1B parameters",
        "description": "Tiny but capable chat model",
        "format": "transformers"
    },
    "phi-2": {
        "name": "microsoft/phi-2",
        "size": "2.7B parameters", 
        "description": "Microsoft's small but powerful model",
        "format": "transformers"
    },
    "qwen-0.5b": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "size": "0.5B parameters",
        "description": "Qwen's ultra-small instruction model",
        "format": "transformers"
    },
    "smollm-135m": {
        "name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "size": "135M parameters",
        "description": "Truly tiny model for basic tasks",
        "format": "transformers"
    },
    "smollm-360m": {
        "name": "HuggingFaceTB/SmolLM2-360M-Instruct", 
        "size": "360M parameters",
        "description": "Small model with better performance",
        "format": "transformers"
    },
    "pythia-70m": {
        "name": "EleutherAI/pythia-70m",
        "size": "70M parameters",
        "description": "Extremely small GPT-NeoX model",
        "format": "transformers"
    },
    "pythia-160m": {
        "name": "EleutherAI/pythia-160m",
        "size": "160M parameters",
        "description": "Small GPT-NeoX model",
        "format": "transformers"
    }
}

# GGUF models for llama.cpp (even more efficient)
GGUF_MODELS = {
    "tinyllama-gguf": {
        "name": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "files": ["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"],
        "size": "~600MB (Q4 quantized)",
        "description": "4-bit quantized TinyLlama"
    },
    "qwen-0.5b-gguf": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "files": ["qwen2.5-0.5b-instruct-q4_k_m.gguf"],
        "size": "~300MB (Q4 quantized)",
        "description": "4-bit quantized Qwen 0.5B"
    }
}

def download_transformers_model(model_id, model_name):
    """Download a transformers format model"""
    print(f"\n📥 Downloading {model_name} ({model_id})...")
    
    local_dir = f"models/small-llms/{model_name}"
    
    try:
        # Download model
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin", "*.safetensors.index.json"]  # Skip unnecessary files
        )
        
        print(f"✅ Downloaded {model_name} to {local_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False

def download_gguf_model(model_info, model_name):
    """Download a GGUF format model"""
    print(f"\n📥 Downloading {model_name} (GGUF format)...")
    
    local_dir = f"models/small-llms/{model_name}"
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        for filename in model_info["files"]:
            print(f"  Downloading {filename}...")
            hf_hub_download(
                repo_id=model_info["name"],
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        
        print(f"✅ Downloaded {model_name} to {local_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False

def save_model_info():
    """Save information about downloaded models"""
    info = {
        "transformers_models": SMALL_MODELS,
        "gguf_models": GGUF_MODELS,
        "usage_notes": {
            "transformers": "Use with transformers library or our inference script",
            "gguf": "Use with llama.cpp or llama-cpp-python for best performance",
            "metal": "These models can use Apple Metal for acceleration"
        }
    }
    
    with open("models/small-llms/model_info.json", "w") as f:
        json.dump(info, f, indent=2)

def main():
    """Download selected small models"""
    print("🤖 Small LLM Downloader for MacBook Pro")
    print("=" * 50)
    
    # Show available models
    print("\n📋 Available Transformers Models:")
    for key, info in SMALL_MODELS.items():
        print(f"  {key}: {info['name']} ({info['size']})")
        print(f"    → {info['description']}")
    
    print("\n📋 Available GGUF Models (for llama.cpp):")
    for key, info in GGUF_MODELS.items():
        print(f"  {key}: {info['name']} ({info['size']})")
        print(f"    → {info['description']}")
    
    # Download the smallest models by default
    selected_models = ["smollm-135m", "smollm-360m", "qwen-0.5b"]
    selected_gguf = ["qwen-0.5b-gguf"]
    
    print(f"\n🎯 Downloading selected models: {selected_models + selected_gguf}")
    
    # Download transformers models
    for model_key in selected_models:
        if model_key in SMALL_MODELS:
            model_info = SMALL_MODELS[model_key]
            download_transformers_model(model_info["name"], model_key)
    
    # Download GGUF models
    for model_key in selected_gguf:
        if model_key in GGUF_MODELS:
            model_info = GGUF_MODELS[model_key]
            download_gguf_model(model_info, model_key)
    
    # Save model information
    save_model_info()
    
    print("\n✨ Download complete! Check models/small-llms/ for your models.")
    print("📖 See inference.py for usage examples.")

if __name__ == "__main__":
    main()