#!/usr/bin/env python3
"""
Download very small LLMs suitable for running on MacBook Pro
These models are optimized for CPU/Metal performance
"""

import os
import json
import argparse
import subprocess
import sys
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
    },
    "gpt-oss-20b": {
        "name": "openai/gpt-oss-20b",
        "size": "20B parameters (~40GB)",
        "description": "OpenAI's open-source 20B model (VERY LARGE - requires significant RAM)",
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

# Ollama models (recommended for best performance)
OLLAMA_MODELS = {
    "llama3.2-1b": {
        "ollama_name": "llama3.2:1b",
        "size": "~700MB",
        "description": "High-quality 1B model, excellent for chat and code"
    },
    "gpt-oss-20b": {
        "ollama_name": "gpt-oss:20b", 
        "size": "~13GB",
        "description": "Large 20B model, GPT-3.5 level performance"
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

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_ollama():
    """Install Ollama via Homebrew"""
    print("📦 Installing Ollama...")
    try:
        # Check if Homebrew is available
        subprocess.run(["brew", "--version"], capture_output=True, check=True)
        
        # Install Ollama
        result = subprocess.run(["brew", "install", "ollama"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama installed successfully!")
            
            # Start the service
            print("🚀 Starting Ollama service...")
            subprocess.run(["brew", "services", "start", "ollama"], capture_output=True)
            print("✅ Ollama service started!")
            return True
        else:
            print(f"❌ Failed to install Ollama: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Homebrew not found. Please install Homebrew first:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Ollama: {e}")
        return False

def setup_ollama_models():
    """Download and setup recommended Ollama models"""
    if not check_ollama_installed():
        print("⚠️  Ollama not found. Installing...")
        if not install_ollama():
            return False
    
    print("\n🎯 Setting up recommended Ollama models...")
    
    # Always install llama3.2:1b (small, high quality)
    print("\n📥 Downloading Llama3.2-1B (recommended)...")
    result = subprocess.run(["ollama", "pull", "llama3.2:1b"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Llama3.2-1B downloaded successfully!")
    else:
        print(f"❌ Failed to download Llama3.2-1B: {result.stderr}")
    
    # Ask about GPT-OSS-20B (large model)
    print("\n⚠️  GPT-OSS-20B is a large model (~13GB).")
    response = input("   Download GPT-OSS-20B? (y/N): ")
    if response.lower() == 'y':
        print("\n📥 Downloading GPT-OSS-20B (this will take a while)...")
        result = subprocess.run(["ollama", "pull", "gpt-oss:20b"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPT-OSS-20B downloaded successfully!")
        else:
            print(f"❌ Failed to download GPT-OSS-20B: {result.stderr}")
    
    print("\n✅ Ollama setup complete!")
    print("🧪 Test with: python models/small-llms/run_small_model_demo.py --model llama3.2-1b \"Hello!\"")
    return True

def save_model_info():
    """Save information about downloaded models"""
    info = {
        "transformers_models": SMALL_MODELS,
        "gguf_models": GGUF_MODELS,
        "ollama_models": OLLAMA_MODELS,
        "usage_notes": {
            "transformers": "Use with transformers library or our inference script",
            "gguf": "Use with llama.cpp or llama-cpp-python for best performance",
            "ollama": "Use with Ollama for optimized performance (recommended)",
            "metal": "These models can use Apple Metal for acceleration"
        }
    }
    
    with open("models/small-llms/model_info.json", "w") as f:
        json.dump(info, f, indent=2)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download small LLMs suitable for MacBook Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default set (all 6 small models)
  python download_small_models.py
  
  # Download specific models
  python download_small_models.py --models pythia-70m,smollm-135m
  
  # List available models without downloading
  python download_small_models.py --list
  
  # Download only GGUF models
  python download_small_models.py --gguf-only
  
  # Download all available models
  python download_small_models.py --all
  
  # Download only small models (skip GPT-OSS-20B)
  python download_small_models.py --small-only
  
  # Setup Ollama with recommended models (recommended)
  python download_small_models.py --setup-ollama
  
  # Download a custom model from HuggingFace
  python download_small_models.py --custom microsoft/phi-1_5 --custom-name phi-1.5
  
  # Download a custom GGUF model
  python download_small_models.py --custom TheBloke/Mistral-7B-v0.1-GGUF --custom-gguf --gguf-file mistral-7b-v0.1.Q4_K_M.gguf
"""
    )
    
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to download (e.g., pythia-70m,smollm-135m)"
    )
    parser.add_argument(
        "--gguf-only",
        action="store_true",
        help="Download only GGUF quantized models"
    )
    parser.add_argument(
        "--no-gguf",
        action="store_true",
        help="Skip GGUF models, download only transformers models"
    )
    parser.add_argument(
        "--small-only",
        action="store_true",
        help="Download only small models (<1GB), skip large models like GPT-OSS-20B"
    )
    parser.add_argument(
        "--setup-ollama",
        action="store_true",
        help="Setup Ollama with recommended models (installs Ollama if needed)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models without downloading"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models (including larger ones)"
    )
    parser.add_argument(
        "--custom",
        type=str,
        help="Download a custom model from HuggingFace (e.g., meta-llama/Llama-2-7b-hf)"
    )
    parser.add_argument(
        "--custom-name",
        type=str,
        help="Local name for the custom model (defaults to last part of model ID)"
    )
    parser.add_argument(
        "--custom-gguf",
        action="store_true",
        help="Treat custom model as GGUF format (download specific .gguf files)"
    )
    parser.add_argument(
        "--gguf-file",
        type=str,
        help="Specific GGUF file to download (e.g., model.Q4_K_M.gguf)"
    )
    
    return parser.parse_args()

def main():
    """Download selected small models"""
    args = parse_arguments()
    
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
    
    print("\n📋 Available Ollama Models (recommended):")
    for key, info in OLLAMA_MODELS.items():
        print(f"  {key}: {info['ollama_name']} ({info['size']})")
        print(f"    → {info['description']}")
    
    # If just listing, exit here
    if args.list:
        return
    
    # Handle Ollama setup
    if args.setup_ollama:
        setup_ollama_models()
        return
    
    # Handle custom model download
    if args.custom:
        print(f"\n🎯 Downloading custom model: {args.custom}")
        
        # Determine local name
        if args.custom_name:
            custom_name = args.custom_name
        else:
            # Use the last part of the model ID as the name
            custom_name = args.custom.split('/')[-1].lower()
        
        if args.custom_gguf:
            # Download as GGUF model
            if not args.gguf_file:
                print("❌ Error: --gguf-file is required when using --custom-gguf")
                return
            
            custom_model_info = {
                "name": args.custom,
                "files": [args.gguf_file],
                "size": "Custom GGUF model",
                "description": f"Custom GGUF model from {args.custom}"
            }
            download_gguf_model(custom_model_info, custom_name)
        else:
            # Download as transformers model
            download_transformers_model(args.custom, custom_name)
        
        print(f"\n✨ Custom model downloaded to models/small-llms/{custom_name}")
        return
    
    # Determine which models to download
    if args.models:
        # Download specific models requested
        requested = [m.strip() for m in args.models.split(",")]
        selected_models = [m for m in requested if m in SMALL_MODELS]
        selected_gguf = [m for m in requested if m in GGUF_MODELS]
        
        # Warn about invalid model names
        invalid = [m for m in requested if m not in SMALL_MODELS and m not in GGUF_MODELS]
        if invalid:
            print(f"\n⚠️  Warning: Unknown models: {invalid}")
    
    elif args.all:
        # Download all available models
        selected_models = list(SMALL_MODELS.keys())
        selected_gguf = list(GGUF_MODELS.keys()) if not args.no_gguf else []
    
    elif args.gguf_only:
        # Download only GGUF models
        selected_models = []
        selected_gguf = ["qwen-0.5b-gguf"]
    
    else:
        # Default: Download the standard set of models
        if args.small_only:
            # Only small models (<1GB)
            selected_models = ["pythia-70m", "pythia-160m", "smollm-135m", "smollm-360m", "qwen-0.5b"]
        else:
            # Include GPT-OSS-20B
            selected_models = ["pythia-70m", "pythia-160m", "smollm-135m", "smollm-360m", "qwen-0.5b", "gpt-oss-20b"]
        selected_gguf = ["qwen-0.5b-gguf"] if not args.no_gguf else []
    
    # Skip GGUF if requested
    if args.no_gguf:
        selected_gguf = []
    
    print(f"\n🎯 Downloading selected models: {selected_models + selected_gguf}")
    
    # Check for large models and warn user
    large_models = [m for m in selected_models if m in ["gpt-oss-20b"]]
    if large_models:
        print(f"\n⚠️  WARNING: The following models are VERY LARGE:")
        for model in large_models:
            print(f"   - {model}: {SMALL_MODELS[model]['size']}")
        print("   These models require significant disk space and RAM (32GB+ recommended)")
        response = input("\n   Continue with download? (y/N): ")
        if response.lower() != 'y':
            print("   Skipping large models...")
            selected_models = [m for m in selected_models if m not in large_models]
    
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