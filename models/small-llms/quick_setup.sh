#!/bin/bash
# Quick setup script for Small LLMs on MacBook Pro
# This will install Ollama and download recommended models

set -e  # Exit on any error

echo "🚀 Quick Setup for Small LLMs on MacBook Pro"
echo "============================================"
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS. Please install Ollama manually:"
    echo "   Visit: https://ollama.com"
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is required but not installed."
    echo "   Install Homebrew first: https://brew.sh"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "📦 Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    brew install ollama
    echo "✅ Ollama installed!"
else
    echo "✅ Ollama already installed!"
fi

echo ""
echo "🚀 Starting Ollama service..."
brew services start ollama
sleep 2  # Give service time to start

echo ""
echo "📥 Downloading recommended models..."
echo "   This may take a few minutes..."

# Download Llama3.2-1B (always recommended)
echo ""
echo "📥 Downloading Llama3.2-1B (~700MB)..."
ollama pull llama3.2:1b
echo "✅ Llama3.2-1B ready!"

# Ask about GPT-OSS-20B
echo ""
echo "⚠️  GPT-OSS-20B is a large model (~13GB)."
echo "   It offers excellent quality but requires more storage and memory."
read -p "   Download GPT-OSS-20B? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Downloading GPT-OSS-20B (~13GB) - this will take a while..."
    ollama pull gpt-oss:20b
    echo "✅ GPT-OSS-20B ready!"
else
    echo "⏭️  Skipping GPT-OSS-20B (you can download later with: ollama pull gpt-oss:20b)"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🧪 Test your setup:"
echo "   python models/small-llms/run_small_model_demo.py --model llama3.2-1b \"Hello!\""
echo ""
echo "📋 List all models:"
echo "   python models/small-llms/run_small_model_demo.py --list"
echo ""
echo "🔧 For transformers models, also run:"
echo "   python models/small-llms/download_small_models.py --small-only"
echo ""
echo "📖 See models/small-llms/README.md for more information!"