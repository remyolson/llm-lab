#!/bin/bash
# LLM Lab Git Cleanup Script
# 
# This script removes large model files and datasets from Git tracking
# after implementing the download-based asset management system.
#
# ⚠️  Run this ONCE after setting up the new .gitignore and download system

set -e

echo "🧹 LLM Lab Git Cleanup Script"
echo "=============================="
echo ""
echo "This script will:"
echo "  1. Remove large model files from Git tracking"
echo "  2. Remove large dataset files from Git tracking"  
echo "  3. Clean up Git LFS artifacts"
echo "  4. Keep files locally for continued use"
echo ""
echo "⚠️  Warning: This will modify your Git history."
echo "Make sure you've committed any other changes first!"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "🔍 Checking Git status..."
if ! git diff-index --quiet HEAD --; then
    echo "❌ You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

echo "✅ Git status clean, proceeding..."
echo ""

# Remove large model files from Git tracking but keep them locally
echo "📦 Removing model files from Git tracking..."

# Remove .gitattributes files that track LFS
echo "  • Removing LFS .gitattributes files..."
find models/small-llms -name ".gitattributes" -delete 2>/dev/null || true

# Remove large files from Git index (but keep local copies)
echo "  • Removing safetensors files from Git..."
git rm --cached models/small-llms/*/model.safetensors 2>/dev/null || true

echo "  • Removing GGUF files from Git..."
git rm --cached models/small-llms/*/*.gguf 2>/dev/null || true

echo "  • Removing ONNX directories from Git..."
git rm -r --cached models/small-llms/*/onnx/ 2>/dev/null || true

echo "  • Removing training run directories from Git..."
git rm -r --cached models/small-llms/*/runs/ 2>/dev/null || true

# Remove dataset directories that might contain large files
echo "📊 Removing large dataset files from Git tracking..."
git rm -r --cached datasets/benchmarking/raw/*/data/ 2>/dev/null || true
git rm -r --cached datasets/fine-tuning/raw/*/data/ 2>/dev/null || true

# Clean up any LFS pointer files that might be problematic
echo "🔧 Cleaning up Git LFS..."
git lfs untrack "models/small-llms/*/*.gguf" 2>/dev/null || true
git lfs untrack "models/small-llms/*/model.safetensors" 2>/dev/null || true
git lfs untrack "models/small-llms/*/onnx/*" 2>/dev/null || true

echo ""
echo "📋 Current repository status:"
echo "----------------------------"

# Show what's been changed
echo "Files removed from Git tracking:"
git status --porcelain | grep "^D " | head -10
if [ $(git status --porcelain | grep "^D " | wc -l) -gt 10 ]; then
    echo "... and $(expr $(git status --porcelain | grep "^D " | wc -l) - 10) more files"
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git status"
echo "  2. Commit the cleanup: git add . && git commit -m 'Switch to download-based asset management'"
echo "  3. Test the download script: python download_assets.py --verify"
echo "  4. Force push to update remote: git push --force-with-lease"
echo ""
echo "💡 Your local model files are still available in the models/ directory."
echo "   Contributors will download them using: python download_assets.py --all"
echo ""