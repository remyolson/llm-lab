#!/bin/bash
# Quick script to run models with the correct virtual environment

# Base directory
BASE_DIR="/Users/ro/Documents/GitHub/lllm-lab"
VENV_PYTHON="$BASE_DIR/venv/bin/python"
SCRIPT_DIR="$BASE_DIR/models/small-llms"

cd "$SCRIPT_DIR"

echo "🔧 Using virtual environment Python: $VENV_PYTHON"
echo "📁 Running from directory: $SCRIPT_DIR"
echo ""

# Run the script with all provided arguments
"$VENV_PYTHON" run_small_model_demo.py "$@"