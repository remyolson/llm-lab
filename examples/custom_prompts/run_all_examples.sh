#!/bin/bash
# Master script to run all custom prompt examples
# Demonstrates the complete CLI interface and template system

echo "🚀 Running All Custom Prompt Examples"
echo "======================================"
echo "This will run examples for:"
echo "  - Customer Service Response Testing"
echo "  - Code Generation Across Complexity Levels"
echo "  - Creative Writing in Multiple Genres"
echo
echo "Results will be saved to ./results/examples/"
echo

# Check if required directories exist
if [ ! -d "scripts" ]; then
  echo "❌ Error: scripts/ directory not found. Please run from project root."
  exit 1
fi

if [ ! -d "templates" ]; then
  echo "❌ Error: templates/ directory not found. Please run from project root."
  exit 1
fi

# Create results directory
mkdir -p results/examples

# Set default models if not specified
if [ -z "$EXAMPLE_MODELS" ]; then
  EXAMPLE_MODELS="gpt-4o-mini,claude-3-haiku"
  echo "Using default models: $EXAMPLE_MODELS"
  echo "Set EXAMPLE_MODELS environment variable to use different models"
  echo
fi

# Option to run quick tests only
if [ "$1" = "--quick" ]; then
  echo "🏃 Running quick tests only (--limit 1 for all examples)"
  export QUICK_MODE="--limit 1"
else
  echo "Running full examples (may take several minutes and use API credits)"
  export QUICK_MODE=""
fi

echo
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Cancelled."
  exit 0
fi

echo
echo "Starting example runs..."
echo

# Run Customer Service Examples
echo "📞 Running Customer Service Examples..."
./examples/custom_prompts/customer_service_examples.sh

echo
echo "⚡ Waiting 5 seconds before next batch..."
sleep 5

# Run Code Generation Examples
echo "💻 Running Code Generation Examples..."
./examples/custom_prompts/code_generation_examples.sh

echo
echo "⚡ Waiting 5 seconds before next batch..."
sleep 5

# Run Creative Writing Examples
echo "✍️ Running Creative Writing Examples..."
./examples/custom_prompts/creative_writing_examples.sh

echo
echo "🎉 All Examples Completed!"
echo "========================="
echo
echo "📊 Results Summary:"
find results/examples -name "*.json" -o -name "*.md" -o -name "*.csv" | wc -l | xargs echo "Total result files:"

echo
echo "📁 Result Directories:"
find results/examples -type d -mindepth 1 -maxdepth 2 | sort

echo
echo "🔍 View Results:"
echo "  JSON files: find results/examples -name '*.json' | head -5"
echo "  Markdown: find results/examples -name '*.md' | head -5"
echo "  CSV files: find results/examples -name '*.csv' | head -5"

echo
echo "📈 Quick Analysis:"
echo "  Customer Service: ls results/examples/customer-service/"
echo "  Code Generation: ls results/examples/code-generation/"
echo "  Creative Writing: ls results/examples/creative-writing/"

echo
echo "💡 Next Steps:"
echo "  1. Review the generated content in results/examples/"
echo "  2. Analyze model performance across different use cases"
echo "  3. Use these examples as templates for your own prompts"
echo "  4. Modify template files in templates/ for your specific needs"
echo
echo "For more information, see: docs/guides/CUSTOM_PROMPT_CLI.md"
