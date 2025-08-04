# Prerequisites for LLM Lab

This guide covers all the prerequisites needed to use LLM Lab effectively. Complete these steps before following any use case guides.

## 🔧 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM (8GB recommended for parallel execution)
- **Storage**: 2GB free space (more for local models)
- **OS**: Windows, macOS, or Linux

### Recommended Specifications
- **Python**: 3.10 or 3.11
- **Memory**: 16GB RAM for optimal performance
- **Storage**: 10GB+ for local model experiments
- **Internet**: Stable connection for API calls

## 🔑 API Keys Setup

### Required API Keys
You need at least **ONE** of the following API keys to get started:

#### Option 1: Google Gemini (Recommended for beginners)
- **Get Key**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Free Tier**: Yes, generous limits
- **Best For**: Fast responses, cost-effective testing
- **Typical Cost**: $0.05-$0.50 per benchmark run

#### Option 2: OpenAI GPT
- **Get Key**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Free Tier**: $5 credit for new accounts
- **Best For**: Wide model selection, GPT-4 access
- **Typical Cost**: $0.10-$5.00 per benchmark run

#### Option 3: Anthropic Claude
- **Get Key**: [Anthropic Console](https://console.anthropic.com/)
- **Free Tier**: Limited
- **Best For**: Long context, advanced reasoning
- **Typical Cost**: $0.15-$3.00 per benchmark run

### 💰 Cost Management Tips

1. **Start with Free Tiers**: Google offers the most generous free tier
2. **Use Cheaper Models First**: Test with `gemini-1.5-flash` or `gpt-4o-mini`
3. **Set Budget Alerts**: Configure spending limits in each provider's console
4. **Monitor Usage**: Check the cost analysis examples to track spending

### Setting Up API Keys

Create a `.env` file in the project root:

```bash
# .env file
GOOGLE_API_KEY=your-google-key-here
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Optional: For future use cases
HUGGINGFACE_TOKEN=your-hf-token-here
```

**Security Note**: Never commit your `.env` file to version control!

## 📦 Environment Setup

### 1. Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/yourusername/lllm-lab.git

# Using SSH
git clone git@github.com:yourusername/lllm-lab.git

cd lllm-lab
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# On Windows PowerShell:
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# For development (includes testing tools)
pip install -r requirements-dev.txt
```

### 4. Verify Installation

Run these commands to ensure everything is set up correctly:

```bash
# Check Python version
python --version  # Should be 3.8+

# Verify key packages
python -c "import anthropic; print('✓ Anthropic SDK installed')"
python -c "import google.generativeai; print('✓ Google AI SDK installed')"
python -c "import openai; print('✓ OpenAI SDK installed')"

# Verify LLM Lab modules
python -c "from src.providers import GoogleProvider; print('✓ LLM Lab modules accessible')"

# Check which providers are available
python -c "
import os
print('Available providers based on API keys:')
if os.getenv('GOOGLE_API_KEY'): print('  ✓ Google Gemini')
if os.getenv('OPENAI_API_KEY'): print('  ✓ OpenAI GPT')
if os.getenv('ANTHROPIC_API_KEY'): print('  ✓ Anthropic Claude')
"
```

## 🚀 Quick Start Verification

Test your setup with a simple benchmark:

```bash
# Test with the smallest, fastest model you have access to
python scripts/run_benchmarks.py --model gemini-1.5-flash --dataset truthfulness --no-csv
```

Expected output:
```
🔬 LLM Lab Benchmark Runner
==================================================
✓ Model configuration loaded
✓ Provider initialized
✓ Dataset loaded
Running evaluations...
```

## 🔍 Troubleshooting Setup

### Common Issues

#### 1. "No module named 'src'" Error
```bash
# Make sure you're in the project root
pwd  # Should show /path/to/lllm-lab

# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. API Key Not Found
```bash
# Check if .env file exists
ls -la .env

# Verify key is set
echo $GOOGLE_API_KEY  # Should show your key
```

#### 3. SSL/Certificate Errors
```bash
# Update certificates
pip install --upgrade certifi

# Or use system certificates
export SSL_CERT_FILE=$(python -m certifi)
export REQUESTS_CA_BUNDLE=$(python -m certifi)
```

#### 4. Permission Denied on Windows
- Run PowerShell as Administrator
- Or use: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## 📚 Additional Setup (Optional)

### For Local Model Development (Use Cases 5 & 6)
```bash
# Install additional dependencies
pip install transformers accelerate bitsandbytes

# Download models (automated by scripts)
python scripts/download_assets.py
```

### For Advanced Analysis
```bash
# Install visualization tools
pip install matplotlib seaborn pandas jupyter
```

### For Contributing/Development
```bash
# Install development tools
pip install pytest black flake8 mypy

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## ✅ Setup Complete!

Once you've completed these prerequisites, you're ready to:
1. Follow any [Use Case How-To Guide](./README.md)
2. Run benchmarks across multiple models
3. Analyze costs and performance
4. Contribute to the project

## 🆘 Getting Help

If you encounter issues:
1. Check the [Troubleshooting Guide](../TROUBLESHOOTING.md)
2. Review [closed issues](https://github.com/yourusername/lllm-lab/issues?q=is%3Aissue+is%3Aclosed)
3. Open a [new issue](https://github.com/yourusername/lllm-lab/issues/new) with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

---

*Last updated: January 2025*