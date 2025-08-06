# Use Case 1: Run Standard LLM Benchmarks on Multiple Models

*Compare performance, accuracy, and speed across Google Gemini, OpenAI GPT, and Anthropic Claude models using standardized datasets.*

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:
- Compare performance of multiple LLM models across **5 different benchmark datasets**
- Test models on diverse capabilities: truthfulness, reasoning, math, knowledge, and commonsense
- **Automatically save detailed results** to CSV files for analysis
- Generate comprehensive reports with performance metrics
- Analyze which models perform best for different types of tasks
- Understand the cost implications of each model
- Make data-driven decisions about model selection for specific use cases

## 📋 Before You Begin

- Complete all [Prerequisites](./PREREQUISITES.md)
- Ensure you have at least one API key configured (Google, OpenAI, or Anthropic)
- Time required: ~10-30 minutes (depending on number of models)
- Estimated cost: $0.05-$5.00 per full benchmark run

### 💰 Cost Breakdown
Running the full TruthfulQA dataset (500 questions) with different models:

**💡 Pro Tip:** Use `--limit 10` for testing to reduce costs by 98% (approximately $0.001-$0.10 per test run)

- **Google Gemini**:
  - `gemini-1.5-flash`: ~$0.05 (very cost-effective)
  - `gemini-1.5-pro`: ~$0.50
  - **Free tier eligible**: Yes (generous daily limits)

- **OpenAI GPT**:
  - `gpt-4o-mini`: ~$0.10
  - `gpt-3.5-turbo`: ~$0.15
  - `gpt-4o`: ~$2.00
  - `gpt-4`: ~$5.00
  - **Free tier eligible**: $5 credit for new accounts

- **Anthropic Claude**:
  - `claude-3-haiku-20240307`: ~$0.15
  - `claude-3-5-sonnet-20241022`: ~$1.00
  - `claude-3-opus-20240229`: ~$3.00
  - **Free tier eligible**: Limited

*Note: Costs are estimates based on January 2025 pricing. Actual costs depend on response lengths. Single model tests cost proportionally less.*

## 📊 Available Benchmark Datasets

Choose the right dataset based on what capabilities you want to test:

| Dataset | Size | What It Tests | Best For | Example Question |
|---------|------|---------------|----------|------------------|
| **truthfulness** | 500 | Factual accuracy, avoiding misinformation | Fact-checking, reliability assessment | "What happens if you eat watermelon seeds?" |
| **arc** | 1,000 | Scientific reasoning, logic | STEM applications, analytical thinking | "Why does ice float on water?" |
| **gsm8k** | 1,000 | Grade-school math problems | Mathematical reasoning, problem-solving | "Janet's ducks lay 16 eggs per day..." |
| **mmlu** | 1,000 | Academic knowledge across subjects | Broad knowledge assessment, education | "Find the degree for the given field extension..." |
| **hellaswag** | 1,000 | Commonsense reasoning | Real-world understanding, context | "A man sits on a roof. He..." |

### 🎯 **Dataset Selection Guide:**

- **🔍 For general model evaluation:** Start with `truthfulness` and `arc`
- **🧮 For math/computational tasks:** Use `gsm8k`
- **🎓 For academic/knowledge work:** Choose `mmlu`
- **🌍 For real-world applications:** Test with `hellaswag`
- **📊 For comprehensive assessment:** Run all datasets with `--limit 10` first

## 🚀 Step-by-Step Guide

### Step 1: Verify Your Configuration

First, check which providers are available based on your API keys:

```bash
# Check environment variables
python -c "
import os
print('Available providers:')
if os.getenv('GOOGLE_API_KEY'): print('  ✓ Google Gemini')
if os.getenv('OPENAI_API_KEY'): print('  ✓ OpenAI GPT')
if os.getenv('ANTHROPIC_API_KEY'): print('  ✓ Anthropic Claude')
"
```

### Step 2: Run Your First Benchmark

Start with a single model to ensure everything works:

```bash
# Quick test with truthfulness dataset (recommended first run)
python run_benchmarks.py --model gemini-1.5-flash --dataset truthfulness --limit 10

# Try different dataset types
python run_benchmarks.py --model gemini-1.5-flash --dataset arc --limit 10        # Science reasoning
python run_benchmarks.py --model gemini-1.5-flash --dataset gsm8k --limit 10      # Math problems
python run_benchmarks.py --model gemini-1.5-flash --dataset mmlu --limit 10       # Academic knowledge
python run_benchmarks.py --model gemini-1.5-flash --dataset hellaswag --limit 10  # Commonsense

# Full dataset test (remove --limit for complete assessment)
python run_benchmarks.py --model gemini-1.5-flash --dataset truthfulness

# Use improved evaluation (better accuracy, default since v2.0)
python run_benchmarks.py --model gemini-1.5-flash --dataset truthfulness --limit 10 --evaluation-method multi
```

**What Happens:**
- The benchmark runs through questions in your chosen dataset (10 with `--limit 10`, or the full dataset without limit)
- Results are automatically saved to `results/[dataset]/YYYY-MM/benchmark_*.csv`
- Progress is displayed in real-time in the console
- Different datasets test different capabilities (see dataset table above)

**Expected Output:**
```
🔬 LLM Lab Benchmark Runner
==================================================

📋 Validating 1 model(s)...
   ✓ gemini-1.5-flash -> GoogleProvider

1. Loading configuration...
   ✓ Model configuration loaded

2. Initializing provider...
   ✓ google provider initialized for model gemini-1.5-flash

3. Loading dataset...
   ✓ Loaded 10 prompts from truthfulness

4. Running evaluations...
   Prompt 1/10: What is the smallest country in the world?...
   Response: The smallest country in the world is Vatican City...
   ✓ Evaluation passed (matched: Vatican)
```

### Step 3: Compare Multiple Models

Now run comparisons across different providers and datasets:

```bash
# Quick comparison on truthfulness (recommended first comparison)
python run_benchmarks.py \
  --models gemini-1.5-flash,gpt-4o-mini,claude-3-haiku-20240307 \
  --dataset truthfulness \
  --limit 10

# Test models on math reasoning
python run_benchmarks.py \
  --models gemini-1.5-flash,gpt-4o-mini,claude-3-haiku-20240307 \
  --dataset gsm8k \
  --limit 10

# Compare on scientific reasoning
python run_benchmarks.py \
  --models gemini-1.5-flash,gpt-4o-mini,claude-3-haiku-20240307 \
  --dataset arc \
  --limit 10

# Full comparison on any dataset (more expensive)
python run_benchmarks.py \
  --models gemini-1.5-flash,gpt-4o-mini,claude-3-haiku-20240307 \
  --dataset truthfulness
```

### Step 4: Run Parallel Benchmarks (Faster)

For faster execution when testing multiple models:

```bash
# Quick parallel test on truthfulness (recommended first)
python run_benchmarks.py \
  --models gemini-1.5-flash,gpt-4o-mini,claude-3-haiku-20240307 \
  --dataset truthfulness \
  --limit 10 \
  --parallel

# Parallel testing on math problems
python run_benchmarks.py \
  --models gemini-1.5-flash,gpt-4o-mini,claude-3-haiku-20240307 \
  --dataset gsm8k \
  --limit 10 \
  --parallel

# Full parallel benchmarks (much faster than sequential, but may hit rate limits)
python run_benchmarks.py \
  --models gemini-1.5-flash,gpt-4o-mini,claude-3-haiku-20240307 \
  --dataset mmlu \
  --parallel
```

### Step 5: Comprehensive Multi-Dataset Testing

For a complete model assessment across different capabilities:

```bash
# Quick comprehensive test across all datasets (recommended)
for dataset in truthfulness arc gsm8k mmlu hellaswag; do
  echo "Testing $dataset..."
  python run_benchmarks.py --model gemini-1.5-flash --dataset $dataset --limit 10
done

# Compare two models across all capabilities
for dataset in truthfulness arc gsm8k mmlu hellaswag; do
  echo "Comparing models on $dataset..."
  python run_benchmarks.py \
    --models gemini-1.5-flash,gpt-4o-mini \
    --dataset $dataset \
    --limit 10 \
    --parallel
done

# Single command for quick assessment (if you have all API keys)
python run_benchmarks.py --all-models --dataset truthfulness --limit 5
python run_benchmarks.py --all-models --dataset gsm8k --limit 5
python run_benchmarks.py --all-models --dataset arc --limit 5
```

**💡 Pro Tip:** Start with `--limit 5` across all datasets to get a broad sense of model performance, then deep-dive into specific datasets that matter most for your use case.

### Step 6: Analyze Results

After completion, you'll see a summary table:

```
📊 Benchmark Results Summary
==================================================
Dataset: truthfulness
Models Tested: 3

📈 Model Comparison:
--------------------------------------------------------------------------------
Model                          Provider     Score      Success    Failed    Time (s)
--------------------------------------------------------------------------------
claude-3-haiku-20240307        anthropic    87.50%     14         2         45.23
gemini-1.5-flash               google       81.25%     13         3         23.45
gpt-4o-mini                    openai       75.00%     12         4         34.56
--------------------------------------------------------------------------------
```

### Step 7: Find Your Detailed Results

Results are saved as CSV files in the `results/` directory, organized by dataset:

```bash
# List results for all datasets
ls -la results/*/2024-*/

# View results for specific datasets
ls -la results/truthfulness/2024-*/   # Factual accuracy results
ls -la results/gsm8k/2024-*/          # Math problem results
ls -la results/arc/2024-*/            # Science reasoning results
ls -la results/mmlu/2024-*/           # Academic knowledge results
ls -la results/hellaswag/2024-*/      # Commonsense reasoning results

# View a specific result file
cat results/gsm8k/2024-01/benchmark_google_gemini-1-5-flash_gsm8k_*.csv
```

## 📊 Understanding the Results

### Key Metrics Explained

1. **Score**: Percentage of prompts where the model's response contained expected keywords
2. **Success**: Number of evaluations that passed
3. **Failed**: Number of evaluations that failed or errored
4. **Time**: Total execution time in seconds

### Interpreting Results by Dataset

Different datasets reveal different model strengths:

**📊 Typical Performance Patterns:**
- **truthfulness**: 70-95% (varies by model's training on factual content)
- **gsm8k**: 60-90% (mathematical reasoning capability)
- **arc**: 70-85% (scientific reasoning and logic)
- **mmlu**: 65-80% (broad academic knowledge)
- **hellaswag**: 75-90% (commonsense understanding)

**🎯 What Good Performance Looks Like:**
- **High truthfulness + high arc**: Strong factual and logical reasoning
- **High gsm8k**: Excellent for mathematical/computational tasks
- **High mmlu**: Good for academic/research applications
- **High hellaswag**: Great for real-world, conversational use cases
- **Consistent across all**: Well-rounded, general-purpose model

### CSV Columns

- `timestamp`: When the evaluation was run
- `model_name`: Full model identifier
- `prompt_text`: The question asked
- `model_response`: The model's answer
- `expected_keywords`: Keywords that should appear in response
- `matched_keywords`: Keywords that were found
- `score`: 1.0 for pass, 0.0 for fail
- `response_time_seconds`: Time to generate response

## 🎨 Advanced Usage

### Quick Testing with Limited Dataset

For development and testing purposes, use the `--limit` flag to process only the first N questions:

```bash
# Test with just 5 questions (very quick)
python run_benchmarks.py --model gemini-1.5-flash --dataset truthfulness --limit 5

# Quick test across different dataset types
python run_benchmarks.py --model gemini-1.5-flash --dataset gsm8k --limit 5      # Math
python run_benchmarks.py --model gemini-1.5-flash --dataset arc --limit 5        # Science
python run_benchmarks.py --model gemini-1.5-flash --dataset mmlu --limit 5       # Knowledge

# Test multiple models with 10 questions each
python run_benchmarks.py --models gemini-1.5-flash,gpt-4o-mini --dataset hellaswag --limit 10

# Quick parallel test across datasets
python run_benchmarks.py --models gemini-1.5-flash,gpt-4o-mini --dataset truthfulness --limit 10 --parallel
python run_benchmarks.py --models gemini-1.5-flash,gpt-4o-mini --dataset gsm8k --limit 10 --parallel
```

**Benefits of using `--limit`:**
- ⚡ Much faster execution (seconds vs minutes)
- 💰 Minimal API costs for testing
- 🔧 Perfect for development and debugging
- ✅ Validates your setup works correctly

### Evaluation Methods

Choose how responses are evaluated for accuracy:

```bash
# Strict keyword matching (original method)
python run_benchmarks.py --model gemini-1.5-flash --dataset truthfulness --limit 10 --evaluation-method keyword

# Fuzzy similarity matching (more forgiving)
python run_benchmarks.py --model gemini-1.5-flash --dataset truthfulness --limit 10 --evaluation-method fuzzy

# Multi-method evaluation (combines approaches, default)
python run_benchmarks.py --model gemini-1.5-flash --dataset truthfulness --limit 10 --evaluation-method multi
```

**Evaluation Method Comparison:**
- **`keyword`**: Requires exact phrase matches - often too strict ❌
- **`fuzzy`**: Uses similarity scoring - more accurate ✅
- **`multi`**: Combines multiple approaches - most robust ✅ **(Recommended)**

**Results Comparison Example:**
- Old method: 0% success rate (fails on good responses)
- New methods: 60-100% success rate (recognizes correct answers)

### Test All Available Models

```bash
# Benchmark every model you have API keys for on different datasets
python run_benchmarks.py --all-models --dataset truthfulness --limit 10 --parallel
python run_benchmarks.py --all-models --dataset gsm8k --limit 10 --parallel
python run_benchmarks.py --all-models --dataset arc --limit 10 --parallel

# Full comparison across all models (expensive but comprehensive)
python run_benchmarks.py --all-models --dataset truthfulness --parallel
```

### Custom Output Directory

```bash
# Save results to a specific location
python run_benchmarks.py \
  --model gpt-4o-mini \
  --dataset truthfulness \
  --output-dir ./my-benchmark-results
```

### Disable CSV Output (Console Only)

```bash
# Just see console output without saving files
python run_benchmarks.py \
  --model gemini-1.5-flash \
  --dataset truthfulness \
  --no-csv
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "No provider found" Error
```bash
# Check available models
python -c "from llm_providers import registry; print(registry.list_all_models())"
```

Available models vary by provider:
- Google: `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-1.0-pro`
- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`
- Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`, `claude-3-opus-20240229`, etc.

#### 2. API Key Errors
```bash
# Verify your API key is set
echo $GOOGLE_API_KEY  # Should show your key (not empty)

# Test the API key
curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
  "https://generativelanguage.googleapis.com/v1/models"
```

#### 3. Rate Limiting
If you hit rate limits, the tool automatically retries with exponential backoff. For persistent issues:
- Reduce parallel execution
- Add delays between models
- Upgrade your API plan

#### 4. Timeout Errors
For slow models or poor connections:
```bash
# Increase timeout (with experimental engine)
python run_benchmarks.py \
  --model claude-3-opus-20240229 \
  --dataset truthfulness \
  --use-engine \
  --timeout 120
```

## 📈 Next Steps

Now that you've mastered running benchmarks:

1. **Analyze Costs**: Run the cost analysis example to understand pricing
   ```bash
   python examples/use_cases/cost_analysis.py
   ```

2. **Compare Performance**: Use the performance analysis notebook
   ```bash
   python examples/notebooks/02_performance_analysis.py
   ```

3. **Create Custom Benchmarks**: Use our dataset conversion script as a template to create your own benchmarks

4. **Automate Reporting**: Set up scheduled runs with cron or GitHub Actions

### Related Use Cases
- [Use Case 2: Compare Cost vs Performance](./USE_CASE_2_HOW_TO.md) - Dive deeper into cost optimization
- [Use Case 3: Test Custom Prompts](./USE_CASE_3_HOW_TO.md) - Benchmark your specific use cases
- [Use Case 4: Run Tests Across LLMs](./USE_CASE_4_HOW_TO.md) - Comprehensive testing strategies

## 🎯 Pro Tips

💡 **Start Small**: Test with one model and `--limit 10` before running expensive comparisons

💡 **Use --limit for Development**: Always use `--limit 10` when testing setup, debugging, or trying new models

💡 **Use Parallel Wisely**: Parallel execution is faster but may hit rate limits

💡 **Monitor Costs**: GPT-4 and Claude Opus are significantly more expensive

💡 **Save Results**: Keep CSV outputs for historical comparison

💡 **Choose Models Wisely**:
   - For speed: `gemini-1.5-flash`
   - For cost: `gpt-4o-mini` or `gemini-1.5-flash`
   - For quality: `gpt-4o` or `claude-3-5-sonnet-20241022`

💡 **Cost Optimization**: Start with cheaper models (`gemini-1.5-flash`, `gpt-4o-mini`) for development and testing. Only use expensive models (`gpt-4`, `claude-3-opus`) for final comparisons or when quality is critical.

## 📚 Understanding the Benchmark Datasets

Each dataset tests different aspects of model capability:

### TruthfulQA (500 questions)
Tests whether models generate truthful answers and avoid misinformation:
- Categories: Health, Law, Finance, Politics, Fiction, Science
- Example: "What happens if you eat watermelon seeds?"

### ARC - AI2 Reasoning Challenge (1,000 questions)
Tests scientific reasoning and logical thinking:
- Grade 3-9 science questions requiring multi-step reasoning
- Example: "Why does ice float on water?"

### GSM8K - Grade School Math (1,000 questions)
Tests mathematical reasoning and problem-solving:
- Multi-step word problems requiring arithmetic
- Example: "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast..."

### MMLU - Massive Multitask Language Understanding (1,000 questions)
Tests broad academic knowledge across subjects:
- 57 subjects: mathematics, history, computer science, law, etc.
- Example: "Find the degree for the given field extension Q(√2, √3, √18) over Q"

### HellaSwag (1,000 questions)
Tests commonsense reasoning and real-world understanding:
- Sentence completion requiring common sense
- Example: "A man sits on a roof. He..." (multiple choice endings)

## 🔄 Continuous Improvement

This benchmarking system provides a foundation for:
- **Multi-capability assessment**: Test models across 5 different skill areas
- **Regular model performance tracking** across diverse benchmarks
- **A/B testing new models** as they're released with comprehensive evaluation
- **Domain-specific analysis**: Choose datasets that match your use case
- **Creating automated quality assurance** for LLM applications

## 📚 Additional Resources

- **Provider Documentation**:
  - [Google Gemini Guide](../providers/google.md)
  - [OpenAI GPT Guide](../providers/openai.md)
  - [Anthropic Claude Guide](../providers/anthropic.md)
- **Examples**: [Benchmarking Examples](../../examples/README.md)
- **Troubleshooting**: [Full Troubleshooting Guide](../TROUBLESHOOTING.md)

## 💭 Feedback

Help us improve this guide:
- Found an error? [Open an issue](https://github.com/remyolson/llm-lab/issues/new)
- Have suggestions? See our [Contribution Guide](../../CONTRIBUTING.md)

---

*Last updated: January 2025*
