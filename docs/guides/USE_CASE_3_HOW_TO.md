# Use Case 3: Test Custom Prompts

*Evaluate models on your specific use cases and prompts with comprehensive metrics and cross-model comparison.*

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:
- **Run custom prompts** across multiple LLM providers simultaneously
- **Use template variables** to create dynamic, reusable prompts
- **Automatically save detailed results** to JSON/CSV/Markdown for analysis
- Generate comprehensive reports with **evaluation metrics** (sentiment, coherence, diversity)
- Analyze which **models perform best** for your specific use cases
- Understand the **cost implications** and optimization strategies
- Make **data-driven decisions** for model selection in production
- Create **A/B testing workflows** for prompt optimization

## 📋 Before You Begin

- Complete all [Prerequisites](./PREREQUISITES.md)
- Ensure you have API keys for at least one provider
- Time required: ~15 minutes
- Estimated cost: $0.10-$2.00 per run

### 💰 Cost Breakdown
Running custom prompts with different providers:

**💡 Pro Tip:** Use `--limit 3` for testing to reduce costs by 95% (approximately $0.005-$0.10 per test run)

- **OpenAI**:
  - `gpt-4o-mini`: ~$0.10 per 100 prompts (most cost-effective)
  - `gpt-4`: ~$1.50 per 100 prompts
  - **Free tier eligible**: $5 credit for new accounts

- **Anthropic**:
  - `claude-3-haiku`: ~$0.25 per 100 prompts
  - `claude-3-sonnet`: ~$1.50 per 100 prompts
  - **Free tier eligible**: Limited usage with API key

- **Google**:
  - `gemini-flash`: ~$0.075 per 100 prompts
  - `gemini-pro`: ~$0.50 per 100 prompts
  - **Free tier eligible**: 15 requests per minute

*Note: Costs are estimates based on December 2024 pricing. Actual costs depend on response lengths. Single model tests cost proportionally less.*

## 📊 Available Prompt Types

Choose the right approach based on what capability you want to test:

| Approach | Complexity | What It Tests | Best For | Example |
|----------|------------|---------------|----------|---------|
| **Simple Prompt** | Basic | Direct model capability | Quick testing, comparisons | "Explain quantum computing" |
| **Template Variables** | Medium | Dynamic content generation | Scalable testing, personalization | "Explain {topic} to a {audience}" |
| **Template Files** | Medium | Structured prompts with context | Complex scenarios, reusability | Code review templates |
| **Conditional Templates** | Advanced | Context-aware responses | Adaptive prompts, workflows | Customer service scenarios |
| **Batch Processing** | Advanced | Large-scale evaluation | Production testing, research | Multiple topics/variations |

### 🎯 **Prompt Selection Guide:**

- **🔍 For model comparison:** Start with simple prompts and `--parallel`
- **🧮 For dynamic content:** Use template variables with JSON input
- **🎓 For complex scenarios:** Create template files with conditions
- **🌍 For production planning:** Use batch processing with metrics
- **📊 For comprehensive assessment:** Run all approaches with `--limit 5` first

## 🚀 Step-by-Step Guide

### Step 1: Quick Start with Simple Prompts

First, test with a simple prompt to ensure everything works:

```bash
# Quick test with a simple prompt (recommended for first run)
python scripts/run_benchmarks.py \
  --custom-prompt "What is machine learning?" \
  --models gpt-4o-mini \
  --limit 1

# Try different model providers
python scripts/run_benchmarks.py \
  --custom-prompt "Explain recursion in programming" \
  --models gpt-4o-mini,claude-3-haiku,gemini-flash \
  --limit 1

# Full comparison without limit (more expensive)
python scripts/run_benchmarks.py \
  --custom-prompt "Write a haiku about programming" \
  --models gpt-4o-mini,claude-3-haiku
```

**What Happens:**
- Your custom prompt is sent to the specified models
- Results are automatically saved to `results/custom_prompts/YYYY-MM/`
- Progress is displayed in real-time in the console
- Evaluation metrics are calculated automatically (length, sentiment, coherence)

**Expected Output:**
```
🚀 Running custom prompt on 1 model(s)...

📝 Using custom prompt: What is machine learning?

Running on gpt-4o-mini...
   ✓ openai provider initialized for model gpt-4o-mini
   ✓ Generated response (2.1s)
   📏 Length: 145 words, 8 sentences
   💭 Sentiment: neutral 😐 (score: 0.2)
   🔗 Coherence: High (score: 0.89)

✅ Completed in 2.1 seconds
```

### Step 2: Using Template Variables

Add dynamic content with template variables:

```bash
# Use template variables for dynamic prompts
python scripts/run_benchmarks.py \
  --custom-prompt "You are {model_name}, an expert in {domain}. Explain {topic} to a {audience}." \
  --prompt-variables '{"domain": "computer science", "topic": "algorithms", "audience": "beginner"}' \
  --models gpt-4,claude-3-sonnet \
  --limit 1

# Compare different audiences
python scripts/run_benchmarks.py \
  --custom-prompt "Explain {concept} for a {level} audience" \
  --prompt-variables '{"concept": "blockchain", "level": "technical"}' \
  --models gpt-4o-mini,claude-3-haiku \
  --limit 1
```

### Step 3: Template Files for Complex Prompts

Create reusable template files for complex scenarios:

```bash
# Create a template file
cat > my_prompt_template.txt << 'EOF'
You are {model_name}, a helpful AI assistant. Today's date is {date}.

{?context}Given the following context:
{context}

{/context}Please provide a {style} explanation of: {topic}

Requirements:
- Use simple language
- Provide concrete examples
- Keep it under {max_words} words
EOF

# Use the template file
python scripts/run_benchmarks.py \
  --prompt-file my_prompt_template.txt \
  --prompt-variables '{"style": "beginner-friendly", "topic": "neural networks", "max_words": "200", "context": "Focus on practical applications"}' \
  --models gpt-4,claude-3-sonnet,gemini-pro \
  --limit 1
```

### Step 4: Parallel Execution for Faster Results

For faster execution when testing multiple models:

```bash
# Quick parallel test (recommended first)
python scripts/run_benchmarks.py \
  --custom-prompt "Compare Python and JavaScript for web development" \
  --models gpt-4o-mini,claude-3-haiku,gemini-flash \
  --limit 1 \
  --parallel

# Parallel testing with template variables
python scripts/run_benchmarks.py \
  --custom-prompt "Write a {type} function in {language}" \
  --prompt-variables '{"type": "sorting", "language": "Python"}' \
  --models gpt-4,claude-3-sonnet,gemini-pro \
  --parallel

# Full parallel comparison (faster than sequential, but may hit rate limits)
python scripts/run_benchmarks.py \
  --prompt-file examples/prompts/code_review_template.txt \
  --prompt-variables '{"language": "python", "code": "def add(a,b): return a+b"}' \
  --models gpt-4o-mini,claude-3-haiku,gemini-flash \
  --parallel
```

### Step 5: Working with Multiple Output Formats

Save results in different formats for various use cases:

```bash
# Save as JSON (default, best for analysis)
python scripts/run_benchmarks.py \
  --custom-prompt "Analyze the pros and cons of {technology}" \
  --prompt-variables '{"technology": "microservices"}' \
  --models gpt-4,claude-3 \
  --output-format json

# Save as CSV (great for spreadsheets)
python scripts/run_benchmarks.py \
  --custom-prompt "Explain {topic}" \
  --prompt-variables '{"topic": "containerization"}' \
  --models gpt-4o-mini,gemini-flash \
  --output-format csv

# Save as Markdown (human-readable reports)
python scripts/run_benchmarks.py \
  --prompt-file examples/prompts/reasoning_template.txt \
  --prompt-variables '{"question": "Why do we use version control?"}' \
  --models claude-3-sonnet \
  --output-format markdown
```

### Step 6: Advanced Metrics and Analysis

Enable comprehensive evaluation metrics:

```bash
# Run with all metrics enabled
python scripts/run_benchmarks.py \
  --custom-prompt "Write a technical blog post about {topic}" \
  --prompt-variables '{"topic": "API design"}' \
  --models gpt-4,claude-3-opus \
  --metrics all \
  --parallel

# Batch processing for trend analysis
for topic in "recursion" "dynamic-programming" "graph-algorithms"; do
  python scripts/run_benchmarks.py \
    --custom-prompt "Explain {topic} with examples" \
    --prompt-variables "{\"topic\": \"$topic\"}" \
    --models gpt-4o-mini,claude-3-haiku \
    --output-dir "./results/cs-concepts" \
    --parallel
done
```

**💡 Pro Tip:** Start with `--limit 1` and single models to test your prompts, then scale up to multiple models and full runs once you're satisfied with the setup.

## 📊 Understanding the Results

### Key Metrics Explained

1. **Response Length**: Word count, sentence count, and average word length
2. **Sentiment Analysis**: Positive/negative/neutral classification with confidence scores
3. **Coherence Score**: Measures logical flow and consistency (0.0-1.0, higher is better)
4. **Response Diversity**: Variation across multiple model responses (when using multiple models)

### Interpreting Results by Use Case

Different prompts reveal different model strengths:

**📊 Typical Performance Patterns:**
- **Creative writing**: GPT-4 often shows higher diversity, Claude-3 more coherent structure
- **Technical explanations**: All models perform well, Gemini often more concise
- **Code generation**: GPT-4 and Claude-3 typically more accurate, better formatting
- **Customer service**: Claude-3 often more empathetic tone, GPT-4 more comprehensive
- **Analysis tasks**: All models comparable, Claude-3 often more balanced perspectives

**🎯 What Good Performance Looks Like:**
- **High coherence (>0.8) + appropriate length**: Well-structured, complete responses
- **Consistent sentiment**: Appropriate emotional tone for the context
- **Low diversity across runs**: Reliable, consistent model behavior
- **High diversity across models**: Good for comparison and getting varied perspectives

### Example Results

```json
{
  "execution_id": "20241204_143022_custom",
  "prompt_template": "Explain {topic} for a {audience} audience",
  "template_variables": {
    "topic": "machine learning",
    "audience": "beginner"
  },
  "models_succeeded": ["gpt-4", "claude-3-sonnet"],
  "responses": [
    {
      "model": "gpt-4",
      "success": true,
      "response": "Machine learning is like teaching computers to learn patterns...",
      "metrics": {
        "response_length": {"words": 156, "sentences": 9},
        "sentiment": {"score": 0.3, "label": "positive"},
        "coherence": {"score": 0.91}
      }
    }
  ],
  "aggregated_metrics": {
    "diversity": {"score": 0.73},
    "average_coherence": 0.89
  }
}
```

### Result Organization

Results are saved as organized files in the `results/` directory:

```bash
# List results for all custom prompts
ls -la results/custom_prompts/*/

# View results for specific dates
ls -la results/custom_prompts/2024-*/

# View a specific result file
cat results/custom_prompts/2024-12-04/20241204_143022_custom.json

# View markdown report
cat results/custom_prompts/2024-12-04/20241204_143022_custom.md
```

## 🎨 Advanced Usage

### Quick Testing with Limited Prompts

For development and testing purposes, use the `--limit` flag:

```bash
# Test with just 1 execution (very quick)
python scripts/run_benchmarks.py \
  --custom-prompt "Explain {concept}" \
  --prompt-variables '{"concept": "APIs"}' \
  --models gpt-4 \
  --limit 1

# Quick test across different models
python scripts/run_benchmarks.py \
  --custom-prompt "Write a function to {task}" \
  --prompt-variables '{"task": "reverse a string"}' \
  --models gpt-4o-mini,claude-3-haiku,gemini-flash \
  --limit 1
```

**Benefits of using `--limit`:**
- ⚡ Much faster execution (seconds vs minutes)
- 💰 Minimal API costs for testing
- 🔧 Perfect for development and debugging
- ✅ Validates your setup works correctly

### Custom Evaluation Metrics

Create domain-specific metrics for your use cases:

```bash
# For code generation prompts
python scripts/run_benchmarks.py \
  --custom-prompt "Write a {language} function to {task}" \
  --prompt-variables '{"language": "Python", "task": "sort a list"}' \
  --models gpt-4,claude-3 \
  --metrics code_quality,syntax_check \
  --parallel

# For creative writing
python scripts/run_benchmarks.py \
  --custom-prompt "Write a {genre} story about {theme}" \
  --prompt-variables '{"genre": "sci-fi", "theme": "time travel"}' \
  --models gpt-4,claude-3-opus \
  --metrics creativity,narrative_structure \
  --parallel
```

### Batch Processing Multiple Scenarios

```bash
# Process multiple related prompts
topics=("recursion" "iteration" "dynamic-programming")
for topic in "${topics[@]}"; do
  python scripts/run_benchmarks.py \
    --custom-prompt "Explain {topic} with a practical coding example" \
    --prompt-variables "{\"topic\": \"$topic\"}" \
    --models gpt-4o-mini,claude-3-haiku \
    --output-dir "./results/algorithms" \
    --parallel
done

# A/B test different prompt formulations
python scripts/run_benchmarks.py \
  --custom-prompt "Explain recursion clearly" \
  --models gpt-4 \
  --limit 3 \
  --output-dir "./results/prompt-a"

python scripts/run_benchmarks.py \
  --custom-prompt "You are a computer science teacher. Explain recursion to a student who's struggling with the concept" \
  --models gpt-4 \
  --limit 3 \
  --output-dir "./results/prompt-b"
```

### Custom Output Directory

```bash
# Save results to a specific location
python scripts/run_benchmarks.py \
  --custom-prompt "Review this code: {code}" \
  --prompt-variables '{"code": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"}' \
  --models gpt-4,claude-3 \
  --output-dir ./my-code-reviews
```

## 🎯 Pro Tips

💡 **Start Small**: Test with one model and `--limit 1` before running expensive comparisons

💡 **Use Templates**: Create reusable `.txt` files for complex prompts you'll use repeatedly

💡 **Enable Parallel**: Use `--parallel` for faster multi-model comparisons

💡 **Monitor Costs**: Start with cheaper models (`gpt-4o-mini`, `claude-3-haiku`, `gemini-flash`) for development

💡 **Save Results**: Keep JSON outputs for historical comparison and trend analysis

💡 **A/B Test Prompts**: Compare different prompt formulations to optimize for your use case

💡 **Use Variables**: Template variables make prompts reusable and enable systematic testing

💡 **Check Rate Limits**: Some providers have strict rate limits - use `--parallel` cautiously

💡 **Analyze Metrics**: Pay attention to coherence scores and diversity metrics for quality assessment

💡 **Batch Process**: For multiple related prompts, use shell loops or scripts for efficiency

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue: "No API key found for provider"
**Solution**:
```bash
# Check your environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY

# Set missing keys
export OPENAI_API_KEY="your-key-here"
```

#### Issue: "Rate limit exceeded"
**Solution**:
- Remove `--parallel` flag for sequential execution
- Use `--limit` to reduce the number of requests
- Wait a few minutes before retrying

#### Issue: "Template variable not found"
**Solution**:
```bash
# Check your JSON syntax
echo '{"topic": "ML", "audience": "beginners"}' | python -m json.tool

# Use single quotes around JSON
--prompt-variables '{"key": "value"}'
```

#### Issue: "Model not supported"
**Solution**:
```bash
# List available models
python scripts/run_benchmarks.py --list-models

# Use supported model names
--models gpt-4o-mini,claude-3-haiku,gemini-flash
```

### Debugging Commands

```bash
# Test API connectivity
python scripts/run_benchmarks.py --custom-prompt "test" --models gpt-4o-mini --limit 1

# Validate template syntax
python -c "
from src.use_cases.custom_prompts import PromptTemplate
t = PromptTemplate('Your template here')
print('Variables:', t.get_required_variables())
"

# Check results directory
ls -la results/custom_prompts/
```

## 📈 Next Steps

Now that you've mastered custom prompt testing:

1. **Analyze Costs vs Performance**: Create systematic comparisons of model cost-effectiveness
   ```bash
   python examples/use_cases/cost_analysis.py
   ```

2. **Set Up A/B Testing**: Use our automated prompt comparison tools
   ```bash
   python examples/use_cases/prompt_ab_testing.py
   ```

3. **Create Custom Metrics**: Build domain-specific evaluation criteria
   ```bash
   python examples/use_cases/custom_metrics_example.py
   ```

4. **Automate with CI/CD**: Set up GitHub Actions for continuous prompt testing

### Related Use Cases
- [Use Case 1: Run Standard Benchmarks](./USE_CASE_1_HOW_TO.md) - Systematic model evaluation
- [Use Case 2: Compare Cost vs Performance](./USE_CASE_2_HOW_TO.md) - Economic analysis
- [Use Case 4: Run Tests Across LLMs](./USE_CASE_4_HOW_TO.md) - Comprehensive testing strategies

## 📚 Understanding the Template System

The template system supports powerful features for dynamic prompt generation:

### Template Variables ({variable})
Basic variable substitution:
- `{topic}` - Topic to discuss
- `{model_name}` - Automatically filled with the current model
- `{date}` - Current date
- `{timestamp}` - Full timestamp

### Conditional Sections ({?condition})
Show content only if a variable exists and is truthy:
```
{?context}Given this context: {context}{/context}
```

### Built-in Variables
Always available without specifying:
- `{timestamp}` - Current ISO timestamp
- `{date}` - Current date (YYYY-MM-DD)
- `{time}` - Current time (HH:MM:SS)
- `{model_name}` - Name of the current model

### Example Template File
```
You are {model_name}, an expert in {domain}.
Generated on: {date}

{?context}
Context: {context}
{/context}

Task: {task}

Requirements:
- Be accurate and helpful
- Use examples when appropriate
{?max_length}- Keep response under {max_length} words{/max_length}
```

## 🔄 Continuous Improvement

This custom prompt system provides a foundation for:
- **Multi-domain assessment**: Test models across different skill areas and domains
- **Regular performance tracking** with metrics trending over time
- **A/B testing new prompt strategies** as models evolve
- **Domain-specific optimization**: Tune prompts for your specific use cases
- **Creating automated quality assurance** for AI-powered applications

## 📚 Additional Resources

- **Provider Documentation**:
  - [OpenAI Guide](../providers/openai.md)
  - [Anthropic Guide](../providers/anthropic.md)
  - [Google Guide](../providers/google.md)
- **Examples**: [Custom Prompt Examples](../../examples/use_cases/)
- **Troubleshooting**: [Full Troubleshooting Guide](../TROUBLESHOOTING.md)

## 💭 Feedback

Help us improve this guide:
- Found an error? [Open an issue](https://github.com/remyolson/llm-lab/issues/new)
- Have suggestions? See our [Contribution Guide](../../CONTRIBUTING.md)

---

*Last updated: December 2024*
