# Custom Prompt Examples

This directory contains working examples demonstrating the complete custom prompt CLI interface and template system.

## 🎯 Quick Start

### Run All Examples
```bash
# From project root
./examples/custom_prompts/run_all_examples.sh

# Quick test mode (uses --limit 1)
./examples/custom_prompts/run_all_examples.sh --quick
```

### Run Individual Example Categories
```bash
# Customer service scenarios
./examples/custom_prompts/customer_service_examples.sh

# Code generation at different complexity levels
./examples/custom_prompts/code_generation_examples.sh

# Creative writing across genres
./examples/custom_prompts/creative_writing_examples.sh
```

## 📁 File Structure

```
examples/custom_prompts/
├── README.md                        # This file
├── run_all_examples.sh             # Master script for all examples
├── customer_service_examples.sh    # Customer service scenarios
├── code_generation_examples.sh     # Code generation examples
└── creative_writing_examples.sh    # Creative writing examples
```

Related template files:
```
templates/
├── customer_service/
│   ├── customer_service_response.txt
│   └── escalation_handling.txt
├── code_generation/
│   ├── code_generation.txt
│   └── code_review.txt
└── creative_writing/
    ├── creative_writing.txt
    └── story_generation.txt
```

## 🎨 Example Categories

### 1. Customer Service Examples (`customer_service_examples.sh`)

**What it demonstrates:**
- Professional customer service responses
- Escalation handling procedures
- Technical support scenarios
- Different response tones (professional, empathetic, friendly)
- Template variable usage with conditionals

**Example scenarios:**
- Basic support request (software crashes)
- Critical escalation (data corruption)
- Technical documentation questions
- Tone comparison testing

**Template features used:**
- Conditional sections (`{?variable}...{/variable}`)
- Required and optional variables
- Built-in variables (`{model_name}`)
- Multi-level complexity

### 2. Code Generation Examples (`code_generation_examples.sh`)

**What it demonstrates:**
- Code generation across complexity levels (beginner → advanced)
- Different programming tasks and algorithms
- Code review and analysis
- Style comparison (functional vs OOP)
- Technical documentation generation

**Example tasks:**
- Beginner: Simple circle area calculation
- Intermediate: CSV data processing with validation
- Advanced: A* pathfinding algorithm implementation
- Code review of security-vulnerable code
- Batch generation of different function types

**Template features used:**
- Nested conditionals based on complexity level
- Multi-paragraph template structure
- Technical requirement specifications
- Code style preferences

### 3. Creative Writing Examples (`creative_writing_examples.sh`)

**What it demonstrates:**
- Multiple creative formats (stories, poetry, dialogue)
- Genre-specific requirements and constraints
- Style variations and tone control
- Character and world-building
- Narrative structure guidance

**Example formats:**
- Science fiction short story (600-800 words)
- Mystery story with atmosphere building
- Free verse poetry with urban themes
- Psychological thriller dialogue
- Fantasy world-building
- Genre comparison (horror, romance, historical fiction, western)

**Template features used:**
- Content-type specific requirements
- Complex character and setting variables
- Style and mood specifications
- Target audience considerations

## 🔧 Configuration Options

### Environment Variables

```bash
# Set default models for all examples
export EXAMPLE_MODELS="gpt-4,claude-3-sonnet,gemini-pro"

# Enable quick mode for all scripts
export QUICK_MODE="--limit 1"
```

### Script Options

```bash
# Quick testing (uses --limit 1 for all examples)
./run_all_examples.sh --quick

# Custom model selection
EXAMPLE_MODELS="gpt-4o-mini,claude-3-haiku" ./run_all_examples.sh

# Individual examples with custom options
python scripts/run_benchmarks.py \
  --prompt-file templates/customer_service/customer_service_response.txt \
  --prompt-variables '{"company_name": "YourCompany"}' \
  --models your-preferred-models \
  --custom-options
```

## 📊 Expected Results

### Output Formats
Examples generate results in multiple formats:
- **JSON**: Structured data with metrics (default)
- **Markdown**: Human-readable reports
- **CSV**: Spreadsheet-compatible data

### Result Organization
```
results/examples/
├── customer-service/
│   ├── basic/              # Standard support scenarios
│   ├── escalation/         # Escalated issue handling
│   ├── technical/          # Technical support
│   └── tones/              # Different response tones
├── code-generation/
│   ├── beginner/           # Simple functions
│   ├── intermediate/       # Data processing
│   ├── advanced/           # Complex algorithms
│   ├── review/             # Code review examples
│   ├── batch/              # Multiple function types
│   └── style-test/         # Style comparisons
└── creative-writing/
    ├── sci-fi/             # Science fiction stories
    ├── mystery/            # Mystery stories
    ├── poetry/             # Poetry generation
    ├── dialogue/           # Screenplay dialogue
    ├── fantasy/            # Fantasy stories
    ├── genres/             # Genre comparisons
    └── style-test/         # Writing style comparisons
```

### Metrics and Analysis
Each example tracks relevant metrics:
- **Response length** (words, sentences)
- **Sentiment analysis** (positive/negative/neutral)
- **Coherence scores** (logical flow)
- **Creativity scores** (for creative writing)
- **Code quality** (for programming tasks)
- **Professional tone** (for customer service)

## 🎯 Usage Patterns

### For Learning
1. **Start with `run_all_examples.sh --quick`** to see basic functionality
2. **Examine individual scripts** to understand CLI patterns
3. **Modify template variables** to test different scenarios
4. **Compare model outputs** to understand strengths/weaknesses

### For Development
1. **Copy and modify templates** for your specific use cases
2. **Use examples as CLI command references**
3. **Extend scripts** with your own test scenarios
4. **Integrate into CI/CD pipelines** for automated testing

### For Production Planning
1. **Analyze cost vs performance** across different models
2. **Test edge cases** with your specific data
3. **Validate template robustness** with various inputs
4. **Establish quality baselines** for your use cases

## 💡 Tips for Success

### Template Modification
- **Start with existing templates** and modify incrementally
- **Test variable substitution** with simple examples first
- **Use conditional sections** to handle optional parameters
- **Document required vs optional variables** in template comments

### CLI Usage
- **Use `--limit 1`** during development to save costs
- **Enable `--parallel`** for faster multi-model comparisons
- **Specify `--output-dir`** to organize results logically
- **Include multiple `--output-format`** options for flexibility

### Model Selection
- **Start with cheaper models** (gpt-4o-mini, claude-3-haiku) for testing
- **Use premium models** (gpt-4, claude-3-opus) for final evaluation
- **Consider task complexity** when choosing models
- **Balance cost, speed, and quality** for your specific needs

## 🐛 Troubleshooting

### Common Issues

**Script fails to run:**
```bash
# Check permissions
chmod +x examples/custom_prompts/*.sh

# Run from project root
cd /path/to/llm-lab
./examples/custom_prompts/run_all_examples.sh
```

**Template variable errors:**
```bash
# Validate JSON syntax
echo '{"key": "value"}' | python -m json.tool

# Check required variables
python -c "
from src.use_cases.custom_prompts import PromptTemplate
t = PromptTemplate(open('templates/customer_service/customer_service_response.txt').read())
print('Required:', t.get_required_variables())
"
```

**API key issues:**
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
```

### Getting Help

- **Check CLI documentation**: `docs/guides/CUSTOM_PROMPT_CLI.md`
- **Review template syntax**: Look at existing template files
- **Test with minimal examples**: Use simple prompts first
- **Enable verbose logging**: Add `--verbose` to CLI commands

---

*These examples provide a comprehensive demonstration of the custom prompt system's capabilities. Use them as starting points for your own prompt engineering workflows.*
