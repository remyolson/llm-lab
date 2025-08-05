# Custom Prompt CLI Interface and Template System

*Complete guide to using the CLI for custom prompt testing with the powerful template engine.*

## 🎯 Overview

The LLM Lab custom prompt system provides a sophisticated CLI interface for testing prompts across multiple LLM providers. This guide covers:

- **CLI Command Reference**: All available flags and options
- **Template System**: Variables, conditionals, and built-in features
- **Working Examples**: Three complete use cases with actual commands
- **Output Formats**: JSON, CSV, and Markdown results
- **Best Practices**: Optimization strategies for different scenarios

## 📋 CLI Command Reference

### Basic Syntax

```bash
python scripts/run_benchmarks.py [OPTIONS]
```

### Core Options

| Option | Description | Example |
|--------|-------------|---------|
| `--custom-prompt TEXT` | Simple prompt string | `--custom-prompt "Explain {topic}"` |
| `--prompt-file PATH` | Path to template file | `--prompt-file ./templates/review.txt` |
| `--prompt-variables JSON` | Template variables | `--prompt-variables '{"topic": "AI"}'` |
| `--models LIST` | Comma-separated model list | `--models gpt-4,claude-3-sonnet` |
| `--parallel` | Run models in parallel | `--parallel` |
| `--limit INTEGER` | Limit number of executions | `--limit 3` |
| `--output-format FORMAT` | Output format(s) | `--output-format json,csv,markdown` |
| `--output-dir PATH` | Custom output directory | `--output-dir ./my-results` |
| `--metrics LIST` | Enable specific metrics | `--metrics sentiment,coherence,all` |

### Advanced Options

| Option | Description | Default |
|--------|-------------|---------|
| `--temperature FLOAT` | Model temperature (0.0-2.0) | `0.7` |
| `--max-tokens INTEGER` | Maximum response tokens | Provider default |
| `--timeout INTEGER` | Request timeout (seconds) | `60` |
| `--retry-count INTEGER` | Number of retries on failure | `3` |
| `--provider-config PATH` | Custom provider configuration | `None` |
| `--cache-responses` | Enable response caching | `False` |
| `--verbose` | Detailed logging output | `False` |

### Model Selection

**Available Models:**
```bash
# OpenAI Models
gpt-4o-mini, gpt-4o, gpt-4, gpt-3.5-turbo

# Anthropic Models  
claude-3-haiku, claude-3-sonnet, claude-3-opus, claude-3-5-sonnet

# Google Models
gemini-flash, gemini-pro, gemini-ultra

# List all available models
python scripts/run_benchmarks.py --list-models
```

## 🎨 Template System Guide

### Basic Variable Substitution

Variables are defined with `{variable_name}` syntax:

```text
You are an expert in {domain}.
Explain {topic} to a {audience} audience.
Keep your response under {word_limit} words.
```

**Usage:**
```bash
python scripts/run_benchmarks.py \
  --prompt-file template.txt \
  --prompt-variables '{"domain": "AI", "topic": "neural networks", "audience": "beginner", "word_limit": 200}'
```

### Built-in Variables

These variables are automatically available:

| Variable | Description | Example |
|----------|-------------|---------|
| `{model_name}` | Current model being used | `gpt-4` |
| `{timestamp}` | ISO timestamp | `2024-08-05T14:30:22Z` |
| `{date}` | Current date | `2024-08-05` |
| `{time}` | Current time | `14:30:22` |

**Template Example:**
```text
You are {model_name}, responding on {date}.
{prompt_content}
```

### Conditional Sections

Use `{?variable}...{/variable}` for conditional content:

```text
You are an AI assistant.

{?context}
Context: {context}

{/context}
{?examples}
Here are some examples:
{examples}

{/examples}
Please answer: {question}

{?constraints}
Constraints:
- {constraints}
{/constraints}
```

**Behavior:**
- If `context` is provided and non-empty, the context section appears
- If `context` is empty/null/undefined, the section is omitted
- Same logic applies to `examples` and `constraints`

### Advanced Template Features

#### Nested Variables
```text
{?user_info}
User: {user_info.name} ({user_info.role})
Department: {user_info.department}
{/user_info}
```

#### Multiple Conditions
```text
{?is_technical}
Technical details: {technical_specs}
{/is_technical}

{?is_beginner}
Simplified explanation: {simple_explanation}
{/is_beginner}
```

## 📁 Template File Organization

### Recommended Structure

```
templates/
├── customer_service/
│   ├── support_response.txt
│   ├── escalation_handling.txt
│   └── feedback_analysis.txt
├── code_generation/
│   ├── function_creation.txt
│   ├── code_review.txt
│   └── debugging_help.txt
├── creative_writing/
│   ├── story_generation.txt
│   ├── character_development.txt
│   └── dialogue_writing.txt
└── analysis/
    ├── document_summary.txt
    ├── data_interpretation.txt
    └── comparison_analysis.txt
```

### Template File Best Practices

1. **Use descriptive filenames**: `customer_support_escalation.txt` not `template1.txt`
2. **Include header comments**: Document the purpose and required variables
3. **Group related templates**: Organize by use case or domain
4. **Version control**: Keep templates in git for collaboration

## 🎯 Complete Working Examples

### Example 1: Customer Service Response Analysis

**Use Case:** Compare how different LLMs handle customer service scenarios with varying tones and complexity levels.

#### Template File: `templates/customer_service_response.txt`

```text
You are {model_name}, a professional customer service representative for {company_name}.

Customer Information:
- Account Type: {customer_type}
- Issue Severity: {severity}
{?previous_interactions}
- Previous Interactions: {previous_interactions}
{/previous_interactions}

Customer Message:
"{customer_message}"

{?company_guidelines}
Company Guidelines:
{company_guidelines}

{/company_guidelines}
Please provide a professional response that:
1. Acknowledges the customer's concern
2. {?resolution_steps}Offers these resolution steps: {resolution_steps}{/resolution_steps}{?no_resolution_steps}Explains next steps for resolution{/no_resolution_steps}
3. Maintains a {tone} tone throughout
4. {?escalation_needed}Includes escalation information{/escalation_needed}

Response length: Keep under {max_words} words.
```

#### CLI Commands

```bash
# Test 1: Standard Support Request
python scripts/run_benchmarks.py \
  --prompt-file templates/customer_service_response.txt \
  --prompt-variables '{
    "company_name": "TechCorp Solutions",
    "customer_type": "Premium",
    "severity": "Medium",
    "customer_message": "My software keeps crashing when I try to export large files. This is very frustrating and affecting my work.",
    "tone": "empathetic and solution-focused",
    "max_words": "150",
    "company_guidelines": "Always offer a callback within 24 hours for premium customers"
  }' \
  --models gpt-4,claude-3-sonnet,gemini-pro \
  --parallel \
  --output-format json,markdown \
  --output-dir ./results/customer-service

# Test 2: Escalation Scenario
python scripts/run_benchmarks.py \
  --prompt-file templates/customer_service_response.txt \
  --prompt-variables '{
    "company_name": "TechCorp Solutions", 
    "customer_type": "Standard",
    "severity": "High",
    "customer_message": "This is my third email about the same issue! Your previous agent promised a fix that never came. I want a refund NOW!",
    "previous_interactions": "Agent Sarah: Promised callback Tuesday. Agent Mike: Said engineering would fix by Friday.",
    "tone": "professional and de-escalating",
    "escalation_needed": true,
    "max_words": "200"
  }' \
  --models gpt-4,claude-3-sonnet \
  --parallel \
  --metrics sentiment,coherence \
  --output-dir ./results/customer-service/escalations

# Test 3: Technical Support  
python scripts/run_benchmarks.py \
  --prompt-file templates/customer_service_response.txt \
  --prompt-variables '{
    "company_name": "DevTools Inc",
    "customer_type": "Enterprise", 
    "severity": "Low",
    "customer_message": "How do I configure SSL certificates for the development environment? The documentation seems outdated.",
    "resolution_steps": "1) Download latest SSL guide 2) Use dev-cert-tool 3) Contact DevOps if issues persist",
    "tone": "helpful and technical",
    "max_words": "250"
  }' \
  --models gpt-4o-mini,claude-3-haiku,gemini-flash \
  --limit 2 \
  --output-format csv

# Batch test multiple severity levels
for severity in "Low" "Medium" "High"; do
  python scripts/run_benchmarks.py \
    --prompt-file templates/customer_service_response.txt \
    --prompt-variables "{\"company_name\": \"ServiceCorp\", \"customer_type\": \"Standard\", \"severity\": \"$severity\", \"customer_message\": \"I need help with my account\", \"tone\": \"professional\", \"max_words\": \"150\"}" \
    --models gpt-4,claude-3-sonnet \
    --output-dir "./results/severity-analysis/$severity"
done
```

#### Expected Output Analysis

**Metrics to Compare:**
- **Sentiment Score**: How empathetic/professional the tone is
- **Response Length**: Adherence to word limits
- **Coherence**: Logical flow and completeness
- **Technical Accuracy**: Proper escalation procedures

**Typical Results:**
- **GPT-4**: Often more detailed, formal language
- **Claude-3-Sonnet**: Better emotional intelligence, natural empathy
- **Gemini-Pro**: Concise, direct, good for technical issues

### Example 2: Code Generation with Complexity Levels

**Use Case:** Generate Python functions with different complexity requirements and compare code quality across models.

#### Template File: `templates/code_generation.txt`

```text
You are {model_name}, an expert Python developer.

Task: Create a Python function with the following specifications:

Function Requirements:
- Name: {function_name}
- Purpose: {function_purpose}
- Complexity Level: {complexity_level}
{?input_parameters}
- Parameters: {input_parameters}
{/input_parameters}
{?return_type}
- Return Type: {return_type}
{/return_type}

{?additional_requirements}
Additional Requirements:
{additional_requirements}

{/additional_requirements}
{?complexity_level}
{?complexity_level=="beginner"}
Requirements for Beginner Level:
- Use simple, readable code
- Include detailed comments
- Avoid complex algorithms
- Handle basic error cases
{/complexity_level}

{?complexity_level=="intermediate"}
Requirements for Intermediate Level:
- Use appropriate data structures
- Include type hints
- Handle edge cases
- Add basic optimization
- Include docstring
{/complexity_level}

{?complexity_level=="advanced"}
Requirements for Advanced Level:
- Implement optimal algorithms
- Handle all edge cases elegantly
- Use advanced Python features appropriately
- Include comprehensive error handling
- Add performance considerations
- Include full documentation
{/complexity_level}
{/complexity_level}

Please provide:
1. The complete function implementation
2. {?include_examples}Usage examples{/include_examples}
3. {?include_tests}Basic test cases{/include_tests}
4. Brief explanation of your approach

{?style_preferences}
Code Style: {style_preferences}
{/style_preferences}
```

#### CLI Commands

```bash
# Test 1: Beginner Level - Simple Function
python scripts/run_benchmarks.py \
  --prompt-file templates/code_generation.txt \
  --prompt-variables '{
    "function_name": "calculate_area",
    "function_purpose": "Calculate the area of a rectangle",
    "complexity_level": "beginner",
    "input_parameters": "length (float), width (float)",
    "return_type": "float",
    "include_examples": true,
    "style_preferences": "Clear variable names, detailed comments"
  }' \
  --models gpt-4,claude-3-sonnet,gemini-pro \
  --parallel \
  --output-format json,markdown \
  --output-dir ./results/code-generation/beginner

# Test 2: Intermediate Level - Data Processing
python scripts/run_benchmarks.py \
  --prompt-file templates/code_generation.txt \
  --prompt-variables '{
    "function_name": "process_user_data",
    "function_purpose": "Process and validate user data from CSV file",
    "complexity_level": "intermediate", 
    "input_parameters": "csv_file_path (str), required_fields (List[str])",
    "return_type": "Dict[str, List[Dict]]",
    "additional_requirements": "Handle missing values, validate email addresses, group by user type",
    "include_examples": true,
    "include_tests": true,
    "style_preferences": "Type hints, PEP 8 compliance"
  }' \
  --models gpt-4,claude-3-sonnet \
  --metrics all \
  --output-dir ./results/code-generation/intermediate

# Test 3: Advanced Level - Algorithm Implementation  
python scripts/run_benchmarks.py \
  --prompt-file templates/code_generation.txt \
  --prompt-variables '{
    "function_name": "optimize_route",
    "function_purpose": "Find optimal route using traveling salesman algorithm",
    "complexity_level": "advanced",
    "input_parameters": "locations (List[Tuple[float, float]]), start_location (int)",
    "return_type": "Tuple[List[int], float]",
    "additional_requirements": "Use genetic algorithm or simulated annealing, handle large datasets efficiently, include performance metrics",
    "include_examples": true,
    "include_tests": true,
    "style_preferences": "Production-ready code, comprehensive documentation"
  }' \
  --models gpt-4,claude-3-opus \
  --timeout 120 \
  --output-dir ./results/code-generation/advanced

# Batch test different function types
declare -a functions=("sort_array:Sort an array using different algorithms" "search_tree:Binary tree search implementation" "validate_json:JSON schema validation")

for func in "${functions[@]}"; do
  IFS=':' read -r name purpose <<< "$func"
  python scripts/run_benchmarks.py \
    --prompt-file templates/code_generation.txt \
    --prompt-variables "{\"function_name\": \"$name\", \"function_purpose\": \"$purpose\", \"complexity_level\": \"intermediate\", \"include_examples\": true}" \
    --models gpt-4o-mini,claude-3-haiku \
    --output-dir "./results/code-generation/batch/$name"
done
```

#### Expected Output Analysis

**Code Quality Metrics:**
- **Syntax Correctness**: Does the code run without errors?
- **Functionality**: Does it meet the requirements?
- **Code Style**: Adherence to Python conventions
- **Documentation**: Quality of comments and docstrings
- **Error Handling**: Robustness of the implementation

### Example 3: Creative Writing with Style Parameters

**Use Case:** Generate creative content with specific style, genre, and narrative requirements.

#### Template File: `templates/creative_writing.txt`

```text
You are {model_name}, a talented {writer_type} with expertise in {genre} writing.

Writing Assignment:
- Type: {content_type}
- Genre: {genre}
- Style: {writing_style}
- Target Length: {target_length}
{?target_audience}
- Target Audience: {target_audience}
{/target_audience}

{?theme}
Central Theme: {theme}
{/theme}

{?setting}
Setting: {setting}
{/setting}

{?characters}
Characters: {characters}
{/characters}

{?mood}
Desired Mood: {mood}
{/mood}

{?constraints}
Creative Constraints:
{constraints}

{/constraints}
{?inspiration}
Inspiration/Reference: {inspiration}
{/inspiration}

Writing Requirements:
{?content_type=="story"}
- Include clear beginning, middle, and end
- Develop characters through actions and dialogue
- Use vivid, sensory descriptions
- Maintain consistent point of view
{/content_type}

{?content_type=="poem"}
- Use appropriate rhythm and meter for {genre}
- Include meaningful imagery and metaphors
- Consider sound and musicality
- Express the theme through literary devices
{/content_type}

{?content_type=="dialogue"}
- Make each character's voice distinct
- Advance plot or reveal character through conversation
- Use natural speech patterns
- Include appropriate subtext
{/content_type}

{?style_notes}
Style Notes: {style_notes}
{/style_notes}

Create an engaging {content_type} that captures the essence of {genre} while maintaining a {writing_style} style.
```

#### CLI Commands

```bash
# Test 1: Science Fiction Short Story
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing.txt \
  --prompt-variables '{
    "writer_type": "science fiction author",
    "content_type": "story",
    "genre": "hard science fiction",
    "writing_style": "descriptive and technical",
    "target_length": "500-800 words",
    "target_audience": "adult sci-fi readers",
    "theme": "AI consciousness and human connection",
    "setting": "Mars colony in 2157",
    "characters": "Dr. Sarah Chen (xenobiologist), ARIA (AI system), Colony Director Marcus Webb",
    "mood": "mysterious and thought-provoking",
    "constraints": "Must include accurate space science, avoid exposition dumps",
    "inspiration": "Similar to Andy Weir'\''s technical approach in The Martian"
  }' \
  --models gpt-4,claude-3-opus,gemini-pro \
  --parallel \
  --output-format markdown \
  --output-dir ./results/creative-writing/sci-fi

# Test 2: Poetry Generation
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing.txt \
  --prompt-variables '{
    "writer_type": "contemporary poet",
    "content_type": "poem", 
    "genre": "free verse",
    "writing_style": "lyrical and introspective",
    "target_length": "20-30 lines",
    "theme": "urban isolation and human connection",
    "setting": "modern city at night",
    "mood": "melancholic but hopeful",
    "style_notes": "Use enjambment, focus on concrete imagery, avoid rhyme scheme"
  }' \
  --models gpt-4,claude-3-sonnet \
  --metrics creativity,coherence \
  --output-dir ./results/creative-writing/poetry

# Test 3: Character Dialogue
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing.txt \
  --prompt-variables '{
    "writer_type": "screenwriter",
    "content_type": "dialogue",
    "genre": "psychological thriller",
    "writing_style": "tense and subtext-heavy",
    "target_length": "300-400 words",
    "characters": "Detective Ray Morrison (gruff, experienced), Dr. Elena Vasquez (psychiatrist, hiding secrets), Marcus Cole (suspect, manipulative)",
    "setting": "police interrogation room",
    "mood": "suspenseful and claustrophobic",
    "constraints": "Each character should have distinct speech patterns. Include stage directions. Create mounting tension through subtext.",
    "theme": "truth vs perception"
  }' \
  --models gpt-4,claude-3-sonnet,gemini-pro \
  --output-format json,markdown \
  --output-dir ./results/creative-writing/dialogue

# Batch test different genres
declare -a genres=("fantasy:magical and epic" "mystery:suspenseful and logical" "romance:emotional and character-driven" "horror:atmospheric and unsettling")

for genre_pair in "${genres[@]}"; do
  IFS=':' read -r genre style <<< "$genre_pair"
  python scripts/run_benchmarks.py \
    --prompt-file templates/creative_writing.txt \
    --prompt-variables "{\"writer_type\": \"professional author\", \"content_type\": \"story\", \"genre\": \"$genre\", \"writing_style\": \"$style\", \"target_length\": \"400-600 words\", \"theme\": \"overcoming adversity\"}" \
    --models gpt-4o-mini,claude-3-haiku \
    --limit 1 \
    --output-dir "./results/creative-writing/genres/$genre"
done

# A/B test different writing styles for same story
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing.txt \
  --prompt-variables '{
    "content_type": "story",
    "genre": "contemporary fiction", 
    "writing_style": "minimalist and sparse",
    "target_length": "400 words",
    "theme": "family reconciliation"
  }' \
  --models gpt-4 \
  --limit 3 \
  --output-dir ./results/creative-writing/style-test/minimalist

python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing.txt \
  --prompt-variables '{
    "content_type": "story",
    "genre": "contemporary fiction",
    "writing_style": "rich and descriptive", 
    "target_length": "400 words",
    "theme": "family reconciliation"
  }' \
  --models gpt-4 \
  --limit 3 \
  --output-dir ./results/creative-writing/style-test/descriptive
```

#### Expected Output Analysis

**Creative Quality Metrics:**
- **Creativity Score**: Originality and uniqueness of content
- **Style Consistency**: Adherence to requested writing style
- **Character Development**: Depth and authenticity of characters
- **Narrative Structure**: Plot coherence and pacing
- **Language Quality**: Prose quality and word choice

## 📊 Output Formats and Analysis

### JSON Output Structure

```json
{
  "execution_id": "20240805_143022_creative_writing",
  "prompt_template": "templates/creative_writing.txt",
  "template_variables": {
    "genre": "science fiction",
    "writing_style": "descriptive"
  },
  "models_requested": ["gpt-4", "claude-3-sonnet"],
  "execution_mode": "parallel",
  "responses": [
    {
      "model": "gpt-4",
      "success": true,
      "response": "In the dim corridors of New Geneva Station...",
      "duration_seconds": 3.2,
      "metrics": {
        "response_length": {"words": 542, "sentences": 34},
        "sentiment": {"score": 0.1, "label": "neutral"},
        "coherence": {"score": 0.91},
        "creativity": {"score": 0.78}
      }
    }
  ],
  "aggregated_metrics": {
    "average_length": 518,
    "creativity_variance": 0.12
  }
}
```

### CSV Output Format

```csv
execution_id,model,success,word_count,sentiment_score,coherence_score,creativity_score,duration_seconds
20240805_143022,gpt-4,true,542,0.1,0.91,0.78,3.2
20240805_143022,claude-3-sonnet,true,494,0.2,0.87,0.82,2.8
```

### Markdown Report Format

```markdown
# Creative Writing Results - Science Fiction Story

**Execution ID:** 20240805_143022_creative_writing
**Date:** 2024-08-05 14:30:22
**Template:** templates/creative_writing.txt

## Prompt Configuration
- **Genre:** Science Fiction
- **Style:** Descriptive and technical
- **Length:** 500-800 words
- **Theme:** AI consciousness and human connection

## Model Comparison

### GPT-4 Response
- **Word Count:** 542 words
- **Coherence:** 0.91 (Excellent)
- **Creativity:** 0.78 (Good)
- **Duration:** 3.2 seconds

**Sample:** "In the dim corridors of New Geneva Station, Dr. Sarah Chen's footsteps echoed against the metallic walls..."

### Claude-3-Sonnet Response  
- **Word Count:** 494 words
- **Coherence:** 0.87 (Very Good)
- **Creativity:** 0.82 (Very Good)
- **Duration:** 2.8 seconds

**Sample:** "The red dust of Mars swirled outside the observation deck as ARIA's sensors detected an anomaly in the colony's behavioral patterns..."

## Summary
Both models produced high-quality science fiction content. GPT-4 showed slightly better coherence, while Claude-3-Sonnet demonstrated higher creativity scores.
```

## 🔧 Best Practices

### Template Design
1. **Use clear variable names**: `{customer_name}` not `{name}`
2. **Provide default values**: Handle missing variables gracefully
3. **Test conditionals**: Verify all conditional paths work
4. **Document requirements**: Comment template files with required variables
5. **Version templates**: Use descriptive filenames and version control

### CLI Usage
1. **Start small**: Use `--limit 1` for testing
2. **Test sequentially first**: Remove `--parallel` when debugging
3. **Use appropriate models**: Match model capability to task complexity
4. **Monitor costs**: Start with cheaper models for development
5. **Save results**: Always specify `--output-dir` for organized results

### Performance Optimization
1. **Cache responses**: Use `--cache-responses` for repeated runs
2. **Batch similar prompts**: Group related tests together
3. **Use parallel execution**: Enable `--parallel` for multi-model tests
4. **Set appropriate timeouts**: Increase for complex creative tasks
5. **Choose optimal models**: Balance cost, speed, and quality

## 🐛 Troubleshooting

### Common Template Issues

**Variable Not Found:**
```bash
# Error: Template variable 'topic' not found
# Solution: Check JSON syntax and variable names
--prompt-variables '{"topic": "AI", "audience": "beginners"}'
```

**Conditional Not Working:**
```bash
# Error: Conditional section not appearing
# Check variable is truthy (not empty/null/false)
--prompt-variables '{"context": "Some context here", "examples": ""}'
# Result: context section appears, examples section is omitted
```

**JSON Parsing Error:**
```bash
# Error: Invalid JSON in prompt-variables
# Solution: Use single quotes around JSON, escape inner quotes
--prompt-variables '{"message": "Say \"hello\""}'
```

### CLI Debugging Commands

```bash
# Test template variable parsing
python -c "
from src.use_cases.custom_prompts import PromptTemplate
t = PromptTemplate(open('template.txt').read())
print('Required variables:', t.get_required_variables())
print('Rendered:', t.render({'var1': 'test'}))
"

# Validate JSON variables
echo '{"topic": "AI"}' | python -m json.tool

# Test single model first
python scripts/run_benchmarks.py --custom-prompt "test" --models gpt-4o-mini --limit 1 --verbose
```

---

*This CLI interface provides maximum flexibility for testing custom prompts across multiple LLM providers with sophisticated templating and evaluation capabilities.*