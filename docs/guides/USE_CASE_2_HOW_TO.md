# How to Compare LLM Provider Cost vs Performance

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:

- Compare costs across different LLM providers (OpenAI, Anthropic, Google, etc.)
- Analyze performance vs cost trade-offs for your specific use cases
- Generate detailed cost reports with visualizations
- Make data-driven decisions about which models to use for different tasks
- Set up budget alerts and monitoring
- Optimize your LLM spending based on actual usage patterns
- Calculate cost-per-quality metrics for informed model selection
- Create professional visualizations for stakeholder presentations

## 📋 Before You Begin

### Prerequisites
- [Initial setup](SETUP.md) completed with API keys configured
- Python 3.8+ installed
- At least 2 different provider API keys for meaningful comparison
- Basic understanding of LLM pricing models

### Time and Cost Estimates
- **Time to complete**: 15-30 minutes
- **Estimated cost**: $0.50-$2.00 (depending on models tested)
- **Skills required**: Basic command line usage

### 💰 Cost Breakdown

| Provider | Model | Input Cost | Output Cost | Typical Test Cost |
|----------|-------|------------|-------------|-------------------|
| OpenAI | GPT-4o | $5.00/1M tokens | $15.00/1M tokens | ~$0.50 |
| OpenAI | GPT-4o-mini | $0.15/1M tokens | $0.60/1M tokens | ~$0.02 |
| OpenAI | GPT-3.5-turbo | $1.00/1M tokens | $2.00/1M tokens | ~$0.05 |
| Anthropic | Claude 3.5 Sonnet | $3.00/1M tokens | $15.00/1M tokens | ~$0.40 |
| Anthropic | Claude 3.5 Haiku | $0.25/1M tokens | $1.25/1M tokens | ~$0.03 |
| Google | Gemini 1.5 Pro | $1.25/1M tokens | $5.00/1M tokens | ~$0.15 |
| Google | Gemini 1.5 Flash | $0.15/1M tokens | $0.60/1M tokens | ~$0.02 |

*Note: Prices as of late 2024. Check provider websites for current rates.*

## 🚀 Step-by-Step Guide

### Step 1: Run the Basic Cost Analysis Example

Start by running the comprehensive cost analysis tool:

```bash
cd examples/use_cases
python cost_analysis.py
```

This will:
- Initialize cost tracking with a $5 daily budget
- Test multiple providers with different prompt lengths
- Find the cheapest provider for each prompt
- Generate a detailed cost report

**Expected Output:**
```
💰 COST OPTIMIZATION DEMONSTRATION
==================================================
✓ Initialized 3 cost-optimized providers
💵 Daily budget: $5.00

📝 Test 1: What is AI?
   Prompt length: 11 characters
   💡 Cheapest provider: google
   ✓ Generated response for $0.0004
   📊 Tokens: 3 in, 125 out
```

### Step 2: Explore Real-World Scenarios

Run the scenario analysis to see costs for different use cases:

```bash
python cost_scenarios.py
```

This analyzes 5 common scenarios:
1. **Customer Service Chatbot** - High volume (500k requests/month)
2. **Code Generation Assistant** - Medium complexity tasks
3. **Content Creation Tool** - Long-form content generation
4. **Data Analysis Assistant** - Batch processing workloads
5. **Real-time Translation** - Low latency requirements

### Step 3: Calculate Cost-Per-Quality Metrics

Understand the relationship between cost and quality:

```bash
python cost_quality_metrics.py
```

This generates:
- Cost per accuracy point
- Cost per satisfaction score
- ROI calculations
- Weighted efficiency scores

**Key Metrics Explained:**
- **Efficiency Score**: Higher is better - balances quality and cost
- **Cost per Accuracy Point**: Lower is better - $ spent per 1% accuracy
- **ROI Score**: Positive means profitable based on assumed revenue

### Step 4: Create Visualizations

Generate professional charts for analysis:

```bash
python cost_visualizations.py
```

This creates:
- Cost vs Accuracy scatter plots
- Provider comparison bar charts
- Volume scaling line graphs
- Performance/cost heatmaps
- Interactive HTML dashboard

### Step 5: Implement Budget Management

Set up budget tracking in your code:

```python
from examples.use_cases.cost_analysis import CostTracker, CostOptimizedProvider

# Initialize with daily budget
tracker = CostTracker(daily_budget=10.0)

# Wrap your provider with cost tracking
optimized_provider = CostOptimizedProvider(
    provider=your_provider,
    provider_name="openai",
    model_name="gpt-4o-mini",
    cost_tracker=tracker
)

# Make cost-aware requests
result = optimized_provider.generate(
    prompt="Your prompt here",
    max_cost=0.01  # 1 cent limit per request
)

if result['success']:
    print(f"Response: {result['response']}")
    print(f"Cost: ${result['cost_estimate'].total_cost:.4f}")
else:
    print(f"Request blocked: {result['error']}")
```

## 📊 Understanding the Results

### Key Metrics Explained

1. **Cost per 1K/1M Tokens**
   - Base pricing unit for all LLM providers
   - Separate rates for input (prompt) and output (response)
   - Total cost = (input_tokens × input_rate) + (output_tokens × output_rate)

2. **Efficiency Score**
   - Combines accuracy, speed, satisfaction, and cost
   - Formula: `(weighted_quality_score) / cost_per_request`
   - Higher scores indicate better value for money

3. **Cost per Accuracy Point**
   - How much you pay for each percentage point of accuracy
   - Formula: `cost_per_request / accuracy_percentage`
   - Useful for comparing models with different accuracy levels

4. **Budget Utilization**
   - Percentage of daily/monthly budget consumed
   - Helps prevent overspending and plan capacity

### Interpreting Cost Reports

The cost analysis generates comprehensive reports with:

```
📊 COST ANALYSIS REPORT
==================================================
💵 BUDGET OVERVIEW
- Daily budget: $10.00
- Total spent today: $3.45
- Remaining budget: $6.55
- Budget utilization: 34.5%

🤖 COST BY PROVIDER
google:
  Total cost: $0.85
  Requests: 234
  Avg per request: $0.0036
```

### Output Data Formats

**JSON Format** (`cost_analysis_YYYYMMDD_HHMMSS.json`):
```json
{
  "daily_budget": 10.0,
  "costs_today": [
    {
      "provider": "google",
      "model": "gemini-1.5-flash",
      "input_tokens": 150,
      "output_tokens": 200,
      "total_cost": 0.000375,
      "timestamp": "2024-11-20T10:30:00"
    }
  ],
  "summary": {
    "daily_spend": 3.45,
    "by_provider": {...},
    "by_model": {...}
  }
}
```

## 🎨 Advanced Usage

### Custom Cost Analysis Scenarios

Create your own scenario analysis:

```python
from examples.use_cases.cost_scenarios import Scenario, analyze_scenario

# Define your use case
my_scenario = Scenario(
    name="Legal Document Review",
    description="Analyzing contracts and legal documents",
    monthly_requests=10_000,
    avg_input_tokens=2000,  # Long documents
    avg_output_tokens=500,   # Summary/analysis
    quality_requirements="Very High - Legal accuracy critical",
    latency_tolerance="High - Batch processing acceptable",
    features_needed=["High accuracy", "Citation support", "Reasoning"],
    recommended_models={
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
        "google": "gemini-1.5-pro"
    }
)

# Analyze costs
results = analyze_scenario(my_scenario)
print(f"Monthly cost range: {results['cost_analysis']['cost_range']}")
```

### Integrating with Benchmark Results

Combine performance and cost data:

```python
from src.benchmark import run_benchmark
from examples.use_cases.cost_quality_metrics import CostQualityAnalyzer

# Run benchmarks
benchmark_results = run_benchmark(
    dataset="truthful_qa",
    models=["gpt-4o-mini", "gemini-1.5-flash"]
)

# Analyze cost-quality trade-offs
analyzer = CostQualityAnalyzer()
for result in benchmark_results:
    metrics = analyzer.analyze_model(
        benchmark=result,
        cost_data=get_cost_data(result.model),
        weights={'accuracy': 0.4, 'latency': 0.3, 'cost': 0.3}
    )
    print(f"{result.model}: Efficiency Score = {metrics.weighted_efficiency_score:.2f}")
```

### Historical Cost Tracking

Implement long-term cost monitoring:

```python
import sqlite3
from datetime import datetime, timedelta

class HistoricalCostTracker:
    def __init__(self, db_path="costs.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()

    def setup_database(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS cost_history (
                timestamp DATETIME,
                provider TEXT,
                model TEXT,
                daily_cost REAL,
                request_count INTEGER
            )
        ''')

    def get_trend(self, days=30):
        """Get cost trend for last N days"""
        query = '''
            SELECT DATE(timestamp) as date,
                   SUM(daily_cost) as total_cost
            FROM cost_history
            WHERE timestamp > datetime('now', '-{} days')
            GROUP BY date
            ORDER BY date
        '''.format(days)
        return self.conn.execute(query).fetchall()
```

### Multi-Model Cost Optimization Strategy

Implement intelligent routing based on task type:

```python
class SmartRouter:
    def __init__(self, cost_tracker):
        self.cost_tracker = cost_tracker
        self.model_capabilities = {
            'gemini-1.5-flash': {'speed': 5, 'cost': 5, 'quality': 3},
            'gpt-4o-mini': {'speed': 4, 'cost': 4, 'quality': 4},
            'claude-3-5-sonnet': {'speed': 3, 'cost': 2, 'quality': 5}
        }

    def select_model(self, task_type, priority='balanced'):
        """Select optimal model based on task requirements"""
        if task_type == 'simple_qa' and priority == 'cost':
            return 'gemini-1.5-flash'
        elif task_type == 'code_generation' and priority == 'quality':
            return 'claude-3-5-sonnet'
        elif priority == 'balanced':
            return 'gpt-4o-mini'

        # Calculate scores for complex routing
        scores = {}
        for model, caps in self.model_capabilities.items():
            if priority == 'cost':
                score = caps['cost'] * 2 + caps['speed']
            elif priority == 'quality':
                score = caps['quality'] * 2 + caps['speed']
            else:  # balanced
                score = sum(caps.values()) / 3
            scores[model] = score

        return max(scores, key=scores.get)
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Error - Module Not Found
**Problem**: `ModuleNotFoundError: No module named 'matplotlib'`

**Solution**:
```bash
pip install matplotlib plotly seaborn pandas numpy
```

#### Issue 2: Incorrect Cost Calculations
**Problem**: Calculated costs don't match provider billing

**Solution**:
1. Verify token counting matches provider's method:
   ```python
   # More accurate token counting
   import tiktoken
   encoder = tiktoken.encoding_for_model("gpt-4")
   actual_tokens = len(encoder.encode(text))
   ```

2. Check for hidden costs:
   - Some providers charge for system prompts
   - Streaming may have different rates
   - Fine-tuned models often cost more

#### Issue 3: Budget Alerts Not Working
**Problem**: Requests exceed budget without warning

**Solution**:
```python
# Enable strict budget enforcement
cost_tracker = CostTracker(daily_budget=10.0)
cost_tracker.strict_mode = True  # Blocks requests over budget

# Add email alerts
def send_budget_alert(remaining_budget):
    if remaining_budget < 1.0:  # Less than $1 remaining
        send_email(
            to="admin@example.com",
            subject="LLM Budget Alert",
            body=f"Only ${remaining_budget:.2f} remaining today!"
        )
```

#### Issue 4: Visualization Errors
**Problem**: Charts not rendering or saving

**Solution**:
```python
# For headless servers
import matplotlib
matplotlib.use('Agg')  # Before importing pyplot
import matplotlib.pyplot as plt

# For Jupyter notebooks
%matplotlib inline  # or %matplotlib notebook
```

## 📈 Next Steps

After mastering cost analysis:
- Explore [Use Case 3: Test Custom Prompts](USE_CASE_3_HOW_TO.md) to optimize costs for your specific use cases
- Try [Use Case 4: Run Tests Across LLMs](USE_CASE_4_HOW_TO.md) to find the most cost-effective model for your needs
- Set up [Use Case 8: Continuous Monitoring](USE_CASE_8_HOW_TO.md) to track costs over time

## 🎯 Pro Tips

💡 **Cost Optimization Hierarchy**:
   1. Use Gemini Flash or GPT-4o-mini for development/testing
   2. Upgrade to GPT-4o or Claude Sonnet for production
   3. Reserve Claude Opus or GPT-4 for critical decisions only

💡 **Token Optimization Techniques**:
   - Use system prompts efficiently (charged on every request)
   - Implement prompt compression for long contexts
   - Set appropriate max_tokens limits
   - Use streaming for early termination

💡 **Smart Caching Strategy**:
   ```python
   from functools import lru_cache
   import hashlib

   @lru_cache(maxsize=1000)
   def cached_llm_call(prompt_hash, model):
       # Cache by prompt hash to save repeated calls
       return llm.generate(prompt)

   prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
   response = cached_llm_call(prompt_hash, model)
   ```

💡 **Budget Management Best Practices**:
   - Set daily AND hourly limits to prevent runaway costs
   - Use different API keys for dev/staging/production
   - Implement cost allocation by project or team
   - Review costs weekly and adjust budgets

💡 **Model Selection Matrix**:
   - **Simple queries**: Gemini Flash (fastest, cheapest)
   - **General purpose**: GPT-4o-mini (good balance)
   - **Complex reasoning**: Claude Sonnet or GPT-4o
   - **Creative tasks**: Claude models excel
   - **Coding tasks**: GPT-4o or Claude Sonnet

💡 **Cost Monitoring Automation**:
   ```python
   # Set up automated daily reports
   schedule.every().day.at("09:00").do(
       lambda: email_cost_report(tracker.get_cost_summary())
   )
   ```

## 📚 Additional Resources

### Official Pricing Pages
- [OpenAI Pricing](https://openai.com/pricing) - GPT-4, GPT-3.5 rates
- [Anthropic Pricing](https://www.anthropic.com/api#pricing) - Claude model costs
- [Google AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) - Gemini pricing
- [Azure OpenAI Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) - Enterprise options

### Cost Optimization Guides
- [Token Optimization Strategies](https://platform.openai.com/docs/guides/optimizing) - OpenAI's official guide
- [Prompt Engineering for Cost](https://www.anthropic.com/index/prompting) - Anthropic's recommendations
- [LLM Cost Calculator](https://llm-price-calculator.com/) - Interactive cost comparison tool

### Related LLM Lab Documentation
- [Use Case 1: Run Benchmarks](USE_CASE_1_HOW_TO.md) - Measure performance alongside cost
- [Use Case 3: Custom Prompts](USE_CASE_3_HOW_TO.md) - Test cost-optimized prompts
- [Use Case 8: Monitoring](USE_CASE_8_HOW_TO.md) - Set up continuous cost tracking

### Example Code Repository
All code examples from this guide are available at:
```
examples/use_cases/
├── cost_analysis.py          # Basic cost tracking
├── cost_scenarios.py         # Real-world scenarios
├── cost_quality_metrics.py   # Quality vs cost analysis
└── cost_visualizations.py    # Visualization examples
```

---

*Last updated: November 2024. For the latest pricing, always check provider websites.*
