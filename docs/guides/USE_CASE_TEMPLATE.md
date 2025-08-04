# Use Case X: [Title of Use Case]

*One-line description of what this use case enables.*

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:
- [Specific capability 1] across [multiple scenarios/datasets/models]
- [Test/compare/analyze] [specific aspects] with [practical benefits]
- **Automatically save detailed results** to [format] for analysis
- Generate comprehensive reports with [specific metrics]
- Analyze which [models/approaches/settings] perform best for [specific tasks]
- Understand the cost implications and optimization strategies
- Make data-driven decisions for [specific domain/use case]

## 📋 Before You Begin

- Complete all [Prerequisites](./PREREQUISITES.md)
- Ensure you have [specific requirement for this use case]
- Time required: ~[X] minutes
- Estimated cost: $[X.XX] per run

### 💰 Cost Breakdown
Running [full operation description] with different [models/providers/settings]:

**💡 Pro Tip:** Use `--limit [N]` for testing to reduce costs by 98% (approximately $0.001-$0.10 per test run)

- **[Provider 1]**:
  - `[model-1]`: ~$[X.XX] ([details/cost-effectiveness notes])
  - `[model-2]`: ~$[X.XX]
  - **Free tier eligible**: [Yes/details] ([specific limits])

- **[Provider 2]**:
  - `[model-1]`: ~$[X.XX]
  - `[model-2]`: ~$[X.XX]  
  - **Free tier eligible**: [Yes/details]

- **[Provider 3]**:
  - `[model-1]`: ~$[X.XX]
  - `[model-2]`: ~$[X.XX]
  - **Free tier eligible**: [Limited/details]

*Note: Costs are estimates based on [date] pricing. Actual costs depend on response lengths. Single model tests cost proportionally less.*

## 📊 Available [Options/Datasets/Approaches]

Choose the right [option/dataset/approach] based on what [capability/outcome] you want to [test/achieve]:

| Option | Size/Scope | What It [Tests/Does] | Best For | Example |
|--------|------------|---------------------|----------|---------|
| **[option-1]** | [details] | [Capability description] | [Use case scenarios] | "[Example question/command]" |
| **[option-2]** | [details] | [Capability description] | [Use case scenarios] | "[Example question/command]" |
| **[option-3]** | [details] | [Capability description] | [Use case scenarios] | "[Example question/command]" |
| **[option-4]** | [details] | [Capability description] | [Use case scenarios] | "[Example question/command]" |
| **[option-5]** | [details] | [Capability description] | [Use case scenarios] | "[Example question/command]" |

### 🎯 **[Selection] Guide:**

- **🔍 For [use case A]:** Start with `[option-1]` and `[option-2]`
- **🧮 For [use case B]:** Use `[option-3]`  
- **🎓 For [use case C]:** Choose `[option-4]`
- **🌍 For [use case D]:** Test with `[option-5]`
- **📊 For comprehensive assessment:** Run all [options] with `--limit [N]` first

## 🚀 Step-by-Step Guide

### Step 1: [Verification/Setup Title]

First, check [prerequisite or setup requirement]:

```bash
# Check environment or verify setup
command_to_verify --status
```

### Step 2: [Initial Test Title]

Start with [simple/single] [operation] to ensure everything works:

```bash
# Quick test with [simple option] (recommended for first run)
command --option [simple-choice] --limit [N]

# Try different [option types/approaches]
command --option [choice-1] --limit [N]        # [Description]
command --option [choice-2] --limit [N]      # [Description]
command --option [choice-3] --limit [N]       # [Description]
command --option [choice-4] --limit [N]  # [Description]

# Full [operation] test (remove --limit for complete assessment)
command --option [choice-1]

# Use [advanced feature] ([better accuracy/performance], default since vX.X)
command --option [choice-1] --limit [N] --advanced-flag [value]
```

**What Happens:**
- [Process description] in your chosen [dataset/option] ([N] with `--limit [N]`, or the full [dataset/scope] without limit)
- Results are automatically saved to `results/[location]/YYYY-MM/[files]`
- Progress is displayed in real-time in the console
- Different [options] test different [capabilities] (see [reference] table above)

**Expected Output:**
```
[Show what successful output looks like with specific examples]
```

### Step 3: [Comparison/Multi-option Title]

Now run comparisons across different [providers/options/settings]:

```bash
# Quick comparison on [option-1] (recommended first comparison)
command \
  --[param] [choice-A],[choice-B],[choice-C] \
  --option [option-1] \
  --limit [N]

# Test [models/settings] on [different capability]
command \
  --[param] [choice-A],[choice-B],[choice-C] \
  --option [option-2] \
  --limit [N]

# Compare on [another capability]  
command \
  --[param] [choice-A],[choice-B],[choice-C] \
  --option [option-3] \
  --limit [N]

# Full comparison on any [option] (more expensive)
command \
  --[param] [choice-A],[choice-B],[choice-C] \
  --option [option-1]
```

### Step 4: [Parallel/Advanced Execution Title]

For faster execution when [testing multiple scenarios]:

```bash
# Quick parallel test on [option-1] (recommended first)
command \
  --[param] [choice-A],[choice-B],[choice-C] \
  --option [option-1] \
  --limit [N] \
  --parallel

# Parallel testing on [different capability]
command \
  --[param] [choice-A],[choice-B],[choice-C] \
  --option [option-2] \
  --limit [N] \
  --parallel

# Full parallel [operations] (much faster than sequential, but may hit rate limits)
command \
  --[param] [choice-A],[choice-B],[choice-C] \
  --option [option-3] \
  --parallel
```

### Step 5: [Comprehensive Multi-Option Testing]

For a complete [assessment/analysis] across different [capabilities/scenarios]:

```bash
# Quick comprehensive test across all [options] (recommended)
for option in [option-1] [option-2] [option-3] [option-4] [option-5]; do
  echo "Testing $option..."
  command --[param] [single-choice] --option $option --limit [N]
done

# Compare [multiple choices] across all [capabilities]
for option in [option-1] [option-2] [option-3] [option-4] [option-5]; do
  echo "Comparing [choices] on $option..."
  command \
    --[param] [choice-A],[choice-B] \
    --option $option \
    --limit [N] \
    --parallel
done

# Single command for quick assessment (if you have all [requirements])
command --all-[param] --option [option-1] --limit [small-N]
command --all-[param] --option [option-2] --limit [small-N]
command --all-[param] --option [option-3] --limit [small-N]
```

**💡 Pro Tip:** Start with `--limit [small-N]` across all [options] to get a broad sense of [performance/results], then deep-dive into specific [options] that matter most for your use case.

### Step 6: [Results Analysis Title]

[Continue pattern for remaining steps...]

## 📊 Understanding the Results

### Key Metrics Explained

1. **[Primary Metric]**: [Percentage/score meaning and calculation method]
2. **[Secondary Metric]**: [Count or measure and significance]  
3. **[Performance Metric]**: [Time/efficiency measure and importance]
4. **[Quality Metric]**: [Accuracy/reliability measure]

### Interpreting Results by [Option/Category]

Different [options/datasets/approaches] reveal different [strengths/capabilities]:

**📊 Typical Performance Patterns:**
- **[option-1]**: X-Y% ([what this indicates about capability])
- **[option-2]**: X-Y% ([what this measures and means])
- **[option-3]**: X-Y% ([capability description])
- **[option-4]**: X-Y% ([what this tests])
- **[option-5]**: X-Y% ([performance indicator meaning])

**🎯 What Good Performance Looks Like:**
- **High [option-1] + high [option-2]**: Strong [capability combination]
- **High [option-3]**: Excellent for [specific use case type]
- **High [option-4]**: Good for [application area]
- **High [option-5]**: Great for [scenario type]
- **Consistent across all**: [Overall capability assessment]

### Example Results

```
📊 [Operation] Results Summary
==================================================
[Context]: [details]
[Scope]: [tested items]

📈 [Comparison Type]:
--------------------------------------------------------------------------------
[Item]                         [Category]     [Score]      [Success]    [Failed]    [Time] 
--------------------------------------------------------------------------------
[item-1]                       [category]     87.50%     14         2         45.23
[item-2]                       [category]     81.25%     13         3         23.45  
[item-3]                       [category]     75.00%     12         4         34.56
--------------------------------------------------------------------------------
```

### [Results Organization/Location]

Results are saved as [format] files in the `results/` directory, organized by [criteria]:

```bash
# List results for all [categories]
ls -la results/*/YYYY-*/

# View results for specific [categories]
ls -la results/[category-1]/YYYY-*/   # [Description]
ls -la results/[category-2]/YYYY-*/   # [Description]  
ls -la results/[category-3]/YYYY-*/   # [Description]
ls -la results/[category-4]/YYYY-*/   # [Description]
ls -la results/[category-5]/YYYY-*/   # [Description]

# View a specific result file
cat results/[category]/YYYY-MM/[filename_pattern]
```

## 🎨 Advanced Usage

### Quick Testing with Limited [Scope/Dataset]

For development and testing purposes, use the `--limit` flag to process only the first N [items/questions/entries]:

```bash
# Test with just [small-N] [items] (very quick) 
command --[param] [choice] --option [option-1] --limit [small-N]

# Quick test across different [option] types
command --[param] [choice] --option [option-1] --limit [small-N]      # [Type A]
command --[param] [choice] --option [option-2] --limit [small-N]        # [Type B]
command --[param] [choice] --option [option-3] --limit [small-N]       # [Type C]

# Test multiple [choices] with [N] [items] each
command --[param] [choice-A],[choice-B] --option [option-4] --limit [N]

# Quick parallel test across [options]
command --[param] [choice-A],[choice-B] --option [option-1] --limit [N] --parallel
command --[param] [choice-A],[choice-B] --option [option-2] --limit [N] --parallel
```

**Benefits of using `--limit`:**
- ⚡ Much faster execution (seconds vs minutes)
- 💰 Minimal [API costs/resource usage] for testing
- 🔧 Perfect for development and debugging
- ✅ Validates your setup works correctly

### [Advanced Evaluation/Processing Methods]

Choose how [results/responses] are [evaluated/processed] for [accuracy/quality]:

```bash
# [Method 1] ([original/strict] method)
command --[param] [choice] --option [option-1] --limit [N] --[method-flag] [method-1]

# [Method 2] ([improved/flexible] method) 
command --[param] [choice] --option [option-1] --limit [N] --[method-flag] [method-2]

# [Method 3] ([combined/advanced] approach, default)
command --[param] [choice] --option [option-1] --limit [N] --[method-flag] [method-3]
```

**[Method] Comparison:**
- **`[method-1]`**: [Description of approach] - [pros/cons] [emoji]
- **`[method-2]`**: [Description of approach] - [pros/cons] [emoji]  
- **`[method-3]`**: [Description of approach] - [pros/cons] [emoji] **(Recommended)**

**Results Comparison Example:**
- Old method: [poor performance description]
- New methods: [improved performance description]

### Test All Available [Options/Choices]

```bash
# [Operation] every [choice] you have [access to] on different [options]
command --all-[param] --option [option-1] --limit [N] --parallel
command --all-[param] --option [option-2] --limit [N] --parallel  
command --all-[param] --option [option-3] --limit [N] --parallel

# Full comparison across all [choices] (expensive but comprehensive)
command --all-[param] --option [option-1] --parallel
```

### Custom Output Directory

```bash
# Save results to a specific location
command \
  --[param] [choice] \
  --option [option-1] \
  --output-dir ./my-[operation]-results
```

### Disable [Output Format] (Console Only)

```bash
# Just see console output without saving files
command \
  --[param] [choice] \
  --option [option-1] \
  --no-[format]
```

## 🎯 Pro Tips

💡 **Start Small**: Test with one [choice] and `--limit [N]` before running expensive [operations]

💡 **Use --limit for Development**: Always use `--limit [N]` when testing setup, debugging, or trying new [options/approaches]

💡 **Use Parallel Wisely**: Parallel execution is faster but may hit rate limits

💡 **Monitor Costs**: [Expensive options] are significantly more expensive

💡 **Save Results**: Keep [output format] outputs for historical comparison

💡 **Choose [Choices] Wisely**:
  - For speed: `[fast-choice]`
  - For cost: `[cheap-choice-1]` or `[cheap-choice-2]`
  - For quality: `[quality-choice-1]` or `[quality-choice-2]`

💡 **Cost Optimization**: Start with cheaper [choices] ([list]) for development and testing. Only use expensive [choices] ([list]) for final comparisons or when quality is critical.

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue: [Common Problem 1]
**Solution**: 
```bash
# Fix command or steps
fix_command --option
```

#### Issue: [Common Problem 2]
**Solution**: [Step-by-step fix]

### Debugging Commands

```bash
# Useful debugging commands
debug_command_1
debug_command_2
```

## 📈 Next Steps

Now that you've mastered [primary capability]:

1. **Analyze [Costs/Performance]**: Run the [analysis type] example to understand [specific aspect]
   ```bash
   python examples/use_cases/[analysis_script].py
   ```

2. **Compare [Advanced Metrics]**: Use the [analysis type] notebook
   ```bash
   python examples/notebooks/[analysis_notebook].py
   ```

3. **Create Custom [Extensions]**: Use our [script/template] as a template to create your own [custom implementations]

4. **Automate [Operations]**: Set up scheduled runs with cron or GitHub Actions

### Related Use Cases
- [Use Case Y: Title](./USE_CASE_Y_HOW_TO.md) - Dive deeper into [specific aspect]
- [Use Case Z: Title](./USE_CASE_Z_HOW_TO.md) - [Different focus/approach]
- [Use Case W: Title](./USE_CASE_W_HOW_TO.md) - Comprehensive [advanced topic]

## 📚 Understanding [The Core Components/Datasets/Features]

Each [component/option/dataset] tests different aspects of [capability/performance]:

### [Component 1] ([size/scope])
[Purpose description]:
- [Key characteristic 1]
- Example: "[Example]"

### [Component 2] ([size/scope])  
[Purpose description]:
- [Key characteristic 1]
- Example: "[Example]"

### [Component 3] ([size/scope])
[Purpose description]:
- [Key characteristic 1]
- Example: "[Example]"

### [Component 4] ([size/scope])
[Purpose description]:
- [Key characteristic 1]
- Example: "[Example]"

### [Component 5] ([size/scope])
[Purpose description]:
- [Key characteristic 1]
- Example: "[Example]"

## 🔄 Continuous Improvement

This [system/approach] provides a foundation for:
- **Multi-[capability] assessment**: [Test/analyze] [items] across [N] different [skill areas/aspects]
- **Regular [performance/quality] tracking** across diverse [benchmarks/scenarios]
- **A/B testing new [options/approaches]** as they're released with comprehensive evaluation
- **Domain-specific analysis**: Choose [options/components] that match your use case
- **Creating automated quality assurance** for [application type] applications

## 📚 Additional Resources

- **Provider Documentation**: 
  - [Provider 1 Guide](../providers/[provider1].md)
  - [Provider 2 Guide](../providers/[provider2].md)
  - [Provider 3 Guide](../providers/[provider3].md)
- **Examples**: [Relevant Examples](../../examples/README.md)
- **Troubleshooting**: [Full Troubleshooting Guide](../TROUBLESHOOTING.md)

## 💭 Feedback

Help us improve this guide:
- Found an error? [Open an issue](https://github.com/yourusername/[repo-name]/issues/new)
- Have suggestions? See our [Contribution Guide](../../CONTRIBUTING.md)

---

*Last updated: [Month Year]*