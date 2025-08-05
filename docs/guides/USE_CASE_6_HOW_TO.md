# Use Case 6: Fine-tuning Local LLMs

*Customize and train local language models on your specific data using efficient methods like LoRA and QLoRA for optimal performance and resource utilization.*

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:
- **Fine-tune local models** using your own datasets with LoRA/QLoRA techniques
- **Prepare training data** in proper formats for instruction tuning and domain adaptation
- **Configure efficient training** with minimal GPU memory requirements using quantization
- **Monitor training progress** with comprehensive metrics and validation curves
- **Evaluate fine-tuned models** against base models and benchmark performance improvements
- **Deploy custom models** for specialized tasks with production-ready inference
- **Iterate and improve** models through systematic hyperparameter optimization

## 📋 Before You Begin

- Complete [Use Case 5: Local Model Development](./USE_CASE_5_HOW_TO.md) first
- Ensure you have a training dataset prepared (500+ examples recommended)
- Time required: ~4-8 hours (including training time)
- Estimated ongoing cost: $0.00 (local compute only, electricity costs minimal)

### 💰 Training Requirements and Costs

Fine-tuning costs depend on model size and training duration, but remain cost-effective compared to cloud alternatives:

**💡 Pro Tip:** Start with 7B parameter models using QLoRA - they provide excellent results while being trainable on consumer hardware

- **LoRA Fine-tuning (Recommended)**:
  - **7B models**: 12-16GB VRAM, 2-4 hours training, ~$0.50 electricity
  - **13B models**: 20-24GB VRAM, 4-8 hours training, ~$1.00 electricity
  - **Free ongoing inference**: $0 per fine-tuned model use vs $0.01-0.03 per API call

- **QLoRA Fine-tuning (Memory Efficient)**:
  - **7B models**: 8-12GB VRAM, 3-6 hours training, ~$0.75 electricity
  - **13B models**: 12-16GB VRAM, 6-12 hours training, ~$1.50 electricity
  - **4-bit quantization**: 50-70% memory reduction with minimal quality loss

- **Full Fine-tuning (Advanced)**:
  - **7B models**: 40-80GB VRAM, 8-24 hours training, ~$5-15 electricity
  - **Requires**: Multiple high-end GPUs or A100/H100 access
  - **Best quality**: Maximum model adaptation for specialized domains

*Note: Training costs include only electricity (~$0.12/kWh). Hardware amortization varies by usage frequency.*

## 📊 Fine-tuning Methods Comparison

Choose the right approach based on your hardware and quality requirements:

| Method | VRAM Need | Training Speed | Final Quality | Best For |
|--------|-----------|----------------|---------------|----------|
| **LoRA** | 12-16GB | Fast (2-4h) | Excellent | General fine-tuning, adapter-based |
| **QLoRA** | 8-12GB | Medium (3-6h) | Very Good | Resource-constrained setups |
| **Full FT** | 40-80GB | Slow (8-24h) | Best | Domain-specific, maximum adaptation |
| **PEFT** | 10-14GB | Fast (2-3h) | Good | Quick prototyping, multiple tasks |

### 🎯 **Method Selection Guide:**

- **🔍 For learning/experimentation:** Start with LoRA on 7B models (balanced performance/resources)
- **🧮 For production applications:** Use QLoRA for efficiency or full fine-tuning for maximum quality
- **🎓 For specialized domains:** Choose full fine-tuning with domain-specific datasets
- **🌍 For multiple tasks:** Use PEFT with task-specific adapters
- **📊 For comparative studies:** Train multiple approaches and benchmark results

## 🚀 Step-by-Step Guide

### Step 1: Install Fine-tuning Dependencies

Set up the required libraries for efficient fine-tuning:

```bash
# Install core fine-tuning libraries
pip install transformers datasets accelerate peft bitsandbytes

# For advanced training features
pip install wandb tensorboard deepspeed  # Optional: experiment tracking

# For dataset processing
pip install pandas scikit-learn nltk

# Verify GPU setup for training
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

### Step 2: Prepare Your Training Dataset

Format your data correctly for instruction tuning:

```bash
# Create a sample dataset for instruction tuning
cat > training_data.jsonl << 'EOF'
{"instruction": "Explain machine learning in simple terms", "input": "", "output": "Machine learning is a way for computers to learn patterns from data without being explicitly programmed. It's like teaching a computer to recognize patterns by showing it many examples."}
{"instruction": "Write a Python function to calculate fibonacci numbers", "input": "n=10", "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"}
{"instruction": "Summarize the key benefits of renewable energy", "input": "", "output": "Key benefits include: reduced carbon emissions, energy independence, job creation, long-term cost savings, and sustainable resource utilization."}
EOF

# Convert to training format
python examples/use_cases/fine_tuning_demo.py \
  --prepare-dataset \
  --input training_data.jsonl \
  --output formatted_training_data.json \
  --format instruction

# Validate dataset format
python examples/use_cases/fine_tuning_demo.py \
  --validate-dataset formatted_training_data.json
```

**Expected Output:**
```
📊 Dataset Analysis:
✓ Total samples: 3
✓ Average instruction length: 47 tokens
✓ Average output length: 89 tokens
✓ Format validation: PASSED
✓ Ready for training
```

### Step 3: Configure LoRA Training Parameters

Set up efficient fine-tuning with LoRA (Low-Rank Adaptation):

```bash
# Create LoRA training configuration
cat > lora_config.yaml << 'EOF'
# LoRA Configuration for 7B Model Fine-tuning
model_name: "mistral-7b"  # or "llama-2-7b", "phi-2"
base_model_path: "~/.cache/llm-lab/models/mistral-7b.gguf"

# LoRA Parameters
lora_config:
  r: 16                    # LoRA rank (4, 8, 16, 32)
  lora_alpha: 32          # LoRA scaling parameter
  target_modules:         # Which layers to apply LoRA
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
  lora_dropout: 0.1       # Dropout for LoRA layers
  bias: "none"            # Bias training strategy

# Training Parameters
training:
  num_epochs: 3
  batch_size: 4           # Adjust based on VRAM
  gradient_accumulation: 4 # Effective batch size = 16
  learning_rate: 2e-4
  warmup_steps: 100
  max_length: 512         # Context length for training
  
# Optimization
optimization:
  gradient_checkpointing: true
  fp16: true              # Mixed precision training
  dataloader_num_workers: 4

# Validation
validation:
  eval_steps: 50
  eval_strategy: "steps"
  save_steps: 100
  validation_split: 0.1
EOF

# Verify configuration
python examples/use_cases/fine_tuning_demo.py \
  --validate-config lora_config.yaml
```

### Step 4: Start Your First Fine-tuning Job

Launch the training process with monitoring:

```bash
# Start LoRA fine-tuning (recommended first approach)
python examples/use_cases/fine_tuning_demo.py \
  --config lora_config.yaml \
  --dataset formatted_training_data.json \
  --output ./fine_tuned_mistral_7b \
  --method lora \
  --monitor

# Alternative: QLoRA for lower memory usage
python examples/use_cases/fine_tuning_demo.py \
  --config lora_config.yaml \
  --dataset formatted_training_data.json \
  --output ./fine_tuned_mistral_7b_qlora \
  --method qlora \
  --quantization 4bit \
  --monitor

# Monitor training progress in real-time
tail -f ./fine_tuned_mistral_7b/training.log
```

**Expected Training Output:**
```
🚀 Starting LoRA Fine-tuning
📋 Model: mistral-7b (7B parameters)
🎯 Method: LoRA (r=16, alpha=32)
📊 Dataset: 3 samples (90% train, 10% validation)
⚡ Hardware: CUDA (RTX 4090, 24GB VRAM)

Epoch 1/3:
  Step 1/3: loss=2.847, lr=1e-4, time=45s
  Step 2/3: loss=2.234, lr=1.5e-4, time=43s
  Step 3/3: loss=1.892, lr=2e-4, time=41s
  Validation: loss=2.156, perplexity=8.64

Epoch 2/3:
  Step 1/3: loss=1.654, lr=2e-4, time=40s
  Step 2/3: loss=1.423, lr=2e-4, time=39s
  Step 3/3: loss=1.298, lr=2e-4, time=38s
  Validation: loss=1.534, perplexity=4.63

✓ Training completed in 3.2 hours
✓ Final validation loss: 1.342
✓ Model saved to: ./fine_tuned_mistral_7b/
```

### Step 5: Test Your Fine-tuned Model

Evaluate the fine-tuned model performance:

```bash
# Test fine-tuned model with new prompts
python examples/use_cases/fine_tuning_demo.py \
  --test-model ./fine_tuned_mistral_7b \
  --prompt "Explain deep learning in simple terms" \
  --compare-base

# Run comprehensive evaluation
python examples/use_cases/fine_tuning_demo.py \
  --evaluate ./fine_tuned_mistral_7b \
  --test-dataset evaluation_prompts.jsonl \
  --metrics accuracy,coherence,relevance

# Compare against base model
python scripts/run_benchmarks.py \
  --providers local \
  --models mistral-7b,./fine_tuned_mistral_7b \
  --custom-prompt "Write a technical explanation of neural networks" \
  --limit 3
```

### Step 6: Monitor and Analyze Training Results

Review training metrics and model performance:

```bash
# View training curves
python examples/use_cases/fine_tuning_demo.py \
  --plot-training ./fine_tuned_mistral_7b/training_log.json

# Generate training report
python examples/use_cases/fine_tuning_demo.py \
  --training-report ./fine_tuned_mistral_7b \
  --output training_analysis.html

# Check model file sizes and structure
ls -la ./fine_tuned_mistral_7b/
# Expected structure:
# adapter_config.json     # LoRA configuration
# adapter_model.bin       # LoRA weights (~100MB)
# training_log.json       # Training metrics
# tokenizer_config.json   # Tokenizer settings
# README.md              # Model documentation
```

**💡 Pro Tip:** Use `--plot-training` to visualize loss curves and identify overfitting early. Good training shows steadily decreasing loss without validation loss increasing.

## 📊 Understanding the Results

### Key Metrics Explained

1. **Training Loss**: Measures how well the model fits training data (lower is better, target: <2.0)
2. **Validation Loss**: Measures generalization to unseen data (should track training loss)
3. **Perplexity**: Exponential of loss, measures prediction uncertainty (lower is better)
4. **Learning Rate**: Optimization step size (typically 1e-4 to 5e-4 for LoRA)
5. **Gradient Norm**: Indicates training stability (should be stable, not exploding)

### Interpreting Training Progress

Different patterns reveal training quality and potential issues:

**📊 Healthy Training Patterns:**
- **Decreasing loss**: Both training and validation loss should decrease together
- **Stable gradients**: Gradient norms should remain consistent (1-10 range)
- **Reasonable perplexity**: Final perplexity should be 2-10 for good models
- **No overfitting**: Validation loss shouldn't diverge significantly from training loss

**🚨 Warning Signs:**
- **Loss plateaus early**: May need higher learning rate or more data
- **Validation loss increases**: Overfitting - reduce epochs or add regularization
- **Exploding gradients**: Gradient norm >100 - reduce learning rate
- **No improvement**: Loss stays constant - check data format and model setup

### Example Training Analysis

```
📊 Fine-tuning Results Summary
==================================================
Model: mistral-7b → fine_tuned_mistral_7b
Method: LoRA (r=16, alpha=32)
Dataset: 1,000 instruction-response pairs

📈 Training Metrics:
--------------------------------------------------------------------------------
Metric                         Initial    Final      Improvement    Status
--------------------------------------------------------------------------------
Training Loss                  3.245      1.342      -58.6%         ✓ Good
Validation Loss               3.189      1.456      -54.3%         ✓ Good
Perplexity                    24.2       4.28       -82.3%         ✓ Excellent
Gradient Norm                 2.45       2.12       Stable         ✓ Good
Train Time                    -          3.2h       -              ✓ Efficient
--------------------------------------------------------------------------------

💡 Analysis: Model shows strong adaptation with good generalization
🎯 Quality: Fine-tuned model outperforms base model on domain-specific tasks
⚡ Efficiency: LoRA achieved 95% of full fine-tuning quality with 15% of memory
```

### Model Output Comparison

```bash
# Compare base vs fine-tuned responses
Prompt: "Explain machine learning model overfitting"

# Base Model Response:
"Overfitting occurs when a model learns the training data too well..."

# Fine-tuned Model Response:
"Overfitting in machine learning happens when your model memorizes the training data instead of learning generalizable patterns. Think of it like studying for a test by memorizing specific questions rather than understanding the underlying concepts. The model performs excellently on training data but poorly on new, unseen data.

Key signs of overfitting:
- High training accuracy, low validation accuracy
- Model performs worse on real-world data
- Training and validation loss curves diverge

Solutions include:
- Use more training data
- Apply regularization techniques
- Reduce model complexity
- Implement early stopping"

💡 Analysis: Fine-tuned model provides more structured, detailed, and practical explanations
```

## 🎨 Advanced Usage

### Multi-Task Fine-tuning

Train a single model on multiple specialized tasks:

```bash
# Prepare multi-task dataset
cat > multi_task_data.jsonl << 'EOF'
{"task": "coding", "instruction": "Write a Python function to reverse a string", "output": "def reverse_string(s):\n    return s[::-1]"}
{"task": "writing", "instruction": "Write a professional email subject line", "output": "RE: Project Timeline Update - Action Required"}
{"task": "analysis", "instruction": "Analyze this data trend", "input": "Sales increased 15% Q1", "output": "The 15% Q1 sales increase indicates strong market performance..."}
EOF

# Train with task-specific adapters
python examples/use_cases/fine_tuning_demo.py \
  --multi-task multi_task_data.jsonl \
  --method multi_lora \
  --output ./multi_task_model \
  --task-adapters coding,writing,analysis
```

### Hyperparameter Optimization

Systematically find optimal training parameters:

```bash
# Define hyperparameter search space
cat > hp_search.yaml << 'EOF'
search_space:
  learning_rate: [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
  lora_r: [4, 8, 16, 32]
  lora_alpha: [16, 32, 64]
  batch_size: [2, 4, 8]
  
search_strategy: "grid"  # or "random" or "bayesian"
max_trials: 20
metric: "validation_loss"
EOF

# Run hyperparameter optimization
python examples/use_cases/fine_tuning_demo.py \
  --hyperparameter-search hp_search.yaml \
  --dataset formatted_training_data.json \
  --output ./hp_optimization_results

# Analyze best parameters
python examples/use_cases/fine_tuning_demo.py \
  --analyze-hp-results ./hp_optimization_results
```

### Continuous Learning Pipeline

Set up automated model improvement:

```python
# continuous_learning.py
from src.use_cases.fine_tuning import ContinuousLearningPipeline

pipeline = ContinuousLearningPipeline(
    base_model="mistral-7b",
    data_source="./new_training_data/",
    update_frequency="weekly",
    evaluation_dataset="./evaluation_set.jsonl"
)

# Monitor for new data and retrain automatically
pipeline.start_monitoring()
```

```bash
# Run continuous learning
python continuous_learning.py --config continuous_config.yaml
```

### Model Merging and Ensembling

Combine multiple fine-tuned models for better performance:

```bash
# Merge LoRA adapters
python examples/use_cases/fine_tuning_demo.py \
  --merge-adapters \
  --models ./fine_tuned_model_1,./fine_tuned_model_2,./fine_tuned_model_3 \
  --weights 0.4,0.3,0.3 \
  --output ./merged_model

# Create ensemble inference
python examples/use_cases/fine_tuning_demo.py \
  --ensemble-inference \
  --models ./model_1,./model_2,./model_3 \
  --prompt "Test ensemble performance" \
  --aggregation weighted_voting
```

## 🎯 Pro Tips

💡 **Start with Quality Data**: 500 high-quality examples beat 5,000 mediocre ones for fine-tuning

💡 **Use LoRA First**: Begin with LoRA (r=16, alpha=32) - it's efficient and effective for most tasks

💡 **Monitor Validation Loss**: Stop training when validation loss stops improving (early stopping)

💡 **Balanced Datasets**: Ensure your training data covers all scenarios your model will encounter

💡 **Incremental Training**: Start with small models (7B) before moving to larger ones (13B+)

💡 **Save Checkpoints**: Configure automatic checkpointing every 100 steps to avoid losing progress

💡 **Temperature Tuning**: Fine-tuned models often work better with lower temperature (0.3-0.7)

💡 **Quantization-Aware**: Train with QLoRA if you plan to deploy with quantization

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue: Out of Memory (OOM) Errors
**Solution**: Reduce memory usage with multiple strategies
```bash
# Reduce batch size and enable gradient checkpointing
python examples/use_cases/fine_tuning_demo.py \
  --config lora_config.yaml \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --gradient-checkpointing \
  --fp16

# Use QLoRA for even lower memory usage
python examples/use_cases/fine_tuning_demo.py \
  --method qlora \
  --quantization 4bit \
  --batch-size 2
```

#### Issue: Training Loss Not Decreasing
**Solution**: Adjust learning rate and check data format
```bash
# Try different learning rates
for lr in 1e-5 2e-5 5e-5 1e-4; do
  echo "Testing LR: $lr"
  python examples/use_cases/fine_tuning_demo.py \
    --quick-test \
    --learning-rate $lr \
    --epochs 1
done

# Validate data format
python examples/use_cases/fine_tuning_demo.py \
  --validate-dataset formatted_training_data.json \
  --verbose
```

#### Issue: Overfitting (Validation Loss Increases)
**Solution**: Apply regularization and reduce training
```bash
# Add dropout and reduce epochs
python examples/use_cases/fine_tuning_demo.py \
  --lora-dropout 0.2 \
  --epochs 2 \
  --early-stopping \
  --patience 3
```

#### Issue: Slow Training Speed
**Solution**: Optimize training configuration
```bash
# Enable all speed optimizations
python examples/use_cases/fine_tuning_demo.py \
  --fp16 \
  --gradient-checkpointing \
  --dataloader-workers 8 \
  --pin-memory \
  --compile-model  # PyTorch 2.0+
```

#### Issue: Poor Fine-tuned Model Quality
**Solution**: Review data and hyperparameters
```bash
# Analyze training data quality
python examples/use_cases/fine_tuning_demo.py \
  --analyze-dataset formatted_training_data.json \
  --show-examples 10

# Try full fine-tuning for comparison
python examples/use_cases/fine_tuning_demo.py \
  --method full \
  --learning-rate 5e-6 \
  --epochs 1  # Short test
```

### Debugging Commands

```bash
# Test model loading and basic inference
python -c "
from src.use_cases.fine_tuning import FineTunedModelLoader
loader = FineTunedModelLoader('./fine_tuned_mistral_7b')
response = loader.generate('Test prompt')
print(f'Model working: {len(response) > 0}')
"

# Monitor GPU usage during training
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv'

# Check training log for errors
grep -E "(ERROR|WARN|Exception)" ./fine_tuned_mistral_7b/training.log

# Validate saved model structure
python examples/use_cases/fine_tuning_demo.py \
  --validate-model ./fine_tuned_mistral_7b
```

## 📈 Next Steps

Now that you've mastered local model fine-tuning:

1. **Deploy Your Model**: Set up production inference with your fine-tuned model
   ```bash
   python examples/use_cases/fine_tuned_inference_server.py \
     --model ./fine_tuned_mistral_7b \
     --port 8000
   ```

2. **Benchmark Performance**: Compare against other models and baselines
   ```bash
   python scripts/run_benchmarks.py \
     --providers local \
     --models ./fine_tuned_mistral_7b,mistral-7b \
     --datasets your_evaluation_set
   ```

3. **Scale Training**: Move to larger models or distributed training
   ```bash
   python examples/use_cases/fine_tuning_demo.py \
     --model llama-2-13b \
     --method qlora \
     --multi-gpu
   ```

4. **Create Model Variants**: Fine-tune for different tasks or domains

### Related Use Cases
- [Use Case 5: Local Model Development](./USE_CASE_5_HOW_TO.md) - Set up local model infrastructure
- [Use Case 4: Cross-LLM Testing](./USE_CASE_4_HOW_TO.md) - Include fine-tuned models in testing
- [Use Case 7: Alignment Research](./USE_CASE_7_HOW_TO.md) - Apply safety and alignment techniques

## 📚 Understanding Fine-tuning Methods

Each fine-tuning approach has specific use cases and trade-offs:

### LoRA (Low-Rank Adaptation)
Efficient adapter-based fine-tuning:
- **Best for**: General fine-tuning, instruction following, style adaptation
- **Strengths**: Memory efficient, fast training, modular (swappable adapters)
- **Example**: "Adapt a coding model to write documentation in your company's style"

### QLoRA (Quantized LoRA)
4-bit quantized training for extreme efficiency:
- **Best for**: Resource-constrained environments, large model fine-tuning
- **Strengths**: Minimal memory usage, works on consumer GPUs, good quality retention
- **Example**: "Fine-tune a 13B model on a single RTX 3090"

### Full Fine-tuning
Complete model parameter updating:
- **Best for**: Domain-specific applications, maximum model adaptation
- **Strengths**: Highest quality results, complete model specialization
- **Example**: "Create a medical diagnosis model from a general language model"

### PEFT (Parameter-Efficient Fine-Tuning) 
General framework for efficient training:
- **Best for**: Multi-task scenarios, rapid prototyping, research
- **Strengths**: Supports multiple adapter types, task switching
- **Example**: "Train one model for coding, writing, and analysis tasks"

## 🔄 Continuous Improvement

This fine-tuning framework enables:
- **Custom model development**: Specialized models for your exact use cases
- **Iterative improvement**: Continuous learning from new data
- **Cost-effective training**: Local fine-tuning vs expensive cloud training
- **Model versioning**: Track and compare different model versions
- **Domain adaptation**: Adapt general models to specific fields or styles

## 📚 Additional Resources

- **Research Papers**: 
  - [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
  - [QLoRA: Efficient Fine-tuning](https://arxiv.org/abs/2305.14314)
  - [PEFT: Parameter-Efficient Fine-tuning](https://arxiv.org/abs/2110.04366)
- **Implementation Guides**: [Fine-tuning Examples](../../examples/use_cases/fine_tuning_demo.py)
- **Hardware Optimization**: [Training Optimization Guide](../TRAINING_OPTIMIZATION.md)

## 💭 Feedback

Help us improve this guide:
- Found an error? [Open an issue](https://github.com/remyolson/llm-lab/issues/new)
- Have suggestions? See our [Contribution Guide](../../CONTRIBUTING.md)

---

*Last updated: January 2025*