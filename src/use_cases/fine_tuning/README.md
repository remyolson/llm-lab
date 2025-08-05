# Fine-tuning Framework for LLM Lab

This module provides a comprehensive framework for fine-tuning language models using parameter-efficient methods like LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA).

## Features

- **LoRA Fine-tuning**: Efficient training with low-rank adapters
- **QLoRA Support**: 4-bit and 8-bit quantized training for larger models
- **Multiple Data Formats**: Support for JSONL, CSV, and Parquet datasets
- **Flexible Configuration**: Comprehensive config management with validation
- **Training Monitoring**: Integration with TensorBoard and Weights & Biases
- **Memory Efficient**: Gradient checkpointing and mixed precision training
- **Benchmark Integration**: Automatic evaluation after training

## Installation

### Required Dependencies

```bash
pip install transformers>=4.35.0 peft>=0.7.0 datasets>=2.14.0 accelerate>=0.24.0
```

### Optional Dependencies

```bash
# For QLoRA (quantized training)
pip install bitsandbytes>=0.41.0

# For monitoring
pip install tensorboard>=2.14.0 wandb>=0.15.0

# For memory profiling
pip install psutil>=5.9.0
```

## Quick Start

### Basic LoRA Fine-tuning

```python
from src.use_cases.fine_tuning import TrainingConfig, LoRAConfig, DataConfig, LoRATrainer

# Configure training
config = TrainingConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    learning_rate=1e-4,
    num_epochs=3,
    batch_size=4,
    lora_config=LoRAConfig(
        r=8,  # LoRA rank
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    ),
    data_config=DataConfig(
        dataset_path="path/to/dataset.jsonl",
        max_seq_length=512,
        validation_split=0.1
    )
)

# Create trainer
trainer = LoRATrainer(config.to_dict())

# Start training
trainer.train(config.data_config.dataset_path)
```

### QLoRA Training (4-bit)

```python
from src.use_cases.fine_tuning import get_template

# Use pre-configured template
config = get_template("qlora_efficient")
config.model_name = "meta-llama/Llama-2-13b-hf"
config.data_config.dataset_path = "path/to/dataset.jsonl"

# QLoRA automatically handles quantization
trainer = LoRATrainer(config.to_dict())
trainer.train(config.data_config.dataset_path)
```

## Dataset Formats

### Instruction Format (Recommended)

```json
{"instruction": "What is machine learning?", "response": "Machine learning is..."}
{"instruction": "Explain neural networks", "response": "Neural networks are..."}
```

### Prompt-Completion Format

```json
{"prompt": "The capital of France is", "completion": " Paris."}
{"prompt": "Machine learning is used for", "completion": " pattern recognition..."}
```

### Plain Text Format

```json
{"text": "This is a complete text example for training."}
{"text": "Another training example with full context."}
```

## Dataset Processing

```python
from src.use_cases.fine_tuning import DatasetProcessor

# Create processor
processor = DatasetProcessor(
    max_length=512,
    validation_split=0.1,
    test_split=0.05
)

# Load and prepare dataset
dataset = processor.prepare_for_training(
    file_path="data.jsonl",
    tokenizer=tokenizer,
    input_format="instruction",
    clean=True,
    validate=True
)
```

## Configuration Options

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 5e-5 | Learning rate for optimization |
| `num_epochs` | 3 | Number of training epochs |
| `batch_size` | 4 | Training batch size |
| `max_seq_length` | 512 | Maximum sequence length |
| `lora_r` | 8 | LoRA rank (lower = more efficient) |
| `lora_alpha` | 16 | LoRA scaling parameter |
| `use_qlora` | False | Enable 4/8-bit quantization |
| `gradient_checkpointing` | False | Trade compute for memory |

### Pre-defined Templates

```python
# Standard LoRA
config = get_template("lora_standard")

# Memory-efficient QLoRA
config = get_template("qlora_efficient")

# Instruction tuning
config = get_template("instruction_tuning")
```

## Monitoring Training

### TensorBoard

```python
config = TrainingConfig(
    logging_backend="tensorboard",
    logging_dir="./logs",
    logging_steps=10
)

# View logs
# tensorboard --logdir ./logs/tensorboard
```

### Weights & Biases

```python
config = TrainingConfig(
    logging_backend="wandb",
    wandb_project="my-fine-tuning",
    wandb_name="llama2-lora",
    wandb_tags=["lora", "instruction"]
)
```

## Memory Optimization

### Estimate Memory Requirements

```python
# Before training
memory_estimate = trainer.estimate_memory_usage()
print(f"Estimated GPU memory: {memory_estimate['total_estimated_gb']:.1f} GB")
```

### Memory-Saving Options

```python
config = TrainingConfig(
    # Use QLoRA for 75% memory reduction
    use_qlora=True,
    bits=4,
    
    # Enable gradient checkpointing
    gradient_checkpointing=True,
    
    # Use mixed precision
    fp16=True,  # or bf16=True for newer GPUs
    
    # Reduce batch size and use accumulation
    batch_size=1,
    gradient_accumulation_steps=16
)
```

## Advanced Usage

### Custom Target Modules

```python
# For different model architectures
lora_config = LoRAConfig(
    # GPT models
    target_modules=["c_attn", "c_proj"],
    
    # T5 models
    # target_modules=["q", "v"],
    
    # Custom regex (PEFT feature)
    # target_modules=[".*attention.*"],
)
```

### Multi-GPU Training

```python
# Automatically handled by the framework
# Just set CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
```

### Checkpoint Management

```python
# Resume from checkpoint
trainer.load_checkpoint("outputs/checkpoints/checkpoint_epoch_2.pt")

# Merge LoRA weights with base model
trainer.merge_and_save("path/to/merged_model")
```

## Integration with LLM Lab

### Benchmark After Training

```python
config = TrainingConfig(
    # ... other settings ...
    run_benchmark_after=True,
    benchmark_dataset="truthfulness",
    benchmark_tasks=["fact_checking"]
)
```

### Use Fine-tuned Model in Benchmarks

```python
# After training, use the model
from src.providers import get_provider_for_model

# Load fine-tuned model
provider = get_provider_for_model("path/to/merged_model")

# Run benchmarks
from benchmarks.runners import MultiModelRunner
runner = MultiModelRunner(models=["fine_tuned_model"])
results = runner.run()
```

## Troubleshooting

### Out of Memory

1. Enable QLoRA: `use_qlora=True, bits=4`
2. Reduce batch size: `batch_size=1`
3. Enable gradient checkpointing: `gradient_checkpointing=True`
4. Reduce sequence length: `max_seq_length=256`
5. Use smaller LoRA rank: `r=4`

### Slow Training

1. Check GPU utilization: `nvidia-smi`
2. Increase batch size if memory allows
3. Use mixed precision: `fp16=True`
4. Ensure data loading isn't bottleneck: `num_workers=4`

### Poor Results

1. Increase LoRA rank: `r=16` or `r=32`
2. Adjust learning rate: try `2e-4` or `5e-5`
3. Train for more epochs
4. Check dataset quality and formatting
5. Try different target modules

## Examples

See `examples/use_cases/fine_tuning_demo.py` for:
- Basic LoRA fine-tuning
- QLoRA with quantization
- Dataset processing examples
- Monitoring setup
- Benchmark integration

## Best Practices

1. **Start Small**: Test with a small dataset first
2. **Monitor Training**: Use TensorBoard/W&B to track progress
3. **Save Checkpoints**: Enable regular checkpoint saving
4. **Validate Data**: Always validate dataset before training
5. **Benchmark**: Compare before/after model performance

## Contributing

To add new features:
1. Extend `BaseTrainer` for new training methods
2. Add new configuration options to `TrainingConfig`
3. Implement additional dataset formats in `DatasetProcessor`
4. Add tests and documentation

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Library](https://github.com/huggingface/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)