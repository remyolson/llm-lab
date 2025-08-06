"""
Demo script for fine-tuning models with LoRA/QLoRA.

This example demonstrates how to use the fine-tuning framework
to train models with parameter-efficient methods.
"""

import argparse
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from use_cases.fine_tuning import (
    DataConfig,
    DatasetProcessor,
    LoRAConfig,
    LoRATrainer,
    TrainingConfig,
)
from use_cases.fine_tuning.config.training_config import get_template


def create_sample_dataset(output_path: str):
    """Create a sample dataset for demonstration."""
    samples = [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris.",
        },
        {
            "instruction": "Explain photosynthesis in simple terms.",
            "response": "Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to make their own food (glucose) and release oxygen.",
        },
        {
            "instruction": "Write a haiku about programming.",
            "response": "Code flows like water\nBugs hide in syntax shadows\nDebugger finds peace",
        },
        {"instruction": "What is 15 + 27?", "response": "15 + 27 = 42"},
        {
            "instruction": "Translate 'Hello, how are you?' to Spanish.",
            "response": "Hola, ¿cómo estás?",
        },
    ]

    # Duplicate samples to create a larger dataset
    samples = samples * 20  # 100 samples total

    # Save as JSONL
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created sample dataset with {len(samples)} examples at: {output_path}")


def demo_basic_fine_tuning():
    """Demo: Basic LoRA fine-tuning."""
    print("\n=== Basic LoRA Fine-tuning Demo ===\n")

    # Create sample dataset
    dataset_path = "sample_instruction_data.jsonl"
    create_sample_dataset(dataset_path)

    # Create configuration
    config = TrainingConfig(
        model_name="gpt2",  # Using small model for demo
        learning_rate=1e-4,
        num_epochs=2,
        batch_size=2,
        output_dir="./demo_outputs/basic_lora",
        lora_config=LoRAConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["c_attn"],  # GPT-2 attention modules
        ),
        data_config=DataConfig(
            dataset_path=dataset_path,
            max_seq_length=128,
            validation_split=0.2,
            input_format="instruction",
        ),
        logging_backend="tensorboard",
        show_progress=True,
    )

    # Validate configuration
    config.validate()

    # Save configuration
    config.save("demo_config.json")
    print("Configuration saved to: demo_config.json")

    # Initialize trainer
    print("\nInitializing LoRA trainer...")
    trainer = LoRATrainer(config.to_dict())

    # Estimate memory usage
    memory_estimate = trainer.estimate_memory_usage()
    print("\nEstimated memory usage:")
    for key, value in memory_estimate.items():
        print(f"  {key}: {value:.2f} GB")

    print("\nNote: To actually run training, you would call:")
    print("  trainer.train(dataset_path)")
    print("\nThis demo stops here to avoid downloading large models.")

    # Clean up
    os.remove(dataset_path)


def demo_qlora_training():
    """Demo: QLoRA training with 4-bit quantization."""
    print("\n=== QLoRA Training Demo ===\n")

    # Use pre-defined template
    config = get_template("qlora_efficient")

    # Customize for demo
    config.model_name = "gpt2"
    config.data_config.dataset_path = "sample_data.jsonl"
    config.num_epochs = 1
    config.output_dir = "./demo_outputs/qlora"

    print("QLoRA Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Quantization: {config.bits}-bit")
    print(f"  LoRA rank: {config.lora_config.r}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  FP16 training: {config.fp16}")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")

    print("\nQLoRA enables training larger models with limited GPU memory!")


def demo_dataset_processing():
    """Demo: Dataset processing pipeline."""
    print("\n=== Dataset Processing Demo ===\n")

    # Create processor
    processor = DatasetProcessor(max_length=128, validation_split=0.2, test_split=0.1)

    # Create sample data in different formats

    # 1. Instruction format
    instruction_data = [
        {"instruction": "What is AI?", "response": "AI is artificial intelligence."},
        {"instruction": "Define ML", "response": "ML stands for machine learning."},
    ]

    # 2. Prompt-completion format
    prompt_data = [
        {"prompt": "The weather today is", "completion": " sunny and warm."},
        {"prompt": "Python is a", "completion": " high-level programming language."},
    ]

    # 3. Text format
    text_data = [
        {"text": "This is a complete text example for training."},
        {"text": "Another example of training text."},
    ]

    # Save as different formats
    with open("instruction_data.jsonl", "w") as f:
        for item in instruction_data:
            f.write(json.dumps(item) + "\n")

    pd_data = pd.DataFrame(prompt_data)
    pd_data.to_csv("prompt_data.csv", index=False)

    # Process datasets
    print("Processing instruction dataset...")
    dataset1 = processor.load_dataset_from_file("instruction_data.jsonl")
    dataset1 = processor.format_dataset(dataset1, "instruction")
    stats1 = processor.validate_dataset(dataset1)
    print(f"  Examples: {stats1['num_examples']}")
    print(f"  Text stats: {stats1.get('text_stats', {})}")

    print("\nProcessing prompt-completion dataset...")
    dataset2 = processor.load_dataset_from_file("prompt_data.csv")
    dataset2 = processor.format_dataset(dataset2, "prompt_completion")

    # Split dataset
    print("\nSplitting dataset...")
    splits = processor.split_dataset(dataset1)
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} examples")

    # Clean up
    os.remove("instruction_data.jsonl")
    os.remove("prompt_data.csv")


def demo_nemotron_dataset():
    """Demo: Using the NVIDIA Nemotron Post-Training Dataset."""
    print("\n=== Nemotron Dataset Demo ===\n")

    from use_cases.fine_tuning.datasets.dataset_registry import DatasetRegistry

    # Get dataset info
    nemotron_info = DatasetRegistry.get_dataset("nemotron-post-training")
    print(f"Dataset: {nemotron_info.name}")
    print(f"Description: {nemotron_info.description}")
    print(f"Size: {nemotron_info.size}")
    print(f"Splits: {', '.join(nemotron_info.splits)}")
    print(f"License: {nemotron_info.license}")

    # Create processor
    processor = DatasetProcessor(max_length=2048)

    print("\nLoading Nemotron 'math' split (streaming mode)...")
    math_dataset = processor.load_nemotron_split(
        split="math",
        streaming=True,
        max_samples=100,  # Limit for demo
    )

    print("Preview of math examples:")
    for i, example in enumerate(math_dataset.take(3)):
        print(f"\nExample {i + 1}:")
        for key in list(example.keys())[:3]:  # Show first 3 fields
            value = (
                str(example[key])[:100] + "..."
                if len(str(example[key])) > 100
                else str(example[key])
            )
            print(f"  {key}: {value}")

    print("\nAvailable splits for different use cases:")
    print("  - chat: General conversation and instruction-following")
    print("  - code: Programming and code generation")
    print("  - math: Mathematical reasoning")
    print("  - stem: Science and technical Q&A")
    print("  - tool_calling: Function calling examples")

    print("\nTo use with fine-tuning:")
    print("  config.data_config.dataset_name = 'nemotron-post-training'")
    print("  config.data_config.dataset_split = 'chat'  # or any other split")


def demo_training_monitoring():
    """Demo: Training monitoring with TensorBoard/W&B."""
    print("\n=== Training Monitoring Demo ===\n")

    config = TrainingConfig(
        model_name="gpt2",
        learning_rate=1e-4,
        num_epochs=3,
        batch_size=4,
        output_dir="./demo_outputs/monitored",
        logging_backend="both",  # Both TensorBoard and W&B
        logging_steps=10,
        wandb_project="llm-lab-demo",
        wandb_name="lora-fine-tuning",
        wandb_tags=["demo", "lora", "gpt2"],
        lora_config=LoRAConfig(r=8),
        data_config=DataConfig(dataset_path="dummy.jsonl"),
    )

    print("Monitoring Configuration:")
    print(f"  TensorBoard logs: {config.logging_dir}/tensorboard")
    print(f"  W&B project: {config.wandb_project}")
    print(f"  Logging interval: every {config.logging_steps} steps")

    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir {config.logging_dir}/tensorboard")

    print("\nTo use W&B:")
    print("  1. Sign up at https://wandb.ai")
    print("  2. Run: wandb login")
    print("  3. Training metrics will be automatically logged")


def demo_benchmark_integration():
    """Demo: Integration with LLM Lab benchmarks."""
    print("\n=== Benchmark Integration Demo ===\n")

    config = TrainingConfig(
        model_name="gpt2",
        num_epochs=2,
        output_dir="./demo_outputs/benchmark",
        # Benchmark settings
        run_benchmark_after=True,
        benchmark_dataset="truthfulness",
        benchmark_tasks=["fact_checking", "common_sense"],
        lora_config=LoRAConfig(r=8),
        data_config=DataConfig(dataset_path="dummy.jsonl"),
    )

    print("After fine-tuning, the model will be benchmarked on:")
    print(f"  Dataset: {config.benchmark_dataset}")
    print(f"  Tasks: {', '.join(config.benchmark_tasks or [])}")

    print("\nThis allows you to measure the impact of fine-tuning on model performance!")


def main():
    """Run all demos."""
    parser = argparse.ArgumentParser(description="Fine-tuning framework demo")
    parser.add_argument(
        "--demo",
        choices=["all", "basic", "qlora", "dataset", "nemotron", "monitoring", "benchmark"],
        default="all",
        help="Which demo to run",
    )

    args = parser.parse_args()

    print("LLM Lab - Fine-tuning Framework Demo")
    print("=" * 50)

    demos = {
        "basic": demo_basic_fine_tuning,
        "qlora": demo_qlora_training,
        "dataset": demo_dataset_processing,
        "nemotron": demo_nemotron_dataset,
        "monitoring": demo_training_monitoring,
        "benchmark": demo_benchmark_integration,
    }

    if args.demo == "all":
        for demo_func in demos.values():
            demo_func()
    else:
        demos[args.demo]()

    print("\n" + "=" * 50)
    print("Demo complete!")
    print("\nNext steps:")
    print("1. Install required dependencies:")
    print("   pip install transformers peft datasets accelerate")
    print("2. For QLoRA: pip install bitsandbytes")
    print("3. For monitoring: pip install tensorboard wandb")
    print("4. Prepare your dataset in supported format")
    print("5. Configure training parameters")
    print("6. Run training with trainer.train()")


if __name__ == "__main__":
    # Add pandas import for demo
    try:
        import pandas as pd
    except ImportError:
        print("Please install pandas: pip install pandas")
        sys.exit(1)

    main()
