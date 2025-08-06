"""
Demo script for using local models with LLM Lab.

This example shows how to:
1. Download a local model
2. Initialize the LocalModelProvider
3. Run benchmarks with local models
4. Compare local vs API-based model performance
"""

import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from use_cases.local_models.download_helper import ModelDownloader
from use_cases.local_models.model_configs import format_prompt
from use_cases.local_models.provider import LocalModelProvider


def demo_model_download():
    """Demo: Download a model."""
    print("=== Model Download Demo ===\n")

    downloader = ModelDownloader()

    # List available models
    print("Available models:")
    from use_cases.local_models.model_configs import MODEL_REGISTRY

    for name, info in MODEL_REGISTRY.items():
        downloaded = "✓" if downloader.is_model_downloaded(name) else " "
        print(f"  [{downloaded}] {name}: {info['name']} ({info['size']})")

    print("\nNote: To download a model, run:")
    print("  python -m src.use_cases.local_models.download_helper --model phi-2")
    print()


def demo_local_inference():
    """Demo: Run inference with a local model."""
    print("=== Local Model Inference Demo ===\n")

    # Check if we have a model downloaded
    downloader = ModelDownloader()

    # Find first available model
    available_model = None
    for model_name in ["phi-2", "mistral-7b", "llama-2-7b"]:
        if downloader.is_model_downloaded(model_name):
            available_model = model_name
            break

    if not available_model:
        print("No models downloaded yet. Please download a model first:")
        print("  python -m src.use_cases.local_models.download_helper --model phi-2")
        return

    print(f"Using model: {available_model}")
    model_path = downloader.get_model_path(available_model)

    try:
        # Initialize provider
        print(f"Loading model from: {model_path}")
        provider = LocalModelProvider(
            model_name=available_model,
            model_path=str(model_path),
            n_ctx=2048,
            temperature=0.7,
            max_tokens=100,
        )

        # Get model info
        info = provider.get_model_info()
        print(f"Model loaded: {info['model_name']}")
        print(f"Context length: {info['context_length']}")
        print(f"Hardware: {info['hardware']['gpu_type']} acceleration")
        print()

        # Test prompts
        test_prompts = [
            "What is 2 + 2?",
            "Write a haiku about programming.",
            "Explain quantum computing in simple terms.",
        ]

        print("Running test prompts...\n")
        for prompt in test_prompts:
            # Format prompt for the model
            formatted_prompt = format_prompt(prompt, available_model)

            print(f"Prompt: {prompt}")
            print("Response: ", end="", flush=True)

            start_time = time.time()
            response = provider.generate(formatted_prompt, stream=False)
            end_time = time.time()

            print(response.strip())
            print(f"Time: {end_time - start_time:.2f}s")
            print("-" * 50)

        # Show memory usage
        memory = provider.get_memory_usage()
        print(f"\nMemory usage: {memory['ram_used_mb']:.1f} MB RAM")
        if memory["vram_used_mb"] > 0:
            print(f"GPU memory: {memory['vram_used_mb']:.1f} MB VRAM")

    except ImportError:
        print("Error: llama-cpp-python is not installed.")
        print("Install it with: pip install llama-cpp-python")
    except Exception as e:
        print(f"Error: {e!s}")


def demo_streaming():
    """Demo: Streaming responses."""
    print("\n=== Streaming Response Demo ===\n")

    downloader = ModelDownloader()
    available_model = None

    for model_name in ["phi-2", "mistral-7b"]:
        if downloader.is_model_downloaded(model_name):
            available_model = model_name
            break

    if not available_model:
        print("No models available for streaming demo.")
        return

    try:
        model_path = downloader.get_model_path(available_model)
        provider = LocalModelProvider(model_name=available_model, model_path=str(model_path))

        prompt = "Tell me a short story about a robot learning to paint."
        formatted_prompt = format_prompt(prompt, available_model)

        print(f"Prompt: {prompt}")
        print("Streaming response:\n")

        # Note: Real streaming would yield tokens as they're generated
        # This is a simplified version
        response = provider.generate(formatted_prompt, stream=True, max_tokens=150)
        print(response)

    except Exception as e:
        print(f"Error: {e!s}")


def demo_benchmark_comparison():
    """Demo: Compare local vs API models."""
    print("\n=== Benchmark Comparison Demo ===\n")

    # Sample benchmark results
    print("Sample benchmark comparison (local vs API models):\n")

    results = {
        "Local Models": {
            "phi-2 (2.7B)": {
                "accuracy": 0.72,
                "latency_ms": 150,
                "cost_per_1k": 0.0,  # Free!
                "tokens_per_sec": 45,
            },
            "mistral-7b": {
                "accuracy": 0.84,
                "latency_ms": 280,
                "cost_per_1k": 0.0,
                "tokens_per_sec": 25,
            },
        },
        "API Models": {
            "gpt-4o-mini": {
                "accuracy": 0.92,
                "latency_ms": 800,
                "cost_per_1k": 0.15,
                "tokens_per_sec": None,
            },
            "claude-3-haiku": {
                "accuracy": 0.89,
                "latency_ms": 600,
                "cost_per_1k": 0.25,
                "tokens_per_sec": None,
            },
        },
    }

    # Display comparison
    for category, models in results.items():
        print(f"{category}:")
        for model, metrics in models.items():
            print(f"  {model}:")
            print(f"    Accuracy: {metrics['accuracy']:.1%}")
            print(f"    Latency: {metrics['latency_ms']}ms")
            print(f"    Cost: ${metrics['cost_per_1k']}/1k tokens")
            if metrics["tokens_per_sec"]:
                print(f"    Speed: {metrics['tokens_per_sec']} tokens/sec")
        print()

    print("Key insights:")
    print("- Local models have zero API costs")
    print("- Lower latency for local inference")
    print("- API models generally higher accuracy")
    print("- Local models offer privacy and offline capability")


def main():
    """Run all demos."""
    print("LLM Lab - Local Model Integration Demo")
    print("=" * 50)
    print()

    # Run demos
    demo_model_download()
    demo_local_inference()
    demo_streaming()
    demo_benchmark_comparison()

    print("\n" + "=" * 50)
    print("Demo complete!")
    print("\nNext steps:")
    print("1. Download more models for testing")
    print("2. Run full benchmarks with local models")
    print("3. Compare performance across different quantization levels")
    print("4. Test on your specific use cases")


if __name__ == "__main__":
    main()
