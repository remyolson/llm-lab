#!/usr/bin/env python3
"""
Simple inference script for small LLMs on MacBook Pro
Supports both transformers and GGUF models
"""

import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Check if MPS (Metal) is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Using device: {device}")


class SmallLLMInference:
    def __init__(self, model_path, model_type="transformers"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.tokenizer = None

    def load_transformers_model(self):
        """Load a transformers format model"""
        print(f"Loading model from {self.model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Fix tokenizer issues for models that lack proper pad tokens
        if self.tokenizer.pad_token is None:
            # Use eos_token as pad_token for most models, but handle special cases
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # For pythia models, ensure we have proper special tokens
        if "pythia" in self.model_path.lower():
            if self.tokenizer.bos_token is None:
                self.tokenizer.bos_token = self.tokenizer.eos_token

        # Load model with appropriate settings for Mac
        # Use float32 for better numerical stability with small models
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # Changed from float16 for stability
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Move to device if needed
        if device == "mps":
            self.model = self.model.to(device)

        print("✅ Model loaded successfully!")

    def generate_transformers(self, prompt, max_new_tokens=100, temperature=0.7):
        """Generate text using transformers model"""
        # Tokenize input with proper max_length to avoid warnings
        max_input_length = 512  # Reasonable limit for most small models
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)

        # Move to device
        if device == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        start_time = time.time()

        # Use safer generation parameters to avoid numerical instability
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # For very small models, use more conservative sampling to avoid numerical issues
        model_size = self.model.num_parameters() if hasattr(self.model, "num_parameters") else 0
        is_very_small = model_size < 200_000_000  # Less than 200M parameters

        # Add stopping criteria to prevent loops
        stop_sequences = ["\n", "Q:", "Human:", "Assistant:", "<|", "User:"]

        if is_very_small or temperature <= 0.1:
            # Use greedy decoding for very small models or low temperature
            generation_kwargs.update(
                {
                    "do_sample": False,
                    "num_beams": 1,
                    "repetition_penalty": 1.2,  # Prevent repetition loops
                    "no_repeat_ngram_size": 2,  # Prevent 2-gram repetitions
                }
            )
        else:
            # Use conservative sampling parameters
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": max(0.1, min(temperature, 1.0)),  # Clamp temperature
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.3,  # Higher penalty for sampling
                    "no_repeat_ngram_size": 3,  # Prevent 3-gram repetitions
                }
            )

        try:
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
        except Exception as e:
            # Fallback to greedy decoding if sampling fails
            print(f"⚠️  Sampling failed, falling back to greedy decoding: {e}")
            fallback_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            with torch.no_grad():
                outputs = self.model.generate(**fallback_kwargs)

        # Decode output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt) :].strip()

        # Clean up response and stop at appropriate sequences
        stop_sequences = ["\nQ:", "\nHuman:", "\nAssistant:", "\nUser:", "\n\n", "<|"]
        for stop_seq in stop_sequences:
            if stop_seq in response:
                response = response.split(stop_seq)[0].strip()
                break

        # For very short or repetitive responses, try to extract meaningful content
        if len(response.strip()) < 5 or response.count("A:") > 2:
            # If response is just repetitive "A:" or too short, provide a fallback
            response = response.replace("A:", "").strip()
            if not response:
                response = "[Model could not generate a clear response]"

        elapsed = time.time() - start_time
        tokens_generated = outputs[0].shape[0] - inputs["input_ids"].shape[1]
        tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

        return {
            "response": response,
            "time": elapsed,
            "tokens_per_second": tokens_per_sec,
            "tokens_generated": tokens_generated,
        }

    def chat(self, message, system_prompt=None, max_new_tokens=200):
        """Chat interface for different model types"""
        # Format prompt based on model type
        if "qwen" in self.model_path.lower():
            # Qwen instruction format
            prompt = f"<|im_start|>system\n{system_prompt or 'You are a helpful assistant.'}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        elif "smollm" in self.model_path.lower():
            # SmolLM instruction format
            prompt = f"<|im_start|>system\n{system_prompt or 'You are a helpful assistant.'}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        elif "tinyllama" in self.model_path.lower():
            # TinyLlama instruction format
            prompt = f"<|system|>\n{system_prompt or 'You are a helpful assistant.'}</s>\n<|user|>\n{message}</s>\n<|assistant|>\n"
        elif "pythia" in self.model_path.lower():
            # Pythia is a base model, not instruction-tuned
            # Use simple completion format that works better for base models
            if "what" in message.lower() and any(
                op in message for op in ["+", "-", "*", "/", "plus", "minus", "times", "divided"]
            ):
                # For math questions, try a more direct approach
                prompt = f"Calculate: {message}\nAnswer:"
            else:
                # For general questions, use completion style
                prompt = f"Question: {message}\nAnswer:"
        else:
            # Generic instruction format for other models
            prompt = f"{system_prompt or ''}\n\nHuman: {message}\n\nAssistant:"

        # Adjust max_new_tokens for very small models to prevent repetition
        if "pythia" in self.model_path.lower():
            # Very small models should generate shorter responses
            max_new_tokens = min(max_new_tokens, 50)
        elif any(size in self.model_path.lower() for size in ["70m", "135m"]):
            # Other very small models
            max_new_tokens = min(max_new_tokens, 75)

        return self.generate_transformers(prompt, max_new_tokens)


def benchmark_model(model_path):
    """Run simple benchmarks on a model"""
    print(f"\n📊 Benchmarking {model_path}")
    print("=" * 50)

    # Initialize model
    llm = SmallLLMInference(model_path)
    llm.load_transformers_model()

    # Test prompts
    test_prompts = [
        {"name": "Simple Q&A", "prompt": "What is the capital of France?", "max_tokens": 50},
        {"name": "Code Generation", "prompt": "Write a Python function to calculate factorial:", "max_tokens": 100},
        {"name": "Creative Writing", "prompt": "Once upon a time in a small village,", "max_tokens": 150},
        {"name": "Math Problem", "prompt": "What is 25 + 37?", "max_tokens": 30},
    ]

    results = []

    for test in test_prompts:
        print(f"\n🔍 Test: {test['name']}")
        print(f"Prompt: {test['prompt']}")

        result = llm.chat(test["prompt"], max_new_tokens=test["max_tokens"])

        print(f"Response: {result['response']}")
        print(f"⏱️  Time: {result['time']:.2f}s")
        print(f"⚡ Speed: {result['tokens_per_second']:.1f} tokens/sec")

        results.append(
            {
                "test": test["name"],
                "prompt": test["prompt"],
                "response": result["response"],
                "time": result["time"],
                "tokens_per_second": result["tokens_per_second"],
            }
        )

    return results


def interactive_chat(model_path):
    """Interactive chat interface"""
    print(f"\n💬 Starting interactive chat with {os.path.basename(model_path)}")
    print("Type 'quit' to exit\n")

    # Initialize model
    llm = SmallLLMInference(model_path)
    llm.load_transformers_model()

    while True:
        user_input = input("\n👤 You: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        print("\n🤖 Assistant: ", end="", flush=True)
        result = llm.chat(user_input, max_new_tokens=200)
        print(result["response"])
        print(f"\n[{result['tokens_per_second']:.1f} tokens/sec]")


def main():
    """Main function to demonstrate usage"""
    print("🤖 Small LLM Inference Demo")
    print("=" * 50)

    # Check for available models
    model_dir = "models/small-llms"
    available_models = []

    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            path = os.path.join(model_dir, item)
            if os.path.isdir(path) and any(f.endswith(".json") for f in os.listdir(path)):
                available_models.append(path)

    if not available_models:
        print("❌ No models found! Run download_small_models.py first.")
        return

    print(f"\n📦 Found {len(available_models)} models:")
    for i, model in enumerate(available_models):
        print(f"  {i+1}. {os.path.basename(model)}")

    # Demo with the first available model
    if available_models:
        model_path = available_models[0]
        print(f"\n🎯 Using model: {os.path.basename(model_path)}")

        # Run benchmark
        benchmark_model(model_path)

        # Optional: Start interactive chat
        # interactive_chat(model_path)


if __name__ == "__main__":
    main()
