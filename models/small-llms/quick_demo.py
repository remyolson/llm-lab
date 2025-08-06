#!/usr/bin/env python3
"""
Quick demo to interact with small LLMs on MacBook
Supports both transformers models and Ollama models
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path


class ThinkingIndicator:
    """Shows a thinking animation while the model processes"""

    def __init__(self):
        self.running = False
        self.thread = None

    def start(self):
        """Start the thinking animation"""
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the thinking animation"""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the line
        print("\r" + " " * 20 + "\r", end="", flush=True)

    def _animate(self):
        """The animation loop"""
        frames = ["🤔 thinking", "🤔 thinking.", "🤔 thinking..", "🤔 thinking..."]
        i = 0
        while self.running:
            print(f"\r{frames[i % len(frames)]}", end="", flush=True)
            time.sleep(0.5)
            i += 1


def check_ollama_available():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_ollama_models():
    """Get list of downloaded Ollama models"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
    except:
        pass
    return []


def chat_with_ollama(model_name, prompt, thinking_indicator=None):
    """Chat with an Ollama model"""
    try:
        if thinking_indicator:
            thinking_indicator.start()

        start_time = time.time()
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )
        end_time = time.time()

        if thinking_indicator:
            thinking_indicator.stop()

        if result.returncode == 0:
            response = result.stdout.strip()
            duration = end_time - start_time
            # Rough token estimation (assuming ~4 chars per token)
            estimated_tokens = len(response) / 4
            tokens_per_second = estimated_tokens / duration if duration > 0 else 0
            return response, tokens_per_second
    except Exception as e:
        if thinking_indicator:
            thinking_indicator.stop()
    return None, 0


def main():
    print("🤖 Small LLM Quick Demo")
    print("=" * 50)

    # Collect all available models
    all_models = []
    model_names_seen = set()  # Track unique model names to avoid duplicates

    # Check for Ollama models
    if check_ollama_available():
        ollama_models = get_ollama_models()
        for model in ollama_models:
            if "llama3.2" in model:
                model_name = "llama3.2-1b"
                if model_name not in model_names_seen:
                    all_models.append((model_name, "ollama", "1B params - High quality (Ollama)"))
                    model_names_seen.add(model_name)
            elif "gpt-oss" in model:
                model_name = "gpt-oss-20b"
                if model_name not in model_names_seen:
                    all_models.append((model_name, "ollama", "20B params - Large model (Ollama)"))
                    model_names_seen.add(model_name)
            else:
                if model not in model_names_seen:
                    all_models.append((model, "ollama", "Ollama model"))
                    model_names_seen.add(model)

    # Check for local transformers models
    base_dir = Path(__file__).parent
    for item in base_dir.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            if item.name not in model_names_seen:  # Only add if not already seen
                size_info = {
                    "smollm-135m": "135M params - Fastest, basic tasks",
                    "smollm-360m": "360M params - Better quality",
                    "qwen-0.5b": "500M params - Best quality",
                    "pythia-70m": "70M params - Ultra tiny",
                    "pythia-160m": "160M params - Very small",
                    "tinyllama": "1.1B params - Larger but capable",
                }
                all_models.append(
                    (
                        item.name,
                        "transformers",
                        size_info.get(item.name, "Transformers model"),
                    )
                )
                model_names_seen.add(item.name)

    # Sort models alphabetically by name
    all_models.sort(key=lambda x: x[0])

    if not all_models:
        print("❌ No models found!")
        print("\n🚀 Quick setup:")
        print("   ./models/small-llms/quick_setup.sh")
        print("   or")
        print("   python models/small-llms/download_small_models.py --setup-ollama")
        return

    print(f"\n📋 Available models ({len(all_models)} found):")
    for i, (name, engine, description) in enumerate(all_models):
        print(f"{i+1:2d}. {name} - {description}")

    # Select model
    print(f"\nSelect a model (1-{len(all_models)}): ", end="")
    choice = input()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(all_models):
            model_name, engine, description = all_models[idx]
        else:
            print("Invalid choice, using first model")
            model_name, engine, description = all_models[0]
    except:
        print("Invalid input, using first model")
        model_name, engine, description = all_models[0]

    print(f"\n✅ Selected: {model_name} ({engine})")

    if engine == "ollama":
        print("💬 Ollama chat interface ready! (type 'quit' to exit)")
    else:
        print("📥 Loading transformers model...")
        try:
            from inference import SmallLLMInference

            model_path = base_dir / model_name
            llm = SmallLLMInference(str(model_path))
            llm.load_transformers_model()
            print("💬 Transformers chat interface ready! (type 'quit' to exit)")
        except ImportError:
            print("❌ inference.py not found. Please ensure all dependencies are installed.")
            return
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return

    print("Tip: Keep questions simple for better results!")
    print("-" * 50)

    while True:
        user_input = input("\n👤 You: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        print("\n🤖 Assistant: ", end="", flush=True)

        # Create thinking indicator
        thinking = ThinkingIndicator()

        if engine == "ollama":
            # Use Ollama
            ollama_name = model_name.replace("llama3.2-1b", "llama3.2:1b").replace("gpt-oss-20b", "gpt-oss:20b")
            response, tokens_per_sec = chat_with_ollama(ollama_name, user_input, thinking)
            if response:
                print(response)
                print(f"\n[⚡ ~{tokens_per_sec:.1f} tokens/sec]")
            else:
                print("❌ Error generating response")
        else:
            # Use transformers - adjust token limits for different model sizes
            if "pythia" in model_name:
                max_tokens = 30  # Very conservative for pythia models
            elif "135m" in model_name or "70m" in model_name:
                max_tokens = 50  # Small models
            else:
                max_tokens = 100  # Larger models

            try:
                thinking.start()
                result = llm.chat(user_input, max_new_tokens=max_tokens)
                thinking.stop()
                print(result["response"])
                print(f"\n[⚡ {result['tokens_per_second']:.1f} tokens/sec]")
            except Exception as e:
                thinking.stop()
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
