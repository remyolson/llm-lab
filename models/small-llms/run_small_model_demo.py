#!/usr/bin/env python3
"""
Small Models Demo Runner
Uses the locally downloaded small models that work well on CPU.

Available models:
- SmolLM-135M: Very fast, basic capabilities
- SmolLM-360M: Fast, better capabilities
- Pythia-70M: Research model, very lightweight
- Pythia-160M: Research model, lightweight
- Qwen-0.5B: Instruction-tuned, good for chat
- Qwen3-4B-Instruct: Advanced instruction-tuned model
- Qwen3-4B-Thinking: Reasoning-focused model

Usage:
    python run_small_model_demo.py
    python run_small_model_demo.py "Your prompt here"
    python run_small_model_demo.py --model qwen-0.5b "Your prompt here"
"""

import sys
import os
import argparse
import json
import shutil
import subprocess
import threading
import time
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


def check_model_available(model_name):
    """Check if a model directory exists."""
    base_dir = Path(__file__).parent
    model_path = base_dir / model_name
    return model_path.exists()


def check_ollama_available():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_ollama_model(model_name, prompt, max_tokens=None):
    """Run a model using Ollama."""
    try:
        print(f"🤖 Using Ollama to run {model_name}...")

        # Map our model names to Ollama model names
        ollama_model_map = {
            "gpt-oss-20b": "gpt-oss:20b",
            "llama3.2-1b": "llama3.2:1b",
        }

        ollama_model = ollama_model_map.get(model_name, model_name)

        # Construct the ollama run command
        cmd = ["ollama", "run", ollama_model, prompt]

        # Show thinking indicator
        thinking = ThinkingIndicator()
        thinking.start()

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        thinking.stop()
        print("🔄 Response generated!")

        if result.returncode == 0:
            response = result.stdout.strip()
            return response
        else:
            print(f"❌ Ollama error: {result.stderr.strip()}")
            return None

    except subprocess.TimeoutExpired:
        thinking.stop()
        print("❌ Ollama request timed out (>2 minutes)")
        return None
    except Exception as e:
        thinking.stop()
        print(f"❌ Error running Ollama: {e}")
        return None


def run_with_transformers(model_path, prompt, max_tokens=100):
    """Run model using transformers (if available)."""
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"📥 Loading model from {model_path}...")

        # Use different settings for large models
        model_name = Path(model_path).name
        if "gpt-oss" in model_name.lower():
            print("🔧 Detected GPT-OSS model, using optimized settings...")
            print("🔧 Attempting to bypass MXFP4 quantization for Mac compatibility...")

            # First, try to temporarily modify the config to bypass quantization
            config_path = Path(model_path) / "config.json"
            config_backup_path = Path(model_path) / "config.json.backup_temp"
            config_modified = False

            try:
                # Backup and modify config if it exists
                if config_path.exists():
                    print("🔧 Temporarily modifying config to disable quantization...")

                    # Backup original config
                    shutil.copy2(config_path, config_backup_path)
                    config_modified = True

                    # Load and modify config
                    with open(config_path, "r") as f:
                        config = json.load(f)

                    # Remove quantization config
                    if "quantization_config" in config:
                        del config["quantization_config"]

                    # Save modified config
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)

                # Now try to create pipeline
                print("🔧 Using official GPT-OSS pipeline configuration...")
                pipe = pipeline(
                    "text-generation",
                    model=model_path,  # Pass model path directly
                    torch_dtype="auto",  # Let transformers auto-detect best dtype
                    device_map="auto",  # Auto device mapping as per README
                    trust_remote_code=True,
                )

            except Exception as e:
                # Restore original config if we modified it
                if config_modified and config_backup_path.exists():
                    shutil.copy2(config_backup_path, config_path)
                    config_backup_path.unlink()  # Remove backup
                raise e

            # Show thinking indicator
            thinking = ThinkingIndicator()
            thinking.start()

            # GPT-OSS uses messages format (harmony response format)
            messages = [
                {"role": "user", "content": prompt},
            ]

            # Generate with GPT-OSS specific settings
            outputs = pipe(
                messages,  # Use messages format, not raw text
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
            )

            thinking.stop()
            print("🔄 Response generated!")

            # GPT-OSS response extraction (different structure)
            response = outputs[0]["generated_text"][-1]["content"]

            # Restore original config
            if config_modified and config_backup_path.exists():
                print("🔧 Restoring original config...")
                shutil.copy2(config_backup_path, config_path)
                config_backup_path.unlink()

            return response

        else:
            # Load model and tokenizer for standard models
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Standard model loading for non-GPT-OSS models
            if "20b" in model_name.lower():
                print("🔧 Detected large model, using memory-optimized settings...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,  # Use float16 for large models
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,  # Use float32 for smaller models
                    low_cpu_mem_usage=True,
                )

            # Create pipeline for standard models
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

            # Show thinking indicator
            thinking = ThinkingIndicator()
            thinking.start()

            # Generate for standard models
            outputs = pipe(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

            thinking.stop()
            print("🔄 Response generated!")

            response = outputs[0]["generated_text"]

            # Try to extract just the new part (after the prompt)
            if response.startswith(prompt):
                response = response[len(prompt) :].strip()

            return response

    except ImportError:
        return None
    except Exception as e:
        # Stop thinking indicator if it was started
        if "thinking" in locals():
            thinking.stop()

        error_str = str(e).lower()
        print(f"❌ Error with transformers: {e}")

        # Provide specific guidance for GPT-OSS issues
        model_name = Path(model_path).name if "model_path" in locals() else ""
        if "gpt-oss" in model_name.lower():
            print(f"\n🔧 GPT-OSS Model Issue Detected:")
            if "gate_up_proj" in error_str or "moe" in error_str:
                print("   This appears to be a Mixture-of-Experts (MoE) runtime issue.")
                print("   The model loaded successfully but failed during inference.")
                print(f"\n💡 Alternative Solution:")
                print(f"   The GPT-OSS README recommends using Ollama for consumer hardware:")
                print(f"   1. Install Ollama: https://ollama.com/download")
                print(f"   2. Run: ollama pull gpt-oss:20b")
                print(f"   3. Run: ollama run gpt-oss:20b")
                print(f"\n   This avoids the MoE compatibility issues with transformers on Mac.")
            elif "mxfp4" in error_str:
                print("   MXFP4 quantization requires CUDA GPUs.")
                print("   Consider using Ollama as recommended in the README.")

        return None


def run_simple_demo(model_name, prompt):
    """Run a simple demo without heavy dependencies."""
    print(f"🤖 Running {model_name} model demo")
    print(f"📝 Prompt: {prompt}")
    print("\n" + "=" * 50)

    # Define which models should use Ollama
    ollama_models = ["gpt-oss-20b", "llama3.2-1b"]

    if model_name in ollama_models:
        # Use Ollama for these models
        if not check_ollama_available():
            print("❌ Ollama is not available!")
            print("💡 Install Ollama:")
            print("   brew install ollama")
            print("   brew services start ollama")
            return False

        response = run_ollama_model(model_name, prompt)
        if response:
            print("💬 Response:")
            print(response)
            return True
        else:
            print(f"💡 Alternative suggestions:")
            if model_name == "gpt-oss-20b":
                print("   The 20B model may be too large for your hardware.")
                print("   Try the smaller model: python run_small_model_demo.py --model llama3.2-1b")
            return False
    else:
        # Use transformers for local models
        model_path = Path(__file__).parent / model_name

        # Try transformers first
        if os.path.exists(model_path):
            response = run_with_transformers(model_path, prompt)
            if response:
                print("💬 Response:")
                print(response)
                return True

        # Fallback message
        print("❌ Could not run model. Possible issues:")
        print("   1. transformers not installed: pip install transformers torch")
        print("   2. Model files not found")
        print("   3. Insufficient memory")
        print(f"\n📁 Looking for model at: {model_path}")
        print(f"   Exists: {model_path.exists()}")

        return False


def list_available_models():
    """List available models."""
    base_dir = Path(__file__).parent

    # Build model list dynamically to avoid duplicates
    all_models = []
    model_names_seen = set()

    # Add Ollama models
    ollama_models = ["gpt-oss-20b", "llama3.2-1b"]
    ollama_descriptions = {
        "gpt-oss-20b": "GPT-OSS 20B - Large model (via Ollama)",
        "llama3.2-1b": "Llama 3.2 1B - Small efficient model (via Ollama)",
    }

    for model_name in ollama_models:
        if model_name not in model_names_seen:
            all_models.append((model_name, ollama_descriptions[model_name]))
            model_names_seen.add(model_name)

    # Add local transformers models
    model_descriptions = {
        "smollm-135m": "SmolLM 135M - Fastest, basic capabilities",
        "smollm-360m": "SmolLM 360M - Fast, better capabilities",
        "pythia-70m": "Pythia 70M - Very lightweight research model",
        "pythia-160m": "Pythia 160M - Lightweight research model",
        "qwen-0.5b": "Qwen 0.5B - Instruction-tuned chat model",
        "qwen3-4b-instruct": "Qwen3 4B Instruct - Advanced instruction-tuned model",
        "qwen3-4b-thinking": "Qwen3 4B Thinking - Reasoning-focused model",
    }

    for model_name, description in model_descriptions.items():
        if model_name not in model_names_seen and (base_dir / model_name).exists():
            all_models.append((model_name, description))
            model_names_seen.add(model_name)

    # Sort alphabetically
    all_models.sort(key=lambda x: x[0])

    print("📋 Available models:")

    for model_dir, description in all_models:
        if model_dir in ollama_models:
            # Check if Ollama model is available
            if check_ollama_available():
                try:
                    # Check if specific model is downloaded in Ollama
                    model_map = {
                        "gpt-oss-20b": "gpt-oss:20b",
                        "llama3.2-1b": "llama3.2:1b",
                    }
                    ollama_name = model_map.get(model_dir, model_dir)
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                    status = "✓" if ollama_name in result.stdout else "📥"
                except:
                    status = "❌"
            else:
                status = "❌"
        else:
            # Check if local model directory exists
            status = "✓" if (base_dir / model_dir).exists() else "❌"

        print(f"   {status} {model_dir}: {description}")

    print("\n📋 Legend:")
    print("   ✓ = Ready to use")
    print("   📥 = Available via Ollama (run: ollama pull <model>)")
    print("   ❌ = Not available")


def main():
    parser = argparse.ArgumentParser(description="Run small language models locally")
    parser.add_argument("prompt", nargs="*", default=None, help="Text prompt for the model")
    parser.add_argument(
        "--model",
        default="smollm-360m",
        choices=[
            "smollm-135m",
            "smollm-360m",
            "pythia-70m",
            "pythia-160m",
            "qwen-0.5b",
            "qwen3-4b-instruct",
            "qwen3-4b-thinking",
            "gpt-oss-20b",
            "llama3.2-1b",
        ],
        help="Model to use",
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    # Get prompt
    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = "Explain quantum computing in simple terms:"

    # Check if model exists (skip check for Ollama models)
    ollama_models = ["gpt-oss-20b", "llama3.2-1b"]
    if args.model not in ollama_models and not check_model_available(args.model):
        print(f"❌ Model {args.model} not found!")
        print("\nAvailable models:")
        list_available_models()
        print(f"\nTo download missing models, run:")
        print(f"   python download_small_models.py")
        sys.exit(1)

    # Run the demo
    success = run_simple_demo(args.model, prompt)

    if not success:
        print(f"\n💡 Try installing required packages:")
        print(f"   pip install transformers torch")
        print(f"\n💡 Or try a different model:")
        print(f'   python {sys.argv[0]} --model smollm-135m "your prompt"')
        sys.exit(1)


if __name__ == "__main__":
    main()
