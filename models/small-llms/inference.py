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
        
        # Load model with appropriate settings for Mac
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Move to device if needed
        if device == "mps":
            self.model = self.model.to(device)
            
        print("✅ Model loaded successfully!")
        
    def generate_transformers(self, prompt, max_new_tokens=100, temperature=0.7):
        """Generate text using transformers model"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to device
        if device == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        elapsed = time.time() - start_time
        tokens_generated = outputs[0].shape[0] - inputs['input_ids'].shape[1]
        tokens_per_sec = tokens_generated / elapsed
        
        return {
            "response": response,
            "time": elapsed,
            "tokens_per_second": tokens_per_sec,
            "tokens_generated": tokens_generated
        }
    
    def chat(self, message, system_prompt=None, max_new_tokens=200):
        """Chat interface for instruction models"""
        # Format prompt based on model
        if "qwen" in self.model_path.lower():
            prompt = f"<|im_start|>system\n{system_prompt or 'You are a helpful assistant.'}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        elif "smollm" in self.model_path.lower():
            prompt = f"<|im_start|>system\n{system_prompt or 'You are a helpful assistant.'}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        elif "tinyllama" in self.model_path.lower():
            prompt = f"<|system|>\n{system_prompt or 'You are a helpful assistant.'}</s>\n<|user|>\n{message}</s>\n<|assistant|>\n"
        else:
            # Generic format
            prompt = f"{system_prompt or ''}\n\nHuman: {message}\n\nAssistant:"
        
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
        {
            "name": "Simple Q&A",
            "prompt": "What is the capital of France?",
            "max_tokens": 50
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to calculate factorial:",
            "max_tokens": 100
        },
        {
            "name": "Creative Writing",
            "prompt": "Once upon a time in a small village,",
            "max_tokens": 150
        },
        {
            "name": "Math Problem",
            "prompt": "What is 25 + 37?",
            "max_tokens": 30
        }
    ]
    
    results = []
    
    for test in test_prompts:
        print(f"\n🔍 Test: {test['name']}")
        print(f"Prompt: {test['prompt']}")
        
        result = llm.chat(test['prompt'], max_new_tokens=test['max_tokens'])
        
        print(f"Response: {result['response']}")
        print(f"⏱️  Time: {result['time']:.2f}s")
        print(f"⚡ Speed: {result['tokens_per_second']:.1f} tokens/sec")
        
        results.append({
            "test": test['name'],
            "prompt": test['prompt'],
            "response": result['response'],
            "time": result['time'],
            "tokens_per_second": result['tokens_per_second']
        })
    
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
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        print("\n🤖 Assistant: ", end='', flush=True)
        result = llm.chat(user_input, max_new_tokens=200)
        print(result['response'])
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
            if os.path.isdir(path) and any(f.endswith('.json') for f in os.listdir(path)):
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