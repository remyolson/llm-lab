#!/usr/bin/env python3
"""
Quick demo to interact with small LLMs on MacBook
"""

import os
import sys
from inference import SmallLLMInference

def main():
    print("🤖 Small LLM Quick Demo")
    print("=" * 50)
    
    # List available models
    model_dir = "models/small-llms"
    models = []
    
    for item in os.listdir(model_dir):
        path = os.path.join(model_dir, item)
        if os.path.isdir(path) and any(f.endswith('.json') for f in os.listdir(path)):
            models.append((item, path))
    
    if not models:
        print("❌ No models found! Run download_small_models.py first.")
        return
    
    print("\nAvailable models:")
    for i, (name, _) in enumerate(models):
        size_info = {
            "smollm-135m": "135M params - Fastest, basic tasks",
            "smollm-360m": "360M params - Better quality",
            "qwen-0.5b": "500M params - Best quality",
            "pythia-70m": "70M params - Ultra tiny",
            "pythia-160m": "160M params - Very small",
            "tinyllama": "1.1B params - Larger but capable"
        }
        print(f"{i+1}. {name} - {size_info.get(name, 'Small model')}")
    
    # Select model
    print("\nSelect a model (1-{}): ".format(len(models)), end='')
    choice = input()
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            model_name, model_path = models[idx]
        else:
            print("Invalid choice, using first model")
            model_name, model_path = models[0]
    except:
        print("Invalid input, using first model")
        model_name, model_path = models[0]
    
    print(f"\n✅ Loading {model_name}...")
    llm = SmallLLMInference(model_path)
    llm.load_transformers_model()
    
    print("\n💬 Chat interface ready! (type 'quit' to exit)")
    print("Tip: These are small models - keep questions simple!")
    print("-" * 50)
    
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        print("\n🤖 Assistant: ", end='', flush=True)
        
        # Use appropriate max tokens based on model size
        max_tokens = 100 if "135m" in model_name or "70m" in model_name else 200
        
        result = llm.chat(user_input, max_new_tokens=max_tokens)
        print(result['response'])
        print(f"\n[⚡ {result['tokens_per_second']:.1f} tokens/sec]")

if __name__ == "__main__":
    main()