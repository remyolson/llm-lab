#!/usr/bin/env python3
"""
Verify model consistency between what's downloaded and what we tell users to download
"""

import os
import json

def check_model_consistency():
    """Check if downloaded models match what we tell users to download"""
    
    # Models that the download script downloads by default
    default_transformers = ["smollm-135m", "smollm-360m", "qwen-0.5b"]
    default_gguf = ["qwen-0.5b-gguf"]
    default_models = default_transformers + default_gguf
    
    # Check what's actually downloaded
    model_dir = "/Users/ro/Documents/GitHub/lllm-lab/models/small-llms"
    downloaded_models = []
    
    for item in os.listdir(model_dir):
        path = os.path.join(model_dir, item)
        if os.path.isdir(path) and not item.startswith('.') and not item.startswith('__'):
            downloaded_models.append(item)
    
    print("🔍 Model Consistency Check")
    print("=" * 50)
    
    print("\n📥 Models the script downloads by default:")
    for model in default_models:
        print(f"  - {model}")
    
    print("\n💾 Models actually downloaded:")
    for model in sorted(downloaded_models):
        print(f"  - {model}")
    
    # Check for discrepancies
    print("\n🔄 Consistency Analysis:")
    
    # Models in script but not downloaded
    missing_from_disk = set(default_models) - set(downloaded_models)
    if missing_from_disk:
        print("❌ Models in script but NOT downloaded:")
        for model in missing_from_disk:
            print(f"  - {model}")
    else:
        print("✅ All default models are downloaded")
    
    # Models downloaded but not in default list
    extra_on_disk = set(downloaded_models) - set(default_models)
    if extra_on_disk:
        print("⚠️  Models downloaded but NOT in default list:")
        for model in extra_on_disk:
            print(f"  - {model}")
    else:
        print("✅ No extra models on disk")
    
    # Final verdict
    print("\n📊 Final Result:")
    if not missing_from_disk and not extra_on_disk:
        print("✅ PERFECT CONSISTENCY! Downloaded models match exactly what new users will get.")
    else:
        print("❌ INCONSISTENCY DETECTED! Action needed to sync models.")
        
        if missing_from_disk:
            print("\nTo fix missing models, run:")
            print("python models/small-llms/download_small_models.py")
        
        if extra_on_disk:
            print("\nExtra models on disk (this is OK, just means you have more than the defaults)")

if __name__ == "__main__":
    check_model_consistency()