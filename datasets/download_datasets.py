#!/usr/bin/env python3
"""
Download popular LLM benchmarking and fine-tuning datasets from Hugging Face
"""

import json
import os
from datasets import load_dataset
import pandas as pd

def save_dataset_sample(dataset, name, category, sample_size=1000):
    """Save a sample of the dataset in both JSON and CSV formats"""
    base_path = f"datasets/{category}/raw/{name}"
    os.makedirs(base_path, exist_ok=True)

    # Convert to pandas for easier manipulation
    if hasattr(dataset, 'to_pandas'):
        df = dataset.to_pandas()
    else:
        df = pd.DataFrame(dataset)

    # Take sample if dataset is large
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    # Save as JSON
    df_sample.to_json(f"{base_path}/sample.json", orient='records', indent=2)

    # Save as CSV
    df_sample.to_csv(f"{base_path}/sample.csv", index=False)

    # Save metadata
    metadata = {
        "name": name,
        "category": category,
        "total_size": len(df),
        "sample_size": len(df_sample),
        "columns": list(df.columns),
        "description": f"Sample of {name} dataset from Hugging Face"
    }

    with open(f"{base_path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved {name} ({len(df_sample)} samples)")
    return metadata

def download_benchmarking_datasets():
    """Download popular benchmarking datasets"""
    print("\n=== Downloading Benchmarking Datasets ===")

    datasets_info = []

    # MMLU (Measuring Massive Multitask Language Understanding)
    try:
        print("Downloading MMLU...")
        mmlu = load_dataset("cais/mmlu", "all", split="test", streaming=True)
        mmlu_samples = list(mmlu.take(1000))
        info = save_dataset_sample(mmlu_samples, "mmlu", "benchmarking")
        datasets_info.append(info)
    except Exception as e:
        print(f"Error downloading MMLU: {e}")

    # HellaSwag
    try:
        print("Downloading HellaSwag...")
        hellaswag = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
        hellaswag_samples = list(hellaswag.take(1000))
        info = save_dataset_sample(hellaswag_samples, "hellaswag", "benchmarking")
        datasets_info.append(info)
    except Exception as e:
        print(f"Error downloading HellaSwag: {e}")

    # ARC (AI2 Reasoning Challenge)
    try:
        print("Downloading ARC...")
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", streaming=True)
        arc_samples = list(arc.take(1000))
        info = save_dataset_sample(arc_samples, "arc", "benchmarking")
        datasets_info.append(info)
    except Exception as e:
        print(f"Error downloading ARC: {e}")

    # TruthfulQA
    try:
        print("Downloading TruthfulQA...")
        truthfulqa = load_dataset("truthful_qa", "generation", split="validation", streaming=True)
        truthfulqa_samples = list(truthfulqa.take(500))
        info = save_dataset_sample(truthfulqa_samples, "truthfulqa", "benchmarking")
        datasets_info.append(info)
    except Exception as e:
        print(f"Error downloading TruthfulQA: {e}")

    # GSM8K (Grade School Math)
    try:
        print("Downloading GSM8K...")
        gsm8k = load_dataset("gsm8k", "main", split="test", streaming=True)
        gsm8k_samples = list(gsm8k.take(1000))
        info = save_dataset_sample(gsm8k_samples, "gsm8k", "benchmarking")
        datasets_info.append(info)
    except Exception as e:
        print(f"Error downloading GSM8K: {e}")

    return datasets_info

def download_finetuning_datasets():
    """Download popular fine-tuning datasets"""
    print("\n=== Downloading Fine-tuning Datasets ===")

    datasets_info = []

    # Alpaca
    try:
        print("Downloading Alpaca...")
        alpaca = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        alpaca_samples = list(alpaca.take(1000))
        info = save_dataset_sample(alpaca_samples, "alpaca", "fine-tuning")
        datasets_info.append(info)
    except Exception as e:
        print(f"Error downloading Alpaca: {e}")

    # ShareGPT (cleaned version)
    try:
        print("Downloading ShareGPT...")
        sharegpt = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        sharegpt_samples = list(sharegpt.take(1000))
        info = save_dataset_sample(sharegpt_samples, "sharegpt", "fine-tuning")
        datasets_info.append(info)
    except Exception as e:
        print(f"Error downloading ShareGPT: {e}")

    # OpenOrca
    try:
        print("Downloading OpenOrca sample...")
        openorca = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        openorca_samples = list(openorca.take(1000))
        info = save_dataset_sample(openorca_samples, "openorca", "fine-tuning")
        datasets_info.append(info)
    except Exception as e:
        print(f"Error downloading OpenOrca: {e}")

    # Dolly 15k
    try:
        print("Downloading Dolly-15k...")
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        info = save_dataset_sample(dolly, "dolly-15k", "fine-tuning", sample_size=2000)
        datasets_info.append(info)
    except Exception as e:
        print(f"Error downloading Dolly-15k: {e}")

    # WizardLM Evol Instruct
    try:
        print("Downloading WizardLM Evol Instruct...")
        wizard = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train", streaming=True)
        wizard_samples = list(wizard.take(1000))
        info = save_dataset_sample(wizard_samples, "wizardlm", "fine-tuning")
        datasets_info.append(info)
    except Exception as e:
        print(f"Error downloading WizardLM: {e}")

    return datasets_info

def main():
    """Main function to download all datasets"""
    print("Starting dataset downloads...")

    all_datasets = []

    # Download benchmarking datasets
    benchmarking_datasets = download_benchmarking_datasets()
    all_datasets.extend(benchmarking_datasets)

    # Download fine-tuning datasets
    finetuning_datasets = download_finetuning_datasets()
    all_datasets.extend(finetuning_datasets)

    # Save overall manifest
    manifest = {
        "total_datasets": len(all_datasets),
        "benchmarking_datasets": len(benchmarking_datasets),
        "finetuning_datasets": len(finetuning_datasets),
        "datasets": all_datasets
    }

    with open("datasets/manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Successfully downloaded {len(all_datasets)} datasets!")
    print("Check datasets/manifest.json for details")

if __name__ == "__main__":
    main()
