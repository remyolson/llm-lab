#!/usr/bin/env python3
"""
Dataset Converter for LLM Lab Benchmarks

This script converts raw datasets from datasets/benchmarking/raw/
to the JSONL format expected by the benchmark runner.

Supported datasets:
- ARC (AI2 Reasoning Challenge)
- GSM8K (Grade School Math 8K)
- MMLU (Massive Multitask Language Understanding)
- HellaSwag (Commonsense Natural Language Inference)
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import click


def extract_numeric_answer(answer_text: str) -> Optional[str]:
    """Extract the final numeric answer from GSM8K answer format."""
    # Look for the final answer pattern: #### [answer]
    match = re.search(r"#### (.+?)(?:\n|$)", answer_text)
    if match:
        return match.group(1).strip()

    # Fallback: look for numbers in the last line
    lines = answer_text.strip().split("\n")
    if lines:
        last_line = lines[-1]
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", last_line)
        if numbers:
            return numbers[-1]

    return None


def convert_arc_dataset(input_path: str, output_path: str, limit: Optional[int] = None) -> int:
    """Convert ARC dataset to benchmark format."""
    print(f"Converting ARC dataset from {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converted_count = 0
    with open(output_path, "w") as f:
        for i, item in enumerate(data):
            # Find the correct answer
            answer_key = item["answerKey"]
            choices = item["choices"]["text"]
            labels = (
                item["choices"]["label"] if "label" in item["choices"] else ["A", "B", "C", "D"]
            )

            # Map answer key to choice text
            correct_answer = None
            for j, label in enumerate(labels):
                if label == answer_key and j < len(choices):
                    correct_answer = choices[j]
                    break

            if not correct_answer:
                continue

            # Create prompt with multiple choice format
            prompt = f"{item['question']}\n\nChoices:\n"
            for j, choice in enumerate(choices):
                letter = labels[j] if j < len(labels) else chr(65 + j)
                prompt += f"{letter}. {choice}\n"
            prompt += "\nAnswer:"

            # Expected keywords include the correct choice text and letter
            expected_keywords = [correct_answer, f"{answer_key}.", f"({answer_key})", answer_key]

            entry = {
                "id": f"arc-{i+1:03d}",
                "prompt": prompt.strip(),
                "evaluation_method": "keyword_match",
                "expected_keywords": expected_keywords,
            }

            f.write(json.dumps(entry) + "\n")
            converted_count += 1

    return converted_count


def convert_gsm8k_dataset(input_path: str, output_path: str, limit: Optional[int] = None) -> int:
    """Convert GSM8K dataset to benchmark format."""
    print(f"Converting GSM8K dataset from {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converted_count = 0
    with open(output_path, "w") as f:
        for i, item in enumerate(data):
            # Extract the final answer
            final_answer = extract_numeric_answer(item["answer"])
            if not final_answer:
                continue

            # Create expected keywords (various formats for the answer)
            expected_keywords = [
                final_answer,
                f"${final_answer}",
                f"{final_answer} dollars",
                f"answer is {final_answer}",
                f"= {final_answer}",
            ]

            entry = {
                "id": f"gsm8k-{i+1:03d}",
                "prompt": item["question"],
                "evaluation_method": "keyword_match",
                "expected_keywords": expected_keywords,
            }

            f.write(json.dumps(entry) + "\n")
            converted_count += 1

    return converted_count


def convert_mmlu_dataset(input_path: str, output_path: str, limit: Optional[int] = None) -> int:
    """Convert MMLU dataset to benchmark format."""
    print(f"Converting MMLU dataset from {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converted_count = 0
    with open(output_path, "w") as f:
        for i, item in enumerate(data):
            # Create prompt with multiple choice format
            prompt = (
                f"Subject: {item.get('subject', 'General')}\n\n{item['question']}\n\nChoices:\n"
            )

            choices = item["choices"]
            for j, choice in enumerate(choices):
                letter = chr(65 + j)  # A, B, C, D
                prompt += f"{letter}. {choice}\n"
            prompt += "\nAnswer:"

            # Get correct answer
            correct_idx = item.get("answer", 0)
            if isinstance(correct_idx, str):
                # Convert letter to index
                correct_idx = ord(correct_idx.upper()) - ord("A")

            if 0 <= correct_idx < len(choices):
                correct_answer = choices[correct_idx]
                correct_letter = chr(65 + correct_idx)

                expected_keywords = [
                    correct_answer,
                    f"{correct_letter}.",
                    f"({correct_letter})",
                    correct_letter,
                ]

                entry = {
                    "id": f"mmlu-{i+1:03d}",
                    "prompt": prompt.strip(),
                    "evaluation_method": "keyword_match",
                    "expected_keywords": expected_keywords,
                }

                f.write(json.dumps(entry) + "\n")
                converted_count += 1

    return converted_count


def convert_hellaswag_dataset(
    input_path: str, output_path: str, limit: Optional[int] = None
) -> int:
    """Convert HellaSwag dataset to benchmark format."""
    print(f"Converting HellaSwag dataset from {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converted_count = 0
    with open(output_path, "w") as f:
        for i, item in enumerate(data):
            # Create prompt with context and multiple endings
            context = item["ctx"]
            endings = item["endings"]

            prompt = f"Context: {context}\n\nWhich ending makes the most sense?\n\nChoices:\n"
            for j, ending in enumerate(endings):
                letter = chr(65 + j)  # A, B, C, D
                prompt += f"{letter}. {ending}\n"
            prompt += "\nAnswer:"

            # Get correct answer
            correct_idx = item.get("label", 0)
            if isinstance(correct_idx, str):
                correct_idx = int(correct_idx)

            if 0 <= correct_idx < len(endings):
                correct_ending = endings[correct_idx]
                correct_letter = chr(65 + correct_idx)

                expected_keywords = [
                    correct_ending,
                    f"{correct_letter}.",
                    f"({correct_letter})",
                    correct_letter,
                ]

                entry = {
                    "id": f"hellaswag-{i+1:03d}",
                    "prompt": prompt.strip(),
                    "evaluation_method": "keyword_match",
                    "expected_keywords": expected_keywords,
                }

                f.write(json.dumps(entry) + "\n")
                converted_count += 1

    return converted_count


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["arc", "gsm8k", "mmlu", "hellaswag", "all"]),
    default="all",
    help="Dataset to convert",
)
@click.option(
    "--limit", type=int, default=None, help="Limit number of entries to convert (for testing)"
)
@click.option("--force", is_flag=True, help="Overwrite existing processed datasets")
def main(dataset: str, limit: Optional[int], force: bool):
    """Convert raw datasets to benchmark format."""

    base_path = Path(".")
    raw_path = base_path / "datasets" / "benchmarking" / "raw"
    processed_path = base_path / "datasets" / "benchmarking" / "processed"

    datasets_to_convert = {
        "arc": {
            "input": raw_path / "arc" / "sample.json",
            "output": processed_path / "arc" / "dataset.jsonl",
            "converter": convert_arc_dataset,
        },
        "gsm8k": {
            "input": raw_path / "gsm8k" / "sample.json",
            "output": processed_path / "gsm8k" / "dataset.jsonl",
            "converter": convert_gsm8k_dataset,
        },
        "mmlu": {
            "input": raw_path / "mmlu" / "sample.json",
            "output": processed_path / "mmlu" / "dataset.jsonl",
            "converter": convert_mmlu_dataset,
        },
        "hellaswag": {
            "input": raw_path / "hellaswag" / "sample.json",
            "output": processed_path / "hellaswag" / "dataset.jsonl",
            "converter": convert_hellaswag_dataset,
        },
    }

    if dataset == "all":
        datasets = datasets_to_convert
    else:
        datasets = {dataset: datasets_to_convert[dataset]}

    print("🔄 Dataset Conversion Tool")
    print("=" * 50)

    total_converted = 0

    for name, config in datasets.items():
        input_file = config["input"]
        output_file = config["output"]
        converter = config["converter"]

        # Check if input exists
        if not input_file.exists():
            print(f"❌ {name}: Input file not found: {input_file}")
            continue

        # Check if output exists and force flag
        if output_file.exists() and not force:
            print(f"⚠️  {name}: Output file exists, use --force to overwrite: {output_file}")
            continue

        try:
            count = converter(str(input_file), str(output_file), limit)
            total_converted += count
            print(f"✅ {name}: Converted {count} entries to {output_file}")

        except Exception as e:
            print(f"❌ {name}: Conversion failed: {e}")

    print(f"\n🎉 Total entries converted: {total_converted}")

    if total_converted > 0:
        print("\n📝 Next steps:")
        print("1. Update scripts/run_benchmarks.py to include new datasets")
        print(
            "2. Test benchmarks: python scripts/run_benchmarks.py --model gemini-1.5-flash --dataset [dataset_name] --limit 5"
        )


if __name__ == "__main__":
    main()
