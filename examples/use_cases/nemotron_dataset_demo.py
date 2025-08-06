#!/usr/bin/env python3
"""
Nemotron Dataset Demo

This script demonstrates how to use the NVIDIA Nemotron Post-Training Dataset
for fine-tuning language models. It shows various ways to load and process
the dataset's different splits.

Usage:
    python nemotron_dataset_demo.py
    python nemotron_dataset_demo.py --split math --max-samples 1000
    python nemotron_dataset_demo.py --list-datasets
"""

import argparse
import logging

# Add project root to path
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.use_cases.fine_tuning.datasets.dataset_processor import DatasetProcessor
from src.use_cases.fine_tuning.datasets.dataset_registry import (
    DatasetRegistry,
    DatasetType,
    get_nemotron_dataset,
)

# Setup logging and console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


def display_dataset_info():
    """Display information about the Nemotron dataset."""
    dataset_info = get_nemotron_dataset()

    console.print("\n[bold cyan]NVIDIA Nemotron Post-Training Dataset v1[/bold cyan]")
    console.print(f"[yellow]{dataset_info.description}[/yellow]\n")

    # Create info table
    table = Table(title="Dataset Information", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("HuggingFace Path", dataset_info.hf_path)
    table.add_row("Size", dataset_info.size)
    table.add_row("License", dataset_info.license)
    table.add_row("Format", dataset_info.format)
    table.add_row("Available Splits", ", ".join(dataset_info.splits))
    table.add_row("Tasks", ", ".join(dataset_info.tasks))

    console.print(table)

    # Show special features
    console.print("\n[bold]Special Features:[/bold]")
    for feature in dataset_info.special_features:
        console.print(f"  • {feature}")

    # Show recommended models
    console.print("\n[bold]Recommended Models:[/bold]")
    for model in dataset_info.recommended_models:
        console.print(f"  • {model}")


def list_all_datasets():
    """List all available datasets in the registry."""
    console.print("\n[bold cyan]Available Datasets in Registry[/bold cyan]\n")

    # Group by type
    for dataset_type in DatasetType:
        datasets = DatasetRegistry.list_datasets(dataset_type)
        if datasets:
            console.print(f"\n[bold yellow]{dataset_type.value.upper()} Datasets:[/bold yellow]")

            for dataset in datasets:
                console.print(f"  [cyan]{dataset.name}[/cyan]")
                console.print(f"    Path: {dataset.hf_path}")
                console.print(f"    Size: {dataset.size}")
                console.print(f"    Tasks: {', '.join(dataset.tasks)}")


def load_and_preview_split(split: str, max_samples: int = 5):
    """Load and preview a specific split of the Nemotron dataset."""
    console.print(f"\n[bold cyan]Loading Nemotron '{split}' split...[/bold cyan]")

    # Create processor
    processor = DatasetProcessor()

    try:
        # Load the split (using streaming for efficiency)
        dataset = processor.load_nemotron_split(
            split=split, max_samples=max_samples if max_samples > 5 else None, streaming=True
        )

        console.print(f"[green]✓ Successfully loaded '{split}' split[/green]")

        # Show examples
        console.print(f"\n[bold]First {min(5, max_samples)} examples:[/bold]")

        for i, example in enumerate(dataset.take(min(5, max_samples))):
            console.print(f"\n[yellow]Example {i + 1}:[/yellow]")

            # Display fields based on what's available
            for key, value in example.items():
                if isinstance(value, str):
                    # Truncate long strings for display
                    display_value = value[:200] + "..." if len(value) > 200 else value
                    console.print(f"  [cyan]{key}:[/cyan] {display_value}")
                else:
                    console.print(f"  [cyan]{key}:[/cyan] {value}")

    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        logger.error(f"Failed to load dataset: {e}", exc_info=True)


def generate_loading_code(dataset_name: str = "nemotron-post-training"):
    """Generate and display code for loading the dataset."""
    console.print("\n[bold cyan]Loading Code Example[/bold cyan]")

    code = DatasetRegistry.get_dataset_loading_code(dataset_name)

    # Additional example code for Nemotron
    if dataset_name == "nemotron-post-training":
        code += """
# Example: Process for fine-tuning
from src.use_cases.fine_tuning.datasets.dataset_processor import DatasetProcessor

# Initialize processor
processor = DatasetProcessor(max_length=2048)

# Load specific split with streaming (recommended for large dataset)
math_dataset = processor.load_nemotron_split(
    split="math",
    streaming=True,
    max_samples=10000  # Limit for testing
)

# Or load all splits
all_splits = {}
for split in ["chat", "code", "math", "stem", "tool_calling"]:
    all_splits[split] = processor.load_nemotron_split(split=split, streaming=True)

# Process examples
for example in math_dataset.take(5):
    print(example)  # Process your examples here
"""

    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    panel = Panel(syntax, title="Loading Code", border_style="green")
    console.print(panel)


def demonstrate_processing():
    """Demonstrate processing the dataset for fine-tuning."""
    console.print("\n[bold cyan]Processing Demonstration[/bold cyan]")

    processor = DatasetProcessor(max_length=512)

    # Load a small sample
    console.print("\nLoading chat split samples...")
    dataset = processor.load_nemotron_split(split="chat", streaming=False, max_samples=100)

    console.print(f"[green]✓ Loaded {len(dataset)} examples[/green]")

    # Show dataset structure
    console.print("\n[bold]Dataset Structure:[/bold]")
    if len(dataset) > 0:
        first_example = dataset[0]
        for key in first_example.keys():
            console.print(f"  • {key}: {type(first_example[key]).__name__}")

    # Demonstrate formatting
    console.print("\n[bold]Formatting for Training:[/bold]")
    console.print("The dataset can be formatted for different training objectives:")
    console.print("  • Instruction-following format")
    console.print("  • Prompt-completion format")
    console.print("  • Chat format with system prompts")

    # Show statistics
    console.print("\n[bold]Dataset Statistics:[/bold]")
    if not dataset.features:
        console.print("  Features information not available in streaming mode")
    else:
        for feature, ftype in dataset.features.items():
            console.print(f"  • {feature}: {ftype}")


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="Nemotron Dataset Demo")
    parser.add_argument(
        "--split",
        choices=["chat", "code", "math", "stem", "tool_calling"],
        help="Specific split to load and preview",
    )
    parser.add_argument(
        "--max-samples", type=int, default=5, help="Maximum number of samples to preview"
    )
    parser.add_argument(
        "--list-datasets", action="store_true", help="List all available datasets in registry"
    )
    parser.add_argument(
        "--show-code", action="store_true", help="Show code examples for loading the dataset"
    )
    parser.add_argument(
        "--process-demo", action="store_true", help="Demonstrate dataset processing"
    )

    args = parser.parse_args()

    console.print("[bold magenta]╔══════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║    Nemotron Dataset Demo                ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════╝[/bold magenta]")

    if args.list_datasets:
        list_all_datasets()
    elif args.show_code:
        generate_loading_code()
    elif args.process_demo:
        demonstrate_processing()
    elif args.split:
        load_and_preview_split(args.split, args.max_samples)
    else:
        # Default: show dataset info
        display_dataset_info()

        # Show example loading code
        console.print("\n" + "=" * 50)
        generate_loading_code()

        console.print("\n[dim]Run with --help to see all options[/dim]")


if __name__ == "__main__":
    main()
