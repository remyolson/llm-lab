#!/usr/bin/env python3
"""
Comprehensive Fine-tuning Demo for Cloud and Local Models
========================================================

This example demonstrates all fine-tuning techniques from Use Case 6:
- Cloud model fine-tuning (OpenAI, Anthropic)
- Local model fine-tuning (LoRA, QLoRA, Full)
- Dataset preparation and validation
- Training monitoring and evaluation
- Model comparison and deployment

Usage:
    # Prepare training data
    python fine_tuning_complete_demo.py --prepare-dataset --input data.jsonl --output training_data.json

    # Cloud fine-tuning (OpenAI)
    python fine_tuning_complete_demo.py --cloud-finetune --provider openai --dataset training_data.json

    # Local LoRA fine-tuning
    python fine_tuning_complete_demo.py --local-finetune --method lora --model mistral-7b --dataset training_data.json

    # Compare fine-tuned models
    python fine_tuning_complete_demo.py --compare-models --base gpt-3.5-turbo --finetuned ft:gpt-3.5-turbo:xxx

    # Full workflow
    python fine_tuning_complete_demo.py --full-workflow --dataset training_data.json
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Import from the main library
from providers import get_provider
from utils import setup_logging


@dataclass
class TrainingExample:
    """Represents a single training example."""

    instruction: str
    input: str = ""
    output: str = ""

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI fine-tuning format."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": self.instruction + (f"\n{self.input}" if self.input else ""),
            },
            {"role": "assistant", "content": self.output},
        ]
        return {"messages": messages}

    def to_local_format(self) -> Dict[str, Any]:
        """Convert to local model training format."""
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


@dataclass
class TrainingMetrics:
    """Training metrics for tracking progress."""

    epoch: int
    step: int
    loss: float
    learning_rate: float
    validation_loss: Optional[float] = None
    perplexity: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.perplexity is None and self.loss is not None:
            self.perplexity = np.exp(self.loss)


class DatasetPreparer:
    """Prepares and validates training datasets."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def prepare_dataset(
        self, input_file: str, output_file: str, format: str = "instruction"
    ) -> Dict[str, Any]:
        """Prepare dataset from various formats."""
        self.logger.info(f"Preparing dataset from {input_file}")

        # Load raw data
        raw_data = self._load_raw_data(input_file)

        # Convert to training examples
        examples = self._convert_to_examples(raw_data, format)

        # Validate dataset
        validation_results = self._validate_dataset(examples)

        # Save prepared dataset
        self._save_dataset(examples, output_file, format)

        return {
            "num_examples": len(examples),
            "validation": validation_results,
            "output_file": output_file,
        }

    def _load_raw_data(self, input_file: str) -> List[Dict]:
        """Load raw data from file."""
        data = []

        if input_file.endswith(".jsonl"):
            with open(input_file) as f:
                for line in f:
                    data.append(json.loads(line))
        elif input_file.endswith(".json"):
            with open(input_file) as f:
                data = json.load(f)
        elif input_file.endswith(".csv"):
            df = pd.read_csv(input_file)
            data = df.to_dict("records")
        else:
            raise ValueError(f"Unsupported file format: {input_file}")

        return data

    def _convert_to_examples(self, raw_data: List[Dict], format: str) -> List[TrainingExample]:
        """Convert raw data to training examples."""
        examples = []

        for item in raw_data:
            if format == "instruction":
                example = TrainingExample(
                    instruction=item.get("instruction", item.get("prompt", "")),
                    input=item.get("input", ""),
                    output=item.get("output", item.get("response", "")),
                )
            elif format == "chat":
                # Convert chat format to instruction format
                messages = item.get("messages", [])
                if len(messages) >= 2:
                    instruction = messages[-2].get("content", "")
                    output = messages[-1].get("content", "")
                    example = TrainingExample(instruction=instruction, output=output)
                else:
                    continue
            else:
                raise ValueError(f"Unknown format: {format}")

            examples.append(example)

        return examples

    def _validate_dataset(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Validate dataset quality."""
        validation_results = {
            "total_examples": len(examples),
            "avg_instruction_length": 0,
            "avg_output_length": 0,
            "empty_outputs": 0,
            "duplicate_instructions": 0,
            "quality_score": 0,
        }

        if not examples:
            return validation_results

        # Calculate statistics
        instruction_lengths = []
        output_lengths = []
        instructions_seen = set()

        for example in examples:
            inst_len = len(example.instruction.split())
            out_len = len(example.output.split())

            instruction_lengths.append(inst_len)
            output_lengths.append(out_len)

            if out_len == 0:
                validation_results["empty_outputs"] += 1

            if example.instruction in instructions_seen:
                validation_results["duplicate_instructions"] += 1
            instructions_seen.add(example.instruction)

        validation_results["avg_instruction_length"] = np.mean(instruction_lengths)
        validation_results["avg_output_length"] = np.mean(output_lengths)

        # Calculate quality score
        quality_factors = [
            1.0 - (validation_results["empty_outputs"] / len(examples)),
            1.0 - (validation_results["duplicate_instructions"] / len(examples)),
            min(1.0, validation_results["avg_output_length"] / 50),  # Prefer longer outputs
            min(1.0, len(examples) / 100),  # Prefer larger datasets
        ]
        validation_results["quality_score"] = np.mean(quality_factors)

        return validation_results

    def _save_dataset(self, examples: List[TrainingExample], output_file: str, format: str):
        """Save prepared dataset."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            if format == "instruction":
                data = [example.to_local_format() for example in examples]
                json.dump(data, f, indent=2)
            else:
                # Save as JSONL for OpenAI format
                for example in examples:
                    json.dump(example.to_openai_format(), f)
                    f.write("\n")


class CloudFineTuner:
    """Handles fine-tuning for cloud providers."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.provider = get_provider(provider_name)
        self.logger = logging.getLogger(__name__)

    def upload_dataset(self, dataset_file: str) -> str:
        """Upload dataset to provider."""
        self.logger.info(f"Uploading dataset to {self.provider_name}")

        if self.provider_name == "openai":
            # Simulate OpenAI file upload
            # In practice:
            # with open(dataset_file, 'rb') as f:
            #     response = openai.File.create(file=f, purpose='fine-tune')
            # return response['id']
            return f"file-{int(time.time())}"
        else:
            self.logger.warning(f"Upload not implemented for {self.provider_name}")
            return "simulated-file-id"

    def start_fine_tuning(
        self, training_file: str, base_model: str, suffix: str = None
    ) -> Dict[str, Any]:
        """Start fine-tuning job."""
        self.logger.info(f"Starting fine-tuning on {base_model}")

        if self.provider_name == "openai":
            # Simulate OpenAI fine-tuning
            # In practice:
            # response = openai.FineTuningJob.create(
            #     training_file=training_file,
            #     model=base_model,
            #     suffix=suffix
            # )
            job_id = f"ftjob-{int(time.time())}"

            return {
                "job_id": job_id,
                "status": "pending",
                "model": base_model,
                "created_at": datetime.now().isoformat(),
            }
        else:
            return {
                "job_id": f"job-{int(time.time())}",
                "status": "not_implemented",
                "provider": self.provider_name,
            }

    def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor fine-tuning job progress."""
        # Simulate job monitoring
        return {
            "job_id": job_id,
            "status": "completed",
            "fine_tuned_model": f"ft:{job_id}:model",
            "metrics": {"training_loss": 1.234, "validation_loss": 1.456, "epochs_completed": 3},
        }


class LocalFineTuner:
    """Handles fine-tuning for local models."""

    def __init__(self, model_name: str, method: str = "lora"):
        self.model_name = model_name
        self.method = method
        self.logger = logging.getLogger(__name__)
        self.training_history = []

    def configure_training(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Configure training parameters."""
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                "model_name": self.model_name,
                "method": self.method,
                "lora_config": {
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                    "lora_dropout": 0.1,
                },
                "training": {
                    "num_epochs": 3,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    "learning_rate": 2e-4,
                    "warmup_steps": 100,
                    "max_length": 512,
                },
                "optimization": {
                    "gradient_checkpointing": True,
                    "fp16": True,
                    "dataloader_num_workers": 4,
                },
            }

        return config

    def train(self, dataset_file: str, output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run local model fine-tuning."""
        self.logger.info(f"Starting {self.method} fine-tuning for {self.model_name}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Simulate training process
        num_epochs = config["training"]["num_epochs"]
        steps_per_epoch = 100  # Simulated

        for epoch in range(num_epochs):
            for step in range(steps_per_epoch):
                # Simulate training metrics
                progress = (epoch * steps_per_epoch + step) / (num_epochs * steps_per_epoch)
                loss = 3.0 * (1 - progress) + 0.5 * np.random.random()
                learning_rate = config["training"]["learning_rate"] * (1 - progress * 0.5)

                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    step=step + 1,
                    loss=loss,
                    learning_rate=learning_rate,
                    validation_loss=loss + 0.1 * np.random.random(),
                )

                self.training_history.append(metrics)

                # Log progress
                if step % 20 == 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}/{num_epochs}, Step {step + 1}/{steps_per_epoch}: "
                        f"loss={loss:.3f}, lr={learning_rate:.6f}"
                    )

        # Save training results
        self._save_training_results(output_path)

        return {
            "status": "completed",
            "output_dir": str(output_path),
            "final_loss": self.training_history[-1].loss,
            "total_steps": len(self.training_history),
            "method": self.method,
        }

    def _save_training_results(self, output_path: Path):
        """Save training results and model artifacts."""
        # Save training history
        history_data = [asdict(m) for m in self.training_history]
        with open(output_path / "training_history.json", "w") as f:
            json.dump(history_data, f, indent=2)

        # Save model configuration
        if self.method == "lora":
            adapter_config = {
                "base_model": self.model_name,
                "method": "lora",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            }
            with open(output_path / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f, indent=2)

        # Create README
        readme_content = f"""
# Fine-tuned Model: {self.model_name}

## Training Information
- Base Model: {self.model_name}
- Fine-tuning Method: {self.method}
- Training Date: {datetime.now().strftime("%Y-%m-%d")}
- Final Loss: {self.training_history[-1].loss:.3f}
- Total Steps: {len(self.training_history)}

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("{self.model_name}")
model = PeftModel.from_pretrained(base_model, "{output_path}")
```

## Training Configuration
See `adapter_config.json` for detailed configuration.
"""

        with open(output_path / "README.md", "w") as f:
            f.write(readme_content)


class ModelEvaluator:
    """Evaluates and compares fine-tuned models."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compare_models(
        self, base_model: str, fine_tuned_model: str, test_prompts: List[str]
    ) -> Dict[str, Any]:
        """Compare base and fine-tuned models."""
        self.logger.info(f"Comparing {base_model} vs {fine_tuned_model}")

        results = {
            "base_model": base_model,
            "fine_tuned_model": fine_tuned_model,
            "comparisons": [],
            "metrics": {},
        }

        for prompt in test_prompts:
            # Get responses from both models
            base_response = self._get_response(base_model, prompt)
            ft_response = self._get_response(fine_tuned_model, prompt)

            # Compare responses
            comparison = {
                "prompt": prompt,
                "base_response": base_response["content"],
                "ft_response": ft_response["content"],
                "base_latency": base_response["latency"],
                "ft_latency": ft_response["latency"],
            }

            results["comparisons"].append(comparison)

        # Calculate aggregate metrics
        results["metrics"] = self._calculate_metrics(results["comparisons"])

        return results

    def _get_response(self, model: str, prompt: str) -> Dict[str, Any]:
        """Get response from a model."""
        start_time = time.time()

        try:
            # Determine if it's a fine-tuned model
            if model.startswith("ft:") or "/" in model:
                # Fine-tuned model - extract provider
                provider_name = "openai"  # Default
            else:
                provider_name = "openai"

            provider = get_provider(provider_name)
            response = provider.complete(prompt=prompt, model=model, max_tokens=200)

            latency = time.time() - start_time

            return {
                "content": response.get("content", ""),
                "latency": latency,
                "tokens": response.get("usage", {}).get("completion_tokens", 0),
            }

        except Exception as e:
            self.logger.error(f"Error getting response from {model}: {e}")
            return {"content": f"Error: {e!s}", "latency": time.time() - start_time, "tokens": 0}

    def _calculate_metrics(self, comparisons: List[Dict]) -> Dict[str, Any]:
        """Calculate comparison metrics."""
        base_latencies = [c["base_latency"] for c in comparisons]
        ft_latencies = [c["ft_latency"] for c in comparisons]

        return {
            "avg_base_latency": np.mean(base_latencies),
            "avg_ft_latency": np.mean(ft_latencies),
            "latency_improvement": (np.mean(base_latencies) - np.mean(ft_latencies))
            / np.mean(base_latencies)
            * 100,
            "response_similarity": self._calculate_similarity(comparisons),
            "total_comparisons": len(comparisons),
        }

    def _calculate_similarity(self, comparisons: List[Dict]) -> float:
        """Calculate average similarity between responses."""
        # Simplified similarity calculation
        similarities = []

        for comp in comparisons:
            base_words = set(comp["base_response"].lower().split())
            ft_words = set(comp["ft_response"].lower().split())

            if base_words or ft_words:
                similarity = len(base_words & ft_words) / len(base_words | ft_words)
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0


def visualize_training_history(history_file: str, output_file: str):
    """Visualize training history."""
    with open(history_file) as f:
        history = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(history)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training History", fontsize=16)

    # Plot loss
    axes[0, 0].plot(df.index, df["loss"], label="Training Loss", color="blue")
    if "validation_loss" in df.columns:
        axes[0, 0].plot(df.index, df["validation_loss"], label="Validation Loss", color="red")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot learning rate
    axes[0, 1].plot(df.index, df["learning_rate"], color="green")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Learning Rate")
    axes[0, 1].set_title("Learning Rate Schedule")
    axes[0, 1].grid(True)

    # Plot perplexity
    if "perplexity" in df.columns:
        axes[1, 0].plot(df.index, df["perplexity"], color="purple")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Perplexity")
        axes[1, 0].set_title("Perplexity Over Time")
        axes[1, 0].grid(True)

    # Plot loss by epoch
    epoch_df = df.groupby("epoch").agg({"loss": "mean", "validation_loss": "mean"})
    axes[1, 1].bar(
        epoch_df.index - 0.2, epoch_df["loss"], width=0.4, label="Training", color="blue"
    )
    if "validation_loss" in epoch_df.columns:
        axes[1, 1].bar(
            epoch_df.index + 0.2,
            epoch_df["validation_loss"],
            width=0.4,
            label="Validation",
            color="red",
        )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Average Loss")
    axes[1, 1].set_title("Average Loss by Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main demo execution."""
    parser = argparse.ArgumentParser(description="Fine-tuning Complete Demo")

    # Operation modes
    parser.add_argument("--prepare-dataset", action="store_true", help="Prepare training dataset")
    parser.add_argument("--cloud-finetune", action="store_true", help="Run cloud fine-tuning")
    parser.add_argument("--local-finetune", action="store_true", help="Run local fine-tuning")
    parser.add_argument("--compare-models", action="store_true", help="Compare models")
    parser.add_argument("--full-workflow", action="store_true", help="Run complete workflow")

    # Data parameters
    parser.add_argument("--input", help="Input data file")
    parser.add_argument("--output", help="Output file/directory")
    parser.add_argument("--dataset", help="Training dataset file")
    parser.add_argument("--format", default="instruction", help="Dataset format")

    # Model parameters
    parser.add_argument("--provider", default="openai", help="Cloud provider")
    parser.add_argument("--model", default="mistral-7b", help="Model name")
    parser.add_argument("--base", help="Base model for comparison")
    parser.add_argument("--finetuned", help="Fine-tuned model for comparison")
    parser.add_argument(
        "--method", default="lora", choices=["lora", "qlora", "full"], help="Fine-tuning method"
    )

    # Training parameters
    parser.add_argument("--config", help="Training configuration file")
    parser.add_argument("--suffix", help="Model suffix for cloud fine-tuning")

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        if args.prepare_dataset:
            # Prepare dataset
            logger.info("Preparing training dataset...")
            preparer = DatasetPreparer()

            results = preparer.prepare_dataset(
                args.input or "training_data.jsonl",
                args.output or "prepared_dataset.json",
                args.format,
            )

            print("\n📊 Dataset Preparation Complete")
            print("=" * 50)
            print(f"Total Examples: {results['num_examples']}")
            print(f"Quality Score: {results['validation']['quality_score']:.2f}")
            print(
                f"Avg Instruction Length: {results['validation']['avg_instruction_length']:.1f} words"
            )
            print(f"Avg Output Length: {results['validation']['avg_output_length']:.1f} words")
            print(f"Output saved to: {results['output_file']}")

        elif args.cloud_finetune:
            # Cloud fine-tuning
            logger.info(f"Starting cloud fine-tuning with {args.provider}...")

            tuner = CloudFineTuner(args.provider)

            # Upload dataset
            file_id = tuner.upload_dataset(args.dataset)
            print(f"✅ Dataset uploaded: {file_id}")

            # Start fine-tuning
            job_info = tuner.start_fine_tuning(file_id, args.model or "gpt-3.5-turbo", args.suffix)
            print(f"✅ Fine-tuning job started: {job_info['job_id']}")

            # Monitor job
            print("\n⏳ Monitoring job progress...")
            time.sleep(2)  # Simulate wait

            status = tuner.monitor_job(job_info["job_id"])
            print("\n✅ Job completed!")
            print(f"Fine-tuned model: {status['fine_tuned_model']}")
            print(f"Training loss: {status['metrics']['training_loss']:.3f}")

        elif args.local_finetune:
            # Local fine-tuning
            logger.info(f"Starting local {args.method} fine-tuning...")

            tuner = LocalFineTuner(args.model, args.method)

            # Configure training
            config = tuner.configure_training(args.config)

            print("\n🔧 Training Configuration")
            print("=" * 50)
            print(f"Model: {config['model_name']}")
            print(f"Method: {config['method']}")
            print(f"Epochs: {config['training']['num_epochs']}")
            print(f"Batch Size: {config['training']['batch_size']}")
            print(f"Learning Rate: {config['training']['learning_rate']}")

            # Run training
            print("\n🚀 Starting training...")
            results = tuner.train(
                args.dataset, args.output or f"./fine_tuned_{args.model}_{args.method}", config
            )

            print("\n✅ Training completed!")
            print(f"Output directory: {results['output_dir']}")
            print(f"Final loss: {results['final_loss']:.3f}")

            # Visualize training history
            history_file = Path(results["output_dir"]) / "training_history.json"
            if history_file.exists():
                plot_file = Path(results["output_dir"]) / "training_history.png"
                visualize_training_history(str(history_file), str(plot_file))
                print(f"📊 Training visualization saved to: {plot_file}")

        elif args.compare_models:
            # Compare models
            logger.info("Comparing base and fine-tuned models...")

            evaluator = ModelEvaluator()

            # Test prompts
            test_prompts = [
                "Explain machine learning in simple terms",
                "Write a Python function to sort a list",
                "What are the benefits of exercise?",
                "How does photosynthesis work?",
                "Describe the water cycle",
            ]

            results = evaluator.compare_models(
                args.base or "gpt-3.5-turbo",
                args.finetuned or "ft:gpt-3.5-turbo:example",
                test_prompts,
            )

            print("\n📊 Model Comparison Results")
            print("=" * 50)
            print(f"Base Model: {results['base_model']}")
            print(f"Fine-tuned Model: {results['fine_tuned_model']}")
            print("\nMetrics:")
            print(f"  Avg Base Latency: {results['metrics']['avg_base_latency']:.2f}s")
            print(f"  Avg Fine-tuned Latency: {results['metrics']['avg_ft_latency']:.2f}s")
            print(f"  Latency Improvement: {results['metrics']['latency_improvement']:.1f}%")
            print(f"  Response Similarity: {results['metrics']['response_similarity']:.2f}")

            # Show example comparisons
            print("\n📝 Example Comparisons:")
            for i, comp in enumerate(results["comparisons"][:2]):
                print(f"\nPrompt {i + 1}: {comp['prompt']}")
                print(f"Base: {comp['base_response'][:100]}...")
                print(f"Fine-tuned: {comp['ft_response'][:100]}...")

        elif args.full_workflow:
            # Complete workflow
            logger.info("Running complete fine-tuning workflow...")

            # Step 1: Prepare dataset
            print("\n1️⃣ Preparing Dataset")
            preparer = DatasetPreparer()
            prep_results = preparer.prepare_dataset(
                args.dataset, "prepared_dataset.json", "instruction"
            )
            print(f"✅ Dataset prepared: {prep_results['num_examples']} examples")

            # Step 2: Local fine-tuning
            print("\n2️⃣ Running Local Fine-tuning")
            local_tuner = LocalFineTuner("mistral-7b", "lora")
            config = local_tuner.configure_training()
            train_results = local_tuner.train("prepared_dataset.json", "./fine_tuned_model", config)
            print(f"✅ Training completed: {train_results['output_dir']}")

            # Step 3: Visualize results
            print("\n3️⃣ Generating Visualizations")
            history_file = Path(train_results["output_dir"]) / "training_history.json"
            plot_file = Path(train_results["output_dir"]) / "training_history.png"
            visualize_training_history(str(history_file), str(plot_file))
            print(f"✅ Visualization saved: {plot_file}")

            # Step 4: Summary
            print("\n📊 Workflow Complete!")
            print("=" * 50)
            print(f"Dataset Quality: {prep_results['validation']['quality_score']:.2f}")
            print(f"Final Training Loss: {train_results['final_loss']:.3f}")
            print(f"Total Training Steps: {train_results['total_steps']}")
            print(f"Output Directory: {train_results['output_dir']}")

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"Error in fine-tuning demo: {e}")
        raise


if __name__ == "__main__":
    main()
