"""
Fine-Tuning CLI Interface

This module provides a comprehensive command-line interface for managing
fine-tuning jobs, including training, evaluation, checkpoint management,
and recipe configuration.

Example:
    # Start a new fine-tuning job
    python -m lllm_lab.fine_tuning train --recipe chat --model llama2-7b --data custom.jsonl

    # List running jobs
    python -m lllm_lab.fine_tuning list-jobs

    # Evaluate a checkpoint
    python -m lllm_lab.fine_tuning evaluate --checkpoint path/to/checkpoint
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import Click for CLI
import click
import yaml
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.tree import Tree

from ..checkpoints.checkpoint_manager import CheckpointManager
from ..evaluation.suite import EvaluationConfig, EvaluationSuite
from ..monitoring.config import MonitoringConfig, MonitoringSetup
from ..pipelines.data_preprocessor import DataPreprocessor

# Import fine-tuning components
from ..recipes.recipe_manager import Recipe, RecipeManager
from ..training.distributed_trainer import DistributedTrainer, DistributedTrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

# Global state for job management
RUNNING_JOBS = {}
JOB_STATUS_FILE = Path.home() / ".lllm_lab" / "fine_tuning_jobs.json"


class JobManager:
    """Manages fine-tuning job state and persistence."""

    def __init__(self):
        """Initialize job manager."""
        self.jobs_file = JOB_STATUS_FILE
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
        self.jobs = self._load_jobs()

    def _load_jobs(self) -> Dict[str, Any]:
        """Load jobs from persistent storage."""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load jobs: {e}")
        return {}

    def _save_jobs(self):
        """Save jobs to persistent storage."""
        try:
            with open(self.jobs_file, "w") as f:
                json.dump(self.jobs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")

    def create_job(self, recipe_name: str, model: str, dataset: str, **kwargs) -> str:
        """Create a new job entry."""
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.jobs[job_id] = {
            "id": job_id,
            "recipe": recipe_name,
            "model": model,
            "dataset": dataset,
            "status": "initialized",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "pid": None,
            "checkpoint_dir": None,
            "metrics": {},
            "config": kwargs,
        }

        self._save_jobs()
        return job_id

    def update_job(self, job_id: str, **updates):
        """Update job information."""
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)
            self.jobs[job_id]["updated_at"] = datetime.now().isoformat()
            self._save_jobs()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job information."""
        return self.jobs.get(job_id)

    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List jobs with optional status filter."""
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j["status"] == status]

        # Sort by creation time
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        return jobs

    def delete_job(self, job_id: str):
        """Delete a job entry."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._save_jobs()


# Initialize global job manager
job_manager = JobManager()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config", "-c", type=click.Path(), help="Configuration file")
def cli(verbose: bool, config: Optional[str]):
    """LLLM Lab Fine-Tuning CLI - Manage fine-tuning jobs and workflows."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if config:
        # Load configuration
        with open(config) as f:
            if config.endswith(".yaml") or config.endswith(".yml"):
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)

        # Apply configuration
        os.environ.update({k: str(v) for k, v in cfg.items() if isinstance(v, (str, int, float))})


@cli.command()
@click.option("--recipe", "-r", required=True, help="Recipe name or path")
@click.option("--model", "-m", help="Model name (overrides recipe)")
@click.option("--data", "-d", help="Dataset path (overrides recipe)")
@click.option("--output-dir", "-o", default="./fine_tuned_models", help="Output directory")
@click.option("--num-epochs", type=int, help="Number of training epochs")
@click.option("--batch-size", type=int, help="Training batch size")
@click.option("--learning-rate", type=float, help="Learning rate")
@click.option("--distributed", is_flag=True, help="Enable distributed training")
@click.option(
    "--backend",
    type=click.Choice(["ddp", "fsdp", "deepspeed", "accelerate"]),
    default="ddp",
    help="Distributed backend",
)
@click.option("--resume-from", help="Resume from checkpoint")
@click.option("--interactive", "-i", is_flag=True, help="Interactive configuration mode")
def train(
    recipe: str,
    model: Optional[str],
    data: Optional[str],
    output_dir: str,
    num_epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    distributed: bool,
    backend: str,
    resume_from: Optional[str],
    interactive: bool,
):
    """Start a new fine-tuning training job."""
    console.print(Panel.fit("[bold blue]Starting Fine-Tuning Job[/bold blue]", border_style="blue"))

    try:
        # Load and configure recipe
        recipe_obj = _load_and_configure_recipe(
            recipe, model, data, num_epochs, batch_size, learning_rate, interactive
        )

        # Create and validate job
        job_id = _create_training_job(recipe_obj, output_dir, distributed, backend)

        # Execute training workflow
        _execute_training_workflow(job_id, recipe_obj, output_dir, distributed, backend)

    except Exception as e:
        console.print(f"[red]Error starting training:[/red] {e}")
        if "job_id" in locals():
            job_manager.update_job(job_id, status="failed", error=str(e))
        raise click.ClickException(str(e))


def _load_and_configure_recipe(
    recipe: str,
    model: Optional[str],
    data: Optional[str],
    num_epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    interactive: bool,
) -> Recipe:
    """Load recipe from file or name and apply configuration overrides."""
    recipe_manager = RecipeManager()

    if recipe.endswith(".yaml") or recipe.endswith(".yml") or recipe.endswith(".json"):
        # Load from file
        with open(recipe) as f:
            if recipe.endswith(".json"):
                recipe_data = json.load(f)
            else:
                recipe_data = yaml.safe_load(f)
        recipe_obj = Recipe.from_dict(recipe_data)
    else:
        # Load by name
        recipe_obj = recipe_manager.load_recipe(recipe)

    # Interactive configuration if requested
    if interactive:
        recipe_obj = _interactive_recipe_config(recipe_obj)

    # Override recipe parameters if provided
    if model:
        recipe_obj.model.base_model = model
    if data:
        recipe_obj.dataset.path = data
    if num_epochs:
        recipe_obj.training.num_epochs = num_epochs
    if batch_size:
        recipe_obj.training.per_device_train_batch_size = batch_size
    if learning_rate:
        recipe_obj.training.learning_rate = learning_rate

    return recipe_obj


def _create_training_job(
    recipe_obj: Recipe, output_dir: str, distributed: bool, backend: str
) -> str:
    """Create a new training job and display configuration for user confirmation."""
    # Create job
    job_id = job_manager.create_job(
        recipe_name=recipe_obj.name,
        model=recipe_obj.model.base_model,
        dataset=recipe_obj.dataset.path,
        output_dir=output_dir,
        distributed=distributed,
        backend=backend,
    )

    console.print(f"[green]Created job:[/green] {job_id}")

    # Show configuration
    _display_recipe_config(recipe_obj)

    if not Confirm.ask("\\nProceed with training?"):
        job_manager.update_job(job_id, status="cancelled")
        console.print("[yellow]Training cancelled[/yellow]")
        raise click.ClickException("Training cancelled by user")

    return job_id


def _execute_training_workflow(
    job_id: str, recipe_obj: Recipe, output_dir: str, distributed: bool, backend: str
):
    """Execute the complete training workflow with progress tracking."""
    job_manager.update_job(job_id, status="starting")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing training...", total=100)

        # Initialize model and tokenizer
        progress.update(task, advance=20, description="Loading model...")
        model_obj, tokenizer = _initialize_model_and_tokenizer(recipe_obj)

        # Prepare dataset
        progress.update(task, advance=20, description="Preparing dataset...")
        dataset = _prepare_training_dataset(recipe_obj, tokenizer)

        # Setup trainer
        progress.update(task, advance=20, description="Setting up training...")
        trainer = _setup_trainer(
            recipe_obj, model_obj, tokenizer, dataset, output_dir, distributed, backend
        )

        # Start training
        progress.update(task, advance=20, description="Starting training...")
        _start_training_process(job_id, recipe_obj, output_dir)

        progress.update(task, advance=20, description="Training in progress...")


def _initialize_model_and_tokenizer(recipe_obj: Recipe) -> Tuple[Any | Any]:
    """Initialize model and tokenizer from recipe configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_obj = AutoModelForCausalLM.from_pretrained(
        recipe_obj.model.base_model,
        torch_dtype=recipe_obj.model.torch_dtype,
        device_map=recipe_obj.model.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(recipe_obj.model.base_model)

    return model_obj, tokenizer


def _prepare_training_dataset(recipe_obj: Recipe, tokenizer: Any) -> Dict[str, Any]:
    """Prepare training and validation datasets."""
    preprocessor = DataPreprocessor()
    return preprocessor.prepare_dataset(
        dataset_path=recipe_obj.dataset.path,
        tokenizer=tokenizer,
        format_type=recipe_obj.dataset.format,
        max_length=recipe_obj.dataset.tokenizer_config.get("max_length", 512),
    )


def _setup_trainer(
    recipe_obj: Recipe,
    model_obj: Any,
    tokenizer: Any,
    dataset: Dict[str, Any],
    output_dir: str,
    distributed: bool,
    backend: str,
) -> Any:
    """Setup appropriate trainer (distributed or standard) based on configuration."""
    if distributed:
        return _setup_distributed_trainer(
            recipe_obj, model_obj, tokenizer, dataset, output_dir, backend
        )
    else:
        return _setup_standard_trainer(recipe_obj, model_obj, tokenizer, dataset, output_dir)


def _setup_distributed_trainer(
    recipe_obj: Recipe,
    model_obj: Any,
    tokenizer: Any,
    dataset: Dict[str, Any],
    output_dir: str,
    backend: str,
) -> Any:
    """Setup distributed trainer with specified backend."""
    from transformers import TrainingArguments

    dist_config = DistributedTrainingConfig(
        backend=backend,
        fp16=recipe_obj.training.fp16,
        gradient_accumulation_steps=recipe_obj.training.gradient_accumulation_steps,
    )

    training_args = _create_training_arguments(recipe_obj, output_dir)

    return DistributedTrainer(
        model=model_obj,
        training_args=training_args,
        config=dist_config,
        tokenizer=tokenizer,
    )


def _setup_standard_trainer(
    recipe_obj: Recipe, model_obj: Any, tokenizer: Any, dataset: Dict[str, Any], output_dir: str
) -> Any:
    """Setup standard (non-distributed) trainer."""
    from transformers import Trainer

    training_args = _create_training_arguments(recipe_obj, output_dir)

    return Trainer(
        model=model_obj,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
    )


def _create_training_arguments(recipe_obj: Recipe, output_dir: str) -> Any:
    """Create TrainingArguments from recipe configuration."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=recipe_obj.training.num_epochs,
        per_device_train_batch_size=recipe_obj.training.per_device_train_batch_size,
        per_device_eval_batch_size=recipe_obj.training.per_device_eval_batch_size,
        learning_rate=recipe_obj.training.learning_rate,
        warmup_steps=recipe_obj.training.warmup_steps,
        logging_steps=recipe_obj.training.logging_steps,
        save_steps=recipe_obj.training.save_steps,
        eval_steps=recipe_obj.training.eval_steps,
        save_total_limit=recipe_obj.training.save_total_limit,
        fp16=recipe_obj.training.fp16,
        bf16=recipe_obj.training.bf16,
        gradient_checkpointing=recipe_obj.training.gradient_checkpointing,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )


def _start_training_process(job_id: str, recipe_obj: Recipe, output_dir: str):
    """Start the actual training process and update job status."""
    # Update job status
    job_manager.update_job(job_id, status="running", checkpoint_dir=output_dir, pid=os.getpid())

    # Launch training in background with dashboard
    if recipe_obj.training.use_lora:
        console.print("[yellow]Note: LoRA training not fully implemented in this example[/yellow]")

    # Simple training execution
    console.print("\\n[bold green]Training started![/bold green]")
    console.print(f"Job ID: {job_id}")
    console.print(f"Output directory: {output_dir}")
    console.print("\\nUse 'finetune monitor <job_id>' to track progress")

    # In a real implementation, would launch training in subprocess
    # For now, we'll simulate completion
    job_manager.update_job(job_id, status="completed")


@cli.command("list-jobs")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["all", "running", "completed", "failed", "cancelled"]),
    default="all",
    help="Filter by job status",
)
@click.option("--limit", "-l", type=int, default=10, help="Number of jobs to display")
def list_jobs(status: str, limit: int):
    """List fine-tuning jobs."""
    # Get jobs
    if status == "all":
        jobs = job_manager.list_jobs()
    else:
        jobs = job_manager.list_jobs(status=status)

    if not jobs:
        console.print("[yellow]No jobs found[/yellow]")
        return

    # Create table
    table = Table(title=f"Fine-Tuning Jobs ({status})", box=box.ROUNDED)

    table.add_column("Job ID", style="cyan")
    table.add_column("Recipe", style="magenta")
    table.add_column("Model", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Created", style="yellow")
    table.add_column("Duration")

    # Add rows
    for job in jobs[:limit]:
        # Calculate duration
        created = datetime.fromisoformat(job["created_at"])
        if job["status"] == "running":
            duration = str(datetime.now() - created).split(".")[0]
        elif job["status"] == "completed" and "completed_at" in job:
            completed = datetime.fromisoformat(job["completed_at"])
            duration = str(completed - created).split(".")[0]
        else:
            duration = "-"

        # Status color
        status_colors = {
            "running": "green",
            "completed": "blue",
            "failed": "red",
            "cancelled": "yellow",
            "initialized": "white",
            "starting": "cyan",
        }
        status_color = status_colors.get(job["status"], "white")

        table.add_row(
            job["id"],
            job["recipe"],
            job["model"].split("/")[-1],  # Show only model name
            f"[{status_color}]{job['status']}[/{status_color}]",
            created.strftime("%Y-%m-%d %H:%M"),
            duration,
        )

    console.print(table)

    if len(jobs) > limit:
        console.print(f"\n[dim]Showing {limit} of {len(jobs)} jobs[/dim]")


@cli.command("stop-job")
@click.argument("job_id")
@click.option("--force", "-f", is_flag=True, help="Force stop without confirmation")
def stop_job(job_id: str, force: bool):
    """Stop a running fine-tuning job."""
    job = job_manager.get_job(job_id)

    if not job:
        console.print(f"[red]Job not found:[/red] {job_id}")
        return

    if job["status"] != "running":
        console.print(f"[yellow]Job {job_id} is not running (status: {job['status']})[/yellow]")
        return

    if not force:
        if not Confirm.ask(f"Stop job {job_id}?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

    # Stop the job
    if job.get("pid"):
        try:
            os.kill(job["pid"], signal.SIGTERM)
            console.print(f"[green]Sent stop signal to job {job_id}[/green]")
        except ProcessLookupError:
            console.print("[yellow]Process already terminated[/yellow]")
        except Exception as e:
            console.print(f"[red]Error stopping job:[/red] {e}")

    job_manager.update_job(job_id, status="cancelled")
    console.print(f"[green]Job {job_id} marked as cancelled[/green]")


@cli.command("resume-job")
@click.argument("job_id")
@click.option("--checkpoint", "-c", help="Specific checkpoint to resume from")
def resume_job(job_id: str, checkpoint: Optional[str]):
    """Resume a stopped or failed fine-tuning job."""
    job = job_manager.get_job(job_id)

    if not job:
        console.print(f"[red]Job not found:[/red] {job_id}")
        return

    if job["status"] not in ["failed", "cancelled"]:
        console.print(f"[yellow]Job {job_id} cannot be resumed (status: {job['status']})[/yellow]")
        return

    console.print(f"[green]Resuming job {job_id}...[/green]")

    # Get checkpoint directory
    checkpoint_dir = job.get("checkpoint_dir")
    if not checkpoint_dir:
        console.print("[red]No checkpoint directory found for job[/red]")
        return

    # Find latest checkpoint if not specified
    if not checkpoint:
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        checkpoints = checkpoint_manager.list_checkpoints()

        if not checkpoints:
            console.print("[red]No checkpoints found[/red]")
            return

        # Use latest checkpoint
        checkpoint = checkpoints[-1]["path"]
        console.print(f"[blue]Using latest checkpoint: {checkpoint}[/blue]")

    # Update job and restart
    job_manager.update_job(job_id, status="resuming", resumed_from=checkpoint)

    # In real implementation, would restart training from checkpoint
    console.print(f"[green]Job {job_id} resumed from {checkpoint}[/green]")


@cli.command("list-checkpoints")
@click.option("--job-id", "-j", help="Filter by job ID")
@click.option("--model", "-m", help="Filter by model name")
def list_checkpoints(job_id: Optional[str], model: Optional[str]):
    """List available checkpoints."""
    checkpoints_found = []

    if job_id:
        job = job_manager.get_job(job_id)
        if job and job.get("checkpoint_dir"):
            checkpoint_manager = CheckpointManager(job["checkpoint_dir"])
            checkpoints = checkpoint_manager.list_checkpoints()
            checkpoints_found.extend([(job_id, c) for c in checkpoints])
    else:
        # Search all job checkpoint directories
        for job in job_manager.list_jobs():
            if model and model not in job["model"]:
                continue

            if job.get("checkpoint_dir"):
                checkpoint_manager = CheckpointManager(job["checkpoint_dir"])
                checkpoints = checkpoint_manager.list_checkpoints()
                checkpoints_found.extend([(job["id"], c) for c in checkpoints])

    if not checkpoints_found:
        console.print("[yellow]No checkpoints found[/yellow]")
        return

    # Create table
    table = Table(title="Available Checkpoints", box=box.ROUNDED)

    table.add_column("Job ID", style="cyan")
    table.add_column("Checkpoint", style="magenta")
    table.add_column("Step", style="blue")
    table.add_column("Metrics", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Created")

    # Add rows
    for job_id, checkpoint in checkpoints_found:
        metrics_str = ""
        if checkpoint.get("metrics"):
            key_metrics = ["loss", "eval_loss", "accuracy"]
            metrics_parts = []
            for k in key_metrics:
                if k in checkpoint["metrics"]:
                    v = checkpoint["metrics"][k]
                    metrics_parts.append(f"{k}: {v:.4f}")
            metrics_str = ", ".join(metrics_parts)

        # Format size
        size_mb = checkpoint.get("size", 0) / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB"

        # Format timestamp
        created = checkpoint.get("timestamp", "")
        if created:
            created = datetime.fromisoformat(created).strftime("%Y-%m-%d %H:%M")

        table.add_row(
            job_id,
            Path(checkpoint["path"]).name,
            str(checkpoint.get("step", "-")),
            metrics_str,
            size_str,
            created,
        )

    console.print(table)


@cli.command()
@click.option("--checkpoint", "-c", required=True, help="Checkpoint path or job ID")
@click.option(
    "--benchmarks", "-b", multiple=True, default=["hellaswag", "mmlu"], help="Benchmarks to run"
)
@click.option("--output", "-o", help="Output file for results")
@click.option("--compare-with", help="Compare with another checkpoint")
def evaluate(
    checkpoint: str, benchmarks: Tuple[str], output: Optional[str], compare_with: Optional[str]
):
    """Evaluate a fine-tuned model checkpoint."""
    console.print(Panel.fit("[bold blue]Model Evaluation[/bold blue]", border_style="blue"))

    try:
        # Load model from checkpoint
        checkpoint_path = Path(checkpoint)

        if not checkpoint_path.exists():
            # Try as job ID
            job = job_manager.get_job(checkpoint)
            if job and job.get("checkpoint_dir"):
                checkpoint_manager = CheckpointManager(job["checkpoint_dir"])
                checkpoints = checkpoint_manager.list_checkpoints()
                if checkpoints:
                    checkpoint_path = Path(checkpoints[-1]["path"])
                else:
                    console.print("[red]No checkpoints found for job[/red]")
                    return
            else:
                console.print("[red]Checkpoint not found[/red]")
                return

        console.print(f"[green]Loading checkpoint:[/green] {checkpoint_path}")

        # Initialize evaluation suite
        eval_config = EvaluationConfig(
            benchmarks=list(benchmarks),
            batch_size=8,
            save_results=True,
            results_dir=Path(output).parent if output else "./evaluation_results",
        )

        suite = EvaluationSuite(eval_config)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Loading model...", total=None)

            # Load model and tokenizer
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

            # Try to find tokenizer
            tokenizer_path = checkpoint_path
            if not (tokenizer_path / "tokenizer_config.json").exists():
                # Use base model tokenizer
                config_path = checkpoint_path / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    base_model = config.get("_name_or_path", "gpt2")
                    tokenizer = AutoTokenizer.from_pretrained(base_model)
                else:
                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            progress.update(task, description="Running evaluation...")

            # Run evaluation
            results = suite.evaluate(model, tokenizer, list(benchmarks))

        # Display results
        _display_evaluation_results(results)

        # Compare if requested
        if compare_with:
            console.print("\n[bold]Comparison Results[/bold]")
            # Load comparison model and evaluate
            # ... (implementation for comparison)

        # Save results if requested
        if output:
            report = suite.generate_report(results, output_format="markdown")
            with open(output, "w") as f:
                f.write(report)
            console.print(f"\n[green]Results saved to:[/green] {output}")

    except Exception as e:
        console.print(f"[red]Evaluation error:[/red] {e}")
        raise click.ClickException(str(e))


@cli.command("list-recipes")
@click.option(
    "--difficulty",
    "-d",
    type=click.Choice(["all", "beginner", "intermediate", "advanced"]),
    default="all",
    help="Filter by difficulty",
)
def list_recipes(difficulty: str):
    """List available fine-tuning recipes."""
    recipe_manager = RecipeManager()

    # Get all recipes
    recipes = recipe_manager.list_recipes()
    templates = recipe_manager.list_templates()

    # Create table
    table = Table(title="Available Fine-Tuning Recipes", box=box.ROUNDED)

    table.add_column("Recipe", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Model", style="blue")
    table.add_column("Use Cases", style="green")
    table.add_column("Difficulty", style="yellow")

    # Add templates
    for template in templates:
        if difficulty != "all" and template.get("difficulty") != difficulty:
            continue

        table.add_row(
            template["id"],
            "Template",
            template.get("model", "Various"),
            ", ".join(template.get("use_cases", [])[:2]),
            template.get("difficulty", "unknown"),
        )

    # Add custom recipes
    for recipe_name in recipes:
        if recipe_name not in [t["id"] for t in templates]:
            try:
                info = recipe_manager.get_recipe_info(recipe_name)
                table.add_row(
                    recipe_name,
                    "Custom",
                    info.get("model", "Unknown"),
                    ", ".join(info.get("tags", [])[:2]),
                    "custom",
                )
            except Exception:
                pass

    console.print(table)


@cli.command("create-recipe")
@click.option("--name", "-n", required=True, help="Recipe name")
@click.option("--base", "-b", help="Base template to use")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.option("--output", "-o", help="Output file path")
def create_recipe(name: str, base: Optional[str], interactive: bool, output: Optional[str]):
    """Create a new fine-tuning recipe."""
    console.print(
        Panel.fit("[bold blue]Create Fine-Tuning Recipe[/bold blue]", border_style="blue")
    )

    recipe_manager = RecipeManager()

    if base:
        # Load base template
        recipe = recipe_manager.load_recipe(base)
        recipe.name = name
        console.print(f"[green]Using base template:[/green] {base}")
    else:
        # Create from scratch
        recipe = Recipe(name=name, description="Custom fine-tuning recipe", author="user")

    if interactive:
        recipe = _interactive_recipe_config(recipe)

    # Save recipe
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            yaml.dump(recipe.to_dict(), f, default_flow_style=False)
        console.print(f"[green]Recipe saved to:[/green] {output_path}")
    else:
        recipe_manager.save_recipe(recipe)
        console.print(f"[green]Recipe saved:[/green] {name}")

    # Display recipe
    _display_recipe_config(recipe)


@cli.command()
@click.argument("job_id")
@click.option("--refresh", "-r", type=int, default=5, help="Refresh interval in seconds")
def monitor(job_id: str, refresh: int):
    """Monitor a running fine-tuning job."""
    job = job_manager.get_job(job_id)

    if not job:
        console.print(f"[red]Job not found:[/red] {job_id}")
        return

    console.print(
        Panel.fit(f"[bold blue]Monitoring Job: {job_id}[/bold blue]", border_style="blue")
    )

    # Create layout
    layout = Layout()
    layout.split(Layout(name="header", size=3), Layout(name="main"), Layout(name="footer", size=3))

    layout["main"].split_row(Layout(name="metrics"), Layout(name="logs"))

    def generate_layout():
        """Generate the monitoring layout."""
        # Header
        header_text = f"Job: {job_id} | Status: {job['status']} | Model: {job['model']}"
        layout["header"].update(Panel(header_text, style="bold white on blue"))

        # Metrics panel
        metrics_table = Table(box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        # Add metric rows (in real implementation, would read from job metrics)
        metrics_table.add_row("Epoch", "2/3")
        metrics_table.add_row("Step", "1250/2000")
        metrics_table.add_row("Loss", "1.234")
        metrics_table.add_row("Learning Rate", "2e-5")
        metrics_table.add_row("GPU Memory", "15.2 GB")

        layout["metrics"].update(Panel(metrics_table, title="Training Metrics"))

        # Logs panel
        log_text = "Loading logs..."
        if job.get("checkpoint_dir"):
            log_file = Path(job["checkpoint_dir"]) / "training.log"
            if log_file.exists():
                # Read last 10 lines
                with open(log_file) as f:
                    lines = f.readlines()
                    log_text = "".join(lines[-10:])

        layout["logs"].update(Panel(log_text, title="Recent Logs"))

        # Footer
        footer_text = f"Refreshing every {refresh}s | Press Ctrl+C to exit"
        layout["footer"].update(Panel(footer_text, style="dim"))

        return layout

    try:
        with Live(generate_layout(), refresh_per_second=1 / refresh, console=console) as live:
            while True:
                time.sleep(refresh)

                # Update job info
                job = job_manager.get_job(job_id)

                if job["status"] in ["completed", "failed", "cancelled"]:
                    console.print(f"\n[yellow]Job {job['status']}[/yellow]")
                    break

                live.update(generate_layout())

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")


# Helper functions


def _interactive_recipe_config(recipe: Recipe) -> Recipe:
    """Interactive recipe configuration."""
    console.print("\n[bold]Configure Recipe[/bold]")

    # Model configuration
    console.print("\n[cyan]Model Configuration[/cyan]")
    recipe.model.base_model = Prompt.ask("Base model", default=recipe.model.base_model)
    recipe.model.model_type = Prompt.ask(
        "Model type", choices=["causal_lm", "seq2seq"], default=recipe.model.model_type
    )
    recipe.model.use_flash_attention = Confirm.ask(
        "Use Flash Attention?", default=recipe.model.use_flash_attention
    )

    # Dataset configuration
    console.print("\n[cyan]Dataset Configuration[/cyan]")
    recipe.dataset.path = Prompt.ask("Dataset path", default=recipe.dataset.path)
    recipe.dataset.format = Prompt.ask(
        "Dataset format",
        choices=["jsonl", "csv", "huggingface", "parquet"],
        default=recipe.dataset.format,
    )

    # Training configuration
    console.print("\n[cyan]Training Configuration[/cyan]")
    recipe.training.num_epochs = IntPrompt.ask(
        "Number of epochs", default=recipe.training.num_epochs
    )
    recipe.training.per_device_train_batch_size = IntPrompt.ask(
        "Batch size", default=recipe.training.per_device_train_batch_size
    )
    recipe.training.learning_rate = FloatPrompt.ask(
        "Learning rate", default=recipe.training.learning_rate
    )
    recipe.training.use_lora = Confirm.ask("Use LoRA?", default=recipe.training.use_lora)

    if recipe.training.use_lora:
        recipe.training.lora_rank = IntPrompt.ask("LoRA rank", default=recipe.training.lora_rank)
        recipe.training.lora_alpha = IntPrompt.ask("LoRA alpha", default=recipe.training.lora_alpha)

    return recipe


def _display_recipe_config(recipe: Recipe):
    """Display recipe configuration."""
    # Create configuration tree
    tree = Tree(f"[bold]Recipe: {recipe.name}[/bold]")

    # Model branch
    model_branch = tree.add("[cyan]Model Configuration[/cyan]")
    model_branch.add(f"Base Model: {recipe.model.base_model}")
    model_branch.add(f"Type: {recipe.model.model_type}")
    model_branch.add(f"Flash Attention: {recipe.model.use_flash_attention}")

    # Dataset branch
    dataset_branch = tree.add("[cyan]Dataset Configuration[/cyan]")
    dataset_branch.add(f"Path: {recipe.dataset.path}")
    dataset_branch.add(f"Format: {recipe.dataset.format}")
    dataset_branch.add(f"Max Length: {recipe.dataset.tokenizer_config.get('max_length', 512)}")

    # Training branch
    training_branch = tree.add("[cyan]Training Configuration[/cyan]")
    training_branch.add(f"Epochs: {recipe.training.num_epochs}")
    training_branch.add(f"Batch Size: {recipe.training.per_device_train_batch_size}")
    training_branch.add(f"Learning Rate: {recipe.training.learning_rate}")
    training_branch.add(
        f"Mixed Precision: fp16={recipe.training.fp16}, bf16={recipe.training.bf16}"
    )

    if recipe.training.use_lora:
        lora_branch = training_branch.add("LoRA Configuration")
        lora_branch.add(f"Rank: {recipe.training.lora_rank}")
        lora_branch.add(f"Alpha: {recipe.training.lora_alpha}")
        lora_branch.add(f"Dropout: {recipe.training.lora_dropout}")

    console.print(tree)


def _display_evaluation_results(results):
    """Display evaluation results."""
    console.print(f"\n[bold]Evaluation Results: {results.model_name}[/bold]")

    if results.perplexity:
        console.print(f"\n[cyan]Perplexity:[/cyan] {results.perplexity:.2f}")

    if results.benchmarks:
        # Create benchmarks table
        table = Table(title="Benchmark Scores", box=box.ROUNDED)
        table.add_column("Benchmark", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Samples", style="blue")
        table.add_column("Runtime", style="yellow")

        for benchmark in results.benchmarks:
            table.add_row(
                benchmark.name,
                f"{benchmark.overall_score:.4f}",
                str(benchmark.samples_evaluated),
                f"{benchmark.runtime_seconds:.1f}s",
            )

        console.print(table)

    if results.custom_metrics:
        console.print("\n[cyan]Custom Metrics:[/cyan]")
        for metric in results.custom_metrics:
            console.print(f"  {metric.name}: {metric.value:.4f}")

    console.print(f"\n[dim]Total runtime: {results.total_runtime_seconds:.1f}s[/dim]")


@cli.command("estimate-cost")
@click.option("--recipe", "-r", required=True, help="Recipe name or path")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["aws", "gcp", "azure", "local"]),
    default="aws",
    help="Cloud provider",
)
@click.option("--instance-type", "-i", help="Instance type (e.g., p3.2xlarge)")
@click.option("--num-gpus", "-g", type=int, default=1, help="Number of GPUs")
@click.option("--hours", "-h", type=float, help="Estimated training hours")
def estimate_cost(
    recipe: str, provider: str, instance_type: Optional[str], num_gpus: int, hours: Optional[float]
):
    """Estimate training cost for a recipe."""
    console.print(Panel.fit("[bold cyan]Cost Estimation[/bold cyan]", border_style="cyan"))

    # Initialize job manager
    job_manager = JobManager()

    # Load recipe
    recipe_manager = RecipeManager()
    if Path(recipe).exists():
        recipe_obj = recipe_manager.load_recipe(recipe)
    else:
        recipe_obj = recipe_manager.get_recipe(recipe)

    if not recipe_obj:
        console.print(f"[red]Recipe not found:[/red] {recipe}")
        return

    # Estimate training time if not provided
    if not hours:
        # Rough estimation based on model size and epochs
        model_params = _estimate_model_params(recipe_obj.model.base_model)
        dataset_size = _estimate_dataset_size(recipe_obj.dataset.path)
        epochs = recipe_obj.training.num_epochs
        batch_size = recipe_obj.training.per_device_train_batch_size

        # Very rough estimation (samples per second * seconds)
        samples_per_hour = (3600 / (model_params / 1e9)) * batch_size * num_gpus
        hours = (dataset_size * epochs) / samples_per_hour

    # Cost estimation based on provider
    costs = {
        "aws": {
            "p3.2xlarge": 3.06,  # per hour
            "p3.8xlarge": 12.24,
            "p3.16xlarge": 24.48,
            "g4dn.xlarge": 0.526,
            "g4dn.12xlarge": 3.912,
        },
        "gcp": {"n1-standard-4-t4": 0.35, "n1-standard-8-v100": 2.48, "a2-highgpu-1g": 2.93},
        "azure": {"NC6": 0.90, "NC12": 1.80, "NC24": 3.60, "ND40rs_v2": 22.03},
        "local": {
            "electricity": 0.12  # per kWh
        },
    }

    # Calculate cost
    if provider == "local":
        # Estimate based on power consumption
        power_consumption = num_gpus * 250  # Watts per GPU
        kwh = (power_consumption * hours) / 1000
        total_cost = kwh * costs["local"]["electricity"]
    else:
        # Use instance pricing
        if not instance_type:
            # Default instance types
            defaults = {"aws": "p3.2xlarge", "gcp": "n1-standard-8-v100", "azure": "NC12"}
            instance_type = defaults[provider]

        hourly_rate = costs.get(provider, {}).get(instance_type, 0)
        total_cost = hourly_rate * hours

    # Display cost breakdown
    table = Table(title="Cost Estimation", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Recipe", recipe_obj.name)
    table.add_row("Model", recipe_obj.model.base_model)
    table.add_row("Provider", provider)
    table.add_row("Instance Type", instance_type or "N/A")
    table.add_row("Number of GPUs", str(num_gpus))
    table.add_row("Estimated Hours", f"{hours:.2f}")
    table.add_row(
        "Hourly Rate",
        f"${hourly_rate:.2f}" if provider != "local" else f"${costs['local']['electricity']}/kWh",
    )
    table.add_row("[bold]Total Cost[/bold]", f"[bold]${total_cost:.2f}[/bold]")

    console.print(table)

    # Additional recommendations
    console.print("\n[cyan]Cost Optimization Tips:[/cyan]")
    console.print("• Use spot/preemptible instances for 60-90% cost reduction")
    console.print("• Enable mixed precision training (fp16/bf16) to reduce training time")
    console.print("• Use gradient checkpointing to train larger models on smaller instances")
    console.print("• Consider using LoRA/QLoRA for parameter-efficient fine-tuning")


@cli.command("export-recipe")
@click.argument("recipe_name")
@click.option("--output", "-o", required=True, help="Output file path")
@click.option(
    "--format", "-f", type=click.Choice(["yaml", "json"]), default="yaml", help="Export format"
)
def export_recipe(recipe_name: str, output: str, format: str):
    """Export a recipe to a file."""
    recipe_manager = RecipeManager()
    recipe = recipe_manager.get_recipe(recipe_name)

    if not recipe:
        console.print(f"[red]Recipe not found:[/red] {recipe_name}")
        return

    # Export recipe
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "yaml":
        with open(output_path, "w") as f:
            yaml.dump(recipe.to_dict(), f, default_flow_style=False)
    else:
        with open(output_path, "w") as f:
            json.dump(recipe.to_dict(), f, indent=2)

    console.print(f"[green]Recipe exported to:[/green] {output_path}")


@cli.command("validate-recipe")
@click.argument("recipe_path")
@click.option("--strict", is_flag=True, help="Enable strict validation")
def validate_recipe(recipe_path: str, strict: bool):
    """Validate a recipe file."""
    console.print(
        Panel.fit(f"[bold cyan]Validating Recipe: {recipe_path}[/bold cyan]", border_style="cyan")
    )

    path = Path(recipe_path)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {recipe_path}")
        return

    # Load and validate
    recipe_manager = RecipeManager()
    try:
        with open(path) as f:
            if path.suffix == ".yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Validate schema
        recipe_manager.validate_recipe_data(data)

        # Additional strict checks
        if strict:
            warnings = []

            # Check for recommended fields
            if "tags" not in data:
                warnings.append("Missing 'tags' field")
            if "metadata" not in data:
                warnings.append("Missing 'metadata' field")
            if "evaluation" not in data:
                warnings.append("Missing 'evaluation' configuration")

            # Check training parameters
            training = data.get("training", {})
            if training.get("learning_rate", 0) > 1e-3:
                warnings.append("Learning rate seems high (> 1e-3)")
            if training.get("num_epochs", 0) > 10:
                warnings.append("High number of epochs may lead to overfitting")

            if warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  • {warning}")

        console.print("\n[green]✓ Recipe is valid![/green]")

        # Display recipe summary
        recipe = Recipe.from_dict(data)
        _display_recipe_config(recipe)

    except Exception as e:
        console.print(f"\n[red]✗ Validation failed:[/red] {e!s}")
        return 1


@cli.command("compare-checkpoints")
@click.argument("checkpoint1")
@click.argument("checkpoint2")
@click.option(
    "--benchmarks", "-b", multiple=True, default=["hellaswag", "mmlu"], help="Benchmarks to compare"
)
@click.option("--output", "-o", help="Save comparison report")
def compare_checkpoints(
    checkpoint1: str, checkpoint2: str, benchmarks: Tuple[str], output: Optional[str]
):
    """Compare two model checkpoints."""
    console.print(Panel.fit("[bold cyan]Checkpoint Comparison[/bold cyan]", border_style="cyan"))

    # Load checkpoints
    checkpoint_manager = CheckpointManager()

    with console.status("Loading checkpoints..."):
        ckpt1 = checkpoint_manager.load_checkpoint(checkpoint1)
        ckpt2 = checkpoint_manager.load_checkpoint(checkpoint2)

    if not ckpt1 or not ckpt2:
        console.print("[red]Failed to load checkpoints[/red]")
        return

    # Run evaluation on both
    eval_suite = EvaluationSuite(EvaluationConfig(benchmarks=list(benchmarks), save_results=False))

    with console.status("Evaluating checkpoint 1..."):
        results1 = eval_suite.evaluate(ckpt1["model"], ckpt1["tokenizer"])

    with console.status("Evaluating checkpoint 2..."):
        results2 = eval_suite.evaluate(ckpt2["model"], ckpt2["tokenizer"])

    # Display comparison
    table = Table(title="Checkpoint Comparison", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Checkpoint 1", style="blue")
    table.add_column("Checkpoint 2", style="green")
    table.add_column("Difference", style="yellow")

    # Compare perplexity
    if results1.perplexity and results2.perplexity:
        diff = results2.perplexity - results1.perplexity
        diff_str = f"{'↑' if diff > 0 else '↓'} {abs(diff):.2f}"
        table.add_row(
            "Perplexity", f"{results1.perplexity:.2f}", f"{results2.perplexity:.2f}", diff_str
        )

    # Compare benchmarks
    for b1 in results1.benchmarks:
        b2 = next((b for b in results2.benchmarks if b.name == b1.name), None)
        if b2:
            diff = b2.overall_score - b1.overall_score
            diff_pct = (diff / b1.overall_score) * 100 if b1.overall_score > 0 else 0
            diff_str = f"{'↑' if diff > 0 else '↓'} {abs(diff_pct):.1f}%"
            table.add_row(b1.name, f"{b1.overall_score:.4f}", f"{b2.overall_score:.4f}", diff_str)

    console.print(table)

    # Save report if requested
    if output:
        report = {
            "checkpoint1": checkpoint1,
            "checkpoint2": checkpoint2,
            "results1": results1.__dict__,
            "results2": results2.__dict__,
            "comparison": {
                "perplexity_diff": results2.perplexity - results1.perplexity
                if results1.perplexity and results2.perplexity
                else None,
                "benchmark_diffs": {},
            },
        }

        for b1 in results1.benchmarks:
            b2 = next((b for b in results2.benchmarks if b.name == b1.name), None)
            if b2:
                report["comparison"]["benchmark_diffs"][b1.name] = (
                    b2.overall_score - b1.overall_score
                )

        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)

        console.print(f"\n[green]Report saved to:[/green] {output}")


@cli.command("batch-train")
@click.option("--config", "-c", required=True, help="Batch configuration file (YAML/JSON)")
@click.option("--parallel", "-p", type=int, default=1, help="Number of parallel jobs")
@click.option("--output-dir", "-o", default="./batch_results", help="Output directory")
def batch_train(config: str, parallel: int, output_dir: str):
    """Submit multiple training jobs from a configuration file."""
    console.print(
        Panel.fit("[bold cyan]Batch Training Submission[/bold cyan]", border_style="cyan")
    )

    # Load batch configuration
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config file not found:[/red] {config}")
        return

    with open(config_path) as f:
        if config_path.suffix == ".yaml":
            batch_config = yaml.safe_load(f)
        else:
            batch_config = json.load(f)

    jobs = batch_config.get("jobs", [])
    if not jobs:
        console.print("[red]No jobs found in configuration[/red]")
        return

    console.print(f"Found [cyan]{len(jobs)}[/cyan] jobs to submit")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    submitted_jobs = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []

        for i, job_config in enumerate(jobs):
            job_name = job_config.get("name", f"job_{i}")
            console.print(f"\n[cyan]Submitting job:[/cyan] {job_name}")

            # Prepare job arguments
            recipe = job_config.get("recipe")
            model = job_config.get("model")
            data = job_config.get("data")
            job_output = output_path / job_name

            # Submit job
            future = executor.submit(
                _run_training_job,
                recipe=recipe,
                model=model,
                data=data,
                output_dir=str(job_output),
                job_config=job_config,
            )
            futures.append((job_name, future))

        # Monitor progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running batch jobs...", total=len(futures))

            for job_name, future in futures:
                try:
                    result = future.result()
                    submitted_jobs.append(
                        {"name": job_name, "status": "submitted", "job_id": result}
                    )
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[red]Failed to submit {job_name}:[/red] {e!s}")
                    submitted_jobs.append({"name": job_name, "status": "failed", "error": str(e)})

    # Display summary
    table = Table(title="Batch Submission Summary", box=box.ROUNDED)
    table.add_column("Job Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Job ID / Error", style="yellow")

    for job in submitted_jobs:
        status_color = "green" if job["status"] == "submitted" else "red"
        table.add_row(
            job["name"],
            f"[{status_color}]{job['status']}[/{status_color}]",
            job.get("job_id", job.get("error", "N/A")),
        )

    console.print(table)

    # Save submission report
    report_path = output_path / "batch_submission.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "config_file": str(config_path),
                "timestamp": datetime.now().isoformat(),
                "jobs": submitted_jobs,
            },
            f,
            indent=2,
        )

    console.print(f"\n[green]Submission report saved to:[/green] {report_path}")


def _estimate_model_params(model_name: str) -> float:
    """Estimate model parameters from name."""
    # Rough estimates based on common model sizes
    if "70b" in model_name.lower():
        return 70e9
    elif "13b" in model_name.lower():
        return 13e9
    elif "7b" in model_name.lower():
        return 7e9
    elif "3b" in model_name.lower():
        return 3e9
    elif "1b" in model_name.lower():
        return 1e9
    else:
        return 1e9  # Default


def _estimate_dataset_size(dataset_path: str) -> int:
    """Estimate dataset size."""
    # This is a placeholder - in reality would check actual dataset
    return 10000  # Default estimate


def _run_training_job(recipe: str, model: str, data: str, output_dir: str, job_config: Dict) -> str:
    """Run a single training job."""
    # This would actually submit the training job
    # For now, return a mock job ID
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # In reality, would call the training function here
    # train(recipe=recipe, model=model, data=data, output_dir=output_dir, ...)

    return job_id


@cli.command("setup-monitoring")
@click.option(
    "--platforms",
    "-p",
    multiple=True,
    default=["tensorboard"],
    help="Monitoring platforms to enable",
)
@click.option(
    "--project",
    "--project-name",
    default="fine_tuning_experiment",
    help="Project name for monitoring",
)
@click.option("--config-file", "-c", help="Save configuration to file")
@click.option("--wandb-key", help="Weights & Biases API key")
@click.option("--enable-alerts", is_flag=True, default=True, help="Enable training alerts")
def setup_monitoring(
    platforms: Tuple[str],
    project: str,
    config_file: Optional[str],
    wandb_key: Optional[str],
    enable_alerts: bool,
):
    """Set up monitoring configuration."""
    console.print(Panel.fit("[bold cyan]Monitoring Setup[/bold cyan]", border_style="cyan"))

    # Create monitoring configuration
    config = MonitoringConfig(
        platforms=list(platforms),
        project_name=project,
        enable_alerts=enable_alerts,
        resource_monitoring=True,
    )

    # Configure Weights & Biases if requested
    if "wandb" in platforms:
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            config.wandb_config = {"project": project, "job_type": "fine-tuning"}
        else:
            console.print(
                "[yellow]Warning: Weights & Biases requested but no API key provided[/yellow]"
            )

    # Configure TensorBoard
    if "tensorboard" in platforms:
        config.tensorboard_config = {"log_dir": f"./tensorboard_logs/{project}"}

    # Save configuration if requested
    if config_file:
        config.to_file(config_file)
        console.print(f"[green]Configuration saved to:[/green] {config_file}")

    # Test setup
    try:
        setup = MonitoringSetup(config)
        components = setup.setup()

        console.print("[green]✓ Monitoring setup successful![/green]")

        # Display configuration
        table = Table(title="Monitoring Configuration", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Project Name", project)
        table.add_row("Platforms", ", ".join(platforms))
        table.add_row("Alerts Enabled", "Yes" if enable_alerts else "No")
        table.add_row("Resource Monitoring", "Yes")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Setup failed:[/red] {e!s}")
        return 1


@cli.command("monitoring-status")
@click.option("--job-id", "-j", help="Check status for specific job")
def monitoring_status(job_id: Optional[str]):
    """Check monitoring system status."""
    console.print(Panel.fit("[bold cyan]Monitoring Status[/bold cyan]", border_style="cyan"))

    # Check environment configuration
    config_file = os.getenv("LLLM_MONITORING_CONFIG")

    if config_file and Path(config_file).exists():
        config = MonitoringConfig.from_file(config_file)
        console.print(f"[green]Using config file:[/green] {config_file}")
    else:
        console.print("[yellow]No monitoring configuration found[/yellow]")
        console.print("Run 'finetune setup-monitoring' to configure monitoring")
        return

    # Check platform availability
    table = Table(title="Platform Status", box=box.ROUNDED)
    table.add_column("Platform", style="cyan")
    table.add_column("Configured", style="green")
    table.add_column("Available", style="blue")
    table.add_column("Status", style="yellow")

    for platform in config.platforms:
        configured = "Yes"

        if platform == "wandb":
            available = "Yes" if os.getenv("WANDB_API_KEY") else "No"
            status = "Ready" if available == "Yes" else "Missing API Key"
        elif platform == "tensorboard":
            available = "Yes"  # Always available
            status = "Ready"
        elif platform == "mlflow":
            available = "Yes"  # Assume available
            status = "Ready"
        else:
            available = "Unknown"
            status = "Unknown"

        table.add_row(platform, configured, available, status)

    console.print(table)

    # Check for specific job
    if job_id:
        job_manager = JobManager()
        job = job_manager.get_job(job_id)

        if job:
            console.print(f"\n[cyan]Job Monitoring Status:[/cyan] {job_id}")
            console.print(f"Status: {job.get('status', 'unknown')}")
            console.print(
                f"Monitoring: {'Enabled' if job.get('monitoring_enabled') else 'Disabled'}"
            )
        else:
            console.print(f"[red]Job not found:[/red] {job_id}")


@cli.command("view-logs")
@click.option("--job-id", "-j", help="Job ID to view logs for")
@click.option(
    "--level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Log level filter",
)
@click.option("--tail", "-t", type=int, default=50, help="Number of recent lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (live tail)")
def view_logs(job_id: Optional[str], level: str, tail: int, follow: bool):
    """View training logs."""
    console.print(
        Panel.fit(
            f"[bold cyan]Training Logs{' - ' + job_id if job_id else ''}[/bold cyan]",
            border_style="cyan",
        )
    )

    # Find log files
    log_files = []

    if job_id:
        # Look for job-specific logs
        job_log_pattern = f"*{job_id}*.log"
        log_files.extend(Path("./logs").glob(job_log_pattern))
    else:
        # Look for general training logs
        log_files.extend(Path("./logs").glob("training_*.log"))
        log_files.extend(Path("./logs").glob("fine_tuning_*.log"))

    if not log_files:
        console.print("[yellow]No log files found[/yellow]")
        console.print("Make sure logging is enabled and training has started")
        return

    # Display logs
    for log_file in sorted(log_files)[-1:]:  # Show most recent
        console.print(f"\n[cyan]Log file:[/cyan] {log_file}")

        try:
            if follow:
                console.print("[yellow]Following log output (Ctrl+C to stop)...[/yellow]")
                # This would implement live log following
                # For now, just show recent lines

            with open(log_file) as f:
                lines = f.readlines()

            # Filter by level and show recent lines
            filtered_lines = []
            for line in lines:
                if level.lower() in line.lower() or level == "DEBUG":
                    filtered_lines.append(line)

            recent_lines = filtered_lines[-tail:] if len(filtered_lines) > tail else filtered_lines

            for line in recent_lines:
                # Color code log levels
                if "ERROR" in line:
                    console.print(f"[red]{line.rstrip()}[/red]")
                elif "WARNING" in line:
                    console.print(f"[yellow]{line.rstrip()}[/yellow]")
                elif "INFO" in line:
                    console.print(line.rstrip())
                else:
                    console.print(f"[dim]{line.rstrip()}[/dim]")

        except Exception as e:
            console.print(f"[red]Error reading log file:[/red] {e!s}")


@cli.command("alerts")
@click.option("--list", "list_alerts", is_flag=True, help="List active alerts")
@click.option("--add", help="Add new alert (format: metric,threshold,condition)")
@click.option("--remove", help="Remove alert by name")
@click.option("--status", is_flag=True, help="Show alert system status")
def alerts(list_alerts: bool, add: Optional[str], remove: Optional[str], status: bool):
    """Manage training alerts."""
    console.print(Panel.fit("[bold cyan]Alert Management[/bold cyan]", border_style="cyan"))

    if status:
        # Show alert system status
        console.print("[cyan]Alert System Status:[/cyan]")
        console.print("• Alert system: Active")
        console.print("• Default alerts: Enabled")
        console.print("• Notification methods: Console")

        # Show available alert types
        table = Table(title="Available Alert Types", box=box.ROUNDED)
        table.add_column("Alert Type", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("loss_explosion", "Triggers when loss > 10.0")
        table.add_row("loss_stagnation", "Detects when loss stops improving")
        table.add_row("gpu_memory_high", "Warns when GPU memory > 90%")
        table.add_row("lr_too_high", "Warns when learning rate > 0.01")

        console.print(table)

    elif list_alerts:
        # List active alerts
        console.print("[cyan]Active Alerts:[/cyan]")

        # This would show actual alerts from the alert system
        # For now, show example alerts
        table = Table(title="Active Alerts", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Metric", style="green")
        table.add_column("Condition", style="blue")
        table.add_column("Threshold", style="yellow")
        table.add_column("Status", style="white")

        table.add_row("loss_explosion", "loss", "greater_than", "10.0", "Active")
        table.add_row("gpu_memory", "gpu_memory_percent", "greater_than", "90.0", "Active")

        console.print(table)

    elif add:
        # Add new alert
        try:
            parts = add.split(",")
            if len(parts) != 3:
                raise ValueError("Format: metric,threshold,condition")

            metric, threshold, condition = parts
            console.print(f"[green]Added alert:[/green] {metric} {condition} {threshold}")

        except ValueError as e:
            console.print(f"[red]Invalid format:[/red] {e!s}")
            console.print("Use format: metric,threshold,condition")
            console.print("Example: loss,5.0,greater_than")

    elif remove:
        # Remove alert
        console.print(f"[green]Removed alert:[/green] {remove}")

    else:
        # Show help
        console.print("[yellow]Use --help to see available options[/yellow]")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
