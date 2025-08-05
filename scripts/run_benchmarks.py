"""
LLM Lab Benchmark Runner CLI

This module provides a command-line interface for running LLM benchmarks
using various providers and datasets.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click

from src.config import MODEL_DEFAULTS, ConfigurationError, get_api_key, get_model_config
from src.evaluation import keyword_match
from src.evaluation.improved_evaluation import multi_method_evaluation
from src.providers import registry, get_provider_for_model
from src.providers.exceptions import ProviderError, InvalidCredentialsError
from src.logging import CSVResultLogger
from src.use_cases.custom_prompts import PromptTemplate
from benchmarks import MultiModelBenchmarkRunner, ExecutionMode


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to call (should be a callable)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be multiplied by 2^attempt)

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries - 1:  # Don't sleep on last attempt
                delay = base_delay * (2**attempt)
                click.echo(
                    f"      ⟳ Retry {attempt + 1}/{max_retries} after {delay:.1f}s delay...",
                    err=True,
                )
                time.sleep(delay)
            else:
                click.echo(f"      ✗ All {max_retries} retries exhausted", err=True)

    raise last_exception


# Import all providers to ensure they register themselves
from src.providers import GoogleProvider, OpenAIProvider, AnthropicProvider

# Available benchmark datasets
DATASETS = {
    "truthfulness": "datasets/benchmarking/processed/truthfulqa/dataset.jsonl",
    "arc": "datasets/benchmarking/processed/arc/dataset.jsonl",
    "gsm8k": "datasets/benchmarking/processed/gsm8k/dataset.jsonl",
    "mmlu": "datasets/benchmarking/processed/mmlu/dataset.jsonl",
    "hellaswag": "datasets/benchmarking/processed/hellaswag/dataset.jsonl",
}


def load_dataset(dataset_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load and parse a JSONL dataset file.

    Args:
        dataset_path: Path to the JSONL dataset file
        limit: Optional limit on number of entries to load (for testing)

    Returns:
        List of parsed dataset entries

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    entries = []
    errors = []

    with open(dataset_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                entry = json.loads(line)

                # Validate required fields
                required_fields = ["id", "prompt", "evaluation_method"]
                missing_fields = [field for field in required_fields if field not in entry]

                if missing_fields:
                    errors.append(f"Line {line_num}: Missing required fields: {missing_fields}")
                    continue

                # Validate evaluation method specific fields
                if entry["evaluation_method"] == "keyword_match":
                    if "expected_keywords" not in entry:
                        errors.append(
                            f"Line {line_num}: Missing 'expected_keywords' for keyword_match method"
                        )
                        continue
                    if not isinstance(entry["expected_keywords"], list):
                        errors.append(f"Line {line_num}: 'expected_keywords' must be a list")
                        continue

                entries.append(entry)

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON parsing error - {e!s}")

    if errors:
        click.echo("\n   ⚠️  Dataset loading warnings:", err=True)
        for error in errors[:5]:  # Show first 5 errors
            click.echo(f"      - {error}", err=True)
        if len(errors) > 5:
            click.echo(f"      ... and {len(errors) - 5} more errors", err=True)

    if not entries:
        raise ValueError("No valid entries found in dataset")

    # Apply limit if specified (useful for testing)
    if limit is not None and limit > 0:
        entries = entries[:limit]

    return entries


def run_benchmark(
    model_name: str,
    dataset_name: str,
    limit: Optional[int] = None,
    evaluation_method: str = "multi",
) -> dict:
    """
    Orchestrate the benchmark execution.

    Args:
        model_name: Name of the LLM model to use
        dataset_name: Name of the benchmark dataset to run
        limit: Optional limit on number of dataset entries to process (for testing)
        evaluation_method: Evaluation method to use ('keyword', 'fuzzy', 'multi')

    Returns:
        dict: Results of the benchmark run including scores and details
    """
    start_time = datetime.now()

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "start_time": start_time.isoformat(),
        "total_prompts": 0,
        "successful_evaluations": 0,
        "failed_evaluations": 0,
        "evaluations": [],
        "model_config": {},
    }

    try:
        # Step 1: Load configuration and API key
        click.echo("\n1. Loading configuration...")

        # Load model configuration
        try:
            model_config = get_model_config()
            results["model_config"] = model_config
            click.echo("   ✓ Model configuration loaded")
            click.echo(f"     - Temperature: {model_config.get('temperature', 0.7)}")
            click.echo(f"     - Max tokens: {model_config.get('max_tokens', 1000)}")
            click.echo(f"     - Timeout: {model_config.get('timeout_seconds', 30)}s")
        except ConfigurationError as e:
            click.echo(f"   ⚠️  Using default model configuration: {e!s}", err=True)
            model_config = MODEL_DEFAULTS
            results["model_config"] = model_config

        # Step 2: Initialize provider for the model
        click.echo("\n2. Initializing provider...")
        try:
            # Get provider class for the model
            provider_class = get_provider_for_model(model_name)
            provider_name = provider_class.__name__.replace("Provider", "").lower()
            results["provider"] = provider_name

            # Create provider instance with model name
            provider = provider_class(model_name)

            # Initialize and validate credentials
            provider.initialize()
            click.echo(f"   ✓ {provider_name} provider initialized for model {model_name}")

        except ValueError as e:
            click.echo(f"   ✗ {e!s}", err=True)
            results["error"] = str(e)
            return results
        except InvalidCredentialsError as e:
            click.echo(f"   ✗ Invalid credentials: {e!s}", err=True)
            results["error"] = f"Invalid credentials: {e!s}"
            return results
        except Exception as e:
            click.echo(f"   ✗ Failed to initialize provider: {e!s}", err=True)
            results["error"] = f"Provider initialization failed: {e!s}"
            return results

        # Step 3: Load and validate dataset
        click.echo("\n3. Loading dataset...")
        dataset_path = DATASETS.get(dataset_name)

        if not dataset_path:
            raise ConfigurationError(f"Unknown dataset: {dataset_name}")

        # Note: Dataset validation is handled by load_dataset() function below

        # Load dataset entries
        entries = load_dataset(dataset_path, limit)

        results["total_prompts"] = len(entries)
        click.echo(f"   ✓ Loaded {len(entries)} prompts from {dataset_name}")

        # Step 4: Run evaluations
        click.echo("\n4. Running evaluations...")
        for i, entry in enumerate(entries, 1):
            prompt_start_time = datetime.now()

            click.echo(f"\n   Prompt {i}/{len(entries)}: {entry['prompt'][:50]}...")

            eval_metadata = {
                "prompt_id": entry.get("id", f"unknown_{i}"),
                "prompt": entry["prompt"],
                "timestamp": prompt_start_time.isoformat(),
                "model_name": model_name,
                "provider": provider_name,
                "benchmark_name": dataset_name,
                "evaluation_method": entry.get("evaluation_method", "unknown"),
                "expected_keywords": entry.get("expected_keywords", []),
            }

            try:
                # Generate response from model with retry logic
                max_retries = model_config.get("max_retries", 3)
                retry_delay = model_config.get("retry_delay", 1.0)

                def generate_with_prompt():
                    return provider.generate(entry["prompt"])

                try:
                    response = retry_with_backoff(
                        generate_with_prompt, max_retries=max_retries, base_delay=retry_delay
                    )
                except Exception as e:
                    # Log the specific error type for better debugging
                    error_msg = str(e).lower()
                    if "api_key" in error_msg or "authentication" in error_msg:
                        raise Exception(f"Authentication error: {e!s}")
                    elif "quota" in error_msg:
                        raise Exception(f"API quota exceeded: {e!s}")
                    elif "timeout" in error_msg:
                        raise Exception(f"Request timeout: {e!s}")
                    else:
                        raise

                response_time = (datetime.now() - prompt_start_time).total_seconds()

                click.echo(f"   Response: {response[:100]}...")
                click.echo(f"   Generation time: {response_time:.2f}s")

                eval_metadata["response"] = response
                eval_metadata["response_time_seconds"] = response_time

                # Evaluate response based on method
                if entry["evaluation_method"] == "keyword_match":
                    # Choose evaluation method based on user preference
                    if evaluation_method == "keyword":
                        eval_result = keyword_match(response, entry["expected_keywords"])
                    elif evaluation_method == "fuzzy":
                        from src.evaluation.improved_evaluation import fuzzy_keyword_match

                        eval_result = fuzzy_keyword_match(response, entry["expected_keywords"])
                    else:  # multi (default)
                        eval_result = multi_method_evaluation(response, entry["expected_keywords"])

                    # Merge evaluation result with metadata
                    full_result = {**eval_metadata, **eval_result}
                    results["evaluations"].append(full_result)

                    if eval_result["success"]:
                        results["successful_evaluations"] += 1
                        click.echo(
                            f"   ✓ Evaluation passed (matched: {', '.join(eval_result['matched_keywords'])})"
                        )
                    else:
                        results["failed_evaluations"] += 1
                        click.echo("   ✗ Evaluation failed (no keyword match)")
                else:
                    # Unsupported evaluation method
                    click.echo(
                        f"   ⚠️  Unsupported evaluation method: {entry['evaluation_method']}",
                        err=True,
                    )
                    results["failed_evaluations"] += 1
                    results["evaluations"].append(
                        {
                            **eval_metadata,
                            "success": False,
                            "score": 0.0,
                            "error": f"Unsupported evaluation method: {entry['evaluation_method']}",
                        }
                    )

            except Exception as e:
                click.echo(f"   ✗ Error during evaluation: {e!s}", err=True)
                results["failed_evaluations"] += 1
                results["evaluations"].append(
                    {
                        **eval_metadata,
                        "success": False,
                        "score": 0.0,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )

        # Calculate overall score and finalize results
        end_time = datetime.now()
        results["end_time"] = end_time.isoformat()
        results["total_duration_seconds"] = (end_time - start_time).total_seconds()

        if results["total_prompts"] > 0:
            results["overall_score"] = results["successful_evaluations"] / results["total_prompts"]

            # Calculate average response time
            response_times = [
                eval_data.get("response_time_seconds", 0)
                for eval_data in results["evaluations"]
                if "response_time_seconds" in eval_data
            ]
            if response_times:
                results["average_response_time_seconds"] = sum(response_times) / len(response_times)
            else:
                results["average_response_time_seconds"] = 0.0
        else:
            results["overall_score"] = 0.0
            results["average_response_time_seconds"] = 0.0

    except Exception as e:
        click.echo(f"\n✗ Fatal error: {e!s}", err=True)
        results["error"] = str(e)
        results["end_time"] = datetime.now().isoformat()
        results["total_duration_seconds"] = (datetime.now() - start_time).total_seconds()

    return results


def run_custom_prompt(
    model_name: str,
    prompt_content: str,
    template_variables: dict = None,
) -> dict:
    """
    Run a custom prompt on a single model.
    
    Args:
        model_name: Name of the LLM model to use
        prompt_content: The prompt text (may contain template variables)
        template_variables: Optional dict of variables to substitute in the prompt
        
    Returns:
        dict: Results including the model response
    """
    start_time = datetime.now()
    
    results = {
        "model": model_name,
        "prompt": prompt_content,
        "template_variables": template_variables or {},
        "start_time": start_time.isoformat(),
    }
    
    try:
        # Apply template variables using the template engine
        template = PromptTemplate(prompt_content, name="custom_prompt")
        
        # Add model_name to the context
        render_context = {"model_name": model_name}
        if template_variables:
            render_context.update(template_variables)
        
        # Check for missing variables before rendering
        missing_vars = template.validate_context(render_context)
        if missing_vars:
            click.echo(f"⚠️  Missing template variables: {', '.join(missing_vars)}", err=True)
            click.echo(f"   Available variables: {', '.join(template.get_all_variables())}")
        
        # Render the prompt (non-strict mode to allow missing variables)
        rendered_prompt = template.render(render_context, strict=False)
        results["rendered_prompt"] = rendered_prompt
        results["template_info"] = {
            "required_variables": list(template.get_required_variables()),
            "all_variables": list(template.get_all_variables()),
            "missing_variables": missing_vars
        }
            
        # Get provider and initialize
        provider_class = get_provider_for_model(model_name)
        provider_name = provider_class.__name__.replace("Provider", "").lower()
        results["provider"] = provider_name
        
        provider = provider_class(model_name)
        provider.initialize()
        
        # Get model configuration
        model_config = get_model_config()
        
        # Generate response
        prompt_start_time = datetime.now()
        response = retry_with_backoff(
            lambda: provider.generate(
                rendered_prompt,
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 1000),
            ),
            max_retries=model_config.get("max_retries", 3),
            base_delay=model_config.get("retry_delay", 1.0),
        )
        prompt_end_time = datetime.now()
        
        results["response"] = response
        results["response_time_seconds"] = (prompt_end_time - prompt_start_time).total_seconds()
        results["success"] = True
        
        # Basic metrics (will be expanded in task 3.4)
        results["metrics"] = {
            "response_length": len(response),
            "response_words": len(response.split()),
            "response_lines": len(response.splitlines()),
        }
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        results["error_type"] = type(e).__name__
        
    end_time = datetime.now()
    results["end_time"] = end_time.isoformat()
    results["total_duration_seconds"] = (end_time - start_time).total_seconds()
    
    return results


@click.command()
@click.option(
    "--model",
    type=str,
    help="Single model name to benchmark (e.g., gemini-1.5-flash, gpt-4, claude-3-opus-20240229)",
)
@click.option(
    "--models",
    type=str,
    help="Comma-separated list of models to benchmark (e.g., gemini-1.5-flash,gpt-4,claude-3-opus-20240229)",
)
@click.option("--all-models", is_flag=True, default=False, help="Benchmark all available models")
@click.option(
    "--provider",
    type=str,
    help="(Deprecated) Use --model instead. Provider name for backward compatibility.",
)
@click.option(
    "--dataset",
    type=click.Choice(list(DATASETS.keys()), case_sensitive=False),
    required=True,
    help="Benchmark dataset to run",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of dataset entries to process (e.g., --limit 10 for testing)",
)
@click.option(
    "--evaluation-method",
    type=click.Choice(["keyword", "fuzzy", "multi"], case_sensitive=False),
    default="multi",
    help="Evaluation method: keyword (strict), fuzzy (similarity), multi (combined)",
)
@click.option(
    "--parallel", is_flag=True, default=False, help="Run benchmarks for multiple models in parallel"
)
@click.option("--no-csv", is_flag=True, default=False, help="Disable CSV output logging")
@click.option(
    "--output-dir", type=click.Path(), default="./results", help="Directory for saving CSV results"
)
@click.option(
    "--use-engine",
    is_flag=True,
    default=False,
    help="Use the new multi-model execution engine (experimental)",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Timeout in seconds per model (only with --use-engine)",
)
# Custom Prompt Options
@click.option(
    "--custom-prompt",
    type=str,
    default=None,
    help="Run a custom prompt instead of a dataset (e.g., 'What is 2+2?')",
)
@click.option(
    "--prompt-file",
    type=click.Path(exists=True),
    default=None,
    help="Load custom prompt from a file",
)
@click.option(
    "--prompt-variables",
    type=str,
    default=None,
    help='JSON string of template variables (e.g., \'{"context": "math", "difficulty": "easy"}\')',
)
def main(
    model: Optional[str],
    models: Optional[str],
    all_models: bool,
    provider: Optional[str],
    dataset: str,
    limit: Optional[int],
    evaluation_method: str,
    parallel: bool,
    no_csv: bool,
    output_dir: str,
    use_engine: bool,
    timeout: Optional[int],
    custom_prompt: Optional[str],
    prompt_file: Optional[str],
    prompt_variables: Optional[str],
):
    """
    Run LLM benchmarks with specified models and dataset.

    Examples:
        # Single model
        python run_benchmarks.py --model gemini-1.5-flash --dataset truthfulness

        # Quick test with first 10 entries
        python run_benchmarks.py --model gemini-1.5-flash --dataset truthfulness --limit 10

        # Multiple models
        python run_benchmarks.py --models gemini-1.5-flash,gpt-4,claude-3-opus-20240229 --dataset truthfulness

        # All available models
        python run_benchmarks.py --all-models --dataset truthfulness

        # Parallel execution
        python run_benchmarks.py --models gpt-4,claude-3-opus-20240229 --dataset truthfulness --parallel
        
        # Custom prompt (inline)
        python run_benchmarks.py --model gpt-4 --custom-prompt "What is 2+2?"
        
        # Custom prompt from file
        python run_benchmarks.py --model claude-3-opus-20240229 --prompt-file ./prompts/test.txt
        
        # Custom prompt with template variables
        python run_benchmarks.py --models gemini-1.5-flash,gpt-4 --custom-prompt "Solve this {difficulty} math problem: {problem}" --prompt-variables '{"difficulty": "easy", "problem": "2+2"}'
        
        # Compare multiple models on custom prompt
        python run_benchmarks.py --models gpt-4,claude-3-opus-20240229,gemini-1.5-flash --custom-prompt "Write a haiku about programming" --parallel
    """
    click.echo("🔬 LLM Lab Benchmark Runner")
    click.echo(f"{'=' * 50}")

    # Determine which models to benchmark
    models_to_benchmark = []

    # Handle backward compatibility with --provider
    if provider:
        click.echo("⚠️  Warning: --provider is deprecated. Use --model instead.", err=True)
        # Map provider to default model
        provider_model_map = {
            "google": "gemini-1.5-flash",
            "openai": "gpt-3.5-turbo",
            "anthropic": "claude-3-sonnet-20240229",
        }
        if provider in provider_model_map:
            models_to_benchmark = [provider_model_map[provider]]
        else:
            click.echo(f"❌ Unknown provider: {provider}", err=True)
            sys.exit(1)

    # Handle model selection options
    elif all_models:
        models_to_benchmark = registry.list_all_models()
        if not models_to_benchmark:
            click.echo("❌ No models available in registry", err=True)
            sys.exit(1)
    elif models:
        # Parse comma-separated list
        models_to_benchmark = [m.strip() for m in models.split(",") if m.strip()]
        if not models_to_benchmark:
            click.echo("❌ No valid models specified", err=True)
            sys.exit(1)
    elif model:
        models_to_benchmark = [model]
    else:
        click.echo(
            "❌ You must specify at least one model using --model, --models, or --all-models",
            err=True,
        )
        sys.exit(1)

    # Validate all models exist
    click.echo(f"\n📋 Validating {len(models_to_benchmark)} model(s)...")
    invalid_models = []
    for model_name in models_to_benchmark:
        try:
            provider_class = get_provider_for_model(model_name)
            click.echo(f"   ✓ {model_name} -> {provider_class.__name__}")
        except ValueError:
            invalid_models.append(model_name)
            click.echo(f"   ✗ {model_name} -> No provider found", err=True)

    if invalid_models:
        available = registry.list_all_models()
        click.echo(f"\n❌ Invalid models: {', '.join(invalid_models)}", err=True)
        click.echo(f"Available models: {', '.join(available)}", err=True)
        sys.exit(1)

    # Handle custom prompt options
    is_custom_prompt_mode = False
    prompt_content = None
    template_variables = {}
    
    # Validate mutually exclusive options
    if custom_prompt and prompt_file:
        click.echo("❌ Cannot use both --custom-prompt and --prompt-file", err=True)
        sys.exit(1)
    
    if (custom_prompt or prompt_file) and dataset != "truthfulness":
        click.echo("⚠️  Warning: --dataset is ignored when using custom prompts", err=True)
    
    # Process custom prompt
    if custom_prompt or prompt_file:
        is_custom_prompt_mode = True
        
        if custom_prompt:
            prompt_content = custom_prompt
            click.echo(f"\n📝 Using custom prompt: {custom_prompt[:50]}...")
        else:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read().strip()
            click.echo(f"\n📝 Loaded custom prompt from: {prompt_file}")
        
        # Parse template variables if provided
        if prompt_variables:
            try:
                template_variables = json.loads(prompt_variables)
                click.echo(f"   ✓ Template variables: {template_variables}")
            except json.JSONDecodeError as e:
                click.echo(f"❌ Invalid JSON in --prompt-variables: {e}", err=True)
                sys.exit(1)

    try:
        # Run benchmarks or custom prompts
        all_results = []

        if is_custom_prompt_mode:
            # Custom prompt mode
            click.echo(f"\n🚀 Running custom prompt on {len(models_to_benchmark)} model(s)...")
            
            if parallel and len(models_to_benchmark) > 1:
                # Parallel execution for custom prompts
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(models_to_benchmark), 4)
                ) as executor:
                    future_to_model = {
                        executor.submit(
                            run_custom_prompt, model_name, prompt_content, template_variables
                        ): model_name
                        for model_name in models_to_benchmark
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_model):
                        model_name = future_to_model[future]
                        try:
                            result = future.result()
                            all_results.append(result)
                            if result.get("success"):
                                click.echo(f"✓ {model_name}: Response received ({result['metrics']['response_words']} words)")
                            else:
                                click.echo(f"✗ {model_name}: {result.get('error', 'Unknown error')}", err=True)
                        except Exception as e:
                            click.echo(f"✗ Exception for {model_name}: {e!s}", err=True)
                            all_results.append({
                                "model": model_name,
                                "success": False,
                                "error": str(e),
                                "error_type": type(e).__name__
                            })
            else:
                # Sequential execution for custom prompts
                for model_name in models_to_benchmark:
                    click.echo(f"\n📤 Sending prompt to {model_name}...")
                    result = run_custom_prompt(model_name, prompt_content, template_variables)
                    all_results.append(result)
                    
                    if result.get("success"):
                        click.echo(f"✓ Response received ({result['metrics']['response_words']} words)")
                        click.echo(f"\n📝 Response:\n{result['response']}")
                    else:
                        click.echo(f"✗ Error: {result.get('error', 'Unknown error')}", err=True)
        
        elif use_engine:
            # Use the new multi-model execution engine
            click.echo(f"\n🔧 Using multi-model execution engine...")

            # Define progress callback
            def progress_callback(model_name: str, current: int, total: int):
                click.echo(f"   [{current}/{total}] Processing: {model_name}")

            # Create a wrapper function that includes the limit and evaluation_method parameters
            def benchmark_function_with_limit(model_name: str, dataset_name: str) -> dict:
                return run_benchmark(model_name, dataset_name, limit, evaluation_method)

            # Create the execution engine
            execution_mode = ExecutionMode.PARALLEL if parallel else ExecutionMode.SEQUENTIAL
            runner = MultiModelBenchmarkRunner(
                benchmark_function=benchmark_function_with_limit,
                execution_mode=execution_mode,
                max_workers=4,
                timeout_per_model=timeout,
                progress_callback=progress_callback,
            )

            # Run benchmarks
            benchmark_result = runner.run(models_to_benchmark, dataset)

            # Convert results to the expected format
            for model_result in benchmark_result.model_results:
                all_results.append(model_result.to_dict())

            # Display execution summary
            stats = benchmark_result.summary_stats
            click.echo(
                f"\n✓ Execution completed: {stats['successful_models']}/{stats['total_models']} models successful"
            )
            if stats["best_model"]:
                click.echo(
                    f"   Best model: {stats['best_model']} (Score: {stats['best_score']:.2%})"
                )
            if stats["failed_models"] > 0:
                click.echo(f"   Failed models: {stats['failed_models']}")

        elif parallel and len(models_to_benchmark) > 1:
            # Parallel execution
            click.echo(
                f"\n🚀 Running benchmarks in parallel for {len(models_to_benchmark)} models..."
            )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(models_to_benchmark), 4)
            ) as executor:
                # Submit all benchmarks
                future_to_model = {
                    executor.submit(
                        run_benchmark, model_name, dataset, limit, evaluation_method
                    ): model_name
                    for model_name in models_to_benchmark
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        if "error" not in result:
                            click.echo(
                                f"✓ Completed: {model_name} (Score: {result.get('overall_score', 0):.2%})"
                            )
                        else:
                            click.echo(f"✗ Failed: {model_name} - {result['error']}", err=True)
                    except Exception as e:
                        click.echo(f"✗ Exception for {model_name}: {e!s}", err=True)
                        all_results.append(
                            {
                                "model": model_name,
                                "dataset": dataset,
                                "error": str(e),
                                "overall_score": 0.0,
                            }
                        )
        else:
            # Sequential execution
            click.echo(
                f"\n📊 Running benchmarks for {len(models_to_benchmark)} model(s) sequentially..."
            )

            for model_name in models_to_benchmark:
                click.echo(f"\n{'=' * 50}")
                click.echo(f"Benchmarking: {model_name}")
                click.echo(f"{'=' * 50}")

                results = run_benchmark(model_name, dataset, limit, evaluation_method)
                all_results.append(results)

        # Display summary
        click.echo(f"\n{'=' * 50}")
        if is_custom_prompt_mode:
            click.echo("📊 Custom Prompt Results Summary")
        else:
            click.echo("📊 Benchmark Results Summary")
        click.echo(f"{'=' * 50}")

        if is_custom_prompt_mode:
            # Custom prompt results display
            if len(all_results) == 1:
                # Single model custom prompt result
                result = all_results[0]
                click.echo(f"\nModel: {result['model']}")
                click.echo(f"Provider: {result.get('provider', 'unknown')}")
                click.echo(f"Success: {'✓' if result.get('success') else '✗'}")
                
                if result.get('success'):
                    click.echo(f"\n📊 Response Metrics:")
                    metrics = result.get('metrics', {})
                    click.echo(f"  Length: {metrics.get('response_length', 0)} characters")
                    click.echo(f"  Words: {metrics.get('response_words', 0)}")
                    click.echo(f"  Lines: {metrics.get('response_lines', 0)}")
                    click.echo(f"  Response Time: {result.get('response_time_seconds', 0):.2f}s")
                else:
                    click.echo(f"Error: {result.get('error', 'Unknown error')}")
            else:
                # Multiple model custom prompt results
                click.echo(f"\nPrompt: {prompt_content[:100]}{'...' if len(prompt_content) > 100 else ''}")
                click.echo(f"Models Tested: {len(all_results)}")
                click.echo("\n📈 Model Comparison:")
                click.echo("-" * 80)
                click.echo(
                    f"{'Model':<30} {'Provider':<12} {'Status':<10} {'Words':<10} {'Time (s)':<10}"
                )
                click.echo("-" * 80)
                
                for result in all_results:
                    if result.get('success'):
                        click.echo(
                            f"{result['model']:<30} "
                            f"{result.get('provider', 'unknown'):<12} "
                            f"{'Success':<10} "
                            f"{result.get('metrics', {}).get('response_words', 0):<10} "
                            f"{result.get('response_time_seconds', 0):<10.2f}"
                        )
                    else:
                        click.echo(
                            f"{result['model']:<30} "
                            f"{result.get('provider', 'unknown'):<12} "
                            f"{'Failed':<10} "
                            f"{'N/A':<10} "
                            f"{'N/A':<10}"
                        )
                click.echo("-" * 80)
        elif len(all_results) == 1:
            # Single model results
            results = all_results[0]
            click.echo(f"Model: {results['model']}")
            click.echo(f"Provider: {results.get('provider', 'unknown')}")
            click.echo(f"Dataset: {results['dataset']}")
            click.echo(f"Total Prompts: {results['total_prompts']}")
            click.echo(f"Successful: {results['successful_evaluations']}")
            click.echo(f"Failed: {results['failed_evaluations']}")
            click.echo(f"Overall Score: {results.get('overall_score', 0):.2%}")

            # Display timing information
            if "total_duration_seconds" in results:
                click.echo("\n⏱️  Performance Metrics:")
                click.echo(f"Total Duration: {results['total_duration_seconds']:.2f}s")
                if "average_response_time_seconds" in results:
                    click.echo(
                        f"Avg Response Time: {results['average_response_time_seconds']:.2f}s"
                    )

            # Display model configuration used
            if results.get("model_config"):
                click.echo("\n⚙️  Model Configuration:")
                config = results["model_config"]
                click.echo(f"Temperature: {config.get('temperature', 'N/A')}")
                click.echo(f"Max Tokens: {config.get('max_tokens', 'N/A')}")
                click.echo(f"Max Retries: {config.get('max_retries', 'N/A')}")
        else:
            # Multiple model results - show comparison table
            click.echo(f"Dataset: {dataset}")
            click.echo(f"Models Tested: {len(all_results)}")
            click.echo("\n📈 Model Comparison:")
            click.echo("-" * 80)
            click.echo(
                f"{'Model':<30} {'Provider':<12} {'Score':<10} {'Success':<10} {'Failed':<10} {'Time (s)':<10}"
            )
            click.echo("-" * 80)

            for results in sorted(
                all_results, key=lambda x: x.get("overall_score", 0), reverse=True
            ):
                if "error" in results:
                    click.echo(
                        f"{results['model']:<30} {'ERROR':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}"
                    )
                else:
                    click.echo(
                        f"{results['model']:<30} "
                        f"{results.get('provider', 'unknown'):<12} "
                        f"{results.get('overall_score', 0):<10.2%} "
                        f"{results['successful_evaluations']:<10} "
                        f"{results['failed_evaluations']:<10} "
                        f"{results.get('total_duration_seconds', 0):<10.2f}"
                    )
            click.echo("-" * 80)

        # Save results to CSV if enabled
        if not no_csv:
            try:
                click.echo("\n💾 Saving results to CSV...")
                logger = CSVResultLogger(output_dir)
                saved_count = 0

                for results in all_results:
                    if results.get("evaluations"):
                        csv_path = logger.write_results(results)
                        saved_count += 1

                if saved_count > 0:
                    click.echo(f"   ✓ Saved results for {saved_count} model(s) to: {output_dir}")
                else:
                    click.echo("   ⚠️  No evaluation results to save", err=True)
            except Exception as e:
                click.echo(f"   ⚠️  Failed to save CSV: {e!s}", err=True)

        # Exit with appropriate code
        total_errors = sum(1 for r in all_results if "error" in r)
        total_failures = sum(r.get("failed_evaluations", 0) for r in all_results)

        if total_errors == len(all_results):
            sys.exit(1)  # All models failed
        elif total_failures > 0:
            sys.exit(2)  # Some evaluations failed
        else:
            sys.exit(0)  # All good

    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Benchmark interrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {e!s}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
