"""Prompt runner for executing custom prompts across multiple LLM models.

This module provides the PromptRunner class which handles:
- Dynamic loading of model providers
- Parallel execution across multiple models
- Retry logic for transient failures
- Response collection with metadata and timing
- Progress reporting for long-running executions
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import concurrent.futures
import json
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import get_model_config
from .providers import get_provider_for_model, registry
from .providers.exceptions import InvalidCredentialsError
from .template_engine import PromptTemplate


@dataclass
class ModelResponse:
    """Container for a model's response with metadata."""

    model: str
    provider: str
    prompt: str
    rendered_prompt: str
    response: Optional[str]
    success: bool
    error: Optional[str]
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    retry_count: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str | Any]:
        """Convert to dictionary with JSON-serializable values."""
        result = asdict(self)
        result["start_time"] = self.start_time.isoformat()
        result["end_time"] = self.end_time.isoformat()
        return result


@dataclass
class ExecutionResult:
    """Container for the complete execution result across all models."""

    prompt_template: str
    template_variables: Dict[str, Any]
    models_requested: List[str]
    models_succeeded: List[str]
    models_failed: List[str]
    total_duration_seconds: float
    responses: List[ModelResponse]
    execution_mode: str  # 'sequential' or 'parallel'

    def to_dict(self) -> Dict[str | Any]:
        """Convert to dictionary with JSON-serializable values."""
        return {
            "prompt_template": self.prompt_template,
            "template_variables": self.template_variables,
            "models_requested": self.models_requested,
            "models_succeeded": self.models_succeeded,
            "models_failed": self.models_failed,
            "total_duration_seconds": self.total_duration_seconds,
            "responses": [r.to_dict() for r in self.responses],
            "execution_mode": self.execution_mode,
        }


class PromptRunner:
    """Executes prompts across multiple models with parallel execution support.

    This class provides methods to:
    - Run prompts on single or multiple models
    - Execute in parallel or sequential mode
    - Handle retries and failures gracefully
    - Collect detailed response metadata
    - Report progress during execution
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 300.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """Initialize the prompt runner.

        Args:
            max_retries: Maximum number of retries per model
            retry_delay: Base delay between retries (exponential backoff)
            timeout: Maximum time per model execution in seconds
            progress_callback: Optional callback for progress updates
                              Called with (message: str, progress: float)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.progress_callback = progress_callback or self._default_progress
        self.model_config = get_model_config()

    def _default_progress(self, message: str, progress: float) -> None:
        """Default progress callback that prints to console."""
        print(f"[{progress:3.0f}%] {message}")

    def _report_progress(self, message: str, current: int, total: int) -> None:
        """Report progress using the configured callback."""
        progress = (current / total * 100) if total > 0 else 0
        self.progress_callback(message, progress)

    def run_single(
        self,
        prompt: str | PromptTemplate,
        model: str,
        template_variables: Optional[Dict[str, Any]] = None,
        **generation_kwargs,
    ) -> ModelResponse:
        """Run a prompt on a single model.

        Args:
            prompt: Prompt string or PromptTemplate instance
            model: Model identifier (e.g., 'gpt-4', 'claude-3-sonnet')
            template_variables: Variables for template substitution
            **generation_kwargs: Additional arguments for the provider's generate method

        Returns:
            ModelResponse with the result
        """
        start_time = datetime.now()
        retry_count = 0

        # Prepare template and render prompt
        if isinstance(prompt, str):
            template = PromptTemplate(prompt, name="custom_prompt")
        else:
            template = prompt

        # Prepare rendering context
        render_context = {"model_name": model}
        if template_variables:
            render_context.update(template_variables)

        # Render the prompt
        try:
            rendered_prompt = template.render(render_context, strict=False)
        except Exception as e:
            end_time = datetime.now()
            return ModelResponse(
                model=model,
                provider="unknown",
                prompt=template.template_str,
                rendered_prompt="",
                response=None,
                success=False,
                error=f"Template rendering error: {e!s}",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                retry_count=0,
                metadata={},
            )

        # Get provider for model
        try:
            provider_class = get_provider_for_model(model)
            provider_name = provider_class.__name__.replace("Provider", "").lower()
        except ValueError as e:
            end_time = datetime.now()
            return ModelResponse(
                model=model,
                provider="unknown",
                prompt=template.template_str,
                rendered_prompt=rendered_prompt,
                response=None,
                success=False,
                error=str(e),
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                retry_count=0,
                metadata={},
            )

        # Initialize provider
        try:
            provider = provider_class(model)
            provider.initialize()
        except InvalidCredentialsError as e:
            end_time = datetime.now()
            return ModelResponse(
                model=model,
                provider=provider_name,
                prompt=template.template_str,
                rendered_prompt=rendered_prompt,
                response=None,
                success=False,
                error=f"Invalid credentials: {e!s}",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                retry_count=0,
                metadata={},
            )
        except Exception as e:
            end_time = datetime.now()
            return ModelResponse(
                model=model,
                provider=provider_name,
                prompt=template.template_str,
                rendered_prompt=rendered_prompt,
                response=None,
                success=False,
                error=f"Provider initialization failed: {e!s}",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                retry_count=0,
                metadata={},
            )

        # Prepare generation parameters
        gen_params = {
            "temperature": self.model_config.get("temperature", 0.7),
            "max_tokens": self.model_config.get("max_tokens", 1000),
        }
        gen_params.update(generation_kwargs)

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = provider.generate(rendered_prompt, **gen_params)
                end_time = datetime.now()

                return ModelResponse(
                    model=model,
                    provider=provider_name,
                    prompt=template.template_str,
                    rendered_prompt=rendered_prompt,
                    response=response,
                    success=True,
                    error=None,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(end_time - start_time).total_seconds(),
                    retry_count=retry_count,
                    metadata={
                        "generation_params": gen_params,
                        "template_variables": template_variables or {},
                    },
                )

            except Exception as e:
                retry_count += 1
                last_error = str(e)

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)
                    continue

                # Final attempt failed
                end_time = datetime.now()
                return ModelResponse(
                    model=model,
                    provider=provider_name,
                    prompt=template.template_str,
                    rendered_prompt=rendered_prompt,
                    response=None,
                    success=False,
                    error=f"Failed after {retry_count} retries: {last_error}",
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(end_time - start_time).total_seconds(),
                    retry_count=retry_count,
                    metadata={
                        "generation_params": gen_params,
                        "template_variables": template_variables or {},
                    },
                )

    def run_multiple(
        self,
        prompt: str | PromptTemplate,
        models: List[str],
        template_variables: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        **generation_kwargs,
    ) -> ExecutionResult:
        """Run a prompt across multiple models.

        Args:
            prompt: Prompt string or PromptTemplate instance
            models: List of model identifiers
            template_variables: Variables for template substitution
            parallel: Whether to execute in parallel
            max_workers: Maximum number of parallel workers (None for default)
            **generation_kwargs: Additional arguments for generate method

        Returns:
            ExecutionResult with all responses
        """
        execution_start = datetime.now()

        # Prepare template
        if isinstance(prompt, str):
            template = PromptTemplate(prompt, name="custom_prompt")
        else:
            template = prompt

        responses: List[ModelResponse] = []

        if parallel and len(models) > 1:
            # Parallel execution
            self._report_progress("Starting parallel execution", 0, len(models))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_model = {
                    executor.submit(
                        self.run_single, template, model, template_variables, **generation_kwargs
                    ): model
                    for model in models
                }

                # Collect results as they complete
                completed = 0
                for future in concurrent.futures.as_completed(future_to_model):
                    model = future_to_model[future]
                    completed += 1

                    try:
                        response = future.result(timeout=self.timeout)
                        responses.append(response)

                        status = "✓" if response.success else "✗"
                        self._report_progress(f"{status} Completed {model}", completed, len(models))
                    except concurrent.futures.TimeoutError:
                        # Create timeout response
                        responses.append(
                            ModelResponse(
                                model=model,
                                provider="unknown",
                                prompt=template.template_str,
                                rendered_prompt="",
                                response=None,
                                success=False,
                                error=f"Timeout after {self.timeout} seconds",
                                start_time=execution_start,
                                end_time=datetime.now(),
                                duration_seconds=self.timeout,
                                retry_count=0,
                                metadata={},
                            )
                        )
                        self._report_progress(f"✗ Timeout for {model}", completed, len(models))
                    except Exception as e:
                        # Create error response
                        responses.append(
                            ModelResponse(
                                model=model,
                                provider="unknown",
                                prompt=template.template_str,
                                rendered_prompt="",
                                response=None,
                                success=False,
                                error=f"Unexpected error: {e!s}",
                                start_time=execution_start,
                                end_time=datetime.now(),
                                duration_seconds=0,
                                retry_count=0,
                                metadata={"traceback": traceback.format_exc()},
                            )
                        )
                        self._report_progress(f"✗ Error for {model}", completed, len(models))
        else:
            # Sequential execution
            self._report_progress("Starting sequential execution", 0, len(models))

            for i, model in enumerate(models):
                response = self.run_single(template, model, template_variables, **generation_kwargs)
                responses.append(response)

                status = "✓" if response.success else "✗"
                self._report_progress(f"{status} Completed {model}", i + 1, len(models))

        # Calculate summary statistics
        execution_end = datetime.now()
        total_duration = (execution_end - execution_start).total_seconds()

        models_succeeded = [r.model for r in responses if r.success]
        models_failed = [r.model for r in responses if not r.success]

        return ExecutionResult(
            prompt_template=template.template_str,
            template_variables=template_variables or {},
            models_requested=models,
            models_succeeded=models_succeeded,
            models_failed=models_failed,
            total_duration_seconds=total_duration,
            responses=responses,
            execution_mode="parallel" if parallel else "sequential",
        )

    def run_from_file(
        self,
        prompt_file: str | Path,
        models: List[str],
        template_variables: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
        **generation_kwargs,
    ) -> ExecutionResult:
        """Run a prompt from a file across multiple models.

        Args:
            prompt_file: Path to the prompt template file
            models: List of model identifiers
            template_variables: Variables for template substitution
            parallel: Whether to execute in parallel
            **generation_kwargs: Additional arguments for generate method

        Returns:
            ExecutionResult with all responses
        """
        template = PromptTemplate.from_file(prompt_file)
        return self.run_multiple(
            template, models, template_variables, parallel, **generation_kwargs
        )

    def list_available_models(self) -> List[str]:
        """Get list of all available models from the registry.

        Returns:
            List of model identifiers
        """
        return registry.list_all_models()

    def save_results(
        self, result: ExecutionResult, output_path: str | Path, format: str = "json"
    ) -> None:
        """Save execution results to a file.

        Args:
            result: ExecutionResult to save
            output_path: Path to save the results
            format: Output format ('json' or 'jsonl')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
        elif format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for response in result.responses:
                    f.write(json.dumps(response.to_dict()) + "\n")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'jsonl'")


# Convenience functions
def run_prompt_on_models(
    prompt: str,
    models: List[str],
    template_variables: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
    progress: bool = True,
) -> ExecutionResult:
    """Convenience function to run a prompt on multiple models.

    Args:
        prompt: The prompt template string
        models: List of model identifiers
        template_variables: Variables for template substitution
        parallel: Whether to execute in parallel
        progress: Whether to show progress

    Returns:
        ExecutionResult with all responses
    """
    runner = PromptRunner(progress_callback=None if not progress else None)
    return runner.run_multiple(prompt, models, template_variables, parallel)
