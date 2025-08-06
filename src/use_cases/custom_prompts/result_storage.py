"""Result storage and formatting system for custom prompt executions.

This module provides:
- Standardized result storage format compatible with benchmark infrastructure
- JSON and CSV formatters for results export
- Result comparison functionality
- Result viewer utility
- Result caching to avoid re-running identical prompts
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import csv
import hashlib
import io
import json
import pickle
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .prompt_runner import ExecutionResult, ModelResponse


@dataclass
class CustomPromptResult:
    """Standardized result format for custom prompt executions."""

    # Execution identification
    execution_id: str
    execution_timestamp: datetime

    # Prompt information
    prompt_template: str
    template_variables: Dict[str, Any]
    prompt_hash: str  # For caching

    # Execution details
    models_requested: List[str]
    models_succeeded: List[str]
    models_failed: List[str]
    execution_mode: str  # 'sequential' or 'parallel'
    total_duration_seconds: float

    # Results
    responses: List[ModelResponse]
    metrics: Optional[Dict[str, Any]] = None
    aggregated_metrics: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_execution_result(
        cls,
        exec_result: ExecutionResult,
        execution_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        aggregated_metrics: Optional[Dict[str, Any]] = None,
    ) -> "CustomPromptResult":
        """Create from ExecutionResult with optional metrics."""
        # Generate execution ID if not provided
        if not execution_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exec_id_parts = [timestamp]
            if exec_result.models_requested:
                exec_id_parts.append(exec_result.models_requested[0][:10])
            execution_id = "_".join(exec_id_parts)

        # Calculate prompt hash for caching
        prompt_data = {
            "template": exec_result.prompt_template,
            "variables": exec_result.template_variables,
        }
        prompt_hash = hashlib.md5(json.dumps(prompt_data, sort_keys=True).encode()).hexdigest()

        return cls(
            execution_id=execution_id,
            execution_timestamp=datetime.now(),
            prompt_template=exec_result.prompt_template,
            template_variables=exec_result.template_variables,
            prompt_hash=prompt_hash,
            models_requested=exec_result.models_requested,
            models_succeeded=exec_result.models_succeeded,
            models_failed=exec_result.models_failed,
            execution_mode=exec_result.execution_mode,
            total_duration_seconds=exec_result.total_duration_seconds,
            responses=exec_result.responses,
            metrics=metrics,
            aggregated_metrics=aggregated_metrics,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable values."""
        result = asdict(self)
        result["execution_timestamp"] = self.execution_timestamp.isoformat()
        result["responses"] = [r.to_dict() for r in self.responses]
        return result


class ResultFormatter:
    """Base class for result formatters."""

    def format(self, result: CustomPromptResult) -> str:
        """Format the result for output."""
        raise NotImplementedError


class JSONFormatter(ResultFormatter):
    """Format results as JSON."""

    def __init__(self, indent: int = 2, include_responses: bool = True):
        self.indent = indent
        self.include_responses = include_responses

    def format(self, result: CustomPromptResult) -> str:
        """Format as indented JSON."""
        data = result.to_dict()

        if not self.include_responses:
            # Create summary without full responses
            data["responses"] = [
                {
                    "model": r.model,
                    "success": r.success,
                    "response_preview": r.response[:100] + "..." if r.response else None,
                    "error": r.error,
                }
                for r in result.responses
            ]

        return json.dumps(data, indent=self.indent)


class CSVFormatter(ResultFormatter):
    """Format results as CSV."""

    def format(self, result: CustomPromptResult) -> str:
        """Format as CSV with one row per model response."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Define columns
        columns = [
            "execution_id",
            "timestamp",
            "prompt_template",
            "model",
            "success",
            "response_length",
            "duration_seconds",
            "error",
        ]

        # Add metric columns if available
        if result.metrics and result.responses:
            sample_metrics = next((r for r in result.responses if r.model in result.metrics), None)
            if sample_metrics and result.metrics.get(sample_metrics.model):
                metric_keys = list(result.metrics[sample_metrics.model].keys())
                columns.extend([f"metric_{k}" for k in metric_keys])

        # Write header
        writer.writerow(columns)

        # Write data rows
        for response in result.responses:
            row = [
                result.execution_id,
                result.execution_timestamp.isoformat(),
                result.prompt_template[:50] + "...",
                response.model,
                response.success,
                len(response.response) if response.response else 0,
                response.duration_seconds,
                response.error or "",
            ]

            # Add metrics if available
            if result.metrics and response.model in result.metrics:
                model_metrics = result.metrics[response.model]
                for key in metric_keys:
                    value = model_metrics.get(key, "")
                    # Flatten nested metrics
                    if isinstance(value, dict):
                        value = json.dumps(value)
                    row.append(value)

            writer.writerow(row)

        return output.getvalue()


class MarkdownFormatter(ResultFormatter):
    """Format results as Markdown for documentation."""

    def format(self, result: CustomPromptResult) -> str:
        """Format as Markdown report."""
        lines = []

        # Header
        lines.append("# Custom Prompt Execution Report")
        lines.append(f"\n**Execution ID**: `{result.execution_id}`")
        lines.append(f"**Timestamp**: {result.execution_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Duration**: {result.total_duration_seconds:.2f} seconds")
        lines.append(f"**Mode**: {result.execution_mode}")

        # Prompt information
        lines.append("\n## Prompt")
        lines.append("```")
        lines.append(result.prompt_template)
        lines.append("```")

        if result.template_variables:
            lines.append("\n**Variables**:")
            for key, value in result.template_variables.items():
                lines.append(f"- `{key}`: {value}")

        # Model results summary
        lines.append("\n## Results Summary")
        lines.append(f"- **Requested**: {', '.join(result.models_requested)}")
        lines.append(f"- **Succeeded**: {', '.join(result.models_succeeded) or 'None'}")
        lines.append(f"- **Failed**: {', '.join(result.models_failed) or 'None'}")

        # Individual responses
        lines.append("\n## Model Responses")
        for response in result.responses:
            lines.append(f"\n### {response.model}")
            if response.success:
                lines.append("✅ **Success**")
                lines.append(f"- Duration: {response.duration_seconds:.2f}s")
                lines.append(f"- Response length: {len(response.response)} chars")
                lines.append("\n**Response:**")
                lines.append("```")
                lines.append(
                    response.response[:500] + "..."
                    if len(response.response) > 500
                    else response.response
                )
                lines.append("```")
            else:
                lines.append("❌ **Failed**")
                lines.append(f"- Error: {response.error}")

        # Metrics if available
        if result.aggregated_metrics:
            lines.append("\n## Metrics")
            for metric_name, metric_data in result.aggregated_metrics.items():
                lines.append(f"\n### {metric_name}")
                if isinstance(metric_data, dict):
                    for key, value in metric_data.items():
                        lines.append(f"- {key}: {value}")
                else:
                    lines.append(f"- Value: {metric_data}")

        return "\n".join(lines)


class ResultStorage:
    """Manages storage and retrieval of custom prompt results."""

    def __init__(self, storage_dir: str | Path):
        """Initialize storage with a directory path."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache database
        self.cache_db = self.storage_dir / "prompt_cache.db"
        self._init_cache_db()

    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt_template TEXT,
                    template_variables TEXT,
                    result_data BLOB,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON prompt_cache(created_at)
            """)

    def save(self, result: CustomPromptResult, format: str = "json") -> Path:
        """Save result to file and update cache.

        Args:
            result: The result to save
            format: Output format ('json', 'csv', 'markdown')

        Returns:
            Path to the saved file
        """
        # Create subdirectory for the date
        date_dir = self.storage_dir / result.execution_timestamp.strftime("%Y-%m-%d")
        date_dir.mkdir(exist_ok=True)

        # Generate filename
        filename = f"{result.execution_id}.{format}"
        filepath = date_dir / filename

        # Format and save
        formatter = self._get_formatter(format)
        content = formatter.format(result)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # Update cache
        self._update_cache(result)

        return filepath

    def _get_formatter(self, format: str) -> ResultFormatter:
        """Get appropriate formatter for the format."""
        formatters = {
            "json": JSONFormatter(),
            "csv": CSVFormatter(),
            "markdown": MarkdownFormatter(),
            "md": MarkdownFormatter(),
        }

        if format not in formatters:
            raise ValueError(f"Unknown format: {format}. Use one of: {list(formatters.keys())}")

        return formatters[format]

    def _update_cache(self, result: CustomPromptResult):
        """Update the prompt cache with this result."""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO prompt_cache
                (prompt_hash, prompt_template, template_variables, result_data, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    result.prompt_hash,
                    result.prompt_template,
                    json.dumps(result.template_variables),
                    pickle.dumps(result),
                    result.execution_timestamp,
                    datetime.now(),
                ),
            )

    def check_cache(
        self, prompt_template: str, template_variables: Dict[str, Any]
    ) -> Optional[CustomPromptResult]:
        """Check if this prompt has been run before.

        Args:
            prompt_template: The prompt template
            template_variables: Template variables

        Returns:
            Cached result if found, None otherwise
        """
        # Calculate hash
        prompt_data = {"template": prompt_template, "variables": template_variables}
        prompt_hash = hashlib.md5(json.dumps(prompt_data, sort_keys=True).encode()).hexdigest()

        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute(
                """
                SELECT result_data FROM prompt_cache
                WHERE prompt_hash = ?
            """,
                (prompt_hash,),
            )

            row = cursor.fetchone()
            if row:
                # Update last accessed time
                conn.execute(
                    """
                    UPDATE prompt_cache
                    SET last_accessed = ?
                    WHERE prompt_hash = ?
                """,
                    (datetime.now(), prompt_hash),
                )

                return pickle.loads(row[0])

        return None

    def list_results(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List stored results with optional filtering.

        Args:
            start_date: Filter results after this date
            end_date: Filter results before this date
            model: Filter results for this model

        Returns:
            List of result summaries
        """
        results = []

        # Search all date directories
        for date_dir in sorted(self.storage_dir.glob("*")):
            if not date_dir.is_dir() or not re.match(r"\d{4}-\d{2}-\d{2}", date_dir.name):
                continue

            # Check date filter
            dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
            if start_date and dir_date < start_date:
                continue
            if end_date and dir_date > end_date:
                continue

            # Load JSON files in this directory
            for json_file in date_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    # Apply model filter if specified
                    if model and model not in data.get("models_requested", []):
                        continue

                    # Create summary
                    summary = {
                        "execution_id": data["execution_id"],
                        "timestamp": data["execution_timestamp"],
                        "prompt_preview": data["prompt_template"][:50] + "...",
                        "models": data["models_requested"],
                        "success_rate": len(data["models_succeeded"])
                        / len(data["models_requested"]),
                        "file_path": str(json_file),
                    }
                    results.append(summary)

                except Exception:
                    continue

        return results


class ResultComparator:
    """Compare results across different models or executions."""

    @staticmethod
    def compare_responses(result: CustomPromptResult) -> Dict[str, Any]:
        """Compare responses within a single execution.

        Returns:
            Comparison analysis including agreement, divergence, etc.
        """
        if len(result.models_succeeded) < 2:
            return {"error": "Need at least 2 successful responses to compare"}

        comparison = {
            "models": result.models_succeeded,
            "response_lengths": {},
            "common_phrases": [],
            "unique_content": {},
            "agreement_score": 0.0,
        }

        # Get successful responses
        responses = {r.model: r.response for r in result.responses if r.success and r.response}

        # Compare lengths
        for model, response in responses.items():
            comparison["response_lengths"][model] = len(response.split())

        # Find common phrases (simple approach)
        all_words = []
        for response in responses.values():
            all_words.extend(response.lower().split())

        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Common words appearing in all responses
        num_models = len(responses)
        common_words = [
            word for word, count in word_freq.items() if count >= num_models and len(word) > 4
        ]
        comparison["common_phrases"] = common_words[:10]

        # Calculate simple agreement score
        if common_words:
            total_unique_words = len(set(all_words))
            comparison["agreement_score"] = len(common_words) / total_unique_words

        return comparison

    @staticmethod
    def compare_metrics(results: List[CustomPromptResult]) -> Dict[str, Any]:
        """Compare metrics across multiple executions.

        Returns:
            Statistical comparison of metrics
        """
        if not results:
            return {"error": "No results to compare"}

        comparison = {"execution_count": len(results), "metric_trends": {}, "model_rankings": {}}

        # Aggregate metrics by model
        model_metrics = {}
        for result in results:
            if result.metrics:
                for model, metrics in result.metrics.items():
                    if model not in model_metrics:
                        model_metrics[model] = []
                    model_metrics[model].append(metrics)

        # Calculate trends
        for model, metric_list in model_metrics.items():
            # Example: track coherence scores over time
            coherence_scores = []
            for metrics in metric_list:
                if isinstance(metrics, dict) and "coherence" in metrics:
                    if isinstance(metrics["coherence"], dict):
                        coherence_scores.append(metrics["coherence"].get("score", 0))

            if coherence_scores:
                import statistics

                comparison["metric_trends"][model] = {
                    "coherence_mean": statistics.mean(coherence_scores),
                    "coherence_std": statistics.stdev(coherence_scores)
                    if len(coherence_scores) > 1
                    else 0,
                }

        return comparison


# Convenience functions
def save_execution_result(
    exec_result: ExecutionResult,
    storage_dir: str | Path,
    format: str = "json",
    include_metrics: bool = True,
) -> Path:
    """Save an execution result with optional metrics.

    Args:
        exec_result: The execution result to save
        storage_dir: Directory to save results
        format: Output format
        include_metrics: Whether to calculate and include metrics

    Returns:
        Path to saved file
    """
    # Convert to CustomPromptResult
    custom_result = CustomPromptResult.from_execution_result(exec_result)

    # Add metrics if requested
    if include_metrics:
        from .evaluation_metrics import MetricSuite

        suite = MetricSuite()

        # Calculate individual metrics
        individual_metrics = {}
        for response in exec_result.responses:
            if response.success and response.response:
                individual_metrics[response.model] = suite.evaluate(response.response)

        custom_result.metrics = individual_metrics

        # Calculate aggregated metrics if multiple responses
        if len([r for r in exec_result.responses if r.success]) > 1:
            successful_responses = [
                r.response for r in exec_result.responses if r.success and r.response
            ]
            batch_results = suite.evaluate_batch(successful_responses)
            custom_result.aggregated_metrics = batch_results.get("aggregated", {})

    # Save
    storage = ResultStorage(storage_dir)
    return storage.save(custom_result, format)


def view_result(file_path: str | Path) -> None:
    """View a saved result file in the console.

    Args:
        file_path: Path to the result file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    # Determine format from extension
    ext = file_path.suffix.lower()

    if ext == ".json":
        with open(file_path) as f:
            data = json.load(f)
        print(json.dumps(data, indent=2))
    elif ext in [".md", ".markdown"]:
        with open(file_path) as f:
            print(f.read())
    elif ext == ".csv":
        with open(file_path) as f:
            print(f.read())
    else:
        print(f"Unknown file format: {ext}")
