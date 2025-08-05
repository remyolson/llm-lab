"""Custom Prompts Use Case Module.

This module provides functionality for running custom prompts across multiple LLM providers
with support for template variables, evaluation metrics, and result storage.
"""

from .template_engine import PromptTemplate, TemplateError, ValidationError
from .prompt_runner import (
    PromptRunner,
    ModelResponse,
    ExecutionResult,
    run_prompt_on_models,
)
from .evaluation_metrics import (
    MetricResult,
    BaseMetric,
    ResponseLengthMetric,
    SentimentMetric,
    CoherenceMetric,
    ResponseDiversityMetric,
    CustomMetric,
    MetricSuite,
    evaluate_response,
    evaluate_responses,
)
from .result_storage import (
    CustomPromptResult,
    ResultFormatter,
    JSONFormatter,
    CSVFormatter,
    MarkdownFormatter,
    ResultStorage,
    ResultComparator,
    save_execution_result,
    view_result,
)

__all__ = [
    # Template Engine
    "PromptTemplate",
    "TemplateError",
    "ValidationError",
    # Prompt Runner
    "PromptRunner",
    "ModelResponse",
    "ExecutionResult",
    "run_prompt_on_models",
    # Evaluation Metrics
    "MetricResult",
    "BaseMetric",
    "ResponseLengthMetric",
    "SentimentMetric",
    "CoherenceMetric",
    "ResponseDiversityMetric",
    "CustomMetric",
    "MetricSuite",
    "evaluate_response",
    "evaluate_responses",
    # Result Storage
    "CustomPromptResult",
    "ResultFormatter",
    "JSONFormatter",
    "CSVFormatter",
    "MarkdownFormatter",
    "ResultStorage",
    "ResultComparator",
    "save_execution_result",
    "view_result",
]