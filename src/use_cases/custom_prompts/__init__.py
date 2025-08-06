"""Custom Prompts Use Case Module.

This module provides functionality for running custom prompts across multiple LLM providers
with support for template variables, evaluation metrics, and result storage.
"""

from .evaluation_metrics import (
    BaseMetric,
    CoherenceMetric,
    CustomMetric,
    MetricResult,
    MetricSuite,
    ResponseDiversityMetric,
    ResponseLengthMetric,
    SentimentMetric,
    evaluate_response,
    evaluate_responses,
)
from .prompt_runner import (
    ExecutionResult,
    ModelResponse,
    PromptRunner,
    run_prompt_on_models,
)
from .result_storage import (
    CSVFormatter,
    CustomPromptResult,
    JSONFormatter,
    MarkdownFormatter,
    ResultComparator,
    ResultFormatter,
    ResultStorage,
    save_execution_result,
    view_result,
)
from .template_engine import PromptTemplate, TemplateError, ValidationError

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
