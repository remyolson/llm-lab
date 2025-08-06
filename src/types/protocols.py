"""
Protocol classes for structural typing in LLM Lab

This module provides Protocol classes that define interfaces for key components
using structural subtyping, enabling flexible implementations while maintaining
type safety.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

from typing_extensions import Doc, Protocol, runtime_checkable

# Import BaseModel for type annotation (avoid circular imports)
if TYPE_CHECKING:
    from pydantic import BaseModel

from .core import APIResponse, ErrorResponse, ModelParameters, ProviderInfo
from .custom import ExperimentId, ModelId, TaskId
from .evaluation import BenchmarkResult, EvaluationResult, MetricResult

# TypeVar definitions with proper bounds and constraints

# Generic type variables
T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")  # For key types
V = TypeVar("V")  # For value types

# Bounded type variables for evaluation system
ResultType = TypeVar("ResultType", bound=EvaluationResult)
MetricType = TypeVar("MetricType", bound=MetricResult)
BenchmarkType = TypeVar("BenchmarkType", bound=BenchmarkResult)

# Bounded type variables for configuration system
ConfigType = TypeVar("ConfigType", bound=Dict[str, Any])
SettingsType = TypeVar("SettingsType", bound="BaseModel")  # For Pydantic models
ProviderConfigType = TypeVar("ProviderConfigType", bound=Dict[str, Any])

# Bounded type variables for provider system
ProviderType = TypeVar("ProviderType", bound="Provider")
ResponseType = TypeVar("ResponseType", bound=str)  # Provider response type
ModelInfoType = TypeVar("ModelInfoType", bound=ProviderInfo)

# Bounded type variables for data processing
DataType = TypeVar("DataType")  # For generic data processing
ProcessorInputType = TypeVar("ProcessorInputType")
ProcessorOutputType = TypeVar("ProcessorOutputType")

# Constrained type variables for specific value types
NumericType = TypeVar("NumericType", int, float)  # Only int or float
StringOrBytesType = TypeVar("StringOrBytesType", str, bytes)  # Only str or bytes
IdentifierType = TypeVar("IdentifierType", str, ModelId, TaskId, ExperimentId)  # ID types

# Collection type variables
SequenceType = TypeVar("SequenceType", bound="Sequence")
MappingType = TypeVar("MappingType", bound="Mapping")
IterableType = TypeVar("IterableType", bound="Iterable")

# Protocol-bounded type variables
SerializableType = TypeVar("SerializableType", bound="Serializable")
ComparableType = TypeVar("ComparableType", bound="Comparable")
HashableType = TypeVar("HashableType", bound="Hashable")


# Annotated literal types with metadata for protocols
EvaluationMethodType = Annotated[
    Literal["semantic_similarity", "exact_match", "fuzzy_match", "bleu", "rouge", "custom"],
    Doc("Method used for evaluation"),
]

BenchmarkNameType = Annotated[
    Literal["glue", "superglue", "hellaswag", "arc", "mmlu", "custom"],
    Doc("Name of the benchmark dataset"),
]

ModelCapabilityType = Annotated[
    Literal["text_generation", "embeddings", "chat_completion", "function_calling", "fine_tuning"],
    Doc("Capability supported by the model"),
]

ProviderNameType = Annotated[
    Literal["openai", "anthropic", "google", "azure", "huggingface", "local", "custom"],
    Doc("Name of the LLM provider"),
]

ProcessingStatusType = Annotated[
    Literal["pending", "processing", "completed", "failed", "cancelled"],
    Doc("Status of a processing operation"),
]

QualityLevelType = Annotated[
    Literal["low", "medium", "high", "premium"], Doc("Quality level for processing or evaluation")
]

PriorityLevelType = Annotated[
    Literal["low", "normal", "high", "urgent"], Doc("Priority level for task scheduling")
]

ResourceTypeType = Annotated[
    Literal["cpu", "gpu", "memory", "storage", "network"], Doc("Type of computational resource")
]

DataFormatType = Annotated[
    Literal["json", "jsonl", "csv", "parquet", "hdf5", "pickle"],
    Doc("Format for data serialization"),
]

CompressionMethodType = Annotated[
    Literal["none", "gzip", "bzip2", "lzma", "zstd"], Doc("Data compression method")
]


@runtime_checkable
class Provider(Protocol):
    """Protocol for LLM providers.

    Defines the interface that all LLM provider implementations must follow.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text response from the model."""
        ...

    def get_model_info(self) -> ProviderInfo:
        """Get information about the current model."""
        ...

    def validate_credentials(self) -> bool:
        """Validate provider credentials."""
        ...

    def get_default_parameters(self) -> ModelParameters:
        """Get default generation parameters."""
        ...

    @property
    def model_name(self) -> str:
        """Current model name."""
        ...

    @property
    def provider_name(self) -> str:
        """Provider name identifier."""
        ...


@runtime_checkable
class BatchProvider(Protocol):
    """Protocol for providers supporting batch generation."""

    def batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate responses for multiple prompts."""
        ...

    def get_batch_size_limit(self) -> int:
        """Get maximum supported batch size."""
        ...


@runtime_checkable
class StreamingProvider(Protocol):
    """Protocol for providers supporting streaming responses."""

    def stream_generate(self, prompt: str, **kwargs: Any):
        """Generate streaming response."""
        ...

    def supports_streaming(self) -> bool:
        """Check if streaming is supported."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for evaluation methods.

    Defines the interface for all evaluation implementations.
    """

    def evaluate(self, prompt: str, response: str, **kwargs: Any) -> EvaluationResult:
        """Evaluate a model response."""
        ...

    def batch_evaluate(self, data: List[Dict[str, str]]) -> List[EvaluationResult]:
        """Evaluate multiple prompt-response pairs."""
        ...

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about the evaluation method."""
        ...

    @property
    def method_name(self) -> str:
        """Evaluation method name."""
        ...

    @property
    def supports_batch(self) -> bool:
        """Whether batch evaluation is supported."""
        ...


@runtime_checkable
class BenchmarkRunner(Protocol):
    """Protocol for benchmark execution.

    Defines interface for running standardized benchmarks.
    """

    def run_benchmark(self, model: Provider, benchmark_name: str, **kwargs: Any) -> BenchmarkResult:
        """Run a specific benchmark."""
        ...

    def list_benchmarks(self) -> List[str]:
        """List available benchmarks."""
        ...

    def get_benchmark_info(self, benchmark_name: str) -> Dict[str, Any]:
        """Get benchmark metadata."""
        ...


@runtime_checkable
class MetricCalculator(Protocol):
    """Protocol for metric calculation.

    Defines interface for computing specific metrics.
    """

    def calculate(self, **kwargs: Any) -> MetricResult:
        """Calculate the metric value."""
        ...

    def supports_data_type(self, data_type: str) -> bool:
        """Check if metric supports specific data type."""
        ...

    @property
    def metric_name(self) -> str:
        """Metric name identifier."""
        ...

    @property
    def higher_is_better(self) -> bool:
        """Whether higher values indicate better performance."""
        ...


@runtime_checkable
class Logger(Protocol):
    """Protocol for result logging.

    Defines interface for logging evaluation results and data.
    """

    def log_result(self, result: EvaluationResult) -> None:
        """Log an evaluation result."""
        ...

    def log_benchmark(self, result: BenchmarkResult) -> None:
        """Log a benchmark result."""
        ...

    def export_results(self, format: str) -> str:
        """Export logged results in specified format."""
        ...

    def clear_results(self) -> None:
        """Clear all logged results."""
        ...


@runtime_checkable
class ConfigurationManager(Protocol):
    """Protocol for configuration management.

    Defines interface for loading and managing configurations.
    """

    def load_configuration(self, **kwargs: Any) -> Dict[str, Any]:
        """Load configuration from sources."""
        ...

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
        ...

    def get_config_summary(self) -> Dict[str, str]:
        """Get configuration summary."""
        ...

    def reload_configuration(self) -> Dict[str, Any]:
        """Reload configuration from sources."""
        ...


@runtime_checkable
class CacheProvider(Protocol):
    """Protocol for caching implementations.

    Defines interface for caching data and results.
    """

    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL."""
        ...

    def delete(self, key: str) -> bool:
        """Delete cached value."""
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


@runtime_checkable
class DataValidator(Protocol):
    """Protocol for data validation.

    Defines interface for validating various data types.
    """

    def validate(self, data: Any, **kwargs: Any) -> bool:
        """Validate data according to rules."""
        ...

    def get_validation_errors(self, data: Any, **kwargs: Any) -> List[str]:
        """Get detailed validation errors."""
        ...

    def supports_type(self, data_type: type) -> bool:
        """Check if validator supports data type."""
        ...


@runtime_checkable
class ModelManager(Protocol):
    """Protocol for model management.

    Defines interface for loading and managing models.
    """

    def load_model(self, model_id: ModelId) -> Provider:
        """Load a model by ID."""
        ...

    def unload_model(self, model_id: ModelId) -> None:
        """Unload a model."""
        ...

    def list_models(self) -> List[ModelId]:
        """List available models."""
        ...

    def get_model_info(self, model_id: ModelId) -> ProviderInfo:
        """Get model information."""
        ...

    def is_model_loaded(self, model_id: ModelId) -> bool:
        """Check if model is currently loaded."""
        ...


@runtime_checkable
class ExperimentTracker(Protocol):
    """Protocol for experiment tracking.

    Defines interface for tracking experiments and runs.
    """

    def start_experiment(self, experiment_id: ExperimentId, **metadata: Any) -> None:
        """Start tracking an experiment."""
        ...

    def log_metric(self, experiment_id: ExperimentId, metric_name: str, value: float) -> None:
        """Log a metric for an experiment."""
        ...

    def log_parameter(self, experiment_id: ExperimentId, param_name: str, value: Any) -> None:
        """Log a parameter for an experiment."""
        ...

    def finish_experiment(self, experiment_id: ExperimentId) -> None:
        """Mark experiment as finished."""
        ...

    def get_experiment_info(self, experiment_id: ExperimentId) -> Dict[str, Any]:
        """Get experiment metadata and results."""
        ...


@runtime_checkable
class TaskRunner(Protocol[T]):
    """Generic protocol for task execution.

    Defines interface for running various types of tasks.
    """

    def run_task(self, task_id: TaskId, **kwargs: Any) -> T:
        """Execute a task and return result."""
        ...

    def get_task_status(self, task_id: TaskId) -> str:
        """Get current task status."""
        ...

    def cancel_task(self, task_id: TaskId) -> bool:
        """Cancel a running task."""
        ...

    def list_tasks(self) -> List[TaskId]:
        """List all tasks."""
        ...


@runtime_checkable
class Plugin(Protocol):
    """Protocol for plugin implementations.

    Defines base interface for all plugin types.
    """

    def initialize(self, **config: Any) -> None:
        """Initialize the plugin."""
        ...

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        ...

    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        ...

    @property
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    def version(self) -> str:
        """Plugin version."""
        ...


@runtime_checkable
class APIClient(Protocol):
    """Protocol for API client implementations.

    Defines interface for making API requests.
    """

    def request(self, endpoint: str, method: str = "GET", **kwargs: Any) -> APIResponse:
        """Make an API request."""
        ...

    def get(self, endpoint: str, **kwargs: Any) -> APIResponse:
        """Make a GET request."""
        ...

    def post(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> APIResponse:
        """Make a POST request."""
        ...

    def put(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> APIResponse:
        """Make a PUT request."""
        ...

    def delete(self, endpoint: str, **kwargs: Any) -> APIResponse:
        """Make a DELETE request."""
        ...

    def handle_error(self, error: Exception) -> ErrorResponse:
        """Handle API errors."""
        ...


# Generic protocols for common patterns
@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        ...

    def to_json(self) -> str:
        """Convert to JSON string."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Create instance from dictionary."""
        ...


@runtime_checkable
class Comparable(Protocol):
    """Protocol for comparable objects."""

    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...


@runtime_checkable
class Hashable(Protocol):
    """Protocol for hashable objects."""

    def __hash__(self) -> int: ...
    def __eq__(self, other: Any) -> bool: ...
