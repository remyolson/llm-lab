"""
Custom types and type aliases for LLM Lab

This module provides NewType definitions, type aliases, and custom type
definitions for domain-specific typing throughout the LLM Lab framework.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

from typing import Annotated, Any, Callable, Dict, List, Literal, NewType

from typing_extensions import Doc, TypeAlias

# Domain-specific ID types using NewType for strong typing
ModelId = NewType("ModelId", str)
TaskId = NewType("TaskId", str)
ExperimentId = NewType("ExperimentId", str)
ProviderId = NewType("ProviderId", str)
BenchmarkId = NewType("BenchmarkId", str)
EvaluationId = NewType("EvaluationId", str)
SessionId = NewType("SessionId", str)


# Complex type aliases for common patterns
ResponseData: TypeAlias = Dict[str, Any] | List[Any] | str | int | float | None
ValidationInput: TypeAlias = str | int | float | bool | List[str]
CacheValue: TypeAlias = str | int | float | bool | Dict[str, Any] | List[Any]
MetricValue: TypeAlias = int | float
ScoreValue: TypeAlias = int | float


# JSON-serializable types
JSONSerializable: TypeAlias = (
    str | int | float | bool | None | Dict[str, "JSONSerializable"] | List["JSONSerializable"]
)


# Configuration value types (more specific than the one in config.py)
ConfigValue: TypeAlias = str | int | float | bool | List[str]
NestedConfigValue: TypeAlias = str | int | float | bool | List[str] | Dict[str, "NestedConfigValue"]


# Provider-specific types with Literal types
ProviderName: TypeAlias = Literal["openai", "anthropic", "google", "local", "custom"]
ModelName: TypeAlias = Annotated[str, Doc("Model name identifier (e.g., 'gpt-4', 'claude-3')")]
APIKey: TypeAlias = Annotated[
    str, Doc("API key for provider authentication - should be kept secure")
]


# Evaluation-specific types with Literal constraints
EvaluationMethod: TypeAlias = Literal[
    "semantic_similarity", "exact_match", "fuzzy_match", "bleu", "rouge", "bert_score", "custom"
]
BenchmarkName: TypeAlias = Literal["glue", "superglue", "hellaswag", "arc", "mmlu", "custom"]


# File and path types
FilePath: TypeAlias = str
DirectoryPath: TypeAlias = str
URL: TypeAlias = str


# Time and duration types
Timestamp: TypeAlias = str  # ISO format timestamp
Duration: TypeAlias = float  # Duration in seconds


# Numeric constraint types with Annotated metadata
Percentage: TypeAlias = Annotated[float, Doc("Percentage value between 0.0 and 1.0")]
Temperature: TypeAlias = Annotated[float, Doc("Model temperature parameter, typically 0.0 to 2.0")]
Probability: TypeAlias = Annotated[float, Doc("Probability value between 0.0 and 1.0")]


# Batch and processing types
BatchSize: TypeAlias = int
WorkerCount: TypeAlias = int
BufferSize: TypeAlias = int


# Network and server types
Port: TypeAlias = int
HostAddress: TypeAlias = str
TimeoutSeconds: TypeAlias = int


# Status and state types with Literal constraints
Status: TypeAlias = Literal["pending", "running", "completed", "failed", "cancelled"]
LogLevel: TypeAlias = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment: TypeAlias = Literal["development", "testing", "staging", "production"]


# Function signature types for common patterns
ErrorHandler: TypeAlias = (
    None | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], bool]
)

RetryStrategy: TypeAlias = Literal["fixed", "exponential", "linear", "custom"]


# Generic callback types
Callback: TypeAlias = Callable[[], None]
DataCallback: TypeAlias = Callable[[Any], None]
ErrorCallback: TypeAlias = Callable[[Exception], None]
ProgressCallback: TypeAlias = Callable[[float], None]  # 0.0 to 1.0


# Validation types
ValidationRule: TypeAlias = Callable[[Any], bool]
ValidationMessage: TypeAlias = str
ValidationError: TypeAlias = tuple[ValidationMessage, Any]  # (message, invalid_value)


# Chart and visualization types (for analytics components)
ChartType: TypeAlias = Literal["line", "bar", "scatter", "histogram", "pie", "heatmap"]
ChartData: TypeAlias = Dict[str, List[float] | List[str] | List[int]]


# Plugin and extension types
PluginName: TypeAlias = str
PluginVersion: TypeAlias = str
ExtensionPoint: TypeAlias = str


# Training and fine-tuning types
LearningRate: TypeAlias = float
EpochCount: TypeAlias = int
StepCount: TypeAlias = int
LossValue: TypeAlias = float


# Memory and resource types
MemorySize: TypeAlias = int  # in bytes
GPUMemorySize: TypeAlias = int  # in bytes
CPUCount: TypeAlias = int


# Comparison and ranking types
RankingScore: TypeAlias = float
ComparisonResult: TypeAlias = Literal["better", "worse", "equivalent", "incomparable"]
ConfidenceScore: TypeAlias = Annotated[float, Doc("Confidence score between 0.0 and 1.0")]


# Annotated types with validation constraints and metadata
ValidatedEmail: TypeAlias = Annotated[str, Doc("Email address that should be validated before use")]

ValidatedURL: TypeAlias = Annotated[str, Doc("URL that should be validated and accessible")]

ValidatedFilePath: TypeAlias = Annotated[str, Doc("File path that should exist and be readable")]

ValidatedDirectoryPath: TypeAlias = Annotated[
    str, Doc("Directory path that should exist and be accessible")
]

PositiveInt: TypeAlias = Annotated[int, Doc("Positive integer greater than 0")]

NonNegativeInt: TypeAlias = Annotated[int, Doc("Non-negative integer (>= 0)")]

PositiveFloat: TypeAlias = Annotated[float, Doc("Positive float greater than 0.0")]

NonNegativeFloat: TypeAlias = Annotated[float, Doc("Non-negative float (>= 0.0)")]

BoundedFloat: TypeAlias = Annotated[float, Doc("Float value with specified bounds")]

NormalizedString: TypeAlias = Annotated[
    str, Doc("String that has been normalized (lowercase, stripped)")
]

SanitizedString: TypeAlias = Annotated[str, Doc("String that has been sanitized for safe usage")]

HashedString: TypeAlias = Annotated[str, Doc("String that represents a hash value (e.g., SHA-256)")]

Base64String: TypeAlias = Annotated[str, Doc("Base64 encoded string")]

JSONString: TypeAlias = Annotated[str, Doc("String containing valid JSON data")]

SemVersion: TypeAlias = Annotated[str, Doc("Semantic version string (e.g., '1.2.3')")]

RegexPattern: TypeAlias = Annotated[str, Doc("String containing a valid regex pattern")]

MimeType: TypeAlias = Annotated[
    str, Doc("MIME type string (e.g., 'text/plain', 'application/json')")
]

LanguageCode: TypeAlias = Annotated[str, Doc("ISO language code (e.g., 'en', 'fr', 'zh')")]

CountryCode: TypeAlias = Annotated[str, Doc("ISO country code (e.g., 'US', 'GB', 'FR')")]

Timezone: TypeAlias = Annotated[str, Doc("Timezone identifier (e.g., 'America/New_York', 'UTC')")]

CurrencyCode: TypeAlias = Annotated[str, Doc("ISO currency code (e.g., 'USD', 'EUR', 'GBP')")]

ColorHex: TypeAlias = Annotated[str, Doc("Hexadecimal color code (e.g., '#FF0000' for red)")]

IPAddress: TypeAlias = Annotated[str, Doc("IP address (IPv4 or IPv6)")]

MACAddress: TypeAlias = Annotated[str, Doc("MAC address in standard format")]

UUID: TypeAlias = Annotated[str, Doc("Universally unique identifier string")]

# Model-specific annotated types
ModelConfigDict: TypeAlias = Annotated[
    Dict[str, Any], Doc("Dictionary containing model configuration parameters")
]

EvaluationResults: TypeAlias = Annotated[
    Dict[str, float | int | List[float]],
    Doc("Dictionary containing evaluation results with metric names as keys"),
]

BenchmarkResults: TypeAlias = Annotated[
    Dict[str, Dict[str, float]],
    Doc("Nested dictionary with benchmark names and their detailed results"),
]

TokenCount: TypeAlias = Annotated[int, Doc("Number of tokens (must be positive)")]

ModelParameters: TypeAlias = Annotated[
    Dict[str, float | int | str | bool], Doc("Dictionary containing model inference parameters")
]

PromptTemplate: TypeAlias = Annotated[
    str, Doc("Template string with placeholders for dynamic content")
]

SystemPrompt: TypeAlias = Annotated[
    str, Doc("System prompt that defines model behavior and constraints")
]

UserPrompt: TypeAlias = Annotated[str, Doc("User input prompt for the model")]

AssistantResponse: TypeAlias = Annotated[str, Doc("Generated response from the assistant/model")]

# Training and fine-tuning annotated types
DatasetPath: TypeAlias = Annotated[str, Doc("Path to training/evaluation dataset")]

CheckpointPath: TypeAlias = Annotated[str, Doc("Path to model checkpoint for saving/loading")]

LearningRateSchedule: TypeAlias = Annotated[
    str, Doc("Learning rate schedule strategy (e.g., 'linear', 'cosine', 'constant')")
]

Optimizer: TypeAlias = Annotated[str, Doc("Optimizer name (e.g., 'adam', 'sgd', 'adamw')")]

LossFunction: TypeAlias = Annotated[
    str, Doc("Loss function name (e.g., 'cross_entropy', 'mse', 'cosine')")
]

MetricName: TypeAlias = Annotated[
    str, Doc("Name of evaluation metric (e.g., 'accuracy', 'f1_score', 'bleu')")
]

HyperparameterName: TypeAlias = Annotated[str, Doc("Name of hyperparameter being tuned")]

HyperparameterValue: TypeAlias = Annotated[
    float | int | str | bool, Doc("Value of a hyperparameter")
]

# Resource and performance annotated types
GPUUtilization: TypeAlias = Annotated[float, Doc("GPU utilization percentage (0.0 to 100.0)")]

MemoryUsage: TypeAlias = Annotated[int, Doc("Memory usage in bytes")]

LatencyMs: TypeAlias = Annotated[float, Doc("Latency measurement in milliseconds")]

Throughput: TypeAlias = Annotated[float, Doc("Processing throughput (requests/tokens per second)")]

Bandwidth: TypeAlias = Annotated[float, Doc("Network bandwidth in bytes per second")]

# Quality and reliability annotated types
AccuracyScore: TypeAlias = Annotated[float, Doc("Accuracy score between 0.0 and 1.0")]

PrecisionScore: TypeAlias = Annotated[float, Doc("Precision score between 0.0 and 1.0")]

RecallScore: TypeAlias = Annotated[float, Doc("Recall score between 0.0 and 1.0")]

F1Score: TypeAlias = Annotated[float, Doc("F1 score between 0.0 and 1.0")]

BLEUScore: TypeAlias = Annotated[float, Doc("BLEU score between 0.0 and 1.0")]

ROUGEScore: TypeAlias = Annotated[float, Doc("ROUGE score between 0.0 and 1.0")]

Perplexity: TypeAlias = Annotated[float, Doc("Model perplexity (lower is better)")]

CosineSimilarity: TypeAlias = Annotated[float, Doc("Cosine similarity score between -1.0 and 1.0")]

SemanticSimilarity: TypeAlias = Annotated[
    float, Doc("Semantic similarity score between 0.0 and 1.0")
]
