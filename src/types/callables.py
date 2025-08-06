"""
Enhanced Callable Signatures and Type Guards for LLM Lab

This module provides sophisticated callable type definitions, TypeGuard functions,
and enhanced function signatures for better type safety at runtime.
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Callable,
    Concatenate,
    ContextManager,
    Generic,
    Iterator,
    ParamSpec,
    Protocol,
    TypeGuard,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

from typing_extensions import Annotated, Doc, Literal

from .core import APIResponse, ErrorResponse, ModelParameters, ProviderInfo
from .custom import (
    APIKey,
    BenchmarkName,
    EvaluationMethod,
    ModelName,
    ProviderName,
)
from .evaluation import BenchmarkResult, EvaluationResult, MetricResult
from .protocols import (
    PriorityLevelType,
    ProcessingStatusType,
    QualityLevelType,
)

# ParamSpec and TypeVar definitions for callable signatures
P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")
E = TypeVar("E", bound=Exception)

# Specialized TypeVars for different callable contexts
ModelT = TypeVar("ModelT", bound=str)
PromptT = TypeVar("PromptT", bound=str)
ResponseT = TypeVar("ResponseT", bound=str)
ConfigT = TypeVar("ConfigT", bound=dict)
ResultT = TypeVar("ResultT")


# Core function signatures for LLM operations
TextGenerationFunction = Annotated[
    Callable[[PromptT, ModelParameters], ResponseT],
    Doc("Function that generates text from a prompt using model parameters"),
]

BatchGenerationFunction = Annotated[
    Callable[[list[PromptT], ModelParameters], list[ResponseT]],
    Doc("Function that generates text for multiple prompts in batch"),
]

StreamingGenerationFunction = Annotated[
    Callable[[PromptT, ModelParameters], Iterator[str]],
    Doc("Function that generates text with streaming response"),
]

AsyncGenerationFunction = Annotated[
    Callable[[PromptT, ModelParameters], Awaitable[ResponseT]],
    Doc("Async function for text generation"),
]

AsyncStreamingGenerationFunction = Annotated[
    Callable[[PromptT, ModelParameters], AsyncIterator[str]],
    Doc("Async function that streams generated text"),
]


# Evaluation function signatures
EvaluationFunction = Annotated[
    Callable[[str, str], EvaluationResult],
    Doc("Function that evaluates model output against expected result"),
]

MetricCalculationFunction = Annotated[
    Callable[[Any, Any], float], Doc("Function that calculates a numeric metric from inputs")
]

BenchmarkFunction = Annotated[
    Callable[[str, dict], BenchmarkResult], Doc("Function that runs a benchmark evaluation")
]

CustomEvaluationFunction = Annotated[
    Callable[[str, dict], MetricResult], Doc("Custom evaluation function with flexible inputs")
]


# Callback function signatures
ProgressCallback = Annotated[
    Callable[[float], None], Doc("Callback function for progress updates (0.0 to 1.0)")
]

ErrorCallback = Annotated[Callable[[Exception], None], Doc("Callback function for error handling")]

CompletionCallback = Annotated[
    Callable[[ResultT], None], Doc("Callback function called upon successful completion")
]

StatusUpdateCallback = Annotated[
    Callable[[ProcessingStatusType, dict[str, Any]], None],
    Doc("Callback for status updates with metadata"),
]


# Validation function signatures
ValidationFunction = Annotated[
    Callable[[Any], bool], Doc("Function that validates an input and returns True/False")
]

TypeGuardFunction = Annotated[
    Callable[[Any], TypeGuard[T]], Doc("TypeGuard function for runtime type checking")
]

ValidatorFunction = Annotated[
    Callable[[T], T], Doc("Function that validates and returns the same type")
]

SanitizationFunction = Annotated[Callable[[str], str], Doc("Function that sanitizes string input")]


# Configuration and factory functions
ConfigurationLoader = Annotated[
    Callable[[str], dict[str, Any]], Doc("Function that loads configuration from a source")
]

ProviderFactory = Annotated[
    Callable[[ProviderName, ModelName], "Provider"],
    Doc("Factory function that creates provider instances"),
]

ModelFactory = Annotated[
    Callable[[ModelName, dict[str, Any]], Any], Doc("Factory function for creating model instances")
]


# Data processing function signatures
DataProcessor = Annotated[
    Callable[[T], R], Doc("Function that processes data from type T to type R")
]

BatchDataProcessor = Annotated[
    Callable[[list[T]], list[R]], Doc("Function that processes batches of data")
]

StreamingDataProcessor = Annotated[
    Callable[[Iterator[T]], Iterator[R]], Doc("Function that processes streaming data")
]

AsyncDataProcessor = Annotated[
    Callable[[T], Awaitable[R]], Doc("Async function for data processing")
]


# Filter and predicate functions
PredicateFunction = Annotated[
    Callable[[T], bool], Doc("Function that returns True/False for a given input")
]

FilterFunction = Annotated[
    Callable[[list[T]], list[T]], Doc("Function that filters a list based on criteria")
]

SortKeyFunction = Annotated[Callable[[T], Any], Doc("Function that extracts sort key from an item")]


# Aggregation and reduction functions
AggregationFunction = Annotated[
    Callable[[list[T]], R], Doc("Function that aggregates a list into a single result")
]

ReductionFunction = Annotated[Callable[[T, T], T], Doc("Function that reduces two items into one")]

AccumulatorFunction = Annotated[
    Callable[[R, T], R], Doc("Function for accumulating values (like in reduce)")
]


# Caching and memoization functions
CacheKeyFunction = Annotated[
    Callable[P, str], Doc("Function that generates cache keys from function arguments")
]

CacheSerializer = Annotated[
    Callable[[Any], str], Doc("Function that serializes values for caching")
]

CacheDeserializer = Annotated[Callable[[str], Any], Doc("Function that deserializes cached values")]


# Retry and resilience functions
RetryCondition = Annotated[
    Callable[[Exception], bool], Doc("Function that determines if an operation should be retried")
]

BackoffStrategy = Annotated[
    Callable[[int], float], Doc("Function that calculates retry delay based on attempt number")
]

CircuitBreakerPredicate = Annotated[
    Callable[[Exception], bool], Doc("Function that determines if circuit breaker should open")
]


# Protocol definitions for complex callable interfaces
@runtime_checkable
class GenerativeModel(Protocol):
    """Protocol for generative models with callable interface."""

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...

    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate text for multiple prompts."""
        ...

    def stream_generate(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Generate text with streaming output."""
        ...


@runtime_checkable
class EvaluationMetric(Protocol):
    """Protocol for evaluation metrics with callable interface."""

    def __call__(self, prediction: str, target: str) -> float:
        """Calculate metric score."""
        ...

    def batch_evaluate(self, predictions: list[str], targets: list[str]) -> list[float]:
        """Evaluate multiple prediction-target pairs."""
        ...


@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for data transformation with callable interface."""

    def __call__(self, data: T) -> R:
        """Transform data."""
        ...

    def inverse_transform(self, data: R) -> T:
        """Reverse transformation if possible."""
        ...


@runtime_checkable
class AsyncCallable(Protocol[P, T]):
    """Protocol for async callable objects."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[T]:
        """Async callable interface."""
        ...


@runtime_checkable
class ContextualCallable(Protocol[P, T]):
    """Protocol for callables that manage context."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> ContextManager[T]:
        """Return a context manager."""
        ...


# Advanced callable type constructors
def make_typed_callback(callback_type: type[T]) -> Callable[[Callable[..., Any]], Callable[..., T]]:
    """Create a typed callback decorator.

    Args:
        callback_type: Expected return type for the callback

    Returns:
        Decorator that ensures callback returns correct type
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)
            if not isinstance(result, callback_type):
                raise TypeError(f"Callback must return {callback_type}, got {type(result)}")
            return result

        return wrapper

    return decorator


def make_async_version(sync_func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """Convert a synchronous function to async version.

    Args:
        sync_func: Synchronous function to convert

    Returns:
        Async version of the function
    """
    import asyncio

    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: sync_func(*args, **kwargs))

    return async_wrapper


def make_memoized(
    func: Callable[P, T], cache_key_func: CacheKeyFunction[P] | None = None
) -> Callable[P, T]:
    """Create a memoized version of a function.

    Args:
        func: Function to memoize
        cache_key_func: Optional custom cache key function

    Returns:
        Memoized version of the function
    """
    cache: dict[str, T] = {}

    def memoized_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if cache_key_func:
            key = cache_key_func(*args, **kwargs)
        else:
            key = str(hash((args, tuple(sorted(kwargs.items())))))

        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_wrapper


def make_retry_wrapper(
    func: Callable[P, T],
    max_attempts: int = 3,
    retry_condition: RetryCondition | None = None,
    backoff_strategy: BackoffStrategy | None = None,
) -> Callable[P, T]:
    """Create a retry wrapper for a function.

    Args:
        func: Function to wrap with retry logic
        max_attempts: Maximum number of retry attempts
        retry_condition: Condition to determine if retry should happen
        backoff_strategy: Strategy for calculating retry delays

    Returns:
        Function with retry capabilities
    """
    import time

    def retry_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        last_exception = None

        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Check if we should retry
                if retry_condition and not retry_condition(e):
                    break

                # Don't sleep on the last attempt
                if attempt < max_attempts - 1:
                    if backoff_strategy:
                        delay = backoff_strategy(attempt)
                    else:
                        delay = 2**attempt  # Exponential backoff
                    time.sleep(delay)

        # If we get here, all attempts failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("All retry attempts failed")

    return retry_wrapper


# TypeGuard functions for runtime callable validation
def is_callable_with_signature(
    obj: Any, expected_args: int, expected_return_type: type[T] | None = None
) -> TypeGuard[Callable[..., T]]:
    """Check if object is callable with expected signature.

    Args:
        obj: Object to check
        expected_args: Expected number of arguments
        expected_return_type: Expected return type

    Returns:
        True if object matches callable signature
    """
    if not callable(obj):
        return False

    try:
        import inspect

        sig = inspect.signature(obj)
        if len(sig.parameters) != expected_args:
            return False

        if expected_return_type and sig.return_annotation != expected_return_type:
            return False

        return True
    except (ValueError, TypeError):
        return False


def is_async_callable(obj: Any) -> TypeGuard[Callable[..., Awaitable[Any]]]:
    """Check if object is an async callable.

    Args:
        obj: Object to check

    Returns:
        True if object is async callable
    """
    import asyncio

    return callable(obj) and asyncio.iscoroutinefunction(obj)


def is_generator_function(obj: Any) -> TypeGuard[Callable[..., Iterator[Any]]]:
    """Check if object is a generator function.

    Args:
        obj: Object to check

    Returns:
        True if object is a generator function
    """
    import inspect

    return callable(obj) and inspect.isgeneratorfunction(obj)


def is_context_manager_function(obj: Any) -> TypeGuard[Callable[..., ContextManager[Any]]]:
    """Check if object is a context manager function.

    Args:
        obj: Object to check

    Returns:
        True if object returns a context manager
    """
    if not callable(obj):
        return False

    try:
        import inspect

        sig = inspect.signature(obj)
        return_annotation = sig.return_annotation

        # Check if return type is ContextManager or has __enter__ and __exit__
        if hasattr(return_annotation, "__enter__") and hasattr(return_annotation, "__exit__"):
            return True

        # Check type hints
        if hasattr(return_annotation, "__origin__"):
            origin = return_annotation.__origin__
            return origin is ContextManager or origin is AsyncContextManager

        return False
    except (ValueError, TypeError, AttributeError):
        return False


# Higher-order function type constructors
def curry(func: Callable[..., T]) -> Callable[..., Callable[..., T]]:
    """Create a curried version of a function.

    Args:
        func: Function to curry

    Returns:
        Curried function
    """
    import functools

    @functools.wraps(func)
    def curried(*args: Any, **kwargs: Any) -> Callable[..., T] | T:
        try:
            return func(*args, **kwargs)
        except TypeError:
            # Not enough arguments, return partial function
            return functools.partial(func, *args, **kwargs)

    return curried


def compose(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose multiple functions into a single function.

    Args:
        *functions: Functions to compose (applied right to left)

    Returns:
        Composed function
    """

    def composed(x: Any) -> Any:
        result = x
        for func in reversed(functions):
            result = func(result)
        return result

    return composed


def pipe(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Pipe data through multiple functions (left to right).

    Args:
        *functions: Functions to pipe through (applied left to right)

    Returns:
        Piped function
    """

    def piped(x: Any) -> Any:
        result = x
        for func in functions:
            result = func(result)
        return result

    return piped
