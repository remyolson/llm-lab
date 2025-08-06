"""
Final Annotations and Method Overloads

This module provides comprehensive final decorators, method overloads,
and advanced type annotations to complete the type safety system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    final,
    get_type_hints,
    overload,
)

from typing_extensions import Never, assert_never, override

from .core import ModelParameters, ProviderInfo
from .custom import APIKey, ModelName, ProviderName
from .evaluation import EvaluationResult, MetricResult

T = TypeVar("T")
R = TypeVar("R")
E = TypeVar("E", bound=Exception)


# Final utility classes that should not be subclassed
@final
class ImmutableConfig:
    """Immutable configuration class that cannot be subclassed or modified."""

    def __init__(self, **kwargs: Any):
        object.__setattr__(self, "_data", dict(kwargs))
        object.__setattr__(self, "_frozen", True)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> Never:
        if getattr(self, "_frozen", False):
            raise AttributeError(f"Cannot modify immutable configuration")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> Never:
        raise AttributeError(f"Cannot delete attributes from immutable configuration")

    @final
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (returns a copy)."""
        return dict(self._data)

    @final
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._data.get(key, default)

    @final
    def keys(self) -> Iterator[str]:
        """Get configuration keys."""
        return iter(self._data.keys())

    @final
    def values(self) -> Iterator[Any]:
        """Get configuration values."""
        return iter(self._data.values())

    @final
    def items(self) -> Iterator[tuple[str, Any]]:
        """Get configuration items."""
        return iter(self._data.items())


@final
class SingletonMeta(type):
    """Metaclass for creating singleton instances."""

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@final
@dataclass(frozen=True)
class TypedResult(Generic[T]):
    """Immutable result container with type safety."""

    value: T
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @final
    @classmethod
    def success_result(cls, value: T, metadata: Optional[Dict[str, Any]] = None) -> TypedResult[T]:
        """Create a successful result."""
        return cls(value=value, success=True, metadata=metadata or {})

    @final
    @classmethod
    def error_result(cls, error: str, metadata: Optional[Dict[str, Any]] = None) -> TypedResult[T]:
        """Create an error result."""
        return cls(value=None, success=False, error=error, metadata=metadata or {})

    @final
    def unwrap(self) -> T:
        """Unwrap the value, raising an exception if error."""
        if not self.success or self.error:
            raise ValueError(f"Cannot unwrap error result: {self.error}")
        return self.value

    @final
    def unwrap_or(self, default: T) -> T:
        """Unwrap the value or return default if error."""
        return self.value if self.success else default


# Abstract base classes with final methods
class BaseProcessor(ABC, Generic[T, R]):
    """Base processor with final common methods."""

    @abstractmethod
    def process(self, data: T) -> R:
        """Process data (must be implemented by subclasses)."""
        ...

    @final
    def process_batch(self, data_list: List[T]) -> List[R]:
        """Process a batch of data (cannot be overridden)."""
        return [self.process(item) for item in data_list]

    @final
    def process_with_error_handling(self, data: T, default: Optional[R] = None) -> TypedResult[R]:
        """Process data with error handling (cannot be overridden)."""
        try:
            result = self.process(data)
            return TypedResult.success_result(result)
        except Exception as e:
            return TypedResult.error_result(str(e))

    @final
    def validate_input(self, data: T) -> bool:
        """Validate input data (can be extended but not overridden)."""
        return data is not None


class BaseEvaluator(ABC):
    """Base evaluator with final common methods."""

    @abstractmethod
    def evaluate(self, prediction: str, target: str) -> float:
        """Evaluate prediction against target."""
        ...

    @final
    def evaluate_batch(self, predictions: List[str], targets: List[str]) -> List[float]:
        """Evaluate batch (cannot be overridden)."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        return [self.evaluate(pred, tgt) for pred, tgt in zip(predictions, targets)]

    @final
    def aggregate_scores(self, scores: List[float]) -> Dict[str, float]:
        """Aggregate scores (cannot be overridden)."""
        if not scores:
            return {"mean": 0.0, "std": 0.0, "count": 0}

        import statistics

        return {
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }


# Method overload patterns for common operations
class EnhancedModelInterface:
    """Enhanced model interface with comprehensive overloads."""

    # Simple generation overload
    @overload
    def generate(self, prompt: str) -> str: ...

    # Generation with parameters overload
    @overload
    def generate(self, prompt: str, temperature: float) -> str: ...

    # Generation with full parameters overload
    @overload
    def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 100, top_p: float = 1.0
    ) -> str: ...

    # Generation with metadata return overload
    @overload
    def generate(
        self, prompt: str, return_metadata: Literal[True], **kwargs: Any
    ) -> Dict[str, Any]: ...

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_p: float = 1.0,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> str | Dict[str, Any]:
        """Generate text with comprehensive overloading."""
        # Implementation would go here
        response = f"Generated response for: {prompt}"

        if return_metadata:
            return {
                "response": response,
                "prompt": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    **kwargs,
                },
            }
        return response

    # Batch generation overloads
    @overload
    def batch_generate(self, prompts: List[str]) -> List[str]: ...

    @overload
    def batch_generate(
        self, prompts: List[str], return_metadata: Literal[True]
    ) -> List[Dict[str, Any]]: ...

    def batch_generate(
        self, prompts: List[str], return_metadata: bool = False, **kwargs: Any
    ) -> List[str] | List[Dict[str, Any]]:
        """Batch generate with overloads."""
        if return_metadata:
            return [self.generate(prompt, return_metadata=True, **kwargs) for prompt in prompts]
        else:
            return [self.generate(prompt, **kwargs) for prompt in prompts]


class EnhancedEvaluationInterface:
    """Enhanced evaluation interface with comprehensive overloads."""

    # Simple evaluation overload
    @overload
    def evaluate(self, prediction: str, target: str) -> float: ...

    # Evaluation with method specification
    @overload
    def evaluate(self, prediction: str, target: str, method: Literal["exact_match"]) -> float: ...

    @overload
    def evaluate(
        self, prediction: str, target: str, method: Literal["fuzzy_match"], threshold: float = 0.8
    ) -> float: ...

    # Evaluation with detailed results
    @overload
    def evaluate(
        self, prediction: str, target: str, return_details: Literal[True]
    ) -> EvaluationResult: ...

    def evaluate(
        self,
        prediction: str,
        target: str,
        method: str = "exact_match",
        threshold: float = 0.8,
        return_details: bool = False,
        **kwargs: Any,
    ) -> float | EvaluationResult:
        """Evaluate with comprehensive overloading."""
        # Simple scoring logic
        if method == "exact_match":
            score = 1.0 if prediction.strip().lower() == target.strip().lower() else 0.0
        elif method == "fuzzy_match":
            # Simplified fuzzy matching
            from difflib import SequenceMatcher

            score = SequenceMatcher(None, prediction.lower(), target.lower()).ratio()
            score = 1.0 if score >= threshold else 0.0
        else:
            score = 0.5  # Default score

        if return_details:
            return EvaluationResult(
                {
                    "score": score,
                    "method": method,
                    "prediction": prediction,
                    "target": target,
                    "parameters": {"threshold": threshold, **kwargs},
                }
            )
        return score


# Advanced overload patterns
class FlexibleConfigLoader:
    """Configuration loader with flexible overloads."""

    @overload
    def load(self, source: str) -> Dict[str, Any]: ...

    @overload
    def load(self, source: Dict[str, Any]) -> Dict[str, Any]: ...

    @overload
    def load(self, source: str, format: Literal["json"]) -> Dict[str, Any]: ...

    @overload
    def load(self, source: str, format: Literal["yaml"]) -> Dict[str, Any]: ...

    @overload
    def load(self, source: str, strict: Literal[True]) -> ImmutableConfig: ...

    def load(
        self,
        source: str | Dict[str, Any],
        format: str = "auto",
        strict: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any] | ImmutableConfig:
        """Load configuration with flexible input handling."""
        if isinstance(source, dict):
            config_data = source
        else:
            # Simplified loading logic
            config_data = {"loaded_from": source, "format": format}

        if strict:
            return ImmutableConfig(**config_data)
        return config_data


# Final decorator for methods that should not be overridden
def final_method(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to mark methods as final (not overrideable)."""
    func.__final__ = True
    return func


# Runtime final checking decorator
def enforce_final(cls: type) -> type:
    """Class decorator to enforce final methods at runtime."""
    original_init_subclass = cls.__init_subclass__

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Check if any final methods are being overridden
        for base in cls.__mro__[1:]:  # Skip self
            for name, method in base.__dict__.items():
                if hasattr(method, "__final__") and name in cls.__dict__:
                    raise TypeError(f"Cannot override final method {name} in {cls.__name__}")

        if original_init_subclass:
            original_init_subclass(**kwargs)

    cls.__init_subclass__ = __init_subclass__
    return cls


# Never type utilities for exhaustive checking
def assert_exhaustive(value: Never) -> Never:
    """Assert that a value should never be reached."""
    assert_never(value)


def check_exhaustive_enum(value: Any, enum_class: type) -> None:
    """Check that all enum values are handled."""
    if not isinstance(value, enum_class):
        raise TypeError(f"Expected {enum_class}, got {type(value)}")


# Type-safe builder pattern with final methods
class TypedBuilder(Generic[T]):
    """Builder pattern with type safety and final methods."""

    def __init__(self, target_type: type[T]):
        self._target_type = target_type
        self._data: Dict[str, Any] = {}

    @final
    def set(self, key: str, value: Any) -> TypedBuilder[T]:
        """Set a value (cannot be overridden)."""
        self._data[key] = value
        return self

    @final
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value (cannot be overridden)."""
        return self._data.get(key, default)

    @final
    def build(self) -> T:
        """Build the target object (cannot be overridden)."""
        try:
            return self._target_type(**self._data)
        except TypeError as e:
            raise ValueError(f"Cannot build {self._target_type.__name__}: {e}")

    def validate(self) -> bool:
        """Validate the builder state (can be overridden)."""
        return True


# Final validation utilities
@final
class TypeValidator:
    """Final validation utilities that cannot be subclassed."""

    @staticmethod
    def validate_type(value: Any, expected_type: type) -> bool:
        """Validate that value is of expected type."""
        return isinstance(value, expected_type)

    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float) -> bool:
        """Validate that value is within range."""
        return min_val <= value <= max_val

    @staticmethod
    def validate_enum(value: Any, enum_class: type) -> bool:
        """Validate that value is valid enum member."""
        try:
            return value in enum_class.__members__.values()
        except AttributeError:
            return False

    @staticmethod
    def validate_callable(value: Any, signature: Optional[str] = None) -> bool:
        """Validate that value is callable with optional signature check."""
        if not callable(value):
            return False

        if signature:
            try:
                import inspect

                sig = inspect.signature(value)
                return str(sig) == signature
            except (ValueError, TypeError):
                return False

        return True


# Comprehensive overload example for a complete interface
class CompleteProviderInterface:
    """Complete provider interface with all overload patterns."""

    # Initialize overloads
    @overload
    def __init__(self, model_name: str): ...

    @overload
    def __init__(self, model_name: str, api_key: str): ...

    @overload
    def __init__(self, model_name: str, config: Dict[str, Any]): ...

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize provider with flexible arguments."""
        self.model_name = model_name
        self.api_key = api_key
        self.config = config or {}
        self.config.update(kwargs)

    # Generation method with all overload patterns
    @overload
    def generate(self, prompt: str) -> str: ...

    @overload
    def generate(self, prompt: str, stream: Literal[True]) -> Iterator[str]: ...

    @overload
    def generate(self, prompt: str, return_usage: Literal[True]) -> Dict[str, Any]: ...

    @overload
    def generate(
        self, prompt: str, stream: Literal[True], return_usage: Literal[True]
    ) -> Iterator[Dict[str, Any]]: ...

    def generate(
        self, prompt: str, stream: bool = False, return_usage: bool = False, **kwargs: Any
    ) -> str | Iterator[str] | Dict[str, Any] | Iterator[Dict[str, Any]]:
        """Generate with comprehensive overloading support."""
        # Simplified implementation for demonstration
        base_response = f"Response to: {prompt}"
        usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": 10}

        if stream and return_usage:
            # Return iterator of dicts with usage info
            def stream_with_usage():
                for chunk in [f"Chunk {i}" for i in range(3)]:
                    yield {"content": chunk, "usage": usage}

            return stream_with_usage()
        elif stream:
            # Return iterator of strings
            def stream_content():
                for chunk in [f"Chunk {i}" for i in range(3)]:
                    yield chunk

            return stream_content()
        elif return_usage:
            # Return dict with usage
            return {"content": base_response, "usage": usage}
        else:
            # Return simple string
            return base_response
