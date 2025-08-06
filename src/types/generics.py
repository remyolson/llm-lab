"""
Generic base classes and utilities for LLM Lab

This module provides generic base classes that implement common patterns
with proper type constraints and bounds for better type safety.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from typing_extensions import Protocol

from .protocols import (
    DataType,
    IdentifierType,
    K,
    MetricType,
    NumericType,
    ProcessorInputType,
    ProcessorOutputType,
    ResultType,
    T,
    U,
    V,
)


class GenericProcessor(Generic[ProcessorInputType, ProcessorOutputType], ABC):
    """Generic base class for data processors with type safety.

    Provides a consistent interface for processing data with proper
    input and output type constraints.
    """

    @abstractmethod
    def process(self, data: ProcessorInputType) -> ProcessorOutputType:
        """Process input data and return output."""
        ...

    @abstractmethod
    def validate_input(self, data: ProcessorInputType) -> bool:
        """Validate input data before processing."""
        ...

    def batch_process(self, data_list: List[ProcessorInputType]) -> List[ProcessorOutputType]:
        """Process a batch of data items."""
        results = []
        for data in data_list:
            if self.validate_input(data):
                results.append(self.process(data))
            else:
                raise ValueError(f"Invalid input data: {data}")
        return results


class GenericEvaluator(Generic[T, ResultType], ABC):
    """Generic base class for evaluators with result type constraints.

    Ensures all evaluators return properly typed results that conform
    to the evaluation framework.
    """

    @abstractmethod
    def evaluate(self, data: T, **kwargs: Any) -> ResultType:
        """Evaluate data and return typed result."""
        ...

    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of this evaluation method."""
        ...

    def batch_evaluate(self, data_list: List[T], **kwargs: Any) -> List[ResultType]:
        """Evaluate a batch of data items."""
        return [self.evaluate(data, **kwargs) for data in data_list]


class GenericCache(Generic[K, V]):
    """Generic cache implementation with type-safe key-value operations."""

    def __init__(self, max_size: Optional[int] = None):
        self._cache: Dict[K, V] = {}
        self._max_size = max_size

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get value by key with optional default."""
        return self._cache.get(key, default)

    def set(self, key: K, value: V) -> None:
        """Set key-value pair in cache."""
        if self._max_size and len(self._cache) >= self._max_size:
            # Remove oldest item (FIFO eviction)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value

    def delete(self, key: K) -> bool:
        """Delete key from cache, return True if existed."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def keys(self) -> List[K]:
        """Get all cache keys."""
        return list(self._cache.keys())

    def values(self) -> List[V]:
        """Get all cache values."""
        return list(self._cache.values())


class GenericRepository(Generic[T, IdentifierType], ABC):
    """Generic repository pattern with type-safe operations."""

    @abstractmethod
    def get(self, id_: IdentifierType) -> Optional[T]:
        """Get item by ID."""
        ...

    @abstractmethod
    def save(self, item: T) -> IdentifierType:
        """Save item and return its ID."""
        ...

    @abstractmethod
    def delete(self, id_: IdentifierType) -> bool:
        """Delete item by ID, return True if existed."""
        ...

    @abstractmethod
    def list_all(self) -> List[T]:
        """List all items in repository."""
        ...

    def exists(self, id_: IdentifierType) -> bool:
        """Check if item exists by ID."""
        return self.get(id_) is not None

    def get_or_raise(self, id_: IdentifierType, exception_type: Type[Exception] = ValueError) -> T:
        """Get item by ID or raise exception if not found."""
        item = self.get(id_)
        if item is None:
            raise exception_type(f"Item not found with ID: {id_}")
        return item


class GenericFactory(Generic[T], ABC):
    """Generic factory pattern with type constraints."""

    @abstractmethod
    def create(self, **kwargs: Any) -> T:
        """Create new instance with given parameters."""
        ...

    @abstractmethod
    def supports_type(self, type_name: str) -> bool:
        """Check if factory supports creating given type."""
        ...

    def create_batch(self, count: int, **kwargs: Any) -> List[T]:
        """Create multiple instances with same parameters."""
        return [self.create(**kwargs) for _ in range(count)]


class GenericCollectionAggregator(Generic[T, U]):
    """Generic aggregator for collections with type safety."""

    def __init__(self, aggregation_func: callable[[List[T]], U]):
        self._aggregation_func = aggregation_func

    def aggregate(self, items: List[T]) -> U:
        """Aggregate list of items using configured function."""
        if not items:
            raise ValueError("Cannot aggregate empty collection")
        return self._aggregation_func(items)

    def aggregate_by_key(self, items: List[T], key_func: callable[[T], K]) -> Dict[K, U]:
        """Aggregate items grouped by key function."""
        groups: Dict[K, List[T]] = {}

        for item in items:
            key = key_func(item)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        return {key: self.aggregate(group_items) for key, group_items in groups.items()}


class GenericValidator(Generic[T], ABC):
    """Generic validator with type constraints."""

    @abstractmethod
    def validate(self, data: T) -> bool:
        """Validate data and return True if valid."""
        ...

    @abstractmethod
    def get_validation_errors(self, data: T) -> List[str]:
        """Get detailed validation error messages."""
        ...

    def validate_batch(self, data_list: List[T]) -> Dict[int, List[str]]:
        """Validate batch and return errors by index."""
        errors = {}
        for i, data in enumerate(data_list):
            item_errors = self.get_validation_errors(data)
            if item_errors:
                errors[i] = item_errors
        return errors

    def is_valid_batch(self, data_list: List[T]) -> bool:
        """Check if all items in batch are valid."""
        return all(self.validate(data) for data in data_list)


class GenericMetricCalculator(Generic[DataType, NumericType]):
    """Generic metric calculator with numeric type constraints."""

    def __init__(self, metric_name: str, calculation_func: callable[[DataType], NumericType]):
        self.metric_name = metric_name
        self._calculation_func = calculation_func

    def calculate(self, data: DataType) -> NumericType:
        """Calculate metric value for given data."""
        return self._calculation_func(data)

    def calculate_batch(self, data_list: List[DataType]) -> List[NumericType]:
        """Calculate metrics for batch of data."""
        return [self.calculate(data) for data in data_list]

    def calculate_statistics(self, data_list: List[DataType]) -> Dict[str, NumericType]:
        """Calculate basic statistics for batch of data."""
        values = self.calculate_batch(data_list)
        if not values:
            return {}

        # Convert to float for statistics calculations
        float_values = [float(v) for v in values]

        return {
            "count": len(values),
            "min": min(float_values),
            "max": max(float_values),
            "mean": sum(float_values) / len(float_values),
            "sum": sum(float_values),
        }


# Type-safe builder pattern
class GenericBuilder(Generic[T], ABC):
    """Generic builder pattern with type safety."""

    def __init__(self):
        self._reset()

    @abstractmethod
    def _reset(self) -> None:
        """Reset builder state."""
        ...

    @abstractmethod
    def build(self) -> T:
        """Build the final object."""
        ...

    def reset(self) -> "GenericBuilder[T]":
        """Reset builder and return self for chaining."""
        self._reset()
        return self
