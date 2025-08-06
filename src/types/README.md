# Type System Documentation - LLM Lab

This directory contains comprehensive type definitions for the LLM Lab framework, providing type safety, better IDE support, and clear data contracts.

## Overview

The LLM Lab type system replaces generic `Dict[str, Any]` and `Any` patterns with specific TypedDict classes, Protocol interfaces, and custom type definitions. This improves code quality, IDE autocomplete, and catches errors at development time.

## Architecture

### Module Structure

```
src/types/
├── __init__.py          # Public API exports
├── core.py             # Core infrastructure types
├── evaluation.py       # Evaluation and benchmark types
├── config.py           # Configuration management types
├── custom.py           # Custom types and aliases
├── protocols.py        # Protocol classes for interfaces
└── README.md           # This documentation
```

### Type Categories

1. **TypedDict Classes** - Structured data with required/optional fields
2. **Protocol Classes** - Interface definitions for structural typing
3. **NewType Definitions** - Strong typing for domain-specific IDs
4. **Type Aliases** - Complex union types and shortcuts
5. **Generic Types** - Parameterized types with bounds

## Core Types (`core.py`)

### Infrastructure Types

```python
from src.types import ProviderInfo, ModelParameters, APIResponse

# Provider information with type safety
provider_info: ProviderInfo = {
    "model_name": "gpt-4o-mini",
    "provider": "openai",
    "max_tokens": 4000,
    "capabilities": ["text", "chat"],
    # Optional fields
    "version": "1.0",
    "provider_id": "openai-1"
}

# Model parameters with validation
params: ModelParameters = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    # Optional evaluation parameters
    "eval_temperature_conservative": 0.1
}

# Standardized API responses
response: APIResponse = {
    "success": True,
    "data": {"result": "Generated text"},
    "timestamp": "2024-01-01T00:00:00Z"
}
```

### Configuration Types

```python
from src.types import ConfigDict, NetworkConfig, SystemConfig

# Nested configuration structure
config: ConfigDict = {
    "network": {
        "default_timeout": 30,
        "generation_timeout": 300
    },
    "system": {
        "default_batch_size": 8,
        "max_workers": 4
    }
}
```

## Evaluation Types (`evaluation.py`)

### Evaluation Results

```python
from src.types import EvaluationResult, BenchmarkResult, MetricResult

# Type-safe evaluation result
result: EvaluationResult = {
    "score": 0.85,
    "method": "semantic_similarity",
    "timestamp": "2024-01-01T00:00:00Z",
    # Optional fields
    "confidence": 0.92,
    "details": {"similarity_type": "cosine"}
}

# Benchmark results with model info
benchmark: BenchmarkResult = {
    "benchmark_name": "hellaswag",
    "task": "commonsense_reasoning",
    "score": 0.78,
    "metrics": {"accuracy": 0.78, "f1": 0.76},
    "model_info": provider_info  # Reuse ProviderInfo
}
```

### Method Combinations

```python
from src.types import MethodResult, CombinedResult

# Individual method results
method_results: List[MethodResult] = [
    {
        "method_name": "fuzzy_match",
        "score": 0.82,
        "weight": 0.3
    },
    {
        "method_name": "semantic_similarity",
        "score": 0.87,
        "weight": 0.7
    }
]

# Combined evaluation result
combined: CombinedResult = {
    "overall_score": 0.85,
    "method_results": method_results,
    "combination_method": "weighted_average"
}
```

## Configuration Types (`config.py`)

### Validation and Management

```python
from src.types import ValidationConfig, ValidationResult, ConfigurationSummary

# Validation configuration
validation: ValidationConfig = {
    "required_keys": ["api_key", "model_name"],
    "optional_keys": ["temperature", "max_tokens"],
    "strict_mode": True
}

# Validation results
result: ValidationResult = {
    "is_valid": False,
    "errors": [
        {
            "error_type": "missing_key",
            "message": "Required key 'api_key' not found",
            "field_path": "providers.openai.api_key"
        }
    ]
}
```

## Custom Types (`custom.py`)

### Strong ID Types

```python
from src.types import ModelId, TaskId, ExperimentId

# Strong typing prevents mixing up different ID types
model_id: ModelId = ModelId("gpt-4o-mini")
task_id: TaskId = TaskId("evaluation-001")
experiment_id: ExperimentId = ExperimentId("exp-20240101")

# Type checker will catch this error:
# task_id = model_id  # Error: Cannot assign ModelId to TaskId
```

### Type Aliases

```python
from src.types import ResponseData, ValidationInput, CacheValue

# Complex union types with clear meaning
def process_response(data: ResponseData) -> str:
    # data can be Dict, List, str, int, float, or None
    if isinstance(data, dict):
        return str(data.get("result", ""))
    elif isinstance(data, (list, str, int, float)):
        return str(data)
    else:
        return ""

# Validation inputs with specific types
def validate_input(value: ValidationInput) -> bool:
    # value can be str, int, float, bool, or List[str]
    return value is not None
```

## Protocol Classes (`protocols.py`)

### Structural Typing

```python
from src.types import Provider, Evaluator, Logger

# Any class implementing these methods satisfies the protocol
class MyCustomProvider:
    def generate(self, prompt: str, **kwargs) -> str:
        return "Generated text"

    def get_model_info(self) -> ProviderInfo:
        return {"model_name": "custom", "provider": "local", ...}

    def validate_credentials(self) -> bool:
        return True

    # ... other required methods

# Type checker knows this implements Provider protocol
provider: Provider = MyCustomProvider()
```

### Runtime Checking

```python
from src.types import Provider, Evaluator

# Runtime isinstance() checks with @runtime_checkable protocols
def use_provider(obj: Any) -> None:
    if isinstance(obj, Provider):
        # Safe to call provider methods
        result = obj.generate("Hello world")
        info = obj.get_model_info()
```

### Generic Protocols

```python
from src.types import TaskRunner
from src.types.custom import EvaluationResult

# Generic protocol with type parameter
runner: TaskRunner[EvaluationResult] = MyEvaluationRunner()
result: EvaluationResult = runner.run_task(task_id)
```

## Usage Patterns

### Migration from Generic Types

#### Before (Generic Types)
```python
def evaluate_response(response: str, data: Dict[str, Any]) -> Dict[str, Any]:
    # Unclear what 'data' contains or what's returned
    result = {"score": 0.8, "method": "fuzzy"}
    return result
```

#### After (Typed)
```python
def evaluate_response(response: str, config: ValidationConfig) -> EvaluationResult:
    # Clear input/output contracts
    result: EvaluationResult = {
        "score": 0.8,
        "method": "fuzzy_match",
        "timestamp": datetime.now().isoformat()
    }
    return result
```

### Optional Fields

```python
# Use NotRequired for optional fields
class MyTypedDict(TypedDict):
    required_field: str
    optional_field: NotRequired[int]

# Valid usage
data: MyTypedDict = {"required_field": "value"}  # OK
data_with_optional: MyTypedDict = {
    "required_field": "value",
    "optional_field": 42
}  # Also OK
```

### Inheritance

```python
class BaseResult(TypedDict):
    timestamp: str
    success: bool

class EvaluationResult(BaseResult):
    score: float
    method: str
    # Inherits timestamp and success from BaseResult
```

## IDE Support

### Autocomplete

With TypedDict classes, IDEs provide:
- **Field completion** when accessing dictionary keys
- **Type hints** for field values
- **Error detection** for missing required fields
- **Refactoring support** when renaming fields

### Error Detection

```python
# IDE will highlight errors
result: EvaluationResult = {
    "score": "invalid",  # Error: Expected float, got str
    "method": "fuzzy_match",
    # Missing required 'timestamp' field
}
```

## Runtime Validation

### TypedDict Structure Validation

```python
def validate_structure(data: Dict[str, Any], expected_type: type) -> bool:
    """Validate dictionary matches TypedDict structure."""
    if hasattr(expected_type, '__annotations__'):
        required_fields = getattr(expected_type, '__required_keys__', set())
        for field in required_fields:
            if field not in data:
                return False
    return True

# Usage
is_valid = validate_structure(result_dict, EvaluationResult)
```

### Protocol Validation

```python
from src.types import Provider

def validate_provider(obj: Any) -> bool:
    """Check if object implements Provider protocol."""
    return isinstance(obj, Provider)
```

## mypy Configuration

### Recommended Settings

```ini
# mypy.ini
[mypy]
python_version = 3.8
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_generics = True
disallow_any_unimported = True

# Per-module overrides for gradual typing
[mypy-src.legacy.*]
ignore_errors = True

[mypy-third_party_module]
ignore_missing_imports = True
```

### Running Type Checks

```bash
# Check entire codebase
mypy src/

# Check specific modules
mypy src/types/ src/providers/ src/evaluation/

# Check with strict mode
mypy --strict src/core_module.py
```

## Testing Types

### TypedDict Tests

```python
def test_evaluation_result_structure():
    """Test EvaluationResult TypedDict structure."""
    result: EvaluationResult = {
        "score": 0.85,
        "method": "semantic_similarity",
        "timestamp": "2024-01-01T00:00:00Z"
    }

    # Verify required fields
    assert "score" in result
    assert "method" in result
    assert "timestamp" in result

    # Verify types
    assert isinstance(result["score"], float)
    assert isinstance(result["method"], str)
```

### Protocol Tests

```python
def test_provider_protocol():
    """Test Provider protocol implementation."""
    provider = MyCustomProvider()

    # Runtime protocol check
    assert isinstance(provider, Provider)

    # Test required methods
    assert hasattr(provider, "generate")
    assert hasattr(provider, "get_model_info")
    assert hasattr(provider, "validate_credentials")

    # Test method signatures
    result = provider.generate("test prompt")
    assert isinstance(result, str)

    info = provider.get_model_info()
    assert isinstance(info, dict)
    assert "model_name" in info
```

## Migration Guide

### Step 1: Import Types

```python
# Add type imports
from src.types import (
    EvaluationResult,
    ProviderInfo,
    ModelParameters,
    APIResponse
)
```

### Step 2: Replace Return Types

```python
# Before
def get_model_info() -> Dict[str, Any]:
    return {"model_name": "gpt-4", "provider": "openai"}

# After
def get_model_info() -> ProviderInfo:
    return {"model_name": "gpt-4", "provider": "openai", ...}
```

### Step 3: Replace Parameter Types

```python
# Before
def evaluate(prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
    pass

# After
def evaluate(prompt: str, config: ValidationConfig) -> EvaluationResult:
    pass
```

### Step 4: Update Function Bodies

```python
# Before
result = {
    "score": 0.8,
    "method": "fuzzy"
}

# After
result: EvaluationResult = {
    "score": 0.8,
    "method": "fuzzy_match",
    "timestamp": datetime.now().isoformat()
}
```

## Best Practices

### 1. Use Specific Types

```python
# Avoid
data: Dict[str, Any] = get_data()

# Prefer
data: EvaluationResult = get_evaluation_data()
```

### 2. Leverage NotRequired

```python
class FlexibleConfig(TypedDict):
    required_field: str
    optional_field: NotRequired[int]
    debug_info: NotRequired[Dict[str, str]]
```

### 3. Use Protocols for Interfaces

```python
# Define clear interfaces
class Processable(Protocol):
    def process(self, data: Any) -> str: ...
    def validate(self) -> bool: ...

# Implementation flexibility
def handle_item(item: Processable) -> str:
    if item.validate():
        return item.process("data")
    return "invalid"
```

### 4. Combine with Type Guards

```python
def is_evaluation_result(obj: Any) -> bool:
    """Type guard for EvaluationResult."""
    return (
        isinstance(obj, dict) and
        "score" in obj and
        "method" in obj and
        "timestamp" in obj
    )

# Usage with type narrowing
if is_evaluation_result(data):
    # Type checker knows data is EvaluationResult
    score = data["score"]  # No type errors
```

## Common Patterns

### Builder Pattern with Types

```python
class EvaluationResultBuilder:
    def __init__(self):
        self._result: Dict[str, Any] = {}

    def with_score(self, score: float) -> 'EvaluationResultBuilder':
        self._result["score"] = score
        return self

    def with_method(self, method: str) -> 'EvaluationResultBuilder':
        self._result["method"] = method
        return self

    def build(self) -> EvaluationResult:
        # Add required fields if missing
        if "timestamp" not in self._result:
            self._result["timestamp"] = datetime.now().isoformat()

        return cast(EvaluationResult, self._result)
```

### Factory Pattern with Protocols

```python
def create_provider(provider_type: str) -> Provider:
    """Factory for creating providers."""
    if provider_type == "openai":
        return OpenAIProvider()
    elif provider_type == "local":
        return LocalProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_type}")

# Type checker ensures return value implements Provider
provider = create_provider("openai")
result = provider.generate("Hello")  # Type safe
```

This type system provides a solid foundation for type-safe development in LLM Lab, improving code quality, IDE support, and developer experience while maintaining runtime flexibility where needed.
