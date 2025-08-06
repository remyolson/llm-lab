# Use Case 4: Run Tests Across LLMs

*Execute comprehensive test suites for validation, regression testing, and continuous integration across multiple LLM providers.*

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:
- **Create comprehensive test suites** for LLM outputs with pytest integration
- **Define expected behaviors** with appropriate tolerance levels for variations
- **Implement regression testing** to catch performance degradation
- **Set up automated CI/CD pipelines** for continuous LLM validation
- **Use fuzzy matching** and semantic assertions for robust testing
- **Generate detailed test reports** with performance metrics
- **Monitor model consistency** across different versions and providers
- **Build reliable quality gates** for production deployments

## 📋 Before You Begin

- Complete all [Prerequisites](./PREREQUISITES.md)
- Have API keys for at least two LLM providers
- Install pytest and testing dependencies: `pip install pytest pytest-html pytest-xdist difflib`
- Time required: ~30 minutes setup + testing time
- Estimated cost: $2-$10 per full test suite run

### 💰 Cost Breakdown

Cross-LLM testing costs depend on test suite size and model selection:

**💡 Pro Tip:** Use `--limit 5` during development to reduce costs by 90% (~$0.20-$1.00 per test run)

- **Small Test Suite** (10-20 test cases):
  - `gpt-4o-mini` + `claude-3-haiku` + `gemini-flash`: ~$0.50-$2.00
  - Premium models (`gpt-4` + `claude-3-opus`): ~$3.00-$8.00

- **Medium Test Suite** (50-100 test cases):
  - Cost-effective models: ~$2.00-$6.00
  - Premium models: ~$10.00-$25.00

- **Large Test Suite** (200+ test cases):
  - Cost-effective models: ~$8.00-$20.00
  - Premium models: ~$40.00-$100.00

*Note: Costs are estimates based on December 2024 pricing. Use `--parallel` and caching to optimize performance.*

## 🧪 Testing Architecture Overview

### Testing Approach Hierarchy

| Level | Purpose | Scope | Automation | Example |
|-------|---------|-------|------------|---------|
| **Unit Tests** | Individual prompt validation | Single model/prompt | High | Response format validation |
| **Integration Tests** | Cross-model consistency | Multiple models | High | Semantic similarity checks |
| **Regression Tests** | Performance monitoring | Model versions | High | Accuracy drift detection |
| **Acceptance Tests** | Business requirement validation | End-to-end workflows | Medium | Customer satisfaction criteria |
| **Performance Tests** | Speed and cost analysis | Resource usage | Medium | Response time benchmarks |

### Test Suite Structure

```
tests/
├── llm_tests/
│   ├── conftest.py              # Pytest fixtures and configuration
│   ├── test_prompt_responses.py # Basic response validation
│   ├── test_cross_model.py      # Cross-model consistency tests
│   ├── test_regression.py       # Regression detection tests
│   ├── test_performance.py      # Performance benchmarks
│   └── fixtures/                # Test data and expected outputs
├── integration/
│   ├── test_end_to_end.py      # Complete workflow tests
│   └── test_business_logic.py   # Domain-specific validation
└── reports/                     # Generated test reports
```

## 🚀 Step-by-Step Guide

### Step 1: Set Up Testing Infrastructure

First, create the testing framework:

```bash
# Install testing dependencies
pip install pytest pytest-html pytest-xdist pytest-asyncio pytest-mock

# Create test directory structure
mkdir -p tests/llm_tests/fixtures
mkdir -p tests/integration
mkdir -p tests/reports

# Create pytest configuration
cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --tb=short
    --html=tests/reports/report.html
    --self-contained-html
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    regression: marks tests as regression tests
    expensive: marks tests that use significant API credits
EOF
```

### Step 2: Create Test Fixtures and Configuration

Create `tests/llm_tests/conftest.py`:

```python
import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from unittest.mock import Mock

# Import your LLM testing utilities
import sys
sys.path.append('src')
from use_cases.custom_prompts.prompt_runner import PromptRunner
from use_cases.custom_prompts.evaluation_metrics import MetricSuite

@dataclass
class TestCase:
    """Structure for LLM test cases."""
    name: str
    prompt: str
    expected_patterns: List[str]
    forbidden_patterns: List[str]
    min_length: int
    max_length: int
    models_to_test: List[str]
    tolerance_level: str = "medium"  # strict, medium, loose
    timeout: int = 60

@dataclass
class ExpectedOutput:
    """Expected output specification."""
    content_type: str  # text, code, json, etc.
    required_elements: List[str]
    format_requirements: Dict[str, Any]
    quality_thresholds: Dict[str, float]

@pytest.fixture(scope="session")
def test_models():
    """Default models for testing."""
    return ["gpt-4o-mini", "claude-3-haiku", "gemini-flash"]

@pytest.fixture(scope="session")
def premium_models():
    """Premium models for comprehensive testing."""
    return ["gpt-4", "claude-3-sonnet", "gemini-pro"]

@pytest.fixture
def prompt_runner():
    """Configured prompt runner for testing."""
    return PromptRunner(
        max_retries=2,
        retry_delay=1.0,
        timeout=60
    )

@pytest.fixture
def metric_suite():
    """Evaluation metrics for test validation."""
    suite = MetricSuite()
    # Add any custom metrics needed for testing
    return suite

@pytest.fixture
def tolerance_config():
    """Tolerance levels for different test scenarios."""
    return {
        "strict": {
            "semantic_similarity_threshold": 0.90,
            "length_variance": 0.10,
            "format_compliance": 1.0
        },
        "medium": {
            "semantic_similarity_threshold": 0.75,
            "length_variance": 0.25,
            "format_compliance": 0.90
        },
        "loose": {
            "semantic_similarity_threshold": 0.60,
            "length_variance": 0.40,
            "format_compliance": 0.80
        }
    }

@pytest.fixture
def test_cases():
    """Load test cases from fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    test_cases = []

    # Load from JSON files
    for test_file in fixtures_dir.glob("*.json"):
        with open(test_file) as f:
            data = json.load(f)
            for case_data in data.get("test_cases", []):
                test_cases.append(TestCase(**case_data))

    return test_cases

@pytest.fixture
def mock_responses():
    """Mock responses for testing without API calls."""
    return {
        "gpt-4": {
            "success": True,
            "response": "This is a mock GPT-4 response for testing purposes.",
            "duration": 1.5
        },
        "claude-3-haiku": {
            "success": True,
            "response": "Mock Claude response with similar content and structure.",
            "duration": 1.2
        }
    }

# Async support
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Test data fixtures
@pytest.fixture
def customer_service_test_data():
    """Customer service test scenarios."""
    return [
        {
            "prompt": "Respond to a customer complaint about delayed delivery",
            "expected_tone": "empathetic",
            "required_elements": ["acknowledgment", "apology", "solution"],
            "forbidden_patterns": ["not our fault", "deal with it"]
        },
        {
            "prompt": "Handle a billing inquiry from a premium customer",
            "expected_tone": "professional",
            "required_elements": ["account verification", "detailed explanation"],
            "forbidden_patterns": ["I don't know", "that's not possible"]
        }
    ]

@pytest.fixture
def code_generation_test_data():
    """Code generation test scenarios."""
    return [
        {
            "prompt": "Write a Python function to calculate fibonacci numbers",
            "expected_patterns": ["def ", "fibonacci", "return"],
            "forbidden_patterns": ["import sys", "global "],
            "format_requirements": {"language": "python", "has_docstring": True}
        },
        {
            "prompt": "Create a JavaScript function for input validation",
            "expected_patterns": ["function", "validate", "return"],
            "forbidden_patterns": ["eval(", "innerHTML"],
            "format_requirements": {"language": "javascript", "has_error_handling": True}
        }
    ]
```

### Step 3: Implement Basic Response Validation Tests

Create `tests/llm_tests/test_prompt_responses.py`:

```python
import pytest
import re
from typing import List, Dict, Any
from difflib import SequenceMatcher

class TestPromptResponses:
    """Basic validation tests for LLM responses."""

    def test_response_format_validation(self, prompt_runner, test_models):
        """Test that responses follow expected format requirements."""

        prompt = "Explain the concept of machine learning in 2-3 paragraphs."

        for model in test_models:
            result = prompt_runner.run_single(prompt, model)

            assert result.success, f"{model} failed to generate response"
            assert result.response, f"{model} returned empty response"

            # Basic format checks
            assert len(result.response.strip()) > 50, f"{model} response too short"
            assert len(result.response.split('.')) >= 4, f"{model} response lacks paragraph structure"

    def test_content_appropriateness(self, prompt_runner, test_models):
        """Test that responses are appropriate and safe."""

        test_prompts = [
            "Write a professional email template",
            "Explain quantum computing to a beginner",
            "Create a recipe for chocolate chip cookies"
        ]

        forbidden_patterns = [
            r'\b(fuck|shit|damn)\b',  # Profanity
            r'<script.*?>',           # Potential XSS
            r'DROP TABLE',            # SQL injection patterns
        ]

        for prompt in test_prompts:
            for model in test_models:
                result = prompt_runner.run_single(prompt, model)

                assert result.success, f"{model} failed for prompt: {prompt}"

                # Check for inappropriate content
                for pattern in forbidden_patterns:
                    assert not re.search(pattern, result.response, re.IGNORECASE), \
                        f"{model} generated inappropriate content: {pattern}"

    @pytest.mark.parametrize("complexity_level", ["beginner", "intermediate", "advanced"])
    def test_response_complexity_adaptation(self, prompt_runner, test_models, complexity_level):
        """Test that models adapt response complexity appropriately."""

        prompt = f"Explain neural networks for a {complexity_level} audience"
        complexity_indicators = {
            "beginner": ["simple", "basic", "easy", "like"],
            "intermediate": ["however", "moreover", "specifically", "algorithm"],
            "advanced": ["optimization", "gradient", "backpropagation", "tensor"]
        }

        for model in test_models:
            result = prompt_runner.run_single(prompt, model)

            assert result.success, f"{model} failed for {complexity_level} complexity"

            # Check for appropriate complexity indicators
            expected_indicators = complexity_indicators[complexity_level]
            found_indicators = sum(1 for indicator in expected_indicators
                                 if indicator.lower() in result.response.lower())

            assert found_indicators >= 1, \
                f"{model} didn't adapt to {complexity_level} complexity level"

    def test_instruction_following(self, prompt_runner, test_models):
        """Test that models follow specific instructions."""

        test_cases = [
            {
                "prompt": "List exactly 5 benefits of renewable energy. Use bullet points.",
                "validation": lambda response: (
                    response.count('•') == 5 or response.count('*') == 5 or response.count('-') == 5,
                    "Should contain exactly 5 bullet points"
                )
            },
            {
                "prompt": "Write a haiku about programming. Include the word 'debug'.",
                "validation": lambda response: (
                    'debug' in response.lower(),
                    "Should contain the word 'debug'"
                )
            },
            {
                "prompt": "Respond with only 'YES' or 'NO': Is Python a programming language?",
                "validation": lambda response: (
                    response.strip().upper() in ['YES', 'NO'],
                    "Should respond with only YES or NO"
                )
            }
        ]

        for case in test_cases:
            for model in test_models:
                result = prompt_runner.run_single(case["prompt"], model)

                assert result.success, f"{model} failed for instruction: {case['prompt']}"

                is_valid, error_msg = case["validation"](result.response)
                assert is_valid, f"{model} didn't follow instructions: {error_msg}"

class TestSpecializedDomains:
    """Tests for domain-specific capabilities."""

    def test_code_generation_validity(self, prompt_runner, test_models):
        """Test that generated code is syntactically valid."""

        code_prompts = [
            {
                "prompt": "Write a Python function to sort a list of integers",
                "language": "python",
                "required_patterns": [r"def\s+\w+", r"return\s+"],
            },
            {
                "prompt": "Create a JavaScript function to validate email addresses",
                "language": "javascript",
                "required_patterns": [r"function\s+\w+", r"return\s+"],
            }
        ]

        for case in code_prompts:
            for model in test_models:
                result = prompt_runner.run_single(case["prompt"], model)

                assert result.success, f"{model} failed for code generation"

                # Extract code blocks
                code_blocks = re.findall(r'```\w*\n(.*?)\n```', result.response, re.DOTALL)

                assert len(code_blocks) > 0, f"{model} didn't provide code blocks"

                # Check for required patterns
                code_content = '\n'.join(code_blocks)
                for pattern in case["required_patterns"]:
                    assert re.search(pattern, code_content), \
                        f"{model} code missing required pattern: {pattern}"

    def test_creative_writing_quality(self, prompt_runner, test_models):
        """Test creative writing capabilities."""

        creative_prompts = [
            "Write a short story about a robot learning to paint",
            "Create a poem about the changing seasons",
            "Write dialogue between two characters arguing about time travel"
        ]

        quality_indicators = [
            r'\b\w+ly\b',  # Adverbs for descriptive language
            r'[.!?]{1}',   # Proper punctuation
            r'\".*?\"',    # Dialogue in quotes
        ]

        for prompt in creative_prompts:
            for model in test_models:
                result = prompt_runner.run_single(prompt, model)

                assert result.success, f"{model} failed for creative prompt"
                assert len(result.response.split()) >= 50, f"{model} creative response too short"

                # Check for creative writing indicators
                quality_score = sum(1 for pattern in quality_indicators
                                  if re.search(pattern, result.response))

                assert quality_score >= 1, f"{model} lacks creative writing qualities"
```

### Step 4: Cross-Model Consistency Tests

Create `tests/llm_tests/test_cross_model.py`:

```python
import pytest
from typing import List, Dict, Any
from difflib import SequenceMatcher
import statistics

class TestCrossModelConsistency:
    """Tests for consistency across different LLM models."""

    def test_semantic_consistency(self, prompt_runner, test_models, tolerance_config):
        """Test that different models provide semantically similar responses."""

        test_prompts = [
            "What are the main benefits of cloud computing?",
            "Explain the difference between artificial intelligence and machine learning",
            "List the steps to bake a chocolate cake"
        ]

        for prompt in test_prompts:
            responses = {}

            # Collect responses from all models
            for model in test_models:
                result = prompt_runner.run_single(prompt, model)
                assert result.success, f"{model} failed to respond to: {prompt}"
                responses[model] = result.response

            # Compare semantic similarity between all pairs
            similarities = []
            model_pairs = [(m1, m2) for i, m1 in enumerate(test_models)
                          for m2 in test_models[i+1:]]

            for model1, model2 in model_pairs:
                similarity = self._calculate_semantic_similarity(
                    responses[model1], responses[model2]
                )
                similarities.append(similarity)

                # Assert minimum similarity threshold
                min_threshold = tolerance_config["medium"]["semantic_similarity_threshold"]
                assert similarity >= min_threshold, \
                    f"Low semantic similarity between {model1} and {model2}: {similarity:.3f}"

            # Overall consistency check
            avg_similarity = statistics.mean(similarities)
            assert avg_similarity >= 0.65, f"Overall consistency too low: {avg_similarity:.3f}"

    def test_response_length_consistency(self, prompt_runner, test_models, tolerance_config):
        """Test that response lengths are reasonably consistent."""

        length_sensitive_prompts = [
            "Write a brief summary of photosynthesis",
            "Provide a detailed explanation of blockchain technology",
            "Give a short definition of recursion"
        ]

        for prompt in length_sensitive_prompts:
            lengths = []

            for model in test_models:
                result = prompt_runner.run_single(prompt, model)
                assert result.success, f"{model} failed for length test"
                lengths.append(len(result.response.split()))

            # Check length consistency
            if len(lengths) > 1:
                mean_length = statistics.mean(lengths)
                std_deviation = statistics.stdev(lengths)
                coefficient_of_variation = std_deviation / mean_length

                max_variance = tolerance_config["medium"]["length_variance"]
                assert coefficient_of_variation <= max_variance, \
                    f"Response length too variable: {coefficient_of_variation:.3f} > {max_variance}"

    def test_format_consistency(self, prompt_runner, test_models):
        """Test that models produce consistent output formats."""

        format_tests = [
            {
                "prompt": "List the top 5 programming languages with brief descriptions",
                "expected_format": "list",
                "validation": lambda r: len(re.findall(r'^[-*•]\s', r, re.MULTILINE)) >= 5
            },
            {
                "prompt": "Create a JSON object with name, age, and city fields",
                "expected_format": "json",
                "validation": lambda r: '{' in r and '}' in r and '"name"' in r
            },
            {
                "prompt": "Write Python code to reverse a string",
                "expected_format": "code",
                "validation": lambda r: '```' in r or 'def ' in r
            }
        ]

        for test_case in format_tests:
            format_compliance = []

            for model in test_models:
                result = prompt_runner.run_single(test_case["prompt"], model)
                assert result.success, f"{model} failed format test"

                is_compliant = test_case["validation"](result.response)
                format_compliance.append(is_compliant)

            # Require majority compliance
            compliance_rate = sum(format_compliance) / len(format_compliance)
            assert compliance_rate >= 0.8, \
                f"Format compliance too low: {compliance_rate:.2%} for {test_case['expected_format']}"

    def test_factual_consistency(self, prompt_runner, test_models):
        """Test that models provide consistent factual information."""

        factual_prompts = [
            "What is the capital of France?",
            "When did World War II end?",
            "What is the chemical formula for water?",
            "Who wrote Romeo and Juliet?"
        ]

        expected_answers = [
            ["Paris"],
            ["1945", "September 2, 1945"],
            ["H2O", "H₂O"],
            ["Shakespeare", "William Shakespeare"]
        ]

        for prompt, expected in zip(factual_prompts, expected_answers):
            correct_answers = 0

            for model in test_models:
                result = prompt_runner.run_single(prompt, model)
                assert result.success, f"{model} failed factual question"

                # Check if any expected answer is in the response
                response_lower = result.response.lower()
                if any(ans.lower() in response_lower for ans in expected):
                    correct_answers += 1

            # Require high factual accuracy
            accuracy = correct_answers / len(test_models)
            assert accuracy >= 0.8, f"Factual accuracy too low: {accuracy:.2%} for '{prompt}'"

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""

        # Simple word-based similarity (in practice, you might use sentence transformers)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        jaccard = intersection / union if union > 0 else 0

        # Sequence similarity
        sequence_sim = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

        # Combined similarity score
        return (jaccard * 0.4) + (sequence_sim * 0.6)

class TestModelSpecificBehaviors:
    """Tests for known model-specific behaviors and capabilities."""

    @pytest.mark.parametrize("model,capability", [
        ("gpt-4", "detailed_analysis"),
        ("claude-3-sonnet", "creative_writing"),
        ("gemini-pro", "factual_accuracy")
    ])
    def test_model_strengths(self, prompt_runner, model, capability):
        """Test models on their known strengths."""

        capability_tests = {
            "detailed_analysis": {
                "prompt": "Analyze the economic implications of renewable energy adoption",
                "validation": lambda r: len(r.split()) > 200 and ("economic" in r.lower())
            },
            "creative_writing": {
                "prompt": "Write a creative story about a time-traveling librarian",
                "validation": lambda r: len(r.split()) > 150 and ('"' in r or "'" in r)
            },
            "factual_accuracy": {
                "prompt": "What are the exact dates and locations of the first moon landing?",
                "validation": lambda r: ("1969" in r and "Armstrong" in r)
            }
        }

        if capability in capability_tests:
            test_case = capability_tests[capability]
            result = prompt_runner.run_single(test_case["prompt"], model)

            assert result.success, f"{model} failed {capability} test"
            assert test_case["validation"](result.response), \
                f"{model} didn't demonstrate {capability}"
```

### Step 5: Regression Testing

Create `tests/llm_tests/test_regression.py`:

```python
import pytest
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics

class TestRegressionDetection:
    """Tests to detect performance regression in models."""

    @pytest.fixture
    def baseline_results(self):
        """Load baseline performance results."""
        baseline_file = Path("tests/fixtures/baseline_results.json")
        if baseline_file.exists():
            with open(baseline_file) as f:
                return json.load(f)
        return {}

    def test_response_quality_regression(self, prompt_runner, test_models, metric_suite, baseline_results):
        """Test for regression in response quality metrics."""

        quality_prompts = [
            "Explain the concept of machine learning to a non-technical audience",
            "Write a professional email declining a meeting request",
            "Solve this math problem: If a train travels 60 mph for 2.5 hours, how far does it go?"
        ]

        current_results = {}

        for prompt in quality_prompts:
            for model in test_models:
                result = prompt_runner.run_single(prompt, model)
                assert result.success, f"{model} failed regression test"

                # Evaluate with metrics
                metrics = metric_suite.evaluate(result.response)

                # Store results
                key = f"{model}_{hash(prompt) % 10000}"
                current_results[key] = metrics

                # Compare with baseline if available
                if key in baseline_results:
                    self._check_regression(key, baseline_results[key], metrics)

        # Save current results as new baseline
        self._save_current_results(current_results)

    def test_performance_timing_regression(self, prompt_runner, test_models):
        """Test for regression in response timing."""

        timing_prompts = [
            "What is 2 + 2?",  # Simple prompt
            "Write a 100-word summary of climate change",  # Medium prompt
            "Create a detailed business plan for a coffee shop"  # Complex prompt
        ]

        timing_results = {}

        for prompt in timing_prompts:
            for model in test_models:
                # Run multiple times for statistical significance
                durations = []
                for _ in range(3):
                    result = prompt_runner.run_single(prompt, model)
                    assert result.success, f"{model} failed timing test"
                    durations.append(result.duration_seconds)

                avg_duration = statistics.mean(durations)
                timing_results[f"{model}_{hash(prompt) % 10000}"] = avg_duration

                # Check for reasonable response times
                max_acceptable_time = 30  # seconds
                assert avg_duration <= max_acceptable_time, \
                    f"{model} response time too slow: {avg_duration:.2f}s"

    def test_consistency_regression(self, prompt_runner, test_models):
        """Test for regression in response consistency."""

        consistency_prompt = "Explain the water cycle in simple terms"

        for model in test_models:
            responses = []

            # Generate multiple responses to the same prompt
            for _ in range(5):
                result = prompt_runner.run_single(consistency_prompt, model)
                assert result.success, f"{model} failed consistency test"
                responses.append(result.response)

            # Calculate consistency metrics
            similarities = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    similarity = self._calculate_text_similarity(responses[i], responses[j])
                    similarities.append(similarity)

            avg_consistency = statistics.mean(similarities)

            # Check for minimum consistency threshold
            min_consistency = 0.7
            assert avg_consistency >= min_consistency, \
                f"{model} consistency regression: {avg_consistency:.3f} < {min_consistency}"

    def test_capability_regression(self, prompt_runner, test_models):
        """Test for regression in specific capabilities."""

        capability_tests = [
            {
                "name": "math_solving",
                "prompt": "Calculate 15% of 240",
                "validation": lambda r: "36" in r,
                "critical": True
            },
            {
                "name": "code_generation",
                "prompt": "Write a Python function to check if a number is even",
                "validation": lambda r: "def " in r and "%" in r,
                "critical": True
            },
            {
                "name": "language_translation",
                "prompt": "Translate 'Hello, how are you?' to Spanish",
                "validation": lambda r: "hola" in r.lower() or "cómo" in r.lower(),
                "critical": False
            }
        ]

        failures = []

        for test in capability_tests:
            for model in test_models:
                result = prompt_runner.run_single(test["prompt"], model)

                if not result.success or not test["validation"](result.response):
                    failure_info = {
                        "model": model,
                        "capability": test["name"],
                        "critical": test["critical"]
                    }
                    failures.append(failure_info)

        # Check critical failures
        critical_failures = [f for f in failures if f["critical"]]
        assert len(critical_failures) == 0, \
            f"Critical capability regressions: {critical_failures}"

        # Report non-critical failures
        if failures:
            print(f"Non-critical capability issues: {failures}")

    def _check_regression(self, key: str, baseline: Dict, current: Dict):
        """Check for regression in metrics."""

        regression_threshold = 0.05  # 5% degradation threshold

        for metric_name, baseline_value in baseline.items():
            if metric_name in current:
                current_value = current[metric_name]

                # Calculate percentage change
                if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                    if baseline_value > 0:
                        change = (current_value - baseline_value) / baseline_value

                        # Assert no significant regression
                        assert change >= -regression_threshold, \
                            f"Regression detected in {key}.{metric_name}: {change:.2%} decline"

    def _save_current_results(self, results: Dict):
        """Save current results for future baseline comparison."""

        baseline_file = Path("tests/fixtures/baseline_results.json")
        baseline_file.parent.mkdir(exist_ok=True)

        # Load existing baselines
        existing_baselines = {}
        if baseline_file.exists():
            with open(baseline_file) as f:
                existing_baselines = json.load(f)

        # Update with current results
        existing_baselines.update(results)
        existing_baselines["last_updated"] = datetime.now().isoformat()

        # Save updated baselines
        with open(baseline_file, 'w') as f:
            json.dump(existing_baselines, f, indent=2)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""

    @pytest.mark.slow
    def test_throughput_performance(self, prompt_runner, test_models):
        """Test throughput performance for batch operations."""

        batch_prompts = [f"Count from 1 to {i}" for i in range(1, 21)]

        for model in test_models:
            start_time = pytest.importorskip("time").time()

            successful_responses = 0
            for prompt in batch_prompts:
                result = prompt_runner.run_single(prompt, model)
                if result.success:
                    successful_responses += 1

            total_time = pytest.importorskip("time").time() - start_time
            throughput = successful_responses / total_time  # responses per second

            # Assert minimum throughput
            min_throughput = 0.5  # responses per second
            assert throughput >= min_throughput, \
                f"{model} throughput too low: {throughput:.2f} responses/sec"

            # Assert high success rate
            success_rate = successful_responses / len(batch_prompts)
            assert success_rate >= 0.90, \
                f"{model} success rate too low: {success_rate:.2%}"

    @pytest.mark.expensive
    def test_cost_performance_regression(self, prompt_runner, test_models):
        """Test for cost efficiency regression."""

        # This test would integrate with cost tracking
        # Implementation depends on your cost tracking system
        cost_prompts = [
            "Write a short poem",
            "Explain photosynthesis",
            "Create a simple recipe"
        ]

        for model in test_models:
            total_cost = 0
            total_value = 0

            for prompt in cost_prompts:
                result = prompt_runner.run_single(prompt, model)
                assert result.success, f"{model} failed cost test"

                # Estimate cost (would integrate with actual cost tracking)
                estimated_cost = self._estimate_cost(model, result.response)
                value_score = self._assess_response_value(result.response)

                total_cost += estimated_cost
                total_value += value_score

            # Calculate cost efficiency
            efficiency = total_value / total_cost if total_cost > 0 else 0

            # Assert minimum cost efficiency
            min_efficiency = 1.0  # value points per cost unit
            assert efficiency >= min_efficiency, \
                f"{model} cost efficiency regression: {efficiency:.2f}"

    def _estimate_cost(self, model: str, response: str) -> float:
        """Estimate cost for a response (placeholder implementation)."""
        # This would integrate with actual pricing APIs
        token_count = len(response.split()) * 1.3  # Rough token estimation

        cost_per_token = {
            "gpt-4": 0.00003,
            "gpt-4o-mini": 0.000001,
            "claude-3-sonnet": 0.000015,
            "claude-3-haiku": 0.0000025,
            "gemini-pro": 0.0000005
        }

        return token_count * cost_per_token.get(model, 0.00001)

    def _assess_response_value(self, response: str) -> float:
        """Assess the value/quality of a response (placeholder implementation)."""
        # This would use more sophisticated quality metrics
        length_score = min(len(response.split()) / 50, 2.0)
        completeness_score = 1.0 if len(response.strip()) > 20 else 0.5

        return length_score + completeness_score
```

### Step 6: CI/CD Integration

Create `.github/workflows/llm-testing.yml`:

```yaml
name: LLM Cross-Model Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'
  workflow_dispatch:
    inputs:
      test_suite:
        description: 'Test suite to run'
        required: true
        default: 'full'
        type: choice
        options:
          - 'smoke'
          - 'regression'
          - 'full'
      models:
        description: 'Models to test (comma-separated)'
        required: false
        default: 'gpt-4o-mini,claude-3-haiku,gemini-flash'

env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}

jobs:
  llm-smoke-tests:
    runs-on: ubuntu-latest
    if: github.event.inputs.test_suite == 'smoke' || github.event_name == 'pull_request'

    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        model: ['gpt-4o-mini', 'claude-3-haiku']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-html pytest-xdist pytest-asyncio

    - name: Run smoke tests
      run: |
        pytest tests/llm_tests/test_prompt_responses.py::TestPromptResponses::test_response_format_validation \
               tests/llm_tests/test_prompt_responses.py::TestPromptResponses::test_content_appropriateness \
               -v --tb=short --html=smoke-test-report.html --self-contained-html
      env:
        PYTEST_CURRENT_TEST_MODEL: ${{ matrix.model }}

    - name: Upload smoke test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: smoke-test-results-${{ matrix.model }}-py${{ matrix.python-version }}
        path: smoke-test-report.html

  llm-regression-tests:
    runs-on: ubuntu-latest
    if: github.event.inputs.test_suite == 'regression' || github.event_name == 'schedule'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-html pytest-xdist

    - name: Download baseline results
      uses: actions/download-artifact@v3
      with:
        name: baseline-results
        path: tests/fixtures/
      continue-on-error: true

    - name: Run regression tests
      run: |
        pytest tests/llm_tests/test_regression.py \
               -v --tb=short --html=regression-test-report.html --self-contained-html \
               --maxfail=5

    - name: Upload regression test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: regression-test-results
        path: regression-test-report.html

    - name: Save new baseline
      uses: actions/upload-artifact@v3
      with:
        name: baseline-results
        path: tests/fixtures/baseline_results.json

  llm-full-test-suite:
    runs-on: ubuntu-latest
    if: github.event.inputs.test_suite == 'full' || github.event_name == 'push'

    strategy:
      fail-fast: false
      matrix:
        test_category: ['basic', 'cross_model', 'regression', 'performance']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-html pytest-xdist pytest-asyncio pytest-mock

    - name: Determine test models
      id: models
      run: |
        if [ "${{ github.event.inputs.models }}" != "" ]; then
          echo "models=${{ github.event.inputs.models }}" >> $GITHUB_OUTPUT
        else
          echo "models=gpt-4o-mini,claude-3-haiku,gemini-flash" >> $GITHUB_OUTPUT
        fi

    - name: Run test category - ${{ matrix.test_category }}
      run: |
        case "${{ matrix.test_category }}" in
          "basic")
            pytest tests/llm_tests/test_prompt_responses.py -v --tb=short
            ;;
          "cross_model")
            pytest tests/llm_tests/test_cross_model.py -v --tb=short
            ;;
          "regression")
            pytest tests/llm_tests/test_regression.py -v --tb=short
            ;;
          "performance")
            pytest tests/llm_tests/test_regression.py::TestPerformanceBenchmarks -v --tb=short -m "not expensive"
            ;;
        esac
      env:
        TEST_MODELS: ${{ steps.models.outputs.models }}

    - name: Generate combined report
      if: matrix.test_category == 'basic'
      run: |
        pytest tests/llm_tests/ \
               --html=full-test-report.html --self-contained-html \
               --tb=short -v \
               -m "not slow and not expensive"

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.test_category }}
        path: "*-test-report.html"

  test-report-summary:
    runs-on: ubuntu-latest
    needs: [llm-smoke-tests, llm-regression-tests, llm-full-test-suite]
    if: always()

    steps:
    - name: Download all test results
      uses: actions/download-artifact@v3

    - name: Create test summary
      run: |
        echo "# LLM Testing Summary" > test-summary.md
        echo "" >> test-summary.md
        echo "## Test Results" >> test-summary.md

        # Count test artifacts
        smoke_tests=$(find . -name "smoke-test-results-*" -type d | wc -l)
        echo "- Smoke tests: $smoke_tests runs" >> test-summary.md

        if [ -d "regression-test-results" ]; then
          echo "- Regression tests: ✅ Completed" >> test-summary.md
        else
          echo "- Regression tests: ❌ Not run" >> test-summary.md
        fi

        full_tests=$(find . -name "test-results-*" -type d | wc -l)
        echo "- Full test suite: $full_tests categories" >> test-summary.md

        echo "" >> test-summary.md
        echo "## Available Reports" >> test-summary.md
        find . -name "*.html" -exec basename {} \; | sort | sed 's/^/- /' >> test-summary.md

    - name: Upload summary
      uses: actions/upload-artifact@v3
      with:
        name: test-summary
        path: test-summary.md

  performance-monitoring:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    needs: [llm-full-test-suite]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install matplotlib seaborn pandas

    - name: Generate performance trends
      run: |
        python scripts/generate_performance_trends.py \
               --output-dir performance-trends \
               --days 30

    - name: Upload performance trends
      uses: actions/upload-artifact@v3
      with:
        name: performance-trends
        path: performance-trends/

    - name: Check for performance alerts
      run: |
        python scripts/check_performance_alerts.py \
               --threshold-file performance-thresholds.json \
               --alert-webhook ${{ secrets.SLACK_WEBHOOK_URL }}
```

## 🎯 Running the Tests

### Basic Test Execution

```bash
# Run all tests
pytest tests/llm_tests/

# Run specific test category
pytest tests/llm_tests/test_prompt_responses.py -v

# Run with specific markers
pytest -m "not slow and not expensive" tests/llm_tests/

# Run tests for specific models
pytest tests/llm_tests/ --models="gpt-4o-mini,claude-3-haiku"

# Generate HTML report
pytest tests/llm_tests/ --html=test_report.html --self-contained-html
```

### Advanced Test Execution

```bash
# Parallel test execution
pytest tests/llm_tests/ -n auto

# Run only failed tests from last run
pytest tests/llm_tests/ --lf

# Run with coverage
pytest tests/llm_tests/ --cov=src --cov-report=html

# Run with live logging
pytest tests/llm_tests/ -s --log-cli-level=INFO
```

## 🔧 Custom Assertion Helpers

Create `tests/llm_tests/helpers/assertions.py`:

```python
# tests/llm_tests/helpers/assertions.py

import re
from typing import List, Optional
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords

class LLMAssertions:
    """Custom assertion helpers for LLM testing."""

    @staticmethod
    def assert_contains_concepts(response: str, concepts: List[str], min_matches: int = None):
        """Assert that response contains specified concepts."""
        if min_matches is None:
            min_matches = len(concepts)

        response_lower = response.lower()
        matches = sum(1 for concept in concepts if concept.lower() in response_lower)

        assert matches >= min_matches, \
            f"Response contains {matches}/{len(concepts)} required concepts: {concepts}"

    @staticmethod
    def assert_semantic_similarity(response1: str, response2: str, threshold: float = 0.7):
        """Assert semantic similarity between responses."""
        similarity = LLMAssertions._calculate_semantic_similarity(response1, response2)

        assert similarity >= threshold, \
            f"Semantic similarity too low: {similarity:.3f} < {threshold}"

    @staticmethod
    def assert_response_quality(response: str, min_length: int = 10,
                              max_length: int = None, required_patterns: List[str] = None):
        """Assert response meets quality criteria."""
        word_count = len(response.split())

        assert word_count >= min_length, f"Response too short: {word_count} < {min_length} words"

        if max_length:
            assert word_count <= max_length, f"Response too long: {word_count} > {max_length} words"

        if required_patterns:
            for pattern in required_patterns:
                assert re.search(pattern, response, re.IGNORECASE), \
                    f"Response missing required pattern: {pattern}"

    @staticmethod
    def assert_fuzzy_match(response: str, expected: str, threshold: float = 0.8):
        """Assert fuzzy string matching."""
        similarity = SequenceMatcher(None, response.lower(), expected.lower()).ratio()

        assert similarity >= threshold, \
            f"Fuzzy match failed: {similarity:.3f} < {threshold}"

    @staticmethod
    def assert_structured_response(response: str, structure_type: str):
        """Assert response follows expected structure."""
        structure_validators = {
            "list": lambda r: bool(re.search(r'^[-*•]\s', r, re.MULTILINE)),
            "numbered_list": lambda r: bool(re.search(r'^\d+\.\s', r, re.MULTILINE)),
            "json": lambda r: '{' in r and '}' in r,
            "code": lambda r: '```' in r or 'def ' in r or 'function ' in r,
            "email": lambda r: '@' in r and ('subject:' in r.lower() or 'dear' in r.lower())
        }

        validator = structure_validators.get(structure_type)
        assert validator, f"Unknown structure type: {structure_type}"

        assert validator(response), f"Response doesn't match {structure_type} structure"

    @staticmethod
    def _calculate_semantic_similarity(text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts."""
        # Tokenize and remove stop words
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()

        words1 = set(word.lower() for word in text1.split() if word.lower() not in stop_words)
        words2 = set(word.lower() for word in text2.split() if word.lower() not in stop_words)

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0
```

## 📊 Test Reporting and Analysis

### Example Test Report Generator

Create `scripts/generate_test_report.py`:

```python
#!/usr/bin/env python3
"""
Generate comprehensive test reports for LLM testing.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from jinja2 import Template

def generate_comprehensive_report(test_results_dir: str, output_dir: str):
    """Generate comprehensive test report with visualizations."""

    results_path = Path(test_results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Collect all test results
    all_results = {}
    for result_file in results_path.glob("*.json"):
        with open(result_file) as f:
            data = json.load(f)
            all_results[result_file.stem] = data

    # Generate performance visualizations
    generate_performance_plots(all_results, output_path)

    # Generate HTML report
    generate_html_report(all_results, output_path)

    print(f"Comprehensive test report generated in {output_path}")

def generate_performance_plots(results: dict, output_dir: Path):
    """Generate performance visualization plots."""

    # Response time comparison
    plt.figure(figsize=(12, 6))
    models = []
    response_times = []

    for test_name, test_data in results.items():
        if "performance" in test_name and "response_times" in test_data:
            for model, times in test_data["response_times"].items():
                models.extend([model] * len(times))
                response_times.extend(times)

    if models and response_times:
        df = pd.DataFrame({"Model": models, "Response_Time": response_times})
        sns.boxplot(data=df, x="Model", y="Response_Time")
        plt.title("Response Time Distribution by Model")
        plt.ylabel("Response Time (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "response_times.png", dpi=300)
        plt.close()

    # Success rate comparison
    plt.figure(figsize=(10, 6))
    success_rates = {}

    for test_name, test_data in results.items():
        if "success_rates" in test_data:
            for model, rate in test_data["success_rates"].items():
                if model not in success_rates:
                    success_rates[model] = []
                success_rates[model].append(rate)

    if success_rates:
        models = list(success_rates.keys())
        rates = [statistics.mean(success_rates[model]) for model in models]

        plt.bar(models, rates)
        plt.title("Average Success Rate by Model")
        plt.ylabel("Success Rate (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=45)

        for i, rate in enumerate(rates):
            plt.text(i, rate + 1, f"{rate:.1f}%", ha='center')

        plt.tight_layout()
        plt.savefig(output_dir / "success_rates.png", dpi=300)
        plt.close()

def generate_html_report(results: dict, output_dir: Path):
    """Generate HTML test report."""

    report_template = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Cross-Model Testing Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: #f4f4f4; padding: 20px; margin-bottom: 30px; }
        .summary { background: #e8f5e8; padding: 15px; margin: 20px 0; }
        .failure { background: #ffe8e8; padding: 15px; margin: 20px 0; }
        .chart { margin: 20px 0; text-align: center; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM Cross-Model Testing Report</h1>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        <p><strong>Total Test Suites:</strong> {{ total_suites }}</p>
        <p><strong>Models Tested:</strong> {{ models|join(', ') }}</p>
    </div>

    <div class="summary">
        <h2>Executive Summary</h2>
        <ul>
            <li><strong>Overall Success Rate:</strong> {{ overall_success_rate }}%</li>
            <li><strong>Total Tests Executed:</strong> {{ total_tests }}</li>
            <li><strong>Failed Tests:</strong> {{ failed_tests }}</li>
            <li><strong>Average Response Time:</strong> {{ avg_response_time }}s</li>
        </ul>
    </div>

    {% if failures %}
    <div class="failure">
        <h2>⚠️ Test Failures</h2>
        <ul>
        {% for failure in failures %}
            <li><strong>{{ failure.test_name }}</strong>: {{ failure.error }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}

    <h2>Performance Charts</h2>
    <div class="chart">
        <img src="response_times.png" alt="Response Time Distribution" style="max-width: 100%;">
    </div>
    <div class="chart">
        <img src="success_rates.png" alt="Success Rate Comparison" style="max-width: 100%;">
    </div>

    <h2>Detailed Results</h2>
    <table>
        <thead>
            <tr>
                <th>Test Suite</th>
                <th>Model</th>
                <th>Status</th>
                <th>Success Rate</th>
                <th>Avg Response Time</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>
        {% for result in detailed_results %}
            <tr>
                <td>{{ result.suite }}</td>
                <td>{{ result.model }}</td>
                <td class="{{ 'pass' if result.status == 'PASS' else 'fail' }}">{{ result.status }}</td>
                <td>{{ result.success_rate }}%</td>
                <td>{{ result.avg_time }}s</td>
                <td>{{ result.details }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>

    <h2>Recommendations</h2>
    <ul>
    {% for recommendation in recommendations %}
        <li>{{ recommendation }}</li>
    {% endfor %}
    </ul>

    <hr>
    <p><em>Report generated by LLM Lab Cross-Model Testing Framework</em></p>
</body>
</html>
"""

    # Process results for template
    processed_data = process_results_for_template(results)

    # Render template
    template = Template(report_template)
    html_content = template.render(**processed_data)

    # Save HTML report
    with open(output_dir / "test_report.html", 'w') as f:
        f.write(html_content)

def process_results_for_template(results: dict) -> dict:
    """Process raw results for template rendering."""

    # Extract key metrics
    total_suites = len(results)
    models = set()
    total_tests = 0
    failed_tests = 0
    response_times = []
    failures = []
    detailed_results = []

    for suite_name, suite_data in results.items():
        if isinstance(suite_data, dict):
            if "models" in suite_data:
                models.update(suite_data["models"])
            if "total_tests" in suite_data:
                total_tests += suite_data["total_tests"]
            if "failed_tests" in suite_data:
                failed_tests += suite_data["failed_tests"]
            if "failures" in suite_data:
                failures.extend(suite_data["failures"])
            if "response_times" in suite_data:
                response_times.extend(suite_data["response_times"])

    # Generate recommendations
    recommendations = generate_recommendations(results)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_suites": total_suites,
        "models": list(models),
        "overall_success_rate": round((total_tests - failed_tests) / total_tests * 100, 1) if total_tests > 0 else 0,
        "total_tests": total_tests,
        "failed_tests": failed_tests,
        "avg_response_time": round(statistics.mean(response_times), 2) if response_times else 0,
        "failures": failures,
        "detailed_results": detailed_results,
        "recommendations": recommendations
    }

def generate_recommendations(results: dict) -> list:
    """Generate recommendations based on test results."""

    recommendations = []

    # Analyze common failure patterns
    failure_count = sum(len(suite.get("failures", [])) for suite in results.values() if isinstance(suite, dict))

    if failure_count > 0:
        recommendations.append(f"Address {failure_count} test failures to improve system reliability")

    # Analyze performance patterns
    slow_models = []
    for suite_name, suite_data in results.items():
        if isinstance(suite_data, dict) and "slow_models" in suite_data:
            slow_models.extend(suite_data["slow_models"])

    if slow_models:
        unique_slow_models = list(set(slow_models))
        recommendations.append(f"Consider optimizing performance for: {', '.join(unique_slow_models)}")

    # Generic recommendations
    recommendations.extend([
        "Schedule regular regression testing to catch performance degradation early",
        "Monitor cost efficiency across different models for budget optimization",
        "Consider implementing automated alerting for critical test failures"
    ])

    return recommendations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM test reports")
    parser.add_argument("--results-dir", required=True, help="Directory containing test results")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports")

    args = parser.parse_args()
    generate_comprehensive_report(args.results_dir, args.output_dir)
```

## 🎯 Best Practices Summary

### Test Design Principles

1. **Deterministic Where Possible**: Use fixed seeds and controlled inputs
2. **Tolerance-Based Assertions**: Allow for natural LLM variation
3. **Semantic Validation**: Focus on meaning over exact text matching
4. **Layered Testing**: Unit → Integration → System → Acceptance
5. **Cost-Aware Testing**: Balance thoroughness with API costs

### CI/CD Integration

1. **Tiered Testing**: Smoke tests for PRs, full suites for releases
2. **Model-Specific Pipelines**: Different strategies for different models
3. **Performance Monitoring**: Track metrics over time
4. **Failure Analysis**: Automated issue detection and reporting
5. **Cost Controls**: Budget limits and optimization strategies

### Maintenance Strategies

1. **Regular Baseline Updates**: Keep regression tests current
2. **Model Version Tracking**: Monitor for model updates
3. **Test Data Refresh**: Update test cases regularly
4. **Performance Benchmarking**: Continuous performance monitoring
5. **Documentation**: Keep test documentation current

---

*This comprehensive guide provides a complete framework for testing LLM applications across multiple providers with robust validation, performance monitoring, and CI/CD integration.*
