"""Tests for the prompt template engine."""

from datetime import datetime

import pytest

from use_cases.custom_prompts import PromptTemplate, TemplateError, ValidationError


class TestPromptTemplate:
    """Test cases for PromptTemplate class."""

    def test_simple_template(self):
        """Test basic variable substitution."""
        template = PromptTemplate("Hello {name}!")
        result = template.render({"name": "World"})
        assert result == "Hello World!"

    def test_multiple_variables(self):
        """Test template with multiple variables."""
        template = PromptTemplate("My name is {name} and I am {age} years old.")
        result = template.render({"name": "Alice", "age": 25})
        assert result == "My name is Alice and I am 25 years old."

    def test_builtin_variables(self):
        """Test built-in variables like timestamp."""
        template = PromptTemplate("Current time: {timestamp}")
        result = template.render()
        # Just check it contains a date string
        assert "-" in result and ":" in result

    def test_missing_variable_strict(self):
        """Test strict mode with missing variables."""
        template = PromptTemplate("Hello {name}!")
        with pytest.raises(TemplateError):
            template.render({}, strict=True)

    def test_missing_variable_non_strict(self):
        """Test non-strict mode leaves placeholders."""
        template = PromptTemplate("Hello {name}!")
        result = template.render({}, strict=False)
        assert result == "Hello {name}!"

    def test_conditional_true(self):
        """Test conditional section when true."""
        template = PromptTemplate("Hello{?premium} Premium User{/premium}!")
        result = template.render({"premium": True})
        assert result == "Hello Premium User!"

    def test_conditional_false(self):
        """Test conditional section when false."""
        template = PromptTemplate("Hello{?premium} Premium User{/premium}!")
        result = template.render({"premium": False})
        assert result == "Hello!"

    def test_get_required_variables(self):
        """Test extraction of required variables."""
        template = PromptTemplate("Hello {name}, time is {timestamp}")
        required = template.get_required_variables()
        assert required == {"name"}  # timestamp is built-in

    def test_validate_context(self):
        """Test context validation."""
        template = PromptTemplate("Hello {name} from {city}")
        missing = template.validate_context({"name": "Alice"})
        assert missing == ["city"]

    def test_json_value(self):
        """Test rendering with dict/list values."""
        template = PromptTemplate("Data: {data}")
        result = template.render({"data": {"key": "value", "num": 42}})
        assert '"key": "value"' in result
        assert '"num": 42' in result

    def test_unmatched_braces(self):
        """Test validation catches unmatched braces."""
        with pytest.raises(ValidationError):
            PromptTemplate("Hello {name")

    def test_from_string(self):
        """Test creating template from string."""
        template = PromptTemplate.from_string("Hello {name}!", name="greeting")
        assert template.name == "greeting"
        result = template.render({"name": "Test"})
        assert result == "Hello Test!"


if __name__ == "__main__":
    # Run a simple test
    template = PromptTemplate("Model: {model_name}\nPrompt: {prompt}\nTimestamp: {timestamp}")
    result = template.render({"model_name": "gpt-4", "prompt": "What is 2+2?"})
    print("Template test result:")
    print(result)
    print("\nRequired variables:", template.get_required_variables())
    print("All variables:", template.get_all_variables())
