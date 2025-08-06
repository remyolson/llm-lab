"""Template engine for custom prompts with variable substitution and validation.

This module provides a flexible template engine that supports:
- Variable interpolation with {variable} syntax
- Template validation to detect undefined variables
- Safe string interpolation to prevent code injection
- Nested templates and conditional sections
- Loading templates from files or strings
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class TemplateError(Exception):
    """Base exception for template-related errors."""

    pass


class ValidationError(TemplateError):
    """Raised when template validation fails."""

    pass


class RenderError(TemplateError):
    """Raised when template rendering fails."""

    pass


class PromptTemplate:
    """A flexible prompt template with variable substitution and validation.

    This class provides methods for parsing templates, validating variable
    placeholders, and rendering with provided context. It supports common
    variables like {context}, {question}, {model_name}, {timestamp} and
    custom user-defined variables.

    Example:
        template = PromptTemplate("Hello {name}, today is {timestamp}")
        result = template.render({"name": "Alice"})
    """

    # Pattern to match variables in templates: {variable_name}
    VARIABLE_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

    # Pattern to match conditional sections: {?condition}content{/condition}
    CONDITIONAL_PATTERN = re.compile(r"\{\?([a-zA-Z_][a-zA-Z0-9_]*)\}(.*?)\{/\1\}", re.DOTALL)

    # Built-in variables that are always available
    BUILTIN_VARS = {
        "timestamp": lambda: datetime.now().isoformat(),
        "date": lambda: datetime.now().strftime("%Y-%m-%d"),
        "time": lambda: datetime.now().strftime("%H:%M:%S"),
    }

    def __init__(self, template: str | Path, name: Optional[str] = None):
        """Initialize a prompt template.

        Args:
            template: Template string or path to template file
            name: Optional name for the template (defaults to 'template' or filename)
        """
        if isinstance(template, (str, Path)) and Path(template).exists():
            self.template_path = Path(template)
            self.template_str = self.template_path.read_text(encoding="utf-8")
            self.name = name or self.template_path.stem
        else:
            self.template_path = None
            self.template_str = str(template)
            self.name = name or "template"

        self._validate_syntax()
        self._extract_variables()

    def _validate_syntax(self) -> None:
        """Validate template syntax for proper formatting."""
        # Check for unmatched braces
        open_braces = self.template_str.count("{")
        close_braces = self.template_str.count("}")
        if open_braces != close_braces:
            raise ValidationError(
                f"Unmatched braces in template '{self.name}': "
                f"{open_braces} opening, {close_braces} closing"
            )

        # Check for nested conditionals (not supported in this version)
        conditionals = list(self.CONDITIONAL_PATTERN.finditer(self.template_str))
        for i, match1 in enumerate(conditionals):
            for match2 in conditionals[i + 1 :]:
                if match1.start() < match2.start() < match1.end():
                    raise ValidationError(
                        f"Nested conditionals are not supported in template '{self.name}'"
                    )

    def _extract_variables(self) -> None:
        """Extract all variable names from the template."""
        self.variables: Set[str] = set()
        self.conditionals: Set[str] = set()

        # Extract regular variables
        for match in self.VARIABLE_PATTERN.finditer(self.template_str):
            var_name = match.group(1)
            # Skip if it's part of a conditional syntax
            if not self._is_conditional_syntax(match.start()):
                self.variables.add(var_name)

        # Extract conditional variables
        for match in self.CONDITIONAL_PATTERN.finditer(self.template_str):
            self.conditionals.add(match.group(1))
            # Also extract variables within conditional content
            content = match.group(2)
            for var_match in self.VARIABLE_PATTERN.finditer(content):
                self.variables.add(var_match.group(1))

    def _is_conditional_syntax(self, position: int) -> bool:
        """Check if a position is part of conditional syntax markers."""
        # Check if preceded by ? or followed by / (conditional markers)
        if position > 0 and self.template_str[position - 1] == "?":
            return True
        if position < len(self.template_str) - 1:
            end_pos = self.template_str.find("}", position)
            if end_pos > 0 and position > 0 and self.template_str[position - 1] == "/":
                return True
        return False

    def get_required_variables(self) -> Set[str]:
        """Get all required variables (excluding built-ins and conditionals).

        Returns:
            Set of variable names that must be provided during rendering
        """
        return self.variables - set(self.BUILTIN_VARS.keys()) - self.conditionals

    def get_all_variables(self) -> Set[str]:
        """Get all variables used in the template.

        Returns:
            Set of all variable names (including built-ins and conditionals)
        """
        return self.variables | self.conditionals | set(self.BUILTIN_VARS.keys())

    def validate_context(self, context: Dict[str, Any]) -> List[str]:
        """Validate that all required variables are present in context.

        Args:
            context: Dictionary of variable values

        Returns:
            List of missing variable names (empty if all present)
        """
        required = self.get_required_variables()
        provided = set(context.keys())
        missing = required - provided
        return sorted(list(missing))

    def render(
        self, context: Optional[Dict[str, Any]] = None, strict: bool = True, **kwargs
    ) -> str:
        """Render the template with the provided context.

        Args:
            context: Dictionary of variable values
            strict: If True, raise error on missing variables; if False, leave them as-is
            **kwargs: Additional variables (merged with context)

        Returns:
            Rendered template string

        Raises:
            RenderError: If required variables are missing (in strict mode)
        """
        # Merge context and kwargs
        full_context = {}
        if context:
            full_context.update(context)
        full_context.update(kwargs)

        # Add built-in variables
        for var_name, var_func in self.BUILTIN_VARS.items():
            if var_name not in full_context:
                full_context[var_name] = var_func()

        # Add model_name if not provided but available in context
        if "model_name" not in full_context and "model" in full_context:
            full_context["model_name"] = str(full_context["model"])

        # Validate in strict mode
        if strict:
            missing = self.validate_context(full_context)
            if missing:
                raise RenderError(
                    f"Missing required variables in template '{self.name}': {', '.join(missing)}"
                )

        # Render the template
        result = self.template_str

        # Process conditionals first
        result = self._render_conditionals(result, full_context)

        # Then process regular variables
        result = self._render_variables(result, full_context, strict)

        return result

    def _render_conditionals(self, text: str, context: Dict[str, Any]) -> str:
        """Render conditional sections based on context."""

        def replace_conditional(match):
            condition_var = match.group(1)
            content = match.group(2)

            # Check if condition variable exists and is truthy
            if context.get(condition_var):
                # Render the content if condition is true
                return content
            else:
                # Remove the entire section if condition is false
                return ""

        return self.CONDITIONAL_PATTERN.sub(replace_conditional, text)

    def _render_variables(self, text: str, context: Dict[str, Any], strict: bool) -> str:
        """Render regular variables in the text."""

        def replace_variable(match):
            var_name = match.group(1)

            if var_name in context:
                value = context[var_name]
                # Safe string conversion
                if isinstance(value, (dict, list)):
                    return json.dumps(value, indent=2)
                else:
                    return str(value)
            elif not strict:
                # Keep the original placeholder if not strict
                return match.group(0)
            else:
                # This shouldn't happen if validate_context was called
                raise RenderError(f"Variable '{var_name}' not found in context")

        return self.VARIABLE_PATTERN.sub(replace_variable, text)

    @classmethod
    def from_file(cls, filepath: str | Path, name: Optional[str] = None) -> "PromptTemplate":
        """Create a template from a file.

        Args:
            filepath: Path to the template file
            name: Optional name for the template

        Returns:
            PromptTemplate instance
        """
        path = Path(filepath)
        if not path.exists():
            raise TemplateError(f"Template file not found: {filepath}")
        return cls(path, name=name)

    @classmethod
    def from_string(cls, template: str, name: Optional[str] = None) -> "PromptTemplate":
        """Create a template from a string.

        Args:
            template: Template string
            name: Optional name for the template

        Returns:
            PromptTemplate instance
        """
        return cls(template, name=name)

    def __repr__(self) -> str:
        """String representation of the template."""
        vars_str = ", ".join(sorted(self.get_required_variables()))
        return f"PromptTemplate(name='{self.name}', variables=[{vars_str}])"

    def __str__(self) -> str:
        """String version returns the template string."""
        return self.template_str


# Convenience functions
def load_template(filepath: str | Path) -> PromptTemplate:
    """Load a template from a file."""
    return PromptTemplate.from_file(filepath)


def render_template(template: str, context: Dict[str, Any], **kwargs) -> str:
    """Render a template string with context."""
    tmpl = PromptTemplate.from_string(template)
    return tmpl.render(context, **kwargs)
