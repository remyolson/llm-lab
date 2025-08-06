"""Template engine for documentation generation."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template

logger = logging.getLogger(__name__)


class TemplateEngine:
    """Jinja2-based template engine."""

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the template engine.

        Args:
            template_dir: Directory containing templates
        """
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / "templates"

        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        self.env.filters["format_number"] = self._format_number
        self.env.filters["format_percentage"] = self._format_percentage

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render a template with context.

        Args:
            template_name: Name of the template file
            context: Template context variables

        Returns:
            Rendered string
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            raise

    def render_string(self, template_string: str, context: Dict[str, Any]) -> str:
        """
        Render a template string.

        Args:
            template_string: Template as string
            context: Template context variables

        Returns:
            Rendered string
        """
        try:
            template = Template(template_string)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Error rendering template string: {e}")
            raise

    @staticmethod
    def _format_number(value: float, decimals: int = 2) -> str:
        """Format number with thousand separators."""
        if value is None:
            return "N/A"
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def _format_percentage(value: float, decimals: int = 2) -> str:
        """Format value as percentage."""
        if value is None:
            return "N/A"
        return f"{value * 100:.{decimals}f}%"
