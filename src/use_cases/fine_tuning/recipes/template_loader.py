"""
Template Loader for Pre-configured Recipes

This module handles loading and managing pre-configured recipe templates.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class TemplateLoader:
    """Loads and manages pre-configured recipe templates."""

    # Template metadata
    TEMPLATE_INFO = {
        "instruction_tuning_alpaca": {
            "name": "Instruction Tuning (Alpaca)",
            "description": "General instruction following using Alpaca format",
            "use_cases": ["instruction following", "task completion", "general assistant"],
            "difficulty": "beginner",
            "estimated_time": "2-4 hours",
        },
        "chat_finetuning_chatml": {
            "name": "Chat Fine-tuning (ChatML)",
            "description": "Multi-turn conversational AI using ChatML format",
            "use_cases": ["chatbot", "conversational AI", "customer support"],
            "difficulty": "intermediate",
            "estimated_time": "3-5 hours",
        },
        "code_generation": {
            "name": "Code Generation",
            "description": "Programming and code completion tasks",
            "use_cases": ["code completion", "code generation", "programming assistant"],
            "difficulty": "intermediate",
            "estimated_time": "4-6 hours",
        },
        "domain_adaptation_medical": {
            "name": "Medical Domain Adaptation",
            "description": "Specialized medical/healthcare applications",
            "use_cases": ["medical QA", "clinical support", "healthcare information"],
            "difficulty": "advanced",
            "estimated_time": "6-8 hours",
        },
        "task_specific_summarization": {
            "name": "Text Summarization",
            "description": "Document and article summarization",
            "use_cases": ["news summarization", "document summary", "meeting notes"],
            "difficulty": "intermediate",
            "estimated_time": "3-5 hours",
        },
        "macbook_pro_optimized": {
            "name": "MacBook Pro Optimized",
            "description": "Memory-efficient training for Apple Silicon",
            "use_cases": ["local training", "experimentation", "learning"],
            "difficulty": "beginner",
            "estimated_time": "1-2 hours",
        },
    }

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize template loader.

        Args:
            templates_dir: Directory containing template files
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        self.templates_dir = Path(templates_dir)
        self._template_cache = {}

    def list_templates(self) -> List[Dict[str | any]]:
        """List all available templates with metadata.

        Returns:
            List of template information dictionaries
        """
        templates = []

        for template_id, info in self.TEMPLATE_INFO.items():
            template_file = self.templates_dir / f"{template_id}.yaml"

            if template_file.exists():
                template_info = {"id": template_id, "file": template_file.name, **info}
                templates.append(template_info)
            else:
                logger.warning(f"Template file not found: {template_file}")

        return templates

    def load_template(self, template_id: str) -> Dict[str | any]:
        """Load a specific template.

        Args:
            template_id: Template identifier

        Returns:
            Template configuration dictionary
        """
        # Check cache
        if template_id in self._template_cache:
            return self._template_cache[template_id].copy()

        # Load from file
        template_file = self.templates_dir / f"{template_id}.yaml"

        if not template_file.exists():
            raise FileNotFoundError(f"Template not found: {template_id}")

        with open(template_file) as f:
            template_config = yaml.safe_load(f)

        # Cache and return
        self._template_cache[template_id] = template_config
        return template_config.copy()

    def get_template_info(self, template_id: str) -> Dict[str | any]:
        """Get metadata about a template without loading it.

        Args:
            template_id: Template identifier

        Returns:
            Template metadata
        """
        if template_id not in self.TEMPLATE_INFO:
            raise ValueError(f"Unknown template: {template_id}")

        return self.TEMPLATE_INFO[template_id].copy()

    def get_templates_by_use_case(self, use_case: str) -> List[str]:
        """Find templates suitable for a specific use case.

        Args:
            use_case: Use case description

        Returns:
            List of matching template IDs
        """
        matching_templates = []
        use_case_lower = use_case.lower()

        for template_id, info in self.TEMPLATE_INFO.items():
            template_use_cases = [uc.lower() for uc in info["use_cases"]]

            # Check if use case matches any template use cases
            if any(use_case_lower in uc or uc in use_case_lower for uc in template_use_cases):
                matching_templates.append(template_id)

        return matching_templates

    def get_templates_by_difficulty(self, difficulty: str) -> List[str]:
        """Get templates filtered by difficulty level.

        Args:
            difficulty: Difficulty level (beginner, intermediate, advanced)

        Returns:
            List of template IDs
        """
        difficulty_lower = difficulty.lower()

        return [
            template_id
            for template_id, info in self.TEMPLATE_INFO.items()
            if info["difficulty"] == difficulty_lower
        ]

    def validate_template(self, template_config: Dict[str, any]) -> bool:
        """Validate a template configuration.

        Args:
            template_config: Template configuration to validate

        Returns:
            True if valid
        """
        required_fields = ["name", "description", "model", "dataset", "training"]

        for field in required_fields:
            if field not in template_config:
                logger.error(f"Template missing required field: {field}")
                return False

        # Validate model configuration
        model_config = template_config.get("model", {})
        if "base_model" not in model_config:
            logger.error("Template missing model.base_model")
            return False

        # Validate dataset configuration
        dataset_config = template_config.get("dataset", {})
        if "path" not in dataset_config:
            logger.error("Template missing dataset.path")
            return False

        return True

    def create_custom_template(
        self, base_template_id: str, modifications: Dict[str, any], save_as: Optional[str] = None
    ) -> Dict[str | any]:
        """Create a custom template based on an existing one.

        Args:
            base_template_id: Base template to modify
            modifications: Dictionary of modifications
            save_as: Optional filename to save template

        Returns:
            Modified template configuration
        """
        # Load base template
        base_template = self.load_template(base_template_id)

        # Apply modifications
        custom_template = self._deep_update(base_template.copy(), modifications)

        # Update metadata
        custom_template["name"] = modifications.get("name", f"{base_template['name']}_custom")
        custom_template["version"] = "1.0.0-custom"

        # Save if requested
        if save_as:
            save_path = self.templates_dir / f"{save_as}.yaml"
            with open(save_path, "w") as f:
                yaml.dump(custom_template, f, default_flow_style=False)

            logger.info(f"Saved custom template to {save_path}")

        return custom_template

    def _deep_update(self, base: Dict, updates: Dict) -> Dict:
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_update(base[key], value)
            else:
                base[key] = value
        return base


# Convenience functions
def list_available_templates() -> List[Dict[str | any]]:
    """List all available recipe templates."""
    loader = TemplateLoader()
    return loader.list_templates()


def load_template(template_id: str) -> Dict[str | any]:
    """Load a specific recipe template."""
    loader = TemplateLoader()
    return loader.load_template(template_id)


def get_template_for_use_case(use_case: str) -> Optional[str]:
    """Get the most suitable template for a use case."""
    loader = TemplateLoader()
    templates = loader.get_templates_by_use_case(use_case)

    if templates:
        # Return the first match (could be enhanced with ranking)
        return templates[0]

    return None


# Example usage
if __name__ == "__main__":
    # List all templates
    templates = list_available_templates()
    print(f"Available templates: {len(templates)}")
    for template in templates:
        print(f"  - {template['id']}: {template['name']}")
        print(f"    Use cases: {', '.join(template['use_cases'])}")
        print(f"    Difficulty: {template['difficulty']}")
        print()

    # Load a specific template
    alpaca_template = load_template("instruction_tuning_alpaca")
    print(f"\nLoaded template: {alpaca_template['name']}")
    print(f"Model: {alpaca_template['model']['base_model']}")
    print(f"Dataset: {alpaca_template['dataset']['path']}")

    # Find template for use case
    chat_template_id = get_template_for_use_case("chatbot")
    if chat_template_id:
        print(f"\nRecommended template for 'chatbot': {chat_template_id}")
