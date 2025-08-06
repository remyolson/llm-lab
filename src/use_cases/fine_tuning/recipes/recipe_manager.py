"""
Recipe Manager for Fine-Tuning Pipeline

This module provides a comprehensive recipe management system for defining,
loading, and managing fine-tuning configurations. Recipes are YAML/JSON
configurations that define complete fine-tuning workflows including model
selection, dataset configuration, training hyperparameters, and evaluation metrics.

Example:
    manager = RecipeManager()
    recipe = manager.load_recipe("instruction_tuning")
    validated_recipe = manager.validate_recipe(recipe)
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jsonschema import ValidationError, validate

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model selection and initialization."""

    name: str
    base_model: str
    model_type: str = "causal_lm"  # causal_lm, seq2seq, etc.
    quantization: Optional[str] = None  # int8, int4, etc.
    device_map: str = "auto"
    torch_dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""

    name: str
    path: str
    format: str = "jsonl"  # jsonl, csv, huggingface, parquet
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    max_samples: Optional[int] = None
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    tokenizer_config: Dict[str, Any] = field(
        default_factory=lambda: {"max_length": 512, "padding": "max_length", "truncation": True}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    deepspeed_config: Optional[Dict[str, Any]] = None
    fsdp_config: Optional[Dict[str, Any]] = None

    # LoRA specific parameters
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and benchmarks."""

    metrics: List[str] = field(default_factory=lambda: ["perplexity", "bleu", "rouge", "accuracy"])
    benchmarks: List[str] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    eval_batch_size: int = 8
    num_beams: int = 4
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class Recipe:
    """Complete recipe configuration for fine-tuning."""

    name: str
    description: str
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = "system"
    tags: List[str] = field(default_factory=list)

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert recipe to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "author": self.author,
            "tags": self.tags,
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recipe":
        """Create Recipe from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            author=data.get("author", "system"),
            tags=data.get("tags", []),
            model=ModelConfig(**data.get("model", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            training=TrainingConfig(**data.get("training", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            metadata=data.get("metadata", {}),
        )


class RecipeManager:
    """Manages fine-tuning recipes including loading, validation, and registry."""

    # JSON Schema for recipe validation
    RECIPE_SCHEMA = {
        "type": "object",
        "required": ["name", "description", "model", "dataset", "training"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "description": {"type": "string"},
            "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
            "author": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "model": {
                "type": "object",
                "required": ["name", "base_model"],
                "properties": {
                    "name": {"type": "string"},
                    "base_model": {"type": "string"},
                    "model_type": {"type": "string", "enum": ["causal_lm", "seq2seq"]},
                    "quantization": {"type": ["string", "null"]},
                    "device_map": {"type": "string"},
                    "torch_dtype": {"type": "string"},
                    "load_in_8bit": {"type": "boolean"},
                    "load_in_4bit": {"type": "boolean"},
                    "use_flash_attention": {"type": "boolean"},
                },
            },
            "dataset": {
                "type": "object",
                "required": ["name", "path"],
                "properties": {
                    "name": {"type": "string"},
                    "path": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["jsonl", "csv", "huggingface", "parquet"],
                    },
                    "split_ratios": {
                        "type": "object",
                        "properties": {
                            "train": {"type": "number", "minimum": 0, "maximum": 1},
                            "validation": {"type": "number", "minimum": 0, "maximum": 1},
                            "test": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                    "max_samples": {"type": ["integer", "null"], "minimum": 1},
                    "preprocessing": {"type": "object"},
                    "tokenizer_config": {"type": "object"},
                },
            },
            "training": {
                "type": "object",
                "properties": {
                    "num_epochs": {"type": "integer", "minimum": 1},
                    "per_device_train_batch_size": {"type": "integer", "minimum": 1},
                    "learning_rate": {"type": "number", "minimum": 0},
                    "use_lora": {"type": "boolean"},
                    "lora_rank": {"type": "integer", "minimum": 1},
                    "lora_alpha": {"type": "integer", "minimum": 1},
                    "lora_dropout": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            "evaluation": {
                "type": "object",
                "properties": {
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "benchmarks": {"type": "array", "items": {"type": "string"}},
                    "custom_metrics": {"type": "object"},
                    "custom_eval_function": {"type": "string"},
                    "eval_function_config": {"type": "object"},
                },
            },
            "metadata": {"type": "object"},
        },
    }

    def __init__(self, recipes_dir: Optional[str] = None):
        """Initialize RecipeManager.

        Args:
            recipes_dir: Directory containing recipe files. Defaults to
                        src/use_cases/fine_tuning/recipes/templates/
        """
        if recipes_dir is None:
            self.recipes_dir = Path(__file__).parent / "templates"
        else:
            self.recipes_dir = Path(recipes_dir)

        self.recipes_dir.mkdir(parents=True, exist_ok=True)
        self._recipe_cache: Dict[str, Recipe] = {}
        self._load_builtin_recipes()

    def _load_builtin_recipes(self):
        """Load built-in recipe templates."""
        try:
            from .template_loader import TemplateLoader

            loader = TemplateLoader(self.recipes_dir)

            # Load template metadata
            templates = loader.list_templates()
            logger.info(f"Found {len(templates)} built-in recipe templates")

            # Pre-cache template info for quick access
            self._template_info = {t["id"]: t for t in templates}

        except Exception as e:
            logger.warning(f"Failed to load built-in templates: {e}")
            self._template_info = {}

    def load_recipe(self, recipe_name: str) -> Recipe:
        """Load a recipe by name.

        Args:
            recipe_name: Name of the recipe to load

        Returns:
            Loaded Recipe object

        Raises:
            FileNotFoundError: If recipe file not found
            ValidationError: If recipe validation fails
        """
        # Check cache first
        if recipe_name in self._recipe_cache:
            logger.info(f"Loading recipe '{recipe_name}' from cache")
            return self._recipe_cache[recipe_name]

        # Try loading from file
        recipe_path = None
        for ext in [".yaml", ".yml", ".json"]:
            candidate = self.recipes_dir / f"{recipe_name}{ext}"
            if candidate.exists():
                recipe_path = candidate
                break

        if recipe_path is None:
            raise FileNotFoundError(f"Recipe '{recipe_name}' not found in {self.recipes_dir}")

        logger.info(f"Loading recipe from {recipe_path}")

        # Load recipe data
        with open(recipe_path) as f:
            if recipe_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Validate and create recipe
        self.validate_recipe_data(data)
        recipe = Recipe.from_dict(data)

        # Cache the recipe
        self._recipe_cache[recipe_name] = recipe

        return recipe

    def save_recipe(self, recipe: Recipe, overwrite: bool = False):
        """Save a recipe to file.

        Args:
            recipe: Recipe object to save
            overwrite: Whether to overwrite existing recipe

        Raises:
            FileExistsError: If recipe exists and overwrite=False
        """
        recipe_path = self.recipes_dir / f"{recipe.name}.yaml"

        if recipe_path.exists() and not overwrite:
            raise FileExistsError(f"Recipe '{recipe.name}' already exists")

        # Validate before saving
        recipe_data = recipe.to_dict()
        self.validate_recipe_data(recipe_data)

        # Save to file
        with open(recipe_path, "w") as f:
            yaml.dump(recipe_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved recipe to {recipe_path}")

        # Update cache
        self._recipe_cache[recipe.name] = recipe

    def validate_recipe_data(self, data: Dict[str, Any]):
        """Validate recipe data against schema.

        Args:
            data: Recipe data to validate

        Raises:
            ValidationError: If validation fails
        """
        try:
            validate(instance=data, schema=self.RECIPE_SCHEMA)
        except ValidationError as e:
            logger.error(f"Recipe validation failed: {e.message}")
            raise

        # Additional validation
        if "dataset" in data and "split_ratios" in data["dataset"]:
            ratios = data["dataset"]["split_ratios"]
            total = sum(ratios.values())
            if abs(total - 1.0) > 0.001:
                raise ValidationError(f"Dataset split ratios must sum to 1.0, got {total}")

    def list_recipes(self) -> List[str]:
        """List all available recipes.

        Returns:
            List of recipe names
        """
        recipes = []

        # List files in recipes directory
        for path in self.recipes_dir.iterdir():
            if path.is_file() and path.suffix in [".yaml", ".yml", ".json"]:
                recipes.append(path.stem)

        # Add cached recipes not on disk
        recipes.extend([name for name in self._recipe_cache if name not in recipes])

        return sorted(set(recipes))

    def list_templates(self) -> List[Dict[str, Any]]:
        """List available recipe templates with metadata.

        Returns:
            List of template information dictionaries
        """
        if hasattr(self, "_template_info"):
            return list(self._template_info.values())
        return []

    def get_template_info(self, template_id: str) -> Dict[str, Any]:
        """Get information about a specific template.

        Args:
            template_id: Template identifier

        Returns:
            Template metadata or None
        """
        if hasattr(self, "_template_info"):
            return self._template_info.get(template_id)
        return None

    def get_recipe_info(self, recipe_name: str) -> Dict[str, Any]:
        """Get basic information about a recipe without full loading.

        Args:
            recipe_name: Name of the recipe

        Returns:
            Dictionary with recipe metadata
        """
        recipe = self.load_recipe(recipe_name)
        return {
            "name": recipe.name,
            "description": recipe.description,
            "version": recipe.version,
            "author": recipe.author,
            "tags": recipe.tags,
            "created_at": recipe.created_at,
            "model": recipe.model.base_model,
            "dataset": recipe.dataset.name,
        }

    def create_recipe_from_config(
        self,
        name: str,
        description: str,
        model_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        training_config: Optional[Dict[str, Any]] = None,
        evaluation_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Recipe:
        """Create a new recipe from configuration dictionaries.

        Args:
            name: Recipe name
            description: Recipe description
            model_config: Model configuration dictionary
            dataset_config: Dataset configuration dictionary
            training_config: Training configuration dictionary
            evaluation_config: Evaluation configuration dictionary
            **kwargs: Additional recipe fields (author, tags, etc.)

        Returns:
            New Recipe object
        """
        recipe_data = {
            "name": name,
            "description": description,
            "model": model_config,
            "dataset": dataset_config,
            "training": training_config or {},
            "evaluation": evaluation_config or {},
            **kwargs,
        }

        # Validate the complete recipe data
        self.validate_recipe_data(recipe_data)

        # Create and return recipe
        return Recipe.from_dict(recipe_data)

    def duplicate_recipe(
        self, source_recipe_name: str, new_name: str, modifications: Optional[Dict[str, Any]] = None
    ) -> Recipe:
        """Duplicate an existing recipe with modifications.

        Args:
            source_recipe_name: Name of recipe to duplicate
            new_name: Name for the new recipe
            modifications: Dictionary of fields to modify

        Returns:
            New Recipe object
        """
        # Load source recipe
        source = self.load_recipe(source_recipe_name)

        # Convert to dict and update
        recipe_data = source.to_dict()
        recipe_data["name"] = new_name
        recipe_data["created_at"] = datetime.now().isoformat()

        if modifications:
            # Deep update the recipe data
            self._deep_update(recipe_data, modifications)

        # Create new recipe
        return Recipe.from_dict(recipe_data)

    def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]):
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def export_recipe(self, recipe: Recipe, format: str = "yaml") -> str:
        """Export recipe to string in specified format.

        Args:
            recipe: Recipe to export
            format: Export format ('yaml' or 'json')

        Returns:
            Recipe as formatted string
        """
        recipe_data = recipe.to_dict()

        if format == "json":
            return json.dumps(recipe_data, indent=2)
        else:
            return yaml.dump(recipe_data, default_flow_style=False, sort_keys=False)

    def import_recipe(self, recipe_str: str, format: str = "yaml") -> Recipe:
        """Import recipe from string.

        Args:
            recipe_str: Recipe string
            format: Format of the string ('yaml' or 'json')

        Returns:
            Imported Recipe object
        """
        if format == "json":
            data = json.loads(recipe_str)
        else:
            data = yaml.safe_load(recipe_str)

        self.validate_recipe_data(data)
        return Recipe.from_dict(data)


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = RecipeManager()

    # Create a sample recipe
    recipe = manager.create_recipe_from_config(
        name="instruction_tuning_example",
        description="Example recipe for instruction tuning",
        model_config={
            "name": "llama2-7b-instruct",
            "base_model": "meta-llama/Llama-2-7b-hf",
            "model_type": "causal_lm",
            "use_flash_attention": True,
        },
        dataset_config={
            "name": "alpaca_cleaned",
            "path": "yahma/alpaca-cleaned",
            "format": "huggingface",
        },
        training_config={"num_epochs": 3, "learning_rate": 2e-5, "use_lora": True, "lora_rank": 16},
        author="system",
        tags=["instruction-tuning", "llama2", "lora"],
    )

    # Save recipe
    manager.save_recipe(recipe)

    # List recipes
    print(f"Available recipes: {manager.list_recipes()}")

    # Export recipe
    print("\nRecipe YAML:")
    print(manager.export_recipe(recipe))
