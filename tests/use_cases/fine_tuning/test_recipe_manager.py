"""
Tests for Recipe Manager
"""

import pytest
import json
import yaml
from pathlib import Path
import tempfile
from datetime import datetime

from src.use_cases.fine_tuning.recipes import (
    Recipe,
    RecipeManager,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    EvaluationConfig
)


class TestRecipeManager:
    """Test cases for RecipeManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test recipes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def manager(self, temp_dir):
        """Create a RecipeManager instance with temp directory."""
        return RecipeManager(recipes_dir=temp_dir)
    
    @pytest.fixture
    def sample_recipe_data(self):
        """Sample recipe data for testing."""
        return {
            "name": "test_recipe",
            "description": "Test recipe for unit tests",
            "version": "1.0.0",
            "author": "test_user",
            "tags": ["test", "sample"],
            "model": {
                "name": "test-model",
                "base_model": "meta-llama/Llama-2-7b-hf",
                "model_type": "causal_lm",
                "use_flash_attention": True
            },
            "dataset": {
                "name": "test_dataset",
                "path": "test/dataset/path",
                "format": "jsonl",
                "split_ratios": {
                    "train": 0.8,
                    "validation": 0.1,
                    "test": 0.1
                }
            },
            "training": {
                "num_epochs": 3,
                "learning_rate": 2e-5,
                "use_lora": True,
                "lora_rank": 16
            },
            "evaluation": {
                "metrics": ["perplexity", "accuracy"],
                "benchmarks": ["mmlu", "hellaswag"]
            }
        }
    
    def test_recipe_creation(self):
        """Test creating a Recipe object."""
        recipe = Recipe(
            name="test_recipe",
            description="Test recipe",
            model=ModelConfig(name="test", base_model="llama2"),
            dataset=DatasetConfig(name="test", path="/path/to/data"),
            training=TrainingConfig(),
            evaluation=EvaluationConfig()
        )
        
        assert recipe.name == "test_recipe"
        assert recipe.description == "Test recipe"
        assert recipe.model.name == "test"
        assert recipe.dataset.name == "test"
    
    def test_recipe_to_dict_conversion(self):
        """Test converting Recipe to dictionary."""
        recipe = Recipe(
            name="test_recipe",
            description="Test recipe",
            model=ModelConfig(name="test", base_model="llama2"),
            dataset=DatasetConfig(name="test", path="/path/to/data")
        )
        
        recipe_dict = recipe.to_dict()
        
        assert recipe_dict["name"] == "test_recipe"
        assert recipe_dict["model"]["name"] == "test"
        assert "created_at" in recipe_dict
        assert "training" in recipe_dict
    
    def test_recipe_from_dict_conversion(self, sample_recipe_data):
        """Test creating Recipe from dictionary."""
        recipe = Recipe.from_dict(sample_recipe_data)
        
        assert recipe.name == "test_recipe"
        assert recipe.model.base_model == "meta-llama/Llama-2-7b-hf"
        assert recipe.training.use_lora is True
        assert recipe.evaluation.metrics == ["perplexity", "accuracy"]
    
    def test_recipe_validation_valid(self, manager, sample_recipe_data):
        """Test recipe validation with valid data."""
        # Should not raise any exception
        manager.validate_recipe_data(sample_recipe_data)
    
    def test_recipe_validation_missing_required(self, manager):
        """Test recipe validation with missing required fields."""
        invalid_data = {
            "name": "test_recipe",
            # Missing description
            "model": {
                "name": "test",
                "base_model": "llama2"
            }
            # Missing dataset
        }
        
        with pytest.raises(jsonschema.ValidationError):
            manager.validate_recipe_data(invalid_data)
    
    def test_recipe_validation_invalid_split_ratios(self, manager, sample_recipe_data):
        """Test recipe validation with invalid split ratios."""
        sample_recipe_data["dataset"]["split_ratios"] = {
            "train": 0.6,
            "validation": 0.2,
            "test": 0.1  # Sum is 0.9, not 1.0
        }
        
        with pytest.raises(jsonschema.ValidationError):
            manager.validate_recipe_data(sample_recipe_data)
    
    def test_save_and_load_recipe(self, manager, sample_recipe_data):
        """Test saving and loading a recipe."""
        recipe = Recipe.from_dict(sample_recipe_data)
        
        # Save recipe
        manager.save_recipe(recipe)
        
        # Load recipe
        loaded_recipe = manager.load_recipe("test_recipe")
        
        assert loaded_recipe.name == recipe.name
        assert loaded_recipe.description == recipe.description
        assert loaded_recipe.model.base_model == recipe.model.base_model
    
    def test_save_recipe_overwrite_protection(self, manager, sample_recipe_data):
        """Test that save_recipe prevents overwriting by default."""
        recipe = Recipe.from_dict(sample_recipe_data)
        
        # Save recipe
        manager.save_recipe(recipe)
        
        # Try to save again without overwrite
        with pytest.raises(FileExistsError):
            manager.save_recipe(recipe, overwrite=False)
        
        # Save with overwrite should work
        manager.save_recipe(recipe, overwrite=True)
    
    def test_list_recipes(self, manager, sample_recipe_data):
        """Test listing available recipes."""
        # Initially empty
        assert manager.list_recipes() == []
        
        # Create and save multiple recipes
        for i in range(3):
            recipe_data = sample_recipe_data.copy()
            recipe_data["name"] = f"recipe_{i}"
            recipe = Recipe.from_dict(recipe_data)
            manager.save_recipe(recipe)
        
        # List should contain all recipes
        recipes = manager.list_recipes()
        assert len(recipes) == 3
        assert "recipe_0" in recipes
        assert "recipe_1" in recipes
        assert "recipe_2" in recipes
    
    def test_get_recipe_info(self, manager, sample_recipe_data):
        """Test getting recipe information."""
        recipe = Recipe.from_dict(sample_recipe_data)
        manager.save_recipe(recipe)
        
        info = manager.get_recipe_info("test_recipe")
        
        assert info["name"] == "test_recipe"
        assert info["description"] == "Test recipe for unit tests"
        assert info["model"] == "meta-llama/Llama-2-7b-hf"
        assert info["dataset"] == "test_dataset"
        assert "created_at" in info
    
    def test_create_recipe_from_config(self, manager):
        """Test creating recipe from configuration dictionaries."""
        recipe = manager.create_recipe_from_config(
            name="config_recipe",
            description="Recipe from config",
            model_config={
                "name": "test-model",
                "base_model": "llama2-7b"
            },
            dataset_config={
                "name": "test-data",
                "path": "/data/path"
            },
            training_config={
                "num_epochs": 5,
                "learning_rate": 1e-4
            },
            author="config_test",
            tags=["config", "test"]
        )
        
        assert recipe.name == "config_recipe"
        assert recipe.training.num_epochs == 5
        assert recipe.author == "config_test"
    
    def test_duplicate_recipe(self, manager, sample_recipe_data):
        """Test duplicating a recipe with modifications."""
        original = Recipe.from_dict(sample_recipe_data)
        manager.save_recipe(original)
        
        # Duplicate with modifications
        duplicate = manager.duplicate_recipe(
            "test_recipe",
            "test_recipe_v2",
            modifications={
                "description": "Modified test recipe",
                "training": {
                    "num_epochs": 5,
                    "learning_rate": 1e-4
                }
            }
        )
        
        assert duplicate.name == "test_recipe_v2"
        assert duplicate.description == "Modified test recipe"
        assert duplicate.training.num_epochs == 5
        assert duplicate.training.learning_rate == 1e-4
        # Original values should be preserved
        assert duplicate.model.base_model == original.model.base_model
    
    def test_export_import_recipe_yaml(self, manager, sample_recipe_data):
        """Test exporting and importing recipe as YAML."""
        recipe = Recipe.from_dict(sample_recipe_data)
        
        # Export to YAML
        yaml_str = manager.export_recipe(recipe, format="yaml")
        
        # Import from YAML
        imported = manager.import_recipe(yaml_str, format="yaml")
        
        assert imported.name == recipe.name
        assert imported.model.base_model == recipe.model.base_model
        assert imported.training.use_lora == recipe.training.use_lora
    
    def test_export_import_recipe_json(self, manager, sample_recipe_data):
        """Test exporting and importing recipe as JSON."""
        recipe = Recipe.from_dict(sample_recipe_data)
        
        # Export to JSON
        json_str = manager.export_recipe(recipe, format="json")
        
        # Import from JSON
        imported = manager.import_recipe(json_str, format="json")
        
        assert imported.name == recipe.name
        assert imported.model.base_model == recipe.model.base_model
        assert imported.training.use_lora == recipe.training.use_lora
    
    def test_recipe_caching(self, manager, sample_recipe_data):
        """Test that recipes are cached after loading."""
        recipe = Recipe.from_dict(sample_recipe_data)
        manager.save_recipe(recipe)
        
        # First load
        loaded1 = manager.load_recipe("test_recipe")
        
        # Second load should come from cache
        loaded2 = manager.load_recipe("test_recipe")
        
        # They should be the same object (cached)
        assert loaded1 is loaded2
    
    def test_load_nonexistent_recipe(self, manager):
        """Test loading a recipe that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            manager.load_recipe("nonexistent_recipe")
    
    def test_deep_update(self, manager):
        """Test the deep update utility method."""
        base = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        updates = {
            "b": {
                "d": {
                    "e": 4,
                    "f": 5
                }
            },
            "g": 6
        }
        
        manager._deep_update(base, updates)
        
        assert base["a"] == 1  # Unchanged
        assert base["b"]["c"] == 2  # Unchanged
        assert base["b"]["d"]["e"] == 4  # Updated
        assert base["b"]["d"]["f"] == 5  # Added
        assert base["g"] == 6  # Added


class TestModelConfig:
    """Test cases for ModelConfig dataclass."""
    
    def test_model_config_creation(self):
        """Test creating ModelConfig."""
        config = ModelConfig(
            name="llama2-7b-instruct",
            base_model="meta-llama/Llama-2-7b-hf",
            quantization="int8",
            use_flash_attention=True
        )
        
        assert config.name == "llama2-7b-instruct"
        assert config.quantization == "int8"
        assert config.use_flash_attention is True
        assert config.torch_dtype == "float16"  # Default value
    
    def test_model_config_to_dict(self):
        """Test converting ModelConfig to dict."""
        config = ModelConfig(
            name="test",
            base_model="llama2"
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["name"] == "test"
        assert config_dict["base_model"] == "llama2"
        assert "device_map" in config_dict


class TestDatasetConfig:
    """Test cases for DatasetConfig dataclass."""
    
    def test_dataset_config_defaults(self):
        """Test DatasetConfig default values."""
        config = DatasetConfig(
            name="test_dataset",
            path="/path/to/data"
        )
        
        assert config.format == "jsonl"
        assert config.split_ratios["train"] == 0.8
        assert config.split_ratios["validation"] == 0.1
        assert config.split_ratios["test"] == 0.1
        assert config.tokenizer_config["max_length"] == 512


class TestTrainingConfig:
    """Test cases for TrainingConfig dataclass."""
    
    def test_training_config_lora_defaults(self):
        """Test TrainingConfig LoRA default values."""
        config = TrainingConfig()
        
        assert config.use_lora is True
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert "q_proj" in config.lora_target_modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])