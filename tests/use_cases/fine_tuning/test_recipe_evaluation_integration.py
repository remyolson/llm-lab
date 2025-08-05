"""
Test Recipe-Based Evaluation Integration

This script tests the integration between the recipe system and custom evaluation functions.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
from transformers import AutoTokenizer

from src.use_cases.fine_tuning.evaluation import (
    EvaluationSuite,
    EvaluationConfig,
    CustomEvaluationRegistry,
    create_recipe_evaluation_function
)
from src.use_cases.fine_tuning.recipes import RecipeManager


class TestRecipeEvaluationIntegration:
    """Test recipe-based evaluation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.recipe_manager = RecipeManager()
        self.eval_suite = EvaluationSuite(EvaluationConfig(
            benchmarks=[],
            save_results=False
        ))
        
        # Mock model and tokenizer
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        
    def test_custom_evaluation_registry(self):
        """Test custom evaluation function registry."""
        # Check that evaluation functions are registered
        available = CustomEvaluationRegistry.list_available()
        assert "instruction_following" in available
        assert "code_generation" in available
        assert "domain_specific_medical" in available
        assert "chat_coherence" in available
        assert "summarization_quality" in available
        
        # Test getting a specific evaluation
        eval_fn = CustomEvaluationRegistry.get("instruction_following")
        assert eval_fn is not None
        assert callable(eval_fn)
    
    def test_recipe_with_custom_evaluation(self):
        """Test creating recipe with custom evaluation function."""
        recipe = self.recipe_manager.create_recipe_from_config(
            name="test_recipe_with_eval",
            description="Test recipe with custom evaluation",
            model_config={
                "name": "test-model",
                "base_model": "test/model",
                "model_type": "causal_lm"
            },
            dataset_config={
                "name": "test_dataset",
                "path": "test/path",
                "format": "jsonl"
            },
            training_config={
                "num_epochs": 1,
                "learning_rate": 1e-5
            },
            evaluation_config={
                "benchmarks": ["hellaswag"],
                "custom_eval_function": "instruction_following",
                "eval_function_config": {
                    "test_instructions": ["Test instruction 1", "Test instruction 2"]
                }
            }
        )
        
        assert recipe.evaluation is not None
        assert recipe.evaluation.get("custom_eval_function") == "instruction_following"
        assert "eval_function_config" in recipe.evaluation
    
    def test_create_recipe_evaluation_function(self):
        """Test creating evaluation function from recipe."""
        recipe_dict = {
            "name": "test_recipe",
            "evaluation": {
                "custom_eval_function": "chat_coherence",
                "eval_function_config": {
                    "max_turns": 3
                }
            }
        }
        
        eval_fn = create_recipe_evaluation_function(recipe_dict)
        assert eval_fn is not None
        assert callable(eval_fn)
    
    @patch('torch.no_grad')
    def test_instruction_following_evaluation(self, mock_no_grad):
        """Test instruction following evaluation function."""
        from src.use_cases.fine_tuning.evaluation.custom_evaluations import (
            evaluate_instruction_following
        )
        
        # Mock model generate
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        self.mock_tokenizer.return_tensors = "pt"
        self.mock_tokenizer.decode.return_value = "Generated response"
        self.mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
        
        config = {
            "test_instructions": ["Write a test", "Explain testing"]
        }
        
        results = evaluate_instruction_following(
            self.mock_model,
            self.mock_tokenizer,
            config
        )
        
        assert len(results) > 0
        assert any(r["name"] == "instruction_following_overall" for r in results)
    
    @patch('torch.no_grad')
    def test_code_generation_evaluation(self, mock_no_grad):
        """Test code generation evaluation function."""
        from src.use_cases.fine_tuning.evaluation.custom_evaluations import (
            evaluate_code_generation
        )
        
        # Mock model generate
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        self.mock_tokenizer.decode.return_value = "factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)"
        self.mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
        
        results = evaluate_code_generation(
            self.mock_model,
            self.mock_tokenizer
        )
        
        assert len(results) > 0
        assert any(r["name"] == "code_generation_overall" for r in results)
    
    def test_evaluation_suite_with_recipe(self):
        """Test EvaluationSuite.evaluate_with_recipe method."""
        recipe = {
            "name": "test_recipe",
            "recipe_type": "instruction_tuning",
            "evaluation": {
                "benchmarks": ["hellaswag"],
                "custom_eval_function": "instruction_following",
                "eval_function_config": {
                    "test_instructions": ["Test"]
                }
            }
        }
        
        with patch.object(self.eval_suite, 'evaluate') as mock_evaluate:
            mock_result = Mock()
            mock_result.metadata = {}
            mock_evaluate.return_value = mock_result
            
            result = self.eval_suite.evaluate_with_recipe(
                self.mock_model,
                self.mock_tokenizer,
                recipe
            )
            
            # Verify evaluate was called with custom eval function
            mock_evaluate.assert_called_once()
            call_args = mock_evaluate.call_args
            assert call_args[1]["benchmarks"] == ["hellaswag"]
            assert call_args[1]["custom_eval_fn"] is not None
            
            # Verify recipe metadata was added
            assert result.metadata["recipe_name"] == "test_recipe"
            assert result.metadata["recipe_type"] == "instruction_tuning"
    
    def test_recipe_validation_with_evaluation(self):
        """Test that recipe validation accepts evaluation fields."""
        recipe_data = {
            "name": "valid_recipe_with_eval",
            "description": "Valid recipe with evaluation",
            "recipe_type": "instruction_tuning",
            "version": "1.0.0",
            "author": "test",
            "model": {
                "name": "test-model",
                "base_model": "test/model",
                "model_type": "causal_lm"
            },
            "dataset": {
                "name": "test_dataset",
                "path": "test/path",
                "format": "jsonl"
            },
            "training": {
                "num_epochs": 1,
                "learning_rate": 1e-5
            },
            "evaluation": {
                "benchmarks": ["hellaswag", "mmlu"],
                "custom_eval_function": "instruction_following",
                "eval_function_config": {
                    "test_instructions": ["Test 1", "Test 2"],
                    "min_score": 0.8
                }
            }
        }
        
        # This should not raise an exception
        self.recipe_manager.validate_recipe_data(recipe_data)
    
    def test_medical_domain_evaluation(self):
        """Test medical domain evaluation function."""
        from src.use_cases.fine_tuning.evaluation.custom_evaluations import (
            evaluate_medical_domain
        )
        
        with patch('torch.no_grad'):
            self.mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
            self.mock_tokenizer.decode.return_value = "immune system protein defense"
            self.mock_tokenizer.return_value = {"input_ids": torch.tensor([[1]])}
            
            results = evaluate_medical_domain(
                self.mock_model,
                self.mock_tokenizer
            )
            
            assert len(results) > 0
            assert any(r["name"] == "medical_domain_overall" for r in results)
            
            # Check that accuracy is calculated based on keywords
            medical_result = next(r for r in results if "medical_" in r["name"])
            assert "keywords_found" in medical_result["metadata"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])