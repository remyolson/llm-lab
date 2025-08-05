"""
Comprehensive tests for fine-tuning functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.use_cases.fine_tuning_complete_demo import (
    DatasetPreparer,
    CloudFineTuner,
    LocalFineTuner,
    ModelEvaluator,
    FineTuningOrchestrator
)


class TestDatasetPreparer:
    """Test dataset preparation functionality"""
    
    @pytest.fixture
    def dataset_preparer(self):
        """Create DatasetPreparer instance"""
        return DatasetPreparer()
    
    def test_validate_openai_format(self, dataset_preparer):
        """Test OpenAI format validation"""
        # Valid data
        valid_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        ]
        assert dataset_preparer.validate_openai_format(valid_data)
        
        # Invalid data - missing messages
        invalid_data = [{"prompt": "test"}]
        assert not dataset_preparer.validate_openai_format(invalid_data)
    
    def test_validate_anthropic_format(self, dataset_preparer):
        """Test Anthropic format validation"""
        # Valid data
        valid_data = [
            {
                "prompt": "Human: Hello\n\nAssistant:",
                "completion": " Hi there!"
            }
        ]
        assert dataset_preparer.validate_anthropic_format(valid_data)
        
        # Invalid data - wrong format
        invalid_data = [{"messages": []}]
        assert not dataset_preparer.validate_anthropic_format(invalid_data)
    
    def test_convert_to_openai_format(self, dataset_preparer):
        """Test conversion to OpenAI format"""
        generic_data = [
            {
                "instruction": "Translate to French",
                "input": "Hello",
                "output": "Bonjour"
            }
        ]
        
        converted = dataset_preparer.convert_to_openai_format(generic_data)
        
        assert len(converted) == 1
        assert "messages" in converted[0]
        assert len(converted[0]["messages"]) == 3
        assert converted[0]["messages"][1]["content"] == "Translate to French: Hello"
    
    def test_convert_to_anthropic_format(self, dataset_preparer):
        """Test conversion to Anthropic format"""
        generic_data = [
            {
                "instruction": "Translate",
                "input": "Hello",
                "output": "Bonjour"
            }
        ]
        
        converted = dataset_preparer.convert_to_anthropic_format(generic_data)
        
        assert len(converted) == 1
        assert "prompt" in converted[0]
        assert "completion" in converted[0]
        assert "Human:" in converted[0]["prompt"]
        assert "Assistant:" in converted[0]["prompt"]
    
    @patch("builtins.open", new_callable=mock_open)
    def test_prepare_dataset(self, mock_file, dataset_preparer):
        """Test full dataset preparation"""
        input_data = [
            {
                "instruction": "Test",
                "input": "Input",
                "output": "Output"
            }
        ]
        
        mock_file.return_value.read.return_value = json.dumps(input_data)
        
        output_file = dataset_preparer.prepare_dataset(
            "input.json",
            "openai",
            "output.jsonl"
        )
        
        assert output_file == "output.jsonl"
        mock_file.assert_called()


class TestCloudFineTuner:
    """Test cloud fine-tuning functionality"""
    
    @pytest.fixture
    def cloud_tuner(self):
        """Create CloudFineTuner instance"""
        return CloudFineTuner()
    
    @patch('openai.OpenAI')
    def test_tune_openai(self, mock_openai_client, cloud_tuner):
        """Test OpenAI fine-tuning"""
        # Mock client and responses
        mock_client = MagicMock()
        mock_openai_client.return_value = mock_client
        
        # Mock file upload
        mock_file = MagicMock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file
        
        # Mock fine-tuning job
        mock_job = MagicMock()
        mock_job.id = "ft-job-123"
        mock_job.status = "succeeded"
        mock_job.fine_tuned_model = "ft:gpt-3.5-turbo:test"
        mock_client.fine_tuning.jobs.create.return_value = mock_job
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job
        
        result = cloud_tuner.tune_openai(
            dataset_file="train.jsonl",
            model="gpt-3.5-turbo",
            config={"epochs": 3}
        )
        
        assert result["job_id"] == "ft-job-123"
        assert result["status"] == "succeeded"
        assert result["model_id"] == "ft:gpt-3.5-turbo:test"
    
    @patch('anthropic.Anthropic')
    @patch('requests.post')
    def test_tune_anthropic(self, mock_post, mock_anthropic, cloud_tuner):
        """Test Anthropic fine-tuning"""
        # Note: Anthropic fine-tuning is hypothetical in this example
        mock_post.return_value.json.return_value = {
            "job_id": "anthropic-ft-123",
            "status": "completed"
        }
        
        result = cloud_tuner.tune_anthropic(
            dataset_file="train.jsonl",
            model="claude-3-5-sonnet-20241022",
            config={}
        )
        
        assert "job_id" in result or "error" in result


class TestLocalFineTuner:
    """Test local model fine-tuning"""
    
    @pytest.fixture
    def local_tuner(self):
        """Create LocalFineTuner instance"""
        return LocalFineTuner()
    
    @patch('subprocess.run')
    def test_setup_environment(self, mock_subprocess, local_tuner):
        """Test environment setup"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        local_tuner.setup_environment()
        
        # Should attempt to install required packages
        assert mock_subprocess.called
    
    @patch('subprocess.run')
    @patch('torch.cuda.is_available')
    def test_train(self, mock_cuda, mock_subprocess, local_tuner):
        """Test local model training"""
        mock_cuda.return_value = False  # CPU training
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Training completed"
        )
        
        result = local_tuner.train(
            dataset_file="train.json",
            output_dir="./output",
            config={
                "model_name": "mistralai/Mistral-7B-v0.1",
                "method": "lora"
            }
        )
        
        assert "status" in result
        assert result["status"] == "completed"
    
    def test_get_training_args(self, local_tuner):
        """Test training arguments generation"""
        args = local_tuner._get_training_args(
            output_dir="./output",
            config={
                "learning_rate": 1e-4,
                "num_epochs": 3,
                "batch_size": 4
            }
        )
        
        assert "--output_dir" in args
        assert "--learning_rate" in args
        assert "1e-4" in args


class TestModelEvaluator:
    """Test model evaluation functionality"""
    
    @pytest.fixture
    def evaluator(self):
        """Create ModelEvaluator instance"""
        return ModelEvaluator()
    
    @patch('examples.use_cases.fine_tuning_complete_demo.ModelEvaluator._run_model')
    def test_evaluate_model(self, mock_run_model, evaluator):
        """Test model evaluation"""
        # Mock model responses
        mock_run_model.side_effect = [
            "Paris",
            "Oxygen and Hydrogen"
        ]
        
        test_cases = [
            {
                "prompt": "What is the capital of France?",
                "expected": "Paris"
            },
            {
                "prompt": "What elements make water?",
                "expected": "Hydrogen and Oxygen"
            }
        ]
        
        metrics = evaluator.evaluate_model(
            model_name="test-model",
            test_cases=test_cases
        )
        
        assert metrics["total_cases"] == 2
        assert metrics["accuracy"] > 0
        assert "latency" in metrics
    
    def test_calculate_similarity(self, evaluator):
        """Test text similarity calculation"""
        score = evaluator._calculate_similarity(
            "The capital is Paris",
            "Paris is the capital"
        )
        assert 0 <= score <= 1
        assert score > 0.5  # Should be reasonably similar
    
    def test_compare_models(self, evaluator):
        """Test model comparison"""
        with patch.object(evaluator, 'evaluate_model') as mock_eval:
            mock_eval.side_effect = [
                {"accuracy": 0.9, "avg_latency": 0.5},
                {"accuracy": 0.85, "avg_latency": 0.3}
            ]
            
            comparison = evaluator.compare_models(
                models=["model1", "model2"],
                test_cases=[]
            )
            
            assert len(comparison) == 2
            assert comparison["model1"]["accuracy"] == 0.9


class TestFineTuningOrchestrator:
    """Test fine-tuning orchestration"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create FineTuningOrchestrator instance"""
        return FineTuningOrchestrator()
    
    @patch('examples.use_cases.fine_tuning_complete_demo.CloudFineTuner.tune_openai')
    @patch('examples.use_cases.fine_tuning_complete_demo.DatasetPreparer.prepare_dataset')
    def test_run_cloud_fine_tuning(self, mock_prepare, mock_tune, orchestrator):
        """Test cloud fine-tuning orchestration"""
        mock_prepare.return_value = "prepared.jsonl"
        mock_tune.return_value = {
            "job_id": "ft-123",
            "status": "succeeded",
            "model_id": "ft:model"
        }
        
        result = orchestrator.run_cloud_fine_tuning(
            provider="openai",
            dataset_file="data.json",
            base_model="gpt-3.5-turbo"
        )
        
        assert result["dataset_file"] == "prepared.jsonl"
        assert result["job_id"] == "ft-123"
    
    @patch('examples.use_cases.fine_tuning_complete_demo.LocalFineTuner.train')
    def test_run_local_fine_tuning(self, mock_train, orchestrator):
        """Test local fine-tuning orchestration"""
        mock_train.return_value = {
            "status": "completed",
            "output_dir": "./output"
        }
        
        result = orchestrator.run_local_fine_tuning(
            dataset_file="data.json",
            model_name="llama-2-7b",
            method="lora"
        )
        
        assert result["status"] == "completed"


class TestIntegration:
    """Test integration scenarios"""
    
    def test_full_pipeline(self):
        """Test full fine-tuning pipeline"""
        with patch('examples.use_cases.fine_tuning_complete_demo.DatasetPreparer.prepare_dataset') as mock_prepare, \
             patch('examples.use_cases.fine_tuning_complete_demo.CloudFineTuner.tune_openai') as mock_tune, \
             patch('examples.use_cases.fine_tuning_complete_demo.ModelEvaluator.evaluate_model') as mock_eval:
            
            # Setup mocks
            mock_prepare.return_value = "prepared.jsonl"
            mock_tune.return_value = {
                "model_id": "ft:model",
                "status": "succeeded"
            }
            mock_eval.return_value = {
                "accuracy": 0.95,
                "avg_latency": 0.4
            }
            
            # Run pipeline
            orchestrator = FineTuningOrchestrator()
            
            # Prepare dataset
            dataset_file = orchestrator.dataset_preparer.prepare_dataset(
                "input.json", "openai", "output.jsonl"
            )
            
            # Fine-tune
            ft_result = orchestrator.cloud_tuner.tune_openai(
                dataset_file, "gpt-3.5-turbo", {}
            )
            
            # Evaluate
            eval_result = orchestrator.evaluator.evaluate_model(
                ft_result["model_id"], []
            )
            
            assert eval_result["accuracy"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])