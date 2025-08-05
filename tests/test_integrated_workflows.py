"""
Comprehensive tests for integrated workflow functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.use_cases.integrated_workflow_demo import (
    IntegratedBenchmarkMonitor,
    FineTuningSafetyValidator,
    LocalCloudHybridMonitor,
    ProductionPipeline
)


class TestIntegratedBenchmarkMonitor:
    """Test integrated benchmark and monitoring workflow"""
    
    @pytest.fixture
    def benchmark_monitor(self):
        """Create IntegratedBenchmarkMonitor instance"""
        config = {
            'models': ['gpt-4', 'claude-3-5-sonnet-20241022'],
            'benchmark_schedule': 'daily',
            'cost_threshold': 100.0,
            'alert_config': {
                'email': 'team@example.com'
            }
        }
        return IntegratedBenchmarkMonitor(config)
    
    @patch('examples.use_cases.integrated_workflow_demo.IntegratedBenchmarkMonitor._run_llm')
    def test_run_benchmarks(self, mock_llm, benchmark_monitor):
        """Test running benchmarks"""
        mock_llm.return_value = "Paris"
        
        results = benchmark_monitor.run_benchmarks()
        
        assert len(results) == 2  # Two models
        assert all('latency' in r for r in results)
        assert all('accuracy' in r for r in results)
        assert all('cost' in r for r in results)
    
    def test_monitor_costs(self, benchmark_monitor):
        """Test cost monitoring"""
        benchmark_results = [
            {
                'model': 'gpt-4',
                'cost': 0.05,
                'timestamp': datetime.now().isoformat()
            },
            {
                'model': 'claude-3-5-sonnet-20241022',
                'cost': 0.03,
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        benchmark_monitor.benchmark_results = benchmark_results
        cost_analysis = benchmark_monitor.monitor_costs()
        
        assert cost_analysis['total_cost'] == 0.08
        assert len(cost_analysis['cost_by_model']) == 2
        assert not cost_analysis['threshold_exceeded']
    
    @patch('examples.use_cases.integrated_workflow_demo.IntegratedBenchmarkMonitor._send_alert')
    def test_check_performance_regression(self, mock_alert, benchmark_monitor):
        """Test performance regression detection"""
        current_results = [
            {'model': 'gpt-4', 'latency': 2.5, 'accuracy': 0.85}
        ]
        historical_results = [
            {'model': 'gpt-4', 'latency': 1.5, 'accuracy': 0.90}
        ]
        
        benchmark_monitor.benchmark_results = current_results
        benchmark_monitor.historical_results = historical_results
        
        regressions = benchmark_monitor.check_performance_regression()
        
        assert len(regressions) == 1
        assert regressions[0]['model'] == 'gpt-4'
        assert regressions[0]['latency_regression']
        assert regressions[0]['accuracy_regression']
        
        # Alert should be sent
        mock_alert.assert_called_once()
    
    def test_generate_report(self, benchmark_monitor):
        """Test report generation"""
        benchmark_monitor.benchmark_results = [
            {
                'model': 'gpt-4',
                'latency': 1.5,
                'accuracy': 0.92,
                'cost': 0.05
            }
        ]
        
        report = benchmark_monitor.generate_report()
        
        assert 'summary' in report
        assert 'benchmark_results' in report
        assert 'cost_analysis' in report
        assert 'recommendations' in report


class TestFineTuningSafetyValidator:
    """Test fine-tuning with safety validation workflow"""
    
    @pytest.fixture
    def safety_validator(self):
        """Create FineTuningSafetyValidator instance"""
        safety_config = {
            'toxicity_threshold': 0.7,
            'bias_threshold': 0.8,
            'consensus_models': ['claude-3-5-sonnet-20241022', 'gpt-4']
        }
        return FineTuningSafetyValidator(safety_config)
    
    def test_validate_dataset(self, safety_validator):
        """Test dataset validation"""
        dataset = [
            {
                'instruction': 'Translate to French',
                'input': 'Hello',
                'output': 'Bonjour'
            },
            {
                'instruction': 'Bad instruction',
                'input': 'harmful content',
                'output': 'harmful response'
            }
        ]
        
        with patch.object(safety_validator, '_check_safety') as mock_safety:
            mock_safety.side_effect = [True, False]
            
            clean_dataset = safety_validator.validate_dataset(dataset)
            
            assert len(clean_dataset) == 1
            assert clean_dataset[0]['output'] == 'Bonjour'
    
    @patch('examples.use_cases.integrated_workflow_demo.FineTuningSafetyValidator._run_fine_tuning')
    def test_train_with_safety(self, mock_train, safety_validator):
        """Test training with safety checks"""
        mock_train.return_value = {
            'model_id': 'ft:model-123',
            'status': 'completed'
        }
        
        dataset = [{'instruction': 'test', 'input': 'test', 'output': 'test'}]
        
        result = safety_validator.train_with_safety(
            dataset, 
            'gpt-3.5-turbo',
            {'epochs': 3}
        )
        
        assert result['status'] == 'completed'
        assert result['safety_validated']
    
    @patch('examples.use_cases.integrated_workflow_demo.FineTuningSafetyValidator._run_model')
    def test_evaluate_safety_post_training(self, mock_model, safety_validator):
        """Test post-training safety evaluation"""
        mock_model.return_value = "Safe response"
        
        test_prompts = [
            "What is the capital of France?",
            "Tell me about AI safety"
        ]
        
        with patch.object(safety_validator, '_check_response_safety') as mock_check:
            mock_check.return_value = (True, 0.1)  # Safe, low toxicity
            
            results = safety_validator.evaluate_safety_post_training(
                'ft:model-123',
                test_prompts
            )
            
            assert results['passed']
            assert results['avg_safety_score'] > 0.8
    
    def test_apply_constitutional_ai(self, safety_validator):
        """Test constitutional AI application"""
        responses = [
            "This is a helpful response",
            "This might be slightly problematic"
        ]
        
        with patch.object(safety_validator, '_apply_constitutional_rules') as mock_rules:
            mock_rules.side_effect = [
                "This is a helpful response",
                "This is a corrected, safe response"
            ]
            
            filtered = safety_validator.apply_constitutional_ai(responses)
            
            assert len(filtered) == 2
            assert filtered[1] == "This is a corrected, safe response"


class TestLocalCloudHybridMonitor:
    """Test local-cloud hybrid monitoring workflow"""
    
    @pytest.fixture
    def hybrid_monitor(self):
        """Create LocalCloudHybridMonitor instance"""
        config = {
            'local_models': ['llama-2-7b', 'mistral-7b'],
            'cloud_models': ['gpt-4', 'claude-3-5-sonnet-20241022'],
            'sync_interval': 3600  # 1 hour
        }
        return LocalCloudHybridMonitor(config)
    
    @patch('examples.use_cases.integrated_workflow_demo.LocalCloudHybridMonitor._get_local_metrics')
    @patch('examples.use_cases.integrated_workflow_demo.LocalCloudHybridMonitor._get_cloud_metrics')
    def test_collect_all_metrics(self, mock_cloud, mock_local, hybrid_monitor):
        """Test collecting metrics from all sources"""
        mock_local.return_value = [
            {'model': 'llama-2-7b', 'latency': 0.8, 'throughput': 50}
        ]
        mock_cloud.return_value = [
            {'model': 'gpt-4', 'latency': 1.5, 'cost': 0.05}
        ]
        
        all_metrics = hybrid_monitor.collect_all_metrics()
        
        assert len(all_metrics['local']) == 1
        assert len(all_metrics['cloud']) == 1
        assert all_metrics['local'][0]['model'] == 'llama-2-7b'
        assert all_metrics['cloud'][0]['model'] == 'gpt-4'
    
    def test_compare_performance(self, hybrid_monitor):
        """Test performance comparison between local and cloud"""
        metrics = {
            'local': [
                {'model': 'llama-2-7b', 'latency': 0.8, 'accuracy': 0.85}
            ],
            'cloud': [
                {'model': 'gpt-4', 'latency': 1.5, 'accuracy': 0.92}
            ]
        }
        
        comparison = hybrid_monitor.compare_performance(metrics)
        
        assert comparison['latency_advantage'] == 'local'
        assert comparison['accuracy_advantage'] == 'cloud'
        assert 'recommendations' in comparison
    
    def test_optimize_routing(self, hybrid_monitor):
        """Test request routing optimization"""
        current_load = {
            'local': {'cpu': 60, 'memory': 70},
            'cloud': {'requests_per_minute': 50}
        }
        
        routing_rules = hybrid_monitor.optimize_routing(current_load)
        
        assert 'rules' in routing_rules
        assert any(r['destination'] == 'local' for r in routing_rules['rules'])
        assert any(r['destination'] == 'cloud' for r in routing_rules['rules'])
    
    @patch('grafana_api.GrafanaFace')
    def test_sync_to_grafana(self, mock_grafana, hybrid_monitor):
        """Test Grafana synchronization"""
        mock_client = MagicMock()
        mock_grafana.return_value = mock_client
        
        metrics = {
            'local': [{'model': 'llama-2-7b', 'latency': 0.8}],
            'cloud': [{'model': 'gpt-4', 'latency': 1.5}]
        }
        
        result = hybrid_monitor.sync_to_grafana(metrics)
        
        assert result['success']
        mock_client.datasource.create_datasource.assert_called()


class TestProductionPipeline:
    """Test complete production pipeline"""
    
    @pytest.fixture
    def production_pipeline(self):
        """Create ProductionPipeline instance"""
        config = {
            'models': {
                'primary': 'gpt-4',
                'fallback': 'gpt-3.5-turbo',
                'safety': 'claude-3-5-sonnet-20241022'
            },
            'thresholds': {
                'latency': 2.0,
                'error_rate': 0.05,
                'safety_score': 0.9
            }
        }
        return ProductionPipeline(config)
    
    @patch('examples.use_cases.integrated_workflow_demo.ProductionPipeline._call_model')
    def test_process_request(self, mock_call, production_pipeline):
        """Test request processing with safety checks"""
        mock_call.return_value = ("This is a safe response", 1.5, None)
        
        with patch.object(production_pipeline, '_check_safety') as mock_safety:
            mock_safety.return_value = (True, 0.95)
            
            result = production_pipeline.process_request(
                "What is AI?",
                user_id="user123"
            )
            
            assert result['success']
            assert result['response'] == "This is a safe response"
            assert result['safety_score'] == 0.95
    
    def test_handle_failure(self, production_pipeline):
        """Test failure handling with fallback"""
        with patch.object(production_pipeline, '_call_model') as mock_call:
            # Primary fails, fallback succeeds
            mock_call.side_effect = [
                (None, 0, "Error"),
                ("Fallback response", 1.0, None)
            ]
            
            result = production_pipeline.process_request("Test prompt")
            
            assert result['success']
            assert result['model_used'] == 'gpt-3.5-turbo'
            assert result['fallback_used']
    
    def test_monitor_health(self, production_pipeline):
        """Test health monitoring"""
        # Simulate request history
        production_pipeline.request_history = [
            {'latency': 1.5, 'error': None, 'timestamp': datetime.now()},
            {'latency': 1.8, 'error': None, 'timestamp': datetime.now()},
            {'latency': 0, 'error': 'Timeout', 'timestamp': datetime.now()},
        ]
        
        health = production_pipeline.monitor_health()
        
        assert health['avg_latency'] == pytest.approx(1.65, 0.01)
        assert health['error_rate'] == pytest.approx(0.33, 0.01)
        assert 'status' in health
    
    @patch('examples.use_cases.integrated_workflow_demo.ProductionPipeline._trigger_alert')
    def test_auto_scale(self, mock_alert, production_pipeline):
        """Test auto-scaling triggers"""
        # Simulate high load
        production_pipeline.request_history = [
            {'latency': 3.0, 'error': None, 'timestamp': datetime.now()}
            for _ in range(100)
        ]
        
        scaling_decision = production_pipeline.auto_scale()
        
        assert scaling_decision['scale_up']
        assert scaling_decision['reason'] == 'high_latency'
        mock_alert.assert_called_once()


class TestIntegration:
    """Test integration between workflows"""
    
    def test_benchmark_to_monitoring_flow(self):
        """Test flow from benchmarking to monitoring"""
        # Create integrated system
        benchmark_config = {
            'models': ['gpt-4'],
            'benchmark_schedule': 'hourly',
            'cost_threshold': 10.0,
            'alert_config': {}
        }
        
        monitor = IntegratedBenchmarkMonitor(benchmark_config)
        
        # Run benchmark
        with patch.object(monitor, '_run_llm') as mock_llm:
            mock_llm.return_value = "Test response"
            
            results = monitor.run_benchmarks()
            monitor.track_over_time(results)
            
            # Generate report
            report = monitor.generate_report()
            
            assert 'benchmark_results' in report
            assert 'trends' in report
    
    def test_safety_validated_production_flow(self):
        """Test flow from safety validation to production"""
        # Validate dataset with safety
        validator = FineTuningSafetyValidator({})
        dataset = [{'instruction': 'test', 'input': 'test', 'output': 'test'}]
        
        with patch.object(validator, '_check_safety') as mock_safety:
            mock_safety.return_value = True
            clean_dataset = validator.validate_dataset(dataset)
        
        # Deploy to production with monitoring
        prod_config = {
            'models': {'primary': 'ft:model-123', 'fallback': 'gpt-3.5-turbo'},
            'thresholds': {'safety_score': 0.9}
        }
        pipeline = ProductionPipeline(prod_config)
        
        with patch.object(pipeline, '_call_model') as mock_call:
            mock_call.return_value = ("Safe response", 1.0, None)
            
            result = pipeline.process_request("Test prompt")
            
            assert result['success']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])