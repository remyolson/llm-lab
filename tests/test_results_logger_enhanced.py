"""
Tests for enhanced results logger with multi-model support
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open
import json

from src.logging.results_logger import CSVResultLogger, ResultRecord
import tempfile
import shutil
import os


class TestEnhancedFileNaming:
    """Test the enhanced file naming convention."""
    
    def test_generate_filename_with_model(self):
        """Test filename generation with model name included."""
        logger = CSVResultLogger()
        
        with patch('results_logger.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '20240201_120000'
            
            # Test with model name
            filename = logger.generate_filename('openai', 'truthfulness', 'gpt-4')
            assert filename == 'benchmark_openai_gpt-4_truthfulness_20240201_120000.csv'
            
            # Test with model name containing special characters
            filename = logger.generate_filename('anthropic', 'truthfulness', 'claude-3/opus:2024')
            assert filename == 'benchmark_anthropic_claude-3-opus-2024_truthfulness_20240201_120000.csv'
    
    def test_generate_filename_without_model(self):
        """Test backward compatibility when model name is not provided."""
        logger = CSVResultLogger()
        
        with patch('results_logger.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '20240201_120000'
            
            # Test without model name (backward compatibility)
            filename = logger.generate_filename('openai', 'truthfulness')
            assert filename == 'benchmark_openai_truthfulness_20240201_120000.csv'
    
    def test_generate_filename_special_characters(self):
        """Test that special characters in model names are properly handled."""
        logger = CSVResultLogger()
        
        with patch('results_logger.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '20240201_120000'
            
            # Test various special characters
            test_cases = [
                ('model/with/slashes', 'model-with-slashes'),
                ('model\\with\\backslashes', 'model-with-backslashes'),
                ('model:with:colons', 'model-with-colons'),
                ('model/mixed\\special:chars', 'model-mixed-special-chars')
            ]
            
            for input_model, expected_safe in test_cases:
                filename = logger.generate_filename('provider', 'dataset', input_model)
                assert f'provider_{expected_safe}_dataset_' in filename
    
    def test_write_results_uses_model_name(self):
        """Test that write_results uses model name in filename generation."""
        logger = CSVResultLogger()
        
        # Mock results with model name
        results = {
            'provider': 'openai',
            'model': 'gpt-4-turbo',
            'dataset': 'truthfulness',
            'evaluations': [
                {
                    'timestamp': '2024-02-01T12:00:00',
                    'model_name': 'gpt-4-turbo',
                    'benchmark_name': 'truthfulness',
                    'prompt_id': 'test_1',
                    'prompt': 'Test prompt',
                    'response': 'Test response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        with patch('results_logger.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '20240201_120000'
            
            with patch('builtins.open', mock_open()) as mock_file:
                filepath = logger.write_results(results)
                
                # Check that the filepath includes the model name
                assert 'gpt-4-turbo' in filepath
                assert filepath.endswith('benchmark_openai_gpt-4-turbo_truthfulness_20240201_120000.csv')
    
    def test_write_results_without_model_backward_compatibility(self):
        """Test backward compatibility when model is not in results."""
        logger = CSVResultLogger()
        
        # Mock results without model name (backward compatibility)
        results = {
            'provider': 'openai',
            'dataset': 'truthfulness',
            'evaluations': [
                {
                    'timestamp': '2024-02-01T12:00:00',
                    'model_name': 'gpt-4',  # Only in evaluation data
                    'benchmark_name': 'truthfulness',
                    'prompt_id': 'test_1',
                    'prompt': 'Test prompt',
                    'response': 'Test response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        with patch('results_logger.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '20240201_120000'
            
            with patch('builtins.open', mock_open()) as mock_file:
                filepath = logger.write_results(results)
                
                # Check that the filepath uses old format without model
                assert 'gpt-4' not in filepath
                assert filepath.endswith('benchmark_openai_truthfulness_20240201_120000.csv')


class TestResultRecordEnhancements:
    """Test ResultRecord for any needed enhancements."""
    
    def test_result_record_with_provider_info(self):
        """Test that ResultRecord properly handles provider information."""
        eval_data = {
            'timestamp': '2024-02-01T12:00:00',
            'model_name': 'gpt-4',
            'provider': 'openai',  # Additional provider field
            'benchmark_name': 'truthfulness',
            'prompt_id': 'test_1',
            'prompt': 'Test prompt',
            'response': 'Test response',
            'score': 0.95,
            'success': True
        }
        
        record = ResultRecord(eval_data)
        csv_dict = record.to_csv_dict()
        
        # Verify all standard fields are present
        assert csv_dict['model_name'] == 'gpt-4'
        assert csv_dict['score'] == '0.9500'
        assert csv_dict['success'] == 'pass'


class TestDirectoryStructure:
    """Test the organized directory structure functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_get_organized_path_creates_structure(self, temp_dir):
        """Test that get_organized_path creates the correct directory structure."""
        logger = CSVResultLogger(temp_dir)
        
        # Test with specific date
        test_date = datetime(2024, 2, 15, 12, 0, 0)
        path = logger.get_organized_path('truthfulness', test_date)
        
        # Check path structure
        assert path == Path(temp_dir) / 'truthfulness' / '2024-02'
        assert path.exists()
        assert path.is_dir()
    
    def test_get_organized_path_current_date(self, temp_dir):
        """Test get_organized_path with current date."""
        logger = CSVResultLogger(temp_dir)
        
        with patch('results_logger.datetime') as mock_datetime:
            mock_now = datetime(2024, 3, 20, 15, 30, 0)
            mock_datetime.now.return_value = mock_now
            mock_datetime.strftime = datetime.strftime
            
            path = logger.get_organized_path('benchmark_test')
            
            assert path == Path(temp_dir) / 'benchmark_test' / '2024-03'
            assert path.exists()
    
    def test_write_results_with_organized_dirs(self, temp_dir):
        """Test that write_results uses organized directory structure."""
        logger = CSVResultLogger(temp_dir)
        
        results = {
            'provider': 'openai',
            'model': 'gpt-4',
            'dataset': 'truthfulness',
            'start_time': '2024-02-15T10:30:00',
            'evaluations': [
                {
                    'timestamp': '2024-02-15T10:30:00',
                    'model_name': 'gpt-4',
                    'benchmark_name': 'truthfulness',
                    'prompt_id': 'test_1',
                    'prompt': 'Test prompt',
                    'response': 'Test response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        with patch('results_logger.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '20240215_103000'
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            filepath = logger.write_results(results)
            
            # Check that file is in organized directory
            assert 'truthfulness/2024-02' in filepath
            assert Path(filepath).exists()
            assert Path(filepath).parent.name == '2024-02'
            assert Path(filepath).parent.parent.name == 'truthfulness'
    
    def test_write_results_without_organized_dirs(self, temp_dir):
        """Test that write_results can use flat structure when requested."""
        logger = CSVResultLogger(temp_dir)
        
        results = {
            'provider': 'openai',
            'model': 'gpt-4',
            'dataset': 'truthfulness',
            'evaluations': [
                {
                    'timestamp': '2024-02-15T10:30:00',
                    'model_name': 'gpt-4',
                    'benchmark_name': 'truthfulness',
                    'prompt_id': 'test_1',
                    'prompt': 'Test prompt',
                    'response': 'Test response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        with patch('results_logger.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '20240215_103000'
            
            # Use flat structure
            filepath = logger.write_results(results, use_organized_dirs=False)
            
            # Check that file is in root output directory
            assert Path(filepath).parent == Path(temp_dir)
            assert 'truthfulness/2024-02' not in filepath
    
    def test_multiple_datasets_organization(self, temp_dir):
        """Test organization with multiple datasets."""
        logger = CSVResultLogger(temp_dir)
        
        # Create results for different datasets
        datasets = ['truthfulness', 'reasoning', 'coding']
        test_date = datetime(2024, 2, 15)
        
        for dataset in datasets:
            path = logger.get_organized_path(dataset, test_date)
            assert path.exists()
            assert path == Path(temp_dir) / dataset / '2024-02'
        
        # Check overall structure
        root_contents = list(Path(temp_dir).iterdir())
        assert len(root_contents) == 3
        assert all(d.name in datasets for d in root_contents)
    
    def test_append_result_creates_directories(self, temp_dir):
        """Test that append_result creates necessary directories."""
        logger = CSVResultLogger(temp_dir)
        
        # Create a path that doesn't exist yet
        deep_path = Path(temp_dir) / 'deep' / 'nested' / 'path' / 'results.csv'
        
        eval_data = {
            'timestamp': '2024-02-15T10:30:00',
            'model_name': 'gpt-4',
            'benchmark_name': 'test',
            'prompt_id': 'test_1',
            'prompt': 'Test',
            'response': 'Response',
            'score': 1.0,
            'success': True
        }
        
        # This should create all parent directories
        logger.append_result(deep_path, eval_data)
        
        assert deep_path.exists()
        assert deep_path.parent.exists()


class TestAtomicFileWriting:
    """Test atomic file writing functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_atomic_write_creates_file(self, temp_dir):
        """Test that atomic write creates the file correctly."""
        logger = CSVResultLogger(temp_dir)
        
        results = {
            'provider': 'openai',
            'model': 'gpt-4',
            'dataset': 'test',
            'evaluations': [
                {
                    'timestamp': '2024-02-15T10:30:00',
                    'model_name': 'gpt-4',
                    'benchmark_name': 'test',
                    'prompt_id': 'test_1',
                    'prompt': 'Test prompt',
                    'response': 'Test response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        filepath = logger.write_results(results, use_organized_dirs=False)
        
        # Check file exists and has content
        assert Path(filepath).exists()
        with open(filepath, 'r') as f:
            content = f.read()
            assert 'model_name' in content  # Header
            assert 'gpt-4' in content  # Data
    
    def test_atomic_write_no_partial_files(self, temp_dir):
        """Test that atomic write doesn't leave partial files on error."""
        logger = CSVResultLogger(temp_dir)
        
        # Create results that will cause an error during writing
        results = {
            'provider': 'openai',
            'model': 'gpt-4',
            'dataset': 'test',
            'evaluations': []  # Empty evaluations will cause ValueError
        }
        
        with pytest.raises(ValueError, match="No evaluation results to write"):
            logger.write_results(results, use_organized_dirs=False)
        
        # Check that no partial files exist
        files = list(Path(temp_dir).glob('*'))
        # Should only have subdirectories, no partial CSV files
        csv_files = [f for f in files if f.suffix == '.csv']
        assert len(csv_files) == 0
        
        # Also check for temp files
        temp_files = [f for f in files if '.tmp' in f.name]
        assert len(temp_files) == 0
    
    def test_atomic_append_to_existing_file(self, temp_dir):
        """Test atomic append to an existing file."""
        logger = CSVResultLogger(temp_dir)
        
        # Create initial file
        filepath = Path(temp_dir) / 'test_results.csv'
        initial_data = {
            'timestamp': '2024-02-15T10:30:00',
            'model_name': 'gpt-4',
            'benchmark_name': 'test',
            'prompt_id': 'test_1',
            'prompt': 'First prompt',
            'response': 'First response',
            'score': 1.0,
            'success': True
        }
        
        logger.append_result(filepath, initial_data)
        
        # Read initial content
        with open(filepath, 'r') as f:
            initial_lines = f.readlines()
        
        # Append more data
        append_data = {
            'timestamp': '2024-02-15T10:31:00',
            'model_name': 'gpt-4',
            'benchmark_name': 'test',
            'prompt_id': 'test_2',
            'prompt': 'Second prompt',
            'response': 'Second response',
            'score': 0.9,
            'success': True
        }
        
        logger.append_result(filepath, append_data)
        
        # Verify both records exist
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Should have header + 2 data rows
        assert len(lines) == len(initial_lines) + 1
        assert 'First prompt' in ''.join(lines)
        assert 'Second prompt' in ''.join(lines)
    
    def test_concurrent_atomic_writes(self, temp_dir):
        """Test that atomic writes handle concurrent access safely."""
        import threading
        import time
        
        logger = CSVResultLogger(temp_dir)
        errors = []
        completed = []
        
        def write_results(thread_id):
            try:
                results = {
                    'provider': f'provider_{thread_id}',
                    'model': f'model_{thread_id}',
                    'dataset': 'concurrent_test',
                    'evaluations': [
                        {
                            'timestamp': datetime.now().isoformat(),
                            'model_name': f'model_{thread_id}',
                            'benchmark_name': 'concurrent_test',
                            'prompt_id': f'test_{thread_id}',
                            'prompt': f'Prompt from thread {thread_id}',
                            'response': f'Response from thread {thread_id}',
                            'score': 0.5 + (thread_id * 0.1),
                            'success': True
                        }
                    ]
                }
                
                # Small random delay to increase chance of collision
                time.sleep(0.001 * thread_id)
                
                filepath = logger.write_results(results)
                completed.append((thread_id, filepath))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Launch multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=write_results, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # All should complete without errors
        assert len(errors) == 0
        assert len(completed) == 5
        
        # All files should exist and be valid
        for thread_id, filepath in completed:
            assert Path(filepath).exists()
            with open(filepath, 'r') as f:
                content = f.read()
                assert f'model_{thread_id}' in content
    
    def test_atomic_write_preserves_permissions(self, temp_dir):
        """Test that atomic write preserves file permissions."""
        logger = CSVResultLogger(temp_dir)
        
        results = {
            'provider': 'openai',
            'model': 'gpt-4',
            'dataset': 'test',
            'evaluations': [
                {
                    'timestamp': '2024-02-15T10:30:00',
                    'model_name': 'gpt-4',
                    'benchmark_name': 'test',
                    'prompt_id': 'test_1',
                    'prompt': 'Test',
                    'response': 'Response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        filepath = logger.write_results(results, use_organized_dirs=False)
        
        # Check that file has reasonable permissions (readable)
        stat_info = os.stat(filepath)
        # Check owner can read
        assert stat_info.st_mode & 0o400


class TestMetadataAndIndexing:
    """Test metadata tracking and index generation functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_metadata_file_creation(self, temp_dir):
        """Test that metadata file is created alongside CSV."""
        logger = CSVResultLogger(temp_dir)
        
        results = {
            'provider': 'openai',
            'model': 'gpt-4',
            'dataset': 'test',
            'start_time': '2024-02-15T10:30:00',
            'end_time': '2024-02-15T10:35:00',
            'total_duration_seconds': 300,
            'total_prompts': 100,
            'successful_evaluations': 95,
            'failed_evaluations': 5,
            'overall_score': 0.95,
            'average_response_time_seconds': 3.0,
            'model_config': {'temperature': 0.7, 'max_tokens': 1000},
            'evaluations': [
                {
                    'timestamp': '2024-02-15T10:30:00',
                    'model_name': 'gpt-4',
                    'benchmark_name': 'test',
                    'prompt_id': 'test_1',
                    'prompt': 'Test prompt',
                    'response': 'Test response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        csv_path = logger.write_results(results, use_organized_dirs=False)
        metadata_path = Path(csv_path).with_suffix('.meta.json')
        
        # Check metadata file exists
        assert metadata_path.exists()
        
        # Load and verify metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['provider'] == 'openai'
        assert metadata['model'] == 'gpt-4'
        assert metadata['dataset'] == 'test'
        assert metadata['overall_score'] == 0.95
        assert metadata['total_prompts'] == 100
        assert metadata['model_config']['temperature'] == 0.7
        assert 'created_at' in metadata
    
    def test_index_file_creation(self, temp_dir):
        """Test that index file is created and updated."""
        logger = CSVResultLogger(temp_dir)
        
        # First result
        results1 = {
            'provider': 'openai',
            'model': 'gpt-4',
            'dataset': 'test1',
            'start_time': '2024-02-15T10:00:00',
            'overall_score': 0.90,
            'total_prompts': 50,
            'total_duration_seconds': 150,
            'evaluations': [
                {
                    'timestamp': '2024-02-15T10:00:00',
                    'model_name': 'gpt-4',
                    'benchmark_name': 'test1',
                    'prompt_id': 'test_1',
                    'prompt': 'Test',
                    'response': 'Response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        csv_path1 = logger.write_results(results1, use_organized_dirs=False)
        
        # Check index file exists
        index_path = Path(temp_dir) / 'benchmark_index.json'
        assert index_path.exists()
        
        # Load and verify index
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        assert index_data['version'] == '1.0'
        assert len(index_data['entries']) == 1
        assert index_data['entries'][0]['provider'] == 'openai'
        assert index_data['entries'][0]['model'] == 'gpt-4'
        assert index_data['entries'][0]['dataset'] == 'test1'
        
        # Second result
        results2 = {
            'provider': 'anthropic',
            'model': 'claude-3',
            'dataset': 'test2',
            'start_time': '2024-02-15T11:00:00',
            'overall_score': 0.85,
            'total_prompts': 60,
            'total_duration_seconds': 180,
            'evaluations': [
                {
                    'timestamp': '2024-02-15T11:00:00',
                    'model_name': 'claude-3',
                    'benchmark_name': 'test2',
                    'prompt_id': 'test_1',
                    'prompt': 'Test',
                    'response': 'Response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        csv_path2 = logger.write_results(results2, use_organized_dirs=False)
        
        # Reload index
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Should have 2 entries now
        assert len(index_data['entries']) == 2
        # Newest entry should be first (sorted by timestamp)
        assert index_data['entries'][0]['model'] == 'claude-3'
        assert index_data['entries'][1]['model'] == 'gpt-4'
    
    def test_index_with_organized_directories(self, temp_dir):
        """Test that index correctly handles organized directory structure."""
        logger = CSVResultLogger(temp_dir)
        
        results = {
            'provider': 'openai',
            'model': 'gpt-4',
            'dataset': 'truthfulness',
            'start_time': '2024-02-15T10:00:00',
            'overall_score': 0.90,
            'evaluations': [
                {
                    'timestamp': '2024-02-15T10:00:00',
                    'model_name': 'gpt-4',
                    'benchmark_name': 'truthfulness',
                    'prompt_id': 'test_1',
                    'prompt': 'Test',
                    'response': 'Response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        # Use organized directories
        csv_path = logger.write_results(results, use_organized_dirs=True)
        
        # Load index
        index_path = Path(temp_dir) / 'benchmark_index.json'
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Check that paths are relative and include subdirectories
        entry = index_data['entries'][0]
        assert 'truthfulness/2024-02' in entry['csv_file']
        assert 'truthfulness/2024-02' in entry['metadata_file']
    
    def test_metadata_error_handling(self, temp_dir):
        """Test that metadata errors don't fail main operation."""
        logger = CSVResultLogger(temp_dir)
        
        results = {
            'provider': 'openai',
            'model': 'gpt-4',
            'dataset': 'test',
            'evaluations': [
                {
                    'timestamp': '2024-02-15T10:00:00',
                    'model_name': 'gpt-4',
                    'benchmark_name': 'test',
                    'prompt_id': 'test_1',
                    'prompt': 'Test',
                    'response': 'Response',
                    'score': 1.0,
                    'success': True
                }
            ]
        }
        
        # Mock _write_metadata to raise an error
        original_write_metadata = logger._write_metadata
        
        def failing_write_metadata(*args, **kwargs):
            raise Exception("Simulated metadata error")
        
        logger._write_metadata = failing_write_metadata
        
        # Should still succeed
        csv_path = logger.write_results(results, use_organized_dirs=False)
        assert Path(csv_path).exists()
        
        # Restore original method
        logger._write_metadata = original_write_metadata
    
    def test_index_sorting(self, temp_dir):
        """Test that index entries are sorted by timestamp."""
        logger = CSVResultLogger(temp_dir)
        
        # Create results with different timestamps
        timestamps = [
            '2024-02-15T09:00:00',
            '2024-02-15T11:00:00',
            '2024-02-15T10:00:00'
        ]
        
        for i, ts in enumerate(timestamps):
            results = {
                'provider': 'openai',
                'model': f'model-{i}',
                'dataset': 'test',
                'start_time': ts,
                'evaluations': [
                    {
                        'timestamp': ts,
                        'model_name': f'model-{i}',
                        'benchmark_name': 'test',
                        'prompt_id': 'test_1',
                        'prompt': 'Test',
                        'response': 'Response',
                        'score': 1.0,
                        'success': True
                    }
                ]
            }
            logger.write_results(results, use_organized_dirs=False)
        
        # Load index
        index_path = Path(temp_dir) / 'benchmark_index.json'
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Check sorting (newest first)
        assert index_data['entries'][0]['model'] == 'model-1'  # 11:00
        assert index_data['entries'][1]['model'] == 'model-2'  # 10:00
        assert index_data['entries'][2]['model'] == 'model-0'  # 09:00