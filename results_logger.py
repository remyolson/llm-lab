"""
Results Logger Module for LLM Lab

This module provides comprehensive logging functionality for benchmark results,
including CSV file generation with proper formatting and error handling.

Key features:
- CSV output with configurable columns and formatting
- Automatic directory creation for results
- Text truncation and cleaning for CSV compatibility
- Timestamp formatting in ISO 8601 standard
- Type-safe result records using the ResultRecord class
- Context manager support for safe file operations
- Append mode for incremental result logging

The module ensures all benchmark results are persistently stored in a format
that's easy to analyze with standard data analysis tools.
"""

import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if truncated.
    
    Args:
        text: Text to potentially truncate
        max_length: Maximum allowed length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - 3] + '...'


def clean_multiline_text(text: str) -> str:
    """
    Clean multiline text for CSV storage.
    
    Replaces newlines with spaces and removes extra whitespace.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned single-line text
    """
    if not text:
        return ''

    # Replace various newline types with space
    text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')

    # Collapse multiple spaces
    text = ' '.join(text.split())

    return text


def format_timestamp(dt: Optional[Union[str, datetime]] = None) -> str:
    """
    Format a timestamp in ISO 8601 format.
    
    Args:
        dt: Datetime object or ISO string, or None for current time
        
    Returns:
        ISO 8601 formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    elif isinstance(dt, str):
        # Already formatted, return as-is
        return dt

    return dt.isoformat()


# Define CSV column schema
CSV_COLUMNS = [
    'timestamp',
    'model_name',
    'benchmark_name',
    'prompt_id',
    'prompt_text',
    'model_response',
    'expected_keywords',
    'matched_keywords',
    'score',
    'success',
    'evaluation_method',
    'response_time_seconds',
    'error',
    'error_type'
]


class ResultRecord:
    """Type-safe container for a single benchmark result record."""

    def __init__(self, eval_data: Dict[str, Any]):
        """Initialize from evaluation data dictionary."""
        self.timestamp = eval_data.get('timestamp', '')
        self.model_name = eval_data.get('model_name', '')
        self.benchmark_name = eval_data.get('benchmark_name', '')
        self.prompt_id = eval_data.get('prompt_id', '')
        self.prompt_text = eval_data.get('prompt', '')
        self.model_response = eval_data.get('response', '')
        self.expected_keywords = eval_data.get('expected_keywords', [])
        self.matched_keywords = eval_data.get('matched_keywords', [])
        self.score = eval_data.get('score', 0.0)
        self.success = eval_data.get('success', False)
        self.evaluation_method = eval_data.get('evaluation_method', '')
        self.response_time_seconds = eval_data.get('response_time_seconds', 0.0)
        self.error = eval_data.get('error', '')
        self.error_type = eval_data.get('error_type', '')

    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for CSV writing."""
        return {
            'timestamp': format_timestamp(self.timestamp),
            'model_name': self.model_name,
            'benchmark_name': self.benchmark_name,
            'prompt_id': self.prompt_id,
            'prompt_text': clean_multiline_text(truncate_text(self.prompt_text, 500)),
            'model_response': clean_multiline_text(truncate_text(self.model_response, 1000)),
            'expected_keywords': json.dumps(self.expected_keywords) if self.expected_keywords else '',
            'matched_keywords': json.dumps(self.matched_keywords) if self.matched_keywords else '',
            'score': f"{self.score:.4f}",
            'success': 'pass' if self.success else 'fail',
            'evaluation_method': self.evaluation_method,
            'response_time_seconds': f"{self.response_time_seconds:.3f}" if self.response_time_seconds else '',
            'error': clean_multiline_text(self.error) if self.error else '',
            'error_type': self.error_type
        }


class CSVResultLogger:
    """Handles writing benchmark results to CSV files."""

    def __init__(self, output_dir: str = './results'):
        """
        Initialize the CSV logger.
        
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = Path(output_dir)
        self.ensure_output_dir()

    def ensure_output_dir(self):
        """Ensure the output directory exists."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Failed to create output directory {self.output_dir}: {e!s}")

    def get_organized_path(self, dataset: str, date: Optional[datetime] = None) -> Path:
        """
        Get the organized directory path for storing results.
        
        Directory structure: results/{dataset}/{YYYY-MM}/
        
        Args:
            dataset: Name of the benchmark dataset
            date: Date for organizing (defaults to current date)
            
        Returns:
            Path object for the organized directory
        """
        if date is None:
            date = datetime.now()
        
        # Create subdirectory structure: dataset/YYYY-MM
        year_month = date.strftime('%Y-%m')
        organized_path = self.output_dir / dataset / year_month
        
        # Ensure the directory exists
        try:
            organized_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Failed to create organized directory {organized_path}: {e!s}")
        
        return organized_path

    def generate_filename(self, provider: str, dataset: str, model: Optional[str] = None) -> str:
        """
        Generate a filename for the CSV file.
        
        Args:
            provider: Name of the LLM provider
            dataset: Name of the benchmark dataset
            model: Name of the specific model (optional)
            
        Returns:
            Generated filename with timestamp
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Clean model name for filesystem (replace problematic characters)
        if model:
            safe_model = model.replace('/', '-').replace('\\', '-').replace(':', '-')
            return f"benchmark_{provider}_{safe_model}_{dataset}_{timestamp}.csv"
        else:
            return f"benchmark_{provider}_{dataset}_{timestamp}.csv"

    def write_results(self, results: Dict[str, Any], filename: Optional[str] = None, use_organized_dirs: bool = True) -> str:
        """
        Write benchmark results to a CSV file.
        
        Args:
            results: Complete results dictionary from benchmark run
            filename: Optional specific filename to use
            use_organized_dirs: Whether to use organized directory structure
            
        Returns:
            Path to the created CSV file
        """
        dataset = results.get('dataset', 'unknown')
        
        if not filename:
            filename = self.generate_filename(
                results.get('provider', 'unknown'),
                dataset,
                results.get('model')  # Pass model name if available
            )

        # Determine the directory to use
        if use_organized_dirs:
            # Extract timestamp from results if available
            result_date = None
            if 'start_time' in results:
                try:
                    result_date = datetime.fromisoformat(results['start_time'])
                except (ValueError, TypeError):
                    pass
            
            output_dir = self.get_organized_path(dataset, result_date)
        else:
            output_dir = self.output_dir
        
        filepath = output_dir / filename

        # Extract evaluation records
        evaluations = results.get('evaluations', [])
        if not evaluations:
            raise ValueError("No evaluation results to write")

        # Convert to ResultRecord objects
        records = [ResultRecord(eval_data) for eval_data in evaluations]

        # Define the write function for atomic writing
        def write_csv(file_obj):
            writer = csv.DictWriter(
                file_obj,
                fieldnames=CSV_COLUMNS,
                quoting=csv.QUOTE_MINIMAL,
                escapechar='\\',
                lineterminator='\n'
            )
            
            # Write header
            writer.writeheader()
            
            # Write records
            for record in records:
                writer.writerow(record.to_csv_dict())

        # Write to CSV atomically
        try:
            self._atomic_write(filepath, write_csv)
        except Exception as e:
            raise OSError(f"Failed to write CSV file {filepath}: {e!s}")
        
        # Write metadata and update index (non-critical operations)
        try:
            self._write_metadata(filepath, results)
        except Exception:
            pass  # Already logged in _write_metadata
        
        try:
            self._update_index(filepath, results)
        except Exception:
            pass  # Already logged in _update_index
        
        return str(filepath)

    def _atomic_write(self, filepath: Path, write_func):
        """
        Perform atomic file writing using a temporary file and rename.
        
        This ensures that the file write is atomic - either the entire file
        is written successfully or nothing is written at all. This prevents
        corruption during concurrent access or system failures.
        
        Args:
            filepath: Target file path
            write_func: Function that performs the actual writing to a file object
        """
        # Create temporary file in the same directory as target
        # This ensures we're on the same filesystem for atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f'.{filepath.stem}_',
            suffix='.tmp'
        )
        
        try:
            # Write to temporary file
            with os.fdopen(temp_fd, 'w', newline='', encoding='utf-8') as temp_file:
                write_func(temp_file)
            
            # Atomic rename (on POSIX systems)
            # On Windows, this might not be fully atomic but is still safer
            Path(temp_path).replace(filepath)
            
        except Exception:
            # Clean up temporary file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _atomic_append(self, filepath: Path, record: ResultRecord, write_header: bool):
        """
        Perform atomic append operation.
        
        For appending, we need to copy existing content first, then add new data.
        This is more complex than simple write but ensures atomicity.
        
        Args:
            filepath: Target file path
            record: Record to append
            write_header: Whether to write CSV header
        """
        # If file doesn't exist, just do atomic write
        if not filepath.exists() or write_header:
            def write_new(file_obj):
                writer = csv.DictWriter(
                    file_obj,
                    fieldnames=CSV_COLUMNS,
                    quoting=csv.QUOTE_MINIMAL,
                    escapechar='\\',
                    lineterminator='\n'
                )
                writer.writeheader()
                writer.writerow(record.to_csv_dict())
            
            self._atomic_write(filepath, write_new)
            return
        
        # For existing files, we need to copy and append
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f'.{filepath.stem}_append_',
            suffix='.tmp'
        )
        
        try:
            # Copy existing content
            with open(filepath, 'r', encoding='utf-8') as original:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
                    temp_file.write(original.read())
                    
                    # Append new record
                    writer = csv.DictWriter(
                        temp_file,
                        fieldnames=CSV_COLUMNS,
                        quoting=csv.QUOTE_MINIMAL,
                        escapechar='\\',
                        lineterminator='\n'
                    )
                    writer.writerow(record.to_csv_dict())
            
            # Atomic rename
            Path(temp_path).replace(filepath)
            
        except Exception:
            # Clean up temporary file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def append_result(self, filepath: Union[str, Path], eval_data: Dict[str, Any]):
        """
        Append a single result to an existing CSV file using atomic operations.
        
        Args:
            filepath: Path to the CSV file
            eval_data: Single evaluation result dictionary
        """
        filepath = Path(filepath)
        record = ResultRecord(eval_data)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to determine if we need header
        write_header = not filepath.exists()

        try:
            self._atomic_append(filepath, record, write_header)
        except Exception as e:
            raise OSError(f"Failed to append to CSV file {filepath}: {e!s}")

    def _write_metadata(self, filepath: Path, results: Dict[str, Any]):
        """
        Write metadata file alongside the CSV results.
        
        The metadata file contains information about the benchmark run including:
        - Provider and model information
        - API configuration used
        - Timing information
        - Summary statistics
        
        Args:
            filepath: Path to the CSV file
            results: Complete results dictionary
        """
        metadata = {
            'csv_file': filepath.name,
            'provider': results.get('provider', 'unknown'),
            'model': results.get('model', 'unknown'),
            'dataset': results.get('dataset', 'unknown'),
            'start_time': results.get('start_time'),
            'end_time': results.get('end_time'),
            'total_duration_seconds': results.get('total_duration_seconds', 0),
            'total_prompts': results.get('total_prompts', 0),
            'successful_evaluations': results.get('successful_evaluations', 0),
            'failed_evaluations': results.get('failed_evaluations', 0),
            'overall_score': results.get('overall_score', 0),
            'average_response_time_seconds': results.get('average_response_time_seconds', 0),
            'model_config': results.get('model_config', {}),
            'error': results.get('error'),
            'created_at': datetime.now().isoformat()
        }
        
        # Write metadata file
        metadata_path = filepath.with_suffix('.meta.json')
        
        def write_json(file_obj):
            json.dump(metadata, file_obj, indent=2, ensure_ascii=False)
        
        try:
            self._atomic_write(metadata_path, write_json)
        except Exception as e:
            # Log error but don't fail the main operation
            print(f"Warning: Failed to write metadata file: {e}")

    def _update_index(self, filepath: Path, results: Dict[str, Any]):
        """
        Update the benchmark index file.
        
        The index file maintains a registry of all benchmark runs for easy lookup.
        
        Args:
            filepath: Path to the CSV file
            results: Complete results dictionary
        """
        index_path = self.output_dir / 'benchmark_index.json'
        
        # Create index entry
        index_entry = {
            'id': f"{results.get('provider', 'unknown')}_{results.get('model', 'unknown')}_{results.get('dataset', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'csv_file': str(filepath.relative_to(self.output_dir)),
            'metadata_file': str(filepath.with_suffix('.meta.json').relative_to(self.output_dir)),
            'provider': results.get('provider', 'unknown'),
            'model': results.get('model', 'unknown'),
            'dataset': results.get('dataset', 'unknown'),
            'timestamp': results.get('start_time', datetime.now().isoformat()),
            'overall_score': results.get('overall_score', 0),
            'total_prompts': results.get('total_prompts', 0),
            'duration_seconds': results.get('total_duration_seconds', 0)
        }
        
        # Load existing index or create new one
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    if not isinstance(index_data, dict):
                        index_data = {'version': '1.0', 'entries': []}
            except (json.JSONDecodeError, IOError):
                index_data = {'version': '1.0', 'entries': []}
        else:
            index_data = {'version': '1.0', 'entries': []}
        
        # Add new entry
        index_data['entries'].append(index_entry)
        index_data['last_updated'] = datetime.now().isoformat()
        
        # Sort entries by timestamp (newest first)
        index_data['entries'].sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Write updated index
        def write_json(file_obj):
            json.dump(index_data, file_obj, indent=2, ensure_ascii=False)
        
        try:
            self._atomic_write(index_path, write_json)
        except Exception as e:
            # Log error but don't fail the main operation
            print(f"Warning: Failed to update index file: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # No cleanup needed
        pass
