"""
Data models for analysis results.

This module contains dataclasses for storing and manipulating
benchmark results and comparison data.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ModelResult:
    """Container for a single model's benchmark results."""
    model_name: str
    provider: str
    dataset: str
    timestamp: str
    total_prompts: int
    successful_evaluations: int
    failed_evaluations: int
    overall_score: float
    average_response_time: float
    evaluations: List[Dict[str, Any]]
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'ModelResult':
        """Load model results from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return cls(
            model_name=data.get('model', 'unknown'),
            provider=data.get('provider', 'unknown'),
            dataset=data.get('dataset', 'unknown'),
            timestamp=data.get('start_time', ''),
            total_prompts=data.get('total_prompts', 0),
            successful_evaluations=data.get('successful_evaluations', 0),
            failed_evaluations=data.get('failed_evaluations', 0),
            overall_score=data.get('overall_score', 0.0),
            average_response_time=data.get('average_response_time_seconds', 0.0),
            evaluations=data.get('evaluations', []),
            model_config=data.get('model_config', {})
        )
    
    @classmethod
    def from_csv(cls, csv_path: Union[str, Path]) -> 'ModelResult':
        """Load model results from a CSV file."""
        df = pd.read_csv(csv_path)
        
        # Extract metadata from the first row (assuming consistent metadata)
        if len(df) > 0:
            first_row = df.iloc[0]
            model_name = first_row.get('model_name', 'unknown')
            provider = first_row.get('provider', 'unknown')
            dataset = first_row.get('benchmark_name', 'unknown')
            timestamp = first_row.get('timestamp', '')
        else:
            model_name = provider = dataset = 'unknown'
            timestamp = ''
        
        # Calculate metrics
        total_prompts = len(df)
        successful = len(df[df['success'] == True])
        failed = total_prompts - successful
        overall_score = successful / total_prompts if total_prompts > 0 else 0.0
        
        # Calculate average response time
        response_times = df['response_time_seconds'].dropna()
        avg_response_time = response_times.mean() if len(response_times) > 0 else 0.0
        
        # Convert evaluations to list of dicts
        evaluations = df.to_dict('records')
        
        return cls(
            model_name=model_name,
            provider=provider,
            dataset=dataset,
            timestamp=timestamp,
            total_prompts=total_prompts,
            successful_evaluations=successful,
            failed_evaluations=failed,
            overall_score=overall_score,
            average_response_time=avg_response_time,
            evaluations=evaluations
        )


@dataclass
class ComparisonResult:
    """Container for comparison results between multiple models."""
    models: List[str]
    dataset: str
    comparison_timestamp: str
    prompt_alignment: Dict[str, Dict[str, Any]]  # prompt_id -> {model_name: evaluation}
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {model_name: value}
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    rankings: Dict[str, List[str]] = field(default_factory=dict)  # metric -> ranked model list