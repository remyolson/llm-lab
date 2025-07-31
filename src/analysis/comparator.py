"""
Results Comparison and Reporting Module

This module provides functionality to compare benchmark results across multiple
LLM models and generate comprehensive reports including statistical analysis.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from scipy import stats
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


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
    

class ResultsComparator:
    """
    Compare benchmark results across multiple LLM models.
    
    This class provides functionality to:
    - Load and align results from multiple models
    - Calculate comparative metrics
    - Perform statistical analysis
    - Generate reports in various formats
    """
    
    def __init__(self, results_dir: Union[str, Path] = "./results"):
        """
        Initialize the ResultsComparator.
        
        Args:
            results_dir: Directory containing result files
        """
        self.results_dir = Path(results_dir)
        self.model_results: Dict[str, ModelResult] = {}
        self.comparison_result: Optional[ComparisonResult] = None
        
    def load_results(self, 
                    model_names: Optional[List[str]] = None,
                    dataset: Optional[str] = None,
                    file_format: str = "json") -> Dict[str, ModelResult]:
        """
        Load results for specified models and dataset.
        
        Args:
            model_names: List of model names to load (None = all available)
            dataset: Dataset name to filter by (None = any dataset)
            file_format: File format to load ("json" or "csv")
            
        Returns:
            Dictionary mapping model names to ModelResult objects
        """
        self.model_results.clear()
        
        # Find result files
        pattern = f"*.{file_format}"
        result_files = list(self.results_dir.glob(pattern))
        
        if not result_files:
            logger.warning(f"No {file_format} files found in {self.results_dir}")
            return self.model_results
        
        # Load each result file
        for file_path in result_files:
            try:
                # Load result based on format
                if file_format == "json":
                    result = ModelResult.from_json(file_path)
                else:  # csv
                    result = ModelResult.from_csv(file_path)
                
                # Filter by model name if specified
                if model_names and result.model_name not in model_names:
                    continue
                
                # Filter by dataset if specified
                if dataset and result.dataset != dataset:
                    continue
                
                # Store result
                self.model_results[result.model_name] = result
                logger.info(f"Loaded results for {result.model_name} from {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        return self.model_results
    
    def align_results_by_prompt(self) -> Dict[str, Dict[str, Any]]:
        """
        Align results by prompt ID to enable direct comparison.
        
        Returns:
            Dictionary mapping prompt IDs to model evaluations
        """
        if not self.model_results:
            raise ValueError("No results loaded. Call load_results() first.")
        
        # Create alignment structure
        prompt_alignment = defaultdict(dict)
        
        # Process each model's results
        for model_name, model_result in self.model_results.items():
            for evaluation in model_result.evaluations:
                prompt_id = evaluation.get('prompt_id', 'unknown')
                prompt_alignment[prompt_id][model_name] = evaluation
        
        # Convert to regular dict and filter out prompts not evaluated by all models
        all_models = set(self.model_results.keys())
        complete_alignment = {}
        
        for prompt_id, model_evals in prompt_alignment.items():
            # Only include prompts evaluated by all models
            if set(model_evals.keys()) == all_models:
                complete_alignment[prompt_id] = dict(model_evals)
        
        logger.info(f"Aligned {len(complete_alignment)} prompts across {len(all_models)} models")
        
        return complete_alignment
    
    def calculate_metrics(self, aligned_results: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate comparative metrics for all models.
        
        Args:
            aligned_results: Pre-aligned results (if None, will call align_results_by_prompt)
            
        Returns:
            Dictionary mapping metric names to model scores
        """
        if aligned_results is None:
            aligned_results = self.align_results_by_prompt()
        
        metrics = {
            'overall_accuracy': {},
            'average_response_time': {},
            'consistency_score': {},
            'success_rate': {},
            'average_confidence': {},
            'response_length': {}
        }
        
        # Calculate metrics for each model
        for model_name, model_result in self.model_results.items():
            # Overall accuracy (same as overall_score)
            metrics['overall_accuracy'][model_name] = model_result.overall_score
            
            # Average response time
            metrics['average_response_time'][model_name] = model_result.average_response_time
            
            # Success rate
            total = model_result.total_prompts
            if total > 0:
                metrics['success_rate'][model_name] = model_result.successful_evaluations / total
            else:
                metrics['success_rate'][model_name] = 0.0
            
            # Calculate additional metrics from evaluations
            confidences = []
            response_lengths = []
            
            for evaluation in model_result.evaluations:
                # Confidence (if available)
                if 'confidence' in evaluation:
                    confidences.append(evaluation['confidence'])
                elif 'score' in evaluation:
                    confidences.append(evaluation['score'])
                
                # Response length
                if 'response' in evaluation:
                    response_lengths.append(len(str(evaluation['response'])))
            
            # Average confidence
            if confidences:
                metrics['average_confidence'][model_name] = np.mean(confidences)
            else:
                metrics['average_confidence'][model_name] = 0.0
            
            # Average response length
            if response_lengths:
                metrics['response_length'][model_name] = np.mean(response_lengths)
            else:
                metrics['response_length'][model_name] = 0.0
        
        # Calculate consistency scores (agreement with majority)
        if len(aligned_results) > 0:
            consistency_scores = defaultdict(int)
            total_prompts = len(aligned_results)
            
            for prompt_id, model_evals in aligned_results.items():
                # Get all responses for this prompt
                responses = {}
                for model_name, evaluation in model_evals.items():
                    if evaluation.get('success', False):
                        responses[model_name] = evaluation.get('matched_keywords', [])
                
                # Find majority response (most common set of keywords)
                if responses:
                    # Convert keyword lists to tuples for comparison
                    response_tuples = {m: tuple(sorted(kw)) for m, kw in responses.items()}
                    
                    # Count occurrences of each response
                    response_counts = defaultdict(list)
                    for model, response_tuple in response_tuples.items():
                        response_counts[response_tuple].append(model)
                    
                    # Find majority response
                    majority_response = max(response_counts.keys(), 
                                          key=lambda x: len(response_counts[x]))
                    
                    # Award consistency points to models with majority response
                    for model in response_counts[majority_response]:
                        consistency_scores[model] += 1
            
            # Convert to percentage
            for model_name in self.model_results.keys():
                if total_prompts > 0:
                    metrics['consistency_score'][model_name] = consistency_scores[model_name] / total_prompts
                else:
                    metrics['consistency_score'][model_name] = 0.0
        
        return metrics
    
    def rank_models(self, metrics: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, List[str]]:
        """
        Rank models by each metric.
        
        Args:
            metrics: Pre-calculated metrics (if None, will call calculate_metrics)
            
        Returns:
            Dictionary mapping metric names to ranked lists of model names
        """
        if metrics is None:
            metrics = self.calculate_metrics()
        
        rankings = {}
        
        for metric_name, model_scores in metrics.items():
            # Sort models by score (descending for most metrics)
            if metric_name == 'average_response_time':
                # Lower is better for response time
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
            else:
                # Higher is better for other metrics
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            rankings[metric_name] = [model for model, _ in sorted_models]
        
        return rankings
    
    def calculate_statistical_significance(self, 
                                         model1: str, 
                                         model2: str,
                                         metric: str = 'success') -> Tuple[float, bool]:
        """
        Calculate statistical significance between two models using appropriate tests.
        
        Args:
            model1: First model name
            model2: Second model name
            metric: Metric to test ('success' for binary outcomes, 'response_time' for continuous)
            
        Returns:
            Tuple of (p_value, is_significant) where is_significant uses α=0.05
        """
        if model1 not in self.model_results or model2 not in self.model_results:
            raise ValueError(f"Both models must be loaded: {model1}, {model2}")
        
        results1 = self.model_results[model1]
        results2 = self.model_results[model2]
        
        if metric == 'success':
            # Binary outcome - use Chi-square test
            # Create contingency table
            success1 = results1.successful_evaluations
            fail1 = results1.failed_evaluations
            success2 = results2.successful_evaluations
            fail2 = results2.failed_evaluations
            
            contingency_table = np.array([
                [success1, fail1],
                [success2, fail2]
            ])
            
            # Perform Chi-square test
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            
        elif metric == 'response_time':
            # Continuous data - use t-test
            times1 = [e.get('response_time_seconds', 0) for e in results1.evaluations]
            times2 = [e.get('response_time_seconds', 0) for e in results2.evaluations]
            
            # Remove zeros (failed responses)
            times1 = [t for t in times1 if t > 0]
            times2 = [t for t in times2 if t > 0]
            
            if len(times1) < 2 or len(times2) < 2:
                return 1.0, False  # Not enough data
            
            # Perform Welch's t-test (doesn't assume equal variance)
            _, p_value = stats.ttest_ind(times1, times2, equal_var=False)
            
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Determine significance at α=0.05
        is_significant = p_value < 0.05
        
        return p_value, is_significant
    
    def calculate_confidence_intervals(self, 
                                     model_name: str,
                                     metric: str = 'overall_accuracy',
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence intervals for a model's metric.
        
        Args:
            model_name: Model name
            metric: Metric to calculate CI for
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if model_name not in self.model_results:
            raise ValueError(f"Model not loaded: {model_name}")
        
        result = self.model_results[model_name]
        
        if metric == 'overall_accuracy' or metric == 'success_rate':
            # Binomial proportion confidence interval
            n = result.total_prompts
            if n == 0:
                return 0.0, 0.0
            
            p = result.successful_evaluations / n
            
            # Wilson score interval
            z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            denominator = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denominator
            margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
            
            return max(0, centre - margin), min(1, centre + margin)
            
        elif metric == 'average_response_time':
            # Normal distribution CI for response times
            times = [e.get('response_time_seconds', 0) for e in result.evaluations 
                    if e.get('response_time_seconds', 0) > 0]
            
            if len(times) < 2:
                return 0.0, 0.0
            
            mean = np.mean(times)
            std = np.std(times, ddof=1)
            n = len(times)
            
            # t-distribution for small sample sizes
            t_value = stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
            margin = t_value * std / np.sqrt(n)
            
            return mean - margin, mean + margin
        
        else:
            raise ValueError(f"Unsupported metric for CI: {metric}")
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform comprehensive statistical tests on all model pairs.
        
        Returns:
            Dictionary containing test results
        """
        if len(self.model_results) < 2:
            return {}
        
        model_names = list(self.model_results.keys())
        test_results = {
            'pairwise_success_tests': {},
            'pairwise_time_tests': {},
            'confidence_intervals': {},
            'effect_sizes': {}
        }
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                pair_key = f"{model1}_vs_{model2}"
                
                # Success rate comparison
                try:
                    p_val, is_sig = self.calculate_statistical_significance(
                        model1, model2, 'success'
                    )
                    test_results['pairwise_success_tests'][pair_key] = {
                        'p_value': p_val,
                        'is_significant': is_sig
                    }
                except Exception as e:
                    logger.warning(f"Failed to test {pair_key} success rates: {e}")
                
                # Response time comparison
                try:
                    p_val, is_sig = self.calculate_statistical_significance(
                        model1, model2, 'response_time'
                    )
                    test_results['pairwise_time_tests'][pair_key] = {
                        'p_value': p_val,
                        'is_significant': is_sig
                    }
                except Exception as e:
                    logger.warning(f"Failed to test {pair_key} response times: {e}")
        
        # Confidence intervals for each model
        for model_name in model_names:
            test_results['confidence_intervals'][model_name] = {}
            
            # Accuracy CI
            try:
                lower, upper = self.calculate_confidence_intervals(
                    model_name, 'overall_accuracy'
                )
                test_results['confidence_intervals'][model_name]['accuracy'] = {
                    'lower': lower,
                    'upper': upper,
                    'point_estimate': self.model_results[model_name].overall_score
                }
            except Exception as e:
                logger.warning(f"Failed to calculate accuracy CI for {model_name}: {e}")
            
            # Response time CI
            try:
                lower, upper = self.calculate_confidence_intervals(
                    model_name, 'average_response_time'
                )
                test_results['confidence_intervals'][model_name]['response_time'] = {
                    'lower': lower,
                    'upper': upper,
                    'point_estimate': self.model_results[model_name].average_response_time
                }
            except Exception as e:
                logger.warning(f"Failed to calculate time CI for {model_name}: {e}")
        
        # Calculate effect sizes (Cohen's h for proportions)
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                pair_key = f"{model1}_vs_{model2}"
                
                try:
                    # Cohen's h for difference in proportions
                    p1 = self.model_results[model1].overall_score
                    p2 = self.model_results[model2].overall_score
                    
                    # Arcsine transformation
                    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
                    
                    test_results['effect_sizes'][pair_key] = {
                        'cohens_h': h,
                        'interpretation': self._interpret_cohens_h(abs(h))
                    }
                except Exception as e:
                    logger.warning(f"Failed to calculate effect size for {pair_key}: {e}")
        
        return test_results
    
    def _interpret_cohens_h(self, h: float) -> str:
        """Interpret Cohen's h effect size."""
        if h < 0.2:
            return "small"
        elif h < 0.5:
            return "medium"
        elif h < 0.8:
            return "large"
        else:
            return "very large"
    
    def compare(self, 
                model_names: Optional[List[str]] = None,
                dataset: Optional[str] = None) -> ComparisonResult:
        """
        Perform a complete comparison of models.
        
        Args:
            model_names: List of model names to compare (None = all available)
            dataset: Dataset name to filter by (None = any dataset)
            
        Returns:
            ComparisonResult object containing all comparison data
        """
        # Load results
        self.load_results(model_names, dataset)
        
        if len(self.model_results) < 2:
            raise ValueError("At least 2 models are required for comparison")
        
        # Align results
        prompt_alignment = self.align_results_by_prompt()
        
        # Calculate metrics
        metrics = self.calculate_metrics(prompt_alignment)
        
        # Rank models
        rankings = self.rank_models(metrics)
        
        # Perform statistical tests
        statistical_tests = self.perform_statistical_tests()
        
        # Get dataset name (should be consistent across models)
        dataset_name = next(iter(self.model_results.values())).dataset
        
        # Create comparison result
        self.comparison_result = ComparisonResult(
            models=list(self.model_results.keys()),
            dataset=dataset_name,
            comparison_timestamp=datetime.now().isoformat(),
            prompt_alignment=prompt_alignment,
            metrics=metrics,
            statistical_tests=statistical_tests,
            rankings=rankings
        )
        
        return self.comparison_result
    
    def generate_comparison_csv(self, 
                              output_path: Union[str, Path],
                              include_all_prompts: bool = False) -> None:
        """
        Generate a CSV file with side-by-side model comparisons.
        
        Args:
            output_path: Path to save the CSV file
            include_all_prompts: If True, include all prompts; if False, only aligned prompts
        """
        if not self.comparison_result:
            raise ValueError("No comparison performed. Call compare() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for CSV
        rows = []
        
        if include_all_prompts:
            # Include all prompts from all models
            all_prompts = set()
            prompt_to_text = {}
            
            for model_result in self.model_results.values():
                for evaluation in model_result.evaluations:
                    prompt_id = evaluation.get('prompt_id', 'unknown')
                    all_prompts.add(prompt_id)
                    if prompt_id not in prompt_to_text:
                        prompt_to_text[prompt_id] = evaluation.get('prompt', '')
            
            for prompt_id in sorted(all_prompts):
                row = {
                    'prompt_id': prompt_id,
                    'prompt': prompt_to_text.get(prompt_id, '')
                }
                
                # Add results for each model
                for model_name in self.comparison_result.models:
                    model_prefix = f"{model_name}_"
                    
                    # Find evaluation for this prompt and model
                    evaluation = None
                    for eval_data in self.model_results[model_name].evaluations:
                        if eval_data.get('prompt_id') == prompt_id:
                            evaluation = eval_data
                            break
                    
                    if evaluation:
                        row[f"{model_prefix}success"] = evaluation.get('success', False)
                        row[f"{model_prefix}response"] = evaluation.get('response', '')
                        row[f"{model_prefix}response_time"] = evaluation.get('response_time_seconds', 0)
                        row[f"{model_prefix}matched_keywords"] = ', '.join(evaluation.get('matched_keywords', []))
                    else:
                        row[f"{model_prefix}success"] = None
                        row[f"{model_prefix}response"] = 'NOT_EVALUATED'
                        row[f"{model_prefix}response_time"] = None
                        row[f"{model_prefix}matched_keywords"] = ''
                
                rows.append(row)
        else:
            # Only aligned prompts
            for prompt_id, model_evals in self.comparison_result.prompt_alignment.items():
                row = {
                    'prompt_id': prompt_id,
                    'prompt': next(iter(model_evals.values())).get('prompt', '')
                }
                
                # Add results for each model
                for model_name, evaluation in model_evals.items():
                    model_prefix = f"{model_name}_"
                    row[f"{model_prefix}success"] = evaluation.get('success', False)
                    row[f"{model_prefix}response"] = evaluation.get('response', '')
                    row[f"{model_prefix}response_time"] = evaluation.get('response_time_seconds', 0)
                    row[f"{model_prefix}matched_keywords"] = ', '.join(evaluation.get('matched_keywords', []))
                
                rows.append(row)
        
        # Write to CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved comparison CSV to {output_path}")
        else:
            logger.warning("No data to save to CSV")
    
    def generate_metrics_csv(self, output_path: Union[str, Path]) -> None:
        """
        Generate a CSV file with model metrics summary.
        
        Args:
            output_path: Path to save the CSV file
        """
        if not self.comparison_result:
            raise ValueError("No comparison performed. Call compare() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metrics data
        rows = []
        
        for model_name in self.comparison_result.models:
            row = {'model': model_name}
            
            # Add all metrics
            for metric_name, model_scores in self.comparison_result.metrics.items():
                row[metric_name] = model_scores.get(model_name, 0.0)
            
            # Add confidence intervals if available
            if 'confidence_intervals' in self.comparison_result.statistical_tests:
                ci_data = self.comparison_result.statistical_tests['confidence_intervals'].get(model_name, {})
                
                if 'accuracy' in ci_data:
                    row['accuracy_ci_lower'] = ci_data['accuracy']['lower']
                    row['accuracy_ci_upper'] = ci_data['accuracy']['upper']
                
                if 'response_time' in ci_data:
                    row['response_time_ci_lower'] = ci_data['response_time']['lower']
                    row['response_time_ci_upper'] = ci_data['response_time']['upper']
            
            # Add basic stats from model results
            model_result = self.model_results[model_name]
            row['total_prompts'] = model_result.total_prompts
            row['successful_evaluations'] = model_result.successful_evaluations
            row['failed_evaluations'] = model_result.failed_evaluations
            
            rows.append(row)
        
        # Write to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved metrics CSV to {output_path}")
    
    def generate_statistical_tests_csv(self, output_path: Union[str, Path]) -> None:
        """
        Generate a CSV file with statistical test results.
        
        Args:
            output_path: Path to save the CSV file
        """
        if not self.comparison_result or not self.comparison_result.statistical_tests:
            raise ValueError("No statistical tests performed. Call compare() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare statistical test data
        rows = []
        
        # Pairwise success tests
        for pair_key, test_result in self.comparison_result.statistical_tests.get('pairwise_success_tests', {}).items():
            models = pair_key.split('_vs_')
            row = {
                'test_type': 'success_rate_comparison',
                'model1': models[0],
                'model2': models[1],
                'p_value': test_result['p_value'],
                'is_significant': test_result['is_significant']
            }
            
            # Add effect size if available
            effect_key = pair_key
            if effect_key in self.comparison_result.statistical_tests.get('effect_sizes', {}):
                effect_data = self.comparison_result.statistical_tests['effect_sizes'][effect_key]
                row['cohens_h'] = effect_data['cohens_h']
                row['effect_size'] = effect_data['interpretation']
            
            rows.append(row)
        
        # Pairwise time tests
        for pair_key, test_result in self.comparison_result.statistical_tests.get('pairwise_time_tests', {}).items():
            models = pair_key.split('_vs_')
            row = {
                'test_type': 'response_time_comparison',
                'model1': models[0],
                'model2': models[1],
                'p_value': test_result['p_value'],
                'is_significant': test_result['is_significant'],
                'cohens_h': None,
                'effect_size': None
            }
            rows.append(row)
        
        # Write to CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved statistical tests CSV to {output_path}")
        else:
            logger.warning("No statistical test data to save")
    
    def generate_markdown_report(self, output_path: Union[str, Path]) -> None:
        """
        Generate a comprehensive markdown report with analysis.
        
        Args:
            output_path: Path to save the markdown file
        """
        if not self.comparison_result:
            raise ValueError("No comparison performed. Call compare() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Start building the report
        report_lines = [
            f"# LLM Benchmark Comparison Report",
            f"",
            f"**Dataset**: {self.comparison_result.dataset}  ",
            f"**Models Compared**: {len(self.comparison_result.models)}  ",
            f"**Date**: {self.comparison_result.comparison_timestamp}  ",
            f"**Total Aligned Prompts**: {len(self.comparison_result.prompt_alignment)}  ",
            f"",
            f"## Executive Summary",
            f""
        ]
        
        # Find best model for each metric
        best_models = {}
        for metric, ranking in self.comparison_result.rankings.items():
            if ranking:
                best_models[metric] = ranking[0]
        
        # Summary bullets
        if 'overall_accuracy' in best_models:
            best_acc_model = best_models['overall_accuracy']
            best_acc_score = self.comparison_result.metrics['overall_accuracy'][best_acc_model]
            report_lines.append(f"- **Best Overall Accuracy**: {best_acc_model} ({best_acc_score:.2%})")
        
        if 'average_response_time' in best_models:
            best_time_model = best_models['average_response_time']
            best_time = self.comparison_result.metrics['average_response_time'][best_time_model]
            report_lines.append(f"- **Fastest Response Time**: {best_time_model} ({best_time:.2f}s average)")
        
        if 'consistency_score' in best_models:
            best_cons_model = best_models['consistency_score']
            best_cons = self.comparison_result.metrics['consistency_score'][best_cons_model]
            report_lines.append(f"- **Most Consistent**: {best_cons_model} ({best_cons:.2%} agreement with consensus)")
        
        # Model Performance Table
        report_lines.extend([
            "",
            "## Model Performance Overview",
            "",
            "| Model | Accuracy | Success Rate | Avg Response Time | Consistency |",
            "|-------|----------|--------------|-------------------|-------------|"
        ])
        
        for model in self.comparison_result.models:
            metrics = self.comparison_result.metrics
            accuracy = metrics.get('overall_accuracy', {}).get(model, 0)
            success_rate = metrics.get('success_rate', {}).get(model, 0)
            response_time = metrics.get('average_response_time', {}).get(model, 0)
            consistency = metrics.get('consistency_score', {}).get(model, 0)
            
            report_lines.append(
                f"| {model} | {accuracy:.2%} | {success_rate:.2%} | "
                f"{response_time:.2f}s | {consistency:.2%} |"
            )
        
        # Statistical Significance Section
        if self.comparison_result.statistical_tests:
            report_lines.extend([
                "",
                "## Statistical Analysis",
                "",
                "### Pairwise Comparisons (Success Rate)",
                "",
                "| Model 1 | Model 2 | p-value | Significant? | Effect Size |",
                "|---------|---------|---------|--------------|-------------|"
            ])
            
            for pair_key, test_result in self.comparison_result.statistical_tests.get('pairwise_success_tests', {}).items():
                models = pair_key.split('_vs_')
                p_val = test_result['p_value']
                is_sig = "Yes" if test_result['is_significant'] else "No"
                
                # Get effect size
                effect_info = self.comparison_result.statistical_tests.get('effect_sizes', {}).get(pair_key, {})
                effect_size = effect_info.get('interpretation', 'N/A')
                
                report_lines.append(
                    f"| {models[0]} | {models[1]} | {p_val:.4f} | {is_sig} | {effect_size} |"
                )
            
            # Confidence Intervals
            report_lines.extend([
                "",
                "### Confidence Intervals (95%)",
                "",
                "| Model | Accuracy CI | Response Time CI |",
                "|-------|-------------|------------------|"
            ])
            
            for model, ci_data in self.comparison_result.statistical_tests.get('confidence_intervals', {}).items():
                acc_ci = ci_data.get('accuracy', {})
                time_ci = ci_data.get('response_time', {})
                
                acc_str = f"[{acc_ci.get('lower', 0):.2%}, {acc_ci.get('upper', 0):.2%}]" if acc_ci else "N/A"
                time_str = f"[{time_ci.get('lower', 0):.2f}s, {time_ci.get('upper', 0):.2f}s]" if time_ci else "N/A"
                
                report_lines.append(f"| {model} | {acc_str} | {time_str} |")
        
        # Model Rankings
        report_lines.extend([
            "",
            "## Model Rankings by Metric",
            ""
        ])
        
        for metric_name, ranking in self.comparison_result.rankings.items():
            if ranking:
                # Format metric name
                formatted_name = metric_name.replace('_', ' ').title()
                report_lines.append(f"### {formatted_name}")
                report_lines.append("")
                
                for i, model in enumerate(ranking, 1):
                    score = self.comparison_result.metrics[metric_name].get(model, 0)
                    if metric_name == 'average_response_time':
                        report_lines.append(f"{i}. {model}: {score:.2f}s")
                    else:
                        report_lines.append(f"{i}. {model}: {score:.2%}")
                report_lines.append("")
        
        # Model Strengths and Weaknesses
        report_lines.extend([
            "## Model Analysis",
            ""
        ])
        
        for model in self.comparison_result.models:
            report_lines.append(f"### {model}")
            report_lines.append("")
            
            strengths = []
            weaknesses = []
            
            # Analyze rankings
            for metric, ranking in self.comparison_result.rankings.items():
                position = ranking.index(model) + 1 if model in ranking else len(ranking)
                total = len(ranking)
                
                formatted_metric = metric.replace('_', ' ').title()
                
                if position == 1:
                    strengths.append(f"Best {formatted_metric.lower()}")
                elif position <= total / 3:
                    strengths.append(f"Strong {formatted_metric.lower()}")
                elif position >= 2 * total / 3:
                    weaknesses.append(f"Lower {formatted_metric.lower()}")
            
            # Model result details
            model_result = self.model_results[model]
            if model_result.failed_evaluations > 0:
                fail_rate = model_result.failed_evaluations / model_result.total_prompts
                if fail_rate > 0.1:
                    weaknesses.append(f"High failure rate ({fail_rate:.1%})")
            
            report_lines.append("**Strengths:**")
            if strengths:
                for strength in strengths:
                    report_lines.append(f"- {strength}")
            else:
                report_lines.append("- No standout strengths identified")
            
            report_lines.append("")
            report_lines.append("**Areas for Improvement:**")
            if weaknesses:
                for weakness in weaknesses:
                    report_lines.append(f"- {weakness}")
            else:
                report_lines.append("- No significant weaknesses identified")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # Best overall model
        if 'overall_accuracy' in best_models:
            best_model = best_models['overall_accuracy']
            recommendations.append(
                f"1. **For maximum accuracy**: Use {best_model} which achieved the "
                f"highest overall accuracy of {self.comparison_result.metrics['overall_accuracy'][best_model]:.2%}"
            )
        
        # Best speed/accuracy tradeoff
        if 'overall_accuracy' in self.comparison_result.metrics and 'average_response_time' in self.comparison_result.metrics:
            # Calculate efficiency score (accuracy / response_time)
            efficiency_scores = {}
            for model in self.comparison_result.models:
                acc = self.comparison_result.metrics['overall_accuracy'].get(model, 0)
                time = self.comparison_result.metrics['average_response_time'].get(model, 1)
                if time > 0:
                    efficiency_scores[model] = acc / time
            
            if efficiency_scores:
                best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1])
                recommendations.append(
                    f"2. **For best speed/accuracy balance**: Consider {best_efficiency[0]} "
                    f"with an efficiency score of {best_efficiency[1]:.3f}"
                )
        
        # Cost considerations
        if 'average_response_time' in best_models:
            fastest = best_models['average_response_time']
            recommendations.append(
                f"3. **For fastest responses**: Use {fastest} with average response "
                f"time of {self.comparison_result.metrics['average_response_time'][fastest]:.2f}s"
            )
        
        for rec in recommendations:
            report_lines.append(rec)
        
        # Footer
        report_lines.extend([
            "",
            "---",
            f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved markdown report to {output_path}")
    
    def generate_visualizations(self, output_dir: Union[str, Path]) -> List[str]:
        """
        Generate visualization charts for the comparison results.
        
        Args:
            output_dir: Directory to save visualization files
            
        Returns:
            List of generated file paths
        """
        if not self.comparison_result:
            raise ValueError("No comparison performed. Call compare() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Overall Performance Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = self.comparison_result.models
        metrics_to_plot = ['overall_accuracy', 'success_rate', 'consistency_score']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = [self.comparison_result.metrics[metric].get(model, 0) for model in models]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        performance_chart_path = output_dir / 'performance_comparison.png'
        plt.savefig(performance_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(str(performance_chart_path))
        
        # 2. Response Time Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        response_times = [self.comparison_result.metrics['average_response_time'].get(model, 0) 
                         for model in models]
        
        bars = ax.bar(models, response_times, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}s',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Average Response Time (seconds)')
        ax.set_title('Response Time Comparison')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        response_time_path = output_dir / 'response_time_comparison.png'
        plt.savefig(response_time_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(str(response_time_path))
        
        # 3. Accuracy with Confidence Intervals
        if 'confidence_intervals' in self.comparison_result.statistical_tests:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models_list = []
            accuracies = []
            lower_bounds = []
            upper_bounds = []
            
            for model in models:
                ci_data = self.comparison_result.statistical_tests['confidence_intervals'].get(model, {})
                if 'accuracy' in ci_data:
                    models_list.append(model)
                    accuracies.append(ci_data['accuracy']['point_estimate'])
                    lower_bounds.append(ci_data['accuracy']['lower'])
                    upper_bounds.append(ci_data['accuracy']['upper'])
            
            if models_list:
                x = np.arange(len(models_list))
                
                # Calculate error bars
                yerr_lower = [acc - lower for acc, lower in zip(accuracies, lower_bounds)]
                yerr_upper = [upper - acc for acc, upper in zip(accuracies, upper_bounds)]
                yerr = [yerr_lower, yerr_upper]
                
                bars = ax.bar(x, accuracies, yerr=yerr, capsize=5, 
                             color='lightgreen', edgecolor='darkgreen', alpha=0.7,
                             error_kw={'linewidth': 2, 'ecolor': 'darkgreen'})
                
                # Add value labels
                for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                    ax.annotate(f'{acc:.2%}',
                               xy=(bar.get_x() + bar.get_width() / 2, acc),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
                
                ax.set_xlabel('Models')
                ax.set_ylabel('Accuracy')
                ax.set_title('Model Accuracy with 95% Confidence Intervals')
                ax.set_xticks(x)
                ax.set_xticklabels(models_list)
                ax.set_ylim(0, 1.1)
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                ci_path = output_dir / 'accuracy_confidence_intervals.png'
                plt.savefig(ci_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(str(ci_path))
        
        # 4. Heatmap of Pairwise Statistical Significance
        if 'pairwise_success_tests' in self.comparison_result.statistical_tests:
            # Create matrix for p-values
            n_models = len(models)
            p_value_matrix = np.ones((n_models, n_models))
            
            for pair_key, test_result in self.comparison_result.statistical_tests['pairwise_success_tests'].items():
                models_pair = pair_key.split('_vs_')
                if len(models_pair) == 2 and models_pair[0] in models and models_pair[1] in models:
                    i = models.index(models_pair[0])
                    j = models.index(models_pair[1])
                    p_value = test_result['p_value']
                    p_value_matrix[i, j] = p_value
                    p_value_matrix[j, i] = p_value
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(p_value_matrix, dtype=bool))
            
            # Create custom colormap (red for significant, blue for not significant)
            cmap = sns.diverging_palette(250, 10, as_cmap=True)
            
            sns.heatmap(p_value_matrix, mask=mask, annot=True, fmt='.4f',
                       xticklabels=models, yticklabels=models,
                       cmap=cmap, center=0.05, vmin=0, vmax=0.1,
                       square=True, linewidths=0.5,
                       cbar_kws={"shrink": 0.8, "label": "p-value"})
            
            ax.set_title('Pairwise Statistical Significance (p-values)\nRed indicates significant difference (p < 0.05)')
            
            plt.tight_layout()
            heatmap_path = output_dir / 'statistical_significance_heatmap.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(str(heatmap_path))
        
        # 5. Radar Chart for Multi-metric Comparison
        if len(models) <= 6:  # Radar charts work best with fewer models
            metrics_for_radar = ['overall_accuracy', 'success_rate', 'consistency_score']
            
            # Normalize response time (invert so higher is better)
            if 'average_response_time' in self.comparison_result.metrics:
                max_time = max(self.comparison_result.metrics['average_response_time'].values())
                if max_time > 0:
                    normalized_speed = {}
                    for model in models:
                        time = self.comparison_result.metrics['average_response_time'].get(model, max_time)
                        normalized_speed[model] = 1 - (time / max_time)
                    
                    # Add normalized speed as a metric
                    temp_metrics = self.comparison_result.metrics.copy()
                    temp_metrics['speed_score'] = normalized_speed
                    metrics_for_radar.append('speed_score')
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for model in models:
                values = []
                for metric in metrics_for_radar:
                    if metric == 'speed_score':
                        values.append(normalized_speed.get(model, 0))
                    else:
                        values.append(self.comparison_result.metrics[metric].get(model, 0))
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model)
                ax.fill(angles, values, alpha=0.15)
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_for_radar])
            ax.set_ylim(0, 1)
            ax.set_rlabel_position(30)
            ax.grid(True)
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            plt.title('Multi-Metric Model Comparison', size=20, y=1.08)
            
            radar_path = output_dir / 'model_comparison_radar.png'
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(str(radar_path))
        
        logger.info(f"Generated {len(generated_files)} visualization files in {output_dir}")
        return generated_files