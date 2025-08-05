"""
Performance regression detection system.

This module provides statistical analysis capabilities to detect significant
performance regressions in model benchmarks using various statistical methods
and change point detection algorithms.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics
from scipy import stats
import pandas as pd

from .database import DatabaseManager
from .models import PerformanceMetric, BenchmarkRun

logger = logging.getLogger(__name__)


class RegressionType(Enum):
    """Types of performance regressions that can be detected."""
    ACCURACY_DROP = "accuracy_drop"
    LATENCY_INCREASE = "latency_increase"
    COST_INCREASE = "cost_increase"
    ERROR_RATE_INCREASE = "error_rate_increase"
    THROUGHPUT_DROP = "throughput_drop"


class DetectionMethod(Enum):
    """Statistical methods for regression detection."""
    THRESHOLD_BASED = "threshold_based"
    STATISTICAL_TEST = "statistical_test"
    CHANGE_POINT = "change_point"
    ROLLING_AVERAGE = "rolling_average"
    PERCENTILE_BASED = "percentile_based"


@dataclass
class RegressionConfig:
    """Configuration for regression detection."""
    method: DetectionMethod
    metric_type: str
    metric_name: str
    threshold_percent: Optional[float] = None  # For threshold-based detection
    statistical_confidence: float = 0.95  # For statistical tests
    window_size: int = 10  # For rolling average and change point
    baseline_days: int = 7  # Days to use as baseline
    min_data_points: int = 5  # Minimum data points required
    severity_thresholds: Dict[str, float] = None  # Custom severity levels


@dataclass
class RegressionResult:
    """Result of regression detection analysis."""
    metric_type: str
    metric_name: str
    model_id: int
    detection_method: DetectionMethod
    regression_detected: bool
    severity: str  # 'critical', 'warning', 'info'
    confidence_score: float
    baseline_value: float
    current_value: float
    change_percent: float
    p_value: Optional[float] = None
    change_point_index: Optional[int] = None
    statistical_details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class RegressionDetector:
    """Detects performance regressions in benchmark results."""
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize regression detector.
        
        Args:
            database_manager: Database manager for accessing historical data
        """
        self.db_manager = database_manager
        
        # Default configurations for different metric types
        self.default_configs = {
            'accuracy': RegressionConfig(
                method=DetectionMethod.STATISTICAL_TEST,
                metric_type='accuracy',
                metric_name='overall_accuracy',
                statistical_confidence=0.95,
                severity_thresholds={'critical': 0.05, 'warning': 0.02}
            ),
            'latency': RegressionConfig(
                method=DetectionMethod.THRESHOLD_BASED,
                metric_type='latency',
                metric_name='avg_latency_ms',
                threshold_percent=0.2,  # 20% increase
                severity_thresholds={'critical': 0.5, 'warning': 0.2}
            ),
            'cost': RegressionConfig(
                method=DetectionMethod.ROLLING_AVERAGE,
                metric_type='cost',
                metric_name='total_cost',
                window_size=5,
                threshold_percent=0.15,  # 15% increase
                severity_thresholds={'critical': 0.3, 'warning': 0.15}
            )
        }
        
        logger.info("Regression detector initialized")
    
    def detect_regressions(
        self,
        model_id: int,
        configs: Optional[List[RegressionConfig]] = None,
        days_back: int = 30
    ) -> List[RegressionResult]:
        """
        Detect regressions for a model across multiple metrics.
        
        Args:
            model_id: ID of the model to analyze
            configs: List of detection configurations (uses defaults if None)
            days_back: Number of days of history to analyze
            
        Returns:
            List of regression detection results
        """
        if configs is None:
            configs = list(self.default_configs.values())
        
        results = []
        
        for config in configs:
            try:
                result = self._detect_single_metric_regression(
                    model_id, config, days_back
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(
                    f"Error detecting regression for {config.metric_name}: {e}"
                )
        
        return results
    
    def _detect_single_metric_regression(
        self,
        model_id: int,
        config: RegressionConfig,
        days_back: int
    ) -> Optional[RegressionResult]:
        """Detect regression for a single metric."""
        # Get historical data
        metrics = self.db_manager.get_metric_history(
            model_id=model_id,
            metric_type=config.metric_type,
            metric_name=config.metric_name,
            days=days_back
        )
        
        if len(metrics) < config.min_data_points:
            logger.warning(
                f"Insufficient data for {config.metric_name}: "
                f"{len(metrics)} < {config.min_data_points}"
            )
            return None
        
        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        
        # Apply the appropriate detection method
        if config.method == DetectionMethod.THRESHOLD_BASED:
            return self._threshold_based_detection(values, timestamps, config, model_id)
        elif config.method == DetectionMethod.STATISTICAL_TEST:
            return self._statistical_test_detection(values, timestamps, config, model_id)
        elif config.method == DetectionMethod.CHANGE_POINT:
            return self._change_point_detection(values, timestamps, config, model_id)
        elif config.method == DetectionMethod.ROLLING_AVERAGE:
            return self._rolling_average_detection(values, timestamps, config, model_id)
        elif config.method == DetectionMethod.PERCENTILE_BASED:
            return self._percentile_based_detection(values, timestamps, config, model_id)
        else:
            logger.error(f"Unknown detection method: {config.method}")
            return None
    
    def _threshold_based_detection(
        self,
        values: List[float],
        timestamps: List[datetime],
        config: RegressionConfig,
        model_id: int
    ) -> RegressionResult:
        """Simple threshold-based regression detection."""
        # Split into baseline and recent periods
        split_point = len(values) - max(1, len(values) // 3)
        baseline_values = values[:split_point]
        recent_values = values[split_point:]
        
        baseline_mean = statistics.mean(baseline_values)
        recent_mean = statistics.mean(recent_values)
        
        # Calculate change percentage
        if baseline_mean != 0:
            change_percent = (recent_mean - baseline_mean) / abs(baseline_mean)
        else:
            change_percent = 0.0
        
        # Determine if this is a regression (depends on metric type)
        is_regression = self._is_regression(
            config.metric_type, change_percent, config.threshold_percent
        )
        
        # Determine severity
        severity = self._calculate_severity(change_percent, config)
        
        # Calculate confidence score
        confidence = min(abs(change_percent) / config.threshold_percent, 1.0) if config.threshold_percent else 0.5
        
        return RegressionResult(
            metric_type=config.metric_type,
            metric_name=config.metric_name,
            model_id=model_id,
            detection_method=config.method,
            regression_detected=is_regression,
            severity=severity,
            confidence_score=confidence,
            baseline_value=baseline_mean,
            current_value=recent_mean,
            change_percent=change_percent
        )
    
    def _statistical_test_detection(
        self,
        values: List[float],
        timestamps: List[datetime],
        config: RegressionConfig,
        model_id: int
    ) -> RegressionResult:
        """Statistical test-based regression detection using t-test."""
        # Split into baseline and recent periods
        split_point = len(values) - max(1, len(values) // 3)
        baseline_values = values[:split_point]
        recent_values = values[split_point:]
        
        baseline_mean = statistics.mean(baseline_values)
        recent_mean = statistics.mean(recent_values)
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(baseline_values, recent_values)
        except Exception as e:
            logger.error(f"T-test failed: {e}")
            p_value = 1.0
            t_stat = 0.0
        
        # Calculate change percentage
        if baseline_mean != 0:
            change_percent = (recent_mean - baseline_mean) / abs(baseline_mean)
        else:
            change_percent = 0.0
        
        # Regression detected if p-value is significant and change is in wrong direction
        alpha = 1 - config.statistical_confidence
        is_significant = p_value < alpha
        is_regression = is_significant and self._is_regression(
            config.metric_type, change_percent, 0.01  # 1% minimum change
        )
        
        # Determine severity
        severity = self._calculate_severity(change_percent, config)
        
        # Confidence score based on p-value
        confidence = 1 - p_value if is_significant else 0.0
        
        return RegressionResult(
            metric_type=config.metric_type,
            metric_name=config.metric_name,
            model_id=model_id,
            detection_method=config.method,
            regression_detected=is_regression,
            severity=severity,
            confidence_score=confidence,
            baseline_value=baseline_mean,
            current_value=recent_mean,
            change_percent=change_percent,
            p_value=p_value,
            statistical_details={
                't_statistic': t_stat,
                'degrees_of_freedom': len(baseline_values) + len(recent_values) - 2,
                'alpha': alpha
            }
        )
    
    def _change_point_detection(
        self,
        values: List[float],
        timestamps: List[datetime],
        config: RegressionConfig,
        model_id: int
    ) -> RegressionResult:
        """Change point detection using CUSUM algorithm."""
        if len(values) < config.window_size:
            return self._threshold_based_detection(values, timestamps, config, model_id)
        
        # Simple CUSUM implementation
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 1.0
        
        # Normalize values
        normalized = [(v - mean_val) / std_val for v in values]
        
        # CUSUM parameters
        threshold = 2.0  # Detection threshold
        drift = 0.5     # Expected drift
        
        cusum_pos = 0
        cusum_neg = 0
        change_point = None
        
        for i, val in enumerate(normalized):
            cusum_pos = max(0, cusum_pos + val - drift)
            cusum_neg = max(0, cusum_neg - val - drift)
            
            if cusum_pos > threshold or cusum_neg > threshold:
                change_point = i
                break
        
        # Calculate baseline and current values
        if change_point and change_point < len(values) - 1:
            baseline_values = values[:change_point]
            recent_values = values[change_point:]
        else:
            # Fall back to splitting in half
            split_point = len(values) // 2
            baseline_values = values[:split_point]
            recent_values = values[split_point:]
        
        baseline_mean = statistics.mean(baseline_values)
        recent_mean = statistics.mean(recent_values)
        
        # Calculate change percentage
        if baseline_mean != 0:
            change_percent = (recent_mean - baseline_mean) / abs(baseline_mean)
        else:
            change_percent = 0.0
        
        # Determine if this is a regression
        is_regression = change_point is not None and self._is_regression(
            config.metric_type, change_percent, 0.05  # 5% minimum change
        )
        
        # Determine severity
        severity = self._calculate_severity(change_percent, config)
        
        # Confidence score based on change magnitude
        confidence = min(abs(change_percent), 1.0) if change_point else 0.0
        
        return RegressionResult(
            metric_type=config.metric_type,
            metric_name=config.metric_name,
            model_id=model_id,
            detection_method=config.method,
            regression_detected=is_regression,
            severity=severity,
            confidence_score=confidence,
            baseline_value=baseline_mean,
            current_value=recent_mean,
            change_percent=change_percent,
            change_point_index=change_point,
            statistical_details={
                'cusum_threshold': threshold,
                'drift_parameter': drift,
                'normalized_values': normalized[-5:]  # Last 5 for debugging
            }
        )
    
    def _rolling_average_detection(
        self,
        values: List[float],
        timestamps: List[datetime],
        config: RegressionConfig,
        model_id: int
    ) -> RegressionResult:
        """Rolling average-based regression detection."""
        window_size = min(config.window_size, len(values) // 2)
        
        if len(values) < window_size * 2:
            return self._threshold_based_detection(values, timestamps, config, model_id)
        
        # Calculate rolling averages
        rolling_avgs = []
        for i in range(window_size, len(values) + 1):
            window_values = values[i-window_size:i]
            rolling_avgs.append(statistics.mean(window_values))
        
        # Compare first and last rolling averages
        baseline_avg = rolling_avgs[0]
        current_avg = rolling_avgs[-1]
        
        # Calculate change percentage
        if baseline_avg != 0:
            change_percent = (current_avg - baseline_avg) / abs(baseline_avg)
        else:
            change_percent = 0.0
        
        # Determine if this is a regression
        is_regression = self._is_regression(
            config.metric_type, change_percent, config.threshold_percent
        )
        
        # Determine severity
        severity = self._calculate_severity(change_percent, config)
        
        # Calculate confidence based on trend consistency
        trend_direction = 1 if current_avg > baseline_avg else -1
        consistent_trend = 0
        
        for i in range(1, len(rolling_avgs)):
            if (rolling_avgs[i] - rolling_avgs[i-1]) * trend_direction > 0:
                consistent_trend += 1
        
        confidence = consistent_trend / (len(rolling_avgs) - 1) if len(rolling_avgs) > 1 else 0.0
        
        return RegressionResult(
            metric_type=config.metric_type,
            metric_name=config.metric_name,
            model_id=model_id,
            detection_method=config.method,
            regression_detected=is_regression,
            severity=severity,
            confidence_score=confidence,
            baseline_value=baseline_avg,
            current_value=current_avg,
            change_percent=change_percent,
            statistical_details={
                'rolling_window_size': window_size,
                'rolling_averages': rolling_avgs[-5:],  # Last 5 values
                'trend_consistency': confidence
            }
        )
    
    def _percentile_based_detection(
        self,
        values: List[float],
        timestamps: List[datetime],
        config: RegressionConfig,
        model_id: int
    ) -> RegressionResult:
        """Percentile-based regression detection."""
        # Calculate percentiles for baseline period
        baseline_days = config.baseline_days
        baseline_cutoff = timestamps[-1] - timedelta(days=baseline_days)
        
        baseline_values = []
        recent_values = []
        
        for i, ts in enumerate(timestamps):
            if ts <= baseline_cutoff:
                baseline_values.append(values[i])
            else:
                recent_values.append(values[i])
        
        if len(baseline_values) < 3 or len(recent_values) < 1:
            return self._threshold_based_detection(values, timestamps, config, model_id)
        
        # Calculate percentiles for baseline
        baseline_50th = np.percentile(baseline_values, 50)
        baseline_90th = np.percentile(baseline_values, 90)
        baseline_10th = np.percentile(baseline_values, 10)
        
        current_value = statistics.mean(recent_values)
        
        # Determine if current value is outside normal range
        is_regression = False
        severity = 'info'
        
        if config.metric_type in ['accuracy', 'throughput']:
            # For metrics where higher is better
            if current_value < baseline_10th:
                is_regression = True
                severity = 'critical'
            elif current_value < baseline_50th:
                is_regression = True
                severity = 'warning'
        else:
            # For metrics where lower is better (latency, cost, error_rate)
            if current_value > baseline_90th:
                is_regression = True
                severity = 'critical'
            elif current_value > baseline_50th:
                is_regression = True
                severity = 'warning'
        
        # Calculate change percentage from median
        if baseline_50th != 0:
            change_percent = (current_value - baseline_50th) / abs(baseline_50th)
        else:
            change_percent = 0.0
        
        # Confidence based on how far outside normal range
        if is_regression:
            if severity == 'critical':
                confidence = 0.9
            else:
                confidence = 0.7
        else:
            confidence = 0.1
        
        return RegressionResult(
            metric_type=config.metric_type,
            metric_name=config.metric_name,
            model_id=model_id,
            detection_method=config.method,
            regression_detected=is_regression,
            severity=severity,
            confidence_score=confidence,
            baseline_value=baseline_50th,
            current_value=current_value,
            change_percent=change_percent,
            statistical_details={
                'baseline_10th_percentile': baseline_10th,
                'baseline_50th_percentile': baseline_50th,
                'baseline_90th_percentile': baseline_90th,
                'baseline_sample_size': len(baseline_values),
                'recent_sample_size': len(recent_values)
            }
        )
    
    def _is_regression(
        self,
        metric_type: str,
        change_percent: float,
        threshold: float
    ) -> bool:
        """Determine if a change represents a regression."""
        if abs(change_percent) < threshold:
            return False
        
        # For accuracy, throughput: regression is decrease
        if metric_type in ['accuracy', 'precision', 'recall', 'f1_score', 'throughput']:
            return change_percent < -threshold
        
        # For latency, cost, error_rate: regression is increase
        if metric_type in ['latency', 'cost', 'error_rate', 'memory_usage']:
            return change_percent > threshold
        
        # Default: any significant change is potentially a regression
        return abs(change_percent) > threshold
    
    def _calculate_severity(
        self,
        change_percent: float,
        config: RegressionConfig
    ) -> str:
        """Calculate severity level based on change magnitude."""
        if config.severity_thresholds:
            abs_change = abs(change_percent)
            if abs_change >= config.severity_thresholds.get('critical', 0.2):
                return 'critical'
            elif abs_change >= config.severity_thresholds.get('warning', 0.1):
                return 'warning'
        
        return 'info'
    
    def analyze_model_trends(
        self,
        model_id: int,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze overall trends for a model across all metrics.
        
        Args:
            model_id: ID of the model to analyze
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis results
        """
        # Get all recent benchmark runs for the model
        runs = self.db_manager.list_benchmark_runs(
            model_id=model_id,
            limit=1000
        )
        
        if not runs:
            return {'error': 'No benchmark runs found for model'}
        
        # Group metrics by type
        metric_trends = {}
        
        for run in runs:
            if (datetime.utcnow() - run.timestamp).days > days_back:
                continue
            
            metrics = self.db_manager.get_metrics_for_run(run.id)
            
            for metric in metrics:
                key = f"{metric.metric_type}_{metric.metric_name}"
                if key not in metric_trends:
                    metric_trends[key] = []
                
                metric_trends[key].append({
                    'value': metric.value,
                    'timestamp': metric.timestamp,
                    'run_id': run.id
                })
        
        # Analyze each metric trend
        trend_analysis = {}
        
        for metric_key, data_points in metric_trends.items():
            if len(data_points) < 3:
                continue
            
            # Sort by timestamp
            data_points.sort(key=lambda x: x['timestamp'])
            values = [dp['value'] for dp in data_points]
            
            # Calculate trend using linear regression
            x = list(range(len(values)))
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                trend_analysis[metric_key] = {
                    'data_points': len(values),
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                    'trend_strength': abs(r_value),
                    'first_value': values[0],
                    'last_value': values[-1],
                    'change_percent': (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else 0,
                    'timestamps': {
                        'first': data_points[0]['timestamp'].isoformat(),
                        'last': data_points[-1]['timestamp'].isoformat()
                    }
                }
            except Exception as e:
                logger.error(f"Error calculating trend for {metric_key}: {e}")
        
        return {
            'model_id': model_id,
            'analysis_period_days': days_back,
            'total_runs_analyzed': len(runs),
            'metrics_analyzed': len(trend_analysis),
            'metric_trends': trend_analysis
        }
    
    def get_regression_summary(
        self,
        model_ids: Optional[List[int]] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Get a summary of detected regressions across models.
        
        Args:
            model_ids: List of model IDs to analyze (all if None)
            days_back: Number of days to analyze
            
        Returns:
            Summary of regression detection results
        """
        if model_ids is None:
            models = self.db_manager.list_models(active_only=True)
            model_ids = [m.id for m in models]
        
        all_results = []
        summary_stats = {
            'total_models_analyzed': len(model_ids),
            'total_regressions_detected': 0,
            'regressions_by_severity': {'critical': 0, 'warning': 0, 'info': 0},
            'regressions_by_metric_type': {},
            'models_with_regressions': 0
        }
        
        for model_id in model_ids:
            try:
                results = self.detect_regressions(model_id, days_back=days_back)
                all_results.extend(results)
                
                model_has_regression = False
                for result in results:
                    if result.regression_detected:
                        summary_stats['total_regressions_detected'] += 1
                        summary_stats['regressions_by_severity'][result.severity] += 1
                        
                        metric_type = result.metric_type
                        if metric_type not in summary_stats['regressions_by_metric_type']:
                            summary_stats['regressions_by_metric_type'][metric_type] = 0
                        summary_stats['regressions_by_metric_type'][metric_type] += 1
                        
                        model_has_regression = True
                
                if model_has_regression:
                    summary_stats['models_with_regressions'] += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing model {model_id}: {e}")
        
        return {
            'summary': summary_stats,
            'detailed_results': [
                {
                    'model_id': r.model_id,
                    'metric_type': r.metric_type,
                    'metric_name': r.metric_name,
                    'regression_detected': r.regression_detected,
                    'severity': r.severity,
                    'confidence_score': r.confidence_score,
                    'change_percent': r.change_percent,
                    'detection_method': r.detection_method.value
                }
                for r in all_results if r.regression_detected
            ]
        }