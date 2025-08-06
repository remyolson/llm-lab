"""
Visual Performance Analytics

This module provides comprehensive visual analytics for model performance tracking,
including real-time dashboards, response comparison, task evaluation, behavior
analysis, and regression detection.
"""

from .analysis.behavior_analyzer import BehaviorAnalysis, ModelBehaviorAnalyzer
from .app import VisualAnalyticsApp
from .comparison.response_analyzer import ResponseComparison, ResponseComparisonView
from .dashboard.metrics_dashboard import MetricsDashboard, TrainingMetrics, TrainingRun
from .evaluation.task_evaluator import EvaluationResult, TaskEvaluationPanel, TaskType
from .monitoring.regression_detector import RegressionAlert, RegressionDetector

__all__ = [
    "VisualAnalyticsApp",
    "MetricsDashboard",
    "TrainingMetrics",
    "TrainingRun",
    "ResponseComparisonView",
    "ResponseComparison",
    "TaskEvaluationPanel",
    "TaskType",
    "EvaluationResult",
    "ModelBehaviorAnalyzer",
    "BehaviorAnalysis",
    "RegressionDetector",
    "RegressionAlert",
]
