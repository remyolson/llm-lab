#!/usr/bin/env python3
"""
Regression Testing Suite for LLM Performance Monitoring

This module provides a comprehensive regression testing framework for monitoring
LLM model performance over time, detecting drift, and maintaining quality baselines.

Key Features:
- Baseline performance establishment and tracking
- Automated drift detection with statistical tests
- Performance degradation alerts
- Historical trend analysis
- Multi-dimensional quality metrics
"""

import pytest
import json
import sqlite3
import statistics
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import logging
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import tempfile

# Import existing infrastructure
from src.providers.base import LLMProvider
from src.providers.openai import OpenAIProvider
from src.providers.anthropic import AnthropicProvider
from src.providers.google import GoogleProvider


# ===============================================================================
# DATA STRUCTURES FOR REGRESSION TESTING
# ===============================================================================

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for LLM responses."""
    response_time: float
    token_count: int
    cost: float
    accuracy_score: float = 0.0
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    factual_correctness: float = 0.0
    sentiment_appropriateness: float = 0.0
    length_appropriateness: float = 0.0
    grammar_score: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        quality_metrics = [
            self.accuracy_score,
            self.coherence_score, 
            self.relevance_score,
            self.factual_correctness,
            self.sentiment_appropriateness,
            self.length_appropriateness,
            self.grammar_score
        ]
        return statistics.mean([m for m in quality_metrics if m > 0])


@dataclass
class TestResult:
    """Result from a single regression test."""
    test_id: str
    provider: str
    model: str
    prompt: str
    response: str
    metrics: QualityMetrics
    timestamp: datetime = field(default_factory=datetime.now)
    test_version: str = "1.0"
    environment: str = "test"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result_dict = asdict(self)
        result_dict['timestamp'] = self.timestamp.isoformat()
        result_dict['metrics'] = asdict(self.metrics)
        return result_dict


@dataclass
class BaselineResult:
    """Baseline performance metrics for comparison."""
    test_id: str
    provider: str
    model: str
    baseline_metrics: QualityMetrics
    sample_size: int
    confidence_interval: Tuple[float, float]
    established_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"


@dataclass
class RegressionAlert:
    """Alert for performance regression detection."""
    test_id: str
    provider: str
    model: str
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_percentage: float
    severity: str  # 'warning', 'critical'
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return (f"REGRESSION ALERT [{self.severity.upper()}]: "
                f"{self.provider}/{self.model} - {self.metric_name} "
                f"degraded by {self.degradation_percentage:.1f}% "
                f"(current: {self.current_value:.3f}, baseline: {self.baseline_value:.3f})")


# ===============================================================================
# REGRESSION TESTING DATABASE
# ===============================================================================

class RegressionDatabase:
    """SQLite database for storing regression test results and baselines."""
    
    def __init__(self, db_path: str = "regression_results.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    token_count INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    accuracy_score REAL DEFAULT 0.0,
                    coherence_score REAL DEFAULT 0.0,
                    relevance_score REAL DEFAULT 0.0,
                    factual_correctness REAL DEFAULT 0.0,
                    sentiment_appropriateness REAL DEFAULT 0.0,
                    length_appropriateness REAL DEFAULT 0.0,
                    grammar_score REAL DEFAULT 0.0,
                    overall_score REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    test_version TEXT DEFAULT '1.0',
                    environment TEXT DEFAULT 'test'
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    baseline_value REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    confidence_lower REAL NOT NULL,
                    confidence_upper REAL NOT NULL,
                    established_date TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    version TEXT DEFAULT '1.0',
                    UNIQUE(test_id, provider, model, metric_name)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS regression_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    baseline_value REAL NOT NULL,
                    degradation_percentage REAL NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
    
    def store_result(self, result: TestResult):
        """Store a test result in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO test_results (
                    test_id, provider, model, prompt, response,
                    response_time, token_count, cost,
                    accuracy_score, coherence_score, relevance_score,
                    factual_correctness, sentiment_appropriateness,
                    length_appropriateness, grammar_score, overall_score,
                    timestamp, test_version, environment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.test_id, result.provider, result.model,
                result.prompt, result.response,
                result.metrics.response_time, result.metrics.token_count, result.metrics.cost,
                result.metrics.accuracy_score, result.metrics.coherence_score,
                result.metrics.relevance_score, result.metrics.factual_correctness,
                result.metrics.sentiment_appropriateness, result.metrics.length_appropriateness,
                result.metrics.grammar_score, result.metrics.overall_score(),
                result.timestamp.isoformat(), result.test_version, result.environment
            ))
    
    def get_recent_results(self, test_id: str, provider: str, model: str, 
                          days: int = 7) -> List[TestResult]:
        """Get recent test results for baseline comparison."""
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM test_results 
                WHERE test_id = ? AND provider = ? AND model = ? 
                AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (test_id, provider, model, since_date))
            
            results = []
            for row in cursor.fetchall():
                metrics = QualityMetrics(
                    response_time=row['response_time'],
                    token_count=row['token_count'],
                    cost=row['cost'],
                    accuracy_score=row['accuracy_score'],
                    coherence_score=row['coherence_score'],
                    relevance_score=row['relevance_score'],
                    factual_correctness=row['factual_correctness'],
                    sentiment_appropriateness=row['sentiment_appropriateness'],
                    length_appropriateness=row['length_appropriateness'],
                    grammar_score=row['grammar_score']
                )
                
                result = TestResult(
                    test_id=row['test_id'],
                    provider=row['provider'],
                    model=row['model'],
                    prompt=row['prompt'],
                    response=row['response'],
                    metrics=metrics,
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    test_version=row['test_version'],
                    environment=row['environment']
                )
                results.append(result)
            
            return results
    
    def store_baseline(self, baseline: BaselineResult):
        """Store or update baseline metrics."""
        metrics_to_store = [
            ('response_time', baseline.baseline_metrics.response_time),
            ('overall_score', baseline.baseline_metrics.overall_score()),
            ('accuracy_score', baseline.baseline_metrics.accuracy_score),
            ('coherence_score', baseline.baseline_metrics.coherence_score),
            ('cost', baseline.baseline_metrics.cost)
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for metric_name, value in metrics_to_store:
                conn.execute('''
                    INSERT OR REPLACE INTO baselines (
                        test_id, provider, model, metric_name, baseline_value,
                        sample_size, confidence_lower, confidence_upper,
                        established_date, last_updated, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    baseline.test_id, baseline.provider, baseline.model,
                    metric_name, value, baseline.sample_size,
                    baseline.confidence_interval[0], baseline.confidence_interval[1],
                    baseline.established_date.isoformat(),
                    baseline.last_updated.isoformat(),
                    baseline.version
                ))
    
    def get_baseline(self, test_id: str, provider: str, model: str, 
                    metric_name: str) -> Optional[Tuple[float, Tuple[float, float]]]:
        """Get baseline value and confidence interval for a metric."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT baseline_value, confidence_lower, confidence_upper
                FROM baselines
                WHERE test_id = ? AND provider = ? AND model = ? AND metric_name = ?
            ''', (test_id, provider, model, metric_name))
            
            row = cursor.fetchone()
            if row:
                return row[0], (row[1], row[2])
            return None
    
    def store_alert(self, alert: RegressionAlert):
        """Store regression alert."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO regression_alerts (
                    test_id, provider, model, metric_name,
                    current_value, baseline_value, degradation_percentage,
                    severity, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.test_id, alert.provider, alert.model, alert.metric_name,
                alert.current_value, alert.baseline_value, alert.degradation_percentage,
                alert.severity, alert.timestamp.isoformat()
            ))


# ===============================================================================
# QUALITY ASSESSMENT FUNCTIONS
# ===============================================================================

class QualityAssessor:
    """Assess various quality dimensions of LLM responses."""
    
    @staticmethod
    def assess_accuracy(prompt: str, response: str, expected_answer: str = None) -> float:
        """Assess response accuracy (simplified heuristic-based)."""
        if expected_answer:
            # Simple keyword matching for demonstration
            response_lower = response.lower()
            expected_lower = expected_answer.lower()
            if expected_lower in response_lower:
                return 1.0
            else:
                # Calculate similarity based on common words
                response_words = set(response_lower.split())
                expected_words = set(expected_lower.split())
                if expected_words:
                    overlap = len(response_words.intersection(expected_words))
                    return overlap / len(expected_words)
        
        # For prompts without expected answers, use length and coherence heuristics
        if len(response.strip()) == 0:
            return 0.0
        elif "error" in response.lower() or "sorry" in response.lower():
            return 0.3
        else:
            return 0.8  # Assume reasonable quality for demonstration
    
    @staticmethod
    def assess_coherence(response: str) -> float:
        """Assess response coherence based on structure and flow."""
        if len(response.strip()) == 0:
            return 0.0
        
        # Simple heuristics for coherence
        sentences = response.split('.')
        if len(sentences) < 1:
            return 0.2
        
        # Check for repeated words (incoherence indicator)
        words = response.lower().split()
        if len(words) == 0:
            return 0.0
        
        unique_words = len(set(words))
        word_diversity = unique_words / len(words)
        
        # Score based on diversity and reasonable length
        coherence_score = min(1.0, word_diversity * 1.5)
        if len(response) < 10:
            coherence_score *= 0.5
        
        return coherence_score
    
    @staticmethod
    def assess_relevance(prompt: str, response: str) -> float:
        """Assess how relevant the response is to the prompt."""
        if len(response.strip()) == 0:
            return 0.0
        
        # Simple keyword overlap analysis
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        if len(prompt_words) == 0:
            return 0.5
        
        overlap = len(prompt_words.intersection(response_words))
        relevance = overlap / len(prompt_words)
        
        # Boost score if response addresses the prompt type
        if "?" in prompt and ("yes" in response.lower() or "no" in response.lower() or "because" in response.lower()):
            relevance += 0.3
        
        return min(1.0, relevance)
    
    @staticmethod
    def assess_length_appropriateness(prompt: str, response: str) -> float:
        """Assess if response length is appropriate for the prompt."""
        prompt_len = len(prompt.split())
        response_len = len(response.split())
        
        if response_len == 0:
            return 0.0
        
        # Heuristic: good responses are usually 0.5x to 5x the prompt length
        ratio = response_len / max(prompt_len, 1)
        
        if 0.5 <= ratio <= 5.0:
            return 1.0
        elif 0.2 <= ratio <= 10.0:
            return 0.7
        else:
            return 0.3
    
    @staticmethod
    def assess_grammar(response: str) -> float:
        """Simple grammar assessment based on basic rules."""
        if len(response.strip()) == 0:
            return 0.0
        
        # Basic grammar checks
        score = 1.0
        
        # Check for proper sentence endings
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if sentences:
            for sentence in sentences:
                if len(sentence) > 0 and not sentence[0].isupper():
                    score -= 0.1
        
        # Check for basic punctuation
        if not any(punct in response for punct in '.!?'):
            score -= 0.3
        
        return max(0.0, score)


# ===============================================================================
# REGRESSION TESTING FRAMEWORK
# ===============================================================================

class RegressionTestSuite:
    """Main regression testing framework."""
    
    def __init__(self, db_path: str = "regression_results.db"):
        """Initialize regression test suite."""
        self.db = RegressionDatabase(db_path)
        self.assessor = QualityAssessor()
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for regression tests."""
        logger = logging.getLogger('regression_testing')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_test_case(self, test_id: str, provider: LLMProvider, 
                     prompt: str, expected_answer: str = None,
                     max_tokens: int = 150) -> TestResult:
        """Run a single test case and collect metrics."""
        import time
        
        start_time = time.time()
        try:
            response = provider.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
            response_time = time.time() - start_time
            
            # Assess quality metrics
            metrics = QualityMetrics(
                response_time=response_time,
                token_count=len(response.split()) * 2,  # Rough estimate
                cost=self._estimate_cost(provider, len(response.split()) * 2),
                accuracy_score=self.assessor.assess_accuracy(prompt, response, expected_answer),
                coherence_score=self.assessor.assess_coherence(response),
                relevance_score=self.assessor.assess_relevance(prompt, response),
                factual_correctness=0.8,  # Would need specialized tool
                sentiment_appropriateness=0.9,  # Would need sentiment analysis
                length_appropriateness=self.assessor.assess_length_appropriateness(prompt, response),
                grammar_score=self.assessor.assess_grammar(response)
            )
            
            result = TestResult(
                test_id=test_id,
                provider=provider.__class__.__name__,
                model=provider.model,
                prompt=prompt,
                response=response,
                metrics=metrics
            )
            
            # Store result
            self.db.store_result(result)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            error_response = f"Error: {str(e)}"
            
            metrics = QualityMetrics(
                response_time=response_time,
                token_count=0,
                cost=0.0,
                accuracy_score=0.0,
                coherence_score=0.0,
                relevance_score=0.0,
                factual_correctness=0.0,
                sentiment_appropriateness=0.0,
                length_appropriateness=0.0,
                grammar_score=0.0
            )
            
            result = TestResult(
                test_id=test_id,
                provider=provider.__class__.__name__,
                model=provider.model,
                prompt=prompt,
                response=error_response,
                metrics=metrics
            )
            
            self.db.store_result(result)
            return result
    
    def establish_baseline(self, test_id: str, provider: LLMProvider,
                          prompts: List[Tuple[str, str]], num_runs: int = 10) -> BaselineResult:
        """Establish baseline performance for a test case."""
        self.logger.info(f"Establishing baseline for {test_id} with {provider.__class__.__name__}")
        
        all_results = []
        for prompt, expected in prompts:
            for _ in range(num_runs):
                result = self.run_test_case(test_id, provider, prompt, expected)
                all_results.append(result)
        
        # Calculate baseline metrics
        response_times = [r.metrics.response_time for r in all_results]
        overall_scores = [r.metrics.overall_score() for r in all_results]
        
        # Calculate confidence intervals
        response_time_ci = self._calculate_confidence_interval(response_times)
        overall_score_ci = self._calculate_confidence_interval(overall_scores)
        
        baseline_metrics = QualityMetrics(
            response_time=statistics.mean(response_times),
            token_count=int(statistics.mean([r.metrics.token_count for r in all_results])),
            cost=statistics.mean([r.metrics.cost for r in all_results]),
            accuracy_score=statistics.mean([r.metrics.accuracy_score for r in all_results]),
            coherence_score=statistics.mean([r.metrics.coherence_score for r in all_results]),
            relevance_score=statistics.mean([r.metrics.relevance_score for r in all_results]),
            factual_correctness=statistics.mean([r.metrics.factual_correctness for r in all_results]),
            sentiment_appropriateness=statistics.mean([r.metrics.sentiment_appropriateness for r in all_results]),
            length_appropriateness=statistics.mean([r.metrics.length_appropriateness for r in all_results]),
            grammar_score=statistics.mean([r.metrics.grammar_score for r in all_results])
        )
        
        baseline = BaselineResult(
            test_id=test_id,
            provider=provider.__class__.__name__,
            model=provider.model,
            baseline_metrics=baseline_metrics,
            sample_size=len(all_results),
            confidence_interval=overall_score_ci
        )
        
        self.db.store_baseline(baseline)
        self.logger.info(f"Baseline established: Overall Score = {baseline_metrics.overall_score():.3f}")
        
        return baseline
    
    def check_regression(self, test_id: str, provider: LLMProvider,
                        prompt: str, expected_answer: str = None,
                        alert_threshold: float = 0.1) -> List[RegressionAlert]:
        """Check for performance regression against baseline."""
        # Run current test
        current_result = self.run_test_case(test_id, provider, prompt, expected_answer)
        
        # Check against baselines
        alerts = []
        metrics_to_check = [
            ('response_time', current_result.metrics.response_time, 'lower_is_better'),
            ('overall_score', current_result.metrics.overall_score(), 'higher_is_better'),
            ('accuracy_score', current_result.metrics.accuracy_score, 'higher_is_better'),
            ('coherence_score', current_result.metrics.coherence_score, 'higher_is_better'),
            ('cost', current_result.metrics.cost, 'lower_is_better')
        ]
        
        for metric_name, current_value, direction in metrics_to_check:
            baseline_data = self.db.get_baseline(
                test_id, 
                provider.__class__.__name__, 
                provider.model, 
                metric_name
            )
            
            if baseline_data:
                baseline_value, confidence_interval = baseline_data
                
                # Calculate degradation
                if direction == 'higher_is_better':
                    degradation = (baseline_value - current_value) / baseline_value
                    is_regression = current_value < (baseline_value * (1 - alert_threshold))
                else:  # lower_is_better
                    degradation = (current_value - baseline_value) / baseline_value  
                    is_regression = current_value > (baseline_value * (1 + alert_threshold))
                
                if is_regression:
                    severity = 'critical' if abs(degradation) > 0.25 else 'warning'
                    
                    alert = RegressionAlert(
                        test_id=test_id,
                        provider=provider.__class__.__name__,
                        model=provider.model,
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        degradation_percentage=abs(degradation) * 100,
                        severity=severity
                    )
                    
                    alerts.append(alert)
                    self.db.store_alert(alert)
                    self.logger.warning(str(alert))
        
        return alerts
    
    def _calculate_confidence_interval(self, values: List[float], 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values."""
        if len(values) < 2:
            return (0.0, 0.0)
        
        mean_val = statistics.mean(values)
        std_err = statistics.stdev(values) / (len(values) ** 0.5)
        
        # Use t-distribution for small samples
        from scipy.stats import t
        t_val = t.ppf((1 + confidence) / 2, len(values) - 1)
        margin = t_val * std_err
        
        return (mean_val - margin, mean_val + margin)
    
    def _estimate_cost(self, provider: LLMProvider, token_count: int) -> float:
        """Estimate cost based on provider and token count."""
        cost_per_token = {
            'OpenAIProvider': 0.00002,
            'AnthropicProvider': 0.00001,
            'GoogleProvider': 0.000005
        }
        
        provider_name = provider.__class__.__name__
        return token_count * cost_per_token.get(provider_name, 0.00001)
    
    def generate_trend_report(self, test_id: str, provider: str, model: str, 
                            days: int = 30) -> Dict[str, Any]:
        """Generate trend analysis report."""
        results = self.db.get_recent_results(test_id, provider, model, days)
        
        if not results:
            return {"error": "No results found for the specified parameters"}
        
        # Group by date
        daily_metrics = defaultdict(list)
        for result in results:
            date_key = result.timestamp.date().isoformat()
            daily_metrics[date_key].append(result.metrics)
        
        # Calculate daily averages
        trend_data = {}
        for date, metrics_list in daily_metrics.items():
            trend_data[date] = {
                'avg_response_time': statistics.mean([m.response_time for m in metrics_list]),
                'avg_overall_score': statistics.mean([m.overall_score() for m in metrics_list]),
                'avg_cost': statistics.mean([m.cost for m in metrics_list]),
                'sample_count': len(metrics_list)
            }
        
        return {
            'test_id': test_id,
            'provider': provider,
            'model': model,
            'period_days': days,
            'total_samples': len(results),
            'trend_data': trend_data,
            'generated_at': datetime.now().isoformat()
        }


# ===============================================================================
# TEST FIXTURES AND EXAMPLES
# ===============================================================================

@pytest.fixture
def regression_suite():
    """Fixture for regression test suite with temporary database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        suite = RegressionTestSuite(tmp_file.name)
        yield suite
        # Cleanup
        os.unlink(tmp_file.name)


@pytest.fixture
def sample_test_cases():
    """Sample test cases for regression testing."""
    return [
        ("math_basic", "What is 2 + 2?", "4"),
        ("greeting", "Hello, how are you?", None),
        ("code_simple", "Write a function to add two numbers in Python", None),
        ("reasoning", "Why is testing important?", None),
        ("factual", "What is the capital of France?", "Paris")
    ]


# ===============================================================================
# REGRESSION TEST EXAMPLES
# ===============================================================================

class TestRegressionSuite:
    """Test cases for the regression testing framework."""
    
    def test_baseline_establishment(self, regression_suite, sample_test_cases):
        """Test baseline establishment process."""
        # Mock provider
        from unittest.mock import Mock
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.model = "test-model"
        mock_provider.__class__.__name__ = "TestProvider"
        mock_provider.generate.return_value = "This is a test response."
        
        # Establish baseline for first test case
        test_id, prompt, expected = sample_test_cases[0]
        
        baseline = regression_suite.establish_baseline(
            test_id=test_id,
            provider=mock_provider,
            prompts=[(prompt, expected)],
            num_runs=3
        )
        
        assert baseline.test_id == test_id
        assert baseline.provider == "TestProvider"
        assert baseline.sample_size == 3
        assert baseline.baseline_metrics.overall_score() > 0
    
    def test_regression_detection(self, regression_suite):
        """Test regression detection functionality."""
        from unittest.mock import Mock
        
        # Create mock provider
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.model = "test-model"
        mock_provider.__class__.__name__ = "TestProvider"
        
        # First, establish baseline with good responses
        mock_provider.generate.return_value = "4"  # Good response
        baseline = regression_suite.establish_baseline(
            test_id="math_test",
            provider=mock_provider,
            prompts=[("What is 2 + 2?", "4")],
            num_runs=5
        )
        
        # Now simulate degraded performance
        mock_provider.generate.return_value = "I don't know"  # Poor response
        
        alerts = regression_suite.check_regression(
            test_id="math_test",
            provider=mock_provider,
            prompt="What is 2 + 2?",
            expected_answer="4",
            alert_threshold=0.1
        )
        
        # Should detect regression in accuracy
        assert len(alerts) > 0
        accuracy_alerts = [a for a in alerts if a.metric_name == 'accuracy_score']
        assert len(accuracy_alerts) > 0
    
    def test_trend_analysis(self, regression_suite):
        """Test trend analysis functionality."""
        from unittest.mock import Mock
        
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.model = "test-model"
        mock_provider.__class__.__name__ = "TestProvider"
        mock_provider.generate.return_value = "Test response"
        
        # Generate some test data
        for _ in range(5):
            regression_suite.run_test_case(
                test_id="trend_test",
                provider=mock_provider,
                prompt="Test prompt"
            )
        
        # Generate trend report
        report = regression_suite.generate_trend_report(
            test_id="trend_test",
            provider="TestProvider", 
            model="test-model",
            days=1
        )
        
        assert 'trend_data' in report
        assert report['total_samples'] == 5
    
    @pytest.mark.integration  
    @pytest.mark.skipif(not os.getenv('INTEGRATION_TESTS'), reason="Integration tests disabled")
    def test_real_provider_regression(self, regression_suite):
        """Integration test with real providers."""
        providers_to_test = []
        
        # Try to initialize real providers
        if os.getenv('OPENAI_API_KEY'):
            try:
                provider = OpenAIProvider(model='gpt-4o-mini')
                provider.initialize()
                providers_to_test.append(provider)
            except Exception:
                pass
        
        if not providers_to_test:
            pytest.skip("No real providers available for testing")
        
        provider = providers_to_test[0]
        
        # Establish baseline
        baseline = regression_suite.establish_baseline(
            test_id="integration_math",
            provider=provider,
            prompts=[("What is 5 + 3?", "8")],
            num_runs=3
        )
        
        assert baseline.baseline_metrics.overall_score() > 0.5
        
        # Check for regression (should be minimal with same test)
        alerts = regression_suite.check_regression(
            test_id="integration_math",
            provider=provider,
            prompt="What is 5 + 3?",
            expected_answer="8"
        )
        
        # Should have few or no alerts for identical test
        critical_alerts = [a for a in alerts if a.severity == 'critical']
        assert len(critical_alerts) == 0


if __name__ == "__main__":
    # Example usage
    print("Regression Testing Suite Example")
    print("="*40)
    
    # Create mock provider for demonstration
    from unittest.mock import Mock
    mock_provider = Mock(spec=LLMProvider)
    mock_provider.model = "demo-model"
    mock_provider.__class__.__name__ = "DemoProvider"
    mock_provider.generate.return_value = "2 + 2 equals 4"
    
    # Initialize regression suite
    suite = RegressionTestSuite("demo_regression.db")
    
    # Establish baseline
    print("Establishing baseline...")
    baseline = suite.establish_baseline(
        test_id="demo_math",
        provider=mock_provider,
        prompts=[("What is 2 + 2?", "4")],
        num_runs=5
    )
    
    print(f"Baseline Overall Score: {baseline.baseline_metrics.overall_score():.3f}")
    
    # Test for regression
    print("\nTesting for regression...")
    alerts = suite.check_regression(
        test_id="demo_math",
        provider=mock_provider,
        prompt="What is 2 + 2?",
        expected_answer="4"
    )
    
    if alerts:
        print(f"Found {len(alerts)} regression alerts:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("No regressions detected!")
    
    print("\nTo run the full test suite:")
    print("pytest examples/use_cases/regression_testing_suite.py -v")
    
    # Cleanup demo database
    os.unlink("demo_regression.db")