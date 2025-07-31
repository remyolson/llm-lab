"""
Statistical analysis and insights for performance benchmarks

This module provides advanced statistical analysis capabilities for benchmark results,
including trend analysis, performance comparisons, and actionable insights.
"""

import statistics
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from .benchmark_suite import BenchmarkResult, BenchmarkMetrics
from .benchmark_config import BenchmarkConfig, rate_performance

logger = logging.getLogger(__name__)


@dataclass
class StatisticalSummary:
    """Statistical summary for a set of measurements."""
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentiles: Dict[int, float]
    sample_size: int
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class PerformanceInsight:
    """A single performance insight with supporting data."""
    category: str  # 'warning', 'recommendation', 'observation'
    title: str
    description: str
    impact: str  # 'high', 'medium', 'low'
    supporting_data: Dict[str, Any]
    recommendation: Optional[str] = None


class PerformanceAnalyzer:
    """
    Advanced statistical analysis for performance benchmark results.
    
    Features:
    - Statistical analysis with confidence intervals
    - Performance trend detection
    - Outlier detection and analysis
    - Cross-provider performance comparison
    - Automated insight generation
    - Performance regression detection
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the performance analyzer.
        
        Args:
            confidence_level: Statistical confidence level for intervals
        """
        self.confidence_level = confidence_level
        self.insights: List[PerformanceInsight] = []
        
    def analyze_results(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of benchmark results.
        
        Args:
            results: Dictionary of benchmark results
            
        Returns:
            Dictionary containing analysis results and insights
        """
        logger.info("Starting comprehensive performance analysis...")
        
        analysis = {
            'summary': self._generate_summary_statistics(results),
            'provider_comparison': self._analyze_provider_performance(results),
            'test_category_analysis': self._analyze_test_categories(results),
            'outlier_analysis': self._detect_outliers(results),
            'performance_insights': self._generate_insights(results),
            'recommendations': self._generate_recommendations(results),
            'statistical_significance': self._test_statistical_significance(results)
        }
        
        logger.info(f"Analysis completed. Generated {len(self.insights)} insights.")
        return analysis
    
    def _generate_summary_statistics(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Generate overall summary statistics."""
        if not results:
            return {}
        
        # Collect all metrics
        all_metrics = []
        for result in results.values():
            all_metrics.extend(result.metrics)
        
        successful_metrics = [m for m in all_metrics if m.success]
        
        if not successful_metrics:
            return {'error': 'No successful metrics found'}
        
        # Response time analysis
        response_times = [m.response_time for m in successful_metrics]
        response_stats = self._calculate_statistical_summary(response_times)
        
        # Success rate analysis
        success_rates = []
        for result in results.values():
            success_rates.append(result.success_rate)
        
        success_stats = self._calculate_statistical_summary(success_rates)
        
        # Token efficiency analysis
        token_rates = [m.tokens_per_second for m in successful_metrics if m.tokens_per_second]
        token_stats = self._calculate_statistical_summary(token_rates) if token_rates else None
        
        return {
            'total_tests': len(results),
            'total_requests': len(all_metrics),
            'successful_requests': len(successful_metrics),
            'overall_success_rate': (len(successful_metrics) / len(all_metrics)) * 100,
            'response_time_stats': response_stats,
            'success_rate_stats': success_stats,
            'token_efficiency_stats': token_stats,
            'providers_tested': len(set(r.provider for r in results.values())),
            'test_categories': len(set(r.category for r in results.values()))
        }
    
    def _analyze_provider_performance(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance differences between providers."""
        by_provider = {}
        
        # Group results by provider
        for result in results.values():
            provider_key = f"{result.provider}_{result.model}"
            if provider_key not in by_provider:
                by_provider[provider_key] = []
            by_provider[provider_key].append(result)
        
        provider_analysis = {}
        
        for provider, provider_results in by_provider.items():
            # Collect all metrics for this provider
            all_metrics = []
            for result in provider_results:
                all_metrics.extend(result.metrics)
            
            successful_metrics = [m for m in all_metrics if m.success]
            
            if not successful_metrics:
                continue
            
            # Calculate provider statistics
            response_times = [m.response_time for m in successful_metrics]
            response_stats = self._calculate_statistical_summary(response_times)
            
            # Provider-specific insights
            provider_insights = []
            
            # Check for performance issues
            if response_stats.mean > 5.0:
                provider_insights.append({
                    'type': 'warning',
                    'message': f'High average response time: {response_stats.mean:.2f}s'
                })
            
            # Check for consistency
            if response_stats.std_dev > response_stats.mean * 0.5:
                provider_insights.append({
                    'type': 'warning',
                    'message': f'High response time variability (CV: {(response_stats.std_dev/response_stats.mean)*100:.1f}%)'
                })
            
            # Calculate success rate
            success_rate = (len(successful_metrics) / len(all_metrics)) * 100
            
            provider_analysis[provider] = {
                'total_requests': len(all_metrics),
                'successful_requests': len(successful_metrics),
                'success_rate': success_rate,
                'response_time_stats': response_stats,
                'insights': provider_insights,
                'performance_rating': rate_performance(response_stats.mean, 'response_time', False),
                'reliability_rating': 'excellent' if success_rate >= 95 else 'good' if success_rate >= 85 else 'needs_improvement'
            }
        
        # Cross-provider comparison
        if len(provider_analysis) > 1:
            comparison = self._compare_providers(provider_analysis)
            provider_analysis['comparison'] = comparison
        
        return provider_analysis
    
    def _analyze_test_categories(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance across different test categories."""
        by_category = {}
        
        for result in results.values():
            category = result.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        category_analysis = {}
        
        for category, cat_results in by_category.items():
            all_metrics = []
            for result in cat_results:
                all_metrics.extend(result.metrics)
            
            successful_metrics = [m for m in all_metrics if m.success]
            
            if not successful_metrics:
                continue
            
            response_times = [m.response_time for m in successful_metrics]
            response_stats = self._calculate_statistical_summary(response_times)
            
            success_rate = (len(successful_metrics) / len(all_metrics)) * 100
            
            # Category-specific analysis
            insights = []
            
            # Check if this category is particularly challenging
            if success_rate < 80:
                insights.append({
                    'type': 'warning',
                    'message': f'Low success rate for {category} tests: {success_rate:.1f}%'
                })
            
            # Performance expectations by category
            expected_times = {
                'short': 2.0,
                'medium': 5.0,
                'long': 10.0,
                'throughput': 1.0,
                'token_efficiency': 8.0
            }
            
            expected = expected_times.get(category, 5.0)
            if response_stats.mean > expected * 1.5:
                insights.append({
                    'type': 'observation',
                    'message': f'Response times higher than expected for {category} category'
                })
            
            category_analysis[category] = {
                'total_requests': len(all_metrics),
                'successful_requests': len(successful_metrics),
                'success_rate': success_rate,
                'response_time_stats': response_stats,
                'expected_response_time': expected,
                'performance_vs_expected': response_stats.mean / expected,
                'insights': insights
            }
        
        return category_analysis
    
    def _detect_outliers(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Detect and analyze outliers in performance data."""
        outlier_threshold = BenchmarkConfig.STATISTICS_SETTINGS['outlier_threshold']
        
        all_response_times = []
        outlier_details = []
        
        for result_key, result in results.items():
            successful_metrics = [m for m in result.metrics if m.success]
            if not successful_metrics:
                continue
            
            response_times = [m.response_time for m in successful_metrics]
            if len(response_times) < 3:
                continue
            
            mean_time = statistics.mean(response_times)
            std_dev = statistics.stdev(response_times)
            
            for i, time in enumerate(response_times):
                z_score = abs(time - mean_time) / std_dev if std_dev > 0 else 0
                
                if z_score > outlier_threshold:
                    outlier_details.append({
                        'test': result_key,
                        'provider': result.provider,
                        'model': result.model,
                        'category': result.category,
                        'response_time': time,
                        'z_score': z_score,
                        'deviation_from_mean': time - mean_time
                    })
            
            all_response_times.extend(response_times)
        
        # Overall outlier statistics
        if all_response_times:
            overall_mean = statistics.mean(all_response_times)
            overall_std = statistics.stdev(all_response_times)
            
            severe_outliers = [o for o in outlier_details if o['z_score'] > 3.0]
            moderate_outliers = [o for o in outlier_details if 2.0 < o['z_score'] <= 3.0]
        
        return {
            'total_outliers': len(outlier_details),
            'severe_outliers': len(severe_outliers) if all_response_times else 0,
            'moderate_outliers': len(moderate_outliers) if all_response_times else 0,
            'outlier_threshold': outlier_threshold,
            'outlier_details': outlier_details[:20],  # Limit to first 20
            'outlier_summary': self._summarize_outliers(outlier_details) if outlier_details else {}
        }
    
    def _generate_insights(self, results: Dict[str, BenchmarkResult]) -> List[PerformanceInsight]:
        """Generate actionable performance insights."""
        insights = []
        
        # Provider performance insights
        by_provider = {}
        for result in results.values():
            provider_key = f"{result.provider}_{result.model}"
            if provider_key not in by_provider:
                by_provider[provider_key] = []
            by_provider[provider_key].append(result)
        
        # Analyze each provider
        for provider, provider_results in by_provider.items():
            all_metrics = []
            for result in provider_results:
                all_metrics.extend(result.metrics)
            
            successful_metrics = [m for m in all_metrics if m.success]
            success_rate = (len(successful_metrics) / len(all_metrics)) * 100 if all_metrics else 0
            
            if success_rate < 90:
                insights.append(PerformanceInsight(
                    category='warning',
                    title=f'Low Success Rate - {provider}',
                    description=f'{provider} has a success rate of {success_rate:.1f}%, which is below optimal.',
                    impact='high' if success_rate < 70 else 'medium',
                    supporting_data={'success_rate': success_rate, 'total_requests': len(all_metrics)},
                    recommendation='Check API key validity, rate limits, and network connectivity. Consider implementing retry logic.'
                ))
            
            if successful_metrics:
                avg_response_time = statistics.mean(m.response_time for m in successful_metrics)
                if avg_response_time > 8.0:
                    insights.append(PerformanceInsight(
                        category='warning',
                        title=f'High Response Time - {provider}',
                        description=f'{provider} has an average response time of {avg_response_time:.2f}s, which may impact user experience.',
                        impact='medium',
                        supporting_data={'avg_response_time': avg_response_time},
                        recommendation='Consider using faster models or implementing response caching for common queries.'
                    ))
        
        # Cross-provider insights
        if len(by_provider) > 1:
            provider_speeds = {}
            for provider, provider_results in by_provider.items():
                all_metrics = []
                for result in provider_results:
                    all_metrics.extend(result.metrics)
                
                successful_metrics = [m for m in all_metrics if m.success]
                if successful_metrics:
                    avg_speed = statistics.mean(m.response_time for m in successful_metrics)
                    provider_speeds[provider] = avg_speed
            
            if provider_speeds:
                fastest = min(provider_speeds.items(), key=lambda x: x[1])
                slowest = max(provider_speeds.items(), key=lambda x: x[1])
                
                if slowest[1] > fastest[1] * 2:
                    insights.append(PerformanceInsight(
                        category='observation',
                        title='Significant Performance Difference',
                        description=f'{fastest[0]} is significantly faster than {slowest[0]} (avg {fastest[1]:.2f}s vs {slowest[1]:.2f}s)',
                        impact='medium',
                        supporting_data={'fastest': fastest, 'slowest': slowest},
                        recommendation=f'Consider using {fastest[0]} for time-sensitive applications.'
                    ))
        
        self.insights = insights
        return insights
    
    def _generate_recommendations(self, results: Dict[str, BenchmarkResult]) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze overall performance patterns
        all_metrics = []
        for result in results.values():
            all_metrics.extend(result.metrics)
        
        successful_metrics = [m for m in all_metrics if m.success]
        overall_success_rate = (len(successful_metrics) / len(all_metrics)) * 100 if all_metrics else 0
        
        if overall_success_rate < 95:
            recommendations.append({
                'category': 'reliability',
                'priority': 'high',
                'title': 'Improve Error Handling',
                'description': 'Implement robust error handling and retry mechanisms to improve overall success rate.',
                'action_items': [
                    'Add exponential backoff for rate-limited requests',
                    'Implement circuit breaker pattern for failing providers',
                    'Add request timeout handling',
                    'Monitor and alert on error rates'
                ]
            })
        
        if successful_metrics:
            avg_response_time = statistics.mean(m.response_time for m in successful_metrics)
            if avg_response_time > 5.0:
                recommendations.append({
                    'category': 'performance',
                    'priority': 'medium',
                    'title': 'Optimize Response Times',
                    'description': 'Consider strategies to reduce average response times.',
                    'action_items': [
                        'Implement response caching for common queries',
                        'Use streaming responses where appropriate',
                        'Consider using faster models for simple tasks',
                        'Optimize prompt engineering to reduce token usage'
                    ]
                })
        
        # Provider-specific recommendations
        by_provider = {}
        for result in results.values():
            provider = result.provider
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(result)
        
        for provider, provider_results in by_provider.items():
            provider_metrics = []
            for result in provider_results:
                provider_metrics.extend(result.metrics)
            
            provider_successful = [m for m in provider_metrics if m.success]
            provider_success_rate = (len(provider_successful) / len(provider_metrics)) * 100 if provider_metrics else 0
            
            if provider_success_rate < 85:
                recommendations.append({
                    'category': 'provider_specific',
                    'priority': 'high',
                    'title': f'Address {provider} Reliability Issues',
                    'description': f'{provider} has a low success rate of {provider_success_rate:.1f}%',
                    'action_items': [
                        f'Review {provider} API configuration and credentials',
                        f'Check {provider}-specific rate limits and quotas',
                        f'Monitor {provider} service status and announcements',
                        f'Consider alternative models or providers for critical paths'
                    ]
                })
        
        return recommendations
    
    def _test_statistical_significance(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Test for statistical significance in performance differences."""
        # Group by provider for comparison
        by_provider = {}
        for result in results.values():
            provider_key = f"{result.provider}_{result.model}"
            if provider_key not in by_provider:
                by_provider[provider_key] = []
            by_provider[provider_key].append(result)
        
        if len(by_provider) < 2:
            return {'note': 'Need at least 2 providers for significance testing'}
        
        # Collect response times by provider
        provider_response_times = {}
        for provider, provider_results in by_provider.items():
            response_times = []
            for result in provider_results:
                successful_metrics = [m for m in result.metrics if m.success]
                response_times.extend([m.response_time for m in successful_metrics])
            
            if len(response_times) >= 10:  # Minimum sample size
                provider_response_times[provider] = response_times
        
        # Perform pairwise comparisons
        comparisons = []
        providers = list(provider_response_times.keys())
        
        for i in range(len(providers)):
            for j in range(i + 1, len(providers)):
                provider_a = providers[i]
                provider_b = providers[j]
                
                times_a = provider_response_times[provider_a]
                times_b = provider_response_times[provider_b]
                
                # Simple t-test approximation
                mean_a = statistics.mean(times_a)
                mean_b = statistics.mean(times_b)
                
                # Calculate effect size (Cohen's d)
                pooled_std = math.sqrt((statistics.variance(times_a) + statistics.variance(times_b)) / 2)
                cohens_d = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
                
                # Interpret effect size
                effect_magnitude = (
                    'large' if cohens_d >= 0.8 else
                    'medium' if cohens_d >= 0.5 else
                    'small' if cohens_d >= 0.2 else
                    'negligible'
                )
                
                comparisons.append({
                    'provider_a': provider_a,
                    'provider_b': provider_b,
                    'mean_difference': mean_b - mean_a,
                    'cohens_d': cohens_d,
                    'effect_magnitude': effect_magnitude,
                    'practical_significance': cohens_d >= 0.5,
                    'faster_provider': provider_a if mean_a < mean_b else provider_b
                })
        
        return {
            'pairwise_comparisons': comparisons,
            'significant_differences': [c for c in comparisons if c['practical_significance']],
            'methodology': 'Cohen\'s d effect size analysis (d >= 0.5 considered practically significant)'
        }
    
    def _calculate_statistical_summary(self, values: List[float]) -> StatisticalSummary:
        """Calculate comprehensive statistical summary."""
        if not values:
            return StatisticalSummary(0, 0, 0, 0, 0, {}, 0)
        
        values = sorted(values)
        n = len(values)
        
        mean = statistics.mean(values)
        median = statistics.median(values)
        std_dev = statistics.stdev(values) if n > 1 else 0
        min_val = min(values)
        max_val = max(values)
        
        # Calculate percentiles
        percentiles = {}
        for p in BenchmarkConfig.STATISTICS_SETTINGS['percentiles']:
            index = int((p / 100) * (n - 1))
            percentiles[p] = values[index]
        
        # Calculate confidence interval for mean
        confidence_interval = None
        if n > 1 and std_dev > 0:
            # Simple approximation using t-distribution
            margin_error = 1.96 * (std_dev / math.sqrt(n))  # Approximation for 95% CI
            confidence_interval = (mean - margin_error, mean + margin_error)
        
        return StatisticalSummary(
            mean=mean,
            median=median,
            std_dev=std_dev,
            min_value=min_val,
            max_value=max_val,
            percentiles=percentiles,
            sample_size=n,
            confidence_interval=confidence_interval
        )
    
    def _compare_providers(self, provider_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare provider performance and generate insights."""
        providers = [p for p in provider_analysis.keys() if p != 'comparison']
        
        if len(providers) < 2:
            return {}
        
        # Find best performers
        best_response_time = min(providers, key=lambda p: provider_analysis[p]['response_time_stats'].mean)
        best_success_rate = max(providers, key=lambda p: provider_analysis[p]['success_rate'])
        
        # Calculate relative performance
        baseline_response_time = provider_analysis[best_response_time]['response_time_stats'].mean
        
        relative_performance = {}
        for provider in providers:
            provider_time = provider_analysis[provider]['response_time_stats'].mean
            relative_performance[provider] = {
                'response_time_ratio': provider_time / baseline_response_time,
                'success_rate_diff': provider_analysis[provider]['success_rate'] - provider_analysis[best_success_rate]['success_rate']
            }
        
        return {
            'best_response_time': best_response_time,
            'best_success_rate': best_success_rate,
            'relative_performance': relative_performance,
            'performance_spread': {
                'response_time_range': (
                    min(provider_analysis[p]['response_time_stats'].mean for p in providers),
                    max(provider_analysis[p]['response_time_stats'].mean for p in providers)
                ),
                'success_rate_range': (
                    min(provider_analysis[p]['success_rate'] for p in providers),
                    max(provider_analysis[p]['success_rate'] for p in providers)
                )
            }
        }
    
    def _summarize_outliers(self, outlier_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize outlier patterns."""
        if not outlier_details:
            return {}
        
        # Group by provider
        by_provider = {}
        for outlier in outlier_details:
            provider = outlier['provider']
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(outlier)
        
        # Group by category
        by_category = {}
        for outlier in outlier_details:
            category = outlier['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(outlier)
        
        return {
            'outliers_by_provider': {p: len(outliers) for p, outliers in by_provider.items()},
            'outliers_by_category': {c: len(outliers) for c, outliers in by_category.items()},
            'most_affected_provider': max(by_provider.items(), key=lambda x: len(x[1]))[0] if by_provider else None,
            'most_affected_category': max(by_category.items(), key=lambda x: len(x[1]))[0] if by_category else None,
            'average_z_score': statistics.mean(o['z_score'] for o in outlier_details),
            'max_deviation': max(o['deviation_from_mean'] for o in outlier_details)
        }