"""
Compatibility test runner and reporter

This module provides a comprehensive compatibility testing framework that can
run tests across multiple providers and generate detailed compatibility reports.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import traceback

from tests.providers.fixtures import get_available_providers
from llm_providers.base import LLMProvider
from llm_providers.exceptions import ProviderError, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityTestResult:
    """Result from a single compatibility test."""
    test_name: str
    provider_name: str
    model_name: str
    success: bool
    duration: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProviderCompatibilityProfile:
    """Complete compatibility profile for a provider."""
    provider_name: str
    model_name: str
    basic_generation: bool = False
    parameter_support: Dict[str, bool] = field(default_factory=dict)
    prompt_types: Dict[str, bool] = field(default_factory=dict)
    unicode_support: bool = False
    edge_case_handling: bool = False
    concurrent_requests: bool = False
    features: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_patterns: List[str] = field(default_factory=list)
    overall_score: float = 0.0


class CompatibilityTestRunner:
    """
    Comprehensive compatibility test runner.
    
    This class runs standardized compatibility tests across multiple providers
    and generates detailed compatibility reports.
    """
    
    def __init__(self, output_dir: str = "compatibility_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.test_results: List[CompatibilityTestResult] = []
        self.provider_profiles: Dict[str, ProviderCompatibilityProfile] = {}
        
        # Test configurations
        self.test_prompts = {
            'simple': "Hello",
            'math': "What is 2 + 2?",
            'creative': "Write a haiku about technology",
            'reasoning': "If all roses are flowers and some flowers are red, are some roses red?",
            'multilingual': "Say 'good morning' in Spanish",
            'technical': "Explain what HTTP stands for",
            'long_response': "Write a brief explanation of artificial intelligence"
        }
        
        self.parameter_tests = {
            'temperature_low': {'temperature': 0.1},
            'temperature_medium': {'temperature': 0.5},
            'temperature_high': {'temperature': 0.9},
            'max_tokens_small': {'max_tokens': 10},
            'max_tokens_medium': {'max_tokens': 100},
            'max_tokens_large': {'max_tokens': 300}
        }
        
        self.unicode_tests = [
            "Hello 🌍 world",  # Emoji
            "Café naïve résumé",  # Accented characters
            "こんにちは",  # Japanese
            "Здравствуй",  # Cyrillic
            "\"Quotes\" and 'apostrophes'",  # Quotes
            "Special: @#$%^&*()",  # Special symbols
        ]
        
        self.edge_cases = [
            "",  # Empty string
            " ",  # Whitespace
            "\n\n",  # Newlines
            "a" * 500,  # Long word
            "?",  # Single character
            "12345",  # Numbers only
        ]
    
    def run_comprehensive_compatibility_tests(self, providers: Optional[List[LLMProvider]] = None) -> Dict[str, Any]:
        """
        Run comprehensive compatibility tests across providers.
        
        Args:
            providers: List of providers to test. If None, uses all available providers.
            
        Returns:
            Dictionary containing test results and analysis
        """
        if providers is None:
            providers = get_available_providers()
        
        if not providers:
            logger.error("No providers available for compatibility testing")
            return {}
        
        logger.info(f"Starting compatibility tests for {len(providers)} providers")
        
        # Initialize provider profiles
        for provider in providers:
            provider_key = f"{provider.__class__.__name__}_{provider.model_name}"
            self.provider_profiles[provider_key] = ProviderCompatibilityProfile(
                provider_name=provider.__class__.__name__,
                model_name=provider.model_name
            )
        
        # Run test suites
        self._run_basic_generation_tests(providers)
        self._run_parameter_compatibility_tests(providers)
        self._run_prompt_type_tests(providers)
        self._run_unicode_tests(providers)
        self._run_edge_case_tests(providers)
        self._run_concurrent_tests(providers)
        self._run_feature_tests(providers)
        
        # Calculate compatibility scores
        self._calculate_compatibility_scores()
        
        # Generate analysis
        analysis = self._analyze_compatibility_results()
        
        logger.info(f"Compatibility tests completed. {len(self.test_results)} total test results.")
        
        return {
            'test_results': self.test_results,
            'provider_profiles': self.provider_profiles,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def _run_basic_generation_tests(self, providers: List[LLMProvider]):
        """Test basic text generation capability."""
        logger.info("Running basic generation tests...")
        
        for provider in providers:
            provider_key = f"{provider.__class__.__name__}_{provider.model_name}"
            
            start_time = time.time()
            try:
                response = provider.generate("Hello world", max_tokens=20)
                duration = time.time() - start_time
                
                success = response is not None and len(response.strip()) > 0
                
                self.test_results.append(CompatibilityTestResult(
                    test_name="basic_generation",
                    provider_name=provider.__class__.__name__,
                    model_name=provider.model_name,
                    success=success,
                    duration=duration,
                    details={'response_length': len(response) if response else 0}
                ))
                
                self.provider_profiles[provider_key].basic_generation = success
                self.provider_profiles[provider_key].performance_metrics['basic_generation_time'] = duration
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)
                
                self.test_results.append(CompatibilityTestResult(
                    test_name="basic_generation",
                    provider_name=provider.__class__.__name__,
                    model_name=provider.model_name,
                    success=False,
                    duration=duration,
                    error=error_msg
                ))
                
                self.provider_profiles[provider_key].basic_generation = False
                self.provider_profiles[provider_key].error_patterns.append(f"basic_generation: {type(e).__name__}")
            
            time.sleep(0.5)  # Rate limiting
    
    def _run_parameter_compatibility_tests(self, providers: List[LLMProvider]):
        """Test parameter compatibility."""
        logger.info("Running parameter compatibility tests...")
        
        for provider in providers:
            provider_key = f"{provider.__class__.__name__}_{provider.model_name}"
            
            for param_name, params in self.parameter_tests.items():
                start_time = time.time()
                try:
                    response = provider.generate("Count to 3", **params)
                    duration = time.time() - start_time
                    
                    success = response is not None
                    
                    self.test_results.append(CompatibilityTestResult(
                        test_name=f"parameter_{param_name}",
                        provider_name=provider.__class__.__name__,
                        model_name=provider.model_name,
                        success=success,
                        duration=duration,
                        details={'parameters': params}
                    ))
                    
                    self.provider_profiles[provider_key].parameter_support[param_name] = success
                    
                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = str(e)
                    
                    self.test_results.append(CompatibilityTestResult(
                        test_name=f"parameter_{param_name}",
                        provider_name=provider.__class__.__name__,
                        model_name=provider.model_name,
                        success=False,
                        duration=duration,
                        error=error_msg,
                        details={'parameters': params}
                    ))
                    
                    self.provider_profiles[provider_key].parameter_support[param_name] = False
                    self.provider_profiles[provider_key].error_patterns.append(f"parameter_{param_name}: {type(e).__name__}")
                
                time.sleep(0.3)
    
    def _run_prompt_type_tests(self, providers: List[LLMProvider]):
        """Test different prompt types."""
        logger.info("Running prompt type tests...")
        
        for provider in providers:
            provider_key = f"{provider.__class__.__name__}_{provider.model_name}"
            
            for prompt_type, prompt in self.test_prompts.items():
                start_time = time.time()
                try:
                    response = provider.generate(prompt, max_tokens=100)
                    duration = time.time() - start_time
                    
                    success = response is not None and len(response.strip()) > 0
                    
                    self.test_results.append(CompatibilityTestResult(
                        test_name=f"prompt_type_{prompt_type}",
                        provider_name=provider.__class__.__name__,
                        model_name=provider.model_name,
                        success=success,
                        duration=duration,
                        details={'prompt_type': prompt_type, 'prompt': prompt[:50]}
                    ))
                    
                    self.provider_profiles[provider_key].prompt_types[prompt_type] = success
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self.test_results.append(CompatibilityTestResult(
                        test_name=f"prompt_type_{prompt_type}",
                        provider_name=provider.__class__.__name__,
                        model_name=provider.model_name,
                        success=False,
                        duration=duration,
                        error=str(e),
                        details={'prompt_type': prompt_type}
                    ))
                    
                    self.provider_profiles[provider_key].prompt_types[prompt_type] = False
                
                time.sleep(0.2)
    
    def _run_unicode_tests(self, providers: List[LLMProvider]):
        """Test unicode and special character handling."""
        logger.info("Running unicode compatibility tests...")
        
        for provider in providers:
            provider_key = f"{provider.__class__.__name__}_{provider.model_name}"
            
            unicode_success_count = 0
            
            for i, test_text in enumerate(self.unicode_tests):
                start_time = time.time()
                try:
                    response = provider.generate(f"Echo: {test_text}", max_tokens=50)
                    duration = time.time() - start_time
                    
                    success = response is not None and len(response.strip()) > 0
                    if success:
                        unicode_success_count += 1
                    
                    self.test_results.append(CompatibilityTestResult(
                        test_name=f"unicode_test_{i}",
                        provider_name=provider.__class__.__name__,
                        model_name=provider.model_name,
                        success=success,
                        duration=duration,
                        details={'test_text': test_text[:20]}
                    ))
                    
                except Exception as e:
                    self.test_results.append(CompatibilityTestResult(
                        test_name=f"unicode_test_{i}",
                        provider_name=provider.__class__.__name__,
                        model_name=provider.model_name,
                        success=False,
                        duration=time.time() - start_time,
                        error=str(e)
                    ))
                
                time.sleep(0.2)
            
            # Consider unicode support successful if most tests pass
            self.provider_profiles[provider_key].unicode_support = (
                unicode_success_count / len(self.unicode_tests) >= 0.7
            )
    
    def _run_edge_case_tests(self, providers: List[LLMProvider]):
        """Test edge case handling."""
        logger.info("Running edge case tests...")
        
        for provider in providers:
            provider_key = f"{provider.__class__.__name__}_{provider.model_name}"
            
            edge_case_success_count = 0
            
            for i, edge_case in enumerate(self.edge_cases):
                start_time = time.time()
                try:
                    response = provider.generate(edge_case, max_tokens=30)
                    duration = time.time() - start_time
                    
                    # For edge cases, not crashing is considered success
                    success = True  # If we get here, no exception was raised
                    edge_case_success_count += 1
                    
                    self.test_results.append(CompatibilityTestResult(
                        test_name=f"edge_case_{i}",
                        provider_name=provider.__class__.__name__,
                        model_name=provider.model_name,
                        success=success,
                        duration=duration,
                        details={'edge_case_type': f"case_{i}"}
                    ))
                    
                except Exception as e:
                    self.test_results.append(CompatibilityTestResult(
                        test_name=f"edge_case_{i}",
                        provider_name=provider.__class__.__name__,
                        model_name=provider.model_name,
                        success=False,
                        duration=time.time() - start_time,
                        error=str(e)
                    ))
                
                time.sleep(0.1)
            
            self.provider_profiles[provider_key].edge_case_handling = (
                edge_case_success_count / len(self.edge_cases) >= 0.6
            )
    
    def _run_concurrent_tests(self, providers: List[LLMProvider]):
        """Test concurrent request handling."""
        logger.info("Running concurrent request tests...")
        
        import concurrent.futures
        
        for provider in providers:
            provider_key = f"{provider.__class__.__name__}_{provider.model_name}"
            
            def make_request():
                try:
                    return provider.generate("Count to 3", max_tokens=30)
                except Exception as e:
                    return e
            
            start_time = time.time()
            
            # Run 3 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(make_request) for _ in range(3)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            duration = time.time() - start_time
            
            # Count successful responses
            successful_results = [r for r in results if isinstance(r, str) and r is not None]
            success_rate = len(successful_results) / len(results)
            
            success = success_rate >= 0.5  # At least 50% success rate
            
            self.test_results.append(CompatibilityTestResult(
                test_name="concurrent_requests",
                provider_name=provider.__class__.__name__,
                model_name=provider.model_name,
                success=success,
                duration=duration,
                details={'success_rate': success_rate, 'total_requests': len(results)}
            ))
            
            self.provider_profiles[provider_key].concurrent_requests = success
            self.provider_profiles[provider_key].performance_metrics['concurrent_success_rate'] = success_rate
            
            time.sleep(1.0)  # Longer pause after concurrent tests
    
    def _run_feature_tests(self, providers: List[LLMProvider]):
        """Test advanced features."""
        logger.info("Running feature compatibility tests...")
        
        for provider in providers:
            provider_key = f"{provider.__class__.__name__}_{provider.model_name}"
            
            # Test streaming
            streaming_support = False
            if hasattr(provider, 'generate_stream'):
                try:
                    stream = provider.generate_stream("Count to 3", max_tokens=30)
                    chunks = list(stream)
                    streaming_support = len(chunks) > 0
                except Exception:
                    streaming_support = False
            
            # Test token counting
            token_counting_support = False
            if hasattr(provider, 'count_tokens'):
                try:
                    count = provider.count_tokens("Hello world")
                    token_counting_support = isinstance(count, int) and count > 0
                except Exception:
                    token_counting_support = False
            
            # Test model listing
            model_listing_support = False
            if hasattr(provider, 'list_models'):
                try:
                    models = provider.list_models()
                    model_listing_support = isinstance(models, list) and len(models) > 0
                except Exception:
                    model_listing_support = False
            
            self.provider_profiles[provider_key].features = {
                'streaming': streaming_support,
                'token_counting': token_counting_support,
                'model_listing': model_listing_support
            }
            
            # Record feature test results
            for feature, supported in self.provider_profiles[provider_key].features.items():
                self.test_results.append(CompatibilityTestResult(
                    test_name=f"feature_{feature}",
                    provider_name=provider.__class__.__name__,
                    model_name=provider.model_name,
                    success=supported,
                    duration=0.0,
                    details={'feature': feature}
                ))
    
    def _calculate_compatibility_scores(self):
        """Calculate overall compatibility scores for each provider."""
        for provider_key, profile in self.provider_profiles.items():
            score = 0.0
            max_score = 0.0
            
            # Basic generation (weight: 30%)
            if profile.basic_generation:
                score += 30
            max_score += 30
            
            # Parameter support (weight: 25%)
            param_score = sum(profile.parameter_support.values()) / len(profile.parameter_support) * 25
            score += param_score
            max_score += 25
            
            # Prompt types (weight: 20%)
            if profile.prompt_types:
                prompt_score = sum(profile.prompt_types.values()) / len(profile.prompt_types) * 20
                score += prompt_score
            max_score += 20
            
            # Unicode support (weight: 10%)
            if profile.unicode_support:
                score += 10
            max_score += 10
            
            # Edge case handling (weight: 5%)
            if profile.edge_case_handling:
                score += 5
            max_score += 5
            
            # Concurrent requests (weight: 5%)
            if profile.concurrent_requests:
                score += 5
            max_score += 5
            
            # Features (weight: 5%)
            if profile.features:
                feature_score = sum(profile.features.values()) / len(profile.features) * 5
                score += feature_score
            max_score += 5
            
            profile.overall_score = (score / max_score) * 100 if max_score > 0 else 0
    
    def _analyze_compatibility_results(self) -> Dict[str, Any]:
        """Analyze compatibility test results and generate insights."""
        analysis = {
            'summary': {},
            'provider_rankings': [],
            'compatibility_issues': [],
            'recommendations': []
        }
        
        # Overall summary
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        analysis['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
            'providers_tested': len(self.provider_profiles),
            'test_categories': len(set(r.test_name.split('_')[0] for r in self.test_results if '_' in r.test_name))
        }
        
        # Provider rankings
        ranked_providers = sorted(
            self.provider_profiles.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        analysis['provider_rankings'] = [
            {
                'provider': profile.provider_name,
                'model': profile.model_name,
                'score': profile.overall_score,
                'strengths': self._identify_strengths(profile),
                'weaknesses': self._identify_weaknesses(profile)
            }
            for _, profile in ranked_providers
        ]
        
        # Identify compatibility issues
        for provider_key, profile in self.provider_profiles.items():
            if profile.overall_score < 70:
                analysis['compatibility_issues'].append({
                    'provider': profile.provider_name,
                    'model': profile.model_name,
                    'score': profile.overall_score,
                    'major_issues': profile.error_patterns[:3],
                    'failed_categories': [
                        category for category, success in [
                            ('basic_generation', profile.basic_generation),
                            ('unicode_support', profile.unicode_support),
                            ('edge_case_handling', profile.edge_case_handling),
                            ('concurrent_requests', profile.concurrent_requests)
                        ] if not success
                    ]
                })
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_compatibility_recommendations()
        
        return analysis
    
    def _identify_strengths(self, profile: ProviderCompatibilityProfile) -> List[str]:
        """Identify strengths for a provider profile."""
        strengths = []
        
        if profile.basic_generation:
            strengths.append("reliable_basic_generation")
        
        if sum(profile.parameter_support.values()) / len(profile.parameter_support) > 0.8:
            strengths.append("excellent_parameter_support")
        
        if profile.unicode_support:
            strengths.append("good_unicode_handling")
        
        if profile.concurrent_requests:
            strengths.append("handles_concurrent_requests")
        
        feature_count = sum(profile.features.values())
        if feature_count >= 2:
            strengths.append("rich_feature_set")
        
        return strengths
    
    def _identify_weaknesses(self, profile: ProviderCompatibilityProfile) -> List[str]:
        """Identify weaknesses for a provider profile."""
        weaknesses = []
        
        if not profile.basic_generation:
            weaknesses.append("basic_generation_issues")
        
        if sum(profile.parameter_support.values()) / len(profile.parameter_support) < 0.5:
            weaknesses.append("poor_parameter_support")
        
        if not profile.unicode_support:
            weaknesses.append("unicode_handling_issues")
        
        if not profile.edge_case_handling:
            weaknesses.append("poor_edge_case_handling")
        
        if not profile.concurrent_requests:
            weaknesses.append("concurrent_request_issues")
        
        return weaknesses
    
    def _generate_compatibility_recommendations(self) -> List[Dict[str, str]]:
        """Generate compatibility improvement recommendations."""
        recommendations = []
        
        # Analyze common issues across providers
        all_errors = []
        for profile in self.provider_profiles.values():
            all_errors.extend(profile.error_patterns)
        
        # Rate limiting issues
        rate_limit_errors = [e for e in all_errors if 'rate' in e.lower() or 'limit' in e.lower()]
        if rate_limit_errors:
            recommendations.append({
                'category': 'rate_limiting',
                'title': 'Implement Better Rate Limiting',
                'description': 'Multiple providers are experiencing rate limiting issues',
                'action': 'Add exponential backoff and request queuing'
            })
        
        # Parameter support issues
        param_issues = [e for e in all_errors if 'parameter' in e.lower()]
        if param_issues:
            recommendations.append({
                'category': 'parameters',
                'title': 'Improve Parameter Validation',
                'description': 'Some providers have inconsistent parameter support',
                'action': 'Add parameter validation and fallback mechanisms'
            })
        
        # Unicode/encoding issues
        unicode_issues = [e for e in all_errors if 'unicode' in e.lower() or 'encoding' in e.lower()]
        if unicode_issues:
            recommendations.append({
                'category': 'encoding',
                'title': 'Fix Unicode Handling',
                'description': 'Unicode support is inconsistent across providers',
                'action': 'Ensure proper UTF-8 encoding in all requests'
            })
        
        return recommendations
    
    def generate_compatibility_report(self, filename: Optional[str] = None) -> str:
        """Generate a comprehensive compatibility report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"compatibility_report_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("LLM Provider Compatibility Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {len(self.test_results)}\n")
            f.write(f"Providers Tested: {len(self.provider_profiles)}\n\n")
            
            # Provider profiles
            f.write("PROVIDER COMPATIBILITY PROFILES\n")
            f.write("-" * 40 + "\n\n")
            
            for provider_key, profile in sorted(
                self.provider_profiles.items(),
                key=lambda x: x[1].overall_score,
                reverse=True
            ):
                f.write(f"{profile.provider_name} ({profile.model_name})\n")
                f.write(f"Overall Score: {profile.overall_score:.1f}/100\n")
                f.write(f"Basic Generation: {'✓' if profile.basic_generation else '✗'}\n")
                f.write(f"Unicode Support: {'✓' if profile.unicode_support else '✗'}\n")
                f.write(f"Edge Case Handling: {'✓' if profile.edge_case_handling else '✗'}\n")
                f.write(f"Concurrent Requests: {'✓' if profile.concurrent_requests else '✗'}\n")
                
                # Parameter support
                f.write("Parameter Support:\n")
                for param, supported in profile.parameter_support.items():
                    f.write(f"  {param}: {'✓' if supported else '✗'}\n")
                
                # Features
                f.write("Features:\n")
                for feature, supported in profile.features.items():
                    f.write(f"  {feature}: {'✓' if supported else '✗'}\n")
                
                # Performance metrics
                if profile.performance_metrics:
                    f.write("Performance Metrics:\n")
                    for metric, value in profile.performance_metrics.items():
                        if 'time' in metric:
                            f.write(f"  {metric}: {value:.3f}s\n")
                        else:
                            f.write(f"  {metric}: {value:.2f}\n")
                
                f.write("\n")
            
            # Test results summary
            f.write("TEST RESULTS SUMMARY\n")
            f.write("-" * 40 + "\n\n")
            
            # Group results by test type
            by_test_type = {}
            for result in self.test_results:
                test_type = result.test_name.split('_')[0]
                if test_type not in by_test_type:
                    by_test_type[test_type] = []
                by_test_type[test_type].append(result)
            
            for test_type, results in by_test_type.items():
                success_count = sum(1 for r in results if r.success)
                total_count = len(results)
                success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
                
                f.write(f"{test_type.title()} Tests: {success_count}/{total_count} ({success_rate:.1f}%)\n")
            
            f.write("\n")
            
            # Recommendations
            analysis = self._analyze_compatibility_results()
            if analysis['recommendations']:
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n\n")
                
                for rec in analysis['recommendations']:
                    f.write(f"• {rec['title']}\n")
                    f.write(f"  {rec['description']}\n")
                    f.write(f"  Action: {rec['action']}\n\n")
        
        logger.info(f"Compatibility report saved to: {filepath}")
        return filepath
    
    def export_results_json(self, filename: Optional[str] = None) -> str:
        """Export compatibility results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"compatibility_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert dataclasses to dictionaries for JSON serialization
        export_data = {
            'test_results': [
                {
                    'test_name': r.test_name,
                    'provider_name': r.provider_name,
                    'model_name': r.model_name,
                    'success': r.success,
                    'duration': r.duration,
                    'error': r.error,
                    'details': r.details,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.test_results
            ],
            'provider_profiles': {
                key: {
                    'provider_name': profile.provider_name,
                    'model_name': profile.model_name,
                    'basic_generation': profile.basic_generation,
                    'parameter_support': profile.parameter_support,
                    'prompt_types': profile.prompt_types,
                    'unicode_support': profile.unicode_support,
                    'edge_case_handling': profile.edge_case_handling,
                    'concurrent_requests': profile.concurrent_requests,
                    'features': profile.features,
                    'performance_metrics': profile.performance_metrics,
                    'error_patterns': profile.error_patterns,
                    'overall_score': profile.overall_score
                }
                for key, profile in self.provider_profiles.items()
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Compatibility results exported to: {filepath}")
        return filepath