"""
Benchmark reporting and visualization

This module provides comprehensive reporting capabilities for performance benchmarks,
including text reports, charts, and data export functionality.
"""

import os
import json
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import seaborn as sns
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
    pd = None

from .benchmark_config import BenchmarkConfig, rate_performance
from .benchmark_suite import BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """
    Comprehensive benchmark reporting system.
    
    Features:
    - Text-based reports with statistics
    - Performance comparison tables
    - Chart generation (if matplotlib available)
    - Data export (JSON, CSV)
    - HTML reports with interactive elements
    """
    
    def __init__(self, output_dir: str = "benchmark_reports"):
        """
        Initialize the benchmark reporter.
        
        Args:
            output_dir: Directory to save reports and charts
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available. Chart generation will be disabled.")
    
    def generate_comprehensive_report(self, 
                                    results: Dict[str, BenchmarkResult],
                                    title: str = "LLM Provider Performance Benchmark Report") -> str:
        """
        Generate a comprehensive text report.
        
        Args:
            results: Dictionary of benchmark results
            title: Report title
            
        Returns:
            Formatted text report
        """
        report_lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Header
        report_lines.extend([
            "=" * 80,
            title,
            "=" * 80,
            f"Generated: {timestamp}",
            f"Total Results: {len(results)}",
            ""
        ])
        
        # Group results by provider
        by_provider = self._group_results_by_provider(results)
        
        # Executive Summary
        report_lines.extend(self._generate_executive_summary(by_provider))
        
        # Detailed Provider Analysis
        report_lines.extend(self._generate_provider_analysis(by_provider))
        
        # Performance Comparison Table
        report_lines.extend(self._generate_comparison_table(by_provider))
        
        # Test Category Analysis
        report_lines.extend(self._generate_category_analysis(results))
        
        # Recommendations
        report_lines.extend(self._generate_recommendations(by_provider))
        
        # Footer
        report_lines.extend([
            "",
            "=" * 80,
            "End of Report",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def _group_results_by_provider(self, results: Dict[str, BenchmarkResult]) -> Dict[str, List[BenchmarkResult]]:
        """Group results by provider."""
        by_provider = {}
        for result in results.values():
            provider_key = f"{result.provider}_{result.model}"
            if provider_key not in by_provider:
                by_provider[provider_key] = []
            by_provider[provider_key].append(result)
        return by_provider
    
    def _generate_executive_summary(self, by_provider: Dict[str, List[BenchmarkResult]]) -> List[str]:
        """Generate executive summary section."""
        lines = [
            "EXECUTIVE SUMMARY",
            "-" * 40,
            ""
        ]
        
        # Calculate overall statistics
        all_results = []
        for provider_results in by_provider.values():
            all_results.extend(provider_results)
        
        if not all_results:
            lines.append("No benchmark results available.")
            return lines
        
        # Overall metrics
        total_tests = len(all_results)
        successful_tests = sum(1 for r in all_results if r.success_rate > 80)
        avg_success_rate = statistics.mean(r.success_rate for r in all_results)
        
        lines.extend([
            f"Providers Tested: {len(by_provider)}",
            f"Total Tests Run: {total_tests}",
            f"Overall Success Rate: {avg_success_rate:.1f}%",
            f"Tests with >80% Success: {successful_tests}/{total_tests}",
            ""
        ])
        
        # Top performers
        if all_results:
            # Best average response time
            best_response_time = min(all_results, key=lambda x: x.avg_response_time)
            lines.append(f"Fastest Average Response: {best_response_time.provider} {best_response_time.model} "
                        f"({best_response_time.avg_response_time:.2f}s)")
            
            # Best success rate
            best_success = max(all_results, key=lambda x: x.success_rate)
            lines.append(f"Highest Success Rate: {best_success.provider} {best_success.model} "
                        f"({best_success.success_rate:.1f}%)")
            
            # Most tokens processed
            best_tokens = max(all_results, key=lambda x: x.total_tokens_processed)
            lines.append(f"Most Tokens Processed: {best_tokens.provider} {best_tokens.model} "
                        f"({best_tokens.total_tokens_processed:,} tokens)")
        
        lines.append("")
        return lines
    
    def _generate_provider_analysis(self, by_provider: Dict[str, List[BenchmarkResult]]) -> List[str]:
        """Generate detailed provider analysis."""
        lines = [
            "DETAILED PROVIDER ANALYSIS",
            "-" * 40,
            ""
        ]
        
        for provider_key, provider_results in by_provider.items():
            lines.append(f"{provider_key.upper()}")
            lines.append("=" * len(provider_key))
            
            # Calculate provider statistics
            avg_response_time = statistics.mean(r.avg_response_time for r in provider_results)
            avg_success_rate = statistics.mean(r.success_rate for r in provider_results)
            total_tokens = sum(r.total_tokens_processed for r in provider_results)
            
            lines.extend([
                f"Tests Completed: {len(provider_results)}",
                f"Average Response Time: {avg_response_time:.2f}s",
                f"Average Success Rate: {avg_success_rate:.1f}%",
                f"Total Tokens Processed: {total_tokens:,}",
                ""
            ])
            
            # Test breakdown
            lines.append("Test Results:")
            for result in provider_results:
                status = "✓" if result.success_rate > 80 else "⚠" if result.success_rate > 50 else "✗"
                lines.append(f"  {status} {result.test_name}: {result.success_rate:.1f}% success, "
                           f"{result.avg_response_time:.2f}s avg time")
            
            # Performance rating
            response_time_rating = rate_performance(avg_response_time, "response_time", False)
            lines.extend([
                "",
                f"Overall Rating: {response_time_rating.title()}",
                ""
            ])
        
        return lines
    
    def _generate_comparison_table(self, by_provider: Dict[str, List[BenchmarkResult]]) -> List[str]:
        """Generate performance comparison table."""
        lines = [
            "PERFORMANCE COMPARISON",
            "-" * 40,
            ""
        ]
        
        if not by_provider:
            lines.append("No data available for comparison.")
            return lines
        
        # Create comparison data
        comparison_data = []
        for provider_key, provider_results in by_provider.items():
            avg_response_time = statistics.mean(r.avg_response_time for r in provider_results)
            avg_success_rate = statistics.mean(r.success_rate for r in provider_results)
            total_tokens = sum(r.total_tokens_processed for r in provider_results)
            
            comparison_data.append({
                "provider": provider_key,
                "avg_response_time": avg_response_time,
                "success_rate": avg_success_rate,
                "total_tokens": total_tokens,
                "num_tests": len(provider_results)
            })
        
        # Sort by response time
        comparison_data.sort(key=lambda x: x["avg_response_time"])
        
        # Generate table
        lines.append(f"{'Provider':<25} {'Avg Response (s)':<18} {'Success Rate':<13} {'Tests':<8} {'Tokens':<10}")
        lines.append("-" * 80)
        
        for data in comparison_data:
            lines.append(
                f"{data['provider']:<25} "
                f"{data['avg_response_time']:<18.2f} "
                f"{data['success_rate']:<13.1f}% "
                f"{data['num_tests']:<8d} "
                f"{data['total_tokens']:<10,d}"
            )
        
        lines.append("")
        return lines
    
    def _generate_category_analysis(self, results: Dict[str, BenchmarkResult]) -> List[str]:
        """Generate analysis by test category."""
        lines = [
            "TEST CATEGORY ANALYSIS",
            "-" * 40,
            ""
        ]
        
        # Group by category
        by_category = {}
        for result in results.values():
            category = result.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        for category, cat_results in by_category.items():
            lines.append(f"{category.upper()} Tests:")
            
            avg_response_time = statistics.mean(r.avg_response_time for r in cat_results)
            avg_success_rate = statistics.mean(r.success_rate for r in cat_results)
            
            lines.extend([
                f"  Average Response Time: {avg_response_time:.2f}s",
                f"  Average Success Rate: {avg_success_rate:.1f}%",
                f"  Number of Tests: {len(cat_results)}",
                ""
            ])
        
        return lines
    
    def _generate_recommendations(self, by_provider: Dict[str, List[BenchmarkResult]]) -> List[str]:
        """Generate recommendations based on results."""
        lines = [
            "RECOMMENDATIONS",
            "-" * 40,
            ""
        ]
        
        if not by_provider:
            lines.append("No data available for recommendations.")
            return lines
        
        # Calculate provider rankings
        provider_scores = {}
        for provider_key, provider_results in by_provider.items():
            avg_response_time = statistics.mean(r.avg_response_time for r in provider_results)
            avg_success_rate = statistics.mean(r.success_rate for r in provider_results)
            
            # Simple scoring: success rate weight 60%, speed weight 40%
            # Lower response time is better, so invert it
            speed_score = max(0, 10 - avg_response_time) / 10  # Normalize to 0-1
            success_score = avg_success_rate / 100  # Convert to 0-1
            
            overall_score = (success_score * 0.6) + (speed_score * 0.4)
            provider_scores[provider_key] = overall_score
        
        # Sort by score
        ranked_providers = sorted(provider_scores.items(), key=lambda x: x[1], reverse=True)
        
        lines.extend([
            "Provider Rankings (Overall Performance):",
            ""
        ])
        
        for i, (provider, score) in enumerate(ranked_providers, 1):
            rating = "Excellent" if score > 0.8 else "Good" if score > 0.6 else "Acceptable" if score > 0.4 else "Needs Improvement"
            lines.append(f"{i}. {provider} - {rating} (Score: {score:.2f})")
        
        lines.extend([
            "",
            "Usage Recommendations:",
            ""
        ])
        
        if ranked_providers:
            best_provider = ranked_providers[0][0]
            lines.append(f"• For best overall performance: {best_provider}")
            
            # Find best for specific use cases
            fastest_provider = min(by_provider.items(), 
                                 key=lambda x: statistics.mean(r.avg_response_time for r in x[1]))[0]
            lines.append(f"• For fastest response times: {fastest_provider}")
            
            most_reliable = max(by_provider.items(),
                              key=lambda x: statistics.mean(r.success_rate for r in x[1]))[0]
            lines.append(f"• For highest reliability: {most_reliable}")
        
        lines.extend([
            "",
            "General Recommendations:",
            "• Monitor success rates regularly and investigate failures",
            "• Consider rate limiting to avoid API throttling",
            "• Test with your specific use case prompts for best results",
            "• Keep track of token usage for cost optimization"
        ])
        
        return lines
    
    def save_report(self, 
                   results: Dict[str, BenchmarkResult],
                   filename: Optional[str] = None,
                   title: str = "Performance Benchmark Report") -> str:
        """
        Save comprehensive report to file.
        
        Args:
            results: Benchmark results
            filename: Output filename (auto-generated if None)
            title: Report title
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        report_content = self.generate_comprehensive_report(results, title)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {filepath}")
        return filepath
    
    def export_to_json(self, 
                      results: Dict[str, BenchmarkResult],
                      filename: Optional[str] = None) -> str:
        """
        Export results to JSON format.
        
        Args:
            results: Benchmark results
            filename: Output filename
            
        Returns:
            Path to exported JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert results to JSON-serializable format
        json_data = {
            "benchmark_results": {},
            "summary": self._calculate_summary_stats(results),
            "export_timestamp": datetime.now().isoformat()
        }
        
        for key, result in results.items():
            json_data["benchmark_results"][key] = {
                "test_name": result.test_name,
                "provider": result.provider,
                "model": result.model,
                "category": result.category,
                "avg_response_time": result.avg_response_time,
                "success_rate": result.success_rate,
                "total_tokens_processed": result.total_tokens_processed,
                "avg_tokens_per_second": result.avg_tokens_per_second,
                "metrics": [
                    {
                        "response_time": m.response_time,
                        "tokens_per_second": m.tokens_per_second,
                        "success": m.success,
                        "error": m.error,
                        "timestamp": m.timestamp.isoformat()
                    }
                    for m in result.metrics
                ]
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON export saved to: {filepath}")
        return filepath
    
    def export_to_csv(self, 
                     results: Dict[str, BenchmarkResult],
                     filename: Optional[str] = None) -> str:
        """
        Export results to CSV format.
        
        Args:
            results: Benchmark results
            filename: Output filename
            
        Returns:
            Path to exported CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare CSV data
        csv_data = []
        for result in results.values():
            for metric in result.metrics:
                csv_data.append({
                    "test_name": result.test_name,
                    "provider": result.provider,
                    "model": result.model,
                    "category": result.category,
                    "response_time": metric.response_time,
                    "tokens_per_second": metric.tokens_per_second,
                    "success": metric.success,
                    "error": metric.error,
                    "timestamp": metric.timestamp.isoformat()
                })
        
        if csv_data:
            fieldnames = csv_data[0].keys()
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"CSV export saved to: {filepath}")
        return filepath
    
    def generate_charts(self, 
                       results: Dict[str, BenchmarkResult],
                       chart_types: List[str] = None) -> List[str]:
        """
        Generate performance charts.
        
        Args:
            results: Benchmark results
            chart_types: Types of charts to generate
            
        Returns:
            List of paths to generated chart files
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping chart generation.")
            return []
        
        if chart_types is None:
            chart_types = ["response_time", "success_rate", "throughput", "comparison"]
        
        chart_files = []
        
        try:
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            for chart_type in chart_types:
                if chart_type == "response_time":
                    chart_file = self._generate_response_time_chart(results)
                elif chart_type == "success_rate":
                    chart_file = self._generate_success_rate_chart(results)
                elif chart_type == "throughput":
                    chart_file = self._generate_throughput_chart(results)
                elif chart_type == "comparison":
                    chart_file = self._generate_comparison_chart(results)
                else:
                    logger.warning(f"Unknown chart type: {chart_type}")
                    continue
                
                if chart_file:
                    chart_files.append(chart_file)
                    
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
        
        return chart_files
    
    def _generate_response_time_chart(self, results: Dict[str, BenchmarkResult]) -> Optional[str]:
        """Generate response time comparison chart."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            providers = []
            response_times = []
            
            for result in results.values():
                providers.append(f"{result.provider}\n{result.model}")
                response_times.append(result.avg_response_time)
            
            bars = ax.bar(providers, response_times, color=sns.color_palette("husl", len(providers)))
            
            ax.set_title('Average Response Time by Provider', fontsize=16, fontweight='bold')
            ax.set_ylabel('Response Time (seconds)', fontsize=12)
            ax.set_xlabel('Provider', fontsize=12)
            
            # Add value labels on bars
            for bar, time in zip(bars, response_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{time:.2f}s', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_time_chart_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Response time chart saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating response time chart: {e}")
            return None
    
    def _generate_success_rate_chart(self, results: Dict[str, BenchmarkResult]) -> Optional[str]:
        """Generate success rate comparison chart."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            providers = []
            success_rates = []
            
            for result in results.values():
                providers.append(f"{result.provider}\n{result.model}")
                success_rates.append(result.success_rate)
            
            colors = ['green' if rate >= 90 else 'orange' if rate >= 70 else 'red' for rate in success_rates]
            bars = ax.bar(providers, success_rates, color=colors, alpha=0.7)
            
            ax.set_title('Success Rate by Provider', fontsize=16, fontweight='bold')
            ax.set_ylabel('Success Rate (%)', fontsize=12)
            ax.set_xlabel('Provider', fontsize=12)
            ax.set_ylim(0, 100)
            
            # Add horizontal lines for reference
            ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent (90%+)')
            ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Good (70%+)')
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{rate:.1f}%', ha='center', va='bottom')
            
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"success_rate_chart_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Success rate chart saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating success rate chart: {e}")
            return None
    
    def _generate_throughput_chart(self, results: Dict[str, BenchmarkResult]) -> Optional[str]:
        """Generate throughput comparison chart."""
        try:
            # Filter for throughput-related results
            throughput_results = {k: v for k, v in results.items() if 'throughput' in v.test_name.lower()}
            
            if not throughput_results:
                logger.info("No throughput results found for chart generation")
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            providers = []
            tokens_per_sec = []
            
            for result in throughput_results.values():
                providers.append(f"{result.provider}\n{result.model}")
                avg_tps = result.avg_tokens_per_second or 0
                tokens_per_sec.append(avg_tps)
            
            bars = ax.bar(providers, tokens_per_sec, color=sns.color_palette("viridis", len(providers)))
            
            ax.set_title('Token Processing Throughput by Provider', fontsize=16, fontweight='bold')
            ax.set_ylabel('Tokens per Second', fontsize=12)
            ax.set_xlabel('Provider', fontsize=12)
            
            # Add value labels on bars
            for bar, tps in zip(bars, tokens_per_sec):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{tps:.1f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"throughput_chart_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Throughput chart saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating throughput chart: {e}")
            return None
    
    def _generate_comparison_chart(self, results: Dict[str, BenchmarkResult]) -> Optional[str]:
        """Generate multi-metric comparison chart."""
        try:
            by_provider = self._group_results_by_provider(results)
            
            if len(by_provider) < 2:
                logger.info("Need at least 2 providers for comparison chart")
                return None
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            providers = list(by_provider.keys())
            
            # Response Time Comparison
            response_times = [statistics.mean(r.avg_response_time for r in results) 
                            for results in by_provider.values()]
            ax1.bar(providers, response_times, color='skyblue')
            ax1.set_title('Average Response Time')
            ax1.set_ylabel('Seconds')
            ax1.tick_params(axis='x', rotation=45)
            
            # Success Rate Comparison
            success_rates = [statistics.mean(r.success_rate for r in results) 
                           for results in by_provider.values()]
            ax2.bar(providers, success_rates, color='lightgreen')
            ax2.set_title('Average Success Rate')
            ax2.set_ylabel('Percentage')
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='x', rotation=45)
            
            # Token Processing
            token_rates = [statistics.mean([r.avg_tokens_per_second or 0 for r in results]) 
                          for results in by_provider.values()]
            ax3.bar(providers, token_rates, color='coral')
            ax3.set_title('Average Tokens per Second')
            ax3.set_ylabel('Tokens/Second')
            ax3.tick_params(axis='x', rotation=45)
            
            # Total Tokens Processed
            total_tokens = [sum(r.total_tokens_processed for r in results) 
                          for results in by_provider.values()]
            ax4.bar(providers, total_tokens, color='gold')
            ax4.set_title('Total Tokens Processed')
            ax4.set_ylabel('Tokens')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.suptitle('Provider Performance Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_chart_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison chart saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating comparison chart: {e}")
            return None
    
    def _calculate_summary_stats(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics for export."""
        if not results:
            return {}
        
        all_response_times = []
        all_success_rates = []
        total_tokens = 0
        
        for result in results.values():
            all_response_times.append(result.avg_response_time)
            all_success_rates.append(result.success_rate)
            total_tokens += result.total_tokens_processed
        
        return {
            "total_tests": len(results),
            "avg_response_time": statistics.mean(all_response_times),
            "min_response_time": min(all_response_times),
            "max_response_time": max(all_response_times),
            "avg_success_rate": statistics.mean(all_success_rates),
            "total_tokens_processed": total_tokens,
            "unique_providers": len(set(r.provider for r in results.values()))
        }