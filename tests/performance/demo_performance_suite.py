#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Demo

This script demonstrates the complete performance benchmarking suite for LLM providers.
It showcases all major components: benchmarking, analysis, reporting, and visualization.

Usage:
    python demo_performance_suite.py [--mode {quick,standard,comprehensive}] [--providers openai,anthropic]

Example:
    python demo_performance_suite.py --mode quick --providers openai
    python demo_performance_suite.py --mode standard --providers openai,anthropic
"""

# Import paths fixed - sys.path manipulation removed
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.providers.anthropic import AnthropicProvider
from src.providers.openai import OpenAIProvider
from tests.performance import (
    BenchmarkConfig,
    BenchmarkMode,
    BenchmarkReporter,
    BenchmarkSuite,
    PerformanceAnalyzer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceBenchmarkDemo:
    """
    Comprehensive demonstration of the performance benchmarking suite.

    This class orchestrates the entire benchmarking process:
    1. Provider setup and validation
    2. Benchmark execution
    3. Statistical analysis
    4. Report generation
    5. Results visualization
    """

    def __init__(self, mode: BenchmarkMode = BenchmarkMode.STANDARD):
        self.mode = mode
        self.suite = BenchmarkSuite(mode)
        self.reporter = BenchmarkReporter()
        self.analyzer = PerformanceAnalyzer()
        self.results = {}
        self.analysis = {}

    def setup_providers(self, provider_names: List[str]) -> bool:
        """
        Setup and validate LLM providers for benchmarking.

        Args:
            provider_names: List of provider names to setup

        Returns:
            True if at least one provider was successfully setup
        """
        logger.info(f"Setting up providers: {', '.join(provider_names)}")

        providers_added = 0

        for provider_name in provider_names:
            try:
                if provider_name.lower() == "openai":
                    if not os.getenv("OPENAI_API_KEY"):
                        logger.warning("OpenAI API key not found. Skipping OpenAI provider.")
                        continue

                    # Try with default model first, fallback to gpt-3.5-turbo
                    try:
                        provider = OpenAIProvider(model_name="gpt-4o-mini")
                    except Exception:
                        provider = OpenAIProvider(model_name="gpt-3.5-turbo")

                    self.suite.add_provider(provider)
                    providers_added += 1
                    logger.info(f"Added OpenAI provider with model: {provider.model_name}")

                elif provider_name.lower() == "anthropic":
                    if not os.getenv("ANTHROPIC_API_KEY"):
                        logger.warning("Anthropic API key not found. Skipping Anthropic provider.")
                        continue

                    provider = AnthropicProvider(model_name="claude-3-haiku-20240307")
                    self.suite.add_provider(provider)
                    providers_added += 1
                    logger.info(f"Added Anthropic provider with model: {provider.model_name}")

                else:
                    logger.warning(f"Unknown provider: {provider_name}")

            except Exception as e:
                logger.error(f"Failed to setup {provider_name} provider: {e}")

        if providers_added == 0:
            logger.error("No providers were successfully setup. Please check your API keys.")
            return False

        logger.info(f"Successfully setup {providers_added} provider(s)")
        return True

    def run_benchmarks(self) -> bool:
        """
        Execute the complete benchmark suite.

        Returns:
            True if benchmarks completed successfully
        """
        if not self.suite.providers:
            logger.error("No providers available for benchmarking")
            return False

        logger.info(f"Starting benchmark suite in {self.mode.value} mode...")

        try:
            # Run all benchmarks
            self.results = self.suite.run_all_benchmarks()

            if not self.results:
                logger.error("No benchmark results generated")
                return False

            logger.info(
                f"Benchmarks completed successfully. Generated {len(self.results)} result sets."
            )

            # Display quick summary
            self._display_quick_summary()

            return True

        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            return False

    def analyze_results(self) -> bool:
        """
        Perform comprehensive statistical analysis of benchmark results.

        Returns:
            True if analysis completed successfully
        """
        if not self.results:
            logger.error("No results available for analysis")
            return False

        logger.info("Performing comprehensive performance analysis...")

        try:
            self.analysis = self.analyzer.analyze_results(self.results)

            # Display key insights
            self._display_analysis_insights()

            return True

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return False

    def generate_reports(self) -> List[str]:
        """
        Generate comprehensive reports and visualizations.

        Returns:
            List of generated file paths
        """
        if not self.results:
            logger.error("No results available for reporting")
            return []

        logger.info("Generating comprehensive reports...")

        generated_files = []

        try:
            # Generate text report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_title = f"Performance Benchmark Report - {self.mode.value.title()} Mode"

            report_file = self.reporter.save_report(
                self.results, filename=f"performance_report_{timestamp}.txt", title=report_title
            )
            generated_files.append(report_file)

            # Export to JSON
            json_file = self.reporter.export_to_json(
                self.results, filename=f"benchmark_results_{timestamp}.json"
            )
            generated_files.append(json_file)

            # Export to CSV
            csv_file = self.reporter.export_to_csv(
                self.results, filename=f"benchmark_data_{timestamp}.csv"
            )
            generated_files.append(csv_file)

            # Generate charts (if matplotlib available)
            try:
                chart_files = self.reporter.generate_charts(self.results)
                generated_files.extend(chart_files)
            except Exception as e:
                logger.warning(f"Chart generation failed: {e}")

            logger.info(f"Generated {len(generated_files)} report files")

            return generated_files

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return generated_files

    def _display_quick_summary(self):
        """Display a quick summary of benchmark results."""
        print("\n" + "=" * 80)
        print("QUICK BENCHMARK SUMMARY")
        print("=" * 80)

        summary = self.suite.get_summary_statistics()

        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Providers Tested: {summary.get('providers_tested', 0)}")
        print(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.1f}%")
        print(f"Average Response Time: {summary.get('avg_response_time', 0):.2f}s")

        # Provider breakdown
        provider_summary = summary.get("provider_summary", {})
        if provider_summary:
            print(f"\nProvider Performance:")
            for provider, stats in provider_summary.items():
                print(
                    f"  {provider}: {stats['success_rate']:.1f}% success, "
                    f"{stats['avg_response_time']:.2f}s avg response"
                )

        print("=" * 80)

    def _display_analysis_insights(self):
        """Display key insights from the performance analysis."""
        if not self.analysis:
            return

        print("\n" + "=" * 80)
        print("KEY PERFORMANCE INSIGHTS")
        print("=" * 80)

        # Display high-impact insights
        insights = self.analysis.get("performance_insights", [])
        high_impact_insights = [i for i in insights if i.impact == "high"]

        if high_impact_insights:
            print("HIGH IMPACT ISSUES:")
            for insight in high_impact_insights[:3]:  # Show top 3
                print(f"  ⚠️  {insight.title}")
                print(f"      {insight.description}")
                if insight.recommendation:
                    print(f"      💡 {insight.recommendation}")
                print()

        # Display recommendations
        recommendations = self.analysis.get("recommendations", [])
        high_priority_recs = [r for r in recommendations if r.get("priority") == "high"]

        if high_priority_recs:
            print("HIGH PRIORITY RECOMMENDATIONS:")
            for rec in high_priority_recs[:2]:  # Show top 2
                print(f"  🔧 {rec['title']}")
                print(f"      {rec['description']}")
                for action in rec.get("action_items", [])[:2]:  # Show first 2 actions
                    print(f"      • {action}")
                print()

        # Statistical significance
        significance = self.analysis.get("statistical_significance", {})
        significant_diffs = significance.get("significant_differences", [])

        if significant_diffs:
            print("SIGNIFICANT PERFORMANCE DIFFERENCES:")
            for diff in significant_diffs[:2]:  # Show top 2
                faster = diff["faster_provider"]
                print(f"  📊 {faster} is significantly faster")
                print(f"      Effect size: {diff['effect_magnitude']} (d={diff['cohens_d']:.2f})")
                print()

        print("=" * 80)


def main():
    """Main function for running the performance benchmark demo."""
    parser = argparse.ArgumentParser(description="LLM Provider Performance Benchmark Demo")
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "comprehensive", "stress"],
        default="standard",
        help="Benchmark mode (default: standard)",
    )
    parser.add_argument(
        "--providers", default="openai", help="Comma-separated list of providers (default: openai)"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_reports",
        help="Output directory for reports (default: benchmark_reports)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse providers
    provider_names = [p.strip() for p in args.providers.split(",")]

    # Set output directory
    os.environ["BENCHMARK_OUTPUT_DIR"] = args.output_dir

    try:
        # Convert mode string to enum
        mode = BenchmarkMode(args.mode)

        # Create demo instance
        demo = PerformanceBenchmarkDemo(mode)

        print("🚀 Starting LLM Provider Performance Benchmark Demo")
        print(f"Mode: {mode.value}")
        print(f"Providers: {', '.join(provider_names)}")
        print(f"Output Directory: {args.output_dir}")
        print()

        # Step 1: Setup providers
        if not demo.setup_providers(provider_names):
            print("❌ Failed to setup providers. Exiting.")
            return 1

        # Step 2: Run benchmarks
        print("\n📊 Running benchmarks...")
        if not demo.run_benchmarks():
            print("❌ Benchmark execution failed. Exiting.")
            return 1

        # Step 3: Analyze results
        print("\n🔍 Analyzing results...")
        if not demo.analyze_results():
            print("⚠️  Analysis failed, but continuing with report generation...")

        # Step 4: Generate reports
        print("\n📄 Generating reports...")
        generated_files = demo.generate_reports()

        if generated_files:
            print(f"\n✅ Demo completed successfully!")
            print(f"Generated {len(generated_files)} files:")
            for file_path in generated_files:
                print(f"  📁 {file_path}")
        else:
            print("\n⚠️  Demo completed but no reports were generated.")

        print(f"\nAll outputs saved to: {args.output_dir}/")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
