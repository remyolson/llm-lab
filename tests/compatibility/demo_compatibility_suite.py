#!/usr/bin/env python3
"""
Compatibility Testing Suite Demo

This script demonstrates the comprehensive compatibility testing framework
for LLM providers, including automated testing and detailed reporting.

Usage:
    python demo_compatibility_suite.py [--providers openai,anthropic] [--output-dir reports]

Example:
    python demo_compatibility_suite.py --providers openai
    python demo_compatibility_suite.py --providers openai,anthropic --output-dir compatibility_reports
"""

# Import paths fixed - sys.path manipulation removed
import os
import sys
import argparse
import logging
from datetime import datetime

# Add the project root to the path
, '../../..')))

from tests.compatibility.compatibility_runner import CompatibilityTestRunner
from tests.providers.fixtures import get_available_providers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for running the compatibility testing demo."""
    parser = argparse.ArgumentParser(description='LLM Provider Compatibility Testing Demo')
    parser.add_argument(
        '--providers',
        default='all',
        help='Comma-separated list of providers or "all" (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        default='compatibility_reports',
        help='Output directory for reports (default: compatibility_reports)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'both'],
        default='both',
        help='Report format (default: both)'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        print("🧪 Starting LLM Provider Compatibility Testing Suite")
        print(f"Output Directory: {args.output_dir}")
        print(f"Report Format: {args.format}")
        print()
        
        # Get available providers
        all_providers = get_available_providers()
        
        if not all_providers:
            print("❌ No providers available. Please check your API keys.")
            return 1
        
        # Filter providers if specified
        if args.providers.lower() != 'all':
            requested_providers = [p.strip().lower() for p in args.providers.split(',')]
            filtered_providers = []
            
            for provider in all_providers:
                provider_name = provider.__class__.__name__.lower().replace('provider', '')
                if provider_name in requested_providers:
                    filtered_providers.append(provider)
            
            if not filtered_providers:
                print(f"❌ None of the requested providers ({args.providers}) are available.")
                print(f"Available providers: {', '.join(p.__class__.__name__ for p in all_providers)}")
                return 1
            
            providers_to_test = filtered_providers
        else:
            providers_to_test = all_providers
        
        print(f"🔍 Testing {len(providers_to_test)} provider(s):")
        for provider in providers_to_test:
            print(f"  • {provider.__class__.__name__} ({provider.model_name})")
        print()
        
        # Create compatibility test runner
        runner = CompatibilityTestRunner(output_dir=args.output_dir)
        
        # Run comprehensive compatibility tests
        print("🚀 Running comprehensive compatibility tests...")
        start_time = datetime.now()
        
        results = runner.run_comprehensive_compatibility_tests(providers_to_test)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if not results:
            print("❌ No test results generated.")
            return 1
        
        print(f"✅ Testing completed in {duration:.1f} seconds")
        print()
        
        # Display quick summary
        analysis = results.get('analysis', {})
        summary = analysis.get('summary', {})
        
        print("📊 QUICK SUMMARY")
        print("=" * 40)
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Successful Tests: {summary.get('successful_tests', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"Providers Tested: {summary.get('providers_tested', 0)}")
        print(f"Test Categories: {summary.get('test_categories', 0)}")
        print()
        
        # Display provider rankings
        rankings = analysis.get('provider_rankings', [])
        if rankings:
            print("🏆 PROVIDER RANKINGS")
            print("=" * 40)
            for i, ranking in enumerate(rankings, 1):
                score = ranking['score']
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                print(f"{emoji} {i}. {ranking['provider']} ({ranking['model']})")
                print(f"    Score: {score:.1f}/100")
                
                strengths = ranking.get('strengths', [])
                if strengths:
                    print(f"    Strengths: {', '.join(strengths[:3])}")
                
                weaknesses = ranking.get('weaknesses', [])
                if weaknesses:
                    print(f"    Areas for improvement: {', '.join(weaknesses[:2])}")
                print()
        
        # Display compatibility issues
        issues = analysis.get('compatibility_issues', [])
        if issues:
            print("⚠️  COMPATIBILITY ISSUES")
            print("=" * 40)
            for issue in issues:
                print(f"• {issue['provider']} ({issue['model']})")
                print(f"  Score: {issue['score']:.1f}/100")
                
                failed_categories = issue.get('failed_categories', [])
                if failed_categories:
                    print(f"  Failed categories: {', '.join(failed_categories)}")
                print()
        
        # Display recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print("💡 RECOMMENDATIONS")
            print("=" * 40)
            for rec in recommendations[:3]:  # Show top 3
                print(f"• {rec['title']}")
                print(f"  {rec['description']}")
                print(f"  Action: {rec['action']}")
                print()
        
        # Generate reports
        print("📄 Generating reports...")
        generated_files = []
        
        if args.format in ['text', 'both']:
            report_file = runner.generate_compatibility_report()
            generated_files.append(report_file)
            print(f"  ✅ Text report: {report_file}")
        
        if args.format in ['json', 'both']:
            json_file = runner.export_results_json()
            generated_files.append(json_file)
            print(f"  ✅ JSON export: {json_file}")
        
        print()
        print(f"🎉 Compatibility testing completed successfully!")
        print(f"📁 All outputs saved to: {args.output_dir}/")
        
        # Final summary
        if rankings:
            best_provider = rankings[0]
            print(f"🌟 Top performer: {best_provider['provider']} ({best_provider['model']}) "
                  f"with {best_provider['score']:.1f}/100")
        
        if summary.get('success_rate', 0) >= 90:
            print("✨ Excellent overall compatibility across providers!")
        elif summary.get('success_rate', 0) >= 75:
            print("👍 Good overall compatibility with some areas for improvement.")
        else:
            print("⚠️  Several compatibility issues detected. Review recommendations for improvements.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Compatibility testing failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())