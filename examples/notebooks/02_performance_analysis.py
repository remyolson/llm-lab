#!/usr/bin/env python3
"""
Performance Analysis Example

This example demonstrates how to analyze and compare the performance of different
LLM providers using the LLM Lab framework's built-in performance testing tools.

Usage:
    python examples/notebooks/02_performance_analysis.py

Requirements:
    - API keys configured in .env file
    - Performance testing dependencies (matplotlib, seaborn, pandas)
"""

import os
import sys
import time
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_providers import GoogleProvider, OpenAIProvider, AnthropicProvider

def setup_test_providers():
    """Setup providers with performance-optimized configurations."""
    providers = {}
    
    # Google Gemini - optimized for speed
    if os.getenv('GOOGLE_API_KEY'):
        providers['google_fast'] = GoogleProvider(
            model_name="gemini-1.5-flash",
            temperature=0.1,  # More deterministic = faster
            max_tokens=500,   # Shorter responses = faster
        )
    
    # OpenAI GPT - cost-effective model
    if os.getenv('OPENAI_API_KEY'):
        providers['openai_fast'] = OpenAIProvider(
            model_name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=500,
        )
    
    # Anthropic Claude - fast model
    if os.getenv('ANTHROPIC_API_KEY'):
        providers['anthropic_fast'] = AnthropicProvider(
            model_name="claude-3-5-haiku-20241022",
            temperature=0.1,
            max_tokens=500,
        )
    
    return providers

def run_response_time_test(provider, prompts: List[str], iterations: int = 3) -> Dict[str, Any]:
    """
    Test response times for a provider across multiple prompts and iterations.
    
    Args:
        provider: The LLM provider instance
        prompts: List of test prompts
        iterations: Number of iterations per prompt
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"🔍 Testing response times ({iterations} iterations per prompt)...")
    
    all_times = []
    prompt_results = {}
    
    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        prompt_times = []
        
        for iteration in range(iterations):
            try:
                start_time = time.time()
                response = provider.generate(prompt)
                end_time = time.time()
                
                response_time = end_time - start_time
                prompt_times.append(response_time)
                all_times.append(response_time)
                
                print(f"    Iteration {iteration+1}: {response_time:.2f}s")
                
                # Rate limiting to avoid hitting API limits
                time.sleep(1)
                
            except Exception as e:
                print(f"    Iteration {iteration+1}: Failed - {e}")
                continue
        
        if prompt_times:
            prompt_results[f"prompt_{i+1}"] = {
                'times': prompt_times,
                'mean': statistics.mean(prompt_times),
                'median': statistics.median(prompt_times),
                'std_dev': statistics.stdev(prompt_times) if len(prompt_times) > 1 else 0,
                'min': min(prompt_times),
                'max': max(prompt_times)
            }
    
    # Overall statistics
    if all_times:
        overall_stats = {
            'total_requests': len(all_times),
            'mean_response_time': statistics.mean(all_times),
            'median_response_time': statistics.median(all_times),
            'std_dev': statistics.stdev(all_times) if len(all_times) > 1 else 0,
            'min_response_time': min(all_times),
            'max_response_time': max(all_times),
            'percentile_95': sorted(all_times)[int(0.95 * len(all_times))] if len(all_times) > 20 else max(all_times)
        }
    else:
        overall_stats = {
            'total_requests': 0,
            'mean_response_time': 0,
            'median_response_time': 0,
            'std_dev': 0,
            'min_response_time': 0,
            'max_response_time': 0,
            'percentile_95': 0
        }
    
    return {
        'overall_stats': overall_stats,
        'prompt_results': prompt_results,
        'raw_times': all_times
    }

def measure_throughput(provider, prompt: str, duration_seconds: int = 30) -> Dict[str, Any]:
    """
    Measure throughput (requests per minute) for a provider.
    
    Args:
        provider: The LLM provider instance
        prompt: Test prompt to use
        duration_seconds: How long to run the test
    
    Returns:
        Dictionary with throughput metrics
    """
    print(f"📊 Measuring throughput for {duration_seconds} seconds...")
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    request_times = []
    successful_requests = 0
    failed_requests = 0
    
    while time.time() < end_time:
        try:
            request_start = time.time()
            response = provider.generate(prompt)
            request_end = time.time()
            
            request_times.append(request_end - request_start)
            successful_requests += 1
            
            print(f"  Request {successful_requests}: {request_end - request_start:.2f}s")
            
        except Exception as e:
            failed_requests += 1
            print(f"  Request failed: {str(e)[:50]}...")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    total_requests = successful_requests + failed_requests
    
    return {
        'duration_seconds': total_time,
        'successful_requests': successful_requests,
        'failed_requests': failed_requests,
        'total_requests': total_requests,
        'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
        'requests_per_minute': (successful_requests / total_time) * 60 if total_time > 0 else 0,
        'average_response_time': statistics.mean(request_times) if request_times else 0,
        'request_times': request_times
    }

def analyze_cost_efficiency(provider_name: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze cost efficiency based on performance and estimated pricing.
    
    Args:
        provider_name: Name of the provider
        performance_data: Performance test results
    
    Returns:
        Dictionary with cost analysis
    """
    # Rough cost estimates per 1K tokens (as of 2024)
    cost_estimates = {
        'google_fast': {'input': 0.00015, 'output': 0.0006},    # Gemini 1.5 Flash
        'openai_fast': {'input': 0.00015, 'output': 0.0006},    # GPT-4o mini
        'anthropic_fast': {'input': 0.00025, 'output': 0.00125}  # Claude 3.5 Haiku
    }
    
    if provider_name not in cost_estimates:
        return {'error': 'Cost data not available for this provider'}
    
    costs = cost_estimates[provider_name]
    stats = performance_data['overall_stats']
    
    # Rough token estimates (assuming ~750 tokens per response)
    estimated_input_tokens = 100  # Rough estimate for prompts
    estimated_output_tokens = 500  # Based on max_tokens=500
    
    cost_per_request = (
        (estimated_input_tokens / 1000 * costs['input']) +
        (estimated_output_tokens / 1000 * costs['output'])
    )
    
    return {
        'cost_per_request_usd': cost_per_request,
        'cost_per_minute_at_max_throughput': cost_per_request * (60 / stats['mean_response_time']) if stats['mean_response_time'] > 0 else 0,
        'requests_per_dollar': 1 / cost_per_request if cost_per_request > 0 else float('inf'),
        'cost_efficiency_score': (1 / stats['mean_response_time']) / cost_per_request if stats['mean_response_time'] > 0 and cost_per_request > 0 else 0
    }

def compare_providers_performance(providers: Dict[str, Any], test_prompts: List[str]) -> Dict[str, Any]:
    """
    Compare performance across multiple providers.
    
    Args:
        providers: Dictionary of provider instances
        test_prompts: List of prompts to test with
    
    Returns:
        Dictionary with comparative analysis
    """
    print("\\n🏁 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    results = {}
    
    for provider_name, provider in providers.items():
        print(f"\\n🤖 Testing {provider_name}...")
        print("-" * 30)
        
        try:
            # Response time analysis
            response_time_data = run_response_time_test(provider, test_prompts[:3], iterations=2)
            
            # Throughput analysis (shorter duration for demo)
            throughput_data = measure_throughput(provider, test_prompts[0], duration_seconds=15)
            
            # Cost efficiency analysis
            cost_data = analyze_cost_efficiency(provider_name, response_time_data)
            
            results[provider_name] = {
                'response_times': response_time_data,
                'throughput': throughput_data,
                'cost_analysis': cost_data,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ {provider_name} analysis complete")
            
        except Exception as e:
            print(f"✗ {provider_name} analysis failed: {e}")
            results[provider_name] = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    return results

def generate_performance_report(results: Dict[str, Any]) -> str:
    """
    Generate a human-readable performance report.
    
    Args:
        results: Performance comparison results
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("\\n📈 PERFORMANCE ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary table
    successful_providers = [name for name, data in results.items() if 'error' not in data]
    
    if not successful_providers:
        report.append("❌ No providers completed successfully")
        return "\\n".join(report)
    
    report.append("📊 PERFORMANCE SUMMARY")
    report.append("-" * 40)
    report.append(f"{'Provider':<15} {'Avg Time':<10} {'Throughput':<12} {'Success Rate':<12}")
    report.append("-" * 40)
    
    for provider_name in successful_providers:
        data = results[provider_name]
        avg_time = data['response_times']['overall_stats']['mean_response_time']
        throughput = data['throughput']['requests_per_minute']
        success_rate = data['throughput']['success_rate'] * 100
        
        report.append(f"{provider_name:<15} {avg_time:<10.2f} {throughput:<12.1f} {success_rate:<12.1f}%")
    
    # Detailed analysis
    report.append("\\n🔍 DETAILED ANALYSIS")
    report.append("-" * 40)
    
    for provider_name in successful_providers:
        data = results[provider_name]
        report.append(f"\\n🤖 {provider_name.upper()}")
        
        # Response time stats
        stats = data['response_times']['overall_stats']
        report.append(f"  Response Times:")
        report.append(f"    Mean: {stats['mean_response_time']:.2f}s")
        report.append(f"    Median: {stats['median_response_time']:.2f}s")
        report.append(f"    95th percentile: {stats['percentile_95']:.2f}s")
        report.append(f"    Range: {stats['min_response_time']:.2f}s - {stats['max_response_time']:.2f}s")
        
        # Throughput stats
        throughput = data['throughput']
        report.append(f"  Throughput:")
        report.append(f"    Requests per minute: {throughput['requests_per_minute']:.1f}")
        report.append(f"    Success rate: {throughput['success_rate']*100:.1f}%")
        
        # Cost analysis
        if 'error' not in data['cost_analysis']:
            cost = data['cost_analysis']
            report.append(f"  Cost Efficiency:")
            report.append(f"    Cost per request: ${cost['cost_per_request_usd']:.4f}")
            report.append(f"    Requests per dollar: {cost['requests_per_dollar']:.0f}")
            report.append(f"    Efficiency score: {cost['cost_efficiency_score']:.2f}")
    
    # Recommendations
    report.append("\\n💡 RECOMMENDATIONS")
    report.append("-" * 40)
    
    if len(successful_providers) > 1:
        # Find best performers
        fastest_provider = min(successful_providers, 
                             key=lambda p: results[p]['response_times']['overall_stats']['mean_response_time'])
        
        highest_throughput = max(successful_providers,
                               key=lambda p: results[p]['throughput']['requests_per_minute'])
        
        most_reliable = max(successful_providers,
                          key=lambda p: results[p]['throughput']['success_rate'])
        
        report.append(f"🚀 Fastest responses: {fastest_provider}")
        report.append(f"📊 Highest throughput: {highest_throughput}")
        report.append(f"🛡️  Most reliable: {most_reliable}")
        
        # Cost efficiency
        cost_efficient = []
        for provider_name in successful_providers:
            cost_data = results[provider_name]['cost_analysis']
            if 'error' not in cost_data:
                cost_efficient.append((provider_name, cost_data['cost_efficiency_score']))
        
        if cost_efficient:
            best_value = max(cost_efficient, key=lambda x: x[1])
            report.append(f"💰 Best value: {best_value[0]}")
    
    report.append("\\n📋 Use Case Recommendations:")
    report.append("  • For speed-critical applications: Choose fastest provider")
    report.append("  • For high-load applications: Choose highest throughput provider")
    report.append("  • For cost-sensitive applications: Choose best value provider")
    report.append("  • For mission-critical applications: Choose most reliable provider")
    
    return "\\n".join(report)

def save_performance_data(results: Dict[str, Any], output_dir: str = "examples/results"):
    """
    Save performance analysis data to files.
    
    Args:
        results: Performance analysis results
        output_dir: Directory to save results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw data as JSON
    json_file = Path(output_dir) / f"performance_analysis_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save formatted report
    report = generate_performance_report(results)
    report_file = Path(output_dir) / f"performance_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\\n💾 Results saved:")
    print(f"  Raw data: {json_file}")
    print(f"  Report: {report_file}")
    
    return json_file, report_file

def main():
    """
    Main function demonstrating performance analysis.
    """
    print("📊 LLM Lab - Performance Analysis Example")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        # Setup providers
        print("\\n🔧 Setting up performance-optimized providers...")
        providers = setup_test_providers()
        
        if not providers:
            raise ValueError("No providers available. Please configure API keys in your .env file.")
        
        print(f"✓ Initialized {len(providers)} providers: {list(providers.keys())}")
        
        # Test prompts optimized for performance testing
        test_prompts = [
            "What is Python?",
            "Explain machine learning briefly.",
            "List 3 benefits of cloud computing.",
            "What is artificial intelligence?",
            "Define data science."
        ]
        
        print(f"\\n📝 Using {len(test_prompts)} test prompts")
        print("⚠️  Note: This will make multiple API calls and may take several minutes")
        
        # Confirm before proceeding
        confirm = input("\\nProceed with performance analysis? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Analysis cancelled.")
            return 0
        
        # Run performance comparison
        results = compare_providers_performance(providers, test_prompts)
        
        # Generate and display report
        report = generate_performance_report(results)
        print(report)
        
        # Save results
        save_performance_data(results)
        
        print("\\n🎉 Performance analysis complete!")
        print("\\n💡 Tip: Run this analysis periodically to monitor performance trends")
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())