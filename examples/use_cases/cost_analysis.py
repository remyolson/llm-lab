#!/usr/bin/env python3
"""
Cost Analysis Use Case

This example demonstrates how to analyze and optimize costs when using multiple
LLM providers. It includes cost estimation, budget tracking, and optimization
strategies.

Usage:
    python examples/use_cases/cost_analysis.py

Features:
    - Real-time cost estimation for each provider
    - Budget tracking and alerts
    - Cost-per-token analysis
    - Provider cost comparison
    - Optimization recommendations
"""

import os
import sys
import json
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_providers import GoogleProvider, OpenAIProvider, AnthropicProvider

@dataclass
class CostEstimate:
    """Data class for cost estimation."""
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    provider: str
    model: str
    timestamp: str

class CostTracker:
    """
    Tracks costs across multiple providers and models.
    """
    
    # Cost per 1M tokens (as of late 2024)
    PRICING_DATA = {
        'google': {
            'gemini-1.5-flash': {'input': 0.15, 'output': 0.60},
            'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
            'gemini-1.0-pro': {'input': 0.50, 'output': 1.50}
        },
        'openai': {
            'gpt-4o': {'input': 5.00, 'output': 15.00},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-3.5-turbo': {'input': 1.00, 'output': 2.00}
        },
        'anthropic': {
            'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
            'claude-3-5-haiku-20241022': {'input': 0.25, 'output': 1.25},
            'claude-3-opus-20240229': {'input': 15.00, 'output': 75.00},
            'claude-3-sonnet-20240229': {'input': 3.00, 'output': 15.00},
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25}
        }
    }
    
    def __init__(self, daily_budget: float = 10.0):
        self.daily_budget = daily_budget
        self.costs_today = []
        self.total_costs = []
        self.last_reset = date.today()
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation based on text length.
        More accurate than word count for most use cases.
        """
        # Rough approximation: 1 token ≈ 0.75 words ≈ 4 characters
        return max(1, len(text) // 4)
    
    def estimate_cost(self, provider: str, model: str, input_text: str, output_text: str) -> CostEstimate:
        """
        Estimate the cost for a request.
        
        Args:
            provider: Provider name (google, openai, anthropic)
            model: Model name
            input_text: Input prompt text
            output_text: Generated response text
        
        Returns:
            CostEstimate object with detailed cost breakdown
        """
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        
        # Get pricing data
        if provider not in self.PRICING_DATA:
            raise ValueError(f"Unknown provider: {provider}")
        
        provider_pricing = self.PRICING_DATA[provider]
        if model not in provider_pricing:
            # Use a default model for the provider
            model = list(provider_pricing.keys())[0]
            print(f"⚠️  Model {model} not found, using {model} pricing")
        
        pricing = provider_pricing[model]
        
        # Calculate costs (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost
        
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            provider=provider,
            model=model,
            timestamp=datetime.now().isoformat()
        )
    
    def track_cost(self, cost_estimate: CostEstimate):
        """
        Track a cost and update daily totals.
        
        Args:
            cost_estimate: CostEstimate object from estimate_cost()
        """
        today = date.today()
        
        # Reset daily costs if it's a new day
        if today > self.last_reset:
            self.costs_today = []
            self.last_reset = today
        
        self.costs_today.append(cost_estimate)
        self.total_costs.append(cost_estimate)
    
    def get_daily_spend(self) -> float:
        """Get total spending for today."""
        return sum(cost.total_cost for cost in self.costs_today)
    
    def get_remaining_budget(self) -> float:
        """Get remaining daily budget."""
        return max(0, self.daily_budget - self.get_daily_spend())
    
    def can_afford_request(self, estimated_cost: float) -> bool:
        """Check if a request would exceed the daily budget."""
        return self.get_daily_spend() + estimated_cost <= self.daily_budget
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get a summary of costs by provider and model."""
        summary = {
            'daily_spend': self.get_daily_spend(),
            'remaining_budget': self.get_remaining_budget(),
            'budget_utilization': self.get_daily_spend() / self.daily_budget if self.daily_budget > 0 else 0,
            'requests_today': len(self.costs_today),
            'by_provider': {},
            'by_model': {}
        }
        
        # Group by provider
        for cost in self.costs_today:
            if cost.provider not in summary['by_provider']:
                summary['by_provider'][cost.provider] = {
                    'total_cost': 0,
                    'requests': 0,
                    'total_tokens': 0
                }
            
            summary['by_provider'][cost.provider]['total_cost'] += cost.total_cost
            summary['by_provider'][cost.provider]['requests'] += 1
            summary['by_provider'][cost.provider]['total_tokens'] += cost.input_tokens + cost.output_tokens
        
        # Group by model
        model_key = f"{cost.provider}/{cost.model}"
        for cost in self.costs_today:
            model_key = f"{cost.provider}/{cost.model}"
            if model_key not in summary['by_model']:
                summary['by_model'][model_key] = {
                    'total_cost': 0,
                    'requests': 0,
                    'avg_cost_per_request': 0
                }
            
            summary['by_model'][model_key]['total_cost'] += cost.total_cost
            summary['by_model'][model_key]['requests'] += 1
        
        # Calculate averages
        for model_key in summary['by_model']:
            model_data = summary['by_model'][model_key]
            model_data['avg_cost_per_request'] = model_data['total_cost'] / model_data['requests']
        
        return summary

class CostOptimizedProvider:
    """
    Wrapper around providers that includes cost tracking and optimization.
    """
    
    def __init__(self, provider, provider_name: str, model_name: str, cost_tracker: CostTracker):
        self.provider = provider
        self.provider_name = provider_name
        self.model_name = model_name
        self.cost_tracker = cost_tracker
    
    def generate(self, prompt: str, max_cost: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate a response with cost tracking.
        
        Args:
            prompt: Input prompt
            max_cost: Maximum cost allowed for this request
        
        Returns:
            Dictionary with response and cost information
        """
        # Pre-estimate cost based on input
        estimated_output_length = min(getattr(self.provider, 'max_tokens', 1000), 1000)
        estimated_output_text = "x" * (estimated_output_length * 4)  # Rough character estimate
        
        pre_estimate = self.cost_tracker.estimate_cost(
            self.provider_name, 
            self.model_name, 
            prompt, 
            estimated_output_text
        )
        
        # Check budget constraints
        if not self.cost_tracker.can_afford_request(pre_estimate.total_cost):
            return {
                'success': False,
                'error': f'Request would exceed daily budget. Estimated cost: ${pre_estimate.total_cost:.4f}',
                'estimated_cost': pre_estimate.total_cost,
                'remaining_budget': self.cost_tracker.get_remaining_budget()
            }
        
        # Check max_cost constraint
        if max_cost and pre_estimate.total_cost > max_cost:
            return {
                'success': False,
                'error': f'Request would exceed max cost limit. Estimated: ${pre_estimate.total_cost:.4f}, Limit: ${max_cost:.4f}',
                'estimated_cost': pre_estimate.total_cost,
                'max_cost': max_cost
            }
        
        # Make the actual request
        try:
            start_time = time.time()
            response = self.provider.generate(prompt)
            end_time = time.time()
            
            # Calculate actual cost
            actual_cost = self.cost_tracker.estimate_cost(
                self.provider_name,
                self.model_name,
                prompt,
                response
            )
            
            # Track the cost
            self.cost_tracker.track_cost(actual_cost)
            
            return {
                'success': True,
                'response': response,
                'cost_estimate': actual_cost,
                'response_time': end_time - start_time,
                'cost_accuracy': abs(actual_cost.total_cost - pre_estimate.total_cost) / pre_estimate.total_cost if pre_estimate.total_cost > 0 else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'estimated_cost': pre_estimate.total_cost
            }

def setup_cost_optimized_providers(cost_tracker: CostTracker) -> Dict[str, CostOptimizedProvider]:
    """
    Setup cost-optimized providers with cost tracking.
    """
    providers = {}
    
    # Google Gemini - most cost-effective model
    if os.getenv('GOOGLE_API_KEY'):
        google_provider = GoogleProvider(
            model_name="gemini-1.5-flash",
            temperature=0.1,
            max_tokens=500  # Limit output to control costs
        )
        providers['google'] = CostOptimizedProvider(
            google_provider, 'google', 'gemini-1.5-flash', cost_tracker
        )
    
    # OpenAI - most cost-effective model
    if os.getenv('OPENAI_API_KEY'):
        openai_provider = OpenAIProvider(
            model_name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=500
        )
        providers['openai'] = CostOptimizedProvider(
            openai_provider, 'openai', 'gpt-4o-mini', cost_tracker
        )
    
    # Anthropic - most cost-effective model
    if os.getenv('ANTHROPIC_API_KEY'):
        anthropic_provider = AnthropicProvider(
            model_name="claude-3-5-haiku-20241022",
            temperature=0.1,
            max_tokens=500
        )
        providers['anthropic'] = CostOptimizedProvider(
            anthropic_provider, 'anthropic', 'claude-3-5-haiku-20241022', cost_tracker
        )
    
    return providers

def find_cheapest_provider(providers: Dict[str, CostOptimizedProvider], prompt: str) -> str:
    """
    Find the cheapest provider for a given prompt.
    
    Args:
        providers: Dictionary of cost-optimized providers
        prompt: Test prompt
    
    Returns:
        Name of the cheapest provider
    """
    costs = {}
    
    for name, provider in providers.items():
        # Estimate cost without making actual request
        estimated_output = "x" * 2000  # Assume moderate response length
        cost_estimate = provider.cost_tracker.estimate_cost(
            provider.provider_name,
            provider.model_name,
            prompt,
            estimated_output
        )
        costs[name] = cost_estimate.total_cost
    
    return min(costs, key=costs.get) if costs else None

def demonstrate_cost_optimization():
    """
    Demonstrate various cost optimization strategies.
    """
    print("💰 COST OPTIMIZATION DEMONSTRATION")
    print("=" * 50)
    
    # Setup cost tracking with $5 daily budget
    cost_tracker = CostTracker(daily_budget=5.0)
    providers = setup_cost_optimized_providers(cost_tracker)
    
    if not providers:
        print("❌ No providers available. Please configure API keys.")
        return
    
    print(f"✓ Initialized {len(providers)} cost-optimized providers")
    print(f"💵 Daily budget: ${cost_tracker.daily_budget:.2f}")
    
    # Test prompts with different lengths (cost implications)
    test_prompts = [
        "What is AI?",  # Short prompt
        "Explain machine learning, deep learning, and artificial intelligence in detail.",  # Medium prompt
        "Write a comprehensive guide to getting started with machine learning, including the mathematical foundations, popular algorithms, practical implementation tips, common pitfalls to avoid, and real-world applications across different industries."  # Long prompt
    ]
    
    print("\\n🧪 COST OPTIMIZATION STRATEGIES")
    print("-" * 40)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\\n📝 Test {i}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        print(f"   Prompt length: {len(prompt)} characters")
        
        # Strategy 1: Find cheapest provider
        cheapest = find_cheapest_provider(providers, prompt)
        print(f"   💡 Cheapest provider: {cheapest}")
        
        # Strategy 2: Cost-aware generation
        if cheapest and cheapest in providers:
            result = providers[cheapest].generate(prompt, max_cost=0.01)  # 1 cent limit
            
            if result['success']:
                cost = result['cost_estimate']
                print(f"   ✓ Generated response for ${cost.total_cost:.4f}")
                print(f"   📊 Tokens: {cost.input_tokens} in, {cost.output_tokens} out")
                print(f"   📝 Response preview: {result['response'][:100]}...")
            else:
                print(f"   ❌ Failed: {result['error']}")
        
        # Show running costs
        summary = cost_tracker.get_cost_summary()
        print(f"   💳 Daily spend so far: ${summary['daily_spend']:.4f}")
        print(f"   💰 Remaining budget: ${summary['remaining_budget']:.4f}")
    
    return cost_tracker

def generate_cost_report(cost_tracker: CostTracker) -> str:
    """
    Generate a detailed cost analysis report.
    """
    summary = cost_tracker.get_cost_summary()
    
    report = []
    report.append("\\n📊 COST ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Budget overview
    report.append("💵 BUDGET OVERVIEW")
    report.append("-" * 30)
    report.append(f"Daily budget: ${cost_tracker.daily_budget:.2f}")
    report.append(f"Total spent today: ${summary['daily_spend']:.4f}")
    report.append(f"Remaining budget: ${summary['remaining_budget']:.4f}")
    report.append(f"Budget utilization: {summary['budget_utilization']:.1%}")
    report.append(f"Total requests: {summary['requests_today']}")
    
    if summary['requests_today'] > 0:
        avg_cost = summary['daily_spend'] / summary['requests_today']
        report.append(f"Average cost per request: ${avg_cost:.4f}")
    
    # Provider breakdown
    if summary['by_provider']:
        report.append("\\n🤖 COST BY PROVIDER")
        report.append("-" * 30)
        
        for provider, data in summary['by_provider'].items():
            report.append(f"{provider}:")
            report.append(f"  Total cost: ${data['total_cost']:.4f}")
            report.append(f"  Requests: {data['requests']}")
            report.append(f"  Avg per request: ${data['total_cost']/data['requests']:.4f}")
            report.append(f"  Total tokens: {data['total_tokens']:,}")
    
    # Model breakdown
    if summary['by_model']:
        report.append("\\n🔧 COST BY MODEL")
        report.append("-" * 30)
        
        # Sort by total cost
        sorted_models = sorted(summary['by_model'].items(), 
                             key=lambda x: x[1]['total_cost'], reverse=True)
        
        for model, data in sorted_models:
            report.append(f"{model}:")
            report.append(f"  Total cost: ${data['total_cost']:.4f}")
            report.append(f"  Requests: {data['requests']}")
            report.append(f"  Avg per request: ${data['avg_cost_per_request']:.4f}")
    
    # Optimization recommendations
    report.append("\\n💡 OPTIMIZATION RECOMMENDATIONS")
    report.append("-" * 40)
    
    if summary['budget_utilization'] > 0.8:
        report.append("⚠️  High budget utilization - consider:")
        report.append("   • Using more cost-effective models")
        report.append("   • Reducing max_tokens limits")
        report.append("   • Implementing response caching")
    
    if summary['by_provider']:
        # Find most cost-effective provider
        cheapest_provider = min(summary['by_provider'].items(),
                              key=lambda x: x[1]['total_cost'] / x[1]['requests'])
        report.append(f"💰 Most cost-effective provider: {cheapest_provider[0]}")
        
        # Find most expensive
        if len(summary['by_provider']) > 1:
            expensive_provider = max(summary['by_provider'].items(),
                                   key=lambda x: x[1]['total_cost'] / x[1]['requests'])
            report.append(f"💸 Most expensive provider: {expensive_provider[0]}")
    
    report.append("\\n📋 COST OPTIMIZATION TIPS:")
    report.append("• Use shorter prompts when possible")
    report.append("• Set appropriate max_tokens limits") 
    report.append("• Choose models based on task complexity")
    report.append("• Implement caching for repeated queries")
    report.append("• Monitor usage patterns and adjust budgets")
    
    return "\\n".join(report)

def save_cost_analysis(cost_tracker: CostTracker, output_dir: str = "examples/results"):
    """
    Save cost analysis data and report.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed cost data
    cost_data = {
        'daily_budget': cost_tracker.daily_budget,
        'costs_today': [
            {
                'provider': cost.provider,
                'model': cost.model,
                'input_tokens': cost.input_tokens,
                'output_tokens': cost.output_tokens,
                'input_cost': cost.input_cost,
                'output_cost': cost.output_cost,
                'total_cost': cost.total_cost,
                'timestamp': cost.timestamp
            }
            for cost in cost_tracker.costs_today
        ],
        'summary': cost_tracker.get_cost_summary()
    }
    
    json_file = Path(output_dir) / f"cost_analysis_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(cost_data, f, indent=2, default=str)
    
    # Save report
    report = generate_cost_report(cost_tracker)
    report_file = Path(output_dir) / f"cost_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\\n💾 Cost analysis saved:")
    print(f"  Data: {json_file}")
    print(f"  Report: {report_file}")

def main():
    """
    Main function demonstrating cost analysis and optimization.
    """
    print("💰 LLM Lab - Cost Analysis and Optimization")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        # Run cost optimization demonstration
        cost_tracker = demonstrate_cost_optimization()
        
        if cost_tracker:
            # Generate and display report
            report = generate_cost_report(cost_tracker)
            print(report)
            
            # Save analysis
            save_cost_analysis(cost_tracker)
            
            print("\\n🎉 Cost analysis complete!")
            print("\\n💡 Tips for ongoing cost management:")
            print("  • Run this analysis regularly to track spending patterns")
            print("  • Set daily/monthly budget alerts")
            print("  • Experiment with different models for different use cases")
            print("  • Consider caching responses for repeated queries")
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())