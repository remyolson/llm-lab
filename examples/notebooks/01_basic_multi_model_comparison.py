#!/usr/bin/env python3
"""
Basic Multi-Model Comparison Example

This example demonstrates how to compare responses from multiple LLM providers
using the LLM Lab framework. It's designed as a Python script that can be
converted to a Jupyter notebook.

Usage:
    python examples/notebooks/01_basic_multi_model_comparison.py

Requirements:
    - API keys configured in .env file
    - At least one provider API key (Google, OpenAI, or Anthropic)
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_providers import GoogleProvider, OpenAIProvider, AnthropicProvider

def setup_providers():
    """
    Initialize available providers based on API keys.
    Returns a dictionary of available providers.
    """
    providers = {}
    
    # Google Gemini
    if os.getenv('GOOGLE_API_KEY'):
        try:
            providers['google'] = GoogleProvider(model_name="gemini-1.5-flash")
            print("✓ Google Gemini provider initialized")
        except Exception as e:
            print(f"✗ Google Gemini provider failed: {e}")
    else:
        print("⚠ Google API key not found, skipping Google provider")
    
    # OpenAI GPT
    if os.getenv('OPENAI_API_KEY'):
        try:
            providers['openai'] = OpenAIProvider(model_name="gpt-4o-mini")
            print("✓ OpenAI GPT provider initialized")
        except Exception as e:
            print(f"✗ OpenAI GPT provider failed: {e}")
    else:
        print("⚠ OpenAI API key not found, skipping OpenAI provider")
    
    # Anthropic Claude
    if os.getenv('ANTHROPIC_API_KEY'):
        try:
            providers['anthropic'] = AnthropicProvider(model_name="claude-3-5-haiku-20241022")
            print("✓ Anthropic Claude provider initialized")
        except Exception as e:
            print(f"✗ Anthropic Claude provider failed: {e}")
    else:
        print("⚠ Anthropic API key not found, skipping Anthropic provider")
    
    if not providers:
        raise ValueError("No providers available. Please configure at least one API key in your .env file.")
    
    return providers

def compare_responses(providers, prompt, max_retries=3):
    """
    Get responses from all available providers for a given prompt.
    
    Args:
        providers: Dictionary of initialized providers
        prompt: The prompt to send to all providers
        max_retries: Maximum number of retry attempts for failed requests
    
    Returns:
        Dictionary with provider names as keys and response data as values
    """
    print(f"\n🔄 Comparing responses for prompt: '{prompt[:50]}...'")
    print("=" * 60)
    
    results = {}
    
    for provider_name, provider in providers.items():
        print(f"\n📤 Requesting from {provider_name}...")
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = provider.generate(prompt)
                end_time = time.time()
                
                results[provider_name] = {
                    'response': response,
                    'response_time': end_time - start_time,
                    'success': True,
                    'error': None,
                    'attempt': attempt + 1
                }
                
                print(f"✓ {provider_name} responded in {end_time - start_time:.2f}s")
                break
                
            except Exception as e:
                print(f"✗ {provider_name} attempt {attempt + 1} failed: {str(e)[:100]}...")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    results[provider_name] = {
                        'response': None,
                        'response_time': None,
                        'success': False,
                        'error': str(e),
                        'attempt': attempt + 1
                    }
    
    return results

def analyze_responses(results):
    """
    Analyze and compare the responses from different providers.
    
    Args:
        results: Dictionary of provider results from compare_responses()
    
    Returns:
        Dictionary with analysis metrics
    """
    analysis = {
        'successful_providers': [],
        'failed_providers': [],
        'response_times': {},
        'response_lengths': {},
        'fastest_provider': None,
        'slowest_provider': None,
        'average_response_time': 0
    }
    
    successful_times = []
    
    for provider_name, result in results.items():
        if result['success']:
            analysis['successful_providers'].append(provider_name)
            analysis['response_times'][provider_name] = result['response_time']
            analysis['response_lengths'][provider_name] = len(result['response'])
            successful_times.append(result['response_time'])
        else:
            analysis['failed_providers'].append(provider_name)
    
    if successful_times:
        analysis['average_response_time'] = sum(successful_times) / len(successful_times)
        
        # Find fastest and slowest
        fastest_time = min(analysis['response_times'].values())
        slowest_time = max(analysis['response_times'].values())
        
        for provider, time_taken in analysis['response_times'].items():
            if time_taken == fastest_time:
                analysis['fastest_provider'] = provider
            if time_taken == slowest_time:
                analysis['slowest_provider'] = provider
    
    return analysis

def print_comparison_results(prompt, results, analysis):
    """
    Print a formatted comparison of the results.
    """
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Successful providers: {len(analysis['successful_providers'])}")
    print(f"Failed providers: {len(analysis['failed_providers'])}")
    
    if analysis['successful_providers']:
        print(f"\\nPerformance Summary:")
        print(f"├── Fastest: {analysis['fastest_provider']} ({analysis['response_times'][analysis['fastest_provider']]:.2f}s)")
        print(f"├── Slowest: {analysis['slowest_provider']} ({analysis['response_times'][analysis['slowest_provider']]:.2f}s)")
        print(f"└── Average: {analysis['average_response_time']:.2f}s")
    
    print(f"\\n📝 Detailed Results:")
    print("-" * 40)
    
    for provider_name, result in results.items():
        print(f"\\n🤖 {provider_name.upper()}")
        
        if result['success']:
            print(f"  Status: ✓ Success")
            print(f"  Response time: {result['response_time']:.2f}s")
            print(f"  Response length: {len(result['response'])} characters")
            print(f"  Response preview: {result['response'][:150]}...")
        else:
            print(f"  Status: ✗ Failed")
            print(f"  Error: {result['error']}")
            print(f"  Attempts: {result['attempt']}")

def save_results(prompt, results, analysis, output_dir="examples/results"):
    """
    Save the comparison results to a JSON file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"comparison_{timestamp}.json"
    filepath = Path(output_dir) / filename
    
    data = {
        'prompt': prompt,
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'analysis': analysis
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"\\n💾 Results saved to: {filepath}")
    return filepath

def main():
    """
    Main function demonstrating multi-model comparison.
    """
    print("🚀 LLM Lab - Multi-Model Comparison Example")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        # Setup providers
        print("\\n🔧 Setting up providers...")
        providers = setup_providers()
        
        # Example prompts for comparison
        example_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "What are the key benefits of renewable energy?",
            "Write a short Python function to calculate fibonacci numbers.",
            "Describe the main causes of climate change.",
            "What is the difference between artificial intelligence and machine learning?"
        ]
        
        print(f"\\n📋 Available example prompts:")
        for i, prompt in enumerate(example_prompts, 1):
            print(f"  {i}. {prompt}")
        
        # Interactive prompt selection or custom input
        print("\\n" + "=" * 50)
        choice = input("Choose a prompt (1-5) or press Enter for custom prompt: ").strip()
        
        if choice and choice.isdigit() and 1 <= int(choice) <= len(example_prompts):
            selected_prompt = example_prompts[int(choice) - 1]
        else:
            selected_prompt = input("Enter your custom prompt: ").strip()
            if not selected_prompt:
                selected_prompt = example_prompts[0]  # Default
        
        print(f"\\n🎯 Selected prompt: {selected_prompt}")
        
        # Run comparison
        results = compare_responses(providers, selected_prompt)
        analysis = analyze_responses(results)
        
        # Display results
        print_comparison_results(selected_prompt, results, analysis)
        
        # Save results
        save_results(selected_prompt, results, analysis)
        
        # Summary recommendations
        print("\\n💡 RECOMMENDATIONS")
        print("-" * 30)
        
        if analysis['successful_providers']:
            if len(analysis['successful_providers']) > 1:
                print(f"✓ Multiple providers are working well")
                print(f"✓ Consider using {analysis['fastest_provider']} for speed-critical applications")
                
                # Find longest response
                longest_provider = max(analysis['response_lengths'], key=analysis['response_lengths'].get)
                print(f"✓ Consider using {longest_provider} for detailed responses")
            else:
                provider = analysis['successful_providers'][0]
                print(f"✓ {provider} is working well")
                print(f"✓ Consider adding more providers for comparison")
        
        if analysis['failed_providers']:
            print(f"⚠ Failed providers: {', '.join(analysis['failed_providers'])}")
            print("⚠ Check API keys and rate limits for failed providers")
        
        print("\\n🎉 Comparison complete!")
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())