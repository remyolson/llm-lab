#!/usr/bin/env python3
"""
Complete Benchmarking Workflow Example

This example demonstrates a complete benchmarking workflow using the LLM Lab
framework, including dataset preparation, multi-model execution, results analysis,
and report generation.

Usage:
    python examples/use_cases/benchmarking_workflow.py

Features:
    - Custom dataset creation and validation
    - Multi-provider benchmarking with error handling
    - Statistical analysis of results
    - Performance comparison across models
    - Automated report generation
    - Results export in multiple formats
"""

import os
import sys
import csv
import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_providers import GoogleProvider, OpenAIProvider, AnthropicProvider
from evaluation.keyword_match import evaluate as keyword_evaluate

@dataclass
class BenchmarkPrompt:
    """Data class for benchmark prompts."""
    id: str
    prompt: str
    expected_keywords: List[str]
    category: str
    difficulty: str
    evaluation_method: str = "keyword_match"

@dataclass
class BenchmarkResult:
    """Data class for benchmark results."""
    prompt_id: str
    provider: str
    model: str
    prompt: str
    response: str
    expected_keywords: List[str]
    matched_keywords: List[str]
    score: float
    success: bool
    response_time: float
    error: Optional[str]
    timestamp: str
    category: str
    difficulty: str

class BenchmarkDataset:
    """
    Manages benchmark datasets with validation and categorization.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.prompts: List[BenchmarkPrompt] = []
    
    def add_prompt(self, prompt_id: str, prompt: str, expected_keywords: List[str], 
                   category: str = "general", difficulty: str = "medium"):
        """Add a prompt to the dataset."""
        benchmark_prompt = BenchmarkPrompt(
            id=prompt_id,
            prompt=prompt,
            expected_keywords=expected_keywords,
            category=category,
            difficulty=difficulty
        )
        self.prompts.append(benchmark_prompt)
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the dataset and return statistics."""
        validation = {
            'total_prompts': len(self.prompts),
            'categories': {},
            'difficulties': {},
            'average_prompt_length': 0,
            'average_keywords_per_prompt': 0,
            'issues': []
        }
        
        if not self.prompts:
            validation['issues'].append("Dataset is empty")
            return validation
        
        prompt_lengths = []
        keyword_counts = []
        
        for prompt in self.prompts:
            # Track categories and difficulties
            validation['categories'][prompt.category] = validation['categories'].get(prompt.category, 0) + 1
            validation['difficulties'][prompt.difficulty] = validation['difficulties'].get(prompt.difficulty, 0) + 1
            
            # Track lengths
            prompt_lengths.append(len(prompt.prompt))
            keyword_counts.append(len(prompt.expected_keywords))
            
            # Check for issues
            if len(prompt.prompt.strip()) < 10:
                validation['issues'].append(f"Prompt {prompt.id} is very short")
            
            if len(prompt.expected_keywords) == 0:
                validation['issues'].append(f"Prompt {prompt.id} has no expected keywords")
        
        validation['average_prompt_length'] = statistics.mean(prompt_lengths)
        validation['average_keywords_per_prompt'] = statistics.mean(keyword_counts)
        
        return validation
    
    def get_prompts_by_category(self, category: str) -> List[BenchmarkPrompt]:
        """Get all prompts in a specific category."""
        return [p for p in self.prompts if p.category == category]
    
    def get_prompts_by_difficulty(self, difficulty: str) -> List[BenchmarkPrompt]:
        """Get all prompts of a specific difficulty."""
        return [p for p in self.prompts if p.difficulty == difficulty]

def create_sample_dataset() -> BenchmarkDataset:
    """
    Create a sample benchmark dataset for demonstration.
    """
    dataset = BenchmarkDataset("LLM Knowledge Benchmark")
    
    # Technology category
    dataset.add_prompt(
        "tech_001",
        "What is machine learning and how does it differ from traditional programming?",
        ["machine learning", "algorithms", "data", "patterns", "traditional programming", "rules"],
        "technology",
        "medium"
    )
    
    dataset.add_prompt(
        "tech_002", 
        "Explain the concept of cloud computing and list three major cloud service providers.",
        ["cloud computing", "internet", "servers", "Amazon", "Microsoft", "Google", "AWS", "Azure"],
        "technology",
        "easy"
    )
    
    dataset.add_prompt(
        "tech_003",
        "Describe the key principles of blockchain technology and its applications beyond cryptocurrency.",
        ["blockchain", "decentralized", "immutable", "cryptography", "supply chain", "smart contracts"],
        "technology", 
        "hard"
    )
    
    # Science category
    dataset.add_prompt(
        "sci_001",
        "What causes climate change and what are its main effects on the environment?",
        ["greenhouse gases", "carbon dioxide", "global warming", "temperature", "sea level", "weather patterns"],
        "science",
        "medium"
    )
    
    dataset.add_prompt(
        "sci_002",
        "Explain photosynthesis in simple terms.",
        ["photosynthesis", "plants", "sunlight", "carbon dioxide", "oxygen", "glucose", "chlorophyll"],
        "science",
        "easy"
    )
    
    dataset.add_prompt(
        "sci_003",
        "Describe the process of DNA replication and explain why it's important for cell division.",
        ["DNA", "replication", "nucleotides", "polymerase", "cell division", "genetic information"],
        "science",
        "hard"
    )
    
    # Mathematics category
    dataset.add_prompt(
        "math_001",
        "What is the Pythagorean theorem and when is it used?",
        ["Pythagorean theorem", "right triangle", "hypotenuse", "square", "geometry"],
        "mathematics",
        "easy"
    )
    
    dataset.add_prompt(
        "math_002",
        "Explain the concept of derivatives in calculus and provide a simple example.",
        ["derivatives", "calculus", "rate of change", "slope", "function", "limit"],
        "mathematics",
        "hard"
    )
    
    return dataset

class BenchmarkRunner:
    """
    Runs benchmarks across multiple providers with error handling and progress tracking.
    """
    
    def __init__(self, providers: Dict[str, Any], max_retries: int = 2):
        self.providers = providers
        self.max_retries = max_retries
        self.results: List[BenchmarkResult] = []
    
    def run_single_benchmark(self, provider_name: str, provider: Any, prompt: BenchmarkPrompt) -> BenchmarkResult:
        """
        Run a single benchmark test.
        
        Args:
            provider_name: Name of the provider
            provider: Provider instance
            prompt: BenchmarkPrompt to test
        
        Returns:
            BenchmarkResult with the test outcome
        """
        model_name = getattr(provider, 'model_name', 'unknown')
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                response = provider.generate(prompt.prompt)
                end_time = time.time()
                
                # Evaluate the response
                evaluation = keyword_evaluate(
                    prompt=prompt.prompt,
                    response=response,
                    expected={"expected_keywords": prompt.expected_keywords}
                )
                
                return BenchmarkResult(
                    prompt_id=prompt.id,
                    provider=provider_name,
                    model=model_name,
                    prompt=prompt.prompt,
                    response=response,
                    expected_keywords=prompt.expected_keywords,
                    matched_keywords=evaluation.get('matched_keywords', []),
                    score=evaluation.get('score', 0.0),
                    success=True,
                    response_time=end_time - start_time,
                    error=None,
                    timestamp=datetime.now().isoformat(),
                    category=prompt.category,
                    difficulty=prompt.difficulty
                )
                
            except Exception as e:
                if attempt < self.max_retries:
                    print(f"  Retry {attempt + 1} for {provider_name} on {prompt.id}: {str(e)[:50]}...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return BenchmarkResult(
                        prompt_id=prompt.id,
                        provider=provider_name,
                        model=model_name,
                        prompt=prompt.prompt,
                        response="",
                        expected_keywords=prompt.expected_keywords,
                        matched_keywords=[],
                        score=0.0,
                        success=False,
                        response_time=0.0,
                        error=str(e),
                        timestamp=datetime.now().isoformat(),
                        category=prompt.category,
                        difficulty=prompt.difficulty
                    )
    
    def run_full_benchmark(self, dataset: BenchmarkDataset) -> List[BenchmarkResult]:
        """
        Run the complete benchmark across all providers and prompts.
        
        Args:
            dataset: BenchmarkDataset to run
        
        Returns:
            List of BenchmarkResult objects
        """
        total_tests = len(self.providers) * len(dataset.prompts)
        completed_tests = 0
        
        print(f"🚀 Starting benchmark with {len(self.providers)} providers and {len(dataset.prompts)} prompts")
        print(f"📊 Total tests to run: {total_tests}")
        print("-" * 60)
        
        for provider_name, provider in self.providers.items():
            print(f"\\n🤖 Testing {provider_name}...")
            
            for prompt in dataset.prompts:
                print(f"  📝 {prompt.id} ({prompt.category}/{prompt.difficulty}): {prompt.prompt[:50]}...")
                
                result = self.run_single_benchmark(provider_name, provider, prompt)
                self.results.append(result)
                
                completed_tests += 1
                progress = (completed_tests / total_tests) * 100
                
                if result.success:
                    print(f"    ✓ Score: {result.score:.2f}, Time: {result.response_time:.2f}s [{progress:.1f}%]")
                else:
                    print(f"    ✗ Failed: {result.error[:50]}... [{progress:.1f}%]")
                
                # Rate limiting to be respectful to APIs
                time.sleep(0.5)
        
        print(f"\\n🎉 Benchmark complete! {len(self.results)} tests finished.")
        return self.results

class BenchmarkAnalyzer:
    """
    Analyzes benchmark results and generates insights.
    """
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.successful_results = [r for r in results if r.success]
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall benchmark statistics."""
        total_tests = len(self.results)
        successful_tests = len(self.successful_results)
        
        if not self.successful_results:
            return {
                'total_tests': total_tests,
                'successful_tests': 0,
                'success_rate': 0.0,
                'average_score': 0.0,
                'average_response_time': 0.0
            }
        
        scores = [r.score for r in self.successful_results]
        response_times = [r.response_time for r in self.successful_results]
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests,
            'average_score': statistics.mean(scores),
            'median_score': statistics.median(scores),
            'score_std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
            'average_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'response_time_std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
    
    def analyze_by_provider(self) -> Dict[str, Dict[str, Any]]:
        """Analyze results grouped by provider."""
        provider_analysis = {}
        
        for result in self.results:
            if result.provider not in provider_analysis:
                provider_analysis[result.provider] = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'scores': [],
                    'response_times': [],
                    'categories': {},
                    'difficulties': {}
                }
            
            analysis = provider_analysis[result.provider]
            analysis['total_tests'] += 1
            
            if result.success:
                analysis['successful_tests'] += 1
                analysis['scores'].append(result.score)
                analysis['response_times'].append(result.response_time)
            
            # Track by category and difficulty
            category = result.category
            difficulty = result.difficulty
            
            if category not in analysis['categories']:
                analysis['categories'][category] = {'total': 0, 'successful': 0, 'scores': []}
            analysis['categories'][category]['total'] += 1
            if result.success:
                analysis['categories'][category]['successful'] += 1
                analysis['categories'][category]['scores'].append(result.score)
            
            if difficulty not in analysis['difficulties']:
                analysis['difficulties'][difficulty] = {'total': 0, 'successful': 0, 'scores': []}
            analysis['difficulties'][difficulty]['total'] += 1
            if result.success:
                analysis['difficulties'][difficulty]['successful'] += 1
                analysis['difficulties'][difficulty]['scores'].append(result.score)
        
        # Calculate summary statistics for each provider
        for provider, analysis in provider_analysis.items():
            analysis['success_rate'] = analysis['successful_tests'] / analysis['total_tests'] if analysis['total_tests'] > 0 else 0
            
            if analysis['scores']:
                analysis['average_score'] = statistics.mean(analysis['scores'])
                analysis['median_score'] = statistics.median(analysis['scores'])
            else:
                analysis['average_score'] = 0.0
                analysis['median_score'] = 0.0
            
            if analysis['response_times']:
                analysis['average_response_time'] = statistics.mean(analysis['response_times'])
                analysis['median_response_time'] = statistics.median(analysis['response_times'])
            else:
                analysis['average_response_time'] = 0.0
                analysis['median_response_time'] = 0.0
        
        return provider_analysis
    
    def find_best_performers(self) -> Dict[str, str]:
        """Find the best performing providers in different categories."""
        provider_analysis = self.analyze_by_provider()
        
        if not provider_analysis:
            return {}
        
        best_performers = {}
        
        # Best overall score
        best_score_provider = max(provider_analysis.items(), 
                                key=lambda x: x[1]['average_score'])
        best_performers['highest_average_score'] = best_score_provider[0]
        
        # Fastest response time
        best_speed_provider = min(provider_analysis.items(),
                                key=lambda x: x[1]['average_response_time'] if x[1]['average_response_time'] > 0 else float('inf'))
        best_performers['fastest_response'] = best_speed_provider[0]
        
        # Most reliable (highest success rate)
        best_reliability_provider = max(provider_analysis.items(),
                                      key=lambda x: x[1]['success_rate'])
        best_performers['most_reliable'] = best_reliability_provider[0]
        
        return best_performers

def generate_benchmark_report(analyzer: BenchmarkAnalyzer, dataset: BenchmarkDataset) -> str:
    """
    Generate a comprehensive benchmark report.
    """
    overall_stats = analyzer.get_overall_statistics()
    provider_analysis = analyzer.analyze_by_provider()
    best_performers = analyzer.find_best_performers()
    dataset_validation = dataset.validate_dataset()
    
    report = []
    report.append("📊 COMPREHENSIVE BENCHMARK REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {dataset.name}")
    report.append("")
    
    # Dataset overview
    report.append("📋 DATASET OVERVIEW")
    report.append("-" * 30)
    report.append(f"Total prompts: {dataset_validation['total_prompts']}")
    report.append(f"Categories: {', '.join(dataset_validation['categories'].keys())}")
    report.append(f"Difficulties: {', '.join(dataset_validation['difficulties'].keys())}")
    report.append(f"Average prompt length: {dataset_validation['average_prompt_length']:.0f} characters")
    report.append(f"Average keywords per prompt: {dataset_validation['average_keywords_per_prompt']:.1f}")
    
    # Overall results
    report.append("\\n🎯 OVERALL RESULTS")
    report.append("-" * 30)
    report.append(f"Total tests: {overall_stats['total_tests']}")
    report.append(f"Successful tests: {overall_stats['successful_tests']}")
    report.append(f"Success rate: {overall_stats['success_rate']:.1%}")
    report.append(f"Average score: {overall_stats['average_score']:.2f}")
    report.append(f"Average response time: {overall_stats['average_response_time']:.2f}s")
    
    # Best performers
    if best_performers:
        report.append("\\n🏆 BEST PERFORMERS")
        report.append("-" * 30)
        report.append(f"Highest average score: {best_performers.get('highest_average_score', 'N/A')}")
        report.append(f"Fastest response: {best_performers.get('fastest_response', 'N/A')}")
        report.append(f"Most reliable: {best_performers.get('most_reliable', 'N/A')}")
    
    # Provider comparison
    report.append("\\n🤖 PROVIDER COMPARISON")
    report.append("-" * 50)
    report.append(f"{'Provider':<15} {'Success':<8} {'Avg Score':<10} {'Avg Time':<10}")
    report.append("-" * 50)
    
    for provider, analysis in provider_analysis.items():
        success_rate = f"{analysis['success_rate']:.1%}"
        avg_score = f"{analysis['average_score']:.2f}"
        avg_time = f"{analysis['average_response_time']:.2f}s"
        report.append(f"{provider:<15} {success_rate:<8} {avg_score:<10} {avg_time:<10}")
    
    # Detailed provider analysis
    for provider, analysis in provider_analysis.items():
        report.append(f"\\n📈 {provider.upper()} DETAILED ANALYSIS")
        report.append("-" * 40)
        report.append(f"Success rate: {analysis['success_rate']:.1%} ({analysis['successful_tests']}/{analysis['total_tests']})")
        
        if analysis['scores']:
            report.append(f"Score statistics:")
            report.append(f"  Average: {analysis['average_score']:.2f}")
            report.append(f"  Median: {analysis['median_score']:.2f}")
            report.append(f"  Range: {min(analysis['scores']):.2f} - {max(analysis['scores']):.2f}")
        
        if analysis['response_times']:
            report.append(f"Response time statistics:")
            report.append(f"  Average: {analysis['average_response_time']:.2f}s")
            report.append(f"  Median: {analysis['median_response_time']:.2f}s")
            report.append(f"  Range: {min(analysis['response_times']):.2f}s - {max(analysis['response_times']):.2f}s")
        
        # Category performance
        report.append("Performance by category:")
        for category, cat_data in analysis['categories'].items():
            if cat_data['scores']:
                avg_score = statistics.mean(cat_data['scores'])
                success_rate = cat_data['successful'] / cat_data['total']
                report.append(f"  {category}: {success_rate:.1%} success, {avg_score:.2f} avg score")
        
        # Difficulty performance
        report.append("Performance by difficulty:")
        for difficulty, diff_data in analysis['difficulties'].items():
            if diff_data['scores']:
                avg_score = statistics.mean(diff_data['scores'])
                success_rate = diff_data['successful'] / diff_data['total']
                report.append(f"  {difficulty}: {success_rate:.1%} success, {avg_score:.2f} avg score")
    
    # Recommendations
    report.append("\\n💡 RECOMMENDATIONS")
    report.append("-" * 30)
    
    if len(provider_analysis) > 1:
        best_overall = max(provider_analysis.items(), key=lambda x: x[1]['average_score'])
        report.append(f"• For best accuracy: Use {best_overall[0]} (avg score: {best_overall[1]['average_score']:.2f})")
        
        fastest = min(provider_analysis.items(), key=lambda x: x[1]['average_response_time'] if x[1]['average_response_time'] > 0 else float('inf'))
        report.append(f"• For speed: Use {fastest[0]} (avg time: {fastest[1]['average_response_time']:.2f}s)")
        
        most_reliable = max(provider_analysis.items(), key=lambda x: x[1]['success_rate'])
        report.append(f"• For reliability: Use {most_reliable[0]} (success rate: {most_reliable[1]['success_rate']:.1%})")
    
    report.append("\\n📋 NEXT STEPS:")
    report.append("• Expand dataset with more diverse prompts")
    report.append("• Test with different model configurations")
    report.append("• Implement automated evaluation methods")
    report.append("• Set up continuous benchmarking pipeline")
    
    return "\\n".join(report)

def export_results(results: List[BenchmarkResult], dataset: BenchmarkDataset, 
                  output_dir: str = "examples/results") -> Dict[str, Path]:
    """
    Export benchmark results in multiple formats.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exported_files = {}
    
    # Export as CSV
    csv_file = Path(output_dir) / f"benchmark_results_{timestamp}.csv"
    with open(csv_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
    exported_files['csv'] = csv_file
    
    # Export as JSON
    json_file = Path(output_dir) / f"benchmark_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'dataset': {
                'name': dataset.name,
                'prompts': [asdict(p) for p in dataset.prompts]
            },
            'results': [asdict(r) for r in results],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(results),
                'successful_tests': sum(1 for r in results if r.success)
            }
        }, f, indent=2, default=str)
    exported_files['json'] = json_file
    
    return exported_files

def setup_providers() -> Dict[str, Any]:
    """Setup benchmark providers."""
    providers = {}
    
    if os.getenv('GOOGLE_API_KEY'):
        providers['google'] = GoogleProvider(model_name="gemini-1.5-flash")
    
    if os.getenv('OPENAI_API_KEY'):
        providers['openai'] = OpenAIProvider(model_name="gpt-4o-mini")
    
    if os.getenv('ANTHROPIC_API_KEY'):
        providers['anthropic'] = AnthropicProvider(model_name="claude-3-5-haiku-20241022")
    
    return providers

def main():
    """
    Main function demonstrating the complete benchmarking workflow.
    """
    print("🚀 LLM Lab - Complete Benchmarking Workflow")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        # Step 1: Create and validate dataset
        print("\\n📋 Step 1: Creating benchmark dataset...")
        dataset = create_sample_dataset()
        validation = dataset.validate_dataset()
        
        print(f"✓ Created dataset '{dataset.name}' with {validation['total_prompts']} prompts")
        print(f"✓ Categories: {list(validation['categories'].keys())}")
        print(f"✓ Difficulties: {list(validation['difficulties'].keys())}")
        
        if validation['issues']:
            print(f"⚠️  Dataset issues: {len(validation['issues'])}")
            for issue in validation['issues'][:3]:  # Show first 3 issues
                print(f"   • {issue}")
        
        # Step 2: Setup providers
        print("\\n🔧 Step 2: Setting up providers...")
        providers = setup_providers()
        
        if not providers:
            raise ValueError("No providers available. Please configure API keys in .env file.")
        
        print(f"✓ Initialized {len(providers)} providers: {list(providers.keys())}")
        
        # Confirm before proceeding (this will make API calls)
        print(f"\\n⚠️  This will run {len(providers) * len(dataset.prompts)} API calls")
        confirm = input("Proceed with benchmark? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Benchmark cancelled.")
            return 0
        
        # Step 3: Run benchmark
        print("\\n🏃 Step 3: Running benchmark...")
        runner = BenchmarkRunner(providers, max_retries=2)
        results = runner.run_full_benchmark(dataset)
        
        # Step 4: Analyze results
        print("\\n📊 Step 4: Analyzing results...")
        analyzer = BenchmarkAnalyzer(results)
        
        # Step 5: Generate report
        print("\\n📈 Step 5: Generating report...")
        report = generate_benchmark_report(analyzer, dataset)
        print(report)
        
        # Step 6: Export results
        print("\\n💾 Step 6: Exporting results...")
        exported_files = export_results(results, dataset)
        
        print("\\nExported files:")
        for format_type, file_path in exported_files.items():
            print(f"  {format_type.upper()}: {file_path}")
        
        # Save report
        report_file = Path("examples/results") / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"  REPORT: {report_file}")
        
        print("\\n🎉 Benchmarking workflow complete!")
        print("\\n💡 Next steps:")
        print("  • Review the detailed report for insights")
        print("  • Use the exported data for further analysis")
        print("  • Consider expanding the dataset")
        print("  • Set up automated benchmarking for continuous monitoring")
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())