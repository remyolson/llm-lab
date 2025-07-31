#!/usr/bin/env python3
"""
Results Comparison CLI

This tool provides command-line access to compare benchmark results across
multiple LLM models and generate comprehensive reports.
"""

import click
import json
from pathlib import Path
from typing import List, Optional

from src.analysis.comparator import ResultsComparator


@click.command()
@click.option(
    '--models',
    type=str,
    help='Comma-separated list of models to compare (e.g., gpt-4,claude-3-opus)'
)
@click.option(
    '--dataset',
    type=str,
    help='Dataset name to filter results'
)
@click.option(
    '--results-dir',
    type=click.Path(exists=True),
    default='./results',
    help='Directory containing result files'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default='./comparison_reports',
    help='Directory to save comparison outputs'
)
@click.option(
    '--format',
    type=click.Choice(['json', 'csv'], case_sensitive=False),
    default='json',
    help='Input file format to load'
)
@click.option(
    '--include-all-prompts',
    is_flag=True,
    default=False,
    help='Include all prompts in CSV (not just aligned ones)'
)
@click.option(
    '--skip-visualizations',
    is_flag=True,
    default=False,
    help='Skip generating visualization charts'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output'
)
def main(models: Optional[str], dataset: Optional[str], results_dir: str, 
         output_dir: str, format: str, include_all_prompts: bool,
         skip_visualizations: bool, verbose: bool):
    """
    Compare benchmark results across multiple LLM models.
    
    This tool loads benchmark results, performs statistical analysis,
    and generates comprehensive reports including:
    - Side-by-side CSV comparisons
    - Metrics summary with confidence intervals
    - Statistical significance tests
    - Markdown analysis report
    - Visualization charts
    
    Examples:
        # Compare specific models
        python compare_results.py --models gpt-4,claude-3-opus --dataset truthfulness
        
        # Compare all available models
        python compare_results.py --dataset truthfulness
        
        # Load CSV results instead of JSON
        python compare_results.py --format csv
    """
    click.echo("🔬 LLM Lab Results Comparator")
    click.echo("=" * 50)
    
    # Parse model list
    model_list = None
    if models:
        model_list = [m.strip() for m in models.split(',') if m.strip()]
        click.echo(f"Models to compare: {', '.join(model_list)}")
    else:
        click.echo("Comparing all available models")
    
    if dataset:
        click.echo(f"Dataset filter: {dataset}")
    
    # Initialize comparator
    comparator = ResultsComparator(results_dir)
    
    try:
        # Load results
        click.echo(f"\n📂 Loading results from {results_dir}...")
        loaded_models = comparator.load_results(model_list, dataset, format)
        
        if not loaded_models:
            click.echo("❌ No results found matching criteria", err=True)
            return
        
        click.echo(f"✓ Loaded results for {len(loaded_models)} models:")
        for model_name, model_result in loaded_models.items():
            click.echo(f"  - {model_name}: {model_result.total_prompts} prompts")
        
        # Perform comparison
        click.echo("\n🔄 Performing comparison analysis...")
        comparison_result = comparator.compare(model_list, dataset)
        
        click.echo(f"✓ Aligned {len(comparison_result.prompt_alignment)} prompts across all models")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate outputs
        click.echo(f"\n📝 Generating reports in {output_dir}...")
        
        # 1. Comparison CSV
        comparison_csv_path = output_path / 'model_comparison.csv'
        comparator.generate_comparison_csv(comparison_csv_path, include_all_prompts)
        click.echo(f"  ✓ Side-by-side comparison: {comparison_csv_path}")
        
        # 2. Metrics CSV
        metrics_csv_path = output_path / 'metrics_summary.csv'
        comparator.generate_metrics_csv(metrics_csv_path)
        click.echo(f"  ✓ Metrics summary: {metrics_csv_path}")
        
        # 3. Statistical tests CSV
        stats_csv_path = output_path / 'statistical_tests.csv'
        comparator.generate_statistical_tests_csv(stats_csv_path)
        click.echo(f"  ✓ Statistical tests: {stats_csv_path}")
        
        # 4. Markdown report
        report_path = output_path / 'comparison_report.md'
        comparator.generate_markdown_report(report_path)
        click.echo(f"  ✓ Analysis report: {report_path}")
        
        # 5. Visualizations
        if not skip_visualizations:
            click.echo("\n📊 Generating visualizations...")
            viz_dir = output_path / 'visualizations'
            viz_files = comparator.generate_visualizations(viz_dir)
            click.echo(f"  ✓ Generated {len(viz_files)} charts in {viz_dir}")
            
            if verbose:
                for viz_file in viz_files:
                    click.echo(f"    - {Path(viz_file).name}")
        
        # 6. JSON summary
        summary_path = output_path / 'comparison_summary.json'
        summary_data = {
            'models': comparison_result.models,
            'dataset': comparison_result.dataset,
            'timestamp': comparison_result.comparison_timestamp,
            'total_aligned_prompts': len(comparison_result.prompt_alignment),
            'metrics': comparison_result.metrics,
            'rankings': comparison_result.rankings,
            'statistical_tests': comparison_result.statistical_tests
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        click.echo(f"  ✓ JSON summary: {summary_path}")
        
        # Display summary
        click.echo("\n" + "=" * 50)
        click.echo("📊 Comparison Summary")
        click.echo("=" * 50)
        
        # Show top model for each metric
        for metric, ranking in comparison_result.rankings.items():
            if ranking:
                best_model = ranking[0]
                score = comparison_result.metrics[metric][best_model]
                
                formatted_metric = metric.replace('_', ' ').title()
                if metric == 'average_response_time':
                    click.echo(f"{formatted_metric}: {best_model} ({score:.2f}s)")
                else:
                    click.echo(f"{formatted_metric}: {best_model} ({score:.2%})")
        
        # Show significant differences
        if 'pairwise_success_tests' in comparison_result.statistical_tests:
            sig_pairs = []
            for pair_key, test_result in comparison_result.statistical_tests['pairwise_success_tests'].items():
                if test_result['is_significant']:
                    sig_pairs.append(pair_key.replace('_vs_', ' vs '))
            
            if sig_pairs:
                click.echo(f"\n⚡ Significant differences found between:")
                for pair in sig_pairs:
                    click.echo(f"  - {pair}")
        
        click.echo(f"\n✅ All reports saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()


if __name__ == '__main__':
    main()