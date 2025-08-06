"""Command-line interface for interpretability suite."""

import json
import logging
from pathlib import Path
from typing import Optional

import click
import torch

from .analyzers import ActivationAnalyzer, AttentionAnalyzer, GradientAnalyzer
from .config import ConfigManager, InterpretabilityConfig
from .explanations import ExplanationGenerator, FeatureAttributor
from .visualizers import AttentionVisualizer, DashboardManager

logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config, verbose):
    """LLM Interpretability Suite - Comprehensive toolkit for understanding model behavior."""
    # Set up logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Load configuration
    config_manager = ConfigManager(config)
    ctx.obj = config_manager

    logger.info("Interpretability Suite initialized")


@cli.command()
@click.option("--model-path", "-m", required=True, help="Path to model checkpoint")
@click.option("--input-path", "-i", required=True, help="Path to input data")
@click.option("--output-dir", "-o", default="./analysis_output", help="Output directory")
@click.option(
    "--analysis-type",
    "-t",
    type=click.Choice(["attention", "gradient", "activation", "all"]),
    default="all",
    help="Type of analysis to perform",
)
@click.pass_obj
def analyze(config_manager, model_path, input_path, output_dir, analysis_type):
    """Analyze model behavior on given inputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    # Load inputs
    logger.info(f"Loading inputs from {input_path}")
    if input_path.endswith(".pt"):
        inputs = torch.load(input_path)
    elif input_path.endswith(".json"):
        with open(input_path) as f:
            inputs = json.load(f)
    else:
        raise ValueError(f"Unsupported input format: {input_path}")

    results = {}

    # Perform attention analysis
    if analysis_type in ["attention", "all"]:
        logger.info("Performing attention analysis...")
        analyzer = AttentionAnalyzer(model)
        patterns = analyzer.analyze_attention(inputs)

        # Save results
        attention_output = output_path / "attention_analysis.pt"
        torch.save({"patterns": patterns}, attention_output)
        results["attention"] = str(attention_output)

        # Create visualizations
        visualizer = AttentionVisualizer(style=config_manager.config.visualization.style.value)
        for i, pattern in enumerate(patterns[:5]):  # Visualize first 5
            fig = visualizer.plot_attention_heatmap(
                pattern.attention_weights,
                layer_name=pattern.layer_name,
                save_path=str(output_path / f"attention_{i}.png"),
            )

    # Perform gradient analysis
    if analysis_type in ["gradient", "all"]:
        logger.info("Performing gradient analysis...")
        analyzer = GradientAnalyzer(model)
        gradient_info = analyzer.compute_gradients(inputs)

        # Save results
        gradient_output = output_path / "gradient_analysis.pt"
        analyzer.export_gradients(str(gradient_output))
        results["gradient"] = str(gradient_output)

    # Perform activation analysis
    if analysis_type in ["activation", "all"]:
        logger.info("Performing activation analysis...")
        analyzer = ActivationAnalyzer(model)
        activation_stats = analyzer.analyze_activations(inputs)

        # Save results
        activation_output = output_path / "activation_analysis.pt"
        analyzer.export_activation_stats(str(activation_output))
        results["activation"] = str(activation_output)

    # Save summary
    summary_path = output_path / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Analysis complete. Results saved to {output_dir}")


@cli.command()
@click.option("--model-path", "-m", required=True, help="Path to model")
@click.option("--input-path", "-i", required=True, help="Path to input")
@click.option(
    "--method",
    "-M",
    type=click.Choice(
        [
            "integrated_gradients",
            "gradient_shap",
            "deeplift",
            "occlusion",
            "lime",
            "attention_rollout",
        ]
    ),
    default="integrated_gradients",
    help="Attribution method",
)
@click.option("--target-class", "-t", type=int, help="Target class for attribution")
@click.option("--output-path", "-o", default="attributions.npz", help="Output file")
@click.pass_obj
def attribute(config_manager, model_path, input_path, method, target_class, output_path):
    """Compute feature attributions for model predictions."""
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    # Load inputs
    logger.info(f"Loading inputs from {input_path}")
    inputs = torch.load(input_path) if input_path.endswith(".pt") else torch.tensor(inputs)

    # Compute attributions
    logger.info(f"Computing {method} attributions...")
    attributor = FeatureAttributor(model)
    result = attributor.compute_attributions(inputs, target=target_class, method=method)

    # Get top features
    top_features = attributor.get_top_features(result, top_k=10)
    logger.info("Top features:")
    for idx, score, name in top_features:
        logger.info(f"  {name or idx}: {score:.4f}")

    # Save results
    attributor.export_attributions(output_path)
    logger.info(f"Attributions saved to {output_path}")


@cli.command()
@click.option("--model-path", "-m", required=True, help="Path to model")
@click.option("--input-path", "-i", required=True, help="Path to input")
@click.option(
    "--style",
    "-s",
    type=click.Choice(["technical", "simple", "detailed"]),
    default="technical",
    help="Explanation style",
)
@click.option("--output-path", "-o", default="explanation.json", help="Output file")
@click.pass_obj
def explain(config_manager, model_path, input_path, style, output_path):
    """Generate natural language explanations for predictions."""
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    # Load inputs
    logger.info(f"Loading inputs from {input_path}")
    inputs = torch.load(input_path) if input_path.endswith(".pt") else torch.tensor(inputs)

    # Generate prediction
    with torch.no_grad():
        prediction = model(inputs)

    # Generate explanation
    logger.info("Generating explanation...")
    generator = ExplanationGenerator(model, explanation_style=style)
    explanation = generator.generate_explanation(inputs, prediction)

    # Display explanation
    click.echo("\nExplanation:")
    click.echo(explanation.natural_language)

    # Save explanation
    generator.export_explanations(output_path, format="json")
    logger.info(f"Explanation saved to {output_path}")


@cli.command()
@click.option("--data-dir", "-d", help="Directory with analysis data")
@click.option("--port", "-p", default=8050, help="Dashboard port")
@click.option("--host", "-h", default="127.0.0.1", help="Dashboard host")
@click.pass_obj
def dashboard(config_manager, data_dir, port, host):
    """Launch interactive dashboard for visualization."""
    logger.info(f"Starting dashboard on {host}:{port}")

    # Create dashboard
    dash_manager = DashboardManager(port=port, host=host)

    # Load data if provided
    if data_dir:
        data_path = Path(data_dir)

        # Load attention data
        attention_file = data_path / "attention_analysis.pt"
        if attention_file.exists():
            attention_data = torch.load(attention_file)
            dash_manager.update_data("attention", attention_data)

        # Load gradient data
        gradient_file = data_path / "gradient_analysis.pt"
        if gradient_file.exists():
            gradient_data = torch.load(gradient_file)
            dash_manager.update_data("gradient", gradient_data)

        # Load activation data
        activation_file = data_path / "activation_analysis.pt"
        if activation_file.exists():
            activation_data = torch.load(activation_file)
            dash_manager.update_data("activation", activation_data)

    # Run dashboard
    dash_manager.run(debug=config_manager.config.dashboard.debug)


@cli.command()
@click.option(
    "--preset",
    "-p",
    type=click.Choice(["minimal", "comprehensive", "fast", "research"]),
    required=True,
    help="Configuration preset",
)
@click.option("--output", "-o", default="config.yaml", help="Output configuration file")
@click.pass_obj
def init_config(config_manager, preset, output):
    """Initialize configuration file with preset."""
    logger.info(f"Creating {preset} configuration...")

    # Get preset configuration
    config = config_manager.get_preset(preset)

    # Create new manager with preset
    new_manager = ConfigManager()
    new_manager.config = config

    # Save configuration
    new_manager.save_config(output)
    logger.info(f"Configuration saved to {output}")

    # Validate configuration
    warnings = new_manager.validate_config()
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")


@cli.command()
@click.option("--input-dir", "-i", required=True, help="Directory with analysis results")
@click.option("--output-path", "-o", default="report.html", help="Output report path")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["html", "pdf", "markdown"]),
    default="html",
    help="Report format",
)
@click.pass_obj
def report(config_manager, input_dir, output_path, format):
    """Generate comprehensive interpretability report."""
    logger.info(f"Generating {format} report...")

    input_path = Path(input_dir)

    # Collect all analysis results
    report_data = {"title": "Model Interpretability Report", "sections": []}

    # Add attention analysis section
    attention_file = input_path / "attention_analysis.pt"
    if attention_file.exists():
        data = torch.load(attention_file)
        report_data["sections"].append(
            {
                "title": "Attention Analysis",
                "content": f"Analyzed {len(data.get('patterns', []))} attention patterns",
            }
        )

    # Add gradient analysis section
    gradient_file = input_path / "gradient_analysis.pt"
    if gradient_file.exists():
        data = torch.load(gradient_file)
        report_data["sections"].append(
            {
                "title": "Gradient Analysis",
                "content": f"Computed gradients for {len(data.get('gradient_info', {}))} layers",
            }
        )

    # Add activation analysis section
    activation_file = input_path / "activation_analysis.pt"
    if activation_file.exists():
        data = torch.load(activation_file)
        report_data["sections"].append(
            {
                "title": "Activation Analysis",
                "content": f"Analyzed activations for {len(data.get('activation_stats', {}))} layers",
            }
        )

    # Generate report based on format
    if format == "html":
        html_content = "<html><head><title>{title}</title></head><body>".format(**report_data)
        html_content += f"<h1>{report_data['title']}</h1>"
        for section in report_data["sections"]:
            html_content += f"<h2>{section['title']}</h2>"
            html_content += f"<p>{section['content']}</p>"
        html_content += "</body></html>"

        with open(output_path, "w") as f:
            f.write(html_content)

    elif format == "markdown":
        md_content = f"# {report_data['title']}\n\n"
        for section in report_data["sections"]:
            md_content += f"## {section['title']}\n\n"
            md_content += f"{section['content']}\n\n"

        with open(output_path, "w") as f:
            f.write(md_content)

    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    cli()
