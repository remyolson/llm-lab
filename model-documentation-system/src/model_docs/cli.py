"""Command-line interface for model documentation system."""

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .analyzers import ModelLoader
from .generators import ComplianceGenerator, ModelCardGenerator

console = Console()
logging.basicConfig(level=logging.INFO)


@click.group()
def cli():
    """Model Documentation System CLI."""
    pass


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="model_card.md", help="Output file path")
@click.option("--format", "-f", type=click.Choice(["markdown", "json", "pdf"]), default="markdown")
@click.option("--framework", help="Model framework (auto-detected if not specified)")
def generate(model_path, output, format, framework):
    """Generate model documentation from a model file."""
    console.print(f"[bold blue]Generating documentation for:[/bold blue] {model_path}")

    try:
        # Load model
        model = ModelLoader.load_model(model_path, framework)

        # Generate model card
        generator = ModelCardGenerator()
        model_card = generator.generate_model_card(model)

        # Save documentation
        output_path = generator.save_model_card(model_card, Path(output), format)

        console.print(f"[bold green]✓ Documentation saved to:[/bold green] {output_path}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.argument("model_card", type=click.Path(exists=True))
@click.option("--framework", "-f", multiple=True, default=["eu_ai_act", "model_cards"])
def validate(model_card, framework):
    """Validate model documentation against compliance frameworks."""
    console.print(f"[bold blue]Validating:[/bold blue] {model_card}")

    try:
        import json

        # Load model card
        with open(model_card, "r") as f:
            if model_card.endswith(".json"):
                card_data = json.load(f)
                from .models import ModelCard

                card = ModelCard(**card_data)
            else:
                console.print("[bold red]Only JSON model cards supported for validation[/bold red]")
                raise click.Abort()

        # Run compliance checks
        generator = ComplianceGenerator()

        table = Table(title="Compliance Validation Results")
        table.add_column("Framework", style="cyan")
        table.add_column("Compliant", style="green")
        table.add_column("Score", style="yellow")
        table.add_column("Missing Requirements", style="red")

        for fw in framework:
            report = generator.generate_compliance_report(card, fw)

            table.add_row(
                fw,
                "✓" if report.compliant else "✗",
                f"{report.score:.1%}",
                str(len(report.requirements_missing)),
            )

            if not report.compliant:
                console.print(f"\n[yellow]Missing requirements for {fw}:[/yellow]")
                for req in report.requirements_missing:
                    console.print(f"  • {req}")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.option("--path", "-p", default=".", help="Project path")
def init(path):
    """Initialize a new model documentation project."""
    project_path = Path(path)

    console.print(f"[bold blue]Initializing project at:[/bold blue] {project_path}")

    # Create directory structure
    dirs = ["models", "documentation", "config", "templates"]

    for dir_name in dirs:
        dir_path = project_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        console.print(f"  [green]✓[/green] Created {dir_name}/")

    # Create config file
    config_path = project_path / "config" / "config.yaml"
    if not config_path.exists():
        config_content = """# Model Documentation Configuration

defaults:
  format: markdown
  frameworks:
    - eu_ai_act
    - model_cards

templates:
  model_card: templates/model_card.md.j2

compliance:
  strict_mode: false
  auto_generate_recommendations: true
"""
        with open(config_path, "w") as f:
            f.write(config_content)
        console.print(f"  [green]✓[/green] Created config/config.yaml")

    # Create README
    readme_path = project_path / "README.md"
    if not readme_path.exists():
        readme_content = """# Model Documentation Project

This project contains model documentation generated using the Model Documentation System.

## Structure

- `models/` - Model files
- `documentation/` - Generated documentation
- `config/` - Configuration files
- `templates/` - Custom templates

## Usage

```bash
# Generate documentation
model-docs generate models/my_model.pt -o documentation/my_model.md

# Validate compliance
model-docs validate documentation/my_model.json
```
"""
        with open(readme_path, "w") as f:
            f.write(readme_content)
        console.print(f"  [green]✓[/green] Created README.md")

    console.print("[bold green]✓ Project initialized successfully![/bold green]")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
