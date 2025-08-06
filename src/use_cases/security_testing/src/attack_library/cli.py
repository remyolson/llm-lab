"""Command-line interface for Attack Library System."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import track
from rich.table import Table

from .core.library import AttackLibrary
from .core.models import AttackCategory, AttackSeverity
from .utils.data_loader import DataLoader
from .utils.validators import AttackValidator

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Attack Library System - Comprehensive attack prompt library for LLM security testing."""
    pass


@cli.command()
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
def init(library: str):
    """Initialize a new attack library."""
    library_path = Path(library)

    if library_path.exists():
        if not click.confirm(f"Library file {library} already exists. Overwrite?"):
            console.print("❌ Operation cancelled", style="red")
            return

    # Create empty library
    attack_library = AttackLibrary(library_path)
    attack_library.save_to_file()

    console.print(f"✅ Initialized attack library at {library}", style="green")


@cli.command()
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["jailbreak", "injection", "extraction", "manipulation", "evasion"]),
    help="Filter by category",
)
@click.option(
    "--severity",
    "-s",
    type=click.Choice(["low", "medium", "high", "critical"]),
    help="Filter by severity",
)
@click.option("--limit", default=10, help="Maximum number of results")
def list(library: str, category: Optional[str], severity: Optional[str], limit: int):
    """List attacks in the library."""
    library_path = Path(library)

    if not library_path.exists():
        console.print(f"❌ Library file {library} not found. Run 'init' first.", style="red")
        return

    attack_library = AttackLibrary(library_path)

    # Apply filters
    filters = {}
    if category:
        filters["category"] = category
    if severity:
        filters["severity"] = severity

    filters["limit"] = limit

    attacks = attack_library.search(**filters)

    if not attacks:
        console.print("No attacks found matching the criteria", style="yellow")
        return

    # Create table
    table = Table(title=f"Attack Library ({len(attacks)} attacks)")
    table.add_column("ID", style="cyan", width=12)
    table.add_column("Title", style="magenta", width=30)
    table.add_column("Category", style="green", width=12)
    table.add_column("Severity", style="red", width=10)
    table.add_column("Sophistication", style="blue", width=6)

    for attack in attacks:
        table.add_row(
            attack.id,
            attack.title[:30] + "..." if len(attack.title) > 30 else attack.title,
            attack.category.value,
            attack.severity.value,
            str(attack.sophistication),
        )

    console.print(table)


@cli.command()
@click.argument("attack_id")
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
def show(attack_id: str, library: str):
    """Show detailed information about an attack."""
    library_path = Path(library)

    if not library_path.exists():
        console.print(f"❌ Library file {library} not found", style="red")
        return

    attack_library = AttackLibrary(library_path)
    attack = attack_library.get_attack(attack_id)

    if not attack:
        console.print(f"❌ Attack {attack_id} not found", style="red")
        return

    console.print(f"\n[bold cyan]Attack: {attack.id}[/bold cyan]")
    console.print(f"[bold]Title:[/bold] {attack.title}")
    console.print(f"[bold]Category:[/bold] {attack.category.value}")
    console.print(f"[bold]Severity:[/bold] {attack.severity.value}")
    console.print(f"[bold]Sophistication:[/bold] {attack.sophistication}/5")
    console.print(
        f"[bold]Target Models:[/bold] {', '.join(attack.target_models) if attack.target_models else 'None specified'}"
    )
    console.print(f"[bold]Verified:[/bold] {'Yes' if attack.is_verified else 'No'}")

    console.print(f"\n[bold]Content:[/bold]")
    console.print(f"[italic]{attack.content}[/italic]")

    console.print(f"\n[bold]Metadata:[/bold]")
    console.print(f"  Source: {attack.metadata.source}")
    console.print(f"  Created: {attack.metadata.creation_date.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"  Updated: {attack.metadata.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")

    if attack.metadata.effectiveness_score:
        console.print(f"  Effectiveness: {attack.metadata.effectiveness_score:.2f}")

    if attack.metadata.tags:
        console.print(f"  Tags: {', '.join(sorted(attack.metadata.tags))}")


@cli.command()
@click.argument("data_file")
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
@click.option("--validate", is_flag=True, help="Validate attacks before importing")
def import_data(data_file: str, library: str, validate: bool):
    """Import attacks from a data file."""
    data_path = Path(data_file)
    library_path = Path(library)

    if not data_path.exists():
        console.print(f"❌ Data file {data_file} not found", style="red")
        return

    # Create library if it doesn't exist
    attack_library = AttackLibrary(library_path)

    # Load data
    loader = DataLoader()

    try:
        console.print(f"Loading attacks from {data_file}...")
        attacks = loader.load_from_file(data_path)

        if validate:
            console.print("Validating attacks...")
            validator = AttackValidator()
            report = validator.validate_attack_list(attacks)

            console.print(f"Validation complete:")
            console.print(f"  Valid: {report['valid_attacks']}")
            console.print(f"  Invalid: {report['invalid_attacks']}")
            console.print(f"  Success Rate: {report['validation_rate']:.1%}")

            if report["invalid_attacks"] > 0:
                if not click.confirm(
                    "Some attacks failed validation. Continue importing valid attacks?"
                ):
                    console.print("❌ Import cancelled", style="red")
                    return

        # Import attacks
        imported_count = 0
        duplicate_count = 0

        for attack in track(attacks, description="Importing attacks..."):
            if attack.id in attack_library:
                duplicate_count += 1
                console.print(f"⚠️ Duplicate ID {attack.id}, skipping", style="yellow")
                continue

            attack_library.attacks[attack.id] = attack
            attack_library._update_indices(attack)
            imported_count += 1

        # Save library
        attack_library.save_to_file()
        attack_library._invalidate_cache()

        console.print(f"✅ Import complete:", style="green")
        console.print(f"  Imported: {imported_count} attacks")
        console.print(f"  Duplicates skipped: {duplicate_count}")
        console.print(f"  Total in library: {len(attack_library)}")

    except Exception as e:
        console.print(f"❌ Import failed: {e}", style="red")


@cli.command()
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
@click.option(
    "--format", "-f", type=click.Choice(["json", "csv"]), default="json", help="Export format"
)
@click.option("--output", "-o", help="Output file (default: stdout)")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["jailbreak", "injection", "extraction", "manipulation", "evasion"]),
    help="Filter by category",
)
@click.option(
    "--severity",
    "-s",
    type=click.Choice(["low", "medium", "high", "critical"]),
    help="Filter by severity",
)
def export(
    library: str,
    format: str,
    output: Optional[str],
    category: Optional[str],
    severity: Optional[str],
):
    """Export attacks from the library."""
    library_path = Path(library)

    if not library_path.exists():
        console.print(f"❌ Library file {library} not found", style="red")
        return

    attack_library = AttackLibrary(library_path)

    # Apply filters
    filters = {}
    if category:
        filters["category"] = category
    if severity:
        filters["severity"] = severity

    # Export data
    exported_data = attack_library.export_attacks(format=format, filters=filters)

    if output:
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(exported_data)
        console.print(f"✅ Exported to {output}", style="green")
    else:
        click.echo(exported_data)


@cli.command()
@click.argument("query")
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
@click.option("--limit", default=10, help="Maximum number of results")
def search(query: str, library: str, limit: int):
    """Search attacks by text query."""
    library_path = Path(library)

    if not library_path.exists():
        console.print(f"❌ Library file {library} not found", style="red")
        return

    attack_library = AttackLibrary(library_path)
    attacks = attack_library.search(query=query, limit=limit)

    if not attacks:
        console.print(f"No attacks found for query: {query}", style="yellow")
        return

    console.print(f"Search results for: [bold cyan]{query}[/bold cyan]")

    for i, attack in enumerate(attacks, 1):
        console.print(f"\n{i}. [bold]{attack.title}[/bold] ({attack.id})")
        console.print(f"   Category: {attack.category.value} | Severity: {attack.severity.value}")

        # Show snippet of content
        content_snippet = (
            attack.content[:100] + "..." if len(attack.content) > 100 else attack.content
        )
        console.print(f"   Content: [italic]{content_snippet}[/italic]")


@cli.command()
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
def stats(library: str):
    """Show library statistics."""
    library_path = Path(library)

    if not library_path.exists():
        console.print(f"❌ Library file {library} not found", style="red")
        return

    attack_library = AttackLibrary(library_path)
    stats = attack_library.get_statistics()

    console.print(f"[bold cyan]Attack Library Statistics[/bold cyan]")
    console.print(f"Total Attacks: {stats['total_attacks']}")

    if stats["total_attacks"] == 0:
        console.print("Library is empty")
        return

    # Category distribution
    console.print(f"\n[bold]Categories:[/bold]")
    for category, count in stats["categories"].items():
        percentage = (count / stats["total_attacks"]) * 100
        console.print(f"  {category}: {count} ({percentage:.1f}%)")

    # Severity distribution
    console.print(f"\n[bold]Severities:[/bold]")
    for severity, count in stats["severities"].items():
        percentage = (count / stats["total_attacks"]) * 100
        console.print(f"  {severity}: {count} ({percentage:.1f}%)")

    # Verification stats
    if "verification" in stats:
        verification = stats["verification"]
        console.print(f"\n[bold]Verification:[/bold]")
        console.print(f"  Verified: {verification['verified']}")
        console.print(f"  Unverified: {verification['unverified']}")
        console.print(f"  Verification Rate: {verification['percentage_verified']:.1f}%")

    # Content stats
    if "content_stats" in stats:
        content = stats["content_stats"]
        console.print(f"\n[bold]Content Statistics:[/bold]")
        console.print(f"  Average Length: {content['average_length']:.0f} characters")
        console.print(f"  Total Characters: {content['total_characters']:,}")
        console.print(f"  Total Words: {content['total_words']:,}")


@cli.command()
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
@click.option("--strict", is_flag=True, help="Use strict validation")
def validate(library: str, strict: bool):
    """Validate all attacks in the library."""
    library_path = Path(library)

    if not library_path.exists():
        console.print(f"❌ Library file {library} not found", style="red")
        return

    attack_library = AttackLibrary(library_path)

    if len(attack_library) == 0:
        console.print("Library is empty", style="yellow")
        return

    validator = AttackValidator(strict=strict)
    attacks = list(attack_library.attacks.values())

    console.print(f"Validating {len(attacks)} attacks...")
    report = validator.validate_attack_list(attacks)

    # Display results
    console.print(f"\n[bold cyan]Validation Report[/bold cyan]")
    console.print(f"Total Attacks: {report['total_attacks']}")
    console.print(f"Valid: {report['valid_attacks']} ({report['validation_rate']:.1%})")
    console.print(f"Invalid: {report['invalid_attacks']}")

    if report["duplicate_ids"]:
        console.print(f"Duplicate IDs: {len(report['duplicate_ids'])}")

    if report["errors"]:
        console.print(f"\n[bold red]Errors:[/bold red]")
        for error in report["errors"][:10]:  # Show first 10 errors
            console.print(f"  • {error}")

        if len(report["errors"]) > 10:
            console.print(f"  ... and {len(report['errors']) - 10} more errors")


@cli.command()
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
@click.option("--target", "-t", default=500, help="Target number of attacks")
@click.option("--diversity", "-d", default=0.5, help="Diversity threshold (0.0-1.0)")
@click.option("--output", "-o", help="Output file for expanded library")
def expand(library: str, target: int, diversity: float, output: Optional[str]):
    """Expand attack library to target size with diverse attacks."""
    try:
        from .expansion.attack_expander import AttackExpander
    except ImportError:
        console.print("❌ Attack expansion not available. Missing dependencies.", style="red")
        return

    library_path = Path(library)

    if not library_path.exists():
        console.print(f"❌ Library file {library} not found", style="red")
        return

    # Load existing library
    attack_library = AttackLibrary(library_path)
    base_attacks = list(attack_library.attacks.values())

    if len(base_attacks) == 0:
        console.print("❌ Library is empty. Load some attacks first.", style="red")
        return

    console.print(f"Expanding library from {len(base_attacks)} to {target} attacks...")

    # Initialize expander and expand
    expander = AttackExpander()

    console.print("[bold green]Generating attacks...", style="green")
    expanded_attacks = expander.expand_library(
        base_attacks=base_attacks, target_total=target, diversity_threshold=diversity
    )

    console.print(
        f"✅ Expansion complete! Generated {len(expanded_attacks)} attacks", style="green"
    )

    # Get statistics
    stats = expander.get_expansion_stats(expanded_attacks)

    # Display statistics
    console.print(f"\n[bold cyan]Expansion Statistics[/bold cyan]")
    console.print(f"Total Attacks: {stats['total_attacks']}")
    console.print(f"Average Length: {stats['average_length']:.1f} characters")

    console.print(f"\n[bold]Category Distribution:[/bold]")
    for category, count in stats["category_distribution"].items():
        percentage = (count / stats["total_attacks"]) * 100 if stats["total_attacks"] > 0 else 0
        console.print(f"  {category}: {count} ({percentage:.1f}%)")

    console.print(f"\n[bold]Severity Distribution:[/bold]")
    for severity, count in stats["severity_distribution"].items():
        percentage = (count / stats["total_attacks"]) * 100 if stats["total_attacks"] > 0 else 0
        console.print(f"  {severity}: {count} ({percentage:.1f}%)")

    # Save expanded library
    if output:
        output_path = Path(output)
    else:
        output_path = library_path.with_name(f"expanded_{library_path.name}")

    # Create new library with expanded attacks
    expanded_library = AttackLibrary()
    for attack in expanded_attacks:
        expanded_library.attacks[attack.id] = attack
        expanded_library._update_indices(attack)

    expanded_library.save_to_file(output_path)
    console.print(f"✅ Saved expanded library to {output_path}", style="green")


@cli.command()
def load_initial():
    """Load the initial attack dataset."""
    # Find initial dataset
    current_dir = Path(__file__).parent
    dataset_path = current_dir / "data" / "attacks" / "initial_dataset.json"

    if not dataset_path.exists():
        console.print(f"❌ Initial dataset not found at {dataset_path}", style="red")
        return

    library_path = Path("attack_library.json")

    # Create library
    attack_library = AttackLibrary(library_path)

    try:
        console.print("Loading initial attack dataset...")
        loaded_count = attack_library.load_from_file(dataset_path)

        console.print(f"✅ Loaded {loaded_count} attacks from initial dataset", style="green")

        # Show stats
        stats = attack_library.get_statistics()
        console.print(f"\nLibrary now contains {stats['total_attacks']} attacks:")

        for category, count in stats["categories"].items():
            console.print(f"  {category}: {count}")

    except Exception as e:
        console.print(f"❌ Failed to load initial dataset: {e}", style="red")


@cli.command()
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
@click.option("--count", "-n", default=10, help="Number of attacks to generate")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["jailbreak", "injection", "extraction", "manipulation", "evasion"]),
    help="Attack category",
)
@click.option(
    "--transform",
    "-t",
    multiple=True,
    type=click.Choice(["paraphrase", "encoding", "social_engineering"]),
    help="Apply transformations",
)
@click.option("--save", is_flag=True, help="Save generated attacks to library")
def generate(library: str, count: int, category: Optional[str], transform: tuple, save: bool):
    """Generate new attack prompts using the prompt generation engine."""
    try:
        from .generators.prompt_generator import PromptGenerator
    except ImportError:
        console.print("❌ Prompt generation not available. Missing dependencies.", style="red")
        return

    library_path = Path(library)

    console.print(f"Generating {count} attack prompts...")

    # Initialize generator
    generator = PromptGenerator()

    # Set generation parameters
    kwargs = {}
    if category:
        kwargs["category"] = category

    if transform:
        kwargs["apply_transformations"] = True
        kwargs["transformations"] = list(transform)

    # Generate attacks
    attacks = generator.generate(count=count, **kwargs)

    console.print(f"✅ Generated {len(attacks)} attacks", style="green")

    # Display generated attacks
    for i, attack in enumerate(attacks, 1):
        console.print(f"\n[bold cyan]{i}. {attack.title}[/bold cyan]")
        console.print(f"Category: {attack.category.value} | Severity: {attack.severity.value}")
        console.print(
            f"Content: [italic]{attack.content[:100]}{'...' if len(attack.content) > 100 else ''}[/italic]"
        )

        if transform:
            transform_tags = [tag for tag in attack.metadata.tags if "transform" in tag]
            if transform_tags:
                console.print(f"Transformations: {', '.join(transform_tags)}")

    # Save to library if requested
    if save:
        try:
            attack_library = AttackLibrary(library_path)

            added_count = 0
            for attack in attacks:
                if attack.id not in attack_library:
                    attack_library.attacks[attack.id] = attack
                    attack_library._update_indices(attack)
                    added_count += 1

            attack_library.save_to_file()
            attack_library._invalidate_cache()

            console.print(f"✅ Saved {added_count} new attacks to library", style="green")

        except Exception as e:
            console.print(f"❌ Failed to save attacks: {e}", style="red")


@cli.command()
@click.argument("base_attack_id")
@click.option("--library", "-l", default="attack_library.json", help="Library file path")
@click.option("--count", "-n", default=5, help="Number of variants to generate")
@click.option("--save", is_flag=True, help="Save variants to library")
def variants(base_attack_id: str, library: str, count: int, save: bool):
    """Generate variants of an existing attack."""
    try:
        from .generators.prompt_generator import PromptGenerator
    except ImportError:
        console.print("❌ Prompt generation not available. Missing dependencies.", style="red")
        return

    library_path = Path(library)

    if not library_path.exists():
        console.print(f"❌ Library file {library} not found", style="red")
        return

    # Load library and find base attack
    attack_library = AttackLibrary(library_path)
    base_attack = attack_library.get_attack(base_attack_id)

    if not base_attack:
        console.print(f"❌ Attack {base_attack_id} not found in library", style="red")
        return

    console.print(f"Generating {count} variants of attack: {base_attack.title}")
    console.print(
        f"Base content: [italic]{base_attack.content[:80]}{'...' if len(base_attack.content) > 80 else ''}[/italic]"
    )

    # Generate variants
    generator = PromptGenerator()
    variants_list = generator.generate_variants(base_attack, count=count)

    console.print(f"✅ Generated {len(variants_list)} variants", style="green")

    # Display variants
    for i, variant in enumerate(variants_list, 1):
        console.print(f"\n[bold cyan]Variant {i}:[/bold cyan]")
        console.print(f"Type: {variant.variant_type}")
        console.print(
            f"Content: [italic]{variant.content[:100]}{'...' if len(variant.content) > 100 else ''}[/italic]"
        )

    # Save to library if requested
    if save:
        try:
            added_count = 0
            for variant in variants_list:
                if variant.id not in attack_library:
                    attack_library.attacks[variant.id] = variant
                    attack_library._update_indices(variant)
                    added_count += 1

            attack_library.save_to_file()
            attack_library._invalidate_cache()

            console.print(f"✅ Saved {added_count} variants to library", style="green")

        except Exception as e:
            console.print(f"❌ Failed to save variants: {e}", style="red")


def main():
    """Entry point for CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n❌ Operation cancelled", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
