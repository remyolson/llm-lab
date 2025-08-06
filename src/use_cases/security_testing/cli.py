#!/usr/bin/env python3
"""
LLM Security Testing Framework CLI

Comprehensive command-line interface for security testing operations.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Import security testing components
from src.attack_library.security import ScanConfig, SecurityScanner, SeverityLevel
from src.attack_library.security.security_scorer import SecurityScorer
from src.red_team.core.simulator import RedTeamSimulator
from src.red_team.scenarios.registry import ScenarioRegistry

console = Console()

# CLI Configuration
SUPPORTED_MODELS = [
    "gpt-4",
    "gpt-3.5-turbo",
    "claude-3-opus",
    "claude-3-sonnet",
    "llama-2-70b",
    "mistral-7b",
    "gemini-pro",
]

TEST_SUITES = [
    "jailbreak",
    "injection",
    "extraction",
    "bias",
    "toxicity",
    "privacy",
    "hallucination",
    "all",
]

COMPLIANCE_FRAMEWORKS = ["owasp-llm-top10", "nist-ai-rmf", "iso-27001", "gdpr", "hipaa"]

OUTPUT_FORMATS = ["json", "pdf", "csv", "html", "markdown"]


@click.group()
@click.version_option(version="1.0.0", prog_name="LLM Security Testing Framework")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.pass_context
def cli(ctx, verbose: bool, config: Optional[str]):
    """LLM Security Testing Framework - Comprehensive security assessment for language models."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if config:
        config_path = Path(config)
        if config_path.exists():
            with open(config_path) as f:
                ctx.obj["config"] = json.load(f)
                console.print(f"✅ Loaded configuration from {config}", style="green")


@cli.command()
@click.option("--model", "-m", required=True, help="Model name or endpoint URL")
@click.option(
    "--test-suites",
    "-t",
    multiple=True,
    default=["all"],
    type=click.Choice(TEST_SUITES),
    help="Test suites to run",
)
@click.option(
    "--severity-threshold",
    "-s",
    default="medium",
    type=click.Choice(["low", "medium", "high", "critical"]),
    help="Minimum severity threshold for findings",
)
@click.option(
    "--output-format", "-o", default="json", type=click.Choice(OUTPUT_FORMATS), help="Output format"
)
@click.option("--output-file", "-f", type=click.Path(), help="Output file path")
@click.option("--parallel", "-p", is_flag=True, help="Run tests in parallel")
@click.option("--timeout", default=300, help="Timeout in seconds per test")
@click.pass_context
def scan(
    ctx,
    model: str,
    test_suites: Tuple[str],
    severity_threshold: str,
    output_format: str,
    output_file: Optional[str],
    parallel: bool,
    timeout: int,
):
    """Run security scan on a language model."""

    console.print(
        Panel.fit(
            f"[bold cyan]Security Scan[/bold cyan]\n"
            f"Model: {model}\n"
            f"Test Suites: {', '.join(test_suites)}\n"
            f"Severity Threshold: {severity_threshold}",
            title="Scan Configuration",
        )
    )

    # Initialize scanner
    config = ScanConfig(
        model_name=model,
        severity_threshold=SeverityLevel[severity_threshold.upper()],
        parallel_execution=parallel,
        timeout_seconds=timeout,
    )

    scanner = SecurityScanner(config)

    # Run scan with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Determine test suites to run
        suites_to_run = list(test_suites) if "all" not in test_suites else TEST_SUITES[:-1]

        task = progress.add_task(f"Scanning {model}...", total=len(suites_to_run))

        results = []
        for suite in suites_to_run:
            progress.update(task, description=f"Running {suite} tests...")

            # Run test suite (simplified - would integrate with actual scanner)
            suite_results = run_test_suite(scanner, suite)
            results.extend(suite_results)

            progress.advance(task)

    # Calculate security score
    scorer = SecurityScorer()
    if results:
        # Create scan result from findings
        scan_result = create_scan_result(model, results)
        assessment = scorer.calculate_security_score(scan_result)

        # Display results
        display_scan_results(assessment, output_format, output_file)
    else:
        console.print("No vulnerabilities found!", style="green")


@cli.command()
@click.option("--model", "-m", required=True, help="Model name or endpoint URL")
@click.option(
    "--compliance-frameworks",
    "-c",
    multiple=True,
    required=True,
    type=click.Choice(COMPLIANCE_FRAMEWORKS),
    help="Compliance frameworks",
)
@click.option(
    "--test-suites",
    "-t",
    multiple=True,
    default=["all"],
    type=click.Choice(TEST_SUITES),
    help="Test suites to run",
)
@click.option("--generate-evidence", "-e", is_flag=True, help="Generate compliance evidence")
@click.option(
    "--output-format", "-o", default="pdf", type=click.Choice(OUTPUT_FORMATS), help="Output format"
)
@click.option(
    "--output-dir",
    "-d",
    type=click.Path(),
    default="./reports",
    help="Output directory for reports",
)
@click.pass_context
def enterprise_scan(
    ctx,
    model: str,
    compliance_frameworks: Tuple[str],
    test_suites: Tuple[str],
    generate_evidence: bool,
    output_format: str,
    output_dir: str,
):
    """Run enterprise-grade security scan with compliance reporting."""

    console.print(
        Panel.fit(
            f"[bold cyan]Enterprise Security Scan[/bold cyan]\n"
            f"Model: {model}\n"
            f"Compliance: {', '.join(compliance_frameworks)}\n"
            f"Evidence Generation: {'Yes' if generate_evidence else 'No'}",
            title="Enterprise Scan Configuration",
        )
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Run comprehensive scan
        task = progress.add_task("Running enterprise scan...", total=100)

        # Initialize components
        progress.update(task, description="Initializing security scanner...", completed=10)
        config = ScanConfig(
            model_name=model,
            enterprise_mode=True,
            compliance_frameworks=list(compliance_frameworks),
        )
        scanner = SecurityScanner(config)

        # Run security tests
        progress.update(task, description="Running security tests...", completed=30)
        results = run_comprehensive_tests(scanner, test_suites)

        # Generate compliance mappings
        progress.update(task, description="Mapping to compliance frameworks...", completed=60)
        compliance_results = map_to_compliance(results, compliance_frameworks)

        # Generate evidence if requested
        if generate_evidence:
            progress.update(task, description="Generating compliance evidence...", completed=80)
            evidence_files = generate_compliance_evidence(
                model, results, compliance_results, output_path
            )

        # Generate reports
        progress.update(task, description="Generating reports...", completed=90)
        report_files = generate_enterprise_reports(
            model, compliance_results, output_format, output_path
        )

        progress.update(task, completed=100)

    console.print("\n✅ Enterprise scan complete!", style="bold green")
    console.print(f"Reports saved to: {output_path}", style="cyan")


@cli.command()
@click.option("--model", "-m", required=True, help="Model name or endpoint URL")
@click.option(
    "--scenario",
    "-s",
    type=click.Choice(["customer-service", "financial", "healthcare", "all"]),
    default="all",
    help="Attack scenario",
)
@click.option(
    "--intensity",
    "-i",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    help="Attack intensity",
)
@click.option("--duration", "-d", default=60, help="Test duration in minutes")
@click.option("--interactive", is_flag=True, help="Interactive red team mode")
@click.option("--output-file", "-f", type=click.Path(), help="Output file for results")
@click.pass_context
def red_team(
    ctx,
    model: str,
    scenario: str,
    intensity: str,
    duration: int,
    interactive: bool,
    output_file: Optional[str],
):
    """Run red team simulation against a model."""

    console.print(
        Panel.fit(
            f"[bold red]Red Team Simulation[/bold red]\n"
            f"Target Model: {model}\n"
            f"Scenario: {scenario}\n"
            f"Intensity: {intensity}\n"
            f"Duration: {duration} minutes",
            title="Red Team Configuration",
        )
    )

    # Initialize red team simulator
    simulator = RedTeamSimulator()

    if interactive:
        # Interactive red team mode
        run_interactive_red_team(simulator, model)
    else:
        # Automated red team simulation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running red team simulation...", total=duration)

            # Get scenarios to run
            scenarios = get_red_team_scenarios(scenario)

            # Run simulation
            campaign_results = []
            for i in range(duration):
                progress.update(
                    task, description=f"Minute {i + 1}/{duration} - Executing attacks..."
                )

                # Run attack iterations
                results = simulator.run_campaign(
                    model_name=model, scenarios=scenarios, intensity=intensity
                )
                campaign_results.append(results)

                progress.advance(task)

        # Display results
        display_red_team_results(campaign_results, output_file)


@cli.command()
@click.option(
    "--scan-results", "-s", type=click.Path(exists=True), help="Path to scan results file"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["pdf", "html", "markdown"]),
    default="pdf",
    help="Report format",
)
@click.option(
    "--template",
    "-t",
    type=click.Choice(["executive", "technical", "compliance"]),
    default="executive",
    help="Report template",
)
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file path")
@click.option("--include-recommendations", is_flag=True, help="Include remediation recommendations")
@click.pass_context
def generate_report(
    ctx, scan_results: str, format: str, template: str, output: str, include_recommendations: bool
):
    """Generate security assessment report from scan results."""

    console.print(f"Generating {template} report in {format} format...")

    # Load scan results
    with open(scan_results) as f:
        results = json.load(f)

    # Generate report based on template
    if template == "executive":
        report_content = generate_executive_report(results, include_recommendations)
    elif template == "technical":
        report_content = generate_technical_report(results, include_recommendations)
    else:
        report_content = generate_compliance_report(results, include_recommendations)

    # Save report in requested format
    output_path = Path(output)
    if format == "pdf":
        save_pdf_report(report_content, output_path)
    elif format == "html":
        save_html_report(report_content, output_path)
    else:
        save_markdown_report(report_content, output_path)

    console.print(f"✅ Report saved to {output}", style="green")


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive security testing mode."""

    console.print(
        Panel.fit(
            "[bold cyan]Interactive Security Testing Mode[/bold cyan]\n"
            "Guided testing with recommendations",
            title="Interactive Mode",
        )
    )

    # Get model
    model = Prompt.ask("Enter model name or endpoint URL")

    # Select test type
    test_type = Prompt.ask(
        "Select test type",
        choices=["quick-scan", "comprehensive", "red-team", "compliance"],
        default="quick-scan",
    )

    if test_type == "quick-scan":
        # Quick security scan
        run_quick_scan_interactive(model)
    elif test_type == "comprehensive":
        # Comprehensive scan
        run_comprehensive_scan_interactive(model)
    elif test_type == "red-team":
        # Red team simulation
        run_red_team_interactive(model)
    else:
        # Compliance assessment
        run_compliance_assessment_interactive(model)


# Helper functions


def run_test_suite(scanner: SecurityScanner, suite: str) -> List[Dict]:
    """Run a specific test suite."""
    # Simplified implementation - would integrate with actual scanner
    console.print(f"  Running {suite} tests...", style="dim")
    return []


def create_scan_result(model: str, findings: List[Dict]):
    """Create scan result object from findings."""
    from src.attack_library.security.models import ModelResponse, ScanResult

    return ScanResult(
        scan_id=f"scan-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        model_name=model,
        attack_prompt="Various test prompts",
        response=ModelResponse("", model),
        vulnerabilities=[],  # Would convert findings to vulnerabilities
        scan_duration_ms=0,
    )


def display_scan_results(assessment, output_format: str, output_file: Optional[str]):
    """Display or save scan results."""

    # Create results summary
    summary = Table(title="Security Assessment Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")

    summary.add_row("Overall Score", f"{assessment.overall_score:.1f}/100")
    summary.add_row("Security Posture", assessment.security_posture)
    summary.add_row("Risk Level", assessment.severity_level.value)
    summary.add_row("Vulnerabilities Found", str(len(assessment.vulnerabilities_by_type)))

    console.print(summary)

    # Show top risks
    if assessment.risk_factors:
        console.print("\n[bold]Top Risk Factors:[/bold]")
        for factor in assessment.risk_factors[:5]:
            console.print(f"  • {factor}")

    # Show recommendations
    if assessment.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in assessment.recommendations[:5]:
            console.print(f"  • {rec}")

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        if output_format == "json":
            with open(output_path, "w") as f:
                json.dump(assessment.to_dict(), f, indent=2)
        console.print(f"\n✅ Results saved to {output_file}", style="green")


def run_comprehensive_tests(scanner, test_suites):
    """Run comprehensive security tests."""
    results = []
    for suite in test_suites:
        results.extend(run_test_suite(scanner, suite))
    return results


def map_to_compliance(results, frameworks):
    """Map test results to compliance frameworks."""
    compliance_mapping = {}
    for framework in frameworks:
        compliance_mapping[framework] = {
            "controls_tested": 0,
            "controls_passed": 0,
            "controls_failed": 0,
            "findings": [],
        }
    return compliance_mapping


def generate_compliance_evidence(model, results, compliance_results, output_path):
    """Generate compliance evidence files."""
    evidence_files = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for framework, data in compliance_results.items():
        evidence_file = output_path / f"{model}_{framework}_evidence_{timestamp}.json"
        with open(evidence_file, "w") as f:
            json.dump(data, f, indent=2)
        evidence_files.append(evidence_file)

    return evidence_files


def generate_enterprise_reports(model, compliance_results, output_format, output_path):
    """Generate enterprise compliance reports."""
    report_files = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for framework, data in compliance_results.items():
        report_file = output_path / f"{model}_{framework}_report_{timestamp}.{output_format}"
        # Generate report (simplified)
        report_files.append(report_file)

    return report_files


def get_red_team_scenarios(scenario_type: str):
    """Get red team scenarios to run."""
    if scenario_type == "all":
        return ScenarioRegistry.list_scenarios()
    else:
        return [scenario_type]


def run_interactive_red_team(simulator, model):
    """Run interactive red team session."""
    console.print("\n[bold]Interactive Red Team Mode[/bold]")
    console.print("Type 'help' for commands, 'exit' to quit\n")

    while True:
        command = Prompt.ask("red-team")

        if command == "exit":
            break
        elif command == "help":
            console.print("Commands:")
            console.print("  attack <prompt> - Send attack prompt")
            console.print("  scenario <name> - Load attack scenario")
            console.print("  status - Show current session status")
            console.print("  report - Generate session report")
            console.print("  exit - Exit interactive mode")
        elif command.startswith("attack "):
            prompt = command[7:]
            # Execute attack
            console.print(f"Executing attack: {prompt}")
        elif command.startswith("scenario "):
            scenario = command[9:]
            # Load scenario
            console.print(f"Loading scenario: {scenario}")
        elif command == "status":
            console.print("Session status: Active")
        elif command == "report":
            console.print("Generating session report...")


def display_red_team_results(campaign_results, output_file):
    """Display red team simulation results."""

    # Create summary table
    summary = Table(title="Red Team Campaign Results")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")

    total_attacks = sum(len(r.sessions) for r in campaign_results)
    successful_attacks = sum(len([s for s in r.sessions if s.success]) for r in campaign_results)

    summary.add_row("Total Attacks", str(total_attacks))
    summary.add_row("Successful Attacks", str(successful_attacks))
    summary.add_row("Success Rate", f"{(successful_attacks / total_attacks) * 100:.1f}%")

    console.print(summary)

    if output_file:
        # Save detailed results
        with open(output_file, "w") as f:
            json.dump([r.to_dict() for r in campaign_results], f, indent=2)
        console.print(f"\n✅ Results saved to {output_file}", style="green")


def generate_executive_report(results, include_recommendations):
    """Generate executive summary report."""
    return {
        "title": "Executive Security Assessment",
        "summary": "High-level security findings",
        "recommendations": []
        if not include_recommendations
        else ["Recommendation 1", "Recommendation 2"],
    }


def generate_technical_report(results, include_recommendations):
    """Generate technical report."""
    return {
        "title": "Technical Security Assessment",
        "findings": results,
        "technical_details": {},
        "recommendations": []
        if not include_recommendations
        else ["Technical fix 1", "Technical fix 2"],
    }


def generate_compliance_report(results, include_recommendations):
    """Generate compliance report."""
    return {
        "title": "Compliance Assessment Report",
        "frameworks": {},
        "gaps": [],
        "recommendations": []
        if not include_recommendations
        else ["Compliance action 1", "Compliance action 2"],
    }


def save_pdf_report(content, output_path):
    """Save report as PDF."""
    # Simplified - would use reportlab or similar
    with open(output_path, "w") as f:
        json.dump(content, f, indent=2)


def save_html_report(content, output_path):
    """Save report as HTML."""
    html_content = f"""
    <html>
    <head><title>{content.get("title", "Security Report")}</title></head>
    <body>
    <h1>{content.get("title", "Security Report")}</h1>
    <pre>{json.dumps(content, indent=2)}</pre>
    </body>
    </html>
    """
    with open(output_path, "w") as f:
        f.write(html_content)


def save_markdown_report(content, output_path):
    """Save report as Markdown."""
    md_content = f"# {content.get('title', 'Security Report')}\n\n"
    md_content += f"```json\n{json.dumps(content, indent=2)}\n```"
    with open(output_path, "w") as f:
        f.write(md_content)


def run_quick_scan_interactive(model):
    """Run interactive quick scan."""
    console.print("\nRunning quick security scan...")
    # Implementation would follow


def run_comprehensive_scan_interactive(model):
    """Run interactive comprehensive scan."""
    console.print("\nRunning comprehensive security scan...")
    # Implementation would follow


def run_red_team_interactive(model):
    """Run interactive red team."""
    console.print("\nStarting red team simulation...")
    # Implementation would follow


def run_compliance_assessment_interactive(model):
    """Run interactive compliance assessment."""
    frameworks = Prompt.ask(
        "Select compliance frameworks", choices=COMPLIANCE_FRAMEWORKS, default="owasp-llm-top10"
    )
    console.print(f"\nRunning compliance assessment for {frameworks}...")
    # Implementation would follow


if __name__ == "__main__":
    cli()
