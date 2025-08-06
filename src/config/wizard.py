"""
Configuration Wizard for LLM Lab

Interactive setup wizard for first-time configuration of the LLM Lab framework.
Guides users through setting up API keys, choosing defaults, and configuring
various components.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from .config.settings import (
    BenchmarkConfig,
    DatasetConfig,
    FineTuningConfig,
    LogLevel,
    ModelParameters,
    MonitoringConfig,
    ProviderConfig,
    ProviderType,
    RetryConfig,
    Settings,
)

console = Console()


class ConfigWizard:
    """Interactive configuration wizard for LLM Lab setup."""

    def __init__(self, config_path: Path | None = None):
        """
        Initialize the configuration wizard.

        Args:
            config_path: Path to save configuration (defaults to .env and config.yaml)
        """
        self.config_path = config_path or Path("config.yaml")
        self.env_path = Path(".env")
        self.settings = Settings()
        self.api_keys = {}

    def run(self) -> Settings:
        """
        Run the configuration wizard.

        Returns:
            Configured Settings instance
        """
        self._show_welcome()
        self._configure_api_keys()
        self._configure_providers()
        self._configure_defaults()
        self._configure_advanced()
        self._save_configuration()
        self._show_summary()

        return self.settings

    def _show_welcome(self):
        """Display welcome message."""
        console.clear()
        welcome_text = Text("Welcome to LLM Lab Configuration Wizard", style="bold blue")
        console.print(Panel(welcome_text, title="🚀 Setup", expand=False))
        console.print("\nThis wizard will help you configure LLM Lab for first-time use.\n")

    def _configure_api_keys(self):
        """Configure API keys for various providers."""
        console.print(Panel("API Key Configuration", style="cyan"))
        console.print("Enter your API keys (press Enter to skip):\n")

        providers = [
            ("OpenAI", "OPENAI_API_KEY", "sk-..."),
            ("Anthropic", "ANTHROPIC_API_KEY", "sk-ant-..."),
            ("Google", "GOOGLE_API_KEY", "AIza..."),
        ]

        for provider_name, env_var, example in providers:
            current = os.getenv(env_var)
            if current:
                if Confirm.ask(f"{provider_name} API key already set. Replace?", default=False):
                    key = Prompt.ask(
                        f"  {provider_name} API Key", password=True, default="", show_default=False
                    )
                    if key:
                        self.api_keys[env_var] = key
                else:
                    self.api_keys[env_var] = current
            else:
                key = Prompt.ask(
                    f"  {provider_name} API Key (e.g., {example})",
                    password=True,
                    default="",
                    show_default=False,
                )
                if key:
                    self.api_keys[env_var] = key

        console.print()

    def _configure_providers(self):
        """Configure provider-specific settings."""
        console.print(Panel("Provider Configuration", style="cyan"))

        # Configure each provider that has an API key
        if "OPENAI_API_KEY" in self.api_keys:
            self._configure_openai()

        if "ANTHROPIC_API_KEY" in self.api_keys:
            self._configure_anthropic()

        if "GOOGLE_API_KEY" in self.api_keys:
            self._configure_google()

        # Set default provider
        if self.api_keys:
            provider_choices = []
            if "OPENAI_API_KEY" in self.api_keys:
                provider_choices.append("openai")
            if "ANTHROPIC_API_KEY" in self.api_keys:
                provider_choices.append("anthropic")
            if "GOOGLE_API_KEY" in self.api_keys:
                provider_choices.append("google")

            if len(provider_choices) > 1:
                console.print("\nMultiple providers configured. Choose default:")
                for i, choice in enumerate(provider_choices, 1):
                    console.print(f"  {i}. {choice}")

                choice_idx = IntPrompt.ask(
                    "Select default provider",
                    default=1,
                    choices=[str(i) for i in range(1, len(provider_choices) + 1)],
                )
                self.settings.default_provider = ProviderType(provider_choices[choice_idx - 1])
            elif provider_choices:
                self.settings.default_provider = ProviderType(provider_choices[0])

    def _configure_openai(self):
        """Configure OpenAI provider settings."""
        console.print("\n[bold]OpenAI Configuration:[/bold]")

        default_model = Prompt.ask(
            "  Default model",
            default="gpt-4o-mini",
            choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        )

        config = ProviderConfig(
            type=ProviderType.OPENAI,
            api_key=self.api_keys["OPENAI_API_KEY"],
            default_model=default_model,
        )

        if Confirm.ask("  Configure advanced settings?", default=False):
            config.model_parameters.temperature = float(
                Prompt.ask("    Temperature", default="0.7")
            )
            config.model_parameters.max_tokens = int(Prompt.ask("    Max tokens", default="1000"))
            config.retry_config.max_retries = int(Prompt.ask("    Max retries", default="3"))
            config.retry_config.timeout_seconds = int(
                Prompt.ask("    Timeout (seconds)", default="30")
            )

        self.settings.providers["openai"] = config

    def _configure_anthropic(self):
        """Configure Anthropic provider settings."""
        console.print("\n[bold]Anthropic Configuration:[/bold]")

        default_model = Prompt.ask(
            "  Default model",
            default="claude-3-haiku-20240307",
            choices=[
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022",
            ],
        )

        config = ProviderConfig(
            type=ProviderType.ANTHROPIC,
            api_key=self.api_keys["ANTHROPIC_API_KEY"],
            default_model=default_model,
        )

        if Confirm.ask("  Configure advanced settings?", default=False):
            config.model_parameters.temperature = float(
                Prompt.ask("    Temperature", default="0.7")
            )
            config.model_parameters.max_tokens = int(Prompt.ask("    Max tokens", default="1000"))
            config.retry_config.max_retries = int(Prompt.ask("    Max retries", default="3"))
            config.retry_config.timeout_seconds = int(
                Prompt.ask("    Timeout (seconds)", default="60")
            )

        self.settings.providers["anthropic"] = config

    def _configure_google(self):
        """Configure Google provider settings."""
        console.print("\n[bold]Google Configuration:[/bold]")

        default_model = Prompt.ask(
            "  Default model",
            default="gemini-1.5-flash",
            choices=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
        )

        config = ProviderConfig(
            type=ProviderType.GOOGLE,
            api_key=self.api_keys["GOOGLE_API_KEY"],
            default_model=default_model,
        )

        if Confirm.ask("  Configure advanced settings?", default=False):
            config.model_parameters.temperature = float(
                Prompt.ask("    Temperature", default="0.7")
            )
            config.model_parameters.max_tokens = int(Prompt.ask("    Max tokens", default="1000"))
            config.retry_config.max_retries = int(Prompt.ask("    Max retries", default="3"))
            config.retry_config.timeout_seconds = int(
                Prompt.ask("    Timeout (seconds)", default="30")
            )

        self.settings.providers["google"] = config

    def _configure_defaults(self):
        """Configure default settings."""
        console.print(Panel("Default Settings", style="cyan"))

        # Project settings
        self.settings.project_name = Prompt.ask("Project name", default="LLM Lab")

        self.settings.environment = Prompt.ask(
            "Environment", default="development", choices=["development", "staging", "production"]
        )

        self.settings.debug = Confirm.ask("Enable debug mode?", default=False)

        console.print()

    def _configure_advanced(self):
        """Configure advanced settings."""
        if not Confirm.ask("Configure advanced settings?", default=False):
            return

        console.print(Panel("Advanced Configuration", style="cyan"))

        # Dataset configuration
        console.print("\n[bold]Dataset Settings:[/bold]")
        self.settings.dataset.base_path = Path(
            Prompt.ask("  Dataset directory", default="./datasets")
        )
        self.settings.dataset.auto_download = Confirm.ask("  Auto-download datasets?", default=True)

        # Benchmark configuration
        console.print("\n[bold]Benchmark Settings:[/bold]")
        self.settings.benchmark.output_dir = Path(
            Prompt.ask("  Results directory", default="./results")
        )
        self.settings.benchmark.batch_size = int(Prompt.ask("  Batch size", default="8"))
        self.settings.benchmark.parallel_requests = int(
            Prompt.ask("  Parallel requests", default="4")
        )

        # Monitoring configuration
        console.print("\n[bold]Monitoring Settings:[/bold]")
        log_level = Prompt.ask(
            "  Log level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        self.settings.monitoring.log_level = LogLevel(log_level)
        self.settings.monitoring.enable_telemetry = Confirm.ask(
            "  Enable telemetry?", default=False
        )

        # Fine-tuning configuration
        if Confirm.ask("\nConfigure fine-tuning settings?", default=False):
            console.print("\n[bold]Fine-tuning Settings:[/bold]")
            self.settings.fine_tuning.models_dir = Path(
                Prompt.ask("  Models directory", default="./models")
            )
            self.settings.fine_tuning.default_batch_size = int(
                Prompt.ask("  Default batch size", default="4")
            )
            self.settings.fine_tuning.learning_rate = float(
                Prompt.ask("  Learning rate", default="2e-5")
            )
            self.settings.fine_tuning.num_train_epochs = int(
                Prompt.ask("  Training epochs", default="3")
            )

        console.print()

    def _save_configuration(self):
        """Save configuration to files."""
        console.print(Panel("Saving Configuration", style="green"))

        # Save API keys to .env file
        if self.api_keys:
            console.print(f"Saving API keys to {self.env_path}...")
            with open(self.env_path, "w") as f:
                for key, value in self.api_keys.items():
                    f.write(f"{key}={value}\n")
            console.print(f"  ✓ API keys saved to {self.env_path}")

        # Save settings to YAML file
        console.print(f"Saving configuration to {self.config_path}...")
        self.settings.to_yaml(self.config_path)
        console.print(f"  ✓ Configuration saved to {self.config_path}")

        console.print()

    def _show_summary(self):
        """Display configuration summary."""
        console.print(Panel("Configuration Summary", style="green"))

        # Create summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")

        # Basic settings
        table.add_row("Project Name", self.settings.project_name)
        table.add_row("Environment", self.settings.environment)
        table.add_row("Debug Mode", str(self.settings.debug))

        # Providers
        configured_providers = list(self.settings.providers.keys())
        if configured_providers:
            table.add_row("Configured Providers", ", ".join(configured_providers))
            table.add_row("Default Provider", self.settings.default_provider.value)

        # Paths
        table.add_row("Dataset Directory", str(self.settings.dataset.base_path))
        table.add_row("Results Directory", str(self.settings.benchmark.output_dir))
        table.add_row("Log Directory", str(self.settings.monitoring.log_dir))

        console.print(table)
        console.print("\n✅ Configuration complete! You can now use LLM Lab.\n")

        # Show next steps
        console.print(Panel("Next Steps", style="blue"))
        console.print("1. Verify your configuration:")
        console.print("   python -m src.config.wizard --validate\n")
        console.print("2. Run a test benchmark:")
        console.print("   python scripts/run_benchmarks.py\n")
        console.print("3. Check the documentation:")
        console.print("   https://github.com/yourusername/llm-lab/docs\n")


@click.command()
@click.option("--config-path", type=click.Path(), help="Path to save configuration file")
@click.option("--validate", is_flag=True, help="Validate existing configuration")
@click.option("--reconfigure", is_flag=True, help="Reconfigure existing setup")
def main(config_path: str | None, validate: bool, reconfigure: bool):
    """LLM Lab Configuration Wizard."""

    if validate:
        # Validate existing configuration
        try:
            settings = Settings()
            errors = settings.validate_all()

            if errors:
                console.print("[red]Configuration validation failed:[/red]")
                for error in errors:
                    console.print(f"  ✗ {error}")
                sys.exit(1)
            else:
                console.print("[green]✓ Configuration is valid![/green]")

                # Show current configuration
                table = Table(title="Current Configuration")
                table.add_column("Provider", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Default Model")

                for name, provider in settings.providers.items():
                    status = "✓ Configured" if provider.api_key else "✗ No API Key"
                    table.add_row(name, status, provider.default_model or "N/A")

                console.print(table)
        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")
            sys.exit(1)

    elif reconfigure or not Path(".env").exists():
        # Run configuration wizard
        wizard = ConfigWizard(Path(config_path) if config_path else None)
        wizard.run()

    else:
        console.print("Configuration already exists. Use --reconfigure to modify.")


if __name__ == "__main__":
    main()
