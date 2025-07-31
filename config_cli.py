#!/usr/bin/env python3
"""
Configuration Management CLI

This tool provides command-line access to manage provider configurations,
test credentials, and view available models with their aliases.
"""

import click
import yaml
import json
import os
from pathlib import Path
from typing import Optional
from tabulate import tabulate

from config.provider_config import get_config_manager, reset_config_manager
from llm_providers import get_provider_for_model


@click.group()
def cli():
    """LLM Lab Configuration Management Tool"""
    pass


@cli.command()
@click.option('--provider', '-p', help='Filter by provider name')
@click.option('--include-aliases', '-a', is_flag=True, help='Include model aliases')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), default='table',
              help='Output format')
def list_models(provider: Optional[str], include_aliases: bool, format: str):
    """List all available models and their configurations."""
    manager = get_config_manager()
    
    if provider:
        # Filter to specific provider
        all_models = manager.get_available_models(include_aliases=include_aliases)
        if provider not in all_models:
            click.echo(f"Error: Unknown provider '{provider}'", err=True)
            click.echo(f"Available providers: {', '.join(all_models.keys())}", err=True)
            return
        models = {provider: all_models[provider]}
    else:
        models = manager.get_available_models(include_aliases=include_aliases)
    
    if format == 'json':
        click.echo(json.dumps(models, indent=2))
    elif format == 'yaml':
        click.echo(yaml.dump(models, default_flow_style=False))
    else:  # table format
        # Prepare table data
        table_data = []
        for provider_name, model_list in sorted(models.items()):
            for model in sorted(model_list):
                table_data.append([provider_name, model])
        
        if table_data:
            headers = ['Provider', 'Model']
            click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
        else:
            click.echo("No models found.")


@cli.command()
@click.argument('model_name')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), default='table',
              help='Output format')
def show_config(model_name: str, format: str):
    """Show configuration for a specific model."""
    manager = get_config_manager()
    
    # Resolve alias
    resolved_name = manager.resolve_model_alias(model_name)
    if resolved_name != model_name:
        click.echo(f"Resolved alias: {model_name} -> {resolved_name}")
    
    # Get configuration
    config = manager.get_model_config(resolved_name)
    
    if not config:
        click.echo(f"Error: No configuration found for model '{model_name}'", err=True)
        return
    
    # Get provider info
    provider_name = manager.get_provider_for_model(resolved_name)
    if provider_name:
        provider_defaults = manager.provider_defaults.get(provider_name)
        if provider_defaults:
            config['timeout'] = provider_defaults.timeout
            config['max_retries'] = provider_defaults.max_retries
            config['retry_delay'] = provider_defaults.retry_delay
    
    if format == 'json':
        click.echo(json.dumps(config, indent=2))
    elif format == 'yaml':
        click.echo(yaml.dump(config, default_flow_style=False))
    else:  # table format
        # Flatten config for table display
        table_data = []
        table_data.append(['Provider', config.get('provider', 'N/A')])
        table_data.append(['Model', config.get('model', resolved_name)])
        
        if 'parameters' in config:
            for key, value in config['parameters'].items():
                table_data.append([f'Parameter: {key}', str(value)])
        
        for key in ['timeout', 'max_retries', 'retry_delay']:
            if key in config:
                table_data.append([key.replace('_', ' ').title(), str(config[key])])
        
        click.echo(tabulate(table_data, tablefmt='grid'))


@cli.command()
@click.argument('model_name')
def resolve_alias(model_name: str):
    """Resolve a model alias to its canonical name."""
    manager = get_config_manager()
    resolved = manager.resolve_model_alias(model_name)
    
    if resolved == model_name:
        click.echo(f"{model_name} (no alias)")
    else:
        click.echo(f"{model_name} -> {resolved}")
        
        # Show provider
        provider = manager.get_provider_for_model(resolved)
        if provider:
            click.echo(f"Provider: {provider}")


@cli.command()
@click.option('--provider', '-p', help='Check specific provider')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def check_credentials(provider: Optional[str], verbose: bool):
    """Check if required credentials are configured."""
    manager = get_config_manager()
    
    results = manager.validate_credentials(provider)
    
    if not results:
        click.echo("No providers to check.", err=True)
        return
    
    # Display results
    all_valid = True
    table_data = []
    
    for provider_name, is_valid in sorted(results.items()):
        status = "✓ Valid" if is_valid else "✗ Missing"
        all_valid = all_valid and is_valid
        
        # Get required env vars
        provider_obj = manager.provider_defaults.get(provider_name)
        env_vars = provider_obj.env_vars if provider_obj else []
        
        if verbose:
            for var in env_vars:
                var_status = "✓" if os.getenv(var) else "✗"
                table_data.append([provider_name, var, var_status])
        else:
            env_var_str = ', '.join(env_vars)
            table_data.append([provider_name, env_var_str, status])
    
    if verbose:
        headers = ['Provider', 'Environment Variable', 'Status']
    else:
        headers = ['Provider', 'Required Variables', 'Status']
    
    click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    if not all_valid:
        click.echo("\nSome credentials are missing. Set the required environment variables to use those providers.", err=True)
        exit(1)


@cli.command()
@click.option('--user', '-u', is_flag=True, help='Create user config (~/.llm-lab/config.yaml)')
@click.option('--project', '-p', is_flag=True, help='Create project config (./config/providers.yaml)')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
def init_config(user: bool, project: bool, force: bool):
    """Initialize a configuration file from template."""
    if not user and not project:
        click.echo("Error: Specify --user or --project", err=True)
        return
    
    if user:
        config_path = Path.home() / ".llm-lab" / "config.yaml"
        template_path = Path("config/providers.example.yaml")
    else:
        config_path = Path("config/providers.yaml")
        template_path = Path("config/providers.yaml")
    
    if config_path.exists() and not force:
        click.echo(f"Error: {config_path} already exists. Use --force to overwrite.", err=True)
        return
    
    # Create directory if needed
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy template or create minimal config
    if template_path.exists():
        import shutil
        shutil.copy(template_path, config_path)
        click.echo(f"Created {config_path} from template")
    else:
        # Create minimal config
        minimal_config = {
            "providers": {},
            "aliases": {}
        }
        with open(config_path, 'w') as f:
            yaml.dump(minimal_config, f, default_flow_style=False)
        click.echo(f"Created minimal config at {config_path}")
    
    click.echo(f"\nEdit {config_path} to customize your configuration.")


@cli.command()
@click.argument('model_name')
@click.option('--prompt', '-p', default="Hello, how are you?", help='Test prompt to use')
@click.option('--temperature', '-t', type=float, help='Override temperature')
@click.option('--max-tokens', '-m', type=int, help='Override max tokens')
def test_model(model_name: str, prompt: str, temperature: Optional[float], max_tokens: Optional[int]):
    """Test a model by sending a simple prompt."""
    try:
        # Get provider for model
        provider_class = get_provider_for_model(model_name)
        
        # Create provider instance
        kwargs = {}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        
        provider = provider_class(model_name, **kwargs)
        
        click.echo(f"Testing model: {model_name}")
        click.echo(f"Provider: {provider.provider_name}")
        click.echo(f"Prompt: {prompt}")
        click.echo("-" * 50)
        
        # Initialize and generate
        provider.initialize()
        response = provider.generate(prompt)
        
        click.echo("Response:")
        click.echo(response)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)


@cli.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), default='table',
              help='Output format')
def show_aliases(format: str):
    """Show all configured model aliases."""
    manager = get_config_manager()
    
    # Build alias mapping
    alias_data = []
    for alias, model in sorted(manager.model_aliases.items()):
        if alias != model:  # Only show actual aliases
            provider = manager.get_provider_for_model(model)
            alias_data.append({
                'alias': alias,
                'model': model,
                'provider': provider or 'unknown'
            })
    
    if format == 'json':
        click.echo(json.dumps(alias_data, indent=2))
    elif format == 'yaml':
        click.echo(yaml.dump(alias_data, default_flow_style=False))
    else:  # table format
        if alias_data:
            table_rows = [[item['alias'], item['model'], item['provider']] for item in alias_data]
            headers = ['Alias', 'Model', 'Provider']
            click.echo(tabulate(table_rows, headers=headers, tablefmt='grid'))
        else:
            click.echo("No aliases configured.")


@cli.command()
def validate_config():
    """Validate all configuration files."""
    manager = get_config_manager()
    
    click.echo("Validating configuration files...")
    
    # Check for config files
    config_files = [
        Path("config/providers.yaml"),
        Path.home() / ".llm-lab" / "config.yaml"
    ]
    
    found_files = []
    for config_file in config_files:
        if config_file.exists():
            found_files.append(str(config_file))
            try:
                with open(config_file, 'r') as f:
                    if config_file.suffix in ['.yaml', '.yml']:
                        yaml.safe_load(f)
                    else:
                        json.load(f)
                click.echo(f"✓ {config_file}: Valid")
            except Exception as e:
                click.echo(f"✗ {config_file}: Invalid - {e}", err=True)
    
    if not found_files:
        click.echo("No configuration files found. Using defaults.")
    
    # Check for duplicate aliases
    click.echo("\nChecking for duplicate aliases...")
    seen_aliases = {}
    duplicates = []
    
    for alias, model in manager.model_aliases.items():
        if alias in seen_aliases and seen_aliases[alias] != model:
            duplicates.append((alias, seen_aliases[alias], model))
        seen_aliases[alias] = model
    
    if duplicates:
        click.echo("Warning: Found duplicate aliases:", err=True)
        for alias, model1, model2 in duplicates:
            click.echo(f"  - '{alias}' maps to both '{model1}' and '{model2}'", err=True)
    else:
        click.echo("✓ No duplicate aliases found")
    
    # Summary
    click.echo(f"\nTotal providers: {len(manager.provider_defaults)}")
    click.echo(f"Total models: {sum(len(p.models) for p in manager.provider_defaults.values())}")
    click.echo(f"Total aliases: {len([a for a, m in manager.model_aliases.items() if a != m])}")


if __name__ == '__main__':
    cli()