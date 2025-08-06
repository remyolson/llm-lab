#!/usr/bin/env python3
"""
Configuration CLI Tool for LLM Lab

This module provides command-line utilities for validating, managing, and
migrating LLM Lab configuration files.

Usage:
    python -m src.config.cli validate config.yaml
    python -m src.config.cli generate-template --format yaml
    python -m src.config.cli export-schema
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .manager import ConfigurationError, ConfigurationManager
from .settings import Settings


def validate_config(config_path: Path, strict: bool = True) -> bool:
    """
    Validate a configuration file.

    Args:
        config_path: Path to configuration file
        strict: Enable strict validation mode

    Returns:
        True if valid, False otherwise
    """
    try:
        manager = ConfigurationManager(config_file=config_path, strict_mode=strict)

        print(f"🔍 Validating configuration: {config_path}")

        # Load and validate configuration
        settings = manager.load_configuration()

        # Run validation checks
        errors = settings.validate_all()

        if errors:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"  • {error}")
            return False

        print("✅ Configuration is valid!")

        # Show configuration summary
        summary = manager.get_config_summary()
        print("\n📊 Configuration Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return True

    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def generate_template(format_type: str = "yaml", profile: str = "development") -> str:
    """
    Generate a configuration template.

    Args:
        format_type: Output format (yaml, json, toml)
        profile: Configuration profile

    Returns:
        Template content as string
    """
    # Create default settings
    settings = Settings()

    # Apply profile-specific overrides
    manager = ConfigurationManager(profile=profile)
    profile_config = manager._load_profile_config(profile)

    if profile_config:
        # Apply profile overrides to get profile-specific template
        settings = Settings(**profile_config)

    if format_type.lower() == "yaml":
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            settings.to_yaml(f.name)
            with open(f.name, "r") as rf:
                content = rf.read()
        Path(f.name).unlink()  # Clean up temp file
        return content

    elif format_type.lower() == "json":
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            settings.to_json(f.name)
            with open(f.name, "r") as rf:
                content = rf.read()
        Path(f.name).unlink()  # Clean up temp file
        return content

    else:
        raise ValueError(f"Unsupported format: {format_type}")


def export_schema() -> Dict:
    """
    Export JSON Schema for configuration validation.

    Returns:
        JSON Schema dictionary
    """
    return Settings.model_json_schema()


def migrate_config(old_config_path: Path, new_config_path: Path) -> bool:
    """
    Migrate old configuration format to new format.

    Args:
        old_config_path: Path to old configuration
        new_config_path: Path for new configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🔄 Migrating configuration: {old_config_path} -> {new_config_path}")

        # Load old configuration (basic JSON/YAML support)
        if old_config_path.suffix.lower() in [".yaml", ".yml"]:
            import yaml

            with open(old_config_path, "r") as f:
                old_config = yaml.safe_load(f) or {}
        elif old_config_path.suffix.lower() == ".json":
            with open(old_config_path, "r") as f:
                old_config = json.load(f)
        else:
            print(f"❌ Unsupported old config format: {old_config_path.suffix}")
            return False

        # Create new settings with migration
        new_settings = Settings(**old_config)

        # Save new configuration
        if new_config_path.suffix.lower() in [".yaml", ".yml"]:
            new_settings.to_yaml(new_config_path)
        elif new_config_path.suffix.lower() == ".json":
            new_settings.to_json(new_config_path)
        else:
            print(f"❌ Unsupported new config format: {new_config_path.suffix}")
            return False

        print("✅ Configuration migrated successfully!")
        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Lab Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate config.yaml
  %(prog)s generate-template --format yaml --profile production
  %(prog)s export-schema --output schema.json
  %(prog)s migrate old_config.json new_config.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument("config_path", type=Path, help="Path to configuration file")
    validate_parser.add_argument("--strict", action="store_true", help="Enable strict validation")

    # Generate template command
    template_parser = subparsers.add_parser(
        "generate-template", help="Generate configuration template"
    )
    template_parser.add_argument(
        "--format", choices=["yaml", "json"], default="yaml", help="Output format"
    )
    template_parser.add_argument("--profile", default="development", help="Configuration profile")
    template_parser.add_argument("--output", type=Path, help="Output file (default: stdout)")

    # Export schema command
    schema_parser = subparsers.add_parser("export-schema", help="Export JSON Schema")
    schema_parser.add_argument("--output", type=Path, help="Output file (default: stdout)")

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate configuration format")
    migrate_parser.add_argument("old_config", type=Path, help="Old configuration file")
    migrate_parser.add_argument("new_config", type=Path, help="New configuration file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "validate":
            success = validate_config(args.config_path, args.strict)
            return 0 if success else 1

        elif args.command == "generate-template":
            template = generate_template(args.format, args.profile)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(template)
                print(f"✅ Template saved to: {args.output}")
            else:
                print(template)
            return 0

        elif args.command == "export-schema":
            schema = export_schema()
            schema_json = json.dumps(schema, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(schema_json)
                print(f"✅ Schema saved to: {args.output}")
            else:
                print(schema_json)
            return 0

        elif args.command == "migrate":
            success = migrate_config(args.old_config, args.new_config)
            return 0 if success else 1

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
