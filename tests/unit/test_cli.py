"""
Unit tests for CLI interfaces and command-line functionality.
"""

import argparse
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.mark.unit
class TestCommandLineInterface:
    """Test CLI argument parsing and command handling."""

    def test_basic_argument_parsing(self):
        """Test basic CLI argument parsing."""

        def create_parser():
            parser = argparse.ArgumentParser(description="LLM Lab CLI")

            parser.add_argument(
                "--provider", type=str, default="openai", help="LLM provider to use"
            )
            parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to use")
            parser.add_argument(
                "--temperature", type=float, default=0.7, help="Generation temperature"
            )
            parser.add_argument(
                "--verbose", "-v", action="store_true", help="Enable verbose output"
            )
            parser.add_argument("prompt", type=str, nargs="?", help="Prompt to process")

            return parser

        parser = create_parser()

        # Test basic parsing
        args = parser.parse_args(["--provider", "anthropic", "--model", "claude-3", "Hello world"])
        assert args.provider == "anthropic"
        assert args.model == "claude-3"
        assert args.temperature == 0.7  # Default value
        assert args.prompt == "Hello world"
        assert args.verbose == False

        # Test with flags
        args = parser.parse_args(["--verbose", "--temperature", "0.9", "Test prompt"])
        assert args.verbose == True
        assert args.temperature == 0.9
        assert args.prompt == "Test prompt"

    def test_subcommand_parsing(self):
        """Test CLI subcommand parsing."""

        def create_subcommand_parser():
            parser = argparse.ArgumentParser(description="LLM Lab CLI")
            subparsers = parser.add_subparsers(dest="command", help="Available commands")

            # Generate command
            generate_parser = subparsers.add_parser("generate", help="Generate text")
            generate_parser.add_argument("prompt", type=str, help="Text prompt")
            generate_parser.add_argument("--provider", default="openai")

            # Compare command
            compare_parser = subparsers.add_parser("compare", help="Compare providers")
            compare_parser.add_argument("prompt", type=str, help="Text prompt")
            compare_parser.add_argument("--providers", nargs="+", default=["openai", "anthropic"])

            # Benchmark command
            benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
            benchmark_parser.add_argument("--config", type=str, help="Config file")
            benchmark_parser.add_argument("--output", type=str, help="Output file")

            return parser

        parser = create_subcommand_parser()

        # Test generate subcommand
        args = parser.parse_args(["generate", "Hello world", "--provider", "anthropic"])
        assert args.command == "generate"
        assert args.prompt == "Hello world"
        assert args.provider == "anthropic"

        # Test compare subcommand
        args = parser.parse_args(
            ["compare", "Test prompt", "--providers", "openai", "google", "anthropic"]
        )
        assert args.command == "compare"
        assert args.prompt == "Test prompt"
        assert args.providers == ["openai", "google", "anthropic"]

        # Test benchmark subcommand
        args = parser.parse_args(["benchmark", "--config", "test.yaml", "--output", "results.json"])
        assert args.command == "benchmark"
        assert args.config == "test.yaml"
        assert args.output == "results.json"

    def test_invalid_arguments(self):
        """Test handling of invalid CLI arguments."""

        def create_parser():
            parser = argparse.ArgumentParser(description="LLM Lab CLI")
            parser.add_argument("--temperature", type=float, choices=[0.1, 0.5, 0.7, 1.0])
            parser.add_argument("--max-tokens", type=int)
            parser.add_argument("prompt", type=str)
            return parser

        parser = create_parser()

        # Test invalid temperature
        with pytest.raises(SystemExit):
            with patch("sys.stderr", StringIO()):
                parser.parse_args(["--temperature", "2.0", "test"])

        # Test invalid max-tokens
        with pytest.raises(SystemExit):
            with patch("sys.stderr", StringIO()):
                parser.parse_args(["--max-tokens", "invalid", "test"])

        # Test missing required argument
        with pytest.raises(SystemExit):
            with patch("sys.stderr", StringIO()):
                parser.parse_args(["--temperature", "0.7"])


@pytest.mark.unit
class TestCommandExecution:
    """Test CLI command execution logic."""

    def test_generate_command_execution(self):
        """Test generate command execution."""

        def execute_generate_command(provider, prompt, temperature=0.7, max_tokens=1000):
            # Simulate command execution
            if not provider:
                raise ValueError("Provider is required")
            if not prompt:
                raise ValueError("Prompt is required")
            if not 0 <= temperature <= 2:
                raise ValueError("Temperature must be between 0 and 2")

            return {
                "provider": provider,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response": f"Generated response from {provider}",
                "status": "success",
            }

        # Valid execution
        result = execute_generate_command("openai", "Hello world", 0.7, 1000)
        assert result["status"] == "success"
        assert result["provider"] == "openai"
        assert result["prompt"] == "Hello world"
        assert "Generated response" in result["response"]

        # Invalid parameters
        with pytest.raises(ValueError, match="Provider is required"):
            execute_generate_command("", "Hello world")

        with pytest.raises(ValueError, match="Prompt is required"):
            execute_generate_command("openai", "")

        with pytest.raises(ValueError, match="Temperature must be between"):
            execute_generate_command("openai", "Hello world", 3.0)

    def test_compare_command_execution(self):
        """Test compare command execution."""

        def execute_compare_command(providers, prompt, temperature=0.7):
            if not providers:
                raise ValueError("At least one provider is required")
            if not prompt:
                raise ValueError("Prompt is required")

            results = []
            for provider in providers:
                results.append(
                    {
                        "provider": provider,
                        "response": f"Response from {provider}",
                        "latency": hash(provider) % 1000,  # Mock latency
                        "tokens": len(prompt.split()) * 10,  # Mock token count
                    }
                )

            return {
                "prompt": prompt,
                "results": results,
                "comparison": {
                    "fastest": min(results, key=lambda x: x["latency"])["provider"],
                    "most_tokens": max(results, key=lambda x: x["tokens"])["provider"],
                },
            }

        result = execute_compare_command(["openai", "anthropic"], "Hello world", 0.7)

        assert len(result["results"]) == 2
        assert result["prompt"] == "Hello world"
        assert "fastest" in result["comparison"]
        assert "most_tokens" in result["comparison"]

        # Test with single provider
        single_result = execute_compare_command(["openai"], "Test prompt")
        assert len(single_result["results"]) == 1

    def test_benchmark_command_execution(self, temp_dir):
        """Test benchmark command execution."""

        def execute_benchmark_command(config_file=None, output_file=None, iterations=10):
            # Create mock config if not provided
            if not config_file:
                config = {
                    "providers": ["openai", "anthropic"],
                    "prompts": ["Hello", "What is AI?"],
                    "metrics": ["latency", "tokens", "cost"],
                }
            else:
                # In real implementation, would load from file
                config = {"providers": ["openai"], "prompts": ["Test"]}

            # Run mock benchmarks
            results = []
            for provider in config["providers"]:
                for prompt in config["prompts"]:
                    for i in range(iterations):
                        results.append(
                            {
                                "provider": provider,
                                "prompt": prompt,
                                "iteration": i,
                                "latency": (hash(f"{provider}{prompt}{i}") % 1000) / 1000.0,
                                "tokens": len(prompt) * 5,
                                "cost": 0.001,
                            }
                        )

            benchmark_results = {
                "config": config,
                "results": results,
                "summary": {
                    "total_iterations": len(results),
                    "providers_tested": len(config["providers"]),
                    "prompts_tested": len(config["prompts"]),
                },
            }

            # Save to output file if specified
            if output_file:
                import json

                with open(output_file, "w") as f:
                    json.dump(benchmark_results, f, indent=2)

            return benchmark_results

        # Test with defaults
        results = execute_benchmark_command()
        assert results["summary"]["providers_tested"] == 2
        assert results["summary"]["prompts_tested"] == 2
        assert len(results["results"]) == 40  # 2 providers * 2 prompts * 10 iterations

        # Test with output file
        output_file = temp_dir / "benchmark_results.json"
        results = execute_benchmark_command(output_file=str(output_file))

        assert output_file.exists()

        import json

        with open(output_file) as f:
            saved_results = json.load(f)

        assert saved_results["summary"]["total_iterations"] == len(results["results"])


@pytest.mark.unit
class TestCLIOutputFormatting:
    """Test CLI output formatting and display."""

    def test_table_output_formatting(self):
        """Test formatting output as tables."""

        def format_table(data, headers):
            if not data or not headers:
                return ""

            # Calculate column widths
            widths = [len(h) for h in headers]
            for row in data:
                for i, cell in enumerate(row):
                    widths[i] = max(widths[i], len(str(cell)))

            # Format table
            lines = []

            # Header
            header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
            lines.append(header_line)
            lines.append("-" * len(header_line))

            # Data rows
            for row in data:
                row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
                lines.append(row_line)

            return "\n".join(lines)

        # Test basic table
        headers = ["Provider", "Model", "Latency"]
        data = [
            ["OpenAI", "gpt-4", "1.2s"],
            ["Anthropic", "claude-3", "0.8s"],
            ["Google", "gemini-pro", "1.5s"],
        ]

        table = format_table(data, headers)
        lines = table.split("\n")

        assert "Provider" in lines[0]
        assert "OpenAI" in lines[2]
        assert "claude-3" in lines[3]
        assert len(lines) == 5  # Header + separator + 3 data rows

    def test_json_output_formatting(self):
        """Test formatting output as JSON."""
        import json

        def format_json_output(data, pretty=True):
            if pretty:
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                return json.dumps(data, separators=(",", ":"), ensure_ascii=False)

        test_data = {
            "provider": "openai",
            "model": "gpt-4",
            "response": "Hello world!",
            "metadata": {"tokens": 15, "latency": 1.2},
        }

        # Pretty formatted
        pretty_json = format_json_output(test_data, pretty=True)
        assert '  "provider": "openai"' in pretty_json
        assert pretty_json.count("\n") > 5

        # Compact formatted
        compact_json = format_json_output(test_data, pretty=False)
        assert "\n" not in compact_json
        assert "  " not in compact_json

    def test_progress_bar_formatting(self):
        """Test progress bar formatting."""

        def format_progress_bar(current, total, width=50, fill="█", empty=" "):
            if total <= 0:
                return "[ERROR: Invalid total]"

            progress = min(current / total, 1.0)
            filled_width = int(progress * width)
            bar = fill * filled_width + empty * (width - filled_width)
            percentage = progress * 100

            return f"[{bar}] {percentage:.1f}% ({current}/{total})"

        # Test various progress levels
        assert "0.0%" in format_progress_bar(0, 100)
        assert "50.0%" in format_progress_bar(50, 100)
        assert "100.0%" in format_progress_bar(100, 100)

        # Test edge cases
        assert "100.0%" in format_progress_bar(150, 100)  # Over 100%
        assert "ERROR" in format_progress_bar(50, 0)  # Invalid total

    def test_colored_output_formatting(self):
        """Test colored terminal output formatting."""

        def format_colored_text(text, color=None, style=None):
            colors = {
                "red": "\033[91m",
                "green": "\033[92m",
                "yellow": "\033[93m",
                "blue": "\033[94m",
                "magenta": "\033[95m",
                "cyan": "\033[96m",
            }

            styles = {"bold": "\033[1m", "underline": "\033[4m", "italic": "\033[3m"}

            reset = "\033[0m"

            prefix = ""
            if style and style in styles:
                prefix += styles[style]
            if color and color in colors:
                prefix += colors[color]

            if prefix:
                return f"{prefix}{text}{reset}"
            return text

        # Test colors
        red_text = format_colored_text("Error!", "red")
        assert "\033[91m" in red_text
        assert "\033[0m" in red_text

        # Test styles
        bold_text = format_colored_text("Important", style="bold")
        assert "\033[1m" in bold_text

        # Test combined
        bold_red = format_colored_text("Critical!", "red", "bold")
        assert "\033[1m" in bold_red
        assert "\033[91m" in bold_red

        # Test plain text
        plain = format_colored_text("Normal text")
        assert "\033[" not in plain


@pytest.mark.unit
class TestCLIErrorHandling:
    """Test CLI error handling and user feedback."""

    def test_user_friendly_error_messages(self):
        """Test conversion of technical errors to user-friendly messages."""

        def format_error_message(error):
            error_mappings = {
                "ConnectionError": "Unable to connect to the API. Please check your internet connection.",
                "AuthenticationError": "Authentication failed. Please check your API key.",
                "RateLimitError": "Rate limit exceeded. Please try again later.",
                "InvalidModelError": "Invalid model specified. Please check available models.",
                "ValidationError": "Invalid input parameters. Please check your request.",
                "TimeoutError": "Request timed out. Please try again or reduce the prompt size.",
            }

            error_type = type(error).__name__
            if error_type in error_mappings:
                return f"Error: {error_mappings[error_type]}"
            else:
                return f"Unexpected error: {str(error)}"

        # Test known error types
        class ConnectionError(Exception):
            pass

        class AuthenticationError(Exception):
            pass

        connection_error = ConnectionError("Network unreachable")
        assert "internet connection" in format_error_message(connection_error)

        auth_error = AuthenticationError("Invalid key")
        assert "API key" in format_error_message(auth_error)

        # Test unknown error
        unknown_error = ValueError("Something weird happened")
        assert "Unexpected error" in format_error_message(unknown_error)

    def test_help_text_generation(self):
        """Test automatic help text generation."""

        def generate_command_help(command_info):
            help_text = [f"Usage: llm-lab {command_info['name']} [OPTIONS]"]
            help_text.append("")
            help_text.append(command_info["description"])
            help_text.append("")

            if "options" in command_info:
                help_text.append("Options:")
                for option in command_info["options"]:
                    option_line = f"  {option['flag']:<20} {option['description']}"
                    if "default" in option:
                        option_line += f" (default: {option['default']})"
                    help_text.append(option_line)
                help_text.append("")

            if "examples" in command_info:
                help_text.append("Examples:")
                for example in command_info["examples"]:
                    help_text.append(f"  {example}")
                help_text.append("")

            return "\n".join(help_text)

        command_info = {
            "name": "generate",
            "description": "Generate text using LLM providers.",
            "options": [
                {
                    "flag": "--provider PROVIDER",
                    "description": "LLM provider to use",
                    "default": "openai",
                },
                {
                    "flag": "--temperature TEMP",
                    "description": "Generation temperature",
                    "default": 0.7,
                },
                {"flag": "--verbose", "description": "Enable verbose output"},
            ],
            "examples": [
                'llm-lab generate "Hello world"',
                'llm-lab generate --provider anthropic "Explain AI"',
                'llm-lab generate --temperature 0.9 --verbose "Creative story"',
            ],
        }

        help_text = generate_command_help(command_info)

        assert "Usage: llm-lab generate" in help_text
        assert "Generate text using LLM providers" in help_text
        assert "--provider PROVIDER" in help_text
        assert "(default: openai)" in help_text
        assert "Examples:" in help_text
        assert "llm-lab generate" in help_text

    def test_input_validation_feedback(self):
        """Test input validation with helpful feedback."""

        def validate_cli_input(args):
            errors = []
            warnings = []

            # Validate provider
            valid_providers = ["openai", "anthropic", "google"]
            if hasattr(args, "provider") and args.provider not in valid_providers:
                errors.append(
                    f"Invalid provider '{args.provider}'. Valid options: {', '.join(valid_providers)}"
                )

            # Validate temperature
            if hasattr(args, "temperature"):
                if not isinstance(args.temperature, (int, float)):
                    errors.append("Temperature must be a number")
                elif not 0 <= args.temperature <= 2:
                    errors.append("Temperature must be between 0 and 2")
                elif args.temperature > 1.5:
                    warnings.append("High temperature (>1.5) may produce very random results")

            # Validate max_tokens
            if hasattr(args, "max_tokens"):
                if not isinstance(args.max_tokens, int):
                    errors.append("max_tokens must be an integer")
                elif args.max_tokens <= 0:
                    errors.append("max_tokens must be positive")
                elif args.max_tokens > 32000:
                    warnings.append("Very high max_tokens may be slow and expensive")

            return errors, warnings

        # Mock args object
        class Args:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        # Valid input
        valid_args = Args(provider="openai", temperature=0.7, max_tokens=1000)
        errors, warnings = validate_cli_input(valid_args)
        assert len(errors) == 0
        assert len(warnings) == 0

        # Invalid provider
        invalid_provider_args = Args(provider="invalid", temperature=0.7)
        errors, warnings = validate_cli_input(invalid_provider_args)
        assert len(errors) == 1
        assert "Invalid provider" in errors[0]
        assert "openai, anthropic, google" in errors[0]

        # High temperature warning
        high_temp_args = Args(provider="openai", temperature=1.8)
        errors, warnings = validate_cli_input(high_temp_args)
        assert len(errors) == 0
        assert len(warnings) == 1
        assert "High temperature" in warnings[0]


@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI integration scenarios."""

    def test_end_to_end_cli_workflow(self, temp_dir):
        """Test complete CLI workflow simulation."""

        # Simulate a complete CLI session
        def simulate_cli_session(commands):
            results = []

            for command in commands:
                # Parse command
                parts = command.strip().split()

                if parts[0] == "generate":
                    result = {
                        "command": "generate",
                        "prompt": " ".join(parts[1:]),
                        "response": f"Generated response for: {' '.join(parts[1:])}",
                        "status": "success",
                    }
                elif parts[0] == "compare":
                    result = {
                        "command": "compare",
                        "prompt": " ".join(parts[1:]),
                        "providers": ["openai", "anthropic"],
                        "winner": "openai",
                        "status": "success",
                    }
                elif parts[0] == "benchmark":
                    result = {
                        "command": "benchmark",
                        "tests_run": 10,
                        "avg_latency": 1.2,
                        "status": "success",
                    }
                else:
                    result = {
                        "command": parts[0],
                        "error": f"Unknown command: {parts[0]}",
                        "status": "error",
                    }

                results.append(result)

            return results

        commands = [
            'generate "Hello world"',
            'compare "What is AI?"',
            "benchmark",
            "invalid_command",
        ]

        results = simulate_cli_session(commands)

        assert len(results) == 4
        assert results[0]["status"] == "success"
        assert results[0]["command"] == "generate"
        assert results[1]["status"] == "success"
        assert results[1]["command"] == "compare"
        assert results[2]["status"] == "success"
        assert results[3]["status"] == "error"
        assert "Unknown command" in results[3]["error"]
