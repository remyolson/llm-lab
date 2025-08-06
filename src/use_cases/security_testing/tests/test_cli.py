"""Tests for the LLM Security Testing CLI."""

import json

# Import CLI module
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent))
from cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_scanner():
    """Create a mock security scanner."""
    with patch("cli.SecurityScanner") as mock:
        scanner_instance = MagicMock()
        mock.return_value = scanner_instance
        yield scanner_instance


@pytest.fixture
def mock_scorer():
    """Create a mock security scorer."""
    with patch("cli.SecurityScorer") as mock:
        scorer_instance = MagicMock()
        assessment = MagicMock()
        assessment.overall_score = 75.0
        assessment.security_posture = "Good"
        assessment.severity_level.value = "low"
        assessment.vulnerabilities_by_type = {}
        assessment.risk_factors = ["Test risk factor"]
        assessment.recommendations = ["Test recommendation"]
        assessment.to_dict.return_value = {
            "overall_score": 75.0,
            "security_posture": "Good",
            "severity_level": "low",
        }
        scorer_instance.calculate_security_score.return_value = assessment
        mock.return_value = scorer_instance
        yield scorer_instance


class TestCLICommands:
    """Test suite for CLI commands."""

    def test_cli_version(self, runner):
        """Test version display."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_cli_help(self, runner):
        """Test help display."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "LLM Security Testing Framework" in result.output
        assert "scan" in result.output
        assert "enterprise-scan" in result.output
        assert "red-team" in result.output

    def test_scan_command_basic(self, runner, mock_scanner, mock_scorer):
        """Test basic scan command."""
        result = runner.invoke(
            cli,
            [
                "scan",
                "--model",
                "gpt-4",
                "--test-suites",
                "jailbreak",
                "--severity-threshold",
                "medium",
            ],
        )
        assert result.exit_code == 0
        assert "Security Scan" in result.output
        assert "Model: gpt-4" in result.output

    def test_scan_command_with_output(self, runner, mock_scanner, mock_scorer):
        """Test scan command with output file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = tmp.name

        result = runner.invoke(
            cli,
            [
                "scan",
                "--model",
                "gpt-4",
                "--test-suites",
                "all",
                "--output-format",
                "json",
                "--output-file",
                output_file,
            ],
        )

        assert result.exit_code == 0
        assert Path(output_file).exists()

        # Clean up
        Path(output_file).unlink()

    def test_scan_command_parallel(self, runner, mock_scanner, mock_scorer):
        """Test scan command with parallel execution."""
        result = runner.invoke(
            cli,
            [
                "scan",
                "--model",
                "claude-3-opus",
                "--test-suites",
                "jailbreak",
                "--test-suites",
                "injection",
                "--parallel",
                "--timeout",
                "60",
            ],
        )
        assert result.exit_code == 0

    def test_enterprise_scan_command(self, runner, mock_scanner):
        """Test enterprise scan command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                cli,
                [
                    "enterprise-scan",
                    "--model",
                    "gpt-4",
                    "--compliance-frameworks",
                    "owasp-llm-top10",
                    "--compliance-frameworks",
                    "gdpr",
                    "--output-dir",
                    tmpdir,
                ],
            )
            assert result.exit_code == 0
            assert "Enterprise Security Scan" in result.output
            assert "Compliance: owasp-llm-top10, gdpr" in result.output

    def test_enterprise_scan_with_evidence(self, runner, mock_scanner):
        """Test enterprise scan with evidence generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                cli,
                [
                    "enterprise-scan",
                    "--model",
                    "llama-2-70b",
                    "--compliance-frameworks",
                    "hipaa",
                    "--generate-evidence",
                    "--output-dir",
                    tmpdir,
                ],
            )
            assert result.exit_code == 0
            assert "Evidence Generation: Yes" in result.output

    def test_red_team_command(self, runner):
        """Test red team command."""
        with patch("cli.RedTeamSimulator") as mock_simulator:
            simulator_instance = MagicMock()
            mock_simulator.return_value = simulator_instance

            result = runner.invoke(
                cli,
                [
                    "red-team",
                    "--model",
                    "gpt-3.5-turbo",
                    "--scenario",
                    "customer-service",
                    "--intensity",
                    "high",
                    "--duration",
                    "5",
                ],
            )

            assert result.exit_code == 0
            assert "Red Team Simulation" in result.output
            assert "Target Model: gpt-3.5-turbo" in result.output
            assert "Scenario: customer-service" in result.output

    def test_red_team_interactive(self, runner):
        """Test red team interactive mode."""
        with patch("cli.RedTeamSimulator") as mock_simulator:
            with patch("cli.Prompt.ask", side_effect=["exit"]):
                result = runner.invoke(cli, ["red-team", "--model", "gpt-4", "--interactive"])
                assert result.exit_code == 0
                assert "Interactive Red Team Mode" in result.output

    def test_generate_report_command(self, runner):
        """Test report generation command."""
        # Create a temporary scan results file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump({"overall_score": 80.0, "findings": []}, tmp)
            scan_results = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            output_file = tmp.name

        result = runner.invoke(
            cli,
            [
                "generate-report",
                "--scan-results",
                scan_results,
                "--format",
                "pdf",
                "--template",
                "executive",
                "--output",
                output_file,
            ],
        )

        assert result.exit_code == 0
        assert "Report saved to" in result.output

        # Clean up
        Path(scan_results).unlink()
        Path(output_file).unlink()

    def test_generate_report_with_recommendations(self, runner):
        """Test report generation with recommendations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump({"findings": []}, tmp)
            scan_results = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            output_file = tmp.name

        result = runner.invoke(
            cli,
            [
                "generate-report",
                "--scan-results",
                scan_results,
                "--format",
                "html",
                "--template",
                "technical",
                "--output",
                output_file,
                "--include-recommendations",
            ],
        )

        assert result.exit_code == 0

        # Clean up
        Path(scan_results).unlink()
        Path(output_file).unlink()

    def test_interactive_command(self, runner):
        """Test interactive mode command."""
        with patch("cli.Prompt.ask", side_effect=["gpt-4", "quick-scan"]):
            with patch("cli.run_quick_scan_interactive"):
                result = runner.invoke(cli, ["interactive"])
                assert result.exit_code == 0
                assert "Interactive Security Testing Mode" in result.output

    def test_config_file_loading(self, runner):
        """Test configuration file loading."""
        config_data = {"default_model": "gpt-4", "severity_threshold": "high"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(config_data, tmp)
            config_file = tmp.name

        result = runner.invoke(cli, ["--config", config_file, "--help"])

        assert result.exit_code == 0
        assert "Loaded configuration from" in result.output

        # Clean up
        Path(config_file).unlink()


class TestCLIParameters:
    """Test CLI parameter validation."""

    def test_invalid_model(self, runner):
        """Test with missing model parameter."""
        result = runner.invoke(cli, ["scan"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_invalid_severity_threshold(self, runner):
        """Test with invalid severity threshold."""
        result = runner.invoke(cli, ["scan", "--model", "gpt-4", "--severity-threshold", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_invalid_output_format(self, runner):
        """Test with invalid output format."""
        result = runner.invoke(cli, ["scan", "--model", "gpt-4", "--output-format", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_multiple_test_suites(self, runner, mock_scanner, mock_scorer):
        """Test with multiple test suites."""
        result = runner.invoke(
            cli,
            [
                "scan",
                "--model",
                "gpt-4",
                "--test-suites",
                "jailbreak",
                "--test-suites",
                "injection",
                "--test-suites",
                "extraction",
            ],
        )
        assert result.exit_code == 0
        assert "Test Suites: jailbreak, injection, extraction" in result.output

    def test_invalid_compliance_framework(self, runner):
        """Test with invalid compliance framework."""
        result = runner.invoke(
            cli,
            ["enterprise-scan", "--model", "gpt-4", "--compliance-frameworks", "invalid-framework"],
        )
        assert result.exit_code != 0
        assert "Invalid value" in result.output


class TestCLIOutput:
    """Test CLI output formatting."""

    def test_json_output(self, runner, mock_scanner, mock_scorer):
        """Test JSON output format."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = tmp.name

        result = runner.invoke(
            cli,
            ["scan", "--model", "gpt-4", "--output-format", "json", "--output-file", output_file],
        )

        assert result.exit_code == 0
        assert Path(output_file).exists()

        # Verify JSON is valid
        with open(output_file) as f:
            data = json.load(f)
            assert "overall_score" in data

        # Clean up
        Path(output_file).unlink()

    def test_verbose_output(self, runner, mock_scanner, mock_scorer):
        """Test verbose output mode."""
        result = runner.invoke(cli, ["--verbose", "scan", "--model", "gpt-4"])
        assert result.exit_code == 0
        # In a real implementation, verbose would add more output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
