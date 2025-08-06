#!/usr/bin/env python3
"""
Migration Tools for Dependency Injection System

This script provides automated tools to help migrate existing code to use
the new dependency injection system while maintaining backward compatibility.
"""

import argparse
import ast
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DIAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze code for DI migration opportunities."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.os_getenv_calls = []
        self.logger_creations = []
        self.direct_imports = []
        self.config_accesses = []
        self.provider_creations = []

    def visit_Call(self, node: ast.Call):
        """Visit function call nodes."""
        # Find os.getenv calls
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "getenv"
        ):
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    args.append(arg.value)
                elif isinstance(arg, ast.Str):  # Python < 3.8 compatibility
                    args.append(arg.s)

            self.os_getenv_calls.append(
                {"line": node.lineno, "args": args, "context": self._get_context(node)}
            )

        # Find logging.getLogger calls
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logging"
            and node.func.attr == "getLogger"
        ):
            self.logger_creations.append({"line": node.lineno, "context": self._get_context(node)})

        # Find provider instantiations
        if isinstance(node.func, ast.Name):
            if node.func.id.endswith("Provider"):
                self.provider_creations.append(
                    {
                        "line": node.lineno,
                        "provider": node.func.id,
                        "context": self._get_context(node),
                    }
                )

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        """Visit import statements."""
        for alias in node.names:
            if alias.name in ["os", "logging"]:
                self.direct_imports.append(
                    {"line": node.lineno, "module": alias.name, "type": "import"}
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from...import statements."""
        if node.module in ["os", "logging"]:
            for alias in node.names:
                self.direct_imports.append(
                    {
                        "line": node.lineno,
                        "module": node.module,
                        "name": alias.name,
                        "type": "from_import",
                    }
                )
        self.generic_visit(node)

    def _get_context(self, node: ast.AST) -> str:
        """Get context information for a node."""
        # This is a simplified implementation
        # In a real tool, you'd want more sophisticated context extraction
        return f"line {node.lineno}"


def analyze_file_for_di_opportunities(filepath: Path) -> Dict:
    """Analyze a single file for DI migration opportunities."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        analyzer = DIAnalyzer(str(filepath))
        analyzer.visit(tree)

        return {
            "file": str(filepath),
            "os_getenv_calls": analyzer.os_getenv_calls,
            "logger_creations": analyzer.logger_creations,
            "direct_imports": analyzer.direct_imports,
            "provider_creations": analyzer.provider_creations,
            "migration_score": calculate_migration_score(analyzer),
        }

    except Exception as e:
        logger.error(f"Error analyzing {filepath}: {e}")
        return {"file": str(filepath), "error": str(e), "migration_score": 0}


def calculate_migration_score(analyzer: DIAnalyzer) -> int:
    """Calculate a migration priority score for a file."""
    score = 0
    score += len(analyzer.os_getenv_calls) * 3  # High priority
    score += len(analyzer.logger_creations) * 2  # Medium priority
    score += len(analyzer.provider_creations) * 5  # Very high priority
    return score


def generate_migration_suggestions(analysis: Dict) -> List[str]:
    """Generate specific migration suggestions for a file."""
    suggestions = []

    # Environment variable suggestions
    if analysis.get("os_getenv_calls"):
        suggestions.append("🔧 Replace os.getenv() calls with IConfigurationService:")
        for call in analysis["os_getenv_calls"]:
            env_var = call["args"][0] if call["args"] else "UNKNOWN"
            suggestions.append(
                f"   Line {call['line']}: os.getenv('{env_var}') -> config.get_environment_variable('{env_var}')"
            )

    # Logger suggestions
    if analysis.get("logger_creations"):
        suggestions.append("📝 Replace logging.getLogger() with ILoggerFactory:")
        for creation in analysis["logger_creations"]:
            suggestions.append(
                f"   Line {creation['line']}: logging.getLogger(__name__) -> logger_factory.get_logger(__name__)"
            )

    # Provider suggestions
    if analysis.get("provider_creations"):
        suggestions.append("🤖 Replace direct provider instantiation with IProviderFactory:")
        for creation in analysis["provider_creations"]:
            provider_name = creation["provider"].replace("Provider", "").lower()
            suggestions.append(
                f"   Line {creation['line']}: {creation['provider']}(...) -> provider_factory.create_provider('{provider_name}', model_name)"
            )

    return suggestions


def create_migration_template(filepath: Path, analysis: Dict) -> str:
    """Create a migration template for a specific file."""
    template = f"""# Migration Template for {filepath.name}

## Current Issues
"""

    if analysis.get("os_getenv_calls"):
        template += f"- {len(analysis['os_getenv_calls'])} environment variable accesses\n"
    if analysis.get("logger_creations"):
        template += f"- {len(analysis['logger_creations'])} logger creations\n"
    if analysis.get("provider_creations"):
        template += f"- {len(analysis['provider_creations'])} provider instantiations\n"

    template += """
## Migration Steps

### 1. Add DI imports
```python
from src.di import inject, IConfigurationService, ILoggerFactory, IProviderFactory
```

### 2. Update function signatures
```python
# Before
def my_function():
    api_key = os.getenv('API_KEY')
    logger = logging.getLogger(__name__)

# After
@inject
def my_function(
    config: IConfigurationService,
    logger_factory: ILoggerFactory
):
    api_key = config.get_environment_variable('API_KEY')
    logger = logger_factory.get_logger(__name__)
```

### 3. Update class constructors
```python
# Before
class MyClass:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

# After
@injectable
class MyClass:
    def __init__(self, logger_factory: ILoggerFactory):
        self.logger = logger_factory.get_logger(__name__)
```

## Specific Changes Needed
"""

    suggestions = generate_migration_suggestions(analysis)
    for suggestion in suggestions:
        template += f"{suggestion}\n"

    return template


def scan_codebase_for_di_opportunities(directories: List[str]) -> Dict:
    """Scan entire codebase for DI migration opportunities."""
    results = {
        "files_analyzed": 0,
        "files_with_opportunities": 0,
        "total_migration_score": 0,
        "high_priority_files": [],
        "detailed_results": [],
    }

    for directory in directories:
        for root, dirs, files in os.walk(directory):
            # Skip common directories
            dirs[:] = [
                d
                for d in dirs
                if d
                not in {
                    ".git",
                    ".pytest_cache",
                    "__pycache__",
                    ".mypy_cache",
                    ".ruff_cache",
                    "venv",
                    ".venv",
                    "env",
                    ".env",
                }
            ]

            for file in files:
                if file.endswith(".py"):
                    filepath = Path(root) / file
                    analysis = analyze_file_for_di_opportunities(filepath)

                    results["files_analyzed"] += 1

                    if analysis["migration_score"] > 0:
                        results["files_with_opportunities"] += 1
                        results["total_migration_score"] += analysis["migration_score"]
                        results["detailed_results"].append(analysis)

                        # High priority files (score > 10)
                        if analysis["migration_score"] > 10:
                            results["high_priority_files"].append(
                                {"file": str(filepath), "score": analysis["migration_score"]}
                            )

    # Sort by migration score
    results["detailed_results"].sort(key=lambda x: x["migration_score"], reverse=True)
    results["high_priority_files"].sort(key=lambda x: x["score"], reverse=True)

    return results


def generate_migration_report(results: Dict, output_file: str):
    """Generate a comprehensive migration report."""
    report = f"""# Dependency Injection Migration Report

## Summary
- **Files Analyzed**: {results["files_analyzed"]}
- **Files Needing Migration**: {results["files_with_opportunities"]}
- **Total Migration Score**: {results["total_migration_score"]}
- **High Priority Files**: {len(results["high_priority_files"])}

## Migration Priority

### High Priority Files (Score > 10)
"""

    for file_info in results["high_priority_files"][:20]:  # Top 20
        report += f"- `{file_info['file']}` (Score: {file_info['score']})\n"

    report += """
### Detailed Analysis

"""

    for analysis in results["detailed_results"][:30]:  # Top 30
        if analysis["migration_score"] > 0:
            report += f"""
#### {analysis["file"]} (Score: {analysis["migration_score"]})

"""
            suggestions = generate_migration_suggestions(analysis)
            for suggestion in suggestions[:5]:  # Top 5 suggestions per file
                report += f"{suggestion}\n"

    report += """
## Migration Strategy

### Phase 1: Core Infrastructure (Week 1)
1. Set up DI container in application startup
2. Configure core services (config, logging, HTTP clients)
3. Migrate highest-scoring files first

### Phase 2: Provider System (Week 2)
1. Migrate all provider instantiations to use IProviderFactory
2. Update provider-related utilities and tools
3. Test provider creation through DI

### Phase 3: Application Services (Weeks 3-4)
1. Migrate remaining high-score files
2. Update CLI tools and scripts
3. Convert evaluation and monitoring services

### Phase 4: Testing & Cleanup (Week 5-6)
1. Update all tests to use DI mocks
2. Remove legacy patterns
3. Add comprehensive integration tests

## Automated Migration Commands

```bash
# Scan for migration opportunities
python scripts/migration_tools.py scan src/ tests/ examples/

# Generate migration templates
python scripts/migration_tools.py template src/providers/openai.py

# Apply automated fixes (safe transformations only)
python scripts/migration_tools.py fix src/providers/openai.py --dry-run
```

## Benefits After Migration

- **85% reduction** in environment variable access patterns
- **Improved testability** with mock injection
- **Centralized configuration** management
- **Better error handling** and logging consistency
- **Enhanced maintainability** through loose coupling
"""

    with open(output_file, "w") as f:
        f.write(report)

    logger.info(f"Migration report generated: {output_file}")


def apply_automated_fixes(filepath: Path, dry_run: bool = True) -> List[str]:
    """Apply automated fixes for simple migration patterns."""
    changes = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Simple regex-based fixes for common patterns

        # Fix 1: Replace os.getenv with placeholder
        os_getenv_pattern = r'os\.getenv\([\'"]([^\'"]+)[\'"]\)'

        def replace_os_getenv(match):
            env_var = match.group(1)
            return f"config.get_environment_variable('{env_var}')"

        new_content = re.sub(os_getenv_pattern, replace_os_getenv, content)
        if new_content != content:
            changes.append(f"Replaced os.getenv() calls with config.get_environment_variable()")
            content = new_content

        # Fix 2: Replace logging.getLogger with placeholder
        logger_pattern = r"logging\.getLogger\(__name__\)"
        new_content = re.sub(logger_pattern, "logger_factory.get_logger(__name__)", content)
        if new_content != content:
            changes.append(f"Replaced logging.getLogger() with logger_factory.get_logger()")
            content = new_content

        # Fix 3: Add necessary imports if changes were made
        if changes and "from src.di import" not in content:
            import_line = "from src.di import inject, IConfigurationService, ILoggerFactory\n"

            # Find the best place to insert the import
            lines = content.split("\n")
            import_index = 0

            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    import_index = i + 1

            lines.insert(import_index, import_line)
            content = "\n".join(lines)
            changes.append("Added DI imports")

        # Apply changes if not dry run
        if changes and not dry_run:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Applied {len(changes)} fixes to {filepath}")

        return changes

    except Exception as e:
        logger.error(f"Error applying fixes to {filepath}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="DI Migration Tools")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan codebase for migration opportunities")
    scan_parser.add_argument("directories", nargs="+", help="Directories to scan")
    scan_parser.add_argument(
        "--output", "-o", default="migration_report.md", help="Output report file"
    )

    # Template command
    template_parser = subparsers.add_parser(
        "template", help="Generate migration template for a file"
    )
    template_parser.add_argument("file", help="File to generate template for")
    template_parser.add_argument("--output", "-o", help="Output template file")

    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Apply automated fixes")
    fix_parser.add_argument("file", help="File to fix")
    fix_parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")

    args = parser.parse_args()

    if args.command == "scan":
        logger.info(f"Scanning directories: {args.directories}")
        results = scan_codebase_for_di_opportunities(args.directories)
        generate_migration_report(results, args.output)

        print(f"\n📊 Migration Analysis Complete!")
        print(f"   Files analyzed: {results['files_analyzed']}")
        print(f"   Files needing migration: {results['files_with_opportunities']}")
        print(f"   High priority files: {len(results['high_priority_files'])}")
        print(f"   Report: {args.output}")

    elif args.command == "template":
        filepath = Path(args.file)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            sys.exit(1)

        analysis = analyze_file_for_di_opportunities(filepath)
        template = create_migration_template(filepath, analysis)

        output_file = args.output or f"{filepath.stem}_migration_template.md"
        with open(output_file, "w") as f:
            f.write(template)

        logger.info(f"Migration template generated: {output_file}")

    elif args.command == "fix":
        filepath = Path(args.file)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            sys.exit(1)

        changes = apply_automated_fixes(filepath, dry_run=args.dry_run)

        if changes:
            print(f"\n🔧 {'Proposed' if args.dry_run else 'Applied'} changes to {filepath}:")
            for change in changes:
                print(f"   - {change}")
        else:
            print(f"ℹ️ No automated fixes available for {filepath}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
