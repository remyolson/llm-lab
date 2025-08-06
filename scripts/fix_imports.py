#!/usr/bin/env python3
"""
Import Pattern Refactoring Script

This script automatically fixes common import issues identified by the audit:
- Replaces src. prefix imports with relative imports
- Converts wildcard imports to explicit imports
- Applies consistent import formatting with isort

Usage: python scripts/fix_imports.py [--dry-run] [--file FILE]
"""

import argparse
import ast
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ImportRefactor:
    """Handles refactoring of import statements in Python files."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.changes_made = []
        self.errors = []

        # Map of commonly used src modules to their relative paths
        self.src_module_map = {
            "src.providers": "providers",
            "src.config": "config",
            "src.evaluation": "evaluation",
            "src.analysis": "analysis",
            "src.logging": "logging",
            "src.use_cases": "use_cases",
            "src.utils": "utils",
        }

        # Fixture imports to expand from wildcards
        self.fixture_imports = {
            "fixtures": [
                "mock_anthropic_provider",
                "mock_google_provider",
                "mock_openai_provider",
                "sample_evaluation_data",
                "test_config",
                "temp_config_file",
            ]
        }

    def fix_file(self, file_path: Path) -> bool:
        """Fix import patterns in a single file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            if not original_content.strip():
                return False

            # Parse AST to understand imports
            try:
                tree = ast.parse(original_content)
            except SyntaxError as e:
                self.errors.append(f"Syntax error in {file_path}: {e}")
                return False

            # Fix content
            fixed_content = self._fix_imports_in_content(original_content, file_path)

            if fixed_content != original_content:
                if not self.dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(fixed_content)

                self.changes_made.append(f"Fixed imports in {file_path}")
                return True

        except Exception as e:
            self.errors.append(f"Error processing {file_path}: {e}")

        return False

    def _fix_imports_in_content(self, content: str, file_path: Path) -> str:
        """Fix import patterns in file content."""
        lines = content.split("\n")
        fixed_lines = []
        changes_made_in_file = False

        for line_num, line in enumerate(lines, 1):
            original_line = line

            # Fix src. prefix imports
            if "src." in line and ("import" in line or "from" in line):
                line = self._fix_src_prefix_import(line, file_path)
                if line != original_line:
                    changes_made_in_file = True

            # Fix wildcard imports from fixtures
            if line.strip().startswith("from") and "*" in line and "fixtures" in line:
                line = self._fix_wildcard_import(line)
                if line != original_line:
                    changes_made_in_file = True

            fixed_lines.append(line)

        fixed_content = "\n".join(fixed_lines)

        # Apply isort formatting if changes were made
        if changes_made_in_file and not self.dry_run:
            try:
                # Run isort on the content
                result = subprocess.run(
                    [
                        "python",
                        "-m",
                        "isort",
                        "--stdout",
                        "--profile=black",
                        "--line-length=120",
                        "-",
                    ],
                    input=fixed_content,
                    text=True,
                    capture_output=True,
                    cwd=Path.cwd(),
                )

                if result.returncode == 0 and result.stdout:
                    fixed_content = result.stdout
            except Exception as e:
                print(f"Warning: Could not run isort on {file_path}: {e}")

        return fixed_content

    def _fix_src_prefix_import(self, line: str, file_path: Path) -> str:
        """Fix src. prefix imports with appropriate relative imports."""
        stripped = line.strip()

        # Handle 'from src.module import ...' patterns
        from_match = re.match(r"^(\s*)from\s+src\.([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+(.+)$", line)
        if from_match:
            indent, module_path, imports = from_match.groups()

            # Determine if this is within src/ directory
            if "src/" in str(file_path):
                # Use relative import from within src/
                if module_path.count(".") > 0:
                    # Multi-level import: src.providers.base -> ..providers.base
                    relative_path = "." + module_path.replace(".", ".")
                else:
                    # Single level: src.config -> .config
                    relative_path = "." + module_path

                return f"{indent}from {relative_path} import {imports}"
            else:
                # Outside src/, use absolute import without src prefix
                return f"{indent}from {module_path} import {imports}"

        # Handle 'import src.module' patterns
        import_match = re.match(
            r"^(\s*)import\s+src\.([a-zA-Z_][a-zA-Z0-9_.]*)\s*(as\s+.+)?$", line
        )
        if import_match:
            indent, module_path, alias = import_match.groups()
            alias = alias or ""

            if "src/" in str(file_path):
                # Use relative import from within src/
                relative_path = "." + module_path
                return f"{indent}import {relative_path} {alias}".strip()
            else:
                # Outside src/, use absolute import without src prefix
                return f"{indent}import {module_path} {alias}".strip()

        return line

    def _fix_wildcard_import(self, line: str) -> str:
        """Convert wildcard imports to explicit imports."""
        # Match 'from module import *'
        match = re.match(r"^(\s*)from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+\*\s*$", line)
        if not match:
            return line

        indent, module = match.groups()

        # Replace fixtures wildcard imports
        if "fixtures" in module:
            explicit_imports = ", ".join(self.fixture_imports["fixtures"])
            return f"{indent}from {module} import {explicit_imports}"

        return line

    def run_isort_on_directory(self, directory: Path) -> bool:
        """Run isort on an entire directory."""
        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "isort",
                    str(directory),
                    "--profile=black",
                    "--line-length=120",
                    "--multi-line=3",
                    "--trailing-comma",
                    "--force-grid-wrap=0",
                    "--combine-as",
                    "--src-paths=src,tests,benchmarks,scripts",
                ],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            if result.returncode == 0:
                print(f"✅ Applied isort formatting to {directory}")
                return True
            else:
                print(f"❌ isort failed on {directory}: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error running isort on {directory}: {e}")
            return False


def get_files_to_fix(audit_report_path: str) -> List[Path]:
    """Get list of files that need import fixes from audit report."""
    import json

    files_to_fix = set()

    try:
        with open(audit_report_path, "r") as f:
            audit_data = json.load(f)

        # Get files with src. prefix issues
        src_prefix_files = audit_data.get("detailed_issues", {}).get("src_prefix_files", [])
        files_to_fix.update(src_prefix_files)

        # Get files with wildcard imports
        for star_import in audit_data.get("detailed_issues", {}).get("star_imports", []):
            if ":" in star_import:
                file_path = star_import.split(":", 1)[0]
                files_to_fix.add(file_path)

    except Exception as e:
        print(f"Warning: Could not read audit report {audit_report_path}: {e}")
        # Fallback: scan common directories
        for pattern in ["src/**/*.py", "tests/**/*.py", "scripts/**/*.py", "examples/**/*.py"]:
            files_to_fix.update(Path(".").glob(pattern))

    return [Path(f) for f in files_to_fix if Path(f).exists()]


def main():
    parser = argparse.ArgumentParser(description="Fix import patterns in Python files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without making changes"
    )
    parser.add_argument("--file", type=str, help="Fix imports in a specific file")
    parser.add_argument(
        "--audit-report",
        type=str,
        default="import_audit_report.json",
        help="Path to import audit report",
    )

    args = parser.parse_args()

    refactor = ImportRefactor(dry_run=args.dry_run)

    if args.file:
        # Fix single file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)

        print(f"🔧 Processing {file_path}")
        success = refactor.fix_file(file_path)

        if success:
            print(f"✅ Fixed imports in {file_path}")
        else:
            print(f"ℹ️ No changes needed in {file_path}")

    else:
        # Fix all files from audit report
        files_to_fix = get_files_to_fix(args.audit_report)

        if not files_to_fix:
            print("ℹ️ No files found to fix")
            sys.exit(0)

        print(f"🔍 Found {len(files_to_fix)} files to process")

        if args.dry_run:
            print("🔍 DRY RUN - No changes will be made")

        # Process each file
        for file_path in files_to_fix:
            if file_path.exists():
                refactor.fix_file(file_path)

        # Apply isort to directories if not dry run
        if not args.dry_run and refactor.changes_made:
            print("\n🔄 Applying isort formatting...")
            for directory in ["src", "tests", "scripts", "examples"]:
                if Path(directory).exists():
                    refactor.run_isort_on_directory(Path(directory))

    # Print summary
    print(f"\n📊 Summary:")
    print(f"   ✅ Files modified: {len(refactor.changes_made)}")
    print(f"   ❌ Errors: {len(refactor.errors)}")

    if refactor.changes_made:
        print(f"\n📝 Changes made:")
        for change in refactor.changes_made:
            print(f"   - {change}")

    if refactor.errors:
        print(f"\n❌ Errors encountered:")
        for error in refactor.errors:
            print(f"   - {error}")

    print(f"\n🏁 Import refactoring {'simulation' if args.dry_run else 'completed'}!")


if __name__ == "__main__":
    main()
