#!/usr/bin/env python3
"""
Import Pattern Audit Script

This script analyzes all Python files in the codebase to identify:
- Import patterns and inconsistencies
- src. prefix usage
- Wildcard imports
- Circular dependencies
- Import ordering issues

Usage: python scripts/audit_imports.py
"""

import ast
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze import patterns in Python files."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports = []
        self.from_imports = []
        self.import_order = []
        self.star_imports = []
        self.issues = []

    def visit_Import(self, node: ast.Import):
        """Visit regular import statements."""
        for alias in node.names:
            import_info = {
                "type": "import",
                "module": alias.name,
                "alias": alias.asname,
                "line": node.lineno,
                "col_offset": node.col_offset,
            }
            self.imports.append(import_info)
            self.import_order.append(("import", alias.name, node.lineno))

            # Check for common issues
            if alias.name.startswith("src."):
                self.issues.append(f"Line {node.lineno}: Uses 'src.' prefix import: {alias.name}")

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from ... import statements."""
        module = node.module or ""
        level = node.level

        for alias in node.names:
            import_info = {
                "type": "from_import",
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
                "level": level,
                "line": node.lineno,
                "col_offset": node.col_offset,
            }
            self.from_imports.append(import_info)
            self.import_order.append(("from", module, node.lineno))

            # Check for star imports
            if alias.name == "*":
                self.star_imports.append({"module": module, "line": node.lineno, "level": level})
                self.issues.append(f"Line {node.lineno}: Wildcard import from {module}")

            # Check for src. prefix usage
            if module and module.startswith("src."):
                self.issues.append(f"Line {node.lineno}: Uses 'src.' prefix import: from {module}")

        self.generic_visit(node)


def analyze_file(filepath: Path) -> Dict[str, Any]:
    """Analyze a single Python file for import patterns."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        analyzer = ImportAnalyzer(str(filepath))
        analyzer.visit(tree)

        return {
            "file": str(filepath),
            "imports": analyzer.imports,
            "from_imports": analyzer.from_imports,
            "star_imports": analyzer.star_imports,
            "issues": analyzer.issues,
            "import_count": len(analyzer.imports) + len(analyzer.from_imports),
            "order_violations": detect_order_violations(analyzer.import_order),
        }

    except Exception as e:
        return {
            "file": str(filepath),
            "error": str(e),
            "imports": [],
            "from_imports": [],
            "star_imports": [],
            "issues": [f"Parse error: {e}"],
            "import_count": 0,
            "order_violations": [],
        }


def detect_order_violations(import_order: List[Tuple[str, str, int]]) -> List[str]:
    """Detect PEP 8 import order violations."""
    violations = []

    # Expected order: stdlib, third-party, first-party, local
    stdlib_modules = {
        "os",
        "sys",
        "json",
        "ast",
        "pathlib",
        "collections",
        "typing",
        "logging",
        "time",
        "datetime",
        "threading",
        "concurrent",
        "asyncio",
        "functools",
        "itertools",
        "operator",
        "copy",
        "pickle",
        "re",
        "math",
        "random",
        "unittest",
        "dataclasses",
        "enum",
        "abc",
        "warnings",
        "inspect",
    }

    third_party_indicators = {
        "pytest",
        "click",
        "pydantic",
        "requests",
        "anthropic",
        "openai",
        "google",
        "flask",
        "fastapi",
        "sqlalchemy",
        "pandas",
        "numpy",
        "torch",
        "transformers",
        "yaml",
        "toml",
        "ruff",
        "black",
        "isort",
    }

    first_party_indicators = {"src", "tests", "benchmarks", "scripts"}

    sections = []
    for import_type, module, line_num in import_order:
        module_root = module.split(".")[0] if module else ""

        if module_root in stdlib_modules:
            section = "stdlib"
        elif module_root in third_party_indicators:
            section = "third_party"
        elif module_root in first_party_indicators:
            section = "first_party"
        elif module.startswith("."):
            section = "local"
        else:
            section = "unknown"

        sections.append((section, line_num, module))

    # Check for section ordering violations
    section_order = {"stdlib": 0, "third_party": 1, "first_party": 2, "local": 3, "unknown": 4}

    for i in range(1, len(sections)):
        prev_section, prev_line, prev_module = sections[i - 1]
        curr_section, curr_line, curr_module = sections[i]

        if section_order[curr_section] < section_order[prev_section]:
            violations.append(
                f"Import order violation: {curr_module} (line {curr_line}) "
                f"should come before {prev_module} (line {prev_line})"
            )

    return violations


def find_circular_dependencies(analysis_results: List[Dict[str, Any]]) -> List[str]:
    """Detect potential circular dependencies."""
    # Build a dependency graph
    dependencies = defaultdict(set)

    for result in analysis_results:
        if "error" in result:
            continue

        file_path = result["file"]
        module_name = path_to_module(file_path)

        for imp in result["imports"]:
            dependencies[module_name].add(imp["module"])

        for imp in result["from_imports"]:
            if imp["module"]:
                if imp["level"] > 0:  # Relative import
                    # Convert relative import to absolute
                    target = resolve_relative_import(module_name, imp["module"], imp["level"])
                    if target:
                        dependencies[module_name].add(target)
                else:
                    dependencies[module_name].add(imp["module"])

    # Find cycles using DFS
    cycles = []
    visited = set()
    rec_stack = set()

    def dfs(node, path):
        if node in rec_stack:
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(" -> ".join(cycle))
            return

        if node in visited:
            return

        visited.add(node)
        rec_stack.add(node)

        for neighbor in dependencies.get(node, []):
            if neighbor in dependencies:  # Only follow imports to our own modules
                dfs(neighbor, path + [node])

        rec_stack.remove(node)

    for module in dependencies:
        if module not in visited:
            dfs(module, [])

    return cycles


def path_to_module(file_path: str) -> str:
    """Convert file path to module name."""
    # Remove .py extension and convert slashes to dots
    module = file_path.replace(".py", "").replace("/", ".").replace("\\", ".")

    # Remove common prefixes
    for prefix in ["src.", "tests.", "scripts.", "benchmarks."]:
        if module.startswith(prefix):
            module = module[len(prefix) :]
            break

    return module


def resolve_relative_import(current_module: str, import_module: str, level: int) -> str:
    """Resolve relative import to absolute module name."""
    parts = current_module.split(".")

    # Go up 'level' directories
    for _ in range(level):
        if parts:
            parts.pop()

    if import_module:
        parts.extend(import_module.split("."))

    return ".".join(parts) if parts else None


def generate_report(analysis_results: List[Dict[str, Any]], output_file: str):
    """Generate a comprehensive import audit report."""

    # Collect statistics
    total_files = len(analysis_results)
    files_with_issues = len([r for r in analysis_results if r.get("issues")])
    total_imports = sum(r.get("import_count", 0) for r in analysis_results)
    total_issues = sum(len(r.get("issues", [])) for r in analysis_results)

    # Count different types of issues
    issue_types = Counter()
    star_imports = []
    src_prefix_files = set()

    for result in analysis_results:
        for issue in result.get("issues", []):
            if "Wildcard import" in issue:
                issue_types["wildcard_imports"] += 1
                star_imports.append(f"{result['file']}: {issue}")
            elif "src." in issue:
                issue_types["src_prefix"] += 1
                src_prefix_files.add(result["file"])
            elif "order violation" in issue.lower():
                issue_types["order_violations"] += 1

    # Circular dependencies
    cycles = find_circular_dependencies(analysis_results)

    # Generate report
    report = {
        "summary": {
            "total_files_analyzed": total_files,
            "files_with_issues": files_with_issues,
            "total_imports": total_imports,
            "total_issues": total_issues,
            "issue_types": dict(issue_types),
        },
        "detailed_issues": {
            "star_imports": star_imports,
            "src_prefix_files": list(src_prefix_files),
            "circular_dependencies": cycles,
        },
        "files": analysis_results,
    }

    # Write JSON report
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    # Generate human-readable summary
    summary_file = output_file.replace(".json", "_summary.md")
    with open(summary_file, "w") as f:
        f.write("# Import Pattern Audit Report\n\n")
        f.write(f"**Analysis Date:** {os.popen('date').read().strip()}\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Files Analyzed:** {total_files}\n")
        f.write(f"- **Files with Issues:** {files_with_issues}\n")
        f.write(f"- **Total Import Statements:** {total_imports}\n")
        f.write(f"- **Total Issues Found:** {total_issues}\n\n")

        f.write("## Issue Breakdown\n\n")
        for issue_type, count in issue_types.items():
            f.write(f"- **{issue_type.replace('_', ' ').title()}:** {count}\n")

        if cycles:
            f.write(f"\n## Circular Dependencies ({len(cycles)} found)\n\n")
            for cycle in cycles:
                f.write(f"- {cycle}\n")

        if src_prefix_files:
            f.write(f"\n## Files Using 'src.' Prefix ({len(src_prefix_files)} files)\n\n")
            for file in sorted(src_prefix_files):
                f.write(f"- {file}\n")

        if star_imports:
            f.write(f"\n## Wildcard Imports ({len(star_imports)} found)\n\n")
            for star_import in star_imports:
                f.write(f"- {star_import}\n")

        f.write("\n## Recommendations\n\n")
        f.write("1. **Replace 'src.' prefix imports** with relative imports\n")
        f.write("2. **Convert wildcard imports** to explicit imports\n")
        f.write("3. **Fix import ordering** according to PEP 8\n")
        f.write("4. **Resolve circular dependencies** by restructuring modules\n")
        f.write("5. **Use automated tools** (isort, black, autoflake) for consistency\n")

    print(f"📊 Import audit complete!")
    print(f"📄 Detailed report: {output_file}")
    print(f"📋 Summary report: {summary_file}")
    print(f"\n🔍 Found {total_issues} issues across {files_with_issues} files")


def main():
    """Main function to run the import audit."""
    print("🔍 Starting import pattern audit...")

    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk("."):
        # Skip common directories that shouldn't be analyzed
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
                "node_modules",
                "build",
                "dist",
                ".eggs",
                "*.egg-info",
            }
        ]

        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)

    print(f"📁 Found {len(python_files)} Python files to analyze")

    # Analyze each file
    results = []
    for i, filepath in enumerate(python_files, 1):
        if i % 50 == 0:
            print(f"   Processed {i}/{len(python_files)} files...")

        result = analyze_file(filepath)
        results.append(result)

    print(f"✅ Analyzed {len(python_files)} files")

    # Generate report
    output_file = "import_audit_report.json"
    generate_report(results, output_file)


if __name__ == "__main__":
    main()
