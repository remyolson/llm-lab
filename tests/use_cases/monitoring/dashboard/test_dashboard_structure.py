#!/usr/bin/env python3
"""
Test dashboard structure and basic Python syntax
"""

import ast
import sys
from pathlib import Path


def test_directory_structure():
    """Test that all required directories were created."""
    base_path = Path(__file__).parent / "src" / "use_cases" / "monitoring" / "dashboard"

    required_dirs = [
        "components",
        "templates",
        "static/css",
        "static/js",
        "static/img",
        "api",
        "reports",
        "config",
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if not full_path.exists():
            missing_dirs.append(str(full_path))

    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
        return True


def test_python_files_syntax():
    """Test that Python files have valid syntax."""
    base_path = Path(__file__).parent / "src" / "use_cases" / "monitoring" / "dashboard"

    python_files = [
        "__init__.py",
        "app.py",
        "config/__init__.py",
        "config/settings.py",
        "api/__init__.py",
        "components/__init__.py",
        "run.py",
    ]

    syntax_errors = []
    for file_path in python_files:
        full_path = base_path / file_path
        if not full_path.exists():
            syntax_errors.append(f"File not found: {full_path}")
            continue

        try:
            with open(full_path) as f:
                source = f.read()
            ast.parse(source)
            print(f"✅ {file_path} - syntax OK")
        except SyntaxError as e:
            syntax_errors.append(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            syntax_errors.append(f"Error reading {file_path}: {e}")

    if syntax_errors:
        print("❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"   {error}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True


def test_static_files():
    """Test that static files exist and have content."""
    static_path = (
        Path(__file__).parent / "src" / "use_cases" / "monitoring" / "dashboard" / "static"
    )

    required_files = ["css/dashboard.css", "js/dashboard.js"]

    issues = []
    for file_path in required_files:
        full_path = static_path / file_path
        if not full_path.exists():
            issues.append(f"Missing: {full_path}")
        else:
            try:
                with open(full_path) as f:
                    content = f.read()
                if len(content.strip()) == 0:
                    issues.append(f"Empty: {full_path}")
                else:
                    print(f"✅ {file_path} - {len(content)} characters")
            except Exception as e:
                issues.append(f"Error reading {full_path}: {e}")

    if issues:
        print("❌ Static file issues:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ All required static files exist and have content")
        return True


def test_templates():
    """Test that template files exist and have content."""
    template_path = (
        Path(__file__).parent / "src" / "use_cases" / "monitoring" / "dashboard" / "templates"
    )

    required_templates = ["base.html", "dashboard.html", "error.html", "api_docs.html"]

    issues = []
    for template in required_templates:
        full_path = template_path / template
        if not full_path.exists():
            issues.append(f"Missing: {full_path}")
        else:
            try:
                with open(full_path) as f:
                    content = f.read()
                if len(content.strip()) == 0:
                    issues.append(f"Empty: {full_path}")
                elif "html" not in content.lower():
                    issues.append(f"Not HTML: {full_path}")
                else:
                    print(f"✅ {template} - {len(content)} characters")
            except Exception as e:
                issues.append(f"Error reading {full_path}: {e}")

    if issues:
        print("❌ Template issues:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ All required templates exist and appear to be HTML")
        return True


def test_requirements_file():
    """Test that dashboard requirements file exists."""
    req_path = Path(__file__).parent / "requirements-dashboard.txt"

    if not req_path.exists():
        print("❌ requirements-dashboard.txt not found")
        return False

    try:
        with open(req_path) as f:
            content = f.read()

        # Check for key dependencies
        key_deps = ["Flask", "SQLAlchemy", "socketio"]
        missing_deps = []

        for dep in key_deps:
            if dep.lower() not in content.lower():
                missing_deps.append(dep)

        if missing_deps:
            print(f"❌ Missing key dependencies in requirements: {missing_deps}")
            return False
        else:
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            print(f"✅ requirements-dashboard.txt exists with {len(lines)} dependencies")
            return True

    except Exception as e:
        print(f"❌ Error reading requirements file: {e}")
        return False


def test_configuration_structure():
    """Test that configuration has expected structure."""
    config_path = (
        Path(__file__).parent
        / "src"
        / "use_cases"
        / "monitoring"
        / "dashboard"
        / "config"
        / "settings.py"
    )

    if not config_path.exists():
        print("❌ Configuration file not found")
        return False

    try:
        with open(config_path) as f:
            content = f.read()

        # Check for key configuration classes
        expected_classes = ["DatabaseConfig", "APIConfig", "SecurityConfig", "DashboardConfig"]
        missing_classes = []

        for cls in expected_classes:
            if f"class {cls}" not in content:
                missing_classes.append(cls)

        if missing_classes:
            print(f"❌ Missing configuration classes: {missing_classes}")
            return False
        else:
            print("✅ All expected configuration classes found")
            return True

    except Exception as e:
        print(f"❌ Error reading configuration file: {e}")
        return False


def test_startup_script():
    """Test that startup script exists and is executable."""
    run_path = Path(__file__).parent / "src" / "use_cases" / "monitoring" / "dashboard" / "run.py"

    if not run_path.exists():
        print("❌ run.py not found")
        return False

    # Check if executable
    import stat

    file_stat = run_path.stat()
    if not (file_stat.st_mode & stat.S_IXUSR):
        print("⚠️  run.py is not executable (but exists)")
    else:
        print("✅ run.py is executable")

    # Check content
    try:
        with open(run_path) as f:
            content = f.read()

        if "def main()" not in content:
            print("❌ run.py missing main() function")
            return False

        if "argparse" not in content:
            print("❌ run.py missing argument parsing")
            return False

        print("✅ run.py has expected structure")
        return True

    except Exception as e:
        print(f"❌ Error reading run.py: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Dashboard Framework Structure")
    print("=" * 45)

    tests = [
        test_directory_structure,
        test_python_files_syntax,
        test_static_files,
        test_templates,
        test_requirements_file,
        test_configuration_structure,
        test_startup_script,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print(f"\n📋 Running {test.__name__}...")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            failed += 1

    print("\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed / (passed + failed) * 100:.1f}%")

    if failed == 0:
        print("\n🎉 All tests passed! Dashboard framework structure is correct.")
        print("\n📝 Next Steps:")
        print("1. Install dependencies: pip install -r requirements-dashboard.txt")
        print("2. Run dashboard: python src/use_cases/monitoring/dashboard/run.py")
        print("3. Open browser: http://localhost:8050")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
