#!/usr/bin/env python3
"""Test script to verify project bootstrap was successful."""

# Import paths fixed - sys.path manipulation removed
import importlib
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_directories():
    """Verify all required directories exist."""
    print("Checking directories...")
    directories = [
        "llm_providers",
        "evaluation",
        "benchmarks",
        "benchmarks/truthfulness",
        "results",
    ]

    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Missing directory: {directory}")
            return False
    return True


def check_init_files():
    """Check all __init__.py files are present."""
    print("\nChecking __init__.py files...")
    init_files = [
        "llm_providers/__init__.py",
        "evaluation/__init__.py",
        "benchmarks/__init__.py",
        "benchmarks/truthfulness/__init__.py",
    ]

    for init_file in init_files:
        if os.path.exists(init_file):
            print(f"✓ __init__.py exists: {init_file}")
        else:
            print(f"✗ Missing __init__.py: {init_file}")
            return False
    return True


def check_config_import():
    """Test that config.py can be imported."""
    print("\nChecking config module...")
    try:
        from src import config

        print("✓ config.py can be imported")

        # Check for required constants
        constants = ["DEFAULT_MODEL", "OUTPUT_DIR", "BENCHMARK_NAME"]
        for const in constants:
            if hasattr(config, const):
                print(f"✓ Constant exists: {const} = {getattr(config, const)}")
            else:
                print(f"✗ Missing constant: {const}")
                return False

        # Check for get_provider_config function
        if hasattr(config, "get_provider_config"):
            print("✓ Function exists: get_provider_config")
        else:
            print("✗ Missing function: get_provider_config")
            return False

        return True
    except ImportError as e:
        print(f"✗ Cannot import config.py: {e}")
        return False


def check_env_example():
    """Validate .env.example exists with correct placeholder."""
    print("\nChecking .env.example...")
    if os.path.exists(".env.example"):
        print("✓ .env.example exists")
        with open(".env.example") as f:
            content = f.read()
            if "GOOGLE_API_KEY" in content:
                print("✓ .env.example contains GOOGLE_API_KEY placeholder")
                return True
            else:
                print("✗ .env.example missing GOOGLE_API_KEY placeholder")
                return False
    else:
        print("✗ .env.example not found")
        return False


def check_requirements():
    """Ensure requirements.txt contains all required packages."""
    print("\nChecking requirements.txt...")
    required_packages = ["google-generativeai", "python-dotenv", "click"]

    if os.path.exists("requirements.txt"):
        print("✓ requirements.txt exists")
        with open("requirements.txt") as f:
            content = f.read()
            for package in required_packages:
                if package in content:
                    print(f"✓ Package found: {package}")
                else:
                    print(f"✗ Missing package: {package}")
                    return False
        return True
    else:
        print("✗ requirements.txt not found")
        return False


def check_package_imports():
    """Attempt to import from each package directory."""
    print("\nChecking package imports...")
    packages = ["llm_providers", "evaluation", "benchmarks", "benchmarks.truthfulness"]

    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ Can import: {package}")
        except ImportError as e:
            print(f"✗ Cannot import {package}: {e}")
            return False
    return True


def check_other_files():
    """Check for other important files."""
    print("\nChecking other files...")
    files = [
        (".gitignore", "Git ignore file"),
        ("README.md", "Project documentation"),
        ("venv", "Virtual environment directory"),
    ]

    for filename, description in files:
        if os.path.exists(filename):
            print(f"✓ {description} exists: {filename}")
        else:
            print(f"✗ {description} missing: {filename}")
            # Note: venv is not critical for the test
            if filename != "venv":
                return False
    return True


def main():
    """Run all bootstrap tests."""
    print("=" * 50)
    print("PROJECT BOOTSTRAP VERIFICATION")
    print("=" * 50)

    all_passed = True

    # Run all checks
    checks = [
        check_directories,
        check_init_files,
        check_config_import,
        check_env_example,
        check_requirements,
        check_package_imports,
        check_other_files,
    ]

    for check in checks:
        if not check():
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ ALL CHECKS PASSED! Project bootstrap successful.")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
