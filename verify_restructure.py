#!/usr/bin/env python3
"""
Verification script for fine-tuning directory restructuring
Tests that the new structure works correctly and meets requirements
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_directory_structure():
    """Test that the directory structure meets the 4-level requirement"""
    print("🔍 Testing directory structure...")

    fine_tuning_root = project_root / "src" / "use_cases" / "fine_tuning"
    if not fine_tuning_root.exists():
        return False, "Fine-tuning root directory not found"

    # Check API structure (should be max 4 levels from fine_tuning root)
    api_files = list(fine_tuning_root.glob("api/*.py"))
    if not api_files:
        return False, "No API files found"

    # Check web structure
    web_components = list(fine_tuning_root.glob("web/components/*.tsx"))
    if not web_components:
        return False, "No web components found"

    # Verify no excessive nesting (no more than 4 levels from fine_tuning root)
    max_depth = 0
    for py_file in fine_tuning_root.rglob("*.py"):
        relative_path = py_file.relative_to(fine_tuning_root)
        depth = len(relative_path.parts)
        max_depth = max(max_depth, depth)

    if max_depth > 4:
        return False, f"Maximum depth {max_depth} exceeds 4-level requirement"

    return True, f"✅ Directory structure valid (max depth: {max_depth} levels)"


def test_api_imports():
    """Test that API modules can be imported"""
    print("🔍 Testing API imports...")

    try:
        # Test models import
        from src.use_cases.fine_tuning.api.models import DatasetConfig, ModelConfig

        print("  ✅ API models import successful")

        # Test that we can create model instances
        model_config = ModelConfig(name="test-model", baseModel="gpt-3.5-turbo")
        dataset_config = DatasetConfig(name="test-dataset", path="/tmp/test.jsonl")
        print("  ✅ Model instantiation successful")

        return True, "API imports working correctly"

    except ImportError as e:
        return False, f"API import failed: {e}"
    except Exception as e:
        return False, f"API functionality failed: {e}"


def test_deployment_imports():
    """Test that deployment modules can be imported"""
    print("🔍 Testing deployment imports...")

    try:
        from src.use_cases.fine_tuning.deployment.deploy import (
            DeploymentPipeline,
            DeploymentProvider,
        )

        print("  ✅ Deployment imports successful")

        # Test enum access
        providers = list(DeploymentProvider)
        if len(providers) < 3:
            return False, "Not enough deployment providers found"
        print(f"  ✅ Found {len(providers)} deployment providers")

        return True, "Deployment imports working correctly"

    except ImportError as e:
        return False, f"Deployment import failed: {e}"
    except Exception as e:
        return False, f"Deployment functionality failed: {e}"


def test_api_structure():
    """Test that the API structure has the right endpoints"""
    print("🔍 Testing API structure...")

    try:
        from src.use_cases.fine_tuning.api import ab_testing, datasets, deployments, experiments

        # Check that modules have expected attributes
        required_modules = {
            "experiments": ["router", "create_experiment", "list_experiments"],
            "datasets": ["router", "create_dataset", "list_datasets"],
            "deployments": ["router", "create_deployment", "list_deployments"],
            "ab_testing": ["router", "create_ab_test", "list_ab_tests"],
        }

        for module_name, expected_attrs in required_modules.items():
            module = locals()[module_name]
            for attr in expected_attrs:
                if not hasattr(module, attr):
                    return False, f"Module {module_name} missing {attr}"

        print("  ✅ All API modules have expected structure")
        return True, "API structure is correct"

    except ImportError as e:
        return False, f"API structure test failed: {e}"
    except Exception as e:
        return False, f"API structure verification failed: {e}"


def test_web_structure():
    """Test that web files exist in the right locations"""
    print("🔍 Testing web structure...")

    fine_tuning_root = project_root / "src" / "use_cases" / "fine_tuning"
    web_root = fine_tuning_root / "web"

    if not web_root.exists():
        return False, "Web directory not found"

    # Check required directories
    required_dirs = ["components", "pages", "hooks", "types"]
    for dir_name in required_dirs:
        if not (web_root / dir_name).exists():
            return False, f"Web directory {dir_name} not found"

    # Check that we have React components
    components = list((web_root / "components").glob("*.tsx"))
    if len(components) < 3:
        return False, f"Not enough React components found (got {len(components)})"

    # Check that we have pages
    pages = list((web_root / "pages").glob("*.tsx"))
    if len(pages) < 5:
        return False, f"Not enough pages found (got {len(pages)})"

    print(f"  ✅ Found {len(components)} components and {len(pages)} pages")
    return True, "Web structure is correct"


def test_migration_completeness():
    """Test that migration was complete"""
    print("🔍 Testing migration completeness...")

    # Count files in old location
    old_studio_path = project_root / "src" / "use_cases" / "fine_tuning_studio"
    if not old_studio_path.exists():
        return True, "✅ Old fine_tuning_studio directory removed (migration complete)"

    # Count Python files in old location
    old_py_files = list(old_studio_path.rglob("*.py"))
    old_tsx_files = list(old_studio_path.rglob("*.tsx"))

    if len(old_py_files) + len(old_tsx_files) == 0:
        return True, "✅ No code files remain in old location"

    return (
        False,
        f"Migration incomplete: {len(old_py_files)} .py and {len(old_tsx_files)} .tsx files remain",
    )


def run_all_tests():
    """Run all verification tests"""
    print("🚀 Starting Fine-Tuning Directory Restructure Verification")
    print("=" * 60)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("API Imports", test_api_imports),
        ("Deployment Imports", test_deployment_imports),
        ("API Structure", test_api_structure),
        ("Web Structure", test_web_structure),
        ("Migration Completeness", test_migration_completeness),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success, message = test_func()
            results.append((test_name, success, message))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {message}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ ERROR: {test_name} - {e}")

        print()

    # Summary
    print("=" * 60)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"📊 SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Directory restructuring successful!")
        return True
    else:
        print("⚠️  Some tests failed. Review the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
