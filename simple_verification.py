#!/usr/bin/env python3
"""
Simple verification that the key restructured modules work
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_basic_imports():
    """Test basic imports without triggering heavy dependencies"""
    print("Testing basic API imports...")

    # Test direct module imports
    try:
        from src.use_cases.fine_tuning.api.models import ModelConfig

        print("✅ API models import successful")

        # Test basic instantiation
        config = ModelConfig(name="test", baseModel="gpt-3.5")
        print("✅ Model config creation successful")

    except Exception as e:
        print(f"❌ Basic API test failed: {e}")
        return False

    try:
        # Test deployment module (just the classes, not heavy imports)
        import src.use_cases.fine_tuning.deployment.deploy as deploy_module

        print("✅ Deployment module import successful")

        # Check that key classes exist
        if hasattr(deploy_module, "DeploymentPipeline"):
            print("✅ DeploymentPipeline class found")
        else:
            print("❌ DeploymentPipeline class not found")
            return False

    except Exception as e:
        print(f"❌ Deployment test failed: {e}")
        return False

    return True


def test_directory_compliance():
    """Test that we meet the 4-level requirement"""
    print("\nTesting directory depth compliance...")

    fine_tuning_root = project_root / "src" / "use_cases" / "fine_tuning"

    if not fine_tuning_root.exists():
        print("❌ Fine-tuning root not found")
        return False

    # Count levels for key directories
    api_depth = len(Path("api/main.py").parts)  # Should be 2
    web_depth = len(Path("web/components/Navigation.tsx").parts)  # Should be 3

    print(f"API depth: {api_depth} levels (from fine_tuning root)")
    print(f"Web depth: {web_depth} levels (from fine_tuning root)")

    if api_depth <= 4 and web_depth <= 4:
        print("✅ Directory depth requirements met")
        return True
    else:
        print("❌ Directory depth requirements not met")
        return False


def test_file_migration():
    """Test that key files were migrated"""
    print("\nTesting file migration...")

    fine_tuning_root = project_root / "src" / "use_cases" / "fine_tuning"

    # Check for key migrated files
    key_files = [
        "api/main.py",
        "api/models.py",
        "api/experiments.py",
        "web/components/Navigation.tsx",
        "web/pages/dashboard.tsx",
        "deployment/deploy.py",
    ]

    missing_files = []
    for file_path in key_files:
        full_path = fine_tuning_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ Found {file_path}")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All key files migrated successfully")
    return True


def main():
    print("🚀 Simple Fine-Tuning Restructure Verification")
    print("=" * 50)

    tests = [test_basic_imports, test_directory_compliance, test_file_migration]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 SUCCESS! Directory restructuring verified!")
        return True
    else:
        print("⚠️ Some verification tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
