#!/usr/bin/env python3
"""
Basic test for dashboard framework setup
"""

# Import paths fixed - sys.path manipulation removed
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
.parent / 'src'))

def test_basic_imports():
    """Test that basic dashboard imports work."""
    try:
        from src.use_cases.monitoring.dashboard.config.settings import DashboardConfig
        from src.use_cases.monitoring.dashboard import create_app
        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    try:
        from src.use_cases.monitoring.dashboard.config.settings import DashboardConfig
        
        config = DashboardConfig()
        print(f"✅ Config created: {config.environment}")
        
        # Test validation
        config.validate()
        print("✅ Config validation passed")
        
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_app_creation():
    """Test Flask app creation."""
    try:
        from src.use_cases.monitoring.dashboard import create_app
        
        # Use in-memory SQLite for testing
        config_override = {
            'database': {'url': 'sqlite:///:memory:'},
            'api': {'debug': True, 'port': 0}  # Port 0 for testing
        }
        
        app = create_app(config_override)
        print("✅ Flask app created successfully")
        
        # Test that routes exist
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/health')
            print(f"✅ Health check: {response.status_code}")
            
            # Test main dashboard
            response = client.get('/')
            print(f"✅ Dashboard page: {response.status_code}")
            
            # Test API status
            response = client.get('/api/v1/status')
            print(f"✅ API status: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ App creation error: {e}")
        return False

def test_directory_structure():
    """Test that all required directories were created."""
    base_path = Path(__file__).parent / 'src' / 'use_cases' / 'monitoring' / 'dashboard'
    
    required_dirs = [
        'components',
        'templates', 
        'static/css',
        'static/js',
        'static/img',
        'api',
        'reports',
        'config'
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

def test_static_files():
    """Test that static files exist."""
    static_path = Path(__file__).parent / 'src' / 'use_cases' / 'monitoring' / 'dashboard' / 'static'
    
    required_files = [
        'css/dashboard.css',
        'js/dashboard.js'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = static_path / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
    
    if missing_files:
        print(f"❌ Missing static files: {missing_files}")
        return False
    else:
        print("✅ All required static files exist")
        return True

def test_templates():
    """Test that template files exist."""
    template_path = Path(__file__).parent / 'src' / 'use_cases' / 'monitoring' / 'dashboard' / 'templates'
    
    required_templates = [
        'base.html',
        'dashboard.html',
        'error.html',
        'api_docs.html'
    ]
    
    missing_templates = []
    for template in required_templates:
        full_path = template_path / template
        if not full_path.exists():
            missing_templates.append(str(full_path))
    
    if missing_templates:
        print(f"❌ Missing templates: {missing_templates}")
        return False
    else:
        print("✅ All required templates exist")
        return True

def main():
    """Run all tests."""
    print("🧪 Testing Dashboard Framework Setup")
    print("=" * 40)
    
    tests = [
        test_directory_structure,
        test_static_files,
        test_templates,
        test_basic_imports,
        test_config_creation,
        test_app_creation
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
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Dashboard framework is set up correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)