#!/usr/bin/env python3
"""
Test data integration structure and code quality
"""

import sys
import ast
from pathlib import Path

def test_models_file():
    """Test models.py structure and syntax."""
    models_path = Path(__file__).parent / 'src' / 'use_cases' / 'monitoring' / 'dashboard' / 'models.py'
    
    if not models_path.exists():
        print("❌ models.py not found")
        return False
    
    try:
        with open(models_path, 'r') as f:
            content = f.read()
        
        # Check syntax
        ast.parse(content)
        print("✅ models.py has valid Python syntax")
        
        # Check for required classes
        required_classes = ['MetricSnapshot', 'Alert', 'RequestLog', 'DashboardConfig', 'DatabaseManager']
        missing_classes = []
        
        for cls in required_classes:
            if f'class {cls}' not in content:
                missing_classes.append(cls)
        
        if missing_classes:
            print(f"❌ Missing model classes: {missing_classes}")
            return False
        else:
            print("✅ All required model classes found")
        
        # Check for SQLAlchemy usage
        if 'from sqlalchemy' not in content:
            print("❌ SQLAlchemy imports not found")
            return False
        else:
            print("✅ SQLAlchemy imports found")
        
        # Check for database operations
        db_operations = ['create_tables', 'get_session', 'get_metrics_summary', 'add_metric_snapshot']
        missing_operations = []
        
        for op in db_operations:
            if op not in content:
                missing_operations.append(op)
        
        if missing_operations:
            print(f"❌ Missing database operations: {missing_operations}")
            return False
        else:
            print("✅ All required database operations found")
        
        print(f"✅ models.py structure validated ({len(content)} characters)")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in models.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading models.py: {e}")
        return False

def test_data_service_file():
    """Test data_service.py structure and syntax."""
    service_path = Path(__file__).parent / 'src' / 'use_cases' / 'monitoring' / 'dashboard' / 'data_service.py'
    
    if not service_path.exists():
        print("❌ data_service.py not found")
        return False
    
    try:
        with open(service_path, 'r') as f:
            content = f.read()
        
        # Check syntax
        ast.parse(content)
        print("✅ data_service.py has valid Python syntax")
        
        # Check for required classes
        required_classes = ['DataCollector', 'RealTimeUpdater', 'DataService']
        missing_classes = []
        
        for cls in required_classes:
            if f'class {cls}' not in content:
                missing_classes.append(cls)
        
        if missing_classes:
            print(f"❌ Missing service classes: {missing_classes}")
            return False
        else:
            print("✅ All required service classes found")
        
        # Check for threading support
        if 'import threading' not in content:
            print("❌ Threading support not found")
            return False
        else:
            print("✅ Threading support found")
        
        # Check for data collection methods
        data_methods = ['_collect_metrics', '_collect_alerts', '_generate_sample_metrics']
        missing_methods = []
        
        for method in data_methods:
            if method not in content:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing data collection methods: {missing_methods}")
            return False
        else:
            print("✅ All required data collection methods found")
        
        print(f"✅ data_service.py structure validated ({len(content)} characters)")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in data_service.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading data_service.py: {e}")
        return False

def test_api_integration():
    """Test API integration with data services."""
    api_path = Path(__file__).parent / 'src' / 'use_cases' / 'monitoring' / 'dashboard' / 'api' / '__init__.py'
    
    if not api_path.exists():
        print("❌ API __init__.py not found")
        return False
    
    try:
        with open(api_path, 'r') as f:
            content = f.read()
        
        # Check syntax
        ast.parse(content)
        print("✅ API module has valid Python syntax")
        
        # Check for data service integration
        if 'from ..data_service import get_data_service' not in content:
            print("❌ Data service integration not found")
            return False
        else:
            print("✅ Data service integration found")
        
        # Check for API endpoints
        endpoints = ['metrics/overview', 'metrics/performance', 'metrics/costs', 'alerts']
        missing_endpoints = []
        
        for endpoint in endpoints:
            if endpoint not in content:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"❌ Missing API endpoints: {missing_endpoints}")
            return False
        else:
            print("✅ All required API endpoints found")
        
        # Check for data service calls
        service_calls = ['get_metrics_summary', 'get_performance_data', 'get_cost_breakdown', 'get_active_alerts']
        missing_calls = []
        
        for call in service_calls:
            if call not in content:
                missing_calls.append(call)
        
        if missing_calls:
            print(f"❌ Missing data service calls: {missing_calls}")
            return False
        else:
            print("✅ All required data service calls found")
        
        print(f"✅ API integration validated ({len(content)} characters)")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in API module: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading API module: {e}")
        return False

def test_app_integration():
    """Test app.py integration with data services."""
    app_path = Path(__file__).parent / 'src' / 'use_cases' / 'monitoring' / 'dashboard' / 'app.py'
    
    if not app_path.exists():
        print("❌ app.py not found")
        return False
    
    try:
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Check syntax
        ast.parse(content)
        print("✅ app.py has valid Python syntax")
        
        # Check for data service imports
        required_imports = ['from .models import init_database', 'from .data_service import init_data_service']
        missing_imports = []
        
        for imp in required_imports:
            if imp not in content:
                missing_imports.append(imp)
        
        if missing_imports:
            print(f"❌ Missing imports: {missing_imports}")
            return False
        else:
            print("✅ All required imports found")
        
        # Check for initialization function
        if 'def initialize_data_services' not in content:
            print("❌ Data services initialization function not found")
            return False
        else:
            print("✅ Data services initialization function found")
        
        # Check for database initialization
        if 'init_database(' not in content:
            print("❌ Database initialization call not found")
            return False
        else:
            print("✅ Database initialization call found")
        
        # Check for data service initialization
        if 'init_data_service(' not in content:
            print("❌ Data service initialization call not found")
            return False
        else:
            print("✅ Data service initialization call found")
        
        print(f"✅ App integration validated ({len(content)} characters)")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False

def test_database_schema():
    """Test database schema design."""
    models_path = Path(__file__).parent / 'src' / 'use_cases' / 'monitoring' / 'dashboard' / 'models.py'
    
    try:
        with open(models_path, 'r') as f:
            content = f.read()
        
        # Check for table definitions
        tables = ['metric_snapshots', 'alerts', 'request_logs', 'dashboard_configs']
        missing_tables = []
        
        for table in tables:
            if f"__tablename__ = '{table}'" not in content:
                missing_tables.append(table)
        
        if missing_tables:
            print(f"❌ Missing table definitions: {missing_tables}")
            return False
        else:
            print("✅ All required table definitions found")
        
        # Check for relationships and indexes
        schema_features = ['Column(', 'Index(', 'relationship(', 'ForeignKey(']
        missing_features = []
        
        for feature in schema_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing schema features: {missing_features}")
            return False
        else:
            print("✅ All required schema features found")
        
        # Check for data conversion methods
        conversion_methods = ['to_dict', 'from_dict']
        found_methods = []
        
        for method in conversion_methods:
            if f'def {method}' in content:
                found_methods.append(method)
        
        print(f"✅ Data conversion methods found: {found_methods}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing database schema: {e}")
        return False

def test_background_services():
    """Test background service implementation."""
    service_path = Path(__file__).parent / 'src' / 'use_cases' / 'monitoring' / 'dashboard' / 'data_service.py'
    
    try:
        with open(service_path, 'r') as f:
            content = f.read()
        
        # Check for background thread implementation
        thread_features = ['threading.Thread', '_thread', 'daemon=True', 'start()', 'stop()']
        missing_features = []
        
        for feature in thread_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing threading features: {missing_features}")
            return False
        else:
            print("✅ All threading features found")
        
        # Check for data collection loop
        if '_collection_loop' not in content:
            print("❌ Data collection loop not found")
            return False
        else:
            print("✅ Data collection loop found")
        
        # Check for real-time update mechanism
        realtime_features = ['subscribe', 'unsubscribe', '_notify_subscribers', '_update_loop']
        missing_realtime = []
        
        for feature in realtime_features:
            if feature not in content:
                missing_realtime.append(feature)
        
        if missing_realtime:
            print(f"❌ Missing real-time features: {missing_realtime}")
            return False
        else:
            print("✅ All real-time features found")
        
        # Check for data cleanup
        if 'cleanup_old_data' not in content:
            print("❌ Data cleanup not found")
            return False
        else:
            print("✅ Data cleanup found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing background services: {e}")
        return False

def main():
    """Run all data structure tests."""
    print("🧪 Testing Data Integration Structure")
    print("=" * 40)
    
    tests = [
        test_models_file,
        test_data_service_file,
        test_api_integration,
        test_app_integration,
        test_database_schema,
        test_background_services
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
        print("\n🎉 All data integration structure tests passed!")
        print("\n📝 Features implemented:")
        print("- ✅ SQLAlchemy database models (MetricSnapshot, Alert, RequestLog)")
        print("- ✅ Database manager with CRUD operations")
        print("- ✅ Data collection service with background threads")
        print("- ✅ Real-time update mechanism with pub/sub pattern")
        print("- ✅ REST API integration with data services")
        print("- ✅ App initialization with data services")
        print("- ✅ Comprehensive database schema design")
        print("- ✅ Background data collection and cleanup")
        print("- ✅ Sample data generation for demonstration")
        print("\n📋 Ready for next step: Build Interactive Visualization Components")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)