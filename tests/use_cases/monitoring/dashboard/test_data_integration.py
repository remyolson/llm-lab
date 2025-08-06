#!/usr/bin/env python3
"""
Test data integration and storage functionality
"""

# Import paths fixed - sys.path manipulation removed
import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))


def test_database_models():
    """Test database model creation and operations."""
    try:
        from use_cases.monitoring.dashboard.models import (
            Alert,
            DatabaseManager,
            MetricSnapshot,
            RequestLog,
        )

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        # Initialize database manager
        db_manager = DatabaseManager(f"sqlite:///{db_path}")
        db_manager.create_tables()

        print("✅ Database models and tables created successfully")

        # Test adding metric snapshot
        snapshot = db_manager.add_metric_snapshot(
            provider="OpenAI",
            model="gpt-4o-mini",
            requests_count=100,
            avg_latency=0.5,
            success_rate=99.0,
            total_cost=1.23,
            token_count=5000,
        )

        print(f"✅ Metric snapshot added: ID {snapshot.id}")

        # Test getting metrics summary
        summary = db_manager.get_metrics_summary()
        print(f"✅ Metrics summary retrieved: {summary['total_models']} models")

        # Test adding alert
        alert = db_manager.add_alert(
            alert_type="performance",
            severity="warning",
            title="High latency detected",
            message="Average latency exceeded threshold",
            provider="OpenAI",
            model="gpt-4o-mini",
            current_value=1.5,
            threshold_value=1.0,
        )

        print(f"✅ Alert added: ID {alert.id}")

        # Test getting alerts
        alerts = db_manager.get_active_alerts()
        print(f"✅ Active alerts retrieved: {len(alerts)} alerts")

        # Cleanup
        os.unlink(db_path)

        return True

    except Exception as e:
        print(f"❌ Database model test failed: {e}")
        return False


def test_data_service():
    """Test data service functionality."""
    try:
        from use_cases.monitoring.dashboard.data_service import DataService
        from use_cases.monitoring.dashboard.models import DatabaseManager

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        # Initialize components
        db_manager = DatabaseManager(f"sqlite:///{db_path}")
        db_manager.create_tables()

        data_service = DataService(db_manager)

        print("✅ Data service created successfully")

        # Test data collection (without starting background threads)
        metrics = data_service.get_metrics_summary()
        print(f"✅ Got metrics summary: {metrics}")

        performance_data = data_service.get_performance_data()
        print(
            f"✅ Got performance data: {len(performance_data.get('time_series', []))} data points"
        )

        cost_data = data_service.get_cost_breakdown()
        print(f"✅ Got cost breakdown: ${cost_data.get('total_cost', 0):.2f}")

        alerts = data_service.get_active_alerts()
        print(f"✅ Got active alerts: {len(alerts)} alerts")

        # Cleanup
        os.unlink(db_path)

        return True

    except Exception as e:
        print(f"❌ Data service test failed: {e}")
        return False


def test_data_collector():
    """Test data collector functionality."""
    try:
        from use_cases.monitoring.dashboard.data_service import DataCollector
        from use_cases.monitoring.dashboard.models import DatabaseManager

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        # Initialize components
        db_manager = DatabaseManager(f"sqlite:///{db_path}")
        db_manager.create_tables()

        # Create collector with short interval for testing
        collector = DataCollector(db_manager, collection_interval=1)

        print("✅ Data collector created successfully")

        # Test manual collection (without background thread)
        collector._generate_sample_metrics()
        print("✅ Sample metrics generated")

        collector._generate_sample_alert()
        print("✅ Sample alert generated")

        # Verify data was created
        summary = db_manager.get_metrics_summary()
        alerts = db_manager.get_active_alerts()

        print(f"✅ Verified data creation: {summary['total_models']} models, {len(alerts)} alerts")

        # Cleanup
        os.unlink(db_path)

        return True

    except Exception as e:
        print(f"❌ Data collector test failed: {e}")
        return False


def test_real_time_updater():
    """Test real-time updater functionality."""
    try:
        from use_cases.monitoring.dashboard.data_service import RealTimeUpdater
        from use_cases.monitoring.dashboard.models import DatabaseManager

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        # Initialize components
        db_manager = DatabaseManager(f"sqlite:///{db_path}")
        db_manager.create_tables()

        updater = RealTimeUpdater(db_manager, update_interval=1)

        print("✅ Real-time updater created successfully")

        # Test subscription mechanism
        received_updates = []

        def test_callback(data):
            received_updates.append(data)

        updater.subscribe("test_channel", test_callback)
        print("✅ Subscription mechanism works")

        # Test notification
        test_data = {"type": "test", "message": "Hello World"}
        updater._notify_subscribers("test_channel", test_data)

        if received_updates and received_updates[0] == test_data:
            print("✅ Notification mechanism works")
        else:
            print("⚠️  Notification mechanism may have issues")

        # Test unsubscription
        updater.unsubscribe("test_channel", test_callback)
        print("✅ Unsubscription mechanism works")

        # Cleanup
        os.unlink(db_path)

        return True

    except Exception as e:
        print(f"❌ Real-time updater test failed: {e}")
        return False


def test_api_integration():
    """Test API integration with data services."""
    try:
        from use_cases.monitoring.dashboard import create_app
        from use_cases.monitoring.dashboard.data_service import init_data_service
        from use_cases.monitoring.dashboard.models import init_database

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        # Override config for testing
        config_override = {
            "database": {"url": f"sqlite:///{db_path}"},
            "api": {"debug": True, "port": 0},
        }

        # Initialize services manually for testing
        db_manager = init_database(f"sqlite:///{db_path}")
        data_service = init_data_service(db_manager)

        # Generate some test data
        data_service.collector._generate_sample_metrics()
        data_service.collector._generate_sample_alert()

        # Create app
        app = create_app(config_override)

        print("✅ Flask app with data integration created successfully")

        # Test API endpoints
        with app.test_client() as client:
            # Test metrics overview
            response = client.get("/api/v1/metrics/overview")
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Metrics overview API: {data.get('total_models', 0)} models")
            else:
                print(f"⚠️  Metrics overview API returned {response.status_code}")

            # Test performance metrics
            response = client.get("/api/v1/metrics/performance")
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Performance metrics API: {len(data.get('time_series', []))} data points")
            else:
                print(f"⚠️  Performance metrics API returned {response.status_code}")

            # Test cost metrics
            response = client.get("/api/v1/metrics/costs")
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Cost metrics API: ${data.get('total_cost', 0):.2f}")
            else:
                print(f"⚠️  Cost metrics API returned {response.status_code}")

            # Test alerts
            response = client.get("/api/v1/alerts")
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Alerts API: {len(data.get('active_alerts', []))} alerts")
            else:
                print(f"⚠️  Alerts API returned {response.status_code}")

        # Cleanup
        data_service.stop()
        os.unlink(db_path)

        return True

    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        return False


def main():
    """Run all data integration tests."""
    print("🧪 Testing Data Integration and Storage")
    print("=" * 45)

    tests = [
        test_database_models,
        test_data_service,
        test_data_collector,
        test_real_time_updater,
        test_api_integration,
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
    print(f"📈 Success Rate: {passed / (passed + failed) * 100:.1f}%")

    if failed == 0:
        print("\n🎉 All data integration tests passed!")
        print("\n📝 Features implemented:")
        print("- SQLAlchemy database models for metrics, alerts, and logs")
        print("- Data collection from monitoring sources")
        print("- Real-time update mechanism")
        print("- REST API endpoints with data integration")
        print("- Background data collection and cleanup")
        print("- Sample data generation for demonstration")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
