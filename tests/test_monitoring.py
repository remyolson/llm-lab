"""
Comprehensive tests for monitoring and alerting functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import json
import sqlite3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.use_cases.monitoring_complete_demo import (
    MonitoringDatabase,
    MetricCollector,
    AlertManager,
    DashboardServer,
    MonitoringScheduler,
    ReportGenerator
)


class TestMonitoringDatabase:
    """Test monitoring database functionality"""
    
    @pytest.fixture
    def db(self, tmp_path):
        """Create test database"""
        db_path = tmp_path / "test_monitoring.db"
        return MonitoringDatabase(str(db_path))
    
    def test_database_creation(self, db):
        """Test database tables are created correctly"""
        # Check tables exist
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        expected_tables = {'metrics', 'alerts', 'alert_history', 'benchmarks'}
        assert expected_tables.issubset(tables)
        
        conn.close()
    
    def test_insert_metric(self, db):
        """Test inserting metrics"""
        db.insert_metric(
            model="gpt-4",
            latency=1.5,
            tokens_used=100,
            cost=0.01,
            error=False,
            metadata={"test": "data"}
        )
        
        metrics = db.get_metrics(model="gpt-4", hours=1)
        assert len(metrics) == 1
        assert metrics[0]['latency'] == 1.5
        assert metrics[0]['tokens_used'] == 100
    
    def test_insert_alert(self, db):
        """Test inserting alerts"""
        alert_id = db.insert_alert(
            alert_name="High Latency",
            severity="warning",
            message="Latency exceeds threshold",
            model="gpt-4",
            metadata={"threshold": 2.0}
        )
        
        assert alert_id is not None
        
        # Check alert history
        history = db.get_alert_history(hours=1)
        assert len(history) == 1
        assert history[0]['alert_name'] == "High Latency"
    
    def test_get_aggregated_metrics(self, db):
        """Test metric aggregation"""
        # Insert test metrics
        for i in range(5):
            db.insert_metric(
                model="gpt-4",
                latency=1.0 + i * 0.1,
                tokens_used=100,
                cost=0.01,
                error=False
            )
        
        stats = db.get_aggregated_metrics(model="gpt-4", hours=1)
        
        assert stats['avg_latency'] == pytest.approx(1.2, 0.01)
        assert stats['p95_latency'] >= stats['avg_latency']
        assert stats['total_requests'] == 5
        assert stats['error_rate'] == 0.0
    
    def test_update_alert_status(self, db):
        """Test updating alert status"""
        alert_id = db.insert_alert(
            alert_name="Test Alert",
            severity="info",
            message="Test",
            model="gpt-4"
        )
        
        db.update_alert_status(alert_id, "resolved")
        
        # Verify status updated
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status FROM alerts WHERE id = ?",
            (alert_id,)
        )
        status = cursor.fetchone()[0]
        conn.close()
        
        assert status == "resolved"


class TestMetricCollector:
    """Test metric collection functionality"""
    
    @pytest.fixture
    def collector(self, tmp_path):
        """Create MetricCollector instance"""
        db_path = tmp_path / "test_monitoring.db"
        db = MonitoringDatabase(str(db_path))
        config = {
            'monitoring': {
                'models': [
                    {
                        'provider': 'openai',
                        'model': 'gpt-4',
                        'sla_target': 2.0
                    }
                ]
            }
        }
        return MetricCollector(config, db)
    
    @patch('examples.use_cases.monitoring_complete_demo.MetricCollector._run_test_prompt')
    def test_collect_performance_metrics(self, mock_run, collector):
        """Test performance metric collection"""
        mock_run.return_value = {
            'latency': 1.5,
            'tokens': 50,
            'cost': 0.005,
            'error': False
        }
        
        metrics = collector.collect_performance_metrics()
        
        assert len(metrics) == 1
        assert metrics[0]['model'] == 'gpt-4'
        assert metrics[0]['latency'] == 1.5
        assert not metrics[0]['sla_breach']
    
    @patch('time.time')
    @patch('examples.use_cases.monitoring_complete_demo.MetricCollector._call_model')
    def test_run_test_prompt(self, mock_call, mock_time, collector):
        """Test running test prompts"""
        # Mock timing
        mock_time.side_effect = [0, 1.5]  # 1.5 second execution
        
        # Mock model response
        mock_call.return_value = ("Test response", 50)
        
        result = collector._run_test_prompt("openai", "gpt-4", "Test prompt")
        
        assert result['latency'] == 1.5
        assert result['tokens'] == 50
        assert not result['error']
    
    def test_collect_cost_metrics(self, collector):
        """Test cost metric collection"""
        # Insert test data
        for i in range(10):
            collector.db.insert_metric(
                model="gpt-4",
                latency=1.0,
                tokens_used=100,
                cost=0.01,
                error=False
            )
        
        cost_data = collector.collect_cost_metrics(hours=1)
        
        assert 'total_cost' in cost_data
        assert 'by_model' in cost_data
        assert cost_data['total_cost'] == 0.1  # 10 * 0.01


class TestAlertManager:
    """Test alert management functionality"""
    
    @pytest.fixture
    def alert_manager(self, tmp_path):
        """Create AlertManager instance"""
        db_path = tmp_path / "test_monitoring.db"
        db = MonitoringDatabase(str(db_path))
        config = {
            'alerts': {
                'channels': [
                    {
                        'type': 'email',
                        'recipients': ['test@example.com'],
                        'severity': ['critical', 'warning']
                    }
                ],
                'rules': [
                    {
                        'name': 'High Latency',
                        'condition': 'avg_latency > sla_target * 1.5',
                        'severity': 'warning',
                        'cooldown': 3600
                    }
                ]
            }
        }
        return AlertManager(config, db)
    
    def test_check_alerts(self, alert_manager):
        """Test alert checking"""
        metrics = {
            'gpt-4': {
                'avg_latency': 3.0,
                'sla_target': 2.0,
                'error_rate': 0.0
            }
        }
        
        triggered = alert_manager.check_alerts(metrics)
        
        assert len(triggered) == 1
        assert triggered[0]['rule'] == 'High Latency'
        assert triggered[0]['severity'] == 'warning'
    
    def test_cooldown_check(self, alert_manager):
        """Test alert cooldown functionality"""
        # First alert should trigger
        assert alert_manager._check_cooldown('High Latency', 'gpt-4')
        
        # Second alert within cooldown should not trigger
        assert not alert_manager._check_cooldown('High Latency', 'gpt-4')
    
    @patch('examples.use_cases.monitoring_complete_demo.AlertManager._send_email')
    def test_send_alert(self, mock_email, alert_manager):
        """Test sending alerts"""
        alert = {
            'rule': 'Test Alert',
            'severity': 'warning',
            'model': 'gpt-4',
            'message': 'Test message',
            'value': 3.0,
            'threshold': 2.0
        }
        
        alert_manager.send_alert(alert)
        
        # Email should be sent for warning severity
        mock_email.assert_called_once()
    
    def test_evaluate_condition(self, alert_manager):
        """Test condition evaluation"""
        context = {
            'avg_latency': 3.0,
            'sla_target': 2.0,
            'error_rate': 0.1
        }
        
        # Test various conditions
        assert alert_manager._evaluate_condition(
            'avg_latency > sla_target', context
        )
        assert alert_manager._evaluate_condition(
            'error_rate < 0.5', context
        )
        assert not alert_manager._evaluate_condition(
            'avg_latency < 1.0', context
        )


class TestDashboardServer:
    """Test dashboard server functionality"""
    
    @pytest.fixture
    def dashboard(self, tmp_path):
        """Create DashboardServer instance"""
        db_path = tmp_path / "test_monitoring.db"
        db = MonitoringDatabase(str(db_path))
        config = {
            'dashboard': {
                'refresh_intervals': {'overview': 30}
            }
        }
        
        with patch('examples.use_cases.monitoring_complete_demo.Flask'):
            return DashboardServer(config, db)
    
    def test_get_metrics_endpoint(self, dashboard):
        """Test metrics API endpoint"""
        # Insert test data
        dashboard.db.insert_metric(
            model="gpt-4",
            latency=1.5,
            tokens_used=100,
            cost=0.01,
            error=False
        )
        
        # Mock request context
        with dashboard.app.app_context():
            with dashboard.app.test_request_context('/?model=gpt-4&hours=1'):
                response = dashboard.get_metrics()
                data = json.loads(response.get_data(as_text=True))
                
                assert 'metrics' in data
                assert len(data['metrics']) > 0


class TestMonitoringScheduler:
    """Test scheduled monitoring tasks"""
    
    @pytest.fixture
    def scheduler(self, tmp_path):
        """Create MonitoringScheduler instance"""
        db_path = tmp_path / "test_monitoring.db"
        db = MonitoringDatabase(str(db_path))
        config = {
            'monitoring': {
                'schedule': {
                    'performance_checks': {
                        'frequency': '*/15 minutes'
                    }
                }
            }
        }
        
        collector = MetricCollector(config, db)
        alert_manager = AlertManager({'alerts': {'channels': [], 'rules': []}}, db)
        
        return MonitoringScheduler(config, collector, alert_manager, db)
    
    @patch('examples.use_cases.monitoring_complete_demo.MetricCollector.collect_performance_metrics')
    def test_run_performance_check(self, mock_collect, scheduler):
        """Test performance check task"""
        mock_collect.return_value = [
            {
                'model': 'gpt-4',
                'latency': 1.5,
                'sla_breach': False
            }
        ]
        
        scheduler.run_performance_check()
        
        mock_collect.assert_called_once()
        
        # Check metrics were inserted
        metrics = scheduler.db.get_metrics(hours=1)
        assert len(metrics) > 0


class TestReportGenerator:
    """Test report generation functionality"""
    
    @pytest.fixture
    def report_generator(self, tmp_path):
        """Create ReportGenerator instance"""
        db_path = tmp_path / "test_monitoring.db"
        db = MonitoringDatabase(str(db_path))
        config = {
            'reporting': {
                'email_reports': {
                    'from_address': 'monitor@example.com'
                }
            }
        }
        return ReportGenerator(config, db)
    
    def test_generate_daily_summary(self, report_generator):
        """Test daily summary generation"""
        # Insert test data
        for i in range(24):  # 24 hours of data
            report_generator.db.insert_metric(
                model="gpt-4",
                latency=1.0 + (i % 4) * 0.1,
                tokens_used=100,
                cost=0.01,
                error=i % 10 == 0  # 10% errors
            )
        
        summary = report_generator.generate_daily_summary()
        
        assert 'executive_summary' in summary
        assert 'performance_trends' in summary
        assert 'cost_analysis' in summary
        assert summary['total_requests'] == 24
        assert summary['total_cost'] == 0.24
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_charts(self, mock_savefig, report_generator):
        """Test chart creation"""
        metrics = [
            {
                'timestamp': datetime.now() - timedelta(hours=i),
                'latency': 1.0 + (i % 3) * 0.1,
                'model': 'gpt-4'
            }
            for i in range(10)
        ]
        
        chart_paths = report_generator._create_charts(metrics)
        
        assert 'latency_trend' in chart_paths
        assert mock_savefig.called


class TestIntegration:
    """Test integration between monitoring components"""
    
    def test_full_monitoring_pipeline(self, tmp_path):
        """Test complete monitoring pipeline"""
        # Setup
        db_path = tmp_path / "test_monitoring.db"
        db = MonitoringDatabase(str(db_path))
        
        config = {
            'monitoring': {
                'models': [
                    {
                        'provider': 'openai',
                        'model': 'gpt-4',
                        'sla_target': 2.0
                    }
                ]
            },
            'alerts': {
                'channels': [],
                'rules': [
                    {
                        'name': 'High Latency',
                        'condition': 'avg_latency > 2.5',
                        'severity': 'warning',
                        'cooldown': 3600
                    }
                ]
            }
        }
        
        collector = MetricCollector(config, db)
        alert_manager = AlertManager(config, db)
        
        # Simulate metric collection
        with patch.object(collector, '_run_test_prompt') as mock_run:
            mock_run.return_value = {
                'latency': 3.0,  # Above SLA
                'tokens': 100,
                'cost': 0.01,
                'error': False
            }
            
            metrics = collector.collect_performance_metrics()
            
            # Store metrics
            for metric in metrics:
                db.insert_metric(
                    model=metric['model'],
                    latency=metric['latency'],
                    tokens_used=50,
                    cost=0.01,
                    error=False
                )
        
        # Check alerts
        agg_metrics = {
            'gpt-4': db.get_aggregated_metrics('gpt-4', 1)
        }
        
        triggered = alert_manager.check_alerts(agg_metrics)
        
        assert len(triggered) == 1
        assert triggered[0]['rule'] == 'High Latency'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])