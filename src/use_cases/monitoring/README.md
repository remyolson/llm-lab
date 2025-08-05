# Monitoring System Infrastructure for LLM Lab

A comprehensive monitoring system that provides continuous performance tracking, regression detection, automated alerting, and dashboard capabilities for LLM benchmarking infrastructure.

## Features

- **📊 Database Storage**: SQLAlchemy-based models for storing benchmark results, metrics, and alerts
- **⏰ Scheduled Benchmarking**: APScheduler-powered automated benchmark execution
- **🔍 Regression Detection**: Statistical analysis for performance regression identification
- **🚨 Alert System**: Multi-channel alerting with email, Slack, and webhook notifications
- **🌐 REST API**: Comprehensive API for dashboard integration and external access
- **📈 Performance Analytics**: Trend analysis and statistical insights
- **🔧 Configurable Rules**: Flexible alert rules and detection thresholds

## Installation

### Required Dependencies

```bash
pip install sqlalchemy>=1.4.0 alembic>=1.8.0 fastapi>=0.100.0 uvicorn>=0.22.0
```

### Optional Dependencies

```bash
# For scheduling
pip install apscheduler>=3.10.0

# For statistical analysis
pip install scipy>=1.9.0 pandas>=1.5.0 numpy>=1.21.0

# For notifications
pip install aiohttp>=3.8.0

# For PostgreSQL support
pip install psycopg2-binary>=2.9.0

# For MySQL support
pip install pymysql>=1.0.0
```

## Quick Start

### Basic Setup

```python
from src.use_cases.monitoring import DatabaseManager, create_monitoring_api

# Initialize database
db_manager = DatabaseManager('sqlite:///monitoring.db')
db_manager.initialize_database()

# Create comprehensive monitoring API
api = create_monitoring_api(
    database_url='sqlite:///monitoring.db',
    enable_scheduler=True,
    enable_regression_detection=True,
    enable_alerting=True
)

# Start API server
api.run_server(host="0.0.0.0", port=8000)
```

### Complete System Setup

```python
import asyncio
from src.use_cases.monitoring import (
    DatabaseManager, BenchmarkScheduler, RegressionDetector, 
    AlertManager, MonitoringAPI
)

async def setup_monitoring_system():
    # Initialize database
    db_manager = DatabaseManager('postgresql://user:pass@localhost/monitoring')
    db_manager.initialize_database()
    
    # Set up scheduler with custom benchmark runner
    async def benchmark_runner(model_name, dataset_name, config, run_id):
        # Your benchmark implementation here
        return {'accuracy': 0.85, 'latency_ms': 250, 'cost': 0.001}
    
    scheduler = BenchmarkScheduler(db_manager, benchmark_runner)
    await scheduler.start()
    
    # Initialize regression detection
    regression_detector = RegressionDetector(db_manager)
    
    # Set up alerting with notifications
    alert_manager = AlertManager(db_manager)
    alert_manager.configure_notification_channel(
        NotificationChannel.EMAIL,
        {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-app-password',
            'from_email': 'monitoring@yourcompany.com',
            'to_emails': ['team@yourcompany.com']
        }
    )
    
    # Create comprehensive API
    api = MonitoringAPI(db_manager, scheduler, regression_detector, alert_manager)
    
    return api

# Run the setup
api = asyncio.run(setup_monitoring_system())
```

## Core Components

### 1. Database Management

Comprehensive SQLAlchemy models for storing monitoring data:

```python
from src.use_cases.monitoring import DatabaseManager

db_manager = DatabaseManager('sqlite:///monitoring.db')

# Create model metadata
model = db_manager.create_model({
    'name': 'gpt-4',
    'provider': 'openai',
    'cost_per_input_token': 0.00003,
    'cost_per_output_token': 0.00006,
    'max_tokens': 4096,
    'context_length': 8192,
    'model_type': 'chat'
})

# Create benchmark run
run = db_manager.create_benchmark_run({
    'model_id': model.id,
    'dataset_name': 'truthfulness',
    'run_config': {'sample_size': 1000},
    'status': 'running'
})

# Store performance metrics
metrics = db_manager.create_metrics_batch([
    {
        'run_id': run.id,
        'metric_type': 'accuracy',
        'metric_name': 'overall_accuracy',
        'value': 0.85,
        'unit': 'percentage'
    }
])
```

### 2. Scheduled Benchmarking

Automated benchmark execution with flexible scheduling:

```python
from src.use_cases.monitoring import BenchmarkScheduler
from src.use_cases.monitoring.scheduler import BenchmarkJobConfig, ScheduleType

scheduler = BenchmarkScheduler(db_manager, benchmark_runner)
await scheduler.start()

# Daily benchmarks
scheduler.add_daily_job(
    job_id='gpt4_daily_check',
    model_name='gpt-4',
    dataset_name='truthfulness',
    benchmark_config={'mode': 'full', 'sample_size': 1000},
    hour=2  # 2 AM UTC
)

# Interval-based monitoring
scheduler.add_interval_job(
    job_id='claude_hourly_monitor',
    model_name='claude-3-sonnet',
    dataset_name='reasoning',
    benchmark_config={'mode': 'quick', 'sample_size': 100},
    hours=1
)

# Create complete monitoring suite
created_jobs = scheduler.create_monitoring_suite(
    model_names=['gpt-4', 'claude-3-sonnet', 'llama-2-7b'],
    datasets=['truthfulness', 'reasoning', 'knowledge']
)

print(f"Created {len(created_jobs)} monitoring jobs")
```

### 3. Regression Detection

Statistical analysis for identifying performance regressions:

```python
from src.use_cases.monitoring import RegressionDetector
from src.use_cases.monitoring.regression_detector import RegressionConfig, DetectionMethod

detector = RegressionDetector(db_manager)

# Detect regressions with default configuration
results = detector.detect_regressions(model_id=1, days_back=30)

for result in results:
    if result.regression_detected:
        print(f"Regression detected in {result.metric_name}:")
        print(f"  Change: {result.change_percent*100:.1f}%")
        print(f"  Severity: {result.severity}")
        print(f"  Confidence: {result.confidence_score:.2f}")

# Custom regression detection
custom_config = RegressionConfig(
    method=DetectionMethod.STATISTICAL_TEST,
    metric_type='accuracy',
    metric_name='overall_accuracy',
    statistical_confidence=0.99,
    severity_thresholds={'critical': 0.05, 'warning': 0.02}
)

results = detector.detect_regressions(model_id=1, configs=[custom_config])

# Trend analysis
trends = detector.analyze_model_trends(model_id=1, days_back=30)
print(f"Analyzed {trends['metrics_analyzed']} metric trends")
```

### 4. Alert System

Comprehensive alerting with multiple notification channels:

```python
from src.use_cases.monitoring import AlertManager
from src.use_cases.monitoring.alerting import (
    AlertRule, AlertType, AlertSeverity, NotificationChannel
)

alert_manager = AlertManager(db_manager)

# Configure email notifications
alert_manager.configure_notification_channel(
    NotificationChannel.EMAIL,
    {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'monitoring@company.com',
        'password': 'app-password',
        'to_emails': ['team@company.com']
    }
)

# Configure Slack notifications
alert_manager.configure_notification_channel(
    NotificationChannel.SLACK,
    {'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'}
)

# Add custom alert rule
rule = AlertRule(
    rule_id='critical_accuracy_drop',
    name='Critical Accuracy Drop',
    description='Alert when accuracy drops significantly',
    alert_type=AlertType.PERFORMANCE_REGRESSION,
    severity=AlertSeverity.CRITICAL,
    conditions={'change_percent_threshold': 0.05, 'confidence_threshold': 0.8},
    notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
    cooldown_minutes=60
)

alert_manager.add_alert_rule(rule)

# Process regression results and generate alerts
alerts = await alert_manager.process_regression_results(regression_results)
print(f"Generated {len(alerts)} alerts")

# Manual alert management
alert_manager.acknowledge_alert('alert_123', 'john.doe@company.com')
alert_manager.resolve_alert('alert_123', 'jane.smith@company.com', 'Fixed model configuration')
```

### 5. REST API

Comprehensive API for dashboard integration:

```python
from src.use_cases.monitoring import MonitoringAPI

api = MonitoringAPI(db_manager, scheduler, detector, alert_manager)

# Start API server
await api.start_server(host="0.0.0.0", port=8000)

# API endpoints available:
# GET    /health                              - System health check
# GET    /models                              - List models
# POST   /models                              - Create model
# GET    /benchmark-runs                      - List benchmark runs
# GET    /benchmark-runs/{run_id}/metrics     - Get run metrics
# GET    /scheduler/jobs                      - List scheduled jobs
# POST   /scheduler/jobs                      - Create scheduled job
# POST   /regression-detection/analyze/{model_id} - Analyze regressions
# GET    /alerts                              - Get alerts
# PUT    /alerts/{alert_id}                   - Update alert
# GET    /dashboard/overview                  - Dashboard overview
```

## Configuration

### Database Configuration

```python
# SQLite (default for development)
db_manager = DatabaseManager('sqlite:///monitoring.db')

# PostgreSQL (recommended for production)
db_manager = DatabaseManager('postgresql://user:password@localhost:5432/monitoring')

# MySQL
db_manager = DatabaseManager('mysql+pymysql://user:password@localhost:3306/monitoring')
```

### Scheduler Configuration

```python
from src.use_cases.monitoring.scheduler import BenchmarkJobConfig, ScheduleType

# Cron-based scheduling
job_config = BenchmarkJobConfig(
    job_id='weekly_full_benchmark',
    model_name='gpt-4',
    dataset_name='comprehensive',
    benchmark_config={'mode': 'full', 'sample_size': 5000},
    schedule_type=ScheduleType.CRON,
    schedule_params={'day_of_week': 0, 'hour': 1, 'minute': 0},  # Monday 1 AM
    cooldown_minutes=60
)

# Interval-based scheduling
job_config = BenchmarkJobConfig(
    job_id='quick_health_check',
    model_name='claude-3-sonnet',
    dataset_name='health_check',
    benchmark_config={'mode': 'quick', 'sample_size': 10},
    schedule_type=ScheduleType.INTERVAL,
    schedule_params={'minutes': 15},  # Every 15 minutes
    max_instances=1,
    coalesce=True
)
```

### Detection Method Configuration

```python
from src.use_cases.monitoring.regression_detector import RegressionConfig, DetectionMethod

# Threshold-based detection
threshold_config = RegressionConfig(
    method=DetectionMethod.THRESHOLD_BASED,
    metric_type='latency',
    metric_name='avg_latency_ms',
    threshold_percent=0.2,  # 20% increase triggers alert
    severity_thresholds={'critical': 0.5, 'warning': 0.2}
)

# Statistical test detection
statistical_config = RegressionConfig(
    method=DetectionMethod.STATISTICAL_TEST,
    metric_type='accuracy',
    metric_name='overall_accuracy',
    statistical_confidence=0.95,
    min_data_points=10
)

# Change point detection
changepoint_config = RegressionConfig(
    method=DetectionMethod.CHANGE_POINT,
    metric_type='cost',
    metric_name='total_cost',
    window_size=7,
    baseline_days=14
)
```

## API Reference

### Models Endpoints

```bash
# List all models
GET /models?active_only=true

# Create new model
POST /models
{
  "name": "gpt-4-turbo",
  "provider": "openai",
  "cost_per_input_token": 0.00001,
  "cost_per_output_token": 0.00003
}

# Get specific model
GET /models/1

# Update model
PUT /models/1
{
  "active": false
}
```

### Benchmark Runs Endpoints

```bash
# List benchmark runs
GET /benchmark-runs?model_id=1&status=completed&limit=50

# Get specific run
GET /benchmark-runs/123

# Get run metrics
GET /benchmark-runs/123/metrics
```

### Scheduler Endpoints

```bash
# List scheduled jobs
GET /scheduler/jobs

# Create new job
POST /scheduler/jobs
{
  "job_id": "gpt4_daily_truthfulness",
  "model_name": "gpt-4",
  "dataset_name": "truthfulness",
  "benchmark_config": {"mode": "full", "sample_size": 1000},
  "schedule_type": "cron",
  "schedule_params": {"hour": 2, "minute": 0}
}

# Pause job
POST /scheduler/jobs/gpt4_daily_truthfulness/pause

# Resume job
POST /scheduler/jobs/gpt4_daily_truthfulness/resume

# Remove job
DELETE /scheduler/jobs/gpt4_daily_truthfulness
```

### Regression Detection Endpoints

```bash
# Analyze model for regressions
POST /regression-detection/analyze/1?days_back=30

# Get model trends
GET /regression-detection/trends/1?days_back=30

# Get regression summary
GET /regression-detection/summary?days_back=7
```

### Alert Endpoints

```bash
# Get alerts
GET /alerts?model_id=1&active_only=true

# Acknowledge alert
PUT /alerts/123
{
  "status": "acknowledged",
  "acknowledged_by": "john.doe@company.com"
}

# Resolve alert
PUT /alerts/123
{
  "status": "resolved",
  "resolved_by": "jane.smith@company.com",
  "resolution_notes": "Fixed model configuration issue"
}

# Get alert statistics
GET /alerts/statistics?days_back=7
```

### Dashboard Endpoints

```bash
# System overview
GET /dashboard/overview

# Model performance dashboard
GET /dashboard/model-performance/1?days_back=30
```

## Advanced Features

### Custom Benchmark Runner Integration

```python
async def custom_benchmark_runner(model_name, dataset_name, config, run_id):
    """Custom benchmark runner implementation."""
    
    # Initialize your benchmark framework
    from your_benchmark_framework import BenchmarkRunner
    
    runner = BenchmarkRunner(model_name)
    
    try:
        # Execute benchmark
        results = await runner.run_benchmark(
            dataset=dataset_name,
            **config
        )
        
        # Update database with progress
        db_manager.update_benchmark_run(run_id, {
            'status': 'completed',
            'accuracy': results.accuracy,
            'latency_ms': results.avg_latency,
            'total_cost': results.cost,
            'sample_count': results.samples_processed
        })
        
        # Store detailed metrics
        metrics_data = []
        for metric_name, value in results.detailed_metrics.items():
            metrics_data.append({
                'run_id': run_id,
                'metric_type': metric_name.split('_')[0],
                'metric_name': metric_name,
                'value': value,
                'timestamp': datetime.utcnow()
            })
        
        db_manager.create_metrics_batch(metrics_data)
        
        return {'success': True, 'results': results}
        
    except Exception as e:
        # Handle benchmark failure
        db_manager.update_benchmark_run(run_id, {
            'status': 'failed',
            'error_message': str(e)
        })
        return {'success': False, 'error': str(e)}

# Use custom runner with scheduler
scheduler = BenchmarkScheduler(db_manager, custom_benchmark_runner)
```

### Custom Alert Rules

```python
# Performance degradation alert
performance_rule = AlertRule(
    rule_id='performance_degradation',
    name='Model Performance Degradation',
    description='Detect when model performance degrades across multiple metrics',
    alert_type=AlertType.PERFORMANCE_REGRESSION,
    severity=AlertSeverity.WARNING,
    conditions={
        'metrics_affected_threshold': 2,  # At least 2 metrics affected
        'change_percent_threshold': 0.05,
        'confidence_threshold': 0.7
    },
    notification_channels=[NotificationChannel.SLACK],
    cooldown_minutes=120
)

# Cost anomaly alert
cost_rule = AlertRule(
    rule_id='cost_anomaly',
    name='Unexpected Cost Increase',
    description='Alert when costs increase unexpectedly',
    alert_type=AlertType.COST_ANOMALY,
    severity=AlertSeverity.CRITICAL,
    conditions={
        'cost_increase_threshold': 0.5,  # 50% cost increase
        'time_window_hours': 24
    },
    notification_channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK],
    cooldown_minutes=30,
    escalation_rules={
        'escalate_after_minutes': 60,
        'escalation_channels': [NotificationChannel.EMAIL]
    }
)
```

### Database Maintenance

```python
# Health check
health = db_manager.health_check()
print(f"Database status: {health['status']}")

# Clean up old data
deleted_counts = db_manager.cleanup_old_data(days_to_keep=90)
print(f"Cleaned up: {deleted_counts}")

# Database size information
size_info = db_manager.get_database_size()
print(f"Database size: {size_info.get('size_mb', 'Unknown')} MB")
```

## Running the Demo

```bash
# Run comprehensive demo
python examples/use_cases/monitoring_demo.py

# Start API server separately
python -c "
from src.use_cases.monitoring.api import create_monitoring_api
api = create_monitoring_api()
api.run_server(port=8000)
"

# Access API documentation
# http://localhost:8000/docs
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY examples/ examples/

EXPOSE 8000

CMD ["python", "-c", "from src.use_cases.monitoring.api import create_monitoring_api; api = create_monitoring_api(); api.run_server(host='0.0.0.0', port=8000)"]
```

### Environment Configuration

```bash
# Database
DATABASE_URL=postgresql://user:password@postgres:5432/monitoring

# Scheduler
SCHEDULER_TIMEZONE=UTC
MAX_CONCURRENT_JOBS=5

# Email notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=monitoring@company.com
SMTP_PASSWORD=app-password

# Slack notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

### Production Checklist

- [ ] Configure production database (PostgreSQL recommended)
- [ ] Set up proper logging and monitoring
- [ ] Configure authentication and authorization
- [ ] Set up backup and disaster recovery
- [ ] Configure load balancing for API
- [ ] Set up monitoring alerts for the monitoring system itself
- [ ] Configure rate limiting and security headers
- [ ] Set up proper environment variable management
- [ ] Configure log rotation and retention policies
- [ ] Set up health checks and readiness probes

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```python
   # Check database health
   health = db_manager.health_check()
   if health['status'] != 'healthy':
       print(f"Database issue: {health.get('error')}")
   ```

2. **Scheduler Not Starting**
   ```python
   # Check scheduler status
   if not scheduler._running:
       await scheduler.start()
   
   # Verify jobs are scheduled
   jobs = scheduler.list_jobs()
   print(f"Active jobs: {len(jobs)}")
   ```

3. **Alerts Not Firing**
   ```python
   # Check alert rules
   for rule_id, rule in alert_manager.alert_rules.items():
       print(f"Rule {rule_id}: enabled={rule.enabled}")
   
   # Test notification channels
   test_alert = Alert(...)
   await alert_manager._send_alert_notifications(test_alert, rule)
   ```

4. **API Performance Issues**
   ```python
   # Check database query performance
   import time
   start = time.time()
   runs = db_manager.list_benchmark_runs(limit=100)
   print(f"Query took {time.time() - start:.2f} seconds")
   
   # Add database indexes if needed
   # See models.py for existing indexes
   ```

## Contributing

1. **Adding New Detection Methods**
   - Extend `RegressionDetector` class
   - Add new `DetectionMethod` enum value
   - Implement detection logic
   - Add tests and documentation

2. **Adding New Notification Channels**
   - Extend `NotificationHandler` abstract class
   - Add new `NotificationChannel` enum value
   - Implement notification logic
   - Update `AlertManager` to register handler

3. **Adding New API Endpoints**
   - Add routes to `MonitoringAPI._setup_routes()`
   - Add Pydantic models for request/response
   - Add proper error handling
   - Update API documentation

## References

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [APScheduler Documentation](https://apscheduler.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Statistical Change Point Detection](https://en.wikipedia.org/wiki/Change_detection)
- [Time Series Analysis Methods](https://otexts.com/fpp3/)