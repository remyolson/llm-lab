#!/usr/bin/env python3
"""
Comprehensive Monitoring and Alerting Demo
==========================================

This example demonstrates all monitoring techniques from Use Case 8:
- Automated performance monitoring with schedules
- Real-time dashboards and visualizations  
- Multi-channel alerting (email, Slack, webhooks)
- SLA compliance tracking
- Predictive monitoring and anomaly detection
- Cost optimization monitoring

Usage:
    # Initialize monitoring system
    python monitoring_complete_demo.py --init-monitoring --database postgresql://localhost/monitoring
    
    # Start monitoring service
    python monitoring_complete_demo.py --start-monitoring --config monitoring_config.yaml
    
    # Create performance baseline
    python monitoring_complete_demo.py --create-baseline --iterations 10
    
    # Start dashboard
    python monitoring_complete_demo.py --start-dashboard --port 5000
    
    # Generate report
    python monitoring_complete_demo.py --generate-report --type weekly --output report.pdf
"""

import argparse
import json
import yaml
import asyncio
import sqlite3
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Import from the main library
from src.providers import get_provider
from src.utils import setup_logging


@dataclass
class MetricSnapshot:
    """Represents a single metric measurement."""
    timestamp: datetime
    model: str
    provider: str
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents a monitoring alert."""
    id: str
    name: str
    severity: str
    condition: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLATarget:
    """Service Level Agreement target."""
    metric: str
    target_value: float
    comparison: str  # 'less_than', 'greater_than'
    measurement_window: str  # 'minute', 'hour', 'day'


class MonitoringDatabase:
    """Handles database operations for monitoring data."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        # For demo, use SQLite
        self.conn = sqlite3.connect(':memory:' if 'memory' in self.db_url else 'monitoring.db')
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model TEXT,
                provider TEXT,
                metric_name TEXT,
                value REAL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                name TEXT,
                severity TEXT,
                condition TEXT,
                triggered_at DATETIME,
                resolved_at DATETIME,
                message TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS baselines (
                model TEXT,
                metric_name TEXT,
                baseline_value REAL,
                std_dev REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model, metric_name)
            )
        ''')
        
        self.conn.commit()
    
    def insert_metric(self, metric: MetricSnapshot):
        """Insert a metric snapshot."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO metrics (timestamp, model, provider, metric_name, value, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metric.timestamp,
            metric.model,
            metric.provider,
            metric.metric_name,
            metric.value,
            json.dumps(metric.metadata)
        ))
        self.conn.commit()
    
    def get_metrics(self, model: str = None, metric_name: str = None, 
                   hours: int = 24) -> List[MetricSnapshot]:
        """Retrieve metrics from database."""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT timestamp, model, provider, metric_name, value, metadata
            FROM metrics
            WHERE timestamp > datetime('now', '-{} hours')
        '''.format(hours)
        
        params = []
        if model:
            query += ' AND model = ?'
            params.append(model)
        if metric_name:
            query += ' AND metric_name = ?'
            params.append(metric_name)
        
        query += ' ORDER BY timestamp DESC'
        
        cursor.execute(query, params)
        
        metrics = []
        for row in cursor.fetchall():
            metrics.append(MetricSnapshot(
                timestamp=datetime.fromisoformat(row[0]),
                model=row[1],
                provider=row[2],
                metric_name=row[3],
                value=row[4],
                metadata=json.loads(row[5]) if row[5] else {}
            ))
        
        return metrics
    
    def save_baseline(self, model: str, metric_name: str, 
                     baseline_value: float, std_dev: float):
        """Save performance baseline."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO baselines (model, metric_name, baseline_value, std_dev)
            VALUES (?, ?, ?, ?)
        ''', (model, metric_name, baseline_value, std_dev))
        self.conn.commit()
    
    def get_baseline(self, model: str, metric_name: str) -> Optional[Tuple[float, float]]:
        """Get baseline value and std dev."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT baseline_value, std_dev FROM baselines
            WHERE model = ? AND metric_name = ?
        ''', (model, metric_name))
        
        result = cursor.fetchone()
        return result if result else None


class MetricCollector:
    """Collects performance metrics from LLM providers."""
    
    def __init__(self, db: MonitoringDatabase):
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    async def collect_metrics(self, models: List[Dict[str, Any]]):
        """Collect metrics for all configured models."""
        tasks = []
        
        for model_config in models:
            task = self._collect_model_metrics(model_config)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _collect_model_metrics(self, model_config: Dict[str, Any]):
        """Collect metrics for a single model."""
        provider_name = model_config['provider']
        model_name = model_config['model']
        
        try:
            # Test prompt for performance check
            test_prompt = "Hello, this is a monitoring check. Please respond briefly."
            
            # Measure latency
            start_time = datetime.now()
            provider = get_provider(provider_name)
            response = provider.complete(
                prompt=test_prompt,
                model=model_name,
                max_tokens=50
            )
            latency = (datetime.now() - start_time).total_seconds()
            
            # Store metrics
            metrics = [
                MetricSnapshot(
                    timestamp=datetime.now(),
                    model=model_name,
                    provider=provider_name,
                    metric_name='latency',
                    value=latency
                ),
                MetricSnapshot(
                    timestamp=datetime.now(),
                    model=model_name,
                    provider=provider_name,
                    metric_name='success_rate',
                    value=1.0  # Success
                ),
                MetricSnapshot(
                    timestamp=datetime.now(),
                    model=model_name,
                    provider=provider_name,
                    metric_name='tokens_per_second',
                    value=response.get('usage', {}).get('completion_tokens', 0) / latency
                )
            ]
            
            # Calculate cost
            if 'usage' in response:
                cost = self._calculate_cost(provider_name, model_name, response['usage'])
                metrics.append(MetricSnapshot(
                    timestamp=datetime.now(),
                    model=model_name,
                    provider=provider_name,
                    metric_name='cost_per_request',
                    value=cost
                ))
            
            # Store all metrics
            for metric in metrics:
                self.db.insert_metric(metric)
            
            self.logger.info(f"Collected metrics for {model_name}: latency={latency:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics for {model_name}: {e}")
            
            # Record failure
            self.db.insert_metric(MetricSnapshot(
                timestamp=datetime.now(),
                model=model_name,
                provider=provider_name,
                metric_name='success_rate',
                value=0.0,  # Failure
                metadata={'error': str(e)}
            ))
    
    def _calculate_cost(self, provider: str, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost based on token usage."""
        # Simplified cost calculation
        cost_per_1k_tokens = {
            ('openai', 'gpt-4'): 0.03,
            ('openai', 'gpt-3.5-turbo'): 0.002,
            ('anthropic', 'claude-3-5-sonnet-20241022'): 0.015,
            ('google', 'gemini-1.5-pro'): 0.01
        }
        
        rate = cost_per_1k_tokens.get((provider, model), 0.01)
        total_tokens = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
        
        return (total_tokens / 1000) * rate


class AlertManager:
    """Manages alert rules and notifications."""
    
    def __init__(self, config: Dict[str, Any], db: MonitoringDatabase):
        self.config = config
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.alert_history = {}
    
    def check_alerts(self):
        """Check all alert rules against recent metrics."""
        rules = self.config.get('alerts', {}).get('rules', [])
        
        for rule in rules:
            self._check_rule(rule)
    
    def _check_rule(self, rule: Dict[str, Any]):
        """Check a single alert rule."""
        name = rule['name']
        condition = rule['condition']
        severity = rule['severity']
        cooldown = rule.get('cooldown', 3600)
        
        # Check if in cooldown
        if name in self.alert_history:
            last_alert = self.alert_history[name]
            if (datetime.now() - last_alert).total_seconds() < cooldown:
                return
        
        # Evaluate condition
        if self._evaluate_condition(condition):
            alert = Alert(
                id=f"alert_{int(datetime.now().timestamp())}",
                name=name,
                severity=severity,
                condition=condition,
                triggered_at=datetime.now(),
                message=f"Alert: {name} - Condition '{condition}' triggered"
            )
            
            self._send_alert(alert)
            self.alert_history[name] = datetime.now()
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate alert condition against metrics."""
        # Parse condition (simplified)
        # Example: "avg_latency > sla_target * 1.5"
        
        try:
            # Get recent metrics
            metrics = self.db.get_metrics(hours=1)
            
            if not metrics:
                return False
            
            # Simple evaluation for demo
            if 'avg_latency' in condition:
                latencies = [m.value for m in metrics if m.metric_name == 'latency']
                if latencies:
                    avg_latency = np.mean(latencies)
                    # Check against threshold (simplified)
                    return avg_latency > 2.0  # 2 second threshold
            
            elif 'error_rate' in condition:
                success_rates = [m.value for m in metrics if m.metric_name == 'success_rate']
                if success_rates:
                    error_rate = 1 - np.mean(success_rates)
                    return error_rate > 0.5
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        channels = self.config.get('alerts', {}).get('channels', [])
        
        for channel in channels:
            if alert.severity in channel.get('severity', []):
                self._send_to_channel(alert, channel)
    
    def _send_to_channel(self, alert: Alert, channel: Dict[str, Any]):
        """Send alert to specific channel."""
        channel_type = channel['type']
        
        try:
            if channel_type == 'email':
                self._send_email_alert(alert, channel)
            elif channel_type == 'slack':
                self._send_slack_alert(alert, channel)
            elif channel_type == 'webhook':
                self._send_webhook_alert(alert, channel)
            
            self.logger.info(f"Alert sent to {channel_type}: {alert.name}")
            
        except Exception as e:
            self.logger.error(f"Error sending alert to {channel_type}: {e}")
    
    def _send_email_alert(self, alert: Alert, channel: Dict[str, Any]):
        """Send email alert."""
        # Simplified email sending (would use real SMTP in production)
        recipients = channel.get('recipients', [])
        
        message = MIMEMultipart()
        message['Subject'] = f"[{alert.severity.upper()}] {alert.name}"
        message['From'] = 'monitoring@example.com'
        message['To'] = ', '.join(recipients)
        
        body = f"""
        Alert: {alert.name}
        Severity: {alert.severity}
        Time: {alert.triggered_at}
        
        {alert.message}
        
        Condition: {alert.condition}
        """
        
        message.attach(MIMEText(body, 'plain'))
        
        # In production, would send via SMTP
        self.logger.info(f"Email alert prepared for {recipients}")
    
    def _send_slack_alert(self, alert: Alert, channel: Dict[str, Any]):
        """Send Slack alert."""
        webhook_url = channel.get('webhook_url')
        
        if webhook_url:
            payload = {
                'text': f"*[{alert.severity.upper()}]* {alert.name}",
                'attachments': [{
                    'color': 'danger' if alert.severity == 'critical' else 'warning',
                    'fields': [
                        {'title': 'Time', 'value': str(alert.triggered_at), 'short': True},
                        {'title': 'Condition', 'value': alert.condition, 'short': True},
                        {'title': 'Message', 'value': alert.message}
                    ]
                }]
            }
            
            # In production, would POST to webhook
            self.logger.info(f"Slack alert prepared: {payload}")
    
    def _send_webhook_alert(self, alert: Alert, channel: Dict[str, Any]):
        """Send generic webhook alert."""
        url = channel.get('url')
        
        if url:
            payload = {
                'alert': asdict(alert),
                'timestamp': datetime.now().isoformat()
            }
            
            # In production, would POST to webhook
            self.logger.info(f"Webhook alert prepared: {url}")


class MonitoringScheduler:
    """Handles scheduled monitoring tasks."""
    
    def __init__(self, config: Dict[str, Any], db: MonitoringDatabase):
        self.config = config
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.scheduler = BackgroundScheduler()
        self.collector = MetricCollector(db)
        self.alert_manager = AlertManager(config, db)
    
    def start(self):
        """Start the monitoring scheduler."""
        schedules = self.config.get('schedule', {})
        
        # Schedule performance checks
        if 'performance_checks' in schedules:
            freq = schedules['performance_checks']['frequency']
            self._add_job(self._run_performance_checks, freq, 'performance_checks')
        
        # Schedule cost analysis
        if 'cost_analysis' in schedules:
            freq = schedules['cost_analysis']['frequency']
            self._add_job(self._run_cost_analysis, freq, 'cost_analysis')
        
        # Schedule alert checks
        self._add_job(self._check_alerts, '*/5 minutes', 'alert_checks')
        
        self.scheduler.start()
        self.logger.info("Monitoring scheduler started")
    
    def _add_job(self, func, frequency: str, job_id: str):
        """Add a scheduled job."""
        if frequency.startswith('*/'):
            # Interval format: */5 minutes
            interval_value = int(frequency.split()[0].replace('*/', ''))
            interval_unit = frequency.split()[1]
            
            if interval_unit == 'minutes':
                self.scheduler.add_job(func, 'interval', minutes=interval_value, id=job_id)
            elif interval_unit == 'hours':
                self.scheduler.add_job(func, 'interval', hours=interval_value, id=job_id)
        else:
            # Cron format: daily at 2:00
            parts = frequency.split()
            if parts[0] == 'daily':
                hour, minute = parts[2].split(':')
                self.scheduler.add_job(func, CronTrigger(hour=int(hour), minute=int(minute)), id=job_id)
    
    def _run_performance_checks(self):
        """Run scheduled performance checks."""
        self.logger.info("Running scheduled performance checks")
        models = self.config.get('models', [])
        
        # Run async collection
        asyncio.run(self.collector.collect_metrics(models))
        
        # Check alerts after collection
        self._check_alerts()
    
    def _run_cost_analysis(self):
        """Run scheduled cost analysis."""
        self.logger.info("Running scheduled cost analysis")
        
        # Get cost metrics for the last 24 hours
        metrics = self.db.get_metrics(metric_name='cost_per_request', hours=24)
        
        if metrics:
            # Group by model
            costs_by_model = {}
            for metric in metrics:
                if metric.model not in costs_by_model:
                    costs_by_model[metric.model] = []
                costs_by_model[metric.model].append(metric.value)
            
            # Calculate daily costs
            daily_costs = {}
            for model, costs in costs_by_model.items():
                daily_costs[model] = sum(costs)
            
            # Store aggregated metric
            for model, cost in daily_costs.items():
                self.db.insert_metric(MetricSnapshot(
                    timestamp=datetime.now(),
                    model=model,
                    provider='aggregated',
                    metric_name='daily_cost',
                    value=cost
                ))
            
            self.logger.info(f"Daily costs calculated: {daily_costs}")
    
    def _check_alerts(self):
        """Check for alert conditions."""
        self.alert_manager.check_alerts()
    
    def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown()
        self.logger.info("Monitoring scheduler stopped")


class DashboardServer:
    """Web dashboard for monitoring visualization."""
    
    def __init__(self, db: MonitoringDatabase, port: int = 5000):
        self.db = db
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            return self._render_dashboard()
        
        @self.app.route('/api/metrics/<model>')
        def get_metrics(model):
            metrics = self.db.get_metrics(model=model, hours=24)
            
            # Format for JSON response
            data = {
                'timestamps': [],
                'latency': [],
                'success_rate': [],
                'cost': []
            }
            
            for metric in metrics:
                if metric.metric_name == 'latency':
                    data['timestamps'].append(metric.timestamp.isoformat())
                    data['latency'].append(metric.value)
                elif metric.metric_name == 'success_rate':
                    data['success_rate'].append(metric.value)
                elif metric.metric_name == 'cost_per_request':
                    data['cost'].append(metric.value)
            
            return jsonify(data)
        
        @self.app.route('/api/status')
        def status():
            # Get recent metrics
            recent_metrics = self.db.get_metrics(hours=1)
            
            # Calculate summary
            if recent_metrics:
                latencies = [m.value for m in recent_metrics if m.metric_name == 'latency']
                success_rates = [m.value for m in recent_metrics if m.metric_name == 'success_rate']
                
                summary = {
                    'avg_latency': np.mean(latencies) if latencies else 0,
                    'success_rate': np.mean(success_rates) if success_rates else 0,
                    'total_requests': len(latencies),
                    'last_update': max(m.timestamp for m in recent_metrics).isoformat()
                }
            else:
                summary = {
                    'avg_latency': 0,
                    'success_rate': 0,
                    'total_requests': 0,
                    'last_update': datetime.now().isoformat()
                }
            
            return jsonify(summary)
    
    def _render_dashboard(self):
        """Render the main dashboard HTML."""
        template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Monitoring Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .metric-card {
                    background: white;
                    padding: 20px;
                    margin: 10px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    display: inline-block;
                    min-width: 200px;
                }
                .chart-container {
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                }
                .status-good { color: #4CAF50; }
                .status-warning { color: #FF9800; }
                .status-critical { color: #F44336; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 LLM Monitoring Dashboard</h1>
                
                <div id="status-cards">
                    <div class="metric-card">
                        <h3>Average Latency</h3>
                        <p id="avg-latency" class="metric-value">Loading...</p>
                    </div>
                    <div class="metric-card">
                        <h3>Success Rate</h3>
                        <p id="success-rate" class="metric-value">Loading...</p>
                    </div>
                    <div class="metric-card">
                        <h3>Total Requests</h3>
                        <p id="total-requests" class="metric-value">Loading...</p>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2>Latency Over Time</h2>
                    <canvas id="latencyChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h2>Success Rate</h2>
                    <canvas id="successChart"></canvas>
                </div>
            </div>
            
            <script>
                // Initialize charts
                const latencyCtx = document.getElementById('latencyChart').getContext('2d');
                const latencyChart = new Chart(latencyCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Latency (seconds)',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
                
                // Update dashboard
                function updateDashboard() {
                    // Fetch status
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('avg-latency').textContent = 
                                data.avg_latency.toFixed(2) + 's';
                            document.getElementById('success-rate').textContent = 
                                (data.success_rate * 100).toFixed(1) + '%';
                            document.getElementById('total-requests').textContent = 
                                data.total_requests;
                        });
                    
                    // Fetch metrics (simplified for demo)
                    fetch('/api/metrics/gpt-4')
                        .then(response => response.json())
                        .then(data => {
                            if (data.timestamps.length > 0) {
                                latencyChart.data.labels = data.timestamps.slice(-20);
                                latencyChart.data.datasets[0].data = data.latency.slice(-20);
                                latencyChart.update();
                            }
                        });
                }
                
                // Update every 5 seconds
                updateDashboard();
                setInterval(updateDashboard, 5000);
            </script>
        </body>
        </html>
        '''
        
        return template
    
    def start(self):
        """Start the dashboard server."""
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


class ReportGenerator:
    """Generates monitoring reports."""
    
    def __init__(self, db: MonitoringDatabase):
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    def generate_report(self, report_type: str, date_range: Tuple[datetime, datetime],
                       output_file: str) -> Dict[str, Any]:
        """Generate a monitoring report."""
        self.logger.info(f"Generating {report_type} report")
        
        # Get metrics for date range
        hours = int((date_range[1] - date_range[0]).total_seconds() / 3600)
        metrics = self.db.get_metrics(hours=hours)
        
        if report_type == 'weekly':
            report_data = self._generate_weekly_report(metrics, date_range)
        elif report_type == 'daily':
            report_data = self._generate_daily_report(metrics, date_range)
        else:
            report_data = self._generate_custom_report(metrics, date_range)
        
        # Save report
        self._save_report(report_data, output_file)
        
        return {
            'status': 'completed',
            'output_file': output_file,
            'metrics_analyzed': len(metrics)
        }
    
    def _generate_weekly_report(self, metrics: List[MetricSnapshot], 
                               date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate weekly performance report."""
        # Group metrics by model
        models_data = {}
        
        for metric in metrics:
            if metric.model not in models_data:
                models_data[metric.model] = {
                    'latencies': [],
                    'success_rates': [],
                    'costs': []
                }
            
            if metric.metric_name == 'latency':
                models_data[metric.model]['latencies'].append(metric.value)
            elif metric.metric_name == 'success_rate':
                models_data[metric.model]['success_rates'].append(metric.value)
            elif metric.metric_name == 'cost_per_request':
                models_data[metric.model]['costs'].append(metric.value)
        
        # Calculate summary statistics
        summary = {}
        for model, data in models_data.items():
            summary[model] = {
                'avg_latency': np.mean(data['latencies']) if data['latencies'] else 0,
                'p95_latency': np.percentile(data['latencies'], 95) if data['latencies'] else 0,
                'success_rate': np.mean(data['success_rates']) if data['success_rates'] else 0,
                'total_cost': sum(data['costs']) if data['costs'] else 0,
                'request_count': len(data['latencies'])
            }
        
        return {
            'report_type': 'weekly',
            'date_range': {
                'start': date_range[0].isoformat(),
                'end': date_range[1].isoformat()
            },
            'summary': summary,
            'total_requests': sum(s['request_count'] for s in summary.values()),
            'total_cost': sum(s['total_cost'] for s in summary.values())
        }
    
    def _generate_daily_report(self, metrics: List[MetricSnapshot], 
                              date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate daily summary report."""
        # Similar to weekly but with hourly breakdown
        hourly_data = {}
        
        for metric in metrics:
            hour = metric.timestamp.hour
            if hour not in hourly_data:
                hourly_data[hour] = []
            
            if metric.metric_name == 'latency':
                hourly_data[hour].append(metric.value)
        
        hourly_summary = {}
        for hour, latencies in hourly_data.items():
            hourly_summary[hour] = {
                'avg_latency': np.mean(latencies),
                'request_count': len(latencies)
            }
        
        return {
            'report_type': 'daily',
            'date': date_range[0].date().isoformat(),
            'hourly_breakdown': hourly_summary
        }
    
    def _generate_custom_report(self, metrics: List[MetricSnapshot], 
                               date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate custom report."""
        return {
            'report_type': 'custom',
            'date_range': {
                'start': date_range[0].isoformat(),
                'end': date_range[1].isoformat()
            },
            'total_metrics': len(metrics)
        }
    
    def _save_report(self, report_data: Dict[str, Any], output_file: str):
        """Save report to file."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
        elif output_file.endswith('.html'):
            # Generate HTML report
            html = self._generate_html_report(report_data)
            with open(output_file, 'w') as f:
                f.write(html)
        else:
            # Default to JSON
            with open(output_file + '.json', 'w') as f:
                json.dump(report_data, f, indent=2)
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Performance Report - {report_data['report_type'].title()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background: #f9f9f9; padding: 20px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>📊 LLM Performance Report - {report_data['report_type'].title()}</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>Total Requests: {report_data.get('total_requests', 'N/A')}</p>
                <p>Total Cost: ${report_data.get('total_cost', 0):.2f}</p>
            </div>
            
            <h2>Model Performance</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Avg Latency</th>
                    <th>P95 Latency</th>
                    <th>Success Rate</th>
                    <th>Total Cost</th>
                    <th>Requests</th>
                </tr>
        """
        
        if 'summary' in report_data:
            for model, stats in report_data['summary'].items():
                template += f"""
                <tr>
                    <td>{model}</td>
                    <td>{stats['avg_latency']:.2f}s</td>
                    <td>{stats['p95_latency']:.2f}s</td>
                    <td>{stats['success_rate']*100:.1f}%</td>
                    <td>${stats['total_cost']:.2f}</td>
                    <td>{stats['request_count']}</td>
                </tr>
                """
        
        template += """
            </table>
        </body>
        </html>
        """
        
        return template


def create_baseline(config: Dict[str, Any], db: MonitoringDatabase, iterations: int = 10):
    """Create performance baseline."""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating baseline with {iterations} iterations")
    
    collector = MetricCollector(db)
    models = config.get('models', [])
    
    # Collect baseline metrics
    for i in range(iterations):
        logger.info(f"Baseline iteration {i+1}/{iterations}")
        asyncio.run(collector.collect_metrics(models))
        time.sleep(2)  # Wait between iterations
    
    # Calculate baselines
    for model_config in models:
        model = model_config['model']
        
        # Get collected metrics
        metrics = db.get_metrics(model=model, hours=1)
        
        # Calculate baseline for each metric type
        for metric_name in ['latency', 'success_rate', 'tokens_per_second']:
            values = [m.value for m in metrics if m.metric_name == metric_name]
            
            if values:
                baseline_value = np.mean(values)
                std_dev = np.std(values)
                
                db.save_baseline(model, metric_name, baseline_value, std_dev)
                logger.info(f"Baseline for {model} {metric_name}: {baseline_value:.3f} (±{std_dev:.3f})")
    
    return {'status': 'completed', 'iterations': iterations}


def main():
    """Main demo execution."""
    parser = argparse.ArgumentParser(description='Monitoring Complete Demo')
    
    # Operation modes
    parser.add_argument('--init-monitoring', action='store_true', help='Initialize monitoring')
    parser.add_argument('--start-monitoring', action='store_true', help='Start monitoring service')
    parser.add_argument('--create-baseline', action='store_true', help='Create performance baseline')
    parser.add_argument('--start-dashboard', action='store_true', help='Start web dashboard')
    parser.add_argument('--generate-report', action='store_true', help='Generate report')
    parser.add_argument('--status', action='store_true', help='Show monitoring status')
    
    # Configuration
    parser.add_argument('--config', default='monitoring_config.yaml', help='Configuration file')
    parser.add_argument('--database', default='sqlite:///monitoring.db', help='Database URL')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port')
    
    # Report parameters
    parser.add_argument('--type', default='weekly', help='Report type')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--iterations', type=int, default=10, help='Baseline iterations')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                'models': [
                    {'provider': 'openai', 'model': 'gpt-4', 'priority': 'high', 'sla_target': 2.0},
                    {'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20241022', 'priority': 'high', 'sla_target': 3.0}
                ],
                'schedule': {
                    'performance_checks': {'frequency': '*/5 minutes', 'timeout': 300},
                    'cost_analysis': {'frequency': 'daily at 2:00', 'timeout': 600}
                },
                'alerts': {
                    'channels': [
                        {'type': 'email', 'recipients': ['team@example.com'], 'severity': ['critical', 'warning']}
                    ],
                    'rules': [
                        {'name': 'High Latency', 'condition': 'avg_latency > 2.0', 'severity': 'warning', 'cooldown': 3600}
                    ]
                }
            }
        
        # Initialize database
        db = MonitoringDatabase(args.database)
        
        if args.init_monitoring:
            # Initialize monitoring system
            logger.info("Initializing monitoring system...")
            
            # Save configuration
            with open('monitoring_config.yaml', 'w') as f:
                yaml.dump(config, f)
            
            print("\n✅ Monitoring system initialized")
            print(f"✓ Database: {args.database}")
            print(f"✓ Configuration: monitoring_config.yaml")
            print(f"✓ Models configured: {len(config['models'])}")
            print(f"✓ Alert rules: {len(config['alerts']['rules'])}")
            
        elif args.create_baseline:
            # Create performance baseline
            logger.info("Creating performance baseline...")
            
            result = create_baseline(config, db, args.iterations)
            
            print("\n📊 Baseline Performance Established")
            print("=" * 50)
            
            # Display baseline values
            for model_config in config['models']:
                model = model_config['model']
                print(f"\n{model}:")
                
                for metric in ['latency', 'success_rate', 'tokens_per_second']:
                    baseline = db.get_baseline(model, metric)
                    if baseline:
                        print(f"  - {metric}: {baseline[0]:.3f} (±{baseline[1]:.3f})")
        
        elif args.start_monitoring:
            # Start monitoring service
            logger.info("Starting monitoring service...")
            
            scheduler = MonitoringScheduler(config, db)
            
            try:
                scheduler.start()
                
                print("\n🚀 LLM Monitoring Service Started")
                print("=" * 50)
                print("Active schedules:")
                for job in scheduler.scheduler.get_jobs():
                    print(f"  - {job.id}: Next run at {job.next_run_time}")
                
                print("\nPress Ctrl+C to stop...")
                
                # Keep running
                while True:
                    time.sleep(60)
                    
            except KeyboardInterrupt:
                scheduler.stop()
                print("\n✅ Monitoring service stopped")
        
        elif args.start_dashboard:
            # Start dashboard server
            logger.info(f"Starting dashboard on port {args.port}...")
            
            dashboard = DashboardServer(db, args.port)
            
            print(f"\n🎯 Dashboard started at http://localhost:{args.port}")
            print("Press Ctrl+C to stop...")
            
            try:
                dashboard.start()
            except KeyboardInterrupt:
                print("\n✅ Dashboard stopped")
        
        elif args.generate_report:
            # Generate report
            logger.info(f"Generating {args.type} report...")
            
            generator = ReportGenerator(db)
            
            # Calculate date range
            end_date = datetime.now()
            if args.type == 'weekly':
                start_date = end_date - timedelta(days=7)
            elif args.type == 'daily':
                start_date = end_date - timedelta(days=1)
            else:
                start_date = end_date - timedelta(days=30)
            
            result = generator.generate_report(
                args.type,
                (start_date, end_date),
                args.output or f"{args.type}_report.html"
            )
            
            print(f"\n📊 Report Generated")
            print(f"Type: {args.type}")
            print(f"Period: {start_date.date()} to {end_date.date()}")
            print(f"Output: {result['output_file']}")
            print(f"Metrics analyzed: {result['metrics_analyzed']}")
        
        elif args.status:
            # Show monitoring status
            logger.info("Checking monitoring status...")
            
            # Get recent metrics
            recent_metrics = db.get_metrics(hours=1)
            
            print("\n📊 Monitoring Status")
            print("=" * 50)
            
            if recent_metrics:
                # Group by model
                models_seen = set()
                for metric in recent_metrics:
                    models_seen.add(metric.model)
                
                print(f"Active models: {len(models_seen)}")
                print(f"Recent metrics: {len(recent_metrics)}")
                print(f"Last update: {max(m.timestamp for m in recent_metrics)}")
                
                print("\nRecent performance:")
                for model in models_seen:
                    model_metrics = [m for m in recent_metrics if m.model == model]
                    latencies = [m.value for m in model_metrics if m.metric_name == 'latency']
                    if latencies:
                        print(f"  {model}: {np.mean(latencies):.2f}s avg latency")
            else:
                print("No recent metrics found. Start monitoring to collect data.")
        
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error in monitoring demo: {e}")
        raise


if __name__ == "__main__":
    main()