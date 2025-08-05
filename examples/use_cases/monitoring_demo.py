#!/usr/bin/env python3
"""
Comprehensive demo of the LLM Lab Monitoring System.

This script demonstrates all the key features of the monitoring system:
- Database setup and model management
- Scheduled benchmark execution
- Performance regression detection
- Alert system with notifications
- RESTful API for dashboard integration

Usage:
    python examples/use_cases/monitoring_demo.py
"""

# Import paths fixed - sys.path manipulation removed
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import sys

# Add the project root to the path
, '../..'))

from src.use_cases.monitoring import (
    DatabaseManager, BenchmarkScheduler, RegressionDetector, 
    AlertManager, MonitoringAPI
)
from src.use_cases.monitoring.scheduler import BenchmarkJobConfig, ScheduleType
from src.use_cases.monitoring.alerting import AlertRule, AlertType, AlertSeverity, NotificationChannel
from src.use_cases.monitoring.api import create_monitoring_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def mock_benchmark_runner(
    model_name: str,
    dataset_name: str,
    config: Dict[str, Any],
    run_id: int
) -> Dict[str, Any]:
    """
    Mock benchmark runner for demonstration.
    In a real implementation, this would execute actual benchmarks.
    """
    import random
    import time
    
    logger.info(f"Running benchmark: {model_name} on {dataset_name} (run {run_id})")
    
    # Simulate benchmark execution time
    await asyncio.sleep(2)
    
    # Generate mock results with some variability
    base_accuracy = 0.85
    base_latency = 250.0
    base_cost = 0.001
    
    # Add some randomness to simulate real performance variations
    accuracy = base_accuracy + random.uniform(-0.05, 0.05)
    latency = base_latency + random.uniform(-50, 50)
    cost = base_cost + random.uniform(-0.0002, 0.0002)
    
    # Occasionally simulate a performance regression
    if random.random() < 0.2:  # 20% chance of regression
        accuracy -= 0.1  # Significant accuracy drop
        latency += 100   # Latency spike
        cost += 0.0005   # Cost increase
        logger.warning(f"Simulated performance regression in run {run_id}")
    
    return {
        'accuracy': max(0, min(1, accuracy)),
        'latency_ms': max(50, latency),
        'cost': max(0, cost),
        'success': True
    }


class MonitoringSystemDemo:
    """Comprehensive demonstration of the monitoring system."""
    
    def __init__(self):
        """Initialize the demo with all monitoring components."""
        # Use in-memory SQLite for demo
        self.db_manager = DatabaseManager('sqlite:///monitoring_demo.db')
        
        # Initialize all components
        self.scheduler = BenchmarkScheduler(
            database_manager=self.db_manager,
            benchmark_runner=mock_benchmark_runner
        )
        
        self.regression_detector = RegressionDetector(self.db_manager)
        self.alert_manager = AlertManager(self.db_manager)
        
        # Create API
        self.api = MonitoringAPI(
            database_manager=self.db_manager,
            scheduler=self.scheduler,
            regression_detector=self.regression_detector,
            alert_manager=self.alert_manager
        )
        
        logger.info("Monitoring system demo initialized")
    
    async def setup_database(self):
        """Set up the database with sample data."""
        logger.info("Setting up database...")
        
        # Initialize database tables
        self.db_manager.initialize_database(drop_existing=True)
        
        # Create sample models
        models = [
            {
                'name': 'gpt-4',
                'provider': 'openai',
                'cost_per_input_token': 0.00003,
                'cost_per_output_token': 0.00006,
                'max_tokens': 4096,
                'context_length': 8192,
                'model_type': 'chat'
            },
            {
                'name': 'claude-3-sonnet',
                'provider': 'anthropic',
                'cost_per_input_token': 0.000015,
                'cost_per_output_token': 0.000075,
                'max_tokens': 4096,
                'context_length': 200000,
                'model_type': 'chat'
            },
            {
                'name': 'llama-2-7b',
                'provider': 'local',
                'cost_per_input_token': 0.0,
                'cost_per_output_token': 0.0,
                'max_tokens': 2048,
                'context_length': 4096,
                'model_type': 'chat'
            }
        ]
        
        created_models = []
        for model_data in models:
            model = self.db_manager.create_model(model_data)
            created_models.append(model)
            logger.info(f"Created model: {model.name} (ID: {model.id})")
        
        return created_models
    
    async def demonstrate_scheduling(self, models):
        """Demonstrate benchmark scheduling capabilities."""
        logger.info("Demonstrating benchmark scheduling...")
        
        await self.scheduler.start()
        
        # Create monitoring suite for all models
        datasets = ['truthfulness', 'reasoning', 'knowledge']
        
        for model in models:
            for dataset in datasets:
                # Add a quick test job that runs every 30 seconds for demo
                job_id = f"{model.name}_{dataset}_demo"
                job_config = BenchmarkJobConfig(
                    job_id=job_id,
                    model_name=model.name,
                    dataset_name=dataset,
                    benchmark_config={'mode': 'demo', 'sample_size': 10},
                    schedule_type=ScheduleType.INTERVAL,
                    schedule_params={'seconds': 30},
                    enabled=True
                )
                
                success = self.scheduler.add_job(job_config)
                if success:
                    logger.info(f"Scheduled job: {job_id}")
        
        # Let jobs run for a bit
        logger.info("Letting scheduled jobs run for 2 minutes...")
        await asyncio.sleep(120)
        
        # Show scheduler statistics
        stats = self.scheduler.get_scheduler_stats()
        logger.info(f"Scheduler stats: {stats}")
        
        return stats
    
    async def generate_historical_data(self, models):
        """Generate some historical benchmark data for regression detection."""
        logger.info("Generating historical benchmark data...")
        
        datasets = ['truthfulness', 'reasoning', 'knowledge']
        
        for model in models:
            for dataset in datasets:
                # Create several historical runs
                for i in range(10):
                    run_time = datetime.utcnow() - timedelta(days=i, hours=i*2)
                    
                    # Create benchmark run
                    run_data = {
                        'model_id': model.id,
                        'dataset_name': dataset,
                        'run_config': {'mode': 'historical', 'sample_size': 100},
                        'status': 'completed',
                        'timestamp': run_time,
                        'completed_at': run_time + timedelta(minutes=5),
                        'trigger_type': 'manual'
                    }
                    
                    run = self.db_manager.create_benchmark_run(run_data)
                    
                    # Generate mock results using the same function
                    result = await mock_benchmark_runner(model.name, dataset, {}, run.id)
                    
                    # Update run with results
                    self.db_manager.update_benchmark_run(run.id, {
                        'accuracy': result['accuracy'],
                        'latency_ms': result['latency_ms'],
                        'total_cost': result['cost'],
                        'duration_seconds': 300,
                        'sample_count': 100
                    })
                    
                    # Create individual metrics
                    metrics_data = [
                        {
                            'run_id': run.id,
                            'metric_type': 'accuracy',
                            'metric_name': 'overall_accuracy',
                            'value': result['accuracy'],
                            'unit': 'percentage',
                            'category': 'performance',
                            'timestamp': run_time
                        },
                        {
                            'run_id': run.id,
                            'metric_type': 'latency',
                            'metric_name': 'avg_latency_ms',
                            'value': result['latency_ms'],
                            'unit': 'milliseconds',
                            'category': 'performance',
                            'timestamp': run_time
                        },
                        {
                            'run_id': run.id,
                            'metric_type': 'cost',
                            'metric_name': 'total_cost',
                            'value': result['cost'],
                            'unit': 'dollars',
                            'category': 'cost',
                            'timestamp': run_time
                        }
                    ]
                    
                    self.db_manager.create_metrics_batch(metrics_data)
        
        logger.info("Historical data generation completed")
    
    async def demonstrate_regression_detection(self, models):
        """Demonstrate regression detection capabilities."""
        logger.info("Demonstrating regression detection...")
        
        # Analyze each model for regressions
        all_results = []
        
        for model in models:
            logger.info(f"Analyzing regressions for model: {model.name}")
            
            results = self.regression_detector.detect_regressions(
                model_id=model.id,
                days_back=30
            )
            
            all_results.extend(results)
            
            regressions_found = [r for r in results if r.regression_detected]
            
            if regressions_found:
                logger.warning(f"Found {len(regressions_found)} regressions for {model.name}")
                for regression in regressions_found:
                    logger.warning(
                        f"  - {regression.metric_name}: {regression.change_percent*100:.1f}% change "
                        f"(severity: {regression.severity}, confidence: {regression.confidence_score:.2f})"
                    )
            else:
                logger.info(f"No significant regressions found for {model.name}")
            
            # Get trend analysis
            trends = self.regression_detector.analyze_model_trends(model.id, days_back=30)
            logger.info(f"Analyzed {trends.get('metrics_analyzed', 0)} metric trends for {model.name}")
        
        # Get overall regression summary
        summary = self.regression_detector.get_regression_summary(
            model_ids=[m.id for m in models],
            days_back=30
        )
        
        logger.info(f"Regression summary: {summary['summary']}")
        
        return all_results, summary
    
    async def demonstrate_alerting(self, regression_results):
        """Demonstrate alerting system capabilities."""
        logger.info("Demonstrating alerting system...")
        
        # Configure notification channels
        self.alert_manager.configure_notification_channel(
            NotificationChannel.CONSOLE,
            {'log_level': 'INFO'}
        )
        
        # Process regression results and generate alerts
        alerts = await self.alert_manager.process_regression_results(regression_results)
        
        if alerts:
            logger.info(f"Generated {len(alerts)} alerts from regression results")
            for alert in alerts:
                logger.info(f"  - Alert: {alert.title} (severity: {alert.severity.value})")
        else:
            logger.info("No alerts generated from regression results")
        
        # Simulate a benchmark failure alert
        failure_alert = await self.alert_manager.process_benchmark_failure(
            run_id=999,
            model_id=1,
            error_message="Mock benchmark failure for demonstration"
        )
        
        if failure_alert:
            logger.info(f"Generated benchmark failure alert: {failure_alert.title}")
        
        # Get alert statistics
        stats = self.alert_manager.get_alert_statistics(days_back=30)
        logger.info(f"Alert statistics: {stats}")
        
        return alerts
    
    async def demonstrate_api(self):
        """Demonstrate API capabilities."""
        logger.info("Demonstrating API capabilities...")
        
        # In a real scenario, you would start the API server in a separate process
        # For demo purposes, we'll just show the API structure
        
        logger.info("API endpoints available:")
        routes = []
        for route in self.api.app.routes:
            if hasattr(route, 'methods'):
                for method in route.methods:
                    if method != 'HEAD':
                        routes.append(f"{method} {route.path}")
        
        for route in sorted(routes):
            logger.info(f"  - {route}")
        
        # Demonstrate health check
        # This would normally be done via HTTP request
        logger.info("API health check would return system status")
        
        return len(routes)
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        if self.scheduler:
            await self.scheduler.shutdown()
        
        logger.info("Cleanup completed")
    
    async def run_full_demo(self):
        """Run the complete monitoring system demonstration."""
        logger.info("Starting comprehensive monitoring system demo...")
        
        try:
            # 1. Set up database and models
            models = await self.setup_database()
            
            # 2. Generate historical data
            await self.generate_historical_data(models)
            
            # 3. Demonstrate scheduling (shortened for demo)
            await self.demonstrate_scheduling(models)
            
            # 4. Demonstrate regression detection
            regression_results, summary = await self.demonstrate_regression_detection(models)
            
            # 5. Demonstrate alerting
            alerts = await self.demonstrate_alerting(regression_results)
            
            # 6. Demonstrate API
            api_endpoints = await self.demonstrate_api()
            
            # Final summary
            logger.info("=" * 60)
            logger.info("MONITORING SYSTEM DEMO SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Models created: {len(models)}")
            logger.info(f"Scheduled jobs: {len(self.scheduler.list_jobs())}")
            logger.info(f"Regression analysis: {summary['summary']['total_regressions_detected']} regressions detected")
            logger.info(f"Alerts generated: {len(alerts)}")
            logger.info(f"API endpoints: {api_endpoints}")
            logger.info("=" * 60)
            
            logger.info("Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main function to run the monitoring system demo."""
    demo = MonitoringSystemDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())