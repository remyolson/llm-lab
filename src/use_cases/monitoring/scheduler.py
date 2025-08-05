"""
Scheduled job system for automated benchmark execution.

This module provides scheduling capabilities for running benchmarks at regular
intervals, handling job management, and coordinating with the monitoring system.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, JobExecutionEvent

from .database import DatabaseManager
from .models import BenchmarkRun

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of schedule triggers supported."""
    INTERVAL = "interval"
    CRON = "cron"
    DATE = "date"


@dataclass
class BenchmarkJobConfig:
    """Configuration for a scheduled benchmark job."""
    job_id: str
    model_name: str
    dataset_name: str
    benchmark_config: Dict[str, Any]
    schedule_type: ScheduleType
    schedule_params: Dict[str, Any]
    enabled: bool = True
    max_instances: int = 1
    coalesce: bool = True
    misfire_grace_time: int = 300  # 5 minutes
    metadata: Optional[Dict[str, Any]] = None


class BenchmarkScheduler:
    """Manages scheduled benchmark execution."""
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        benchmark_runner: Optional[Callable] = None,
        timezone: str = 'UTC'
    ):
        """
        Initialize the benchmark scheduler.
        
        Args:
            database_manager: Database manager for storing job results
            benchmark_runner: Function to execute benchmarks
            timezone: Timezone for scheduling (default: UTC)
        """
        self.db_manager = database_manager
        self.benchmark_runner = benchmark_runner
        self.timezone = timezone
        
        # Configure scheduler
        jobstores = {
            'default': SQLAlchemyJobStore(
                engine=database_manager.engine,
                tablename='apscheduler_jobs'
            )
        }
        
        executors = {
            'default': AsyncIOExecutor()
        }
        
        job_defaults = {
            'coalesce': True,
            'max_instances': 1,
            'misfire_grace_time': 300
        }
        
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=timezone
        )
        
        # Set up event listeners
        self.scheduler.add_listener(
            self._job_executed_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )
        
        self.active_jobs: Dict[str, BenchmarkJobConfig] = {}
        self._running = False
        
        logger.info(f"Benchmark scheduler initialized with timezone: {timezone}")
    
    async def start(self) -> None:
        """Start the scheduler."""
        if not self._running:
            self.scheduler.start()
            self._running = True
            logger.info("Benchmark scheduler started")
    
    async def shutdown(self, wait: bool = True) -> None:
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=wait)
            self._running = False
            logger.info("Benchmark scheduler stopped")
    
    def add_job(self, job_config: BenchmarkJobConfig) -> bool:
        """
        Add a new scheduled benchmark job.
        
        Args:
            job_config: Configuration for the benchmark job
            
        Returns:
            True if job was added successfully, False otherwise
        """
        try:
            # Create the appropriate trigger
            trigger = self._create_trigger(job_config)
            
            # Add job to scheduler
            self.scheduler.add_job(
                func=self._execute_benchmark_job,
                trigger=trigger,
                id=job_config.job_id,
                args=[job_config],
                max_instances=job_config.max_instances,
                coalesce=job_config.coalesce,
                misfire_grace_time=job_config.misfire_grace_time,
                replace_existing=True
            )
            
            # Store job config
            self.active_jobs[job_config.job_id] = job_config
            
            logger.info(f"Added scheduled job: {job_config.job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add job {job_config.job_id}: {e}")
            return False
    
    def remove_job(self, job_id: str) -> bool:
        """
        Remove a scheduled job.
        
        Args:
            job_id: ID of the job to remove
            
        Returns:
            True if job was removed successfully, False otherwise
        """
        try:
            self.scheduler.remove_job(job_id)
            self.active_jobs.pop(job_id, None)
            logger.info(f"Removed scheduled job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
            return False
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job."""
        try:
            self.scheduler.pause_job(job_id)
            if job_id in self.active_jobs:
                self.active_jobs[job_id].enabled = False
            logger.info(f"Paused job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        try:
            self.scheduler.resume_job(job_id)
            if job_id in self.active_jobs:
                self.active_jobs[job_id].enabled = True
            logger.info(f"Resumed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a job."""
        job = self.scheduler.get_job(job_id)
        if not job:
            return None
        
        config = self.active_jobs.get(job_id)
        return {
            'job_id': job_id,
            'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
            'enabled': config.enabled if config else True,
            'trigger': str(job.trigger),
            'max_instances': job.max_instances,
            'coalesce': job.coalesce
        }
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs and their status."""
        jobs = []
        for job in self.scheduler.get_jobs():
            config = self.active_jobs.get(job.id)
            jobs.append({
                'job_id': job.id,
                'model_name': config.model_name if config else 'unknown',
                'dataset_name': config.dataset_name if config else 'unknown',
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'enabled': config.enabled if config else True,
                'trigger': str(job.trigger),
                'schedule_type': config.schedule_type.value if config else 'unknown'
            })
        return jobs
    
    def _create_trigger(self, job_config: BenchmarkJobConfig):
        """Create appropriate trigger based on schedule type."""
        if job_config.schedule_type == ScheduleType.INTERVAL:
            return IntervalTrigger(**job_config.schedule_params, timezone=self.timezone)
        elif job_config.schedule_type == ScheduleType.CRON:
            return CronTrigger(**job_config.schedule_params, timezone=self.timezone)
        elif job_config.schedule_type == ScheduleType.DATE:
            return DateTrigger(**job_config.schedule_params, timezone=self.timezone)
        else:
            raise ValueError(f"Unsupported schedule type: {job_config.schedule_type}")
    
    async def _execute_benchmark_job(self, job_config: BenchmarkJobConfig) -> None:
        """Execute a scheduled benchmark job."""
        logger.info(f"Executing scheduled benchmark job: {job_config.job_id}")
        
        # Create benchmark run record
        run_data = {
            'model_id': await self._get_or_create_model_id(job_config.model_name),
            'dataset_name': job_config.dataset_name,
            'run_config': job_config.benchmark_config,
            'status': 'running',
            'scheduled_job_id': job_config.job_id,
            'trigger_type': 'scheduled'
        }
        
        benchmark_run = None
        try:
            benchmark_run = self.db_manager.create_benchmark_run(run_data)
            
            # Execute the benchmark if runner is provided
            if self.benchmark_runner:
                result = await self._run_benchmark_safely(job_config, benchmark_run.id)
                
                # Update run with results
                if result.get('success', False):
                    self.db_manager.complete_benchmark_run(
                        benchmark_run.id,
                        status='completed'
                    )
                else:
                    self.db_manager.complete_benchmark_run(
                        benchmark_run.id,
                        status='failed',
                        error_message=result.get('error', 'Unknown error')
                    )
            else:
                logger.warning(f"No benchmark runner configured for job {job_config.job_id}")
                self.db_manager.complete_benchmark_run(
                    benchmark_run.id,
                    status='failed',
                    error_message='No benchmark runner configured'
                )
                
        except Exception as e:
            logger.error(f"Error executing job {job_config.job_id}: {e}")
            if benchmark_run:
                self.db_manager.complete_benchmark_run(
                    benchmark_run.id,
                    status='failed',
                    error_message=str(e)
                )
    
    async def _run_benchmark_safely(
        self,
        job_config: BenchmarkJobConfig,
        run_id: int
    ) -> Dict[str, Any]:
        """Run benchmark with error handling and timeout."""
        try:
            # Create timeout based on job config or default to 1 hour
            timeout = job_config.metadata.get('timeout', 3600) if job_config.metadata else 3600
            
            # Execute benchmark with timeout
            result = await asyncio.wait_for(
                self.benchmark_runner(
                    model_name=job_config.model_name,
                    dataset_name=job_config.dataset_name,
                    config=job_config.benchmark_config,
                    run_id=run_id
                ),
                timeout=timeout
            )
            
            return {'success': True, 'result': result}
            
        except asyncio.TimeoutError:
            error_msg = f"Benchmark timed out after {timeout} seconds"
            logger.error(f"Job {job_config.job_id}: {error_msg}")
            return {'success': False, 'error': error_msg}
            
        except Exception as e:
            error_msg = f"Benchmark execution failed: {str(e)}"
            logger.error(f"Job {job_config.job_id}: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def _get_or_create_model_id(self, model_name: str) -> int:
        """Get or create model metadata record."""
        model = self.db_manager.get_model_by_name(model_name)
        if not model:
            # Create new model record
            model_data = {
                'name': model_name,
                'provider': 'unknown',  # Will be updated when first used
                'active': True
            }
            model = self.db_manager.create_model(model_data)
        return model.id
    
    def _job_executed_listener(self, event: JobExecutionEvent) -> None:
        """Handle job execution events."""
        if event.exception:
            logger.error(f"Job {event.job_id} failed: {event.exception}")
        else:
            logger.info(f"Job {event.job_id} completed successfully")
    
    # Convenience methods for common scheduling patterns
    def add_daily_job(
        self,
        job_id: str,
        model_name: str,
        dataset_name: str,
        benchmark_config: Dict[str, Any],
        hour: int = 2,
        minute: int = 0
    ) -> bool:
        """Add a daily benchmark job."""
        job_config = BenchmarkJobConfig(
            job_id=job_id,
            model_name=model_name,
            dataset_name=dataset_name,
            benchmark_config=benchmark_config,
            schedule_type=ScheduleType.CRON,
            schedule_params={'hour': hour, 'minute': minute}
        )
        return self.add_job(job_config)
    
    def add_weekly_job(
        self,
        job_id: str,
        model_name: str,
        dataset_name: str,
        benchmark_config: Dict[str, Any],
        day_of_week: int = 0,  # Monday
        hour: int = 2,
        minute: int = 0
    ) -> bool:
        """Add a weekly benchmark job."""
        job_config = BenchmarkJobConfig(
            job_id=job_id,
            model_name=model_name,
            dataset_name=dataset_name,
            benchmark_config=benchmark_config,
            schedule_type=ScheduleType.CRON,
            schedule_params={'day_of_week': day_of_week, 'hour': hour, 'minute': minute}
        )
        return self.add_job(job_config)
    
    def add_interval_job(
        self,
        job_id: str,
        model_name: str,
        dataset_name: str,
        benchmark_config: Dict[str, Any],
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0
    ) -> bool:
        """Add an interval-based benchmark job."""
        if hours == 0 and minutes == 0 and seconds == 0:
            raise ValueError("At least one time interval must be specified")
        
        schedule_params = {}
        if hours > 0:
            schedule_params['hours'] = hours
        if minutes > 0:
            schedule_params['minutes'] = minutes
        if seconds > 0:
            schedule_params['seconds'] = seconds
        
        job_config = BenchmarkJobConfig(
            job_id=job_id,
            model_name=model_name,
            dataset_name=dataset_name,
            benchmark_config=benchmark_config,
            schedule_type=ScheduleType.INTERVAL,
            schedule_params=schedule_params
        )
        return self.add_job(job_config)
    
    def add_one_time_job(
        self,
        job_id: str,
        model_name: str,
        dataset_name: str,
        benchmark_config: Dict[str, Any],
        run_date: datetime
    ) -> bool:
        """Add a one-time benchmark job."""
        job_config = BenchmarkJobConfig(
            job_id=job_id,
            model_name=model_name,
            dataset_name=dataset_name,
            benchmark_config=benchmark_config,
            schedule_type=ScheduleType.DATE,
            schedule_params={'run_date': run_date}
        )
        return self.add_job(job_config)
    
    # Job templates for common scenarios
    def create_monitoring_suite(
        self,
        model_names: List[str],
        datasets: List[str] = None
    ) -> List[str]:
        """
        Create a complete monitoring suite for given models.
        
        Args:
            model_names: List of model names to monitor
            datasets: List of datasets to use (default: standard benchmark datasets)
            
        Returns:
            List of created job IDs
        """
        if datasets is None:
            datasets = ['truthfulness', 'reasoning', 'knowledge']
        
        created_jobs = []
        
        for model_name in model_names:
            for dataset in datasets:
                # Daily detailed benchmarks
                job_id = f"{model_name}_{dataset}_daily"
                if self.add_daily_job(
                    job_id=job_id,
                    model_name=model_name,
                    dataset_name=dataset,
                    benchmark_config={'mode': 'full', 'sample_size': 1000},
                    hour=2
                ):
                    created_jobs.append(job_id)
                
                # Quick hourly checks (smaller sample)
                job_id = f"{model_name}_{dataset}_hourly"
                if self.add_interval_job(
                    job_id=job_id,
                    model_name=model_name,
                    dataset_name=dataset,
                    benchmark_config={'mode': 'quick', 'sample_size': 100},
                    hours=1
                ):
                    created_jobs.append(job_id)
        
        logger.info(f"Created monitoring suite with {len(created_jobs)} jobs")
        return created_jobs
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics and status."""
        return {
            'running': self._running,
            'total_jobs': len(self.scheduler.get_jobs()),
            'active_jobs': len([j for j in self.active_jobs.values() if j.enabled]),
            'paused_jobs': len([j for j in self.active_jobs.values() if not j.enabled]),
            'next_run_times': [
                {
                    'job_id': job.id,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in self.scheduler.get_jobs()
            ]
        }