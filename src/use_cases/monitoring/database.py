"""
Database management for the monitoring system.

This module provides database connection, initialization, and CRUD operations
for the monitoring system models.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager
from datetime import datetime, timedelta

from sqlalchemy import create_engine, event, func, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from .models import (
    Base, ModelMetadata, BenchmarkRun, PerformanceMetric,
    AlertHistory, AggregatedStats
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations for the monitoring system."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL. If None, uses environment variable
                         DATABASE_URL or defaults to SQLite.
        """
        if database_url is None:
            database_url = os.getenv(
                'DATABASE_URL', 
                'sqlite:///monitoring.db'
            )
        
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600    # Recycle connections after 1 hour
        )
        
        # Add SQLite optimizations if using SQLite
        if database_url.startswith('sqlite'):
            @event.listens_for(Engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database manager initialized with URL: {self._mask_password(database_url)}")
    
    def _mask_password(self, url: str) -> str:
        """Mask password in database URL for logging."""
        if '://' in url and '@' in url:
            protocol, rest = url.split('://', 1)
            if '@' in rest:
                credentials, host_part = rest.rsplit('@', 1)
                if ':' in credentials:
                    user, _ = credentials.split(':', 1)
                    return f"{protocol}://{user}:***@{host_part}"
        return url
    
    def initialize_database(self, drop_existing: bool = False) -> None:
        """
        Initialize database tables.
        
        Args:
            drop_existing: If True, drop existing tables before creating new ones.
        """
        try:
            if drop_existing:
                logger.warning("Dropping existing tables")
                Base.metadata.drop_all(bind=self.engine)
            
            logger.info("Creating database tables")
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialization completed successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check database health and return status information.
        
        Returns:
            Dictionary with health check results.
        """
        try:
            with self.get_session() as session:
                # Test basic connectivity
                session.execute(func.count(ModelMetadata.id)).scalar()
                
                # Get table counts
                counts = {
                    'models': session.query(ModelMetadata).count(),
                    'benchmark_runs': session.query(BenchmarkRun).count(),
                    'performance_metrics': session.query(PerformanceMetric).count(),
                    'alert_history': session.query(AlertHistory).count(),
                    'aggregated_stats': session.query(AggregatedStats).count()
                }
                
                return {
                    'status': 'healthy',
                    'database_url': self._mask_password(self.database_url),
                    'table_counts': counts,
                    'checked_at': datetime.utcnow().isoformat()
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'checked_at': datetime.utcnow().isoformat()
            }
    
    # Model Metadata CRUD Operations
    def create_model(self, model_data: Dict[str, Any]) -> ModelMetadata:
        """Create a new model metadata record."""
        with self.get_session() as session:
            model = ModelMetadata(**model_data)
            session.add(model)
            session.flush()
            session.refresh(model)
            return model
    
    def get_model(self, model_id: int) -> Optional[ModelMetadata]:
        """Get model by ID."""
        with self.get_session() as session:
            return session.query(ModelMetadata).filter(
                ModelMetadata.id == model_id
            ).first()
    
    def get_model_by_name(self, name: str) -> Optional[ModelMetadata]:
        """Get model by name."""
        with self.get_session() as session:
            return session.query(ModelMetadata).filter(
                ModelMetadata.name == name
            ).first()
    
    def list_models(self, active_only: bool = True) -> List[ModelMetadata]:
        """List all models, optionally filtering to active ones."""
        with self.get_session() as session:
            query = session.query(ModelMetadata)
            if active_only:
                query = query.filter(ModelMetadata.active == True)
            return query.order_by(ModelMetadata.name).all()
    
    def update_model(self, model_id: int, updates: Dict[str, Any]) -> Optional[ModelMetadata]:
        """Update model metadata."""
        with self.get_session() as session:
            model = session.query(ModelMetadata).filter(
                ModelMetadata.id == model_id
            ).first()
            if model:
                for key, value in updates.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                model.updated_at = datetime.utcnow()
                session.flush()
                session.refresh(model)
            return model
    
    # Benchmark Run CRUD Operations
    def create_benchmark_run(self, run_data: Dict[str, Any]) -> BenchmarkRun:
        """Create a new benchmark run record."""
        with self.get_session() as session:
            run = BenchmarkRun(**run_data)
            session.add(run)
            session.flush()
            session.refresh(run)
            return run
    
    def get_benchmark_run(self, run_id: int) -> Optional[BenchmarkRun]:
        """Get benchmark run by ID."""
        with self.get_session() as session:
            return session.query(BenchmarkRun).filter(
                BenchmarkRun.id == run_id
            ).first()
    
    def list_benchmark_runs(
        self,
        model_id: Optional[int] = None,
        dataset_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[BenchmarkRun]:
        """List benchmark runs with optional filtering."""
        with self.get_session() as session:
            query = session.query(BenchmarkRun)
            
            if model_id:
                query = query.filter(BenchmarkRun.model_id == model_id)
            if dataset_name:
                query = query.filter(BenchmarkRun.dataset_name == dataset_name)
            if status:
                query = query.filter(BenchmarkRun.status == status)
            
            return query.order_by(BenchmarkRun.timestamp.desc()).offset(offset).limit(limit).all()
    
    def update_benchmark_run(self, run_id: int, updates: Dict[str, Any]) -> Optional[BenchmarkRun]:
        """Update benchmark run."""
        with self.get_session() as session:
            run = session.query(BenchmarkRun).filter(
                BenchmarkRun.id == run_id
            ).first()
            if run:
                for key, value in updates.items():
                    if hasattr(run, key):
                        setattr(run, key, value)
                session.flush()
                session.refresh(run)
            return run
    
    def complete_benchmark_run(
        self,
        run_id: int,
        status: str = 'completed',
        error_message: Optional[str] = None
    ) -> Optional[BenchmarkRun]:
        """Mark a benchmark run as completed or failed."""
        updates = {
            'status': status,
            'completed_at': datetime.utcnow()
        }
        if error_message:
            updates['error_message'] = error_message
        
        return self.update_benchmark_run(run_id, updates)
    
    # Performance Metric CRUD Operations
    def create_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Create a new performance metric record."""
        with self.get_session() as session:
            metric = PerformanceMetric(**metric_data)
            session.add(metric)
            session.flush()
            session.refresh(metric)
            return metric
    
    def create_metrics_batch(self, metrics_data: List[Dict[str, Any]]) -> List[PerformanceMetric]:
        """Create multiple performance metrics in a batch."""
        with self.get_session() as session:
            metrics = [PerformanceMetric(**data) for data in metrics_data]
            session.add_all(metrics)
            session.flush()
            for metric in metrics:
                session.refresh(metric)
            return metrics
    
    def get_metrics_for_run(self, run_id: int) -> List[PerformanceMetric]:
        """Get all metrics for a specific benchmark run."""
        with self.get_session() as session:
            return session.query(PerformanceMetric).filter(
                PerformanceMetric.run_id == run_id
            ).order_by(PerformanceMetric.metric_type, PerformanceMetric.metric_name).all()
    
    def get_metric_history(
        self,
        model_id: int,
        metric_type: str,
        metric_name: str,
        days: int = 30
    ) -> List[PerformanceMetric]:
        """Get historical metrics for trend analysis."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            return session.query(PerformanceMetric).join(BenchmarkRun).filter(
                and_(
                    BenchmarkRun.model_id == model_id,
                    PerformanceMetric.metric_type == metric_type,
                    PerformanceMetric.metric_name == metric_name,
                    PerformanceMetric.timestamp >= start_date
                )
            ).order_by(PerformanceMetric.timestamp).all()
    
    # Alert History CRUD Operations
    def create_alert(self, alert_data: Dict[str, Any]) -> AlertHistory:
        """Create a new alert record."""
        with self.get_session() as session:
            alert = AlertHistory(**alert_data)
            session.add(alert)
            session.flush()
            session.refresh(alert)
            return alert
    
    def get_active_alerts(self, model_id: Optional[int] = None) -> List[AlertHistory]:
        """Get active alerts, optionally filtered by model."""
        with self.get_session() as session:
            query = session.query(AlertHistory).filter(
                AlertHistory.status == 'active'
            )
            if model_id:
                query = query.filter(AlertHistory.model_id == model_id)
            
            return query.order_by(AlertHistory.timestamp.desc()).all()
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> Optional[AlertHistory]:
        """Acknowledge an alert."""
        updates = {
            'status': 'acknowledged',
            'acknowledged_at': datetime.utcnow(),
            'acknowledged_by': acknowledged_by
        }
        return self.update_alert(alert_id, updates)
    
    def resolve_alert(
        self,
        alert_id: int,
        resolved_by: str,
        resolution_notes: Optional[str] = None
    ) -> Optional[AlertHistory]:
        """Resolve an alert."""
        updates = {
            'status': 'resolved',
            'resolved_at': datetime.utcnow(),
            'resolved_by': resolved_by
        }
        if resolution_notes:
            updates['resolution_notes'] = resolution_notes
        
        return self.update_alert(alert_id, updates)
    
    def update_alert(self, alert_id: int, updates: Dict[str, Any]) -> Optional[AlertHistory]:
        """Update alert record."""
        with self.get_session() as session:
            alert = session.query(AlertHistory).filter(
                AlertHistory.id == alert_id
            ).first()
            if alert:
                for key, value in updates.items():
                    if hasattr(alert, key):
                        setattr(alert, key, value)
                session.flush()
                session.refresh(alert)
            return alert
    
    # Aggregated Stats CRUD Operations
    def create_aggregated_stats(self, stats_data: Dict[str, Any]) -> AggregatedStats:
        """Create aggregated statistics record."""
        with self.get_session() as session:
            stats = AggregatedStats(**stats_data)
            session.add(stats)
            session.flush()
            session.refresh(stats)
            return stats
    
    def get_aggregated_stats(
        self,
        model_id: Optional[int] = None,
        metric_type: Optional[str] = None,
        period_type: str = 'day',
        days: int = 30
    ) -> List[AggregatedStats]:
        """Get aggregated statistics with optional filtering."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            query = session.query(AggregatedStats).filter(
                and_(
                    AggregatedStats.period_type == period_type,
                    AggregatedStats.period_start >= start_date
                )
            )
            
            if model_id:
                query = query.filter(AggregatedStats.model_id == model_id)
            if metric_type:
                query = query.filter(AggregatedStats.metric_type == metric_type)
            
            return query.order_by(AggregatedStats.period_start).all()
    
    def upsert_aggregated_stats(self, stats_data: Dict[str, Any]) -> AggregatedStats:
        """Create or update aggregated statistics (upsert operation)."""
        with self.get_session() as session:
            # Try to find existing record
            existing = session.query(AggregatedStats).filter(
                and_(
                    AggregatedStats.period_start == stats_data['period_start'],
                    AggregatedStats.period_end == stats_data['period_end'],
                    AggregatedStats.model_id == stats_data.get('model_id'),
                    AggregatedStats.dataset_name == stats_data.get('dataset_name'),
                    AggregatedStats.metric_type == stats_data['metric_type'],
                    AggregatedStats.metric_name == stats_data['metric_name']
                )
            ).first()
            
            if existing:
                # Update existing record
                for key, value in stats_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
                session.flush()
                session.refresh(existing)
                return existing
            else:
                # Create new record
                stats = AggregatedStats(**stats_data)
                session.add(stats)
                session.flush()
                session.refresh(stats)
                return stats
    
    # Utility Methods
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Clean up old data beyond the retention period.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Dictionary with counts of deleted records
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        deleted_counts = {}
        
        with self.get_session() as session:
            # Delete old benchmark runs and their metrics (cascaded)
            runs_deleted = session.query(BenchmarkRun).filter(
                BenchmarkRun.timestamp < cutoff_date
            ).delete()
            deleted_counts['benchmark_runs'] = runs_deleted
            
            # Delete old resolved alerts
            alerts_deleted = session.query(AlertHistory).filter(
                and_(
                    AlertHistory.timestamp < cutoff_date,
                    AlertHistory.status == 'resolved'
                )
            ).delete()
            deleted_counts['alert_history'] = alerts_deleted
            
            # Keep aggregated stats longer (they're small)
            stats_cutoff = datetime.utcnow() - timedelta(days=days_to_keep * 2)
            stats_deleted = session.query(AggregatedStats).filter(
                AggregatedStats.created_at < stats_cutoff
            ).delete()
            deleted_counts['aggregated_stats'] = stats_deleted
        
        logger.info(f"Cleanup completed: {deleted_counts}")
        return deleted_counts
    
    def get_database_size(self) -> Dict[str, Any]:
        """Get database size information."""
        with self.get_session() as session:
            if self.database_url.startswith('sqlite'):
                # For SQLite, get file size
                db_path = self.database_url.replace('sqlite:///', '')
                if os.path.exists(db_path):
                    size_bytes = os.path.getsize(db_path)
                    return {
                        'size_bytes': size_bytes,
                        'size_mb': round(size_bytes / (1024 * 1024), 2),
                        'path': db_path
                    }
            else:
                # For other databases, this would need database-specific queries
                return {'size_info': 'Not available for this database type'}
        
        return {'size_info': 'Unable to determine'}