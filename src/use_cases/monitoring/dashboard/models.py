"""
Database Models for Dashboard

SQLAlchemy models for storing monitoring metrics, alerts, and dashboard data.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean, 
    JSON, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine, func
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()


class MetricSnapshot(Base):
    """
    Stores periodic snapshots of system metrics for time-series analysis.
    """
    __tablename__ = 'metric_snapshots'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    provider = Column(String(50), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    
    # Performance metrics
    requests_count = Column(Integer, default=0)
    avg_latency = Column(Float, default=0.0)
    success_rate = Column(Float, default=100.0)
    error_count = Column(Integer, default=0)
    
    # Cost metrics
    total_cost = Column(Float, default=0.0)
    cost_per_request = Column(Float, default=0.0)
    token_count = Column(Integer, default=0)
    
    # Quality metrics
    avg_quality_score = Column(Float, default=0.0)
    accuracy_score = Column(Float, default=0.0)
    coherence_score = Column(Float, default=0.0)
    
    # System metrics
    cpu_usage = Column(Float, default=0.0)
    memory_usage = Column(Float, default=0.0)
    
    # Additional metadata
    metadata_dict = Column(JSON, default={})
    
    __table_args__ = (
        Index('idx_provider_model_time', 'provider', 'model', 'timestamp'),
        Index('idx_timestamp', 'timestamp'),
    )
    
    @hybrid_property
    def metadata(self):
        """Get metadata as dictionary."""
        return self.metadata_dict or {}
    
    @metadata.setter
    def metadata(self, value):
        """Set metadata dictionary."""
        self.metadata_dict = value or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'provider': self.provider,
            'model': self.model,
            'requests_count': self.requests_count,
            'avg_latency': self.avg_latency,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'total_cost': self.total_cost,
            'cost_per_request': self.cost_per_request,
            'token_count': self.token_count,
            'avg_quality_score': self.avg_quality_score,
            'accuracy_score': self.accuracy_score,
            'coherence_score': self.coherence_score,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'metadata': self.metadata
        }


class Alert(Base):
    """
    Stores alerts and notifications from the monitoring system.
    """
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Alert identification
    alert_type = Column(String(50), nullable=False, index=True)  # performance, cost, error, etc.
    severity = Column(String(20), nullable=False, index=True)  # critical, warning, info
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Alert source
    provider = Column(String(50), index=True)
    model = Column(String(100), index=True)
    metric_name = Column(String(100), index=True)
    
    # Alert values
    current_value = Column(Float)
    threshold_value = Column(Float)
    threshold_operator = Column(String(10))  # >, <, >=, <=, ==, !=
    
    # Alert status
    status = Column(String(20), default='active', index=True)  # active, acknowledged, resolved
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    resolved_at = Column(DateTime)
    resolved_by = Column(String(100))
    
    # Additional data
    details = Column(JSON, default={})
    
    __table_args__ = (
        Index('idx_alert_type_severity', 'alert_type', 'severity'),
        Index('idx_provider_model', 'provider', 'model'),
        Index('idx_status_timestamp', 'status', 'timestamp'),
    )
    
    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == 'active'
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get alert duration."""
        if self.resolved_at:
            return self.resolved_at - self.timestamp
        return datetime.utcnow() - self.timestamp
    
    def acknowledge(self, user: str):
        """Acknowledge the alert."""
        self.status = 'acknowledged'
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = user
    
    def resolve(self, user: str):
        """Resolve the alert."""
        self.status = 'resolved'
        self.resolved_at = datetime.utcnow()
        self.resolved_by = user
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'title': self.title,
            'message': self.message,
            'provider': self.provider,
            'model': self.model,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'threshold_operator': self.threshold_operator,
            'status': self.status,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by,
            'duration_seconds': self.duration.total_seconds() if self.duration else None,
            'details': self.details or {}
        }


class RequestLog(Base):
    """
    Stores individual request logs for detailed analysis.
    """
    __tablename__ = 'request_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Request identification
    request_id = Column(String(100), unique=True, index=True)
    provider = Column(String(50), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    
    # Request details
    prompt_length = Column(Integer)
    response_length = Column(Integer)
    token_count = Column(Integer)
    
    # Performance metrics
    latency = Column(Float, nullable=False)
    success = Column(Boolean, default=True, index=True)
    error_message = Column(Text)
    
    # Cost information
    cost = Column(Float, default=0.0)
    
    # Quality scores
    quality_score = Column(Float)
    accuracy_score = Column(Float)
    coherence_score = Column(Float)
    
    # Additional metadata
    metadata_dict = Column(JSON, default={})
    
    __table_args__ = (
        Index('idx_provider_model_timestamp', 'provider', 'model', 'timestamp'),
        Index('idx_success_timestamp', 'success', 'timestamp'),
    )
    
    @hybrid_property
    def metadata(self):
        """Get metadata as dictionary."""
        return self.metadata_dict or {}
    
    @metadata.setter
    def metadata(self, value):
        """Set metadata dictionary."""
        self.metadata_dict = value or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'request_id': self.request_id,
            'provider': self.provider,
            'model': self.model,
            'prompt_length': self.prompt_length,
            'response_length': self.response_length,
            'token_count': self.token_count,
            'latency': self.latency,
            'success': self.success,
            'error_message': self.error_message,
            'cost': self.cost,
            'quality_score': self.quality_score,
            'accuracy_score': self.accuracy_score,
            'coherence_score': self.coherence_score,
            'metadata': self.metadata
        }


class DashboardConfig(Base):
    """
    Stores dashboard configuration and user preferences.
    """
    __tablename__ = 'dashboard_configs'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Configuration identification
    config_key = Column(String(100), unique=True, nullable=False, index=True)
    config_type = Column(String(50), nullable=False, index=True)  # user, system, default
    
    # Configuration data
    config_data = Column(JSON, nullable=False)
    
    # Metadata
    created_by = Column(String(100))
    description = Column(Text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'config_key': self.config_key,
            'config_type': self.config_type,
            'config_data': self.config_data,
            'created_by': self.created_by,
            'description': self.description
        }


@dataclass
class DatabaseManager:
    """
    Database manager for handling connections and operations.
    """
    database_url: str
    echo: bool = False
    
    def __post_init__(self):
        """Initialize database engine and session."""
        self.engine = create_engine(
            self.database_url,
            echo=self.echo,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary metrics for the dashboard."""
        session = self.get_session()
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            # Get latest snapshots
            latest_snapshots = session.query(MetricSnapshot)\
                .filter(MetricSnapshot.timestamp >= since)\
                .order_by(MetricSnapshot.timestamp.desc())\
                .limit(100).all()
            
            if not latest_snapshots:
                return {
                    'total_models': 0,
                    'total_requests': 0,
                    'avg_latency': 0.0,
                    'total_cost': 0.0,
                    'uptime': 0,
                    'last_updated': datetime.utcnow().isoformat()
                }
            
            # Calculate summary metrics
            unique_models = len(set((s.provider, s.model) for s in latest_snapshots))
            total_requests = sum(s.requests_count for s in latest_snapshots)
            avg_latency = sum(s.avg_latency for s in latest_snapshots) / len(latest_snapshots)
            total_cost = sum(s.total_cost for s in latest_snapshots)
            
            # Get active alerts count
            active_alerts = session.query(Alert)\
                .filter(Alert.status == 'active')\
                .count()
            
            return {
                'total_models': unique_models,
                'total_requests': total_requests,
                'avg_latency': avg_latency,
                'total_cost': total_cost,
                'active_alerts': active_alerts,
                'uptime': hours * 3600,  # Mock uptime
                'last_updated': datetime.utcnow().isoformat()
            }
            
        finally:
            session.close()
    
    def get_performance_data(self, hours: int = 24, provider: str = None, model: str = None) -> Dict[str, Any]:
        """Get performance data for charts."""
        session = self.get_session()
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            query = session.query(MetricSnapshot)\
                .filter(MetricSnapshot.timestamp >= since)\
                .order_by(MetricSnapshot.timestamp.asc())
            
            if provider:
                query = query.filter(MetricSnapshot.provider == provider)
            if model:
                query = query.filter(MetricSnapshot.model == model)
            
            snapshots = query.all()
            
            return {
                'time_series': [s.to_dict() for s in snapshots],
                'providers': list(set(s.provider for s in snapshots)),
                'models': list(set(f"{s.provider}/{s.model}" for s in snapshots))
            }
            
        finally:
            session.close()
    
    def get_cost_breakdown(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost breakdown by provider."""
        session = self.get_session()
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            # Aggregate costs by provider
            cost_data = session.query(
                MetricSnapshot.provider,
                func.sum(MetricSnapshot.total_cost).label('total_cost')
            ).filter(
                MetricSnapshot.timestamp >= since
            ).group_by(
                MetricSnapshot.provider
            ).all()
            
            provider_costs = {provider: float(cost) for provider, cost in cost_data}
            
            # Get daily costs for trends
            daily_costs = session.query(
                func.date(MetricSnapshot.timestamp).label('date'),
                func.sum(MetricSnapshot.total_cost).label('total_cost')
            ).filter(
                MetricSnapshot.timestamp >= since
            ).group_by(
                func.date(MetricSnapshot.timestamp)
            ).all()
            
            daily_cost_data = [
                {'date': str(date), 'cost': float(cost)}
                for date, cost in daily_costs
            ]
            
            return {
                'provider_breakdown': provider_costs,
                'daily_costs': daily_cost_data,
                'total_cost': sum(provider_costs.values())
            }
            
        finally:
            session.close()
    
    def get_active_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get active alerts."""
        session = self.get_session()
        try:
            alerts = session.query(Alert)\
                .filter(Alert.status.in_(['active', 'acknowledged']))\
                .order_by(Alert.timestamp.desc())\
                .limit(limit).all()
            
            return [alert.to_dict() for alert in alerts]
            
        finally:
            session.close()
    
    def add_metric_snapshot(self, **kwargs) -> MetricSnapshot:
        """Add a new metric snapshot."""
        session = self.get_session()
        try:
            snapshot = MetricSnapshot(**kwargs)
            session.add(snapshot)
            session.commit()
            return snapshot
        finally:
            session.close()
    
    def add_alert(self, **kwargs) -> Alert:
        """Add a new alert."""
        session = self.get_session()
        try:
            alert = Alert(**kwargs)
            session.add(alert)
            session.commit()
            return alert
        finally:
            session.close()
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data based on retention policy."""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean up old metric snapshots
            deleted_snapshots = session.query(MetricSnapshot)\
                .filter(MetricSnapshot.timestamp < cutoff_date)\
                .delete()
            
            # Clean up old request logs
            deleted_logs = session.query(RequestLog)\
                .filter(RequestLog.timestamp < cutoff_date)\
                .delete()
            
            # Clean up resolved alerts older than retention period
            deleted_alerts = session.query(Alert)\
                .filter(Alert.timestamp < cutoff_date)\
                .filter(Alert.status == 'resolved')\
                .delete()
            
            session.commit()
            
            return {
                'deleted_snapshots': deleted_snapshots,
                'deleted_logs': deleted_logs,
                'deleted_alerts': deleted_alerts
            }
            
        finally:
            session.close()


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global db_manager
    if db_manager is None:
        raise RuntimeError("Database manager not initialized. Call init_database() first.")
    return db_manager


def init_database(database_url: str, echo: bool = False) -> DatabaseManager:
    """Initialize the global database manager."""
    global db_manager
    db_manager = DatabaseManager(database_url, echo)
    db_manager.create_tables()
    return db_manager