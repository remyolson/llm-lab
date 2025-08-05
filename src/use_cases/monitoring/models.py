"""
Database models for the monitoring system.

This module defines SQLAlchemy models for storing benchmark results,
performance metrics, and monitoring data.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
import json

Base = declarative_base()


class ModelMetadata(Base):
    """Metadata about models being monitored."""
    
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    version = Column(String(100))
    provider = Column(String(100), nullable=False)
    cost_per_input_token = Column(Float)
    cost_per_output_token = Column(Float)
    max_tokens = Column(Integer)
    context_length = Column(Integer)
    model_type = Column(String(50))  # e.g., "chat", "completion", "embedding"
    capabilities = Column(JSON)  # List of capabilities
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    benchmark_runs = relationship("BenchmarkRun", back_populates="model")
    
    # Indexes
    __table_args__ = (
        Index("idx_model_provider", "provider"),
        Index("idx_model_active", "active"),
        Index("idx_model_type", "model_type"),
    )
    
    @validates('name')
    def validate_name(self, key, name):
        if not name or len(name.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return name.strip()
    
    @validates('provider')
    def validate_provider(self, key, provider):
        valid_providers = ['openai', 'anthropic', 'google', 'local', 'custom']
        if provider not in valid_providers:
            raise ValueError(f"Provider must be one of: {valid_providers}")
        return provider
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'provider': self.provider,
            'cost_per_input_token': self.cost_per_input_token,
            'cost_per_output_token': self.cost_per_output_token,
            'max_tokens': self.max_tokens,
            'context_length': self.context_length,
            'model_type': self.model_type,
            'capabilities': self.capabilities,
            'active': self.active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class BenchmarkRun(Base):
    """Records of benchmark execution runs."""
    
    __tablename__ = "benchmark_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_id = Column(Integer, ForeignKey('model_metadata.id'), nullable=False)
    dataset_name = Column(String(100), nullable=False)
    dataset_version = Column(String(50))
    
    # Run configuration
    run_config = Column(JSON)  # Configuration used for the run
    
    # Overall metrics (aggregated from individual metrics)
    accuracy = Column(Float)
    latency_ms = Column(Float)
    total_cost = Column(Float)
    total_tokens_input = Column(Integer)
    total_tokens_output = Column(Integer)
    
    # Run metadata
    status = Column(String(20), default='running')  # running, completed, failed
    error_message = Column(Text)
    duration_seconds = Column(Float)
    sample_count = Column(Integer)
    
    # Scheduling info
    scheduled_job_id = Column(String(100))  # APScheduler job ID
    trigger_type = Column(String(50))  # manual, scheduled, alert
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    model = relationship("ModelMetadata", back_populates="benchmark_runs")
    metrics = relationship("PerformanceMetric", back_populates="run", cascade="all, delete-orphan")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_run_timestamp", "timestamp"),
        Index("idx_run_model_dataset", "model_id", "dataset_name"),
        Index("idx_run_status", "status"),
        Index("idx_run_scheduled_job", "scheduled_job_id"),
        CheckConstraint("status IN ('running', 'completed', 'failed')", name="valid_status"),
    )
    
    @validates('status')
    def validate_status(self, key, status):
        valid_statuses = ['running', 'completed', 'failed']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert run to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'model_id': self.model_id,
            'dataset_name': self.dataset_name,
            'dataset_version': self.dataset_version,
            'run_config': self.run_config,
            'accuracy': self.accuracy,
            'latency_ms': self.latency_ms,
            'total_cost': self.total_cost,
            'total_tokens_input': self.total_tokens_input,
            'total_tokens_output': self.total_tokens_output,
            'status': self.status,
            'error_message': self.error_message,
            'duration_seconds': self.duration_seconds,
            'sample_count': self.sample_count,
            'scheduled_job_id': self.scheduled_job_id,
            'trigger_type': self.trigger_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class PerformanceMetric(Base):
    """Individual performance metrics from benchmark runs."""
    
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('benchmark_runs.id'), nullable=False)
    metric_type = Column(String(100), nullable=False)
    metric_name = Column(String(200), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(50))  # e.g., "percentage", "seconds", "tokens"
    category = Column(String(100))  # e.g., "accuracy", "performance", "cost"
    sample_id = Column(String(100))  # ID of specific sample if applicable
    metadata = Column(JSON)  # Additional metric-specific data
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run = relationship("BenchmarkRun", back_populates="metrics")
    
    # Indexes
    __table_args__ = (
        Index("idx_metric_run_type", "run_id", "metric_type"),
        Index("idx_metric_name", "metric_name"),
        Index("idx_metric_category", "category"),
        Index("idx_metric_timestamp", "timestamp"),
        UniqueConstraint("run_id", "metric_type", "metric_name", "sample_id", 
                        name="unique_metric_per_sample"),
    )
    
    @validates('metric_type')
    def validate_metric_type(self, key, metric_type):
        valid_types = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'latency', 'throughput', 'memory_usage',
            'cost', 'token_count', 'error_rate',
            'custom'
        ]
        if metric_type not in valid_types:
            raise ValueError(f"Metric type must be one of: {valid_types}")
        return metric_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'id': self.id,
            'run_id': self.run_id,
            'metric_type': self.metric_type,
            'metric_name': self.metric_name,
            'value': self.value,
            'unit': self.unit,
            'category': self.category,
            'sample_id': self.sample_id,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class AlertHistory(Base):
    """History of alerts generated by the monitoring system."""
    
    __tablename__ = "alert_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)
    
    # Related entities
    model_id = Column(Integer, ForeignKey('model_metadata.id'))
    run_id = Column(Integer, ForeignKey('benchmark_runs.id'))
    metric_type = Column(String(100))
    
    # Alert data
    trigger_value = Column(Float)
    threshold_value = Column(Float)
    baseline_value = Column(Float)
    
    # Alert status
    status = Column(String(20), default='active')  # active, acknowledged, resolved
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    resolved_at = Column(DateTime)
    resolved_by = Column(String(100))
    resolution_notes = Column(Text)
    
    # Notification tracking
    notification_channels = Column(JSON)  # List of channels notified
    notification_status = Column(JSON)  # Status of each notification
    
    # Relationships
    model = relationship("ModelMetadata")
    run = relationship("BenchmarkRun")
    
    # Indexes
    __table_args__ = (
        Index("idx_alert_timestamp", "timestamp"),
        Index("idx_alert_type_severity", "alert_type", "severity"),
        Index("idx_alert_model", "model_id"),
        Index("idx_alert_status", "status"),
        CheckConstraint("severity IN ('critical', 'warning', 'info')", name="valid_severity"),
        CheckConstraint("status IN ('active', 'acknowledged', 'resolved')", name="valid_alert_status"),
    )
    
    @validates('severity')
    def validate_severity(self, key, severity):
        valid_severities = ['critical', 'warning', 'info']
        if severity not in valid_severities:
            raise ValueError(f"Severity must be one of: {valid_severities}")
        return severity
    
    @validates('status')
    def validate_alert_status(self, key, status):
        valid_statuses = ['active', 'acknowledged', 'resolved']
        if status not in valid_statuses:
            raise ValueError(f"Alert status must be one of: {valid_statuses}")
        return status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'title': self.title,
            'message': self.message,
            'model_id': self.model_id,
            'run_id': self.run_id,
            'metric_type': self.metric_type,
            'trigger_value': self.trigger_value,
            'threshold_value': self.threshold_value,
            'baseline_value': self.baseline_value,
            'status': self.status,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by,
            'resolution_notes': self.resolution_notes,
            'notification_channels': self.notification_channels,
            'notification_status': self.notification_status
        }


class AggregatedStats(Base):
    """Pre-computed aggregated statistics for performance analysis."""
    
    __tablename__ = "aggregated_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20), nullable=False)  # hour, day, week, month
    
    # Aggregation dimensions
    model_id = Column(Integer, ForeignKey('model_metadata.id'))
    dataset_name = Column(String(100))
    metric_type = Column(String(100), nullable=False)
    metric_name = Column(String(200), nullable=False)
    
    # Aggregated values
    count = Column(Integer, nullable=False)
    avg_value = Column(Float, nullable=False)
    min_value = Column(Float, nullable=False)
    max_value = Column(Float, nullable=False)
    std_deviation = Column(Float)
    percentile_25 = Column(Float)
    percentile_50 = Column(Float)  # median
    percentile_75 = Column(Float)
    percentile_95 = Column(Float)
    
    # Trend indicators
    trend_direction = Column(String(10))  # up, down, stable
    trend_strength = Column(Float)  # 0-1 scale
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    model = relationship("ModelMetadata")
    
    # Indexes
    __table_args__ = (
        Index("idx_stats_period", "period_start", "period_end"),
        Index("idx_stats_model_metric", "model_id", "metric_type", "metric_name"),
        Index("idx_stats_dataset", "dataset_name"),
        Index("idx_stats_period_type", "period_type"),
        UniqueConstraint("period_start", "period_end", "model_id", "dataset_name", 
                        "metric_type", "metric_name", name="unique_aggregation"),
        CheckConstraint("period_type IN ('hour', 'day', 'week', 'month')", 
                       name="valid_period_type"),
        CheckConstraint("trend_direction IN ('up', 'down', 'stable')", 
                       name="valid_trend_direction"),
    )
    
    @validates('period_type')
    def validate_period_type(self, key, period_type):
        valid_types = ['hour', 'day', 'week', 'month']
        if period_type not in valid_types:
            raise ValueError(f"Period type must be one of: {valid_types}")
        return period_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'id': self.id,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
            'period_type': self.period_type,
            'model_id': self.model_id,
            'dataset_name': self.dataset_name,
            'metric_type': self.metric_type,
            'metric_name': self.metric_name,
            'count': self.count,
            'avg_value': self.avg_value,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'std_deviation': self.std_deviation,
            'percentile_25': self.percentile_25,
            'percentile_50': self.percentile_50,
            'percentile_75': self.percentile_75,
            'percentile_95': self.percentile_95,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }