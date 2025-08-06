"""
RESTful API for the monitoring dashboard.

This module provides a comprehensive REST API for accessing monitoring data,
managing alerts, viewing performance metrics, and controlling the monitoring
system through a web dashboard or external integrations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Path, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .alerting import AlertManager
from .database import DatabaseManager
from .regression_detector import RegressionDetector
from .scheduler import BenchmarkJobConfig, BenchmarkScheduler, ScheduleType

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class ModelMetadataResponse(BaseModel):
    id: int
    name: str
    version: Optional[str]
    provider: str
    cost_per_input_token: Optional[float]
    cost_per_output_token: Optional[float]
    max_tokens: Optional[int]
    context_length: Optional[int]
    model_type: Optional[str]
    capabilities: Optional[Dict[str, Any]]
    active: bool
    created_at: str
    updated_at: str


class BenchmarkRunResponse(BaseModel):
    id: int
    timestamp: str
    model_id: int
    dataset_name: str
    dataset_version: Optional[str]
    run_config: Optional[Dict[str, Any]]
    accuracy: Optional[float]
    latency_ms: Optional[float]
    total_cost: Optional[float]
    total_tokens_input: Optional[int]
    total_tokens_output: Optional[int]
    status: str
    error_message: Optional[str]
    duration_seconds: Optional[float]
    sample_count: Optional[int]
    scheduled_job_id: Optional[str]
    trigger_type: Optional[str]
    created_at: str
    completed_at: Optional[str]


class PerformanceMetricResponse(BaseModel):
    id: int
    run_id: int
    metric_type: str
    metric_name: str
    value: float
    unit: Optional[str]
    category: Optional[str]
    sample_id: Optional[str]
    metadata: Optional[Dict[str, Any]]
    timestamp: str


class AlertResponse(BaseModel):
    id: int
    timestamp: str
    alert_type: str
    severity: str
    title: str
    message: str
    model_id: Optional[int]
    run_id: Optional[int]
    metric_type: Optional[str]
    trigger_value: Optional[float]
    threshold_value: Optional[float]
    baseline_value: Optional[float]
    status: str
    acknowledged_at: Optional[str]
    acknowledged_by: Optional[str]
    resolved_at: Optional[str]
    resolved_by: Optional[str]
    resolution_notes: Optional[str]
    notification_channels: Optional[List[str]]
    notification_status: Optional[Dict[str, Any]]


class CreateModelRequest(BaseModel):
    name: str = Field(..., description="Model name")
    version: Optional[str] = Field(None, description="Model version")
    provider: str = Field(..., description="Provider name")
    cost_per_input_token: Optional[float] = None
    cost_per_output_token: Optional[float] = None
    max_tokens: Optional[int] = None
    context_length: Optional[int] = None
    model_type: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None


class CreateBenchmarkJobRequest(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    model_name: str = Field(..., description="Model to benchmark")
    dataset_name: str = Field(..., description="Dataset to use")
    benchmark_config: Dict[str, Any] = Field(..., description="Benchmark configuration")
    schedule_type: str = Field(..., description="Schedule type: interval, cron, or date")
    schedule_params: Dict[str, Any] = Field(..., description="Schedule parameters")
    enabled: bool = Field(True, description="Whether job is enabled")
    max_instances: int = Field(1, description="Maximum concurrent instances")
    coalesce: bool = Field(True, description="Whether to coalesce missed runs")
    misfire_grace_time: int = Field(300, description="Misfire grace time in seconds")


class UpdateAlertRequest(BaseModel):
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    status: Optional[str] = None


class HealthCheckResponse(BaseModel):
    status: str
    database: Dict[str, Any]
    scheduler: Optional[Dict[str, Any]]
    components: Dict[str, str]
    timestamp: str


class MonitoringAPI:
    """RESTful API for the monitoring system."""

    def __init__(
        self,
        database_manager: DatabaseManager,
        scheduler: Optional[BenchmarkScheduler] = None,
        regression_detector: Optional[RegressionDetector] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        """
        Initialize monitoring API.

        Args:
            database_manager: Database manager instance
            scheduler: Benchmark scheduler instance
            regression_detector: Regression detector instance
            alert_manager: Alert manager instance
        """
        self.db_manager = database_manager
        self.scheduler = scheduler
        self.regression_detector = regression_detector
        self.alert_manager = alert_manager

        # Initialize FastAPI app
        self.app = FastAPI(
            title="LLM Lab Monitoring API",
            description="REST API for LLM Lab monitoring system",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Set up routes
        self._setup_routes()

        logger.info("Monitoring API initialized")

    def _setup_routes(self):
        """Set up API routes."""

        # Health check
        @self.app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Get system health status."""
            db_health = self.db_manager.health_check()
            scheduler_health = self.scheduler.get_scheduler_stats() if self.scheduler else None

            components = {
                "database": "healthy" if db_health["status"] == "healthy" else "unhealthy",
                "scheduler": "healthy"
                if scheduler_health and scheduler_health["running"]
                else "disabled",
                "regression_detector": "healthy" if self.regression_detector else "disabled",
                "alert_manager": "healthy" if self.alert_manager else "disabled",
            }

            overall_status = (
                "healthy"
                if all(status in ["healthy", "disabled"] for status in components.values())
                else "unhealthy"
            )

            return HealthCheckResponse(
                status=overall_status,
                database=db_health,
                scheduler=scheduler_health,
                components=components,
                timestamp=datetime.utcnow().isoformat(),
            )

        # Model Management
        @self.app.get("/models", response_model=List[ModelMetadataResponse])
        async def list_models(
            active_only: bool = Query(True, description="Show only active models"),
        ):
            """List all models."""
            models = self.db_manager.list_models(active_only=active_only)
            return [ModelMetadataResponse(**model.to_dict()) for model in models]

        @self.app.post(
            "/models", response_model=ModelMetadataResponse, status_code=status.HTTP_201_CREATED
        )
        async def create_model(model_request: CreateModelRequest):
            """Create a new model."""
            try:
                model = self.db_manager.create_model(model_request.dict())
                return ModelMetadataResponse(**model.to_dict())
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/models/{model_id}", response_model=ModelMetadataResponse)
        async def get_model(model_id: int = Path(..., description="Model ID")):
            """Get model by ID."""
            model = self.db_manager.get_model(model_id)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            return ModelMetadataResponse(**model.to_dict())

        @self.app.put("/models/{model_id}", response_model=ModelMetadataResponse)
        async def update_model(
            model_id: int = Path(..., description="Model ID"), updates: Dict[str, Any] = None
        ):
            """Update model metadata."""
            model = self.db_manager.update_model(model_id, updates or {})
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            return ModelMetadataResponse(**model.to_dict())

        # Benchmark Runs
        @self.app.get("/benchmark-runs", response_model=List[BenchmarkRunResponse])
        async def list_benchmark_runs(
            model_id: Optional[int] = Query(None, description="Filter by model ID"),
            dataset_name: Optional[str] = Query(None, description="Filter by dataset name"),
            status: Optional[str] = Query(None, description="Filter by status"),
            limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
            offset: int = Query(0, ge=0, description="Offset for pagination"),
        ):
            """List benchmark runs with optional filtering."""
            runs = self.db_manager.list_benchmark_runs(
                model_id=model_id,
                dataset_name=dataset_name,
                status=status,
                limit=limit,
                offset=offset,
            )
            return [BenchmarkRunResponse(**run.to_dict()) for run in runs]

        @self.app.get("/benchmark-runs/{run_id}", response_model=BenchmarkRunResponse)
        async def get_benchmark_run(run_id: int = Path(..., description="Benchmark run ID")):
            """Get benchmark run by ID."""
            run = self.db_manager.get_benchmark_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Benchmark run not found")
            return BenchmarkRunResponse(**run.to_dict())

        @self.app.get(
            "/benchmark-runs/{run_id}/metrics", response_model=List[PerformanceMetricResponse]
        )
        async def get_run_metrics(run_id: int = Path(..., description="Benchmark run ID")):
            """Get metrics for a benchmark run."""
            metrics = self.db_manager.get_metrics_for_run(run_id)
            return [PerformanceMetricResponse(**metric.to_dict()) for metric in metrics]

        # Performance Metrics
        @self.app.get("/metrics/history")
        async def get_metric_history(
            model_id: int = Query(..., description="Model ID"),
            metric_type: str = Query(..., description="Metric type"),
            metric_name: str = Query(..., description="Metric name"),
            days: int = Query(30, ge=1, le=365, description="Number of days of history"),
        ):
            """Get metric history for trend analysis."""
            metrics = self.db_manager.get_metric_history(model_id, metric_type, metric_name, days)
            return [PerformanceMetricResponse(**metric.to_dict()) for metric in metrics]

        # Job Scheduling
        if self.scheduler:

            @self.app.get("/scheduler/jobs")
            async def list_scheduled_jobs():
                """List all scheduled jobs."""
                return self.scheduler.list_jobs()

            @self.app.post("/scheduler/jobs", status_code=status.HTTP_201_CREATED)
            async def create_scheduled_job(job_request: CreateBenchmarkJobRequest):
                """Create a new scheduled benchmark job."""
                try:
                    job_config = BenchmarkJobConfig(
                        job_id=job_request.job_id,
                        model_name=job_request.model_name,
                        dataset_name=job_request.dataset_name,
                        benchmark_config=job_request.benchmark_config,
                        schedule_type=ScheduleType(job_request.schedule_type),
                        schedule_params=job_request.schedule_params,
                        enabled=job_request.enabled,
                        max_instances=job_request.max_instances,
                        coalesce=job_request.coalesce,
                        misfire_grace_time=job_request.misfire_grace_time,
                    )

                    success = self.scheduler.add_job(job_config)
                    if not success:
                        raise HTTPException(status_code=400, detail="Failed to create job")

                    return {"message": "Job created successfully", "job_id": job_request.job_id}

                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))

            @self.app.get("/scheduler/jobs/{job_id}")
            async def get_job_status(job_id: str = Path(..., description="Job ID")):
                """Get job status."""
                status_info = self.scheduler.get_job_status(job_id)
                if not status_info:
                    raise HTTPException(status_code=404, detail="Job not found")
                return status_info

            @self.app.delete("/scheduler/jobs/{job_id}")
            async def remove_scheduled_job(job_id: str = Path(..., description="Job ID")):
                """Remove a scheduled job."""
                success = self.scheduler.remove_job(job_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Job not found")
                return {"message": "Job removed successfully"}

            @self.app.post("/scheduler/jobs/{job_id}/pause")
            async def pause_job(job_id: str = Path(..., description="Job ID")):
                """Pause a scheduled job."""
                success = self.scheduler.pause_job(job_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Job not found")
                return {"message": "Job paused successfully"}

            @self.app.post("/scheduler/jobs/{job_id}/resume")
            async def resume_job(job_id: str = Path(..., description="Job ID")):
                """Resume a paused job."""
                success = self.scheduler.resume_job(job_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Job not found")
                return {"message": "Job resumed successfully"}

            @self.app.get("/scheduler/stats")
            async def get_scheduler_stats():
                """Get scheduler statistics."""
                return self.scheduler.get_scheduler_stats()

        # Regression Detection
        if self.regression_detector:

            @self.app.post("/regression-detection/analyze/{model_id}")
            async def analyze_model_regressions(
                model_id: int = Path(..., description="Model ID"),
                days_back: int = Query(30, ge=1, le=365, description="Days to analyze"),
            ):
                """Analyze model for performance regressions."""
                try:
                    results = self.regression_detector.detect_regressions(
                        model_id, days_back=days_back
                    )

                    return {
                        "model_id": model_id,
                        "analysis_period_days": days_back,
                        "regressions_detected": len([r for r in results if r.regression_detected]),
                        "total_metrics_analyzed": len(results),
                        "results": [
                            {
                                "metric_type": r.metric_type,
                                "metric_name": r.metric_name,
                                "regression_detected": r.regression_detected,
                                "severity": r.severity,
                                "confidence_score": r.confidence_score,
                                "baseline_value": r.baseline_value,
                                "current_value": r.current_value,
                                "change_percent": r.change_percent,
                                "detection_method": r.detection_method.value,
                                "timestamp": r.timestamp.isoformat(),
                            }
                            for r in results
                        ],
                    }

                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @self.app.get("/regression-detection/trends/{model_id}")
            async def get_model_trends(
                model_id: int = Path(..., description="Model ID"),
                days_back: int = Query(30, ge=1, le=365, description="Days to analyze"),
            ):
                """Get model performance trends."""
                try:
                    trends = self.regression_detector.analyze_model_trends(model_id, days_back)
                    return trends
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @self.app.get("/regression-detection/summary")
            async def get_regression_summary(
                model_ids: Optional[List[int]] = Query(None, description="Model IDs to analyze"),
                days_back: int = Query(7, ge=1, le=365, description="Days to analyze"),
            ):
                """Get regression summary across models."""
                try:
                    summary = self.regression_detector.get_regression_summary(model_ids, days_back)
                    return summary
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

        # Alert Management
        if self.alert_manager:

            @self.app.get("/alerts", response_model=List[AlertResponse])
            async def get_alerts(
                model_id: Optional[int] = Query(None, description="Filter by model ID"),
                active_only: bool = Query(True, description="Show only active alerts"),
            ):
                """Get alerts."""
                if active_only:
                    alerts = self.alert_manager.get_active_alerts(model_id=model_id)
                else:
                    # Get all alerts from database
                    with self.db_manager.get_session() as session:
                        from .models import AlertHistory

                        query = session.query(AlertHistory)
                        if model_id:
                            query = query.filter(AlertHistory.model_id == model_id)
                        alerts = [
                            alert.to_dict()
                            for alert in query.order_by(AlertHistory.timestamp.desc())
                            .limit(100)
                            .all()
                        ]

                return [AlertResponse(**alert) for alert in alerts]

            @self.app.put("/alerts/{alert_id}")
            async def update_alert(
                alert_id: int = Path(..., description="Alert ID"),
                update_request: UpdateAlertRequest = None,
            ):
                """Update alert status."""
                try:
                    if update_request.status == "acknowledged" and update_request.acknowledged_by:
                        success = self.alert_manager.acknowledge_alert(
                            str(alert_id), update_request.acknowledged_by
                        )
                    elif update_request.status == "resolved" and update_request.resolved_by:
                        success = self.alert_manager.resolve_alert(
                            str(alert_id),
                            update_request.resolved_by,
                            update_request.resolution_notes,
                        )
                    else:
                        raise HTTPException(status_code=400, detail="Invalid update request")

                    if not success:
                        raise HTTPException(status_code=404, detail="Alert not found")

                    return {"message": "Alert updated successfully"}

                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @self.app.get("/alerts/statistics")
            async def get_alert_statistics(
                days_back: int = Query(7, ge=1, le=365, description="Days to analyze"),
            ):
                """Get alert statistics."""
                try:
                    stats = self.alert_manager.get_alert_statistics(days_back)
                    return stats
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

        # Dashboard Data Aggregation
        @self.app.get("/dashboard/overview")
        async def get_dashboard_overview():
            """Get dashboard overview data."""
            try:
                # Get basic counts
                models = self.db_manager.list_models(active_only=True)
                recent_runs = self.db_manager.list_benchmark_runs(limit=50)

                # Calculate basic stats
                total_models = len(models)
                recent_runs_count = len(recent_runs)
                failed_runs = len([r for r in recent_runs if r.status == "failed"])

                overview = {
                    "total_active_models": total_models,
                    "recent_benchmark_runs": recent_runs_count,
                    "failed_runs_percentage": (failed_runs / recent_runs_count * 100)
                    if recent_runs_count > 0
                    else 0,
                    "system_health": "healthy",  # Simplified for now
                }

                # Add scheduler info if available
                if self.scheduler:
                    scheduler_stats = self.scheduler.get_scheduler_stats()
                    overview["scheduled_jobs"] = {
                        "total": scheduler_stats["total_jobs"],
                        "active": scheduler_stats["active_jobs"],
                        "paused": scheduler_stats["paused_jobs"],
                    }

                # Add alert info if available
                if self.alert_manager:
                    alert_stats = self.alert_manager.get_alert_statistics(days_back=7)
                    overview["alerts"] = {
                        "total_last_week": alert_stats["total_alerts"],
                        "active": alert_stats["active_alerts"],
                        "resolution_rate": alert_stats["resolution_rate"],
                    }

                return overview

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/dashboard/model-performance/{model_id}")
        async def get_model_performance_dashboard(
            model_id: int = Path(..., description="Model ID"),
            days_back: int = Query(30, ge=1, le=365, description="Days to analyze"),
        ):
            """Get model performance dashboard data."""
            try:
                # Get recent runs
                runs = self.db_manager.list_benchmark_runs(model_id=model_id, limit=100)
                recent_runs = [
                    r for r in runs if (datetime.utcnow() - r.timestamp).days <= days_back
                ]

                if not recent_runs:
                    return {"error": "No recent benchmark runs found"}

                # Calculate performance metrics
                accuracy_values = [r.accuracy for r in recent_runs if r.accuracy is not None]
                latency_values = [r.latency_ms for r in recent_runs if r.latency_ms is not None]
                cost_values = [r.total_cost for r in recent_runs if r.total_cost is not None]

                dashboard_data = {
                    "model_id": model_id,
                    "period_days": days_back,
                    "total_runs": len(recent_runs),
                    "success_rate": len([r for r in recent_runs if r.status == "completed"])
                    / len(recent_runs),
                    "performance": {
                        "avg_accuracy": sum(accuracy_values) / len(accuracy_values)
                        if accuracy_values
                        else None,
                        "avg_latency_ms": sum(latency_values) / len(latency_values)
                        if latency_values
                        else None,
                        "total_cost": sum(cost_values) if cost_values else None,
                    },
                    "recent_runs": [
                        {
                            "id": r.id,
                            "timestamp": r.timestamp.isoformat(),
                            "dataset_name": r.dataset_name,
                            "status": r.status,
                            "accuracy": r.accuracy,
                            "latency_ms": r.latency_ms,
                            "total_cost": r.total_cost,
                        }
                        for r in recent_runs[:10]  # Last 10 runs
                    ],
                }

                # Add trend analysis if regression detector is available
                if self.regression_detector:
                    trends = self.regression_detector.analyze_model_trends(model_id, days_back)
                    dashboard_data["trends"] = trends.get("metric_trends", {})

                return dashboard_data

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def start_server(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Start the API server."""
        config = uvicorn.Config(app=self.app, host=host, port=port, reload=reload, log_level="info")
        server = uvicorn.Server(config)

        logger.info(f"Starting monitoring API server on {host}:{port}")
        await server.serve()

    def run_server(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the API server (blocking)."""
        uvicorn.run(app=self.app, host=host, port=port, reload=reload, log_level="info")


# Factory function for creating the API with all components
def create_monitoring_api(
    database_url: Optional[str] = None,
    enable_scheduler: bool = True,
    enable_regression_detection: bool = True,
    enable_alerting: bool = True,
) -> MonitoringAPI:
    """
    Create a complete monitoring API with all components.

    Args:
        database_url: Database connection URL
        enable_scheduler: Whether to enable job scheduling
        enable_regression_detection: Whether to enable regression detection
        enable_alerting: Whether to enable alerting

    Returns:
        Configured MonitoringAPI instance
    """
    # Initialize database manager
    db_manager = DatabaseManager(database_url)
    db_manager.initialize_database()

    # Initialize optional components
    scheduler = None
    regression_detector = None
    alert_manager = None

    if enable_scheduler:
        scheduler = BenchmarkScheduler(db_manager)

    if enable_regression_detection:
        regression_detector = RegressionDetector(db_manager)

    if enable_alerting:
        alert_manager = AlertManager(db_manager)

    return MonitoringAPI(
        database_manager=db_manager,
        scheduler=scheduler,
        regression_detector=regression_detector,
        alert_manager=alert_manager,
    )
