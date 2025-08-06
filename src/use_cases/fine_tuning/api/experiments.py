"""
Experiment management API endpoints
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from .models import (
    APIResponse,
    Experiment,
    ExperimentCreate,
    ExperimentStatus,
    ExperimentUpdate,
    ListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage (replace with database)
experiments_db: Dict[str, Experiment] = {}


@router.post("/", response_model=APIResponse)
async def create_experiment(experiment: ExperimentCreate, background_tasks: BackgroundTasks):
    """Create a new fine-tuning experiment"""

    experiment_id = str(uuid.uuid4())
    now = datetime.utcnow()

    exp = Experiment(
        id=experiment_id,
        name=experiment.name,
        description=experiment.description,
        recipe=experiment.recipe,
        status=ExperimentStatus.QUEUED,
        createdAt=now,
        updatedAt=now,
    )

    experiments_db[experiment_id] = exp

    # Start experiment in background
    background_tasks.add_task(run_experiment, experiment_id)

    logger.info(f"Created experiment {experiment_id}: {experiment.name}")

    return APIResponse(
        success=True, message="Experiment created successfully", data={"id": experiment_id}
    )


@router.get("/", response_model=ListResponse)
async def list_experiments(
    page: int = 1, page_size: int = 20, status: Optional[ExperimentStatus] = None
):
    """List all experiments with pagination"""

    # Filter by status if provided
    filtered_experiments = list(experiments_db.values())
    if status:
        filtered_experiments = [exp for exp in filtered_experiments if exp.status == status]

    # Sort by creation date (newest first)
    filtered_experiments.sort(key=lambda x: x.createdAt, reverse=True)

    # Pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = filtered_experiments[start_idx:end_idx]

    return ListResponse(
        items=paginated, total=len(filtered_experiments), page=page, pageSize=page_size
    )


@router.get("/{experiment_id}", response_model=Experiment)
async def get_experiment(experiment_id: str):
    """Get experiment details"""

    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return experiments_db[experiment_id]


@router.put("/{experiment_id}", response_model=APIResponse)
async def update_experiment(experiment_id: str, update: ExperimentUpdate):
    """Update experiment details"""

    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp = experiments_db[experiment_id]

    if update.name:
        exp.name = update.name
    if update.description is not None:
        exp.description = update.description
    if update.status:
        exp.status = update.status

    exp.updatedAt = datetime.utcnow()

    logger.info(f"Updated experiment {experiment_id}")

    return APIResponse(success=True, message="Experiment updated successfully")


@router.delete("/{experiment_id}", response_model=APIResponse)
async def delete_experiment(experiment_id: str):
    """Delete an experiment"""

    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Cancel if running
    exp = experiments_db[experiment_id]
    if exp.status in [ExperimentStatus.TRAINING, ExperimentStatus.EVALUATING]:
        exp.status = ExperimentStatus.CANCELLED
        exp.updatedAt = datetime.utcnow()
        # In a real implementation, you'd cancel the actual training process

    del experiments_db[experiment_id]

    logger.info(f"Deleted experiment {experiment_id}")

    return APIResponse(success=True, message="Experiment deleted successfully")


@router.post("/{experiment_id}/start", response_model=APIResponse)
async def start_experiment(experiment_id: str, background_tasks: BackgroundTasks):
    """Start or restart an experiment"""

    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp = experiments_db[experiment_id]

    if exp.status == ExperimentStatus.TRAINING:
        raise HTTPException(status_code=400, detail="Experiment is already running")

    exp.status = ExperimentStatus.QUEUED
    exp.progress = 0.0
    exp.startedAt = datetime.utcnow()
    exp.updatedAt = datetime.utcnow()

    # Start experiment in background
    background_tasks.add_task(run_experiment, experiment_id)

    logger.info(f"Started experiment {experiment_id}")

    return APIResponse(success=True, message="Experiment started successfully")


@router.post("/{experiment_id}/stop", response_model=APIResponse)
async def stop_experiment(experiment_id: str):
    """Stop a running experiment"""

    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp = experiments_db[experiment_id]

    if exp.status not in [ExperimentStatus.TRAINING, ExperimentStatus.EVALUATING]:
        raise HTTPException(status_code=400, detail="Experiment is not running")

    exp.status = ExperimentStatus.CANCELLED
    exp.updatedAt = datetime.utcnow()

    logger.info(f"Stopped experiment {experiment_id}")

    return APIResponse(success=True, message="Experiment stopped successfully")


@router.get("/{experiment_id}/logs")
async def get_experiment_logs(experiment_id: str):
    """Get experiment logs"""

    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp = experiments_db[experiment_id]
    return {"logs": exp.logs}


@router.get("/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    """Get experiment metrics"""

    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp = experiments_db[experiment_id]
    return {"metrics": exp.metrics or {}}


# Background task functions


async def run_experiment(experiment_id: str):
    """Run an experiment in the background"""
    try:
        if experiment_id not in experiments_db:
            return

        exp = experiments_db[experiment_id]

        # Update status to preparing
        exp.status = ExperimentStatus.PREPARING
        exp.updatedAt = datetime.utcnow()
        exp.logs.append(f"[{datetime.utcnow()}] Preparing experiment")

        # Simulate preparation
        await asyncio.sleep(2)

        # Update status to training
        exp.status = ExperimentStatus.TRAINING
        exp.updatedAt = datetime.utcnow()
        exp.logs.append(f"[{datetime.utcnow()}] Starting training")

        # Simulate training with progress updates
        num_epochs = exp.recipe.training.numEpochs
        for epoch in range(num_epochs):
            if exp.status == ExperimentStatus.CANCELLED:
                break

            # Simulate training time
            await asyncio.sleep(5)

            # Update progress
            progress = (epoch + 1) / num_epochs * 0.8  # 80% for training
            exp.progress = progress
            exp.updatedAt = datetime.utcnow()

            # Add training metrics
            loss = 3.0 - (epoch * 0.3)  # Simulate decreasing loss
            accuracy = 0.5 + (epoch * 0.1)  # Simulate increasing accuracy

            exp.logs.append(
                f"[{datetime.utcnow()}] Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.3f}, Accuracy: {accuracy:.3f}"
            )

            if not exp.metrics:
                exp.metrics = {}
            exp.metrics[f"epoch_{epoch + 1}"] = {"loss": loss, "accuracy": accuracy}

        if exp.status != ExperimentStatus.CANCELLED:
            # Update status to evaluating
            exp.status = ExperimentStatus.EVALUATING
            exp.updatedAt = datetime.utcnow()
            exp.logs.append(f"[{datetime.utcnow()}] Starting evaluation")

            # Simulate evaluation
            await asyncio.sleep(3)

            # Update progress to 100%
            exp.progress = 1.0
            exp.status = ExperimentStatus.COMPLETED
            exp.completedAt = datetime.utcnow()
            exp.updatedAt = datetime.utcnow()
            exp.logs.append(f"[{datetime.utcnow()}] Experiment completed successfully")

            # Add final metrics
            if not exp.metrics:
                exp.metrics = {}
            exp.metrics["final"] = {"perplexity": 12.5, "accuracy": 0.87, "f1_score": 0.85}

    except Exception as e:
        logger.error(f"Error running experiment {experiment_id}: {e}")
        if experiment_id in experiments_db:
            exp = experiments_db[experiment_id]
            exp.status = ExperimentStatus.FAILED
            exp.updatedAt = datetime.utcnow()
            exp.logs.append(f"[{datetime.utcnow()}] Experiment failed: {str(e)}")
