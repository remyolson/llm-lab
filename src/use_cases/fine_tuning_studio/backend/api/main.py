"""
FastAPI backend for Fine-Tuning Studio
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fine-Tuning Studio API",
    description="Backend API for managing fine-tuning experiments",
    version="1.0.0",
)

# Configure CORS with configuration system
try:
    from ....config.settings import get_settings

    settings = get_settings()
    cors_origins = settings.server.cors_origins
    cors_allow_credentials = settings.server.cors_allow_credentials
except ImportError:
    # Fallback for backward compatibility
    cors_origins = ["http://localhost:3001", "http://localhost:3000"]
    cors_allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# Pydantic Models
# =====================


class ModelConfig(BaseModel):
    name: str
    baseModel: str
    modelType: str = "causal_lm"
    useFlashAttention: bool = False


class DatasetConfig(BaseModel):
    name: str
    path: str
    format: str = "jsonl"
    splitRatios: Dict[str, float] = {"train": 0.8, "validation": 0.15, "test": 0.05}
    maxSamples: Optional[int] = None
    preprocessing: Optional[Dict[str, Any]] = None


class TrainingConfig(BaseModel):
    numEpochs: int = 3
    perDeviceTrainBatchSize: int = 4
    learningRate: float = 2e-5
    useLora: bool = True
    loraRank: int = 8
    loraAlpha: int = 16
    loraDropout: float = 0.1
    fp16: bool = True
    bf16: bool = False
    gradientCheckpointing: bool = True


class EvaluationConfig(BaseModel):
    metrics: List[str] = ["perplexity", "accuracy"]
    benchmarks: List[str] = []
    customEvalFunction: Optional[str] = None
    evalFunctionConfig: Optional[Dict[str, Any]] = None


class MonitoringConfig(BaseModel):
    platforms: List[str] = ["tensorboard"]
    projectName: Optional[str] = None
    enableAlerts: bool = True
    resourceMonitoring: bool = True


class Recipe(BaseModel):
    name: str
    description: Optional[str] = None
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    evaluation: Optional[EvaluationConfig] = None
    monitoring: Optional[MonitoringConfig] = None


class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    recipe: Recipe
    tags: List[str] = []


class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    recipe: Recipe
    metrics: Optional[Dict[str, Any]]
    createdAt: datetime
    updatedAt: datetime
    userId: str
    version: int
    tags: List[str]


class TrainingJobResponse(BaseModel):
    id: str
    experimentId: str
    status: str
    progress: float
    currentEpoch: Optional[int]
    totalEpochs: Optional[int]
    estimatedTimeRemaining: Optional[int]
    gpuUtilization: Optional[float]
    memoryUsage: Optional[float]
    currentLoss: Optional[float]
    startedAt: Optional[datetime]
    completedAt: Optional[datetime]


class DatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    path: str
    format: str
    size: int
    samples: int
    createdAt: datetime
    updatedAt: datetime
    metadata: Dict[str, Any]


class DeploymentCreate(BaseModel):
    experimentId: str
    name: str
    provider: str = "local"
    config: Optional[Dict[str, Any]] = None


class DeploymentResponse(BaseModel):
    id: str
    name: str
    experimentId: str
    checkpointPath: str
    endpoint: Optional[str]
    status: str
    provider: str
    deployedAt: datetime
    lastHealthCheck: Optional[datetime]
    metrics: Optional[Dict[str, Any]]


class ABTestCreate(BaseModel):
    name: str
    description: Optional[str]
    modelA: str
    modelB: str
    testCases: List[Dict[str, Any]]


class ABTestResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    modelA: str
    modelB: str
    status: str
    testCases: List[Dict[str, Any]]
    results: Optional[Dict[str, Any]]
    createdAt: datetime
    updatedAt: datetime


# =====================
# In-memory storage (replace with database)
# =====================

experiments_db: Dict[str, Dict] = {}
training_jobs_db: Dict[str, Dict] = {}
datasets_db: Dict[str, Dict] = {}
deployments_db: Dict[str, Dict] = {}
ab_tests_db: Dict[str, Dict] = {}

# =====================
# Experiment Endpoints
# =====================


@app.post("/api/experiments", response_model=ExperimentResponse)
async def create_experiment(experiment: ExperimentCreate, background_tasks: BackgroundTasks):
    """Create a new fine-tuning experiment"""
    experiment_id = str(uuid.uuid4())

    experiment_data = {
        "id": experiment_id,
        "name": experiment.name,
        "description": experiment.description,
        "status": "draft",
        "recipe": experiment.recipe.dict(),
        "metrics": None,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow(),
        "userId": "user_123",  # Replace with actual user ID from auth
        "version": 1,
        "tags": experiment.tags,
    }

    experiments_db[experiment_id] = experiment_data

    # Start training job in background
    background_tasks.add_task(start_training_job, experiment_id)

    return ExperimentResponse(**experiment_data)


@app.get("/api/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    page: int = 1, per_page: int = 10, status: Optional[str] = None, search: Optional[str] = None
):
    """List all experiments with pagination and filtering"""
    experiments = list(experiments_db.values())

    # Apply filters
    if status and status != "all":
        experiments = [e for e in experiments if e["status"] == status]

    if search:
        experiments = [
            e
            for e in experiments
            if search.lower() in e["name"].lower()
            or (e.get("description") and search.lower() in e["description"].lower())
        ]

    # Pagination
    start = (page - 1) * per_page
    end = start + per_page

    return [ExperimentResponse(**e) for e in experiments[start:end]]


@app.get("/api/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get a specific experiment by ID"""
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return ExperimentResponse(**experiments_db[experiment_id])


@app.put("/api/experiments/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(experiment_id: str, update: ExperimentUpdate):
    """Update an experiment"""
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    experiment = experiments_db[experiment_id]

    if update.name:
        experiment["name"] = update.name
    if update.description:
        experiment["description"] = update.description
    if update.tags:
        experiment["tags"] = update.tags

    experiment["updatedAt"] = datetime.utcnow()
    experiment["version"] += 1

    return ExperimentResponse(**experiment)


@app.delete("/api/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment"""
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    del experiments_db[experiment_id]
    return {"message": "Experiment deleted successfully"}


@app.post("/api/experiments/{experiment_id}/duplicate", response_model=ExperimentResponse)
async def duplicate_experiment(experiment_id: str):
    """Duplicate an experiment"""
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    original = experiments_db[experiment_id]
    new_id = str(uuid.uuid4())

    duplicate = {
        **original,
        "id": new_id,
        "name": f"{original['name']} (Copy)",
        "status": "draft",
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow(),
        "version": 1,
    }

    experiments_db[new_id] = duplicate
    return ExperimentResponse(**duplicate)


# =====================
# Training Job Endpoints
# =====================


@app.get("/api/training-jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(status: Optional[str] = None):
    """List all training jobs"""
    jobs = list(training_jobs_db.values())

    if status:
        jobs = [j for j in jobs if j["status"] == status]

    return [TrainingJobResponse(**j) for j in jobs]


@app.get("/api/training-jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str):
    """Get a specific training job"""
    if job_id not in training_jobs_db:
        raise HTTPException(status_code=404, detail="Training job not found")

    return TrainingJobResponse(**training_jobs_db[job_id])


@app.post("/api/training-jobs/{job_id}/stop")
async def stop_training_job(job_id: str):
    """Stop a running training job"""
    if job_id not in training_jobs_db:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = training_jobs_db[job_id]
    job["status"] = "cancelled"
    job["completedAt"] = datetime.utcnow()

    return {"message": "Training job stopped"}


# =====================
# Dataset Endpoints
# =====================


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), name: str = None, description: str = None):
    """Upload a new dataset"""
    dataset_id = str(uuid.uuid4())

    # Save file (in production, save to cloud storage)
    file_path = Path(f"/tmp/datasets/{dataset_id}_{file.filename}")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    file_path.write_bytes(content)

    # Analyze dataset
    dataset_data = {
        "id": dataset_id,
        "name": name or file.filename,
        "description": description,
        "path": str(file_path),
        "format": file.filename.split(".")[-1],
        "size": len(content),
        "samples": 1000,  # Calculate actual samples
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow(),
        "metadata": {
            "columns": ["input", "output"],
            "tokenStats": {
                "avgLength": 256,
                "maxLength": 512,
                "minLength": 32,
                "totalTokens": 256000,
            },
            "qualityScore": 0.85,
            "duplicates": 0,
            "formatErrors": 0,
        },
    }

    datasets_db[dataset_id] = dataset_data
    return DatasetResponse(**dataset_data)


@app.get("/api/datasets", response_model=List[DatasetResponse])
async def list_datasets():
    """List all datasets"""
    return [DatasetResponse(**d) for d in datasets_db.values()]


@app.get("/api/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str):
    """Get a specific dataset"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetResponse(**datasets_db[dataset_id])


@app.get("/api/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 10):
    """Preview dataset samples"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Return mock preview data
    return {
        "samples": [
            {
                "id": str(i),
                "content": {"input": f"Sample input {i}", "output": f"Sample output {i}"},
                "tokenCount": 50 + i * 10,
            }
            for i in range(min(limit, 10))
        ]
    }


# =====================
# Deployment Endpoints
# =====================


@app.post("/api/deployments", response_model=DeploymentResponse)
async def create_deployment(deployment: DeploymentCreate, background_tasks: BackgroundTasks):
    """Deploy a fine-tuned model"""
    deployment_id = str(uuid.uuid4())

    deployment_data = {
        "id": deployment_id,
        "name": deployment.name,
        "experimentId": deployment.experimentId,
        "checkpointPath": f"/models/{deployment.experimentId}/checkpoint-final",
        "endpoint": None,
        "status": "deploying",
        "provider": deployment.provider,
        "deployedAt": datetime.utcnow(),
        "lastHealthCheck": None,
        "metrics": None,
    }

    deployments_db[deployment_id] = deployment_data

    # Start deployment in background
    background_tasks.add_task(deploy_model, deployment_id)

    return DeploymentResponse(**deployment_data)


@app.get("/api/deployments", response_model=List[DeploymentResponse])
async def list_deployments():
    """List all deployments"""
    return [DeploymentResponse(**d) for d in deployments_db.values()]


@app.get("/api/deployments/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(deployment_id: str):
    """Get a specific deployment"""
    if deployment_id not in deployments_db:
        raise HTTPException(status_code=404, detail="Deployment not found")

    return DeploymentResponse(**deployments_db[deployment_id])


@app.post("/api/deployments/{deployment_id}/rollback")
async def rollback_deployment(deployment_id: str):
    """Rollback a deployment"""
    if deployment_id not in deployments_db:
        raise HTTPException(status_code=404, detail="Deployment not found")

    deployment = deployments_db[deployment_id]
    deployment["status"] = "inactive"

    return {"message": "Deployment rolled back"}


# =====================
# A/B Testing Endpoints
# =====================


@app.post("/api/ab-tests", response_model=ABTestResponse)
async def create_ab_test(ab_test: ABTestCreate):
    """Create a new A/B test"""
    test_id = str(uuid.uuid4())

    test_data = {
        "id": test_id,
        "name": ab_test.name,
        "description": ab_test.description,
        "modelA": ab_test.modelA,
        "modelB": ab_test.modelB,
        "status": "draft",
        "testCases": ab_test.testCases,
        "results": None,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow(),
    }

    ab_tests_db[test_id] = test_data
    return ABTestResponse(**test_data)


@app.get("/api/ab-tests", response_model=List[ABTestResponse])
async def list_ab_tests():
    """List all A/B tests"""
    return [ABTestResponse(**t) for t in ab_tests_db.values()]


@app.get("/api/ab-tests/{test_id}", response_model=ABTestResponse)
async def get_ab_test(test_id: str):
    """Get a specific A/B test"""
    if test_id not in ab_tests_db:
        raise HTTPException(status_code=404, detail="A/B test not found")

    return ABTestResponse(**ab_tests_db[test_id])


@app.post("/api/ab-tests/{test_id}/run")
async def run_ab_test(test_id: str, background_tasks: BackgroundTasks):
    """Run an A/B test"""
    if test_id not in ab_tests_db:
        raise HTTPException(status_code=404, detail="A/B test not found")

    test = ab_tests_db[test_id]
    test["status"] = "running"

    # Run test in background
    background_tasks.add_task(execute_ab_test, test_id)

    return {"message": "A/B test started"}


# =====================
# Metrics & Analytics Endpoints
# =====================


@app.get("/api/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    """Get metrics for an experiment"""
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Return mock metrics
    return {
        "loss": [2.8, 2.3, 1.9, 1.6, 1.4],
        "accuracy": [0.65, 0.72, 0.78, 0.81, 0.85],
        "perplexity": [16.4, 10.2, 7.1, 5.2, 4.1],
        "learningRate": [2e-5, 2e-5, 1e-5, 1e-5, 5e-6],
        "timestamps": [datetime.utcnow() - timedelta(hours=i) for i in range(5, 0, -1)],
    }


@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    return {
        "systemMetrics": {
            "gpuUtilization": 75,
            "memoryUsage": 60,
            "activeExperiments": len(
                [e for e in experiments_db.values() if e["status"] == "running"]
            ),
            "totalDeployments": len(deployments_db),
        },
        "experiments": list(experiments_db.values())[:5],
        "recentDeployments": list(deployments_db.values())[:5],
        "activeJobs": [j for j in training_jobs_db.values() if j["status"] == "running"][:5],
    }


# =====================
# Health Check
# =====================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


# =====================
# Background Tasks
# =====================


async def start_training_job(experiment_id: str):
    """Start a training job for an experiment"""
    job_id = str(uuid.uuid4())

    job_data = {
        "id": job_id,
        "experimentId": experiment_id,
        "status": "running",
        "progress": 0,
        "currentEpoch": 0,
        "totalEpochs": 3,
        "estimatedTimeRemaining": 3600,
        "gpuUtilization": 0,
        "memoryUsage": 0,
        "currentLoss": None,
        "startedAt": datetime.utcnow(),
        "completedAt": None,
    }

    training_jobs_db[job_id] = job_data

    # Update experiment status
    if experiment_id in experiments_db:
        experiments_db[experiment_id]["status"] = "running"

    # Simulate training progress
    for epoch in range(3):
        await asyncio.sleep(2)  # Simulate training time
        job_data["currentEpoch"] = epoch + 1
        job_data["progress"] = ((epoch + 1) / 3) * 100
        job_data["currentLoss"] = 2.8 - (epoch * 0.5)
        job_data["gpuUtilization"] = 75 + (epoch * 5)
        job_data["memoryUsage"] = 60 + (epoch * 3)

    job_data["status"] = "completed"
    job_data["completedAt"] = datetime.utcnow()

    # Update experiment status
    if experiment_id in experiments_db:
        experiments_db[experiment_id]["status"] = "completed"


async def deploy_model(deployment_id: str):
    """Deploy a model"""
    await asyncio.sleep(5)  # Simulate deployment time

    if deployment_id in deployments_db:
        deployment = deployments_db[deployment_id]
        deployment["status"] = "active"
        deployment["endpoint"] = f"https://api.example.com/models/{deployment_id}"
        deployment["lastHealthCheck"] = datetime.utcnow()
        deployment["metrics"] = {"requestCount": 0, "avgLatency": 0, "errorRate": 0, "uptime": 100}


async def execute_ab_test(test_id: str):
    """Execute an A/B test"""
    await asyncio.sleep(3)  # Simulate test execution

    if test_id in ab_tests_db:
        test = ab_tests_db[test_id]
        test["status"] = "completed"
        test["results"] = {
            "winnerModel": "B",
            "confidenceScore": 0.87,
            "statisticalSignificance": 0.95,
            "metrics": {
                "modelA": {
                    "accuracy": 0.72,
                    "bleuScore": 0.65,
                    "rougeScore": 0.58,
                    "humanPreference": 0.35,
                    "avgLatency": 450,
                },
                "modelB": {
                    "accuracy": 0.85,
                    "bleuScore": 0.78,
                    "rougeScore": 0.71,
                    "humanPreference": 0.65,
                    "avgLatency": 480,
                },
            },
        }


if __name__ == "__main__":
    import uvicorn

    # Get server configuration from settings
    try:
        from ....config.settings import get_settings

        settings = get_settings()
        port = settings.server.api_port
        host = settings.server.api_host
    except ImportError:
        # Fallback for backward compatibility
        port = 8000
        host = "0.0.0.0"

    uvicorn.run(app, host=host, port=port)
