"""
Pydantic models for the Fine-Tuning API
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback BaseModel if pydantic not available
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(*args, **kwargs):
        return None

# =====================
# Core Configuration Models
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


# =====================
# Recipe and Experiment Models
# =====================


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


class ExperimentStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ExperimentStatus] = None


class Experiment(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    recipe: Recipe
    status: ExperimentStatus = ExperimentStatus.QUEUED
    progress: float = 0.0
    createdAt: datetime
    updatedAt: datetime
    startedAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None
    logs: List[str] = []


# =====================
# Dataset Models
# =====================


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    format: str = "jsonl"
    source: str  # file path or URL


class Dataset(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    format: str
    source: str
    size: int
    samples: int
    createdAt: datetime
    metadata: Dict[str, Any] = {}


class DatasetValidation(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    samples: int
    statistics: Dict[str, Any] = {}


# =====================
# Deployment Models
# =====================


class DeploymentTarget(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CLOUD = "cloud"


class DeploymentConfig(BaseModel):
    target: DeploymentTarget
    modelName: str
    apiKey: Optional[str] = None
    hardwareRequirements: Optional[Dict[str, Any]] = None
    scalingConfig: Optional[Dict[str, Any]] = None


class Deployment(BaseModel):
    id: str
    experimentId: str
    target: DeploymentTarget
    status: str
    endpoint: Optional[str] = None
    createdAt: datetime
    completedAt: Optional[datetime] = None
    logs: List[str] = []


# =====================
# A/B Testing Models
# =====================


class ABTestConfig(BaseModel):
    name: str
    description: str
    baselineModel: str
    candidateModel: str
    trafficSplit: float = 0.5
    testCriteria: Dict[str, Any]


class ABTest(BaseModel):
    id: str
    name: str
    description: str
    baselineModel: str
    candidateModel: str
    status: str
    results: Optional[Dict[str, Any]] = None
    createdAt: datetime
    startedAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None


# =====================
# Response Models
# =====================


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


class ListResponse(BaseModel):
    items: List[Any]
    total: int
    page: int = 1
    pageSize: int = 20
