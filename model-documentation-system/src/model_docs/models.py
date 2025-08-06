"""Data models for model documentation system."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Metadata for a machine learning model."""

    name: str
    version: str
    description: Optional[str] = None
    architecture: str
    framework: str
    total_parameters: int
    trainable_parameters: int
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    created_date: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TrainingConfig(BaseModel):
    """Training configuration details."""

    dataset_name: Optional[str] = None
    dataset_size: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    epochs: Optional[int] = None
    optimizer: Optional[str] = None
    loss_function: Optional[str] = None
    hardware: Optional[str] = None
    training_time_hours: Optional[float] = None
    carbon_footprint: Optional[float] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class PerformanceMetrics(BaseModel):
    """Model performance metrics."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    custom_metrics: Dict[str, float] = Field(default_factory=dict)
    evaluation_dataset: Optional[str] = None
    evaluation_date: datetime = Field(default_factory=datetime.now)


class EthicalConsiderations(BaseModel):
    """Ethical considerations and bias analysis."""

    intended_use: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
    potential_biases: List[str] = Field(default_factory=list)
    fairness_metrics: Dict[str, float] = Field(default_factory=dict)
    privacy_considerations: List[str] = Field(default_factory=list)
    environmental_impact: Optional[str] = None


class ModelCard(BaseModel):
    """Complete model card documentation."""

    metadata: ModelMetadata
    training_config: Optional[TrainingConfig] = None
    performance: Optional[PerformanceMetrics] = None
    ethical_considerations: Optional[EthicalConsiderations] = None
    usage_guidelines: Optional[str] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    contact_info: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    changelog: List[Dict[str, Any]] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    """Compliance report for regulatory frameworks."""

    framework: str
    version: str
    assessment_date: datetime = Field(default_factory=datetime.now)
    compliant: bool
    score: Optional[float] = None
    requirements_met: List[str] = Field(default_factory=list)
    requirements_missing: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
