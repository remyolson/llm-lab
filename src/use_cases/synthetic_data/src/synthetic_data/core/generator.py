"""Base class for synthetic data generation."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from .config import GenerationConfig
from .privacy import PrivacyPreserver
from .validator import DataValidator

logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    """Result of a data generation operation."""

    success: bool
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    generation_time: float
    quality_score: Optional[float] = None


class SyntheticDataGenerator(ABC):
    """Base class for synthetic data generation."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        validator: Optional[DataValidator] = None,
        privacy_preserver: Optional[PrivacyPreserver] = None,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            config: Generation configuration
            validator: Data validator instance
            privacy_preserver: Privacy preservation instance
        """
        self.config = config or GenerationConfig()
        self.validator = validator or DataValidator()
        self.privacy_preserver = privacy_preserver or PrivacyPreserver()
        self._generation_history = []

    @abstractmethod
    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a single data record.

        Args:
            **kwargs: Generation parameters

        Returns:
            Generated data record
        """
        pass

    def generate_dataset(
        self,
        count: int,
        batch_size: Optional[int] = None,
        validate: bool = True,
        preserve_privacy: bool = True,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate a dataset with multiple records.

        Args:
            count: Number of records to generate
            batch_size: Batch size for generation
            validate: Whether to validate generated data
            preserve_privacy: Whether to apply privacy preservation
            **kwargs: Additional generation parameters

        Returns:
            Generation result with data and metadata
        """
        batch_size = batch_size or self.config.batch_size
        generated_data = []
        errors = []
        warnings = []
        start_time = datetime.now()

        logger.info(f"Starting generation of {count} records...")

        try:
            # Generate data in batches
            with tqdm(total=count, desc="Generating data") as pbar:
                for i in range(0, count, batch_size):
                    batch_count = min(batch_size, count - i)
                    batch_data = []

                    for _ in range(batch_count):
                        try:
                            record = self.generate_single(**kwargs)
                            batch_data.append(record)
                        except Exception as e:
                            error_msg = f"Error generating record: {str(e)}"
                            logger.error(error_msg)
                            errors.append(error_msg)

                    # Apply privacy preservation if enabled
                    if preserve_privacy and batch_data:
                        batch_data = self.privacy_preserver.apply_privacy(batch_data)

                    # Validate batch if enabled
                    if validate and batch_data:
                        validation_result = self.validator.validate_batch(batch_data)
                        if not validation_result.is_valid:
                            warnings.extend(validation_result.warnings)
                            errors.extend(validation_result.errors)

                    generated_data.extend(batch_data)
                    pbar.update(batch_count)

            # Calculate quality score
            quality_score = None
            if validate and generated_data:
                quality_result = self.validator.calculate_quality_score(generated_data)
                quality_score = quality_result.score

            generation_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = GenerationResult(
                success=len(errors) == 0,
                data=generated_data,
                metadata={
                    "count_requested": count,
                    "count_generated": len(generated_data),
                    "generation_config": self.config.dict(),
                    "timestamp": datetime.now().isoformat(),
                },
                errors=errors,
                warnings=warnings,
                generation_time=generation_time,
                quality_score=quality_score,
            )

            # Store in history
            self._generation_history.append(result)

            logger.info(f"Generation completed: {len(generated_data)}/{count} records")

            return result

        except Exception as e:
            logger.error(f"Fatal error during generation: {str(e)}")
            return GenerationResult(
                success=False,
                data=[],
                metadata={"error": str(e)},
                errors=[str(e)],
                generation_time=(datetime.now() - start_time).total_seconds(),
            )

    def validate_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the quality of generated data.

        Args:
            data: Data to validate

        Returns:
            Validation results
        """
        return self.validator.validate_quality(data)

    def ensure_privacy(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure privacy preservation in generated data.

        Args:
            data: Data to process

        Returns:
            Privacy-preserved data
        """
        return self.privacy_preserver.apply_privacy(data)

    def export_data(
        self,
        data: Union[List[Dict[str, Any]], GenerationResult],
        filepath: str,
        format: str = "json",
        **kwargs,
    ) -> bool:
        """
        Export generated data to file.

        Args:
            data: Data to export
            filepath: Output file path
            format: Export format (json, csv, parquet, jsonl)
            **kwargs: Additional export parameters

        Returns:
            Success status
        """
        try:
            # Extract data if GenerationResult
            if isinstance(data, GenerationResult):
                export_data = data.data
            else:
                export_data = data

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                with open(filepath, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format == "jsonl":
                with open(filepath, "w") as f:
                    for record in export_data:
                        f.write(json.dumps(record, default=str) + "\n")
            elif format == "csv":
                df = pd.DataFrame(export_data)
                df.to_csv(filepath, index=False, **kwargs)
            elif format == "parquet":
                df = pd.DataFrame(export_data)
                df.to_parquet(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Data exported successfully to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return False

    def get_generation_history(self) -> List[GenerationResult]:
        """Get generation history."""
        return self._generation_history

    def clear_history(self):
        """Clear generation history."""
        self._generation_history = []
