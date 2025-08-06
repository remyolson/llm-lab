"""
Dataset management API endpoints
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from .models import APIResponse, Dataset, DatasetCreate, DatasetValidation, ListResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage (replace with database)
datasets_db: Dict[str, Dataset] = {}


@router.post("/", response_model=APIResponse)
async def create_dataset(dataset: DatasetCreate):
    """Create a new dataset"""

    dataset_id = str(uuid.uuid4())

    # Validate dataset file exists and format
    validation = await validate_dataset(dataset.source, dataset.format)
    if not validation.valid:
        raise HTTPException(
            status_code=400, detail=f"Invalid dataset: {'; '.join(validation.errors)}"
        )

    ds = Dataset(
        id=dataset_id,
        name=dataset.name,
        description=dataset.description,
        format=dataset.format,
        source=dataset.source,
        size=validation.statistics.get("file_size", 0),
        samples=validation.samples,
        createdAt=datetime.utcnow(),
        metadata=validation.statistics,
    )

    datasets_db[dataset_id] = ds

    logger.info(f"Created dataset {dataset_id}: {dataset.name}")

    return APIResponse(
        success=True, message="Dataset created successfully", data={"id": dataset_id}
    )


@router.post("/upload", response_model=APIResponse)
async def upload_dataset(
    name: str, description: Optional[str] = None, file: UploadFile = File(...)
):
    """Upload a dataset file"""

    if not file.filename.endswith((".jsonl", ".json", ".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Determine format from file extension
    format_map = {".jsonl": "jsonl", ".json": "json", ".csv": "csv", ".txt": "text"}
    format = format_map[Path(file.filename).suffix]

    # Save uploaded file
    upload_dir = Path("/tmp/datasets")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create dataset
    dataset_create = DatasetCreate(
        name=name, description=description, format=format, source=str(file_path)
    )

    return await create_dataset(dataset_create)


@router.get("/", response_model=ListResponse)
async def list_datasets(page: int = 1, page_size: int = 20, format: Optional[str] = None):
    """List all datasets with pagination"""

    # Filter by format if provided
    filtered_datasets = list(datasets_db.values())
    if format:
        filtered_datasets = [ds for ds in filtered_datasets if ds.format == format]

    # Sort by creation date (newest first)
    filtered_datasets.sort(key=lambda x: x.createdAt, reverse=True)

    # Pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = filtered_datasets[start_idx:end_idx]

    return ListResponse(
        items=paginated, total=len(filtered_datasets), page=page, pageSize=page_size
    )


@router.get("/{dataset_id}", response_model=Dataset)
async def get_dataset(dataset_id: str):
    """Get dataset details"""

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return datasets_db[dataset_id]


@router.delete("/{dataset_id}", response_model=APIResponse)
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ds = datasets_db[dataset_id]

    # Remove file if it exists
    try:
        Path(ds.source).unlink(missing_ok=True)
    except:
        pass  # Ignore file deletion errors

    del datasets_db[dataset_id]

    logger.info(f"Deleted dataset {dataset_id}")

    return APIResponse(success=True, message="Dataset deleted successfully")


@router.get("/{dataset_id}/validate", response_model=DatasetValidation)
async def validate_dataset_endpoint(dataset_id: str):
    """Validate a dataset"""

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ds = datasets_db[dataset_id]
    return await validate_dataset(ds.source, ds.format)


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 10):
    """Preview first N samples from a dataset"""

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ds = datasets_db[dataset_id]

    try:
        samples = []
        file_path = Path(ds.source)

        if ds.format == "jsonl":
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    samples.append(json.loads(line.strip()))

        elif ds.format == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data[:limit]
                else:
                    samples = [data]

        elif ds.format == "csv":
            import csv

            with open(file_path, "r") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= limit:
                        break
                    samples.append(dict(row))

        elif ds.format == "text":
            with open(file_path, "r") as f:
                lines = f.readlines()
                samples = [{"text": line.strip()} for line in lines[:limit]]

        return {"samples": samples, "total_previewed": len(samples), "dataset_total": ds.samples}

    except Exception as e:
        logger.error(f"Error previewing dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail="Error reading dataset file")


@router.get("/{dataset_id}/statistics")
async def get_dataset_statistics(dataset_id: str):
    """Get detailed statistics for a dataset"""

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ds = datasets_db[dataset_id]

    # Calculate detailed statistics
    stats = {
        "basic": {
            "samples": ds.samples,
            "size_bytes": ds.size,
            "format": ds.format,
            "created": ds.createdAt,
        },
        "content": await analyze_dataset_content(ds.source, ds.format),
    }

    return stats


# Helper functions


async def validate_dataset(file_path: str, format: str) -> DatasetValidation:
    """Validate dataset file and return validation results"""

    errors = []
    warnings = []
    samples = 0
    statistics = {}

    try:
        file_path = Path(file_path)

        if not file_path.exists():
            errors.append(f"File does not exist: {file_path}")
            return DatasetValidation(
                valid=False, errors=errors, warnings=warnings, samples=0, statistics={}
            )

        statistics["file_size"] = file_path.stat().st_size

        if format == "jsonl":
            with open(file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            json.loads(line.strip())
                            samples += 1
                        except json.JSONDecodeError as e:
                            errors.append(f"Invalid JSON on line {line_num}: {e}")

        elif format == "json":
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = len(data)
                    else:
                        samples = 1
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON file: {e}")

        elif format == "csv":
            import csv

            try:
                with open(file_path, "r") as f:
                    reader = csv.reader(f)
                    samples = sum(1 for _ in reader) - 1  # Subtract header
                    if samples < 0:
                        samples = 0
                        warnings.append("CSV file appears to be empty")
            except Exception as e:
                errors.append(f"Error reading CSV file: {e}")

        elif format == "text":
            with open(file_path, "r") as f:
                samples = sum(1 for line in f if line.strip())

        # Check minimum samples
        if samples < 10:
            warnings.append(
                f"Dataset has only {samples} samples, which may be too few for training"
            )

        statistics["samples"] = samples
        statistics["format"] = format

    except Exception as e:
        errors.append(f"Error validating dataset: {e}")

    return DatasetValidation(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        samples=samples,
        statistics=statistics,
    )


async def analyze_dataset_content(file_path: str, format: str) -> Dict[str, Any]:
    """Analyze dataset content for detailed statistics"""

    analysis = {"text_stats": {}, "field_analysis": {}, "sample_distribution": {}}

    try:
        file_path = Path(file_path)

        if format == "jsonl":
            texts = []
            fields = set()

            with open(file_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        fields.update(data.keys())

                        # Extract text content for analysis
                        if "text" in data:
                            texts.append(data["text"])
                        elif "prompt" in data and "response" in data:
                            texts.extend([data["prompt"], data["response"]])
                    except:
                        continue

            analysis["field_analysis"]["fields"] = list(fields)
            analysis["field_analysis"]["common_fields"] = [
                "text",
                "prompt",
                "response",
                "instruction",
                "input",
                "output",
            ]

            if texts:
                lengths = [len(text) for text in texts]
                analysis["text_stats"] = {
                    "avg_length": sum(lengths) / len(lengths),
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                    "total_chars": sum(lengths),
                }

    except Exception as e:
        analysis["error"] = str(e)

    return analysis
