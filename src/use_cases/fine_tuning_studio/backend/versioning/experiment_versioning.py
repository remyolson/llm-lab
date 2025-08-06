"""
Experiment versioning system for tracking configurations, datasets, and checkpoints
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dvc.api
import git

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VersionType(Enum):
    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features
    PATCH = "patch"  # Bug fixes


@dataclass
class ExperimentVersion:
    """Represents a version of an experiment"""

    version: str
    experiment_id: str
    parent_version: Optional[str]
    created_at: datetime
    created_by: str
    commit_hash: Optional[str]

    # Versioned components
    config: Dict[str, Any]
    dataset_version: str
    checkpoint_path: str
    metrics: Optional[Dict[str, Any]]

    # Metadata
    description: str
    tags: List[str]
    is_baseline: bool = False
    is_production: bool = False


@dataclass
class DatasetVersion:
    """Represents a version of a dataset"""

    version: str
    dataset_id: str
    path: str
    hash: str
    size: int
    samples: int
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class CheckpointVersion:
    """Represents a model checkpoint version"""

    version: str
    experiment_id: str
    epoch: int
    path: str
    hash: str
    size: int
    metrics: Dict[str, Any]
    created_at: datetime
    is_best: bool = False


class VersioningSystem:
    """Main versioning system for experiments"""

    def __init__(self, base_path: str = "/data/experiments"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize Git repository if not exists
        self.repo = self._init_git_repo()

        # Version storage
        self.versions_db: Dict[str, List[ExperimentVersion]] = {}
        self.dataset_versions_db: Dict[str, List[DatasetVersion]] = {}
        self.checkpoint_versions_db: Dict[str, List[CheckpointVersion]] = {}

    def _init_git_repo(self) -> git.Repo:
        """Initialize Git repository for version control"""
        try:
            repo = git.Repo(self.base_path)
        except git.InvalidGitRepositoryError:
            repo = git.Repo.init(self.base_path)

            # Create .gitignore
            gitignore_path = self.base_path / ".gitignore"
            gitignore_path.write_text(
                "*.pyc\n__pycache__/\n*.log\n*.tmp\n.DS_Store\ncheckpoints/*.bin\n"
            )

            repo.index.add([".gitignore"])
            repo.index.commit("Initial commit")

        return repo

    def create_experiment_version(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        dataset_path: str,
        created_by: str,
        description: str = "",
        parent_version: Optional[str] = None,
        version_type: VersionType = VersionType.MINOR,
    ) -> ExperimentVersion:
        """Create a new version of an experiment"""

        # Generate version number
        version = self._generate_version_number(experiment_id, parent_version, version_type)

        # Create experiment directory
        exp_dir = self.base_path / experiment_id / version
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = exp_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2))

        # Version the dataset
        dataset_version = self._version_dataset(dataset_path, experiment_id)

        # Create checkpoint directory
        checkpoint_path = exp_dir / "checkpoints"
        checkpoint_path.mkdir(exist_ok=True)

        # Git commit
        self.repo.index.add([str(config_path)])
        commit = self.repo.index.commit(
            f"Version {version}: {description or 'New experiment version'}"
        )

        # Create version object
        exp_version = ExperimentVersion(
            version=version,
            experiment_id=experiment_id,
            parent_version=parent_version,
            created_at=datetime.utcnow(),
            created_by=created_by,
            commit_hash=commit.hexsha,
            config=config,
            dataset_version=dataset_version.version,
            checkpoint_path=str(checkpoint_path),
            metrics=None,
            description=description,
            tags=[],
            is_baseline=parent_version is None,
        )

        # Store version
        if experiment_id not in self.versions_db:
            self.versions_db[experiment_id] = []
        self.versions_db[experiment_id].append(exp_version)

        # Save version metadata
        self._save_version_metadata(exp_version)

        logger.info(f"Created experiment version {version} for {experiment_id}")

        return exp_version

    def _generate_version_number(
        self, experiment_id: str, parent_version: Optional[str], version_type: VersionType
    ) -> str:
        """Generate a semantic version number"""

        if parent_version is None:
            return "1.0.0"

        # Parse parent version
        parts = parent_version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        # Increment based on version type
        if version_type == VersionType.MAJOR:
            return f"{major + 1}.0.0"
        elif version_type == VersionType.MINOR:
            return f"{major}.{minor + 1}.0"
        else:  # PATCH
            return f"{major}.{minor}.{patch + 1}"

    def _version_dataset(self, dataset_path: str, experiment_id: str) -> DatasetVersion:
        """Version a dataset using DVC or hash-based versioning"""

        dataset_path = Path(dataset_path)

        # Calculate dataset hash
        dataset_hash = self._calculate_file_hash(dataset_path)

        # Check if this version already exists
        for version in self.dataset_versions_db.get(experiment_id, []):
            if version.hash == dataset_hash:
                return version

        # Create new dataset version
        version_id = f"ds_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Copy dataset to versioned location
        versioned_path = self.base_path / "datasets" / experiment_id / version_id
        versioned_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset_path, versioned_path)

        # Calculate metadata
        file_stats = dataset_path.stat()

        dataset_version = DatasetVersion(
            version=version_id,
            dataset_id=experiment_id,
            path=str(versioned_path),
            hash=dataset_hash,
            size=file_stats.st_size,
            samples=self._count_dataset_samples(dataset_path),
            created_at=datetime.utcnow(),
            metadata={},
        )

        # Store version
        if experiment_id not in self.dataset_versions_db:
            self.dataset_versions_db[experiment_id] = []
        self.dataset_versions_db[experiment_id].append(dataset_version)

        return dataset_version

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _count_dataset_samples(self, dataset_path: Path) -> int:
        """Count number of samples in a dataset"""
        # Simple line count for JSONL files
        if dataset_path.suffix == ".jsonl":
            with open(dataset_path, "r") as f:
                return sum(1 for _ in f)
        return 0

    def save_checkpoint(
        self,
        experiment_id: str,
        version: str,
        epoch: int,
        checkpoint_path: str,
        metrics: Dict[str, Any],
        is_best: bool = False,
    ) -> CheckpointVersion:
        """Save and version a model checkpoint"""

        checkpoint_path = Path(checkpoint_path)

        # Calculate checkpoint hash
        checkpoint_hash = self._calculate_file_hash(checkpoint_path)

        # Create versioned checkpoint path
        exp_version_dir = self.base_path / experiment_id / version / "checkpoints"
        exp_version_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = f"checkpoint_epoch_{epoch}.bin"
        if is_best:
            checkpoint_name = "best_" + checkpoint_name

        versioned_checkpoint = exp_version_dir / checkpoint_name
        shutil.copy2(checkpoint_path, versioned_checkpoint)

        # Create checkpoint version
        checkpoint_version = CheckpointVersion(
            version=f"{version}_epoch_{epoch}",
            experiment_id=experiment_id,
            epoch=epoch,
            path=str(versioned_checkpoint),
            hash=checkpoint_hash,
            size=checkpoint_path.stat().st_size,
            metrics=metrics,
            created_at=datetime.utcnow(),
            is_best=is_best,
        )

        # Store checkpoint version
        if experiment_id not in self.checkpoint_versions_db:
            self.checkpoint_versions_db[experiment_id] = []
        self.checkpoint_versions_db[experiment_id].append(checkpoint_version)

        # Update experiment version metrics if this is the best checkpoint
        if is_best:
            self._update_experiment_metrics(experiment_id, version, metrics)

        logger.info(f"Saved checkpoint for {experiment_id} v{version} epoch {epoch}")

        return checkpoint_version

    def _update_experiment_metrics(self, experiment_id: str, version: str, metrics: Dict[str, Any]):
        """Update experiment version with latest metrics"""
        if experiment_id in self.versions_db:
            for exp_version in self.versions_db[experiment_id]:
                if exp_version.version == version:
                    exp_version.metrics = metrics
                    self._save_version_metadata(exp_version)
                    break

    def _save_version_metadata(self, version: ExperimentVersion):
        """Save version metadata to disk"""
        metadata_path = self.base_path / version.experiment_id / version.version / "metadata.json"
        metadata_path.write_text(json.dumps(asdict(version), indent=2, default=str))

    def get_experiment_versions(self, experiment_id: str) -> List[ExperimentVersion]:
        """Get all versions of an experiment"""
        return self.versions_db.get(experiment_id, [])

    def get_version(self, experiment_id: str, version: str) -> Optional[ExperimentVersion]:
        """Get a specific version of an experiment"""
        versions = self.get_experiment_versions(experiment_id)
        for v in versions:
            if v.version == version:
                return v
        return None

    def compare_versions(self, experiment_id: str, version1: str, version2: str) -> Dict[str | Any]:
        """Compare two versions of an experiment"""

        v1 = self.get_version(experiment_id, version1)
        v2 = self.get_version(experiment_id, version2)

        if not v1 or not v2:
            raise ValueError("Version not found")

        # Compare configurations
        config_diff = self._diff_configs(v1.config, v2.config)

        # Compare metrics
        metrics_diff = {}
        if v1.metrics and v2.metrics:
            metrics_diff = self._diff_metrics(v1.metrics, v2.metrics)

        # Compare datasets
        dataset_changed = v1.dataset_version != v2.dataset_version

        return {
            "versions": {"v1": version1, "v2": version2},
            "config_changes": config_diff,
            "metrics_changes": metrics_diff,
            "dataset_changed": dataset_changed,
            "timestamps": {"v1": v1.created_at, "v2": v2.created_at},
        }

    def _diff_configs(self, config1: Dict, config2: Dict) -> Dict[str | Any]:
        """Find differences between two configurations"""
        diff = {}

        all_keys = set(config1.keys()) | set(config2.keys())

        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)

            if val1 != val2:
                diff[key] = {"old": val1, "new": val2}

        return diff

    def _diff_metrics(self, metrics1: Dict, metrics2: Dict) -> Dict[str | Any]:
        """Compare metrics between versions"""
        diff = {}

        for key in metrics1.keys() & metrics2.keys():
            val1 = metrics1[key]
            val2 = metrics2[key]

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                change = val2 - val1
                change_pct = (change / val1 * 100) if val1 != 0 else 0

                diff[key] = {
                    "old": val1,
                    "new": val2,
                    "change": change,
                    "change_pct": change_pct,
                    "improved": change > 0 if key != "loss" else change < 0,
                }
            else:
                diff[key] = {"old": val1, "new": val2}

        return diff

    def rollback_to_version(
        self, experiment_id: str, target_version: str, created_by: str
    ) -> ExperimentVersion:
        """Rollback an experiment to a previous version"""

        target = self.get_version(experiment_id, target_version)
        if not target:
            raise ValueError(f"Version {target_version} not found")

        # Create new version based on target
        new_version = self.create_experiment_version(
            experiment_id=experiment_id,
            config=target.config,
            dataset_path=self._get_dataset_path(target.dataset_version),
            created_by=created_by,
            description=f"Rollback to version {target_version}",
            parent_version=target_version,
            version_type=VersionType.PATCH,
        )

        # Copy checkpoints from target version
        target_checkpoints = self.base_path / experiment_id / target_version / "checkpoints"
        new_checkpoints = self.base_path / experiment_id / new_version.version / "checkpoints"

        if target_checkpoints.exists():
            shutil.copytree(target_checkpoints, new_checkpoints, dirs_exist_ok=True)

        logger.info(f"Rolled back {experiment_id} to version {target_version}")

        return new_version

    def _get_dataset_path(self, dataset_version: str) -> str:
        """Get the path to a versioned dataset"""
        for datasets in self.dataset_versions_db.values():
            for dataset in datasets:
                if dataset.version == dataset_version:
                    return dataset.path
        raise ValueError(f"Dataset version {dataset_version} not found")

    def export_version(
        self, experiment_id: str, version: str, export_path: str, include_checkpoints: bool = True
    ) -> str:
        """Export a version as a reproducible package"""

        exp_version = self.get_version(experiment_id, version)
        if not exp_version:
            raise ValueError(f"Version {version} not found")

        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        # Create export package
        package_dir = export_path / f"{experiment_id}_v{version}"
        package_dir.mkdir(exist_ok=True)

        # Copy configuration
        config_path = package_dir / "config.json"
        config_path.write_text(json.dumps(exp_version.config, indent=2))

        # Copy dataset reference
        dataset_info = package_dir / "dataset_info.json"
        dataset_info.write_text(
            json.dumps(
                {
                    "version": exp_version.dataset_version,
                    "path": self._get_dataset_path(exp_version.dataset_version),
                },
                indent=2,
            )
        )

        # Copy checkpoints if requested
        if include_checkpoints:
            source_checkpoints = Path(exp_version.checkpoint_path)
            if source_checkpoints.exists():
                dest_checkpoints = package_dir / "checkpoints"
                shutil.copytree(source_checkpoints, dest_checkpoints)

        # Create reproduction script
        script_path = package_dir / "reproduce.py"
        script_path.write_text(self._generate_reproduction_script(exp_version))

        # Create requirements file
        requirements_path = package_dir / "requirements.txt"
        requirements_path.write_text(self._generate_requirements(exp_version))

        # Create README
        readme_path = package_dir / "README.md"
        readme_path.write_text(self._generate_readme(exp_version))

        # Create archive
        archive_path = export_path / f"{experiment_id}_v{version}.tar.gz"
        shutil.make_archive(str(archive_path.with_suffix("")), "gztar", package_dir)

        logger.info(f"Exported {experiment_id} v{version} to {archive_path}")

        return str(archive_path)

    def _generate_reproduction_script(self, version: ExperimentVersion) -> str:
        """Generate a script to reproduce the experiment"""
        return f'''#!/usr/bin/env python3
"""
Reproduction script for experiment {version.experiment_id} version {version.version}
Generated on {datetime.utcnow().isoformat()}
"""

import json
from pathlib import Path

def reproduce_experiment():
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # Load dataset info
    with open("dataset_info.json", "r") as f:
        dataset_info = json.load(f)

    print(f"Reproducing experiment {version.experiment_id} v{version.version}")
    print(f"Configuration: {{config}}")
    print(f"Dataset: {{dataset_info}}")

    # TODO: Add actual training code here
    # from fine_tuning import train_model
    # train_model(config, dataset_info["path"])

if __name__ == "__main__":
    reproduce_experiment()
'''

    def _generate_requirements(self, version: ExperimentVersion) -> str:
        """Generate requirements.txt for the experiment"""
        return """torch>=2.0.0
transformers>=4.30.0
datasets>=2.0.0
accelerate>=0.20.0
peft>=0.4.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
wandb>=0.15.0
"""

    def _generate_readme(self, version: ExperimentVersion) -> str:
        """Generate README for the exported version"""
        return f"""# Experiment: {version.experiment_id} - Version {version.version}

## Description
{version.description}

## Version Information
- Version: {version.version}
- Created: {version.created_at}
- Created By: {version.created_by}
- Parent Version: {version.parent_version or "None (baseline)"}

## Configuration
See `config.json` for full configuration details.

## Dataset
- Version: {version.dataset_version}
- See `dataset_info.json` for dataset details

## Reproduction
To reproduce this experiment:

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the reproduction script:
   ```bash
   python reproduce.py
   ```

## Checkpoints
Model checkpoints are included in the `checkpoints/` directory.

## Metrics
{json.dumps(version.metrics, indent=2) if version.metrics else "No metrics recorded yet"}
"""


# Example usage
if __name__ == "__main__":
    versioning = VersioningSystem()

    # Create an experiment version
    version = versioning.create_experiment_version(
        experiment_id="exp_001",
        config={"model": "llama-2-7b", "learning_rate": 2e-5, "batch_size": 4},
        dataset_path="/data/dataset.jsonl",
        created_by="user123",
        description="Initial experiment with base configuration",
    )

    print(f"Created version: {version.version}")

    # Save a checkpoint
    checkpoint = versioning.save_checkpoint(
        experiment_id="exp_001",
        version=version.version,
        epoch=5,
        checkpoint_path="/models/checkpoint.bin",
        metrics={"loss": 1.23, "accuracy": 0.89},
        is_best=True,
    )

    print(f"Saved checkpoint: {checkpoint.version}")
