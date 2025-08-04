#!/usr/bin/env python3
"""
LLM Lab Asset Downloader

This script downloads all required models and datasets for the LLM Lab project.
Run this after cloning the repository to set up all necessary assets.

Usage:
    python download_assets.py --all                    # Download everything
    python download_assets.py --models                 # Download only models
    python download_assets.py --datasets               # Download only datasets
    python download_assets.py --model qwen-0.5b        # Download specific model
    python download_assets.py --list                   # List available assets
"""

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import shutil


class AssetDownloader:
    """Downloads and manages model and dataset assets for LLM Lab."""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.downloads_dir = self.base_dir / "downloads"
        self.downloads_dir.mkdir(exist_ok=True)

        # Asset definitions with download URLs and metadata
        self.assets = {
            "models": {
                "qwen-0.5b": {
                    "description": "Qwen 0.5B parameter model (Safetensors format)",
                    "files": {
                        "model.safetensors": {
                            "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors",
                            "size_mb": 942,
                            "sha256": None,  # Optional checksum
                        }
                    },
                    "destination": "models/small-llms/qwen-0.5b/",
                },
                "qwen-0.5b-gguf": {
                    "description": "Qwen 0.5B parameter model (GGUF format, quantized)",
                    "files": {
                        "qwen2.5-0.5b-instruct-q4_k_m.gguf": {
                            "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                            "size_mb": 469,
                            "sha256": None,
                        }
                    },
                    "destination": "models/small-llms/qwen-0.5b-gguf/",
                },
                "smollm-135m": {
                    "description": "SmolLM 135M parameter model",
                    "files": {
                        "model.safetensors": {
                            "url": "https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct/resolve/main/model.safetensors",
                            "size_mb": 257,
                            "sha256": None,
                        }
                    },
                    "destination": "models/small-llms/smollm-135m/",
                },
                "smollm-360m": {
                    "description": "SmolLM 360M parameter model",
                    "files": {
                        "model.safetensors": {
                            "url": "https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct/resolve/main/model.safetensors",
                            "size_mb": 690,
                            "sha256": None,
                        }
                    },
                    "destination": "models/small-llms/smollm-360m/",
                },
            },
            "datasets": {
                "truthfulqa-full": {
                    "description": "Complete TruthfulQA dataset",
                    "files": {
                        "TruthfulQA.csv": {
                            "url": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
                            "size_mb": 1,
                            "sha256": None,
                        }
                    },
                    "destination": "datasets/benchmarking/raw/truthfulqa/data/",
                },
                "gsm8k-full": {
                    "description": "Complete GSM8K math reasoning dataset",
                    "files": {
                        "train.jsonl": {
                            "url": "https://raw.githubusercontent.com/openai/grade-school-math/main/grade_school_math/data/train.jsonl",
                            "size_mb": 8,
                            "sha256": None,
                        },
                        "test.jsonl": {
                            "url": "https://raw.githubusercontent.com/openai/grade-school-math/main/grade_school_math/data/test.jsonl",
                            "size_mb": 1,
                            "sha256": None,
                        },
                    },
                    "destination": "datasets/benchmarking/raw/gsm8k/data/",
                },
            },
        }

    def list_assets(self) -> None:
        """List all available assets."""
        print("📦 Available Assets for Download:\n")

        print("🤖 Models:")
        for model_id, info in self.assets["models"].items():
            total_size = sum(f["size_mb"] for f in info["files"].values())
            print(f"  • {model_id}: {info['description']} ({total_size}MB)")

        print("\n📊 Datasets:")
        for dataset_id, info in self.assets["datasets"].items():
            total_size = sum(f["size_mb"] for f in info["files"].values())
            print(f"  • {dataset_id}: {info['description']} ({total_size}MB)")

        print(
            f"\n💾 Total size if downloading everything: {self._calculate_total_size()}MB"
        )

    def _calculate_total_size(self) -> int:
        """Calculate total download size for all assets."""
        total = 0
        for category in self.assets.values():
            for asset in category.values():
                total += sum(f["size_mb"] for f in asset["files"].values())
        return total

    def _download_file(
        self, url: str, destination: Path, expected_size_mb: Optional[int] = None
    ) -> bool:
        """Download a single file with progress indication."""
        print(f"  Downloading {destination.name}...")

        try:
            # Create destination directory
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Download to temporary file first
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                tmp_path = Path(tmp_file.name)

                # Download with progress
                urllib.request.urlretrieve(url, tmp_path, self._progress_hook)

                # Verify size if provided
                if expected_size_mb:
                    actual_size_mb = tmp_path.stat().st_size / (1024 * 1024)
                    if (
                        abs(actual_size_mb - expected_size_mb) > expected_size_mb * 0.05
                    ):  # 5% tolerance
                        print(
                            f"    ⚠️  Size mismatch: expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB"
                        )

                # Move to final destination
                shutil.move(tmp_path, destination)
                print(f"    ✅ Downloaded {destination.name}")
                return True

        except Exception as e:
            print(f"    ❌ Failed to download {destination.name}: {e}")
            # Clean up temporary file if it exists
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()
            return False

    def _progress_hook(self, block_num: int, block_size: int, total_size: int) -> None:
        """Show download progress."""
        if total_size > 0:
            percent = min(100, (block_num * block_size / total_size) * 100)
            if block_num % 100 == 0:  # Update every 100 blocks to avoid spam
                print(f"    Progress: {percent:.1f}%", end="\r")

    def download_asset(self, category: str, asset_id: str) -> bool:
        """Download a specific asset."""
        if category not in self.assets or asset_id not in self.assets[category]:
            print(f"❌ Asset {category}/{asset_id} not found")
            return False

        asset = self.assets[category][asset_id]
        destination_base = self.base_dir / asset["destination"]

        print(f"📥 Downloading {category}/{asset_id}: {asset['description']}")

        success = True
        for filename, file_info in asset["files"].items():
            destination = destination_base / filename

            # Skip if file already exists and has correct size
            if destination.exists():
                actual_size_mb = destination.stat().st_size / (1024 * 1024)
                expected_size_mb = file_info["size_mb"]
                if (
                    abs(actual_size_mb - expected_size_mb) < expected_size_mb * 0.05
                ):  # 5% tolerance
                    print(f"  ✅ {filename} already exists with correct size")
                    continue
                else:
                    print(
                        f"  🔄 {filename} exists but size mismatch, re-downloading..."
                    )

            if not self._download_file(
                file_info["url"], destination, file_info["size_mb"]
            ):
                success = False

        return success

    def download_category(self, category: str) -> bool:
        """Download all assets in a category."""
        if category not in self.assets:
            print(f"❌ Category {category} not found")
            return False

        print(f"📦 Downloading all {category}...")
        success = True

        for asset_id in self.assets[category]:
            if not self.download_asset(category, asset_id):
                success = False
                print(f"❌ Failed to download {category}/{asset_id}")

        return success

    def download_all(self) -> bool:
        """Download all assets."""
        print("🚀 Downloading all assets for LLM Lab...")
        print(f"💾 Total download size: ~{self._calculate_total_size()}MB")
        print("⏱️  This may take a while depending on your internet connection.\n")

        success = True
        for category in self.assets:
            if not self.download_category(category):
                success = False

        if success:
            print("\n🎉 All assets downloaded successfully!")
            print("You can now run benchmarks and experiments.")
        else:
            print("\n⚠️  Some downloads failed. Check the errors above.")

        return success

    def verify_assets(self) -> Dict[str, Dict[str, bool]]:
        """Verify that all assets are present and have correct sizes."""
        print("🔍 Verifying downloaded assets...")

        results = {"models": {}, "datasets": {}}

        for category, assets in self.assets.items():
            for asset_id, asset_info in assets.items():
                destination_base = self.base_dir / asset_info["destination"]
                asset_valid = True

                for filename, file_info in asset_info["files"].items():
                    file_path = destination_base / filename
                    if not file_path.exists():
                        print(f"  ❌ Missing: {category}/{asset_id}/{filename}")
                        asset_valid = False
                    else:
                        # Check file size
                        actual_size_mb = file_path.stat().st_size / (1024 * 1024)
                        expected_size_mb = file_info["size_mb"]
                        if (
                            abs(actual_size_mb - expected_size_mb)
                            > expected_size_mb * 0.05
                        ):
                            print(
                                f"  ⚠️  Size mismatch: {category}/{asset_id}/{filename}"
                            )
                            print(
                                f"      Expected: ~{expected_size_mb}MB, Actual: {actual_size_mb:.1f}MB"
                            )
                            asset_valid = False
                        else:
                            print(f"  ✅ Valid: {category}/{asset_id}/{filename}")

                results[category][asset_id] = asset_valid

        return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Download models and datasets for LLM Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Download all assets")
    group.add_argument("--models", action="store_true", help="Download all models")
    group.add_argument("--datasets", action="store_true", help="Download all datasets")
    group.add_argument(
        "--model", type=str, help="Download specific model (e.g., qwen-0.5b)"
    )
    group.add_argument(
        "--dataset", type=str, help="Download specific dataset (e.g., truthfulqa-full)"
    )
    group.add_argument("--list", action="store_true", help="List available assets")
    group.add_argument("--verify", action="store_true", help="Verify downloaded assets")

    args = parser.parse_args()

    downloader = AssetDownloader()

    if args.list:
        downloader.list_assets()
        return

    if args.verify:
        results = downloader.verify_assets()
        all_valid = all(
            all(asset_results.values()) for asset_results in results.values()
        )
        sys.exit(0 if all_valid else 1)

    if args.all:
        success = downloader.download_all()
    elif args.models:
        success = downloader.download_category("models")
    elif args.datasets:
        success = downloader.download_category("datasets")
    elif args.model:
        success = downloader.download_asset("models", args.model)
    elif args.dataset:
        success = downloader.download_asset("datasets", args.dataset)
    else:
        parser.print_help()
        return

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
