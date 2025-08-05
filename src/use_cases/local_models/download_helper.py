"""
Model download helper for local models.

This module provides functionality to download GGUF models from Hugging Face
with progress tracking, resume capability, and verification.
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, List
from urllib.parse import urlparse
import requests
from tqdm import tqdm
import click

from .model_configs import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Handles downloading and caching of GGUF models."""
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the model downloader.
        
        Args:
            cache_dir: Directory to store downloaded models
        """
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), ".cache", "llm-lab", "models")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "download_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load download metadata from cache."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save download metadata to cache."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get the path to a downloaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the model file if it exists, None otherwise
        """
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        model_info = MODEL_REGISTRY[model_name]
        model_path = self.cache_dir / model_info["filename"]
        
        if model_path.exists():
            return model_path
        
        return None
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded."""
        return self.get_model_path(model_name) is not None
    
    def download_model(self, model_name: str, force: bool = False) -> Path:
        """
        Download a model from Hugging Face.
        
        Args:
            model_name: Name of the model to download
            force: Force re-download even if exists
            
        Returns:
            Path to the downloaded model
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_info = MODEL_REGISTRY[model_name]
        model_path = self.cache_dir / model_info["filename"]
        
        # Check if already downloaded
        if model_path.exists() and not force:
            logger.info(f"Model already downloaded: {model_path}")
            return model_path
        
        # Download the model
        url = model_info["url"]
        logger.info(f"Downloading {model_info['name']} ({model_info['size']})...")
        
        try:
            self._download_file(url, model_path, model_name)
            logger.info(f"Successfully downloaded: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise
    
    def _download_file(self, url: str, dest_path: Path, model_name: str):
        """
        Download a file with progress bar and resume capability.
        
        Args:
            url: URL to download from
            dest_path: Destination file path
            model_name: Model name for metadata
        """
        # Check for partial download
        resume_pos = 0
        mode = 'wb'
        
        if dest_path.exists():
            resume_pos = dest_path.stat().st_size
            mode = 'ab'
        
        # Set up headers for resume
        headers = {}
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'
        
        # Make request
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Get total size
        total_size = int(response.headers.get('content-length', 0))
        if resume_pos > 0:
            total_size += resume_pos
        
        # Download with progress bar
        with open(dest_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=resume_pos,
                unit='B',
                unit_scale=True,
                desc=f"Downloading {model_name}"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Update metadata
        self.metadata[model_name] = {
            "path": str(dest_path),
            "size": dest_path.stat().st_size,
            "url": url
        }
        self._save_metadata()
    
    def verify_model(self, model_name: str) -> bool:
        """
        Verify a downloaded model (basic size check).
        
        Args:
            model_name: Name of the model to verify
            
        Returns:
            True if model appears valid
        """
        model_path = self.get_model_path(model_name)
        if not model_path:
            return False
        
        # Basic verification - check file size is reasonable
        min_size = 100 * 1024 * 1024  # 100MB minimum
        actual_size = model_path.stat().st_size
        
        if actual_size < min_size:
            logger.warning(f"Model file seems too small: {actual_size} bytes")
            return False
        
        return True
    
    def list_downloaded_models(self) -> List[str]:
        """List all downloaded models."""
        downloaded = []
        for model_name in MODEL_REGISTRY:
            if self.is_model_downloaded(model_name):
                downloaded.append(model_name)
        return downloaded
    
    def get_total_cache_size(self) -> int:
        """Get total size of cached models in bytes."""
        total = 0
        for model_file in self.cache_dir.glob("*.gguf"):
            total += model_file.stat().st_size
        return total
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache.
        
        Args:
            model_name: Specific model to remove, or None to clear all
        """
        if model_name:
            model_path = self.get_model_path(model_name)
            if model_path and model_path.exists():
                model_path.unlink()
                logger.info(f"Removed {model_name} from cache")
                if model_name in self.metadata:
                    del self.metadata[model_name]
                    self._save_metadata()
        else:
            # Clear all models
            for model_file in self.cache_dir.glob("*.gguf"):
                model_file.unlink()
            self.metadata = {}
            self._save_metadata()
            logger.info("Cleared all models from cache")


@click.command()
@click.option('--model', '-m', help='Specific model to download')
@click.option('--all', '-a', is_flag=True, help='Download all supported models')
@click.option('--list', '-l', 'list_models', is_flag=True, help='List available models')
@click.option('--status', '-s', is_flag=True, help='Show download status')
@click.option('--cache-dir', help='Custom cache directory')
@click.option('--force', '-f', is_flag=True, help='Force re-download')
@click.option('--clear-cache', is_flag=True, help='Clear model cache')
def main(model, all, list_models, status, cache_dir, force, clear_cache):
    """Download and manage local GGUF models for LLM Lab."""
    
    downloader = ModelDownloader(cache_dir)
    
    if list_models:
        click.echo("Available models:")
        for name, info in MODEL_REGISTRY.items():
            downloaded = "✓" if downloader.is_model_downloaded(name) else " "
            click.echo(f"  [{downloaded}] {name}: {info['name']} ({info['size']})")
        return
    
    if status:
        downloaded = downloader.list_downloaded_models()
        if downloaded:
            click.echo(f"Downloaded models: {', '.join(downloaded)}")
            size_mb = downloader.get_total_cache_size() / (1024 * 1024)
            click.echo(f"Total cache size: {size_mb:.1f} MB")
        else:
            click.echo("No models downloaded yet")
        return
    
    if clear_cache:
        if click.confirm("Clear all downloaded models?"):
            downloader.clear_cache()
            click.echo("Cache cleared")
        return
    
    if model:
        try:
            path = downloader.download_model(model, force)
            click.echo(f"Model ready at: {path}")
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            sys.exit(1)
    
    elif all:
        for model_name in MODEL_REGISTRY:
            if not downloader.is_model_downloaded(model_name) or force:
                try:
                    downloader.download_model(model_name, force)
                except Exception as e:
                    click.echo(f"Failed to download {model_name}: {str(e)}", err=True)
    
    else:
        click.echo("Use --help for usage information")


if __name__ == "__main__":
    main()