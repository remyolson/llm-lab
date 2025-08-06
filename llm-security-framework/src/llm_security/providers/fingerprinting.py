"""Model fingerprinting and identification system."""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ModelCapabilities:
    """Model capability information."""

    max_tokens: int
    supports_streaming: bool = False
    supports_functions: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_system_messages: bool = True
    context_window: Optional[int] = None
    training_cutoff: Optional[str] = None
    languages: List[str] = field(default_factory=lambda: ["en"])
    modalities: List[str] = field(default_factory=lambda: ["text"])


@dataclass
class ModelFingerprint:
    """Unique fingerprint for model identification."""

    provider: str
    model_name: str
    version: Optional[str] = None
    api_version: Optional[str] = None
    capabilities: Optional[ModelCapabilities] = None
    endpoint: Optional[str] = None
    deployment: Optional[str] = None  # For Azure
    fingerprint_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_verified: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate fingerprint hash after initialization."""
        if not self.fingerprint_hash:
            self.fingerprint_hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate unique hash for this fingerprint."""
        # Create deterministic string from key attributes
        components = [
            self.provider,
            self.model_name,
            self.version or "",
            self.api_version or "",
            self.endpoint or "",
            self.deployment or "",
        ]

        # Add capabilities if available
        if self.capabilities:
            cap_dict = asdict(self.capabilities)
            components.append(json.dumps(cap_dict, sort_keys=True))

        # Create hash
        fingerprint_string = "|".join(components)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        if self.last_verified:
            data["last_verified"] = self.last_verified.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelFingerprint":
        """Create from dictionary."""
        if "capabilities" in data and data["capabilities"]:
            data["capabilities"] = ModelCapabilities(**data["capabilities"])

        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        if "last_verified" in data and isinstance(data["last_verified"], str):
            data["last_verified"] = datetime.fromisoformat(data["last_verified"])

        return cls(**data)


class ModelRegistry:
    """Registry for managing model fingerprints and capabilities."""

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry file
        """
        self.registry_path = registry_path or Path("model_registry.json")
        self._fingerprints: Dict[str, ModelFingerprint] = {}
        self._provider_models: Dict[str, Set[str]] = {}

        # Load existing registry
        self.load_registry()

    def register_model(
        self,
        provider: str,
        model_name: str,
        capabilities: Optional[ModelCapabilities] = None,
        **kwargs,
    ) -> ModelFingerprint:
        """
        Register a new model or update existing one.

        Args:
            provider: Provider name
            model_name: Model name
            capabilities: Model capabilities
            **kwargs: Additional fingerprint attributes

        Returns:
            Created or updated ModelFingerprint
        """
        fingerprint = ModelFingerprint(
            provider=provider, model_name=model_name, capabilities=capabilities, **kwargs
        )

        # Store fingerprint
        self._fingerprints[fingerprint.fingerprint_hash] = fingerprint

        # Update provider index
        if provider not in self._provider_models:
            self._provider_models[provider] = set()
        self._provider_models[provider].add(model_name)

        logger.info(f"Registered model: {provider}/{model_name} ({fingerprint.fingerprint_hash})")
        return fingerprint

    def get_fingerprint(self, fingerprint_hash: str) -> Optional[ModelFingerprint]:
        """
        Get fingerprint by hash.

        Args:
            fingerprint_hash: Fingerprint hash

        Returns:
            ModelFingerprint or None
        """
        return self._fingerprints.get(fingerprint_hash)

    def find_fingerprints(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        capability_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ModelFingerprint]:
        """
        Find fingerprints matching criteria.

        Args:
            provider: Provider name to filter by
            model_name: Model name to filter by
            capability_filter: Capability requirements

        Returns:
            List of matching fingerprints
        """
        matches = []

        for fingerprint in self._fingerprints.values():
            # Provider filter
            if provider and fingerprint.provider != provider:
                continue

            # Model name filter
            if model_name and fingerprint.model_name != model_name:
                continue

            # Capability filter
            if capability_filter and fingerprint.capabilities:
                caps = asdict(fingerprint.capabilities)
                if not all(caps.get(key) == value for key, value in capability_filter.items()):
                    continue

            matches.append(fingerprint)

        return matches

    def create_fingerprint_from_adapter(
        self, adapter, test_capabilities: bool = True
    ) -> ModelFingerprint:
        """
        Create fingerprint from model adapter.

        Args:
            adapter: Model adapter instance
            test_capabilities: Whether to test capabilities

        Returns:
            ModelFingerprint
        """
        # Get basic info
        model_info = adapter.get_model_info()

        # Test capabilities if requested
        capabilities = None
        if test_capabilities:
            capabilities = self._test_model_capabilities(adapter)
        elif "capabilities" in model_info:
            cap_data = model_info["capabilities"]
            capabilities = ModelCapabilities(
                max_tokens=cap_data.get("max_tokens", 4096),
                supports_streaming=cap_data.get("streaming", False),
                supports_functions=cap_data.get("functions", False),
                supports_tools=cap_data.get("tools", False),
                supports_vision=cap_data.get("vision", False),
                supports_system_messages=cap_data.get("system_messages", True),
                context_window=cap_data.get("context_window"),
            )

        # Create fingerprint
        fingerprint = self.register_model(
            provider=adapter.provider_name,
            model_name=adapter.model_name,
            capabilities=capabilities,
            api_version=adapter.api_version,
            endpoint=getattr(adapter, "base_url", None),
            deployment=getattr(adapter, "deployment_name", None),
        )

        return fingerprint

    def _test_model_capabilities(self, adapter) -> ModelCapabilities:
        """
        Test model capabilities with actual requests.

        Args:
            adapter: Model adapter

        Returns:
            ModelCapabilities
        """
        capabilities = ModelCapabilities(max_tokens=4096)

        try:
            # Test basic functionality
            test_response = adapter.send_prompt("Hello")
            if test_response.is_success:
                capabilities.max_tokens = test_response.total_tokens or 4096

            # Test streaming (if adapter supports it)
            if hasattr(adapter, "stream_prompt"):
                try:
                    adapter.stream_prompt("Test", callback=lambda x: None)
                    capabilities.supports_streaming = True
                except:
                    pass

            # Test function calling
            if hasattr(adapter, "send_prompt"):
                try:
                    test_functions = [
                        {
                            "name": "test_function",
                            "description": "Test function",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]

                    response = adapter.send_prompt("Call a function", functions=test_functions)

                    if response.is_success and "function_call" in str(response.raw_response):
                        capabilities.supports_functions = True
                except:
                    pass

        except Exception as e:
            logger.warning(f"Error testing capabilities: {e}")

        return capabilities

    def verify_fingerprint(self, fingerprint_hash: str, adapter) -> bool:
        """
        Verify that adapter matches fingerprint.

        Args:
            fingerprint_hash: Fingerprint hash
            adapter: Model adapter

        Returns:
            True if matches, False otherwise
        """
        fingerprint = self.get_fingerprint(fingerprint_hash)
        if not fingerprint:
            return False

        # Check basic attributes
        if (
            fingerprint.provider != adapter.provider_name
            or fingerprint.model_name != adapter.model_name
        ):
            return False

        # Update last verified
        fingerprint.last_verified = datetime.now()

        return True

    def get_compatible_models(self, requirements: Dict[str, Any]) -> List[ModelFingerprint]:
        """
        Find models that meet requirements.

        Args:
            requirements: Capability requirements

        Returns:
            List of compatible models
        """
        compatible = []

        for fingerprint in self._fingerprints.values():
            if not fingerprint.capabilities:
                continue

            caps = asdict(fingerprint.capabilities)

            # Check all requirements
            is_compatible = True
            for req_key, req_value in requirements.items():
                if req_key == "min_tokens":
                    if caps.get("max_tokens", 0) < req_value:
                        is_compatible = False
                        break
                elif req_key in caps:
                    if isinstance(req_value, bool) and not caps[req_key]:
                        is_compatible = False
                        break
                    elif isinstance(req_value, (int, float)) and caps[req_key] < req_value:
                        is_compatible = False
                        break

            if is_compatible:
                compatible.append(fingerprint)

        # Sort by capabilities (more capable models first)
        compatible.sort(
            key=lambda fp: (
                fp.capabilities.max_tokens,
                fp.capabilities.supports_functions,
                fp.capabilities.supports_streaming,
            ),
            reverse=True,
        )

        return compatible

    def load_registry(self) -> None:
        """Load registry from file."""
        if not self.registry_path.exists():
            logger.info("No existing registry found, starting fresh")
            return

        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)

            # Load fingerprints
            for fp_data in data.get("fingerprints", []):
                fingerprint = ModelFingerprint.from_dict(fp_data)
                self._fingerprints[fingerprint.fingerprint_hash] = fingerprint

                # Update provider index
                provider = fingerprint.provider
                if provider not in self._provider_models:
                    self._provider_models[provider] = set()
                self._provider_models[provider].add(fingerprint.model_name)

            logger.info(f"Loaded {len(self._fingerprints)} fingerprints from registry")

        except Exception as e:
            logger.error(f"Error loading registry: {e}")

    def save_registry(self) -> None:
        """Save registry to file."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "fingerprints": [fp.to_dict() for fp in self._fingerprints.values()],
            }

            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self._fingerprints)} fingerprints to registry")

        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        provider_counts = {
            provider: len(models) for provider, models in self._provider_models.items()
        }

        capability_stats = {
            "streaming": 0,
            "functions": 0,
            "tools": 0,
            "vision": 0,
        }

        for fp in self._fingerprints.values():
            if fp.capabilities:
                if fp.capabilities.supports_streaming:
                    capability_stats["streaming"] += 1
                if fp.capabilities.supports_functions:
                    capability_stats["functions"] += 1
                if fp.capabilities.supports_tools:
                    capability_stats["tools"] += 1
                if fp.capabilities.supports_vision:
                    capability_stats["vision"] += 1

        return {
            "total_models": len(self._fingerprints),
            "providers": len(self._provider_models),
            "provider_counts": provider_counts,
            "capability_counts": capability_stats,
        }
