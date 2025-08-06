"""
Deployment pipeline for fine-tuned models
Supports HuggingFace Hub, local serving, and cloud providers
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentProvider(Enum):
    HUGGINGFACE = "huggingface"
    LOCAL_VLLM = "local_vllm"
    LOCAL_TGI = "local_tgi"
    AWS_SAGEMAKER = "aws_sagemaker"
    AZURE_ML = "azure_ml"
    GCP_VERTEX = "gcp_vertex"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""

    provider: DeploymentProvider
    model_path: str
    model_name: str
    experiment_id: str
    checkpoint: str
    hardware_requirements: Dict[str, Any]
    environment_vars: Dict[str, str]
    scaling_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    rollback_enabled: bool = True
    auto_scale: bool = True


class ModelDeployer:
    """Base class for model deployment"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_id = None
        self.status = "initialized"

    async def deploy(self) -> Dict[str, Any]:
        """Deploy the model"""
        raise NotImplementedError

    async def rollback(self) -> bool:
        """Rollback deployment"""
        raise NotImplementedError

    async def health_check(self) -> Dict[str, Any]:
        """Check deployment health"""
        raise NotImplementedError

    async def get_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics"""
        raise NotImplementedError

    async def scale(self, replicas: int) -> bool:
        """Scale deployment"""
        raise NotImplementedError


class HuggingFaceDeployer(ModelDeployer):
    """Deploy models to HuggingFace Hub"""

    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.hub_token = os.getenv("HUGGINGFACE_TOKEN")

    async def deploy(self) -> Dict[str, Any]:
        """Deploy model to HuggingFace Hub"""
        try:
            self.status = "deploying"

            # Prepare model for upload
            model_dir = Path(self.config.model_path)

            # Create model card
            model_card = self._create_model_card()
            model_card_path = model_dir / "README.md"
            model_card_path.write_text(model_card)

            # Upload to HuggingFace Hub
            repo_name = f"{self.config.model_name}-{self.config.experiment_id}"

            # Simulate upload (in production, use huggingface_hub library)
            await asyncio.sleep(3)

            self.deployment_id = f"hf-{repo_name}"
            self.status = "deployed"

            return {
                "success": True,
                "deployment_id": self.deployment_id,
                "url": f"https://huggingface.co/models/{repo_name}",
                "status": self.status,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.status = "failed"
            logger.error(f"HuggingFace deployment failed: {e}")
            return {"success": False, "error": str(e), "status": self.status}

    def _create_model_card(self) -> str:
        """Create model card for HuggingFace"""
        return f"""---
tags:
- fine-tuned
- {self.config.model_name}
license: apache-2.0
language:
- en
metrics:
- perplexity
- accuracy
---

# {self.config.model_name}

This model was fine-tuned using the Fine-Tuning Studio.

## Model Details

- **Base Model**: {self.config.checkpoint}
- **Experiment ID**: {self.config.experiment_id}
- **Training Date**: {datetime.utcnow().isoformat()}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.config.model_name}")
tokenizer = AutoTokenizer.from_pretrained("{self.config.model_name}")

# Use the model
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0])
```

## Training Configuration

```json
{json.dumps(self.config.hardware_requirements, indent=2)}
```

## Performance Metrics

See the evaluation results in the model files.
"""

    async def rollback(self) -> bool:
        """Rollback HuggingFace deployment"""
        if not self.config.rollback_enabled:
            return False

        try:
            # Remove model from HuggingFace Hub
            # In production, use huggingface_hub library to delete repo
            await asyncio.sleep(1)

            self.status = "rolled_back"
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check HuggingFace deployment health"""
        return {
            "healthy": self.status == "deployed",
            "status": self.status,
            "endpoint": f"https://huggingface.co/models/{self.deployment_id}",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics from HuggingFace"""
        return {"downloads": 0, "likes": 0, "usage": "N/A"}

    async def scale(self, replicas: int) -> bool:
        """Scaling not applicable for HuggingFace Hub"""
        return True


class LocalVLLMDeployer(ModelDeployer):
    """Deploy models locally using vLLM"""

    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.process = None
        self.port = 8000

    async def deploy(self) -> Dict[str, Any]:
        """Deploy model using vLLM"""
        try:
            self.status = "deploying"

            # Prepare vLLM command
            cmd = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.config.model_path,
                "--port",
                str(self.port),
                "--host",
                "0.0.0.0",
            ]

            if self.config.hardware_requirements.get("gpu_memory"):
                cmd.extend(["--gpu-memory-utilization", "0.9"])

            if self.config.hardware_requirements.get("tensor_parallel"):
                cmd.extend(
                    [
                        "--tensor-parallel-size",
                        str(self.config.hardware_requirements["tensor_parallel"]),
                    ]
                )

            # Start vLLM server
            self.process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            # Wait for server to start
            await asyncio.sleep(5)

            self.deployment_id = f"vllm-{self.port}"
            self.status = "deployed"

            return {
                "success": True,
                "deployment_id": self.deployment_id,
                "endpoint": f"http://localhost:{self.port}",
                "status": self.status,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.status = "failed"
            logger.error(f"vLLM deployment failed: {e}")
            return {"success": False, "error": str(e), "status": self.status}

    async def rollback(self) -> bool:
        """Stop vLLM server"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self.status = "stopped"
            return True
        return False

    async def health_check(self) -> Dict[str, Any]:
        """Check vLLM server health"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{self.port}/health") as resp:
                    healthy = resp.status == 200

            return {
                "healthy": healthy,
                "status": self.status,
                "endpoint": f"http://localhost:{self.port}",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except:
            return {
                "healthy": False,
                "status": "unhealthy",
                "endpoint": f"http://localhost:{self.port}",
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get vLLM server metrics"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{self.port}/metrics") as resp:
                    metrics = await resp.json()

            return metrics
        except:
            return {"requestCount": 0, "avgLatency": 0, "throughput": 0}

    async def scale(self, replicas: int) -> bool:
        """Scaling not directly supported for single vLLM instance"""
        return False


class AWSSageMakerDeployer(ModelDeployer):
    """Deploy models to AWS SageMaker"""

    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.endpoint_name = None

    async def deploy(self) -> Dict[str, Any]:
        """Deploy model to SageMaker"""
        try:
            self.status = "deploying"

            # Create SageMaker endpoint configuration
            endpoint_config = {
                "ModelName": self.config.model_name,
                "InitialInstanceCount": self.config.scaling_config.get("min_instances", 1),
                "InstanceType": self.config.hardware_requirements.get(
                    "instance_type", "ml.g5.xlarge"
                ),
                "ModelDataUrl": self.config.model_path,
                "Environment": self.config.environment_vars,
            }

            # Simulate SageMaker deployment
            await asyncio.sleep(5)

            self.endpoint_name = f"{self.config.model_name}-endpoint"
            self.deployment_id = f"sagemaker-{self.endpoint_name}"
            self.status = "deployed"

            return {
                "success": True,
                "deployment_id": self.deployment_id,
                "endpoint": self.endpoint_name,
                "status": self.status,
                "region": "us-west-2",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.status = "failed"
            logger.error(f"SageMaker deployment failed: {e}")
            return {"success": False, "error": str(e), "status": self.status}

    async def rollback(self) -> bool:
        """Delete SageMaker endpoint"""
        try:
            # Delete endpoint using boto3
            await asyncio.sleep(2)
            self.status = "rolled_back"
            return True
        except:
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check SageMaker endpoint health"""
        return {
            "healthy": self.status == "deployed",
            "status": self.status,
            "endpoint": self.endpoint_name,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get SageMaker endpoint metrics from CloudWatch"""
        return {"invocations": 0, "modelLatency": 0, "overhead": 0, "errors": 0}

    async def scale(self, replicas: int) -> bool:
        """Update SageMaker endpoint instance count"""
        try:
            # Update endpoint configuration with new instance count
            await asyncio.sleep(1)
            return True
        except:
            return False


class DeploymentPipeline:
    """Main deployment pipeline orchestrator"""

    def __init__(self):
        self.deployers = {
            DeploymentProvider.HUGGINGFACE: HuggingFaceDeployer,
            DeploymentProvider.LOCAL_VLLM: LocalVLLMDeployer,
            DeploymentProvider.AWS_SAGEMAKER: AWSSageMakerDeployer,
        }
        self.active_deployments: Dict[str, ModelDeployer] = {}

    async def deploy_model(
        self,
        provider: str,
        model_path: str,
        model_name: str,
        experiment_id: str,
        checkpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Deploy a model to specified provider"""

        try:
            # Create deployment configuration
            config = DeploymentConfig(
                provider=DeploymentProvider(provider),
                model_path=model_path,
                model_name=model_name,
                experiment_id=experiment_id,
                checkpoint=checkpoint,
                hardware_requirements=kwargs.get("hardware_requirements", {}),
                environment_vars=kwargs.get("environment_vars", {}),
                scaling_config=kwargs.get("scaling_config", {}),
                monitoring_config=kwargs.get("monitoring_config", {}),
                rollback_enabled=kwargs.get("rollback_enabled", True),
                auto_scale=kwargs.get("auto_scale", True),
            )

            # Get appropriate deployer
            deployer_class = self.deployers.get(config.provider)
            if not deployer_class:
                raise ValueError(f"Unsupported provider: {provider}")

            deployer = deployer_class(config)

            # Run pre-deployment checks
            pre_check = await self._pre_deployment_checks(config)
            if not pre_check["success"]:
                return pre_check

            # Deploy the model
            result = await deployer.deploy()

            if result["success"]:
                self.active_deployments[result["deployment_id"]] = deployer

                # Run post-deployment tasks
                await self._post_deployment_tasks(deployer, result)

            return result

        except Exception as e:
            logger.error(f"Deployment pipeline error: {e}")
            return {"success": False, "error": str(e)}

    async def _pre_deployment_checks(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Run pre-deployment validation checks"""
        checks = {
            "model_exists": Path(config.model_path).exists(),
            "resources_available": True,  # Check GPU/CPU availability
            "credentials_valid": True,  # Validate cloud credentials
            "quota_available": True,  # Check cloud quotas
        }

        failed_checks = [k for k, v in checks.items() if not v]

        if failed_checks:
            return {
                "success": False,
                "error": f"Pre-deployment checks failed: {', '.join(failed_checks)}",
            }

        return {"success": True}

    async def _post_deployment_tasks(self, deployer: ModelDeployer, result: Dict[str, Any]):
        """Run post-deployment tasks"""
        # Set up monitoring
        if deployer.config.monitoring_config.get("enabled"):
            await self._setup_monitoring(deployer)

        # Configure auto-scaling
        if deployer.config.auto_scale:
            await self._setup_autoscaling(deployer)

        # Run initial health check
        health = await deployer.health_check()
        logger.info(f"Initial health check: {health}")

    async def _setup_monitoring(self, deployer: ModelDeployer):
        """Set up monitoring for deployment"""
        # Configure monitoring based on provider
        logger.info(f"Setting up monitoring for {deployer.deployment_id}")

    async def _setup_autoscaling(self, deployer: ModelDeployer):
        """Set up auto-scaling for deployment"""
        # Configure auto-scaling based on provider
        logger.info(f"Setting up auto-scaling for {deployer.deployment_id}")

    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment"""
        if deployment_id not in self.active_deployments:
            return False

        deployer = self.active_deployments[deployment_id]
        success = await deployer.rollback()

        if success:
            del self.active_deployments[deployment_id]

        return success

    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status and metrics"""
        if deployment_id not in self.active_deployments:
            return {"error": "Deployment not found"}

        deployer = self.active_deployments[deployment_id]

        health = await deployer.health_check()
        metrics = await deployer.get_metrics()

        return {
            "deployment_id": deployment_id,
            "health": health,
            "metrics": metrics,
            "config": {
                "provider": deployer.config.provider.value,
                "model_name": deployer.config.model_name,
                "auto_scale": deployer.config.auto_scale,
            },
        }

    async def scale_deployment(self, deployment_id: str, replicas: int) -> bool:
        """Scale a deployment"""
        if deployment_id not in self.active_deployments:
            return False

        deployer = self.active_deployments[deployment_id]
        return await deployer.scale(replicas)


# Deployment configuration templates
DEPLOYMENT_TEMPLATES = {
    "huggingface_standard": {
        "provider": "huggingface",
        "hardware_requirements": {},
        "environment_vars": {},
        "scaling_config": {},
        "monitoring_config": {"enabled": False},
    },
    "local_development": {
        "provider": "local_vllm",
        "hardware_requirements": {"gpu_memory": 16, "tensor_parallel": 1},
        "environment_vars": {},
        "scaling_config": {},
        "monitoring_config": {"enabled": True},
    },
    "production_aws": {
        "provider": "aws_sagemaker",
        "hardware_requirements": {"instance_type": "ml.g5.2xlarge"},
        "environment_vars": {},
        "scaling_config": {"min_instances": 1, "max_instances": 10, "target_utilization": 70},
        "monitoring_config": {"enabled": True, "alert_threshold": 90},
    },
}

if __name__ == "__main__":
    # Example usage
    async def main():
        pipeline = DeploymentPipeline()

        # Deploy to HuggingFace
        result = await pipeline.deploy_model(
            provider="huggingface",
            model_path="/path/to/model",
            model_name="my-fine-tuned-model",
            experiment_id="exp-123",
            checkpoint="llama-2-7b",
            **DEPLOYMENT_TEMPLATES["huggingface_standard"],
        )

        print(f"Deployment result: {result}")

        if result["success"]:
            # Check deployment status
            status = await pipeline.get_deployment_status(result["deployment_id"])
            print(f"Deployment status: {status}")

    asyncio.run(main())
