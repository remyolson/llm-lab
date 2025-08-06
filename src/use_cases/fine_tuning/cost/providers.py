"""
Cloud Provider Pricing APIs

This module provides interfaces to fetch real-time pricing from cloud providers.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import boto3
from google.cloud import billing_v1

logger = logging.getLogger(__name__)


@dataclass
class PricingInfo:
    """Pricing information for a service."""

    service: str
    region: str
    instance_type: str
    price_per_hour: float
    currency: str = "USD"
    last_updated: datetime = None
    spot_price: Optional[float] = None

    @property
    def spot_savings_percent(self) -> float:
        """Spot instance savings percentage."""
        if self.spot_price is None:
            return 0.0
        return ((self.price_per_hour - self.spot_price) / self.price_per_hour) * 100


class BasePricingProvider(ABC):
    """Base class for pricing providers."""

    def __init__(self, cache_duration_hours: int = 24):
        """Initialize pricing provider."""
        self.cache_duration_hours = cache_duration_hours
        self._pricing_cache: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def get_compute_pricing(self, region: str, instance_type: str) -> Optional[PricingInfo]:
        """Get compute pricing for instance type."""
        pass

    @abstractmethod
    def get_storage_pricing(self, region: str, storage_type: str) -> Optional[float]:
        """Get storage pricing per GB/month."""
        pass

    @abstractmethod
    def get_network_pricing(self, region: str) -> Optional[float]:
        """Get network pricing per GB."""
        pass

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._pricing_cache:
            return False

        cached_time = self._pricing_cache[cache_key].get("timestamp")
        if not cached_time:
            return False

        expiry_time = cached_time + timedelta(hours=self.cache_duration_hours)
        return datetime.now() < expiry_time

    def _cache_pricing(self, cache_key: str, data: Any):
        """Cache pricing data."""
        self._pricing_cache[cache_key] = {"data": data, "timestamp": datetime.now()}

    def _get_cached_pricing(self, cache_key: str) -> Optional[Any]:
        """Get cached pricing data."""
        if self._is_cache_valid(cache_key):
            return self._pricing_cache[cache_key]["data"]
        return None


class AWSPricing(BasePricingProvider):
    """AWS pricing provider."""

    def __init__(self, cache_duration_hours: int = 24):
        """Initialize AWS pricing provider."""
        super().__init__(cache_duration_hours)
        self._ec2_client = None
        self._pricing_client = None

    def _get_ec2_client(self, region: str = "us-east-1"):
        """Get EC2 client."""
        if self._ec2_client is None:
            try:
                self._ec2_client = boto3.client("ec2", region_name=region)
            except Exception as e:
                logger.error(f"Failed to create EC2 client: {e}")
        return self._ec2_client

    def _get_pricing_client(self):
        """Get Pricing client."""
        if self._pricing_client is None:
            try:
                # Pricing API is only available in us-east-1
                self._pricing_client = boto3.client("pricing", region_name="us-east-1")
            except Exception as e:
                logger.error(f"Failed to create Pricing client: {e}")
        return self._pricing_client

    def get_compute_pricing(self, region: str, instance_type: str) -> Optional[PricingInfo]:
        """Get EC2 compute pricing."""
        cache_key = f"aws_compute_{region}_{instance_type}"
        cached = self._get_cached_pricing(cache_key)
        if cached:
            return cached

        try:
            pricing_client = self._get_pricing_client()
            if not pricing_client:
                return self._get_fallback_pricing(region, instance_type)

            # Get on-demand pricing
            response = pricing_client.get_products(
                ServiceCode="AmazonEC2",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                    {
                        "Type": "TERM_MATCH",
                        "Field": "location",
                        "Value": self._region_to_location(region),
                    },
                    {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                    {"Type": "TERM_MATCH", "Field": "operating-system", "Value": "Linux"},
                    {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                ],
            )

            if not response.get("PriceList"):
                return self._get_fallback_pricing(region, instance_type)

            # Parse pricing data
            price_list = json.loads(response["PriceList"][0])
            terms = price_list.get("terms", {}).get("OnDemand", {})

            if not terms:
                return self._get_fallback_pricing(region, instance_type)

            # Extract hourly price
            term_data = list(terms.values())[0]
            price_dimensions = term_data.get("priceDimensions", {})
            price_data = list(price_dimensions.values())[0]
            hourly_price = float(price_data.get("pricePerUnit", {}).get("USD", 0))

            # Get spot pricing
            spot_price = self._get_spot_pricing(region, instance_type)

            pricing_info = PricingInfo(
                service="EC2",
                region=region,
                instance_type=instance_type,
                price_per_hour=hourly_price,
                spot_price=spot_price,
                last_updated=datetime.now(),
            )

            self._cache_pricing(cache_key, pricing_info)
            return pricing_info

        except Exception as e:
            logger.error(f"Error fetching AWS pricing: {e}")
            return self._get_fallback_pricing(region, instance_type)

    def _get_spot_pricing(self, region: str, instance_type: str) -> Optional[float]:
        """Get current spot pricing."""
        try:
            ec2_client = self._get_ec2_client(region)
            if not ec2_client:
                return None

            response = ec2_client.describe_spot_price_history(
                InstanceTypes=[instance_type], ProductDescriptions=["Linux/UNIX"], MaxResults=1
            )

            if response.get("SpotPriceHistory"):
                return float(response["SpotPriceHistory"][0]["SpotPrice"])

        except Exception as e:
            logger.warning(f"Could not fetch spot pricing: {e}")

        return None

    def _region_to_location(self, region: str) -> str:
        """Convert AWS region to pricing location."""
        region_mapping = {
            "us-east-1": "US East (N. Virginia)",
            "us-east-2": "US East (Ohio)",
            "us-west-1": "US West (N. California)",
            "us-west-2": "US West (Oregon)",
            "eu-west-1": "Europe (Ireland)",
            "eu-central-1": "Europe (Frankfurt)",
            "ap-southeast-1": "Asia Pacific (Singapore)",
            "ap-northeast-1": "Asia Pacific (Tokyo)",
        }
        return region_mapping.get(region, "US East (N. Virginia)")

    def _get_fallback_pricing(self, region: str, instance_type: str) -> PricingInfo:
        """Get fallback pricing when API is unavailable."""
        # Static pricing data (updated as of 2024)
        fallback_prices = {
            "p3.2xlarge": {"hourly": 3.06, "spot": 0.92},
            "p3.8xlarge": {"hourly": 12.24, "spot": 3.67},
            "p3.16xlarge": {"hourly": 24.48, "spot": 7.34},
            "g4dn.xlarge": {"hourly": 0.526, "spot": 0.158},
            "g4dn.2xlarge": {"hourly": 0.752, "spot": 0.226},
            "g4dn.4xlarge": {"hourly": 1.204, "spot": 0.361},
            "g5.xlarge": {"hourly": 1.006, "spot": 0.302},
            "g5.2xlarge": {"hourly": 1.212, "spot": 0.364},
        }

        pricing = fallback_prices.get(instance_type, {"hourly": 1.0, "spot": 0.3})

        return PricingInfo(
            service="EC2",
            region=region,
            instance_type=instance_type,
            price_per_hour=pricing["hourly"],
            spot_price=pricing["spot"],
            last_updated=datetime.now(),
        )

    def get_storage_pricing(self, region: str, storage_type: str = "gp3") -> Optional[float]:
        """Get EBS storage pricing per GB/month."""
        # Static pricing (varies slightly by region, using us-east-1 as base)
        storage_prices = {"gp3": 0.08, "gp2": 0.10, "io2": 0.125, "st1": 0.045, "sc1": 0.025}
        return storage_prices.get(storage_type, 0.08)

    def get_network_pricing(self, region: str) -> Optional[float]:
        """Get data transfer pricing per GB."""
        # First 1 GB free, then $0.09/GB for most regions
        return 0.09


class GCPPricing(BasePricingProvider):
    """Google Cloud Platform pricing provider."""

    def __init__(self, cache_duration_hours: int = 24):
        """Initialize GCP pricing provider."""
        super().__init__(cache_duration_hours)
        self._billing_client = None

    def _get_billing_client(self):
        """Get Cloud Billing client."""
        if self._billing_client is None:
            try:
                self._billing_client = billing_v1.CloudCatalogClient()
            except Exception as e:
                logger.error(f"Failed to create GCP billing client: {e}")
        return self._billing_client

    def get_compute_pricing(self, region: str, instance_type: str) -> Optional[PricingInfo]:
        """Get Compute Engine pricing."""
        cache_key = f"gcp_compute_{region}_{instance_type}"
        cached = self._get_cached_pricing(cache_key)
        if cached:
            return cached

        # For now, use fallback pricing
        # TODO: Implement actual GCP pricing API calls
        return self._get_fallback_pricing(region, instance_type)

    def _get_fallback_pricing(self, region: str, instance_type: str) -> PricingInfo:
        """Get fallback pricing for GCP."""
        fallback_prices = {
            "n1-standard-4-t4": {"hourly": 0.35, "preemptible": 0.105},
            "n1-standard-8-v100": {"hourly": 2.48, "preemptible": 0.744},
            "a2-highgpu-1g": {"hourly": 2.93, "preemptible": 0.879},
            "a2-highgpu-2g": {"hourly": 5.85, "preemptible": 1.755},
            "a2-highgpu-4g": {"hourly": 11.70, "preemptible": 3.51},
        }

        pricing = fallback_prices.get(instance_type, {"hourly": 1.0, "preemptible": 0.3})

        return PricingInfo(
            service="Compute Engine",
            region=region,
            instance_type=instance_type,
            price_per_hour=pricing["hourly"],
            spot_price=pricing["preemptible"],
            last_updated=datetime.now(),
        )

    def get_storage_pricing(self, region: str, storage_type: str = "ssd") -> Optional[float]:
        """Get storage pricing per GB/month."""
        storage_prices = {"ssd": 0.17, "standard": 0.04, "nearline": 0.01, "coldline": 0.004}
        return storage_prices.get(storage_type, 0.17)

    def get_network_pricing(self, region: str) -> Optional[float]:
        """Get network egress pricing per GB."""
        # First 1 GB free per month, then varies by destination
        return 0.12


class AzurePricing(BasePricingProvider):
    """Microsoft Azure pricing provider."""

    def __init__(self, cache_duration_hours: int = 24):
        """Initialize Azure pricing provider."""
        super().__init__(cache_duration_hours)

    def get_compute_pricing(self, region: str, instance_type: str) -> Optional[PricingInfo]:
        """Get Azure VM pricing."""
        cache_key = f"azure_compute_{region}_{instance_type}"
        cached = self._get_cached_pricing(cache_key)
        if cached:
            return cached

        # Use fallback pricing
        return self._get_fallback_pricing(region, instance_type)

    def _get_fallback_pricing(self, region: str, instance_type: str) -> PricingInfo:
        """Get fallback pricing for Azure."""
        fallback_prices = {
            "NC6": {"hourly": 0.90, "spot": 0.27},
            "NC12": {"hourly": 1.80, "spot": 0.54},
            "NC24": {"hourly": 3.60, "spot": 1.08},
            "ND40rs_v2": {"hourly": 22.03, "spot": 6.61},
            "NC6s_v3": {"hourly": 3.06, "spot": 0.92},
            "NC12s_v3": {"hourly": 6.12, "spot": 1.84},
        }

        pricing = fallback_prices.get(instance_type, {"hourly": 1.0, "spot": 0.3})

        return PricingInfo(
            service="Virtual Machines",
            region=region,
            instance_type=instance_type,
            price_per_hour=pricing["hourly"],
            spot_price=pricing["spot"],
            last_updated=datetime.now(),
        )

    def get_storage_pricing(
        self, region: str, storage_type: str = "premium_ssd"
    ) -> Optional[float]:
        """Get storage pricing per GB/month."""
        storage_prices = {"premium_ssd": 0.15, "standard_ssd": 0.075, "standard_hdd": 0.045}
        return storage_prices.get(storage_type, 0.15)

    def get_network_pricing(self, region: str) -> Optional[float]:
        """Get bandwidth pricing per GB."""
        # First 5 GB free per month, then $0.087/GB
        return 0.087


class LocalPricing(BasePricingProvider):
    """Local/on-premises pricing provider."""

    def __init__(self, electricity_rate: float = 0.12, cache_duration_hours: int = 24):
        """Initialize local pricing provider."""
        super().__init__(cache_duration_hours)
        self.electricity_rate = electricity_rate  # per kWh

    def get_compute_pricing(self, region: str, instance_type: str) -> Optional[PricingInfo]:
        """Get local compute pricing based on power consumption."""
        # Estimate based on hardware specifications
        gpu_power_consumption = {
            "rtx_4090": 450,  # watts
            "rtx_3090": 350,
            "rtx_3080": 320,
            "rtx_2080_ti": 250,
            "v100": 300,
            "t4": 70,
            "a100": 400,
        }

        # Extract GPU type from instance_type (simplified)
        gpu_type = "rtx_3080"  # Default
        for gpu in gpu_power_consumption:
            if gpu in instance_type.lower():
                gpu_type = gpu
                break

        # Calculate hourly cost
        gpu_watts = gpu_power_consumption.get(gpu_type, 250)
        cpu_watts = 100  # Approximate CPU power
        total_watts = gpu_watts + cpu_watts

        hourly_kwh = total_watts / 1000
        electricity_cost = hourly_kwh * self.electricity_rate

        # Add depreciation/maintenance costs
        depreciation_cost = 0.10  # $0.10 per hour

        total_hourly_cost = electricity_cost + depreciation_cost

        return PricingInfo(
            service="Local Hardware",
            region=region,
            instance_type=instance_type,
            price_per_hour=total_hourly_cost,
            last_updated=datetime.now(),
        )

    def get_storage_pricing(self, region: str, storage_type: str = "ssd") -> Optional[float]:
        """Get local storage pricing per GB/month."""
        # Very low cost for local storage (mainly depreciation)
        return 0.001

    def get_network_pricing(self, region: str) -> Optional[float]:
        """Get local network pricing per GB."""
        # No cost for local network
        return 0.0


def get_pricing_provider(provider_name: str, **kwargs) -> BasePricingProvider:
    """Get pricing provider by name."""
    providers = {"aws": AWSPricing, "gcp": GCPPricing, "azure": AzurePricing, "local": LocalPricing}

    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown pricing provider: {provider_name}")

    return provider_class(**kwargs)


def compare_pricing_across_providers(
    instance_configs: List[Dict[str, str]], region: str = "us-east-1"
) -> List[PricingInfo]:
    """Compare pricing across multiple providers and instance types."""

    pricing_results = []

    for config in instance_configs:
        provider_name = config["provider"]
        instance_type = config["instance_type"]

        try:
            provider = get_pricing_provider(provider_name)
            pricing = provider.get_compute_pricing(region, instance_type)
            if pricing:
                pricing_results.append(pricing)
        except Exception as e:
            logger.error(f"Error getting pricing for {provider_name} {instance_type}: {e}")

    # Sort by price
    pricing_results.sort(key=lambda x: x.price_per_hour)

    return pricing_results
