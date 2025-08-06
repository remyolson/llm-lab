"""
Cost/Benefit Analyzer for Fine-Tuning Evaluation

This module provides comprehensive cost/benefit analysis leveraging existing
cost_analysis.py from Task 2, including fine-tuning cost calculations,
ROI analysis, break-even analysis, predictive cost modeling, decision matrices,
and multi-provider cost comparison.

Example:
    analyzer = CostBenefitAnalyzer()

    # Analyze fine-tuning cost/benefit
    analysis = analyzer.analyze_fine_tuning(
        comparison_result=comparison,
        training_hours=4.5,
        gpu_cost_per_hour=2.5
    )

    # Generate ROI projection
    roi = analyzer.calculate_roi(
        analysis,
        expected_usage_per_day=1000,
        days=365
    )
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Import plotting
import plotly.graph_objects as go

# Import evaluation components
from ..benchmark_runner import BenchmarkResult, ComparisonResult

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for fine-tuning."""

    compute_cost: float
    storage_cost: float
    data_preparation_cost: float
    evaluation_cost: float
    total_cost: float
    currency: str = "USD"
    breakdown_details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str | Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ROIAnalysis:
    """Return on Investment analysis."""

    initial_investment: float
    monthly_savings: float
    break_even_months: float
    one_year_roi: float
    three_year_roi: float
    net_present_value: float
    internal_rate_return: float
    payback_period_months: float

    def to_dict(self) -> Dict[str | Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CostProjection:
    """Cost projection over time."""

    time_periods: List[str]
    baseline_costs: List[float]
    fine_tuned_costs: List[float]
    cumulative_savings: List[float]
    break_even_point: Optional[int] = None

    def to_dict(self) -> Dict[str | Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProviderComparison:
    """Multi-provider cost comparison."""

    provider_name: str
    base_model_cost_per_1k: float
    fine_tuning_cost: float
    inference_cost_per_1k: float
    monthly_cost_estimate: float
    annual_cost_estimate: float
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)


@dataclass
class CostAnalysisConfig:
    """Configuration for cost analysis."""

    # GPU costs
    gpu_cost_per_hour: float = 2.5  # Default A100 cost
    cpu_cost_per_hour: float = 0.1
    storage_cost_per_gb_month: float = 0.1

    # Model serving costs
    serving_cost_per_hour: float = 0.5
    bandwidth_cost_per_gb: float = 0.12

    # Provider costs (per 1K tokens)
    provider_costs: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "openai": {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            },
            "anthropic": {
                "claude-2": {"input": 0.008, "output": 0.024},
                "claude-instant": {"input": 0.0008, "output": 0.0024},
            },
            "google": {
                "gemini-pro": {"input": 0.00025, "output": 0.0005},
                "palm-2": {"input": 0.0005, "output": 0.001},
            },
        }
    )

    # Business parameters
    discount_rate: float = 0.1  # 10% annual
    inflation_rate: float = 0.03  # 3% annual

    # Usage estimates
    daily_requests: int = 10000
    avg_input_tokens: int = 500
    avg_output_tokens: int = 200


class CostBenefitAnalyzer:
    """Comprehensive cost/benefit analyzer for fine-tuning."""

    def __init__(self, config: Optional[CostAnalysisConfig] = None):
        """Initialize analyzer.

        Args:
            config: Cost analysis configuration
        """
        self.config = config or CostAnalysisConfig()
        self.analysis_cache = {}

    def analyze_fine_tuning(
        self,
        comparison_result: ComparisonResult,
        training_hours: float,
        gpu_type: str = "A100",
        data_size_gb: float = 1.0,
        num_experiments: int = 1,
    ) -> Dict[str | Any]:
        """Analyze cost/benefit of fine-tuning.

        Args:
            comparison_result: Comparison between base and fine-tuned models
            training_hours: Hours spent training
            gpu_type: Type of GPU used
            data_size_gb: Size of training data in GB
            num_experiments: Number of experiments/iterations

        Returns:
            Comprehensive cost/benefit analysis
        """
        # Calculate costs
        cost_breakdown = self._calculate_training_costs(
            training_hours, gpu_type, data_size_gb, num_experiments
        )

        # Calculate benefits
        performance_benefits = self._calculate_performance_benefits(comparison_result)

        # Calculate efficiency gains
        efficiency_gains = self._calculate_efficiency_gains(comparison_result)

        # ROI analysis
        roi_analysis = self.calculate_roi(cost_breakdown, performance_benefits, efficiency_gains)

        # Cost projections
        projections = self._generate_cost_projections(
            cost_breakdown, performance_benefits, efficiency_gains
        )

        # Decision matrix
        decision_matrix = self._create_decision_matrix(
            cost_breakdown, performance_benefits, roi_analysis
        )

        return {
            "cost_breakdown": cost_breakdown,
            "performance_benefits": performance_benefits,
            "efficiency_gains": efficiency_gains,
            "roi_analysis": roi_analysis,
            "projections": projections,
            "decision_matrix": decision_matrix,
            "recommendation": self._generate_recommendation(roi_analysis),
        }

    def _calculate_training_costs(
        self, training_hours: float, gpu_type: str, data_size_gb: float, num_experiments: int
    ) -> CostBreakdown:
        """Calculate detailed training costs.

        Args:
            training_hours: Total training hours
            gpu_type: GPU type used
            data_size_gb: Training data size
            num_experiments: Number of experiments

        Returns:
            Cost breakdown
        """
        # GPU costs
        gpu_costs = {
            "A100": 2.5,
            "V100": 1.5,
            "T4": 0.5,
            "RTX3090": 0.8,
            "M1": 0.0,  # Local machine
        }

        gpu_cost_per_hour = gpu_costs.get(gpu_type, self.config.gpu_cost_per_hour)
        compute_cost = training_hours * gpu_cost_per_hour * num_experiments

        # Storage costs
        storage_months = 1  # Assume 1 month of storage
        storage_cost = data_size_gb * self.config.storage_cost_per_gb_month * storage_months

        # Data preparation costs (estimate 20% of compute time)
        data_prep_hours = training_hours * 0.2
        data_preparation_cost = data_prep_hours * self.config.cpu_cost_per_hour

        # Evaluation costs (estimate 10% of training time)
        eval_hours = training_hours * 0.1
        evaluation_cost = eval_hours * gpu_cost_per_hour

        # Total cost
        total_cost = compute_cost + storage_cost + data_preparation_cost + evaluation_cost

        return CostBreakdown(
            compute_cost=compute_cost,
            storage_cost=storage_cost,
            data_preparation_cost=data_preparation_cost,
            evaluation_cost=evaluation_cost,
            total_cost=total_cost,
            breakdown_details={
                "gpu_type": gpu_type,
                "training_hours": training_hours,
                "num_experiments": num_experiments,
                "data_size_gb": data_size_gb,
                "gpu_cost_per_hour": gpu_cost_per_hour,
            },
        )

    def _calculate_performance_benefits(
        self, comparison_result: ComparisonResult
    ) -> Dict[str | Any]:
        """Calculate performance benefits from fine-tuning.

        Args:
            comparison_result: Model comparison results

        Returns:
            Performance benefits analysis
        """
        # Average improvement
        improvements = []
        for benchmark, imp in comparison_result.improvements.items():
            if isinstance(imp, dict) and "improvement_pct" in imp:
                improvements.append(imp["improvement_pct"])

        avg_improvement = np.mean(improvements) if improvements else 0

        # Quality improvements
        quality_metrics = {
            "accuracy_improvement": avg_improvement,
            "benchmarks_improved": len([i for i in improvements if i > 0]),
            "benchmarks_total": len(improvements),
            "max_improvement": max(improvements) if improvements else 0,
            "min_improvement": min(improvements) if improvements else 0,
        }

        # Estimate business impact
        business_impact = self._estimate_business_impact(avg_improvement)

        return {
            "quality_metrics": quality_metrics,
            "business_impact": business_impact,
            "improvements_by_benchmark": comparison_result.improvements,
        }

    def _calculate_efficiency_gains(self, comparison_result: ComparisonResult) -> Dict[str | Any]:
        """Calculate efficiency gains from fine-tuning.

        Args:
            comparison_result: Model comparison results

        Returns:
            Efficiency gains analysis
        """
        # Model size comparison (if available)
        base_model = comparison_result.base_result.model_version
        ft_model = comparison_result.fine_tuned_result.model_version

        # Inference speed comparison
        base_time = comparison_result.base_result.duration_seconds
        ft_time = comparison_result.fine_tuned_result.duration_seconds

        speed_improvement = ((base_time - ft_time) / base_time * 100) if base_time > 0 else 0

        # Token efficiency (if using smaller model after fine-tuning)
        token_efficiency = {
            "base_model": base_model.model_path,
            "fine_tuned_model": ft_model.model_path,
            "inference_speed_improvement": speed_improvement,
            "reduced_token_usage": 0,  # Would need actual token counts
        }

        # Cost per request comparison
        base_cost_per_request = self._estimate_cost_per_request(base_model.model_path)
        ft_cost_per_request = self._estimate_cost_per_request(
            ft_model.model_path, is_fine_tuned=True
        )

        cost_reduction = (
            ((base_cost_per_request - ft_cost_per_request) / base_cost_per_request * 100)
            if base_cost_per_request > 0
            else 0
        )

        return {
            "token_efficiency": token_efficiency,
            "cost_per_request": {
                "base": base_cost_per_request,
                "fine_tuned": ft_cost_per_request,
                "reduction_pct": cost_reduction,
            },
            "monthly_savings": self._calculate_monthly_savings(
                base_cost_per_request, ft_cost_per_request
            ),
        }

    def calculate_roi(
        self,
        cost_breakdown: CostBreakdown,
        performance_benefits: Dict[str, Any],
        efficiency_gains: Dict[str, Any],
    ) -> ROIAnalysis:
        """Calculate return on investment.

        Args:
            cost_breakdown: Training cost breakdown
            performance_benefits: Performance benefits
            efficiency_gains: Efficiency gains

        Returns:
            ROI analysis
        """
        initial_investment = cost_breakdown.total_cost

        # Monthly savings from efficiency gains
        monthly_savings = efficiency_gains.get("monthly_savings", 0)

        # Additional value from quality improvements
        quality_value = self._monetize_quality_improvements(
            performance_benefits["quality_metrics"]["accuracy_improvement"]
        )

        total_monthly_benefit = monthly_savings + quality_value / 12

        # Break-even analysis
        if total_monthly_benefit > 0:
            break_even_months = initial_investment / total_monthly_benefit
        else:
            break_even_months = float("inf")

        # ROI calculations
        one_year_benefit = total_monthly_benefit * 12
        three_year_benefit = total_monthly_benefit * 36

        one_year_roi = (
            ((one_year_benefit - initial_investment) / initial_investment * 100)
            if initial_investment > 0
            else 0
        )

        three_year_roi = (
            ((three_year_benefit - initial_investment) / initial_investment * 100)
            if initial_investment > 0
            else 0
        )

        # NPV calculation
        npv = self._calculate_npv(
            initial_investment, [total_monthly_benefit] * 36, self.config.discount_rate / 12
        )

        # IRR calculation (simplified)
        irr = self._calculate_irr(initial_investment, [total_monthly_benefit] * 36)

        return ROIAnalysis(
            initial_investment=initial_investment,
            monthly_savings=total_monthly_benefit,
            break_even_months=break_even_months,
            one_year_roi=one_year_roi,
            three_year_roi=three_year_roi,
            net_present_value=npv,
            internal_rate_return=irr,
            payback_period_months=break_even_months,
        )

    def _generate_cost_projections(
        self,
        cost_breakdown: CostBreakdown,
        performance_benefits: Dict[str, Any],
        efficiency_gains: Dict[str, Any],
        months: int = 24,
    ) -> CostProjection:
        """Generate cost projections over time.

        Args:
            cost_breakdown: Training costs
            performance_benefits: Performance benefits
            efficiency_gains: Efficiency gains
            months: Number of months to project

        Returns:
            Cost projections
        """
        time_periods = [f"Month {i + 1}" for i in range(months)]

        # Calculate monthly costs
        base_monthly_cost = self._calculate_monthly_cost("base")
        ft_monthly_cost = self._calculate_monthly_cost("fine_tuned")

        # Add serving costs for fine-tuned model
        ft_serving_cost = self.config.serving_cost_per_hour * 24 * 30  # 24/7 serving
        ft_monthly_cost += ft_serving_cost

        baseline_costs = [base_monthly_cost] * months
        fine_tuned_costs = [ft_monthly_cost] * months

        # Add initial investment to first month
        fine_tuned_costs[0] += cost_breakdown.total_cost

        # Calculate cumulative savings
        cumulative_savings = []
        total_saved = 0
        break_even_point = None

        for i in range(months):
            monthly_saving = baseline_costs[i] - fine_tuned_costs[i]
            total_saved += monthly_saving
            cumulative_savings.append(total_saved)

            if break_even_point is None and total_saved > 0:
                break_even_point = i + 1

        return CostProjection(
            time_periods=time_periods,
            baseline_costs=baseline_costs,
            fine_tuned_costs=fine_tuned_costs,
            cumulative_savings=cumulative_savings,
            break_even_point=break_even_point,
        )

    def compare_providers(
        self, model_size: str = "7B", monthly_requests: int = 100000
    ) -> List[ProviderComparison]:
        """Compare costs across multiple providers.

        Args:
            model_size: Model size category
            monthly_requests: Expected monthly requests

        Returns:
            List of provider comparisons
        """
        comparisons = []

        # Define provider profiles
        provider_profiles = {
            "openai": {
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "fine_tuning_available": True,
                "fine_tuning_cost_multiplier": 3.0,
                "pros": ["High quality", "Easy API", "Good documentation"],
                "cons": ["Higher cost", "Rate limits", "Less control"],
            },
            "anthropic": {
                "models": ["claude-instant", "claude-2"],
                "fine_tuning_available": False,
                "pros": ["Strong reasoning", "Large context", "Safety focused"],
                "cons": ["No fine-tuning", "Limited availability"],
            },
            "google": {
                "models": ["gemini-pro", "palm-2"],
                "fine_tuning_available": True,
                "fine_tuning_cost_multiplier": 2.5,
                "pros": ["Cost effective", "Good performance", "GCP integration"],
                "cons": ["Newer platform", "Limited models"],
            },
            "self_hosted": {
                "models": ["llama-2-7b", "mistral-7b"],
                "fine_tuning_available": True,
                "fine_tuning_cost_multiplier": 1.0,
                "pros": ["Full control", "No rate limits", "Privacy"],
                "cons": ["Infrastructure needed", "Maintenance required"],
            },
        }

        for provider, profile in provider_profiles.items():
            if provider == "self_hosted":
                # Calculate self-hosted costs
                base_cost_per_1k = self._calculate_self_hosted_cost(model_size)
                fine_tuning_cost = 500  # Estimated GPU hours * cost
                inference_cost_per_1k = base_cost_per_1k * 0.7  # Assume optimization
            else:
                # Use provider costs
                model = profile["models"][0]
                if (
                    provider in self.config.provider_costs
                    and model in self.config.provider_costs[provider]
                ):
                    costs = self.config.provider_costs[provider][model]
                    base_cost_per_1k = (costs["input"] + costs["output"]) / 2

                    if profile["fine_tuning_available"]:
                        fine_tuning_cost = 1000 * profile["fine_tuning_cost_multiplier"]
                        inference_cost_per_1k = base_cost_per_1k * 0.8
                    else:
                        fine_tuning_cost = 0
                        inference_cost_per_1k = base_cost_per_1k
                else:
                    continue

            # Calculate monthly costs
            tokens_per_request = self.config.avg_input_tokens + self.config.avg_output_tokens
            monthly_tokens = monthly_requests * tokens_per_request
            monthly_cost = (monthly_tokens / 1000) * inference_cost_per_1k

            comparisons.append(
                ProviderComparison(
                    provider_name=provider,
                    base_model_cost_per_1k=base_cost_per_1k,
                    fine_tuning_cost=fine_tuning_cost,
                    inference_cost_per_1k=inference_cost_per_1k,
                    monthly_cost_estimate=monthly_cost,
                    annual_cost_estimate=monthly_cost * 12,
                    pros=profile.get("pros", []),
                    cons=profile.get("cons", []),
                )
            )

        return sorted(comparisons, key=lambda x: x.annual_cost_estimate)

    def _create_decision_matrix(
        self,
        cost_breakdown: CostBreakdown,
        performance_benefits: Dict[str, Any],
        roi_analysis: ROIAnalysis,
    ) -> Dict[str | Any]:
        """Create decision matrix for fine-tuning.

        Args:
            cost_breakdown: Cost breakdown
            performance_benefits: Performance benefits
            roi_analysis: ROI analysis

        Returns:
            Decision matrix with recommendations
        """
        # Define criteria and weights
        criteria = {
            "performance_improvement": {
                "weight": 0.3,
                "value": performance_benefits["quality_metrics"]["accuracy_improvement"],
                "threshold": 5.0,  # 5% improvement threshold
            },
            "roi": {
                "weight": 0.25,
                "value": roi_analysis.one_year_roi,
                "threshold": 50.0,  # 50% ROI threshold
            },
            "break_even_time": {
                "weight": 0.2,
                "value": -roi_analysis.break_even_months,  # Negative for lower is better
                "threshold": -12,  # 12 months threshold
            },
            "initial_cost": {
                "weight": 0.15,
                "value": -cost_breakdown.total_cost,  # Negative for lower is better
                "threshold": -10000,  # $10k threshold
            },
            "complexity": {
                "weight": 0.1,
                "value": 70,  # Estimated complexity score (0-100)
                "threshold": 50,
            },
        }

        # Calculate weighted score
        total_score = 0
        for criterion, data in criteria.items():
            normalized_value = (
                data["value"] / abs(data["threshold"]) if data["threshold"] != 0 else 0
            )
            weighted_score = normalized_value * data["weight"]
            total_score += weighted_score

            data["normalized_score"] = normalized_value
            data["weighted_score"] = weighted_score
            data["meets_threshold"] = (
                data["value"] >= data["threshold"]
                if data["threshold"] > 0
                else data["value"] <= data["threshold"]
            )

        # Generate recommendation
        recommendation_score = total_score * 100  # Convert to percentage

        if recommendation_score >= 70:
            recommendation = "Strongly Recommended"
            confidence = "High"
        elif recommendation_score >= 50:
            recommendation = "Recommended"
            confidence = "Medium"
        elif recommendation_score >= 30:
            recommendation = "Consider with Caution"
            confidence = "Low"
        else:
            recommendation = "Not Recommended"
            confidence = "High"

        return {
            "criteria": criteria,
            "total_score": recommendation_score,
            "recommendation": recommendation,
            "confidence": confidence,
            "key_factors": self._identify_key_factors(criteria),
        }

    def _identify_key_factors(self, criteria: Dict[str, Any]) -> List[str]:
        """Identify key factors in decision.

        Args:
            criteria: Decision criteria

        Returns:
            List of key factors
        """
        key_factors = []

        for criterion, data in criteria.items():
            if data["meets_threshold"]:
                if criterion == "performance_improvement":
                    key_factors.append(f"Strong performance gains ({data['value']:.1f}%)")
                elif criterion == "roi":
                    key_factors.append(f"Good ROI ({data['value']:.1f}%)")
                elif criterion == "break_even_time":
                    key_factors.append(f"Quick payback ({-data['value']:.1f} months)")
            else:
                if criterion == "performance_improvement":
                    key_factors.append(f"Limited performance gains ({data['value']:.1f}%)")
                elif criterion == "initial_cost":
                    key_factors.append(f"High initial investment (${-data['value']:.0f})")

        return key_factors

    def _estimate_business_impact(self, accuracy_improvement: float) -> Dict[str | float]:
        """Estimate business impact of accuracy improvements.

        Args:
            accuracy_improvement: Percentage improvement in accuracy

        Returns:
            Business impact estimates
        """
        # These are example calculations - should be customized per use case

        # Customer satisfaction improvement
        satisfaction_improvement = accuracy_improvement * 0.5  # Conservative estimate

        # Error reduction
        error_reduction = accuracy_improvement * 1.5  # Errors reduce more than accuracy improves

        # Productivity gains (if used internally)
        productivity_gain = accuracy_improvement * 0.3

        # Revenue impact (very rough estimate)
        revenue_impact_pct = accuracy_improvement * 0.1

        return {
            "satisfaction_improvement_pct": satisfaction_improvement,
            "error_reduction_pct": error_reduction,
            "productivity_gain_pct": productivity_gain,
            "revenue_impact_pct": revenue_impact_pct,
        }

    def _estimate_cost_per_request(self, model_name: str, is_fine_tuned: bool = False) -> float:
        """Estimate cost per request for a model.

        Args:
            model_name: Model name
            is_fine_tuned: Whether model is fine-tuned

        Returns:
            Cost per request
        """
        # Simplified estimation
        tokens_per_request = self.config.avg_input_tokens + self.config.avg_output_tokens

        if "gpt" in model_name.lower():
            provider = "openai"
            base_model = "gpt-3.5-turbo" if "3.5" in model_name else "gpt-4"
        elif "claude" in model_name.lower():
            provider = "anthropic"
            base_model = "claude-instant" if "instant" in model_name else "claude-2"
        elif "gemini" in model_name.lower() or "palm" in model_name.lower():
            provider = "google"
            base_model = "gemini-pro" if "gemini" in model_name else "palm-2"
        else:
            # Self-hosted
            return self._calculate_self_hosted_cost_per_request(tokens_per_request)

        if (
            provider in self.config.provider_costs
            and base_model in self.config.provider_costs[provider]
        ):
            costs = self.config.provider_costs[provider][base_model]
            input_cost = (self.config.avg_input_tokens / 1000) * costs["input"]
            output_cost = (self.config.avg_output_tokens / 1000) * costs["output"]

            total_cost = input_cost + output_cost

            if is_fine_tuned:
                # Fine-tuned models often have slightly lower inference costs
                total_cost *= 0.8

            return total_cost

        return 0.001  # Default fallback

    def _calculate_self_hosted_cost(self, model_size: str) -> float:
        """Calculate self-hosted cost per 1K tokens.

        Args:
            model_size: Model size (7B, 13B, etc.)

        Returns:
            Cost per 1K tokens
        """
        # Estimate based on model size and hardware requirements
        size_to_gpu_hours = {
            "7B": 0.5,  # Can run on single GPU
            "13B": 1.0,  # Needs larger GPU
            "30B": 2.0,  # Multiple GPUs
            "70B": 4.0,  # Multiple high-end GPUs
        }

        gpu_hours = size_to_gpu_hours.get(model_size, 1.0)

        # Assume processing 10K tokens per GPU hour
        cost_per_1k = (gpu_hours * self.config.gpu_cost_per_hour) / 10

        return cost_per_1k

    def _calculate_self_hosted_cost_per_request(self, tokens: int) -> float:
        """Calculate self-hosted cost per request.

        Args:
            tokens: Total tokens in request

        Returns:
            Cost per request
        """
        # Assume 7B model by default
        cost_per_1k = self._calculate_self_hosted_cost("7B")
        return (tokens / 1000) * cost_per_1k

    def _calculate_monthly_cost(self, model_type: str) -> float:
        """Calculate monthly inference cost.

        Args:
            model_type: "base" or "fine_tuned"

        Returns:
            Monthly cost
        """
        requests_per_month = self.config.daily_requests * 30

        if model_type == "base":
            cost_per_request = self._estimate_cost_per_request("gpt-3.5-turbo")
        else:
            cost_per_request = self._estimate_cost_per_request("gpt-3.5-turbo", is_fine_tuned=True)

        return requests_per_month * cost_per_request

    def _calculate_monthly_savings(
        self, base_cost_per_request: float, ft_cost_per_request: float
    ) -> float:
        """Calculate monthly savings from fine-tuning.

        Args:
            base_cost_per_request: Base model cost
            ft_cost_per_request: Fine-tuned model cost

        Returns:
            Monthly savings
        """
        requests_per_month = self.config.daily_requests * 30
        savings_per_request = base_cost_per_request - ft_cost_per_request

        return requests_per_month * savings_per_request

    def _monetize_quality_improvements(self, accuracy_improvement: float) -> float:
        """Convert quality improvements to monetary value.

        Args:
            accuracy_improvement: Percentage accuracy improvement

        Returns:
            Annual monetary value
        """
        # This is highly use-case specific
        # Example: Each 1% improvement = $1000 annual value
        annual_value = accuracy_improvement * 1000

        return annual_value

    def _calculate_npv(
        self, initial_investment: float, cash_flows: List[float], discount_rate: float
    ) -> float:
        """Calculate Net Present Value.

        Args:
            initial_investment: Initial investment (negative)
            cash_flows: List of future cash flows
            discount_rate: Discount rate per period

        Returns:
            NPV
        """
        npv = -initial_investment

        for t, cash_flow in enumerate(cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** (t + 1))

        return npv

    def _calculate_irr(self, initial_investment: float, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return (simplified).

        Args:
            initial_investment: Initial investment
            cash_flows: List of future cash flows

        Returns:
            IRR as percentage
        """
        # Simplified IRR calculation
        # In practice, would use scipy.optimize or numpy.irr

        total_return = sum(cash_flows)
        years = len(cash_flows) / 12  # Convert months to years

        if initial_investment > 0 and years > 0:
            # Simple approximation
            irr = ((total_return / initial_investment) ** (1 / years) - 1) * 100
            return irr

        return 0.0

    def _generate_recommendation(self, roi_analysis: ROIAnalysis) -> str:
        """Generate recommendation based on ROI analysis.

        Args:
            roi_analysis: ROI analysis results

        Returns:
            Recommendation text
        """
        if roi_analysis.break_even_months < 6:
            return "Highly recommended - Quick payback period with strong ROI"
        elif roi_analysis.break_even_months < 12:
            return "Recommended - Reasonable payback period with good ROI"
        elif roi_analysis.break_even_months < 24:
            return "Consider carefully - Longer payback period but positive ROI"
        elif roi_analysis.break_even_months == float("inf"):
            return "Not recommended - No clear payback period"
        else:
            return "Not recommended - Very long payback period"

    def create_visualizations(self, analysis: Dict[str, Any]) -> Dict[str | go.Figure]:
        """Create visualizations for cost/benefit analysis.

        Args:
            analysis: Complete analysis results

        Returns:
            Dictionary of plotly figures
        """
        figures = {}

        # Cost breakdown pie chart
        cost_breakdown = analysis["cost_breakdown"]
        figures["cost_breakdown"] = go.Figure(
            data=[
                go.Pie(
                    labels=["Compute", "Storage", "Data Prep", "Evaluation"],
                    values=[
                        cost_breakdown.compute_cost,
                        cost_breakdown.storage_cost,
                        cost_breakdown.data_preparation_cost,
                        cost_breakdown.evaluation_cost,
                    ],
                    hole=0.3,
                )
            ]
        )
        figures["cost_breakdown"].update_layout(title="Training Cost Breakdown", height=400)

        # ROI timeline
        projections = analysis["projections"]
        figures["roi_timeline"] = go.Figure()

        figures["roi_timeline"].add_trace(
            go.Scatter(
                x=projections.time_periods,
                y=projections.baseline_costs,
                mode="lines",
                name="Baseline Costs",
                line=dict(dash="dash"),
            )
        )

        figures["roi_timeline"].add_trace(
            go.Scatter(
                x=projections.time_periods,
                y=projections.fine_tuned_costs,
                mode="lines",
                name="Fine-Tuned Costs",
            )
        )

        figures["roi_timeline"].add_trace(
            go.Scatter(
                x=projections.time_periods,
                y=projections.cumulative_savings,
                mode="lines",
                name="Cumulative Savings",
                line=dict(color="green"),
            )
        )

        # Add break-even line
        if projections.break_even_point:
            figures["roi_timeline"].add_vline(
                x=projections.time_periods[projections.break_even_point - 1],
                line_dash="dash",
                line_color="red",
                annotation_text="Break-even",
            )

        figures["roi_timeline"].update_layout(
            title="Cost Projection and ROI Timeline",
            xaxis_title="Time",
            yaxis_title="Cost ($)",
            height=500,
        )

        # Decision matrix radar chart
        decision_matrix = analysis["decision_matrix"]
        criteria_names = list(decision_matrix["criteria"].keys())
        criteria_scores = [
            decision_matrix["criteria"][c]["normalized_score"] * 100 for c in criteria_names
        ]

        figures["decision_matrix"] = go.Figure(
            data=go.Scatterpolar(
                r=criteria_scores,
                theta=[c.replace("_", " ").title() for c in criteria_names],
                fill="toself",
                name="Score",
            )
        )

        figures["decision_matrix"].update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-100, 100])),
            showlegend=False,
            title="Decision Matrix Scores",
            height=500,
        )

        return figures


# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = CostBenefitAnalyzer()

    # Create sample comparison result
    from ...fine_tuning.evaluation.suite import (
        BenchmarkResult as EvalBenchmarkResult,
        EvaluationResult,
    )
    from ..benchmark_runner import BenchmarkResult, ComparisonResult, ModelVersion, ModelVersionType

    # Sample data
    base_eval = EvaluationResult(
        model_name="gpt2",
        benchmarks=[
            EvalBenchmarkResult(
                name="hellaswag",
                overall_score=0.45,
                task_scores={"accuracy": 0.45},
                runtime_seconds=120,
                samples_evaluated=1000,
            )
        ],
    )

    ft_eval = EvaluationResult(
        model_name="gpt2-finetuned",
        benchmarks=[
            EvalBenchmarkResult(
                name="hellaswag",
                overall_score=0.55,
                task_scores={"accuracy": 0.55},
                runtime_seconds=115,
                samples_evaluated=1000,
            )
        ],
    )

    comparison = ComparisonResult(
        base_result=BenchmarkResult(
            model_version=ModelVersion(
                version_id="base_001",
                model_path="gpt2",
                version_type=ModelVersionType.BASE,
                created_at=datetime.now(),
            ),
            evaluation_results=base_eval,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=300,
        ),
        fine_tuned_result=BenchmarkResult(
            model_version=ModelVersion(
                version_id="ft_001",
                model_path="gpt2-finetuned",
                version_type=ModelVersionType.FINE_TUNED,
                created_at=datetime.now(),
            ),
            evaluation_results=ft_eval,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=290,
        ),
        improvements={
            "hellaswag": {
                "base_score": 0.45,
                "ft_score": 0.55,
                "improvement": 0.10,
                "improvement_pct": 22.22,
            }
        },
    )

    # Analyze
    analysis = analyzer.analyze_fine_tuning(
        comparison_result=comparison,
        training_hours=24,
        gpu_type="A100",
        data_size_gb=5.0,
        num_experiments=3,
    )

    # Print results
    print("\nCost/Benefit Analysis Results:")
    print(f"\nTotal Training Cost: ${analysis['cost_breakdown'].total_cost:.2f}")
    print(f"Break-even Period: {analysis['roi_analysis'].break_even_months:.1f} months")
    print(f"1-Year ROI: {analysis['roi_analysis'].one_year_roi:.1f}%")
    print(f"\nRecommendation: {analysis['recommendation']}")

    # Compare providers
    print("\n\nProvider Comparison:")
    comparisons = analyzer.compare_providers()
    for comp in comparisons:
        print(f"\n{comp.provider_name}:")
        print(f"  Annual Cost: ${comp.annual_cost_estimate:.2f}")
        print(f"  Fine-tuning Cost: ${comp.fine_tuning_cost:.2f}")
        print(f"  Pros: {', '.join(comp.pros)}")
        print(f"  Cons: {', '.join(comp.cons)}")

    # Create visualizations
    figures = analyzer.create_visualizations(analysis)
    print(f"\nGenerated {len(figures)} visualizations")
