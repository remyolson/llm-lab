"""Attack effectiveness tracking and measurement system."""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TestOutcome(Enum):
    """Test outcome categories."""

    SUCCESS = "success"  # Attack succeeded
    FAILURE = "failure"  # Attack failed
    PARTIAL = "partial"  # Partial success
    ERROR = "error"  # Test error
    BLOCKED = "blocked"  # Blocked by safety measures


@dataclass
class EffectivenessResult:
    """Single effectiveness test result."""

    attack_id: str
    model_name: str
    model_version: Optional[str]
    outcome: TestOutcome
    score: float  # 0.0-1.0 effectiveness score
    response_preview: str  # First 200 chars of response
    timestamp: datetime
    test_context: Dict[str, Any]
    tester_id: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["outcome"] = self.outcome.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectivenessResult":
        """Create from dictionary."""
        data["outcome"] = TestOutcome(data["outcome"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class EffectivenessTracker:
    """System for tracking attack effectiveness across models and time."""

    def __init__(self, data_file: Optional[Path] = None):
        """
        Initialize effectiveness tracker.

        Args:
            data_file: File to store effectiveness data
        """
        self.data_file = data_file or Path("attack_effectiveness.json")
        self.results: List[EffectivenessResult] = []

        # Load existing data
        self.load_data()

    def load_data(self):
        """Load effectiveness data from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, "r") as f:
                    data = json.load(f)
                    self.results = [
                        EffectivenessResult.from_dict(result_data)
                        for result_data in data.get("results", [])
                    ]
                logger.info(f"Loaded {len(self.results)} effectiveness results")
            except Exception as e:
                logger.error(f"Failed to load effectiveness data: {e}")
                self.results = []

    def save_data(self):
        """Save effectiveness data to file."""
        try:
            data = {
                "metadata": {
                    "total_results": len(self.results),
                    "last_updated": datetime.now().isoformat(),
                    "data_version": "1.0",
                },
                "results": [result.to_dict() for result in self.results],
            }

            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.results)} effectiveness results")
        except Exception as e:
            logger.error(f"Failed to save effectiveness data: {e}")

    def record_result(
        self,
        attack_id: str,
        model_name: str,
        outcome: TestOutcome,
        score: float,
        response_preview: str,
        model_version: Optional[str] = None,
        test_context: Optional[Dict[str, Any]] = None,
        tester_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> EffectivenessResult:
        """
        Record a new effectiveness test result.

        Args:
            attack_id: ID of the attack tested
            model_name: Name of the model tested
            outcome: Test outcome
            score: Effectiveness score (0.0-1.0)
            response_preview: Preview of model response
            model_version: Optional model version
            test_context: Additional test context
            tester_id: ID of person/system conducting test
            notes: Optional notes

        Returns:
            Created effectiveness result
        """
        result = EffectivenessResult(
            attack_id=attack_id,
            model_name=model_name,
            model_version=model_version,
            outcome=outcome,
            score=max(0.0, min(1.0, score)),  # Clamp to [0,1]
            response_preview=response_preview[:200],  # Truncate preview
            timestamp=datetime.now(),
            test_context=test_context or {},
            tester_id=tester_id,
            notes=notes,
        )

        self.results.append(result)
        self.save_data()

        logger.info(
            f"Recorded effectiveness result for attack {attack_id} on {model_name}: {outcome.value}"
        )
        return result

    def get_attack_effectiveness(
        self,
        attack_id: str,
        model_name: Optional[str] = None,
        time_window_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get effectiveness statistics for an attack.

        Args:
            attack_id: Attack ID to analyze
            model_name: Optional model filter
            time_window_days: Optional time window in days

        Returns:
            Effectiveness statistics
        """
        # Filter results
        filtered_results = [r for r in self.results if r.attack_id == attack_id]

        if model_name:
            filtered_results = [
                r for r in filtered_results if r.model_name.lower() == model_name.lower()
            ]

        if time_window_days:
            cutoff = datetime.now() - timedelta(days=time_window_days)
            filtered_results = [r for r in filtered_results if r.timestamp >= cutoff]

        if not filtered_results:
            return {
                "attack_id": attack_id,
                "total_tests": 0,
                "success_rate": 0.0,
                "average_score": 0.0,
                "models_tested": [],
                "outcome_distribution": {},
                "latest_test": None,
                "trend_analysis": None,
            }

        # Calculate statistics
        total_tests = len(filtered_results)
        successful_tests = len([r for r in filtered_results if r.outcome == TestOutcome.SUCCESS])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        average_score = sum(r.score for r in filtered_results) / total_tests

        # Models tested
        models_tested = list(set(r.model_name for r in filtered_results))

        # Outcome distribution
        outcome_counts = defaultdict(int)
        for result in filtered_results:
            outcome_counts[result.outcome.value] += 1

        outcome_distribution = {
            outcome: count / total_tests for outcome, count in outcome_counts.items()
        }

        # Latest test
        latest_result = max(filtered_results, key=lambda r: r.timestamp)

        # Trend analysis (if enough data points)
        trend_analysis = None
        if len(filtered_results) >= 5:
            trend_analysis = self._analyze_effectiveness_trend(filtered_results)

        return {
            "attack_id": attack_id,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "average_score": average_score,
            "models_tested": models_tested,
            "outcome_distribution": outcome_distribution,
            "latest_test": {
                "timestamp": latest_result.timestamp.isoformat(),
                "model": latest_result.model_name,
                "outcome": latest_result.outcome.value,
                "score": latest_result.score,
            },
            "trend_analysis": trend_analysis,
        }

    def get_model_vulnerability_profile(
        self, model_name: str, model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get vulnerability profile for a specific model.

        Args:
            model_name: Model to analyze
            model_version: Optional specific version

        Returns:
            Model vulnerability profile
        """
        # Filter results for this model
        model_results = [r for r in self.results if r.model_name.lower() == model_name.lower()]

        if model_version:
            model_results = [r for r in model_results if r.model_version == model_version]

        if not model_results:
            return {
                "model_name": model_name,
                "model_version": model_version,
                "total_tests": 0,
                "overall_vulnerability_score": 0.0,
                "category_vulnerabilities": {},
                "most_effective_attacks": [],
                "temporal_analysis": {},
            }

        total_tests = len(model_results)
        overall_vulnerability_score = sum(r.score for r in model_results) / total_tests

        # Group by attack categories (would need attack category mapping)
        # For now, group by attack_id prefix patterns
        category_scores = defaultdict(list)
        for result in model_results:
            # Simple category detection from attack_id patterns
            if "jailbreak" in result.attack_id.lower():
                category = "jailbreak"
            elif "inject" in result.attack_id.lower():
                category = "injection"
            elif "extract" in result.attack_id.lower():
                category = "extraction"
            elif "manipul" in result.attack_id.lower():
                category = "manipulation"
            elif "evade" in result.attack_id.lower() or "evas" in result.attack_id.lower():
                category = "evasion"
            else:
                category = "other"

            category_scores[category].append(result.score)

        category_vulnerabilities = {
            category: {
                "average_score": sum(scores) / len(scores),
                "test_count": len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
            }
            for category, scores in category_scores.items()
        }

        # Most effective attacks
        attack_effectiveness = defaultdict(list)
        for result in model_results:
            attack_effectiveness[result.attack_id].append(result.score)

        most_effective_attacks = [
            {
                "attack_id": attack_id,
                "average_score": sum(scores) / len(scores),
                "test_count": len(scores),
                "max_score": max(scores),
            }
            for attack_id, scores in attack_effectiveness.items()
        ]
        most_effective_attacks.sort(key=lambda x: x["average_score"], reverse=True)
        most_effective_attacks = most_effective_attacks[:10]  # Top 10

        # Temporal analysis
        temporal_analysis = self._analyze_model_vulnerability_trends(model_results)

        return {
            "model_name": model_name,
            "model_version": model_version,
            "total_tests": total_tests,
            "overall_vulnerability_score": overall_vulnerability_score,
            "category_vulnerabilities": category_vulnerabilities,
            "most_effective_attacks": most_effective_attacks,
            "temporal_analysis": temporal_analysis,
        }

    def _analyze_effectiveness_trend(self, results: List[EffectivenessResult]) -> Dict[str, Any]:
        """Analyze effectiveness trend over time."""
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)

        # Group by time periods (weekly)
        weekly_scores = defaultdict(list)
        for result in sorted_results:
            week_key = result.timestamp.strftime("%Y-W%U")
            weekly_scores[week_key].append(result.score)

        # Calculate weekly averages
        weekly_averages = {
            week: sum(scores) / len(scores) for week, scores in weekly_scores.items()
        }

        # Trend direction
        if len(weekly_averages) >= 2:
            scores = list(weekly_averages.values())
            trend = "improving" if scores[-1] > scores[0] else "declining"
            trend_strength = abs(scores[-1] - scores[0])
        else:
            trend = "insufficient_data"
            trend_strength = 0.0

        return {
            "weekly_averages": weekly_averages,
            "trend_direction": trend,
            "trend_strength": trend_strength,
            "data_points": len(results),
        }

    def _analyze_model_vulnerability_trends(
        self, results: List[EffectivenessResult]
    ) -> Dict[str, Any]:
        """Analyze model vulnerability trends over time."""
        if len(results) < 2:
            return {"insufficient_data": True}

        # Sort by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)

        # Monthly grouping
        monthly_data = defaultdict(lambda: {"scores": [], "success_count": 0, "total_count": 0})

        for result in sorted_results:
            month_key = result.timestamp.strftime("%Y-%m")
            monthly_data[month_key]["scores"].append(result.score)
            monthly_data[month_key]["total_count"] += 1
            if result.outcome == TestOutcome.SUCCESS:
                monthly_data[month_key]["success_count"] += 1

        # Calculate monthly statistics
        monthly_stats = {}
        for month, data in monthly_data.items():
            monthly_stats[month] = {
                "average_vulnerability": sum(data["scores"]) / len(data["scores"]),
                "success_rate": data["success_count"] / data["total_count"],
                "test_count": data["total_count"],
            }

        return {
            "monthly_statistics": monthly_stats,
            "total_time_span_months": len(monthly_stats),
            "earliest_test": sorted_results[0].timestamp.isoformat(),
            "latest_test": sorted_results[-1].timestamp.isoformat(),
        }

    def get_comparative_analysis(
        self, models: List[str], time_window_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare vulnerability across multiple models.

        Args:
            models: List of model names to compare
            time_window_days: Optional time window

        Returns:
            Comparative analysis
        """
        comparison_data = {}

        for model in models:
            profile = self.get_model_vulnerability_profile(model)
            comparison_data[model] = {
                "total_tests": profile["total_tests"],
                "overall_vulnerability_score": profile["overall_vulnerability_score"],
                "category_vulnerabilities": profile["category_vulnerabilities"],
            }

        # Overall ranking
        model_rankings = [
            {
                "model": model,
                "vulnerability_score": data["overall_vulnerability_score"],
                "test_count": data["total_tests"],
            }
            for model, data in comparison_data.items()
            if data["total_tests"] > 0
        ]
        model_rankings.sort(key=lambda x: x["vulnerability_score"], reverse=True)

        return {
            "models_compared": models,
            "individual_profiles": comparison_data,
            "vulnerability_ranking": model_rankings,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def export_effectiveness_data(
        self, output_file: Path, format_type: str = "json", filters: Optional[Dict[str, Any]] = None
    ):
        """
        Export effectiveness data in specified format.

        Args:
            output_file: Output file path
            format_type: Export format ('json', 'csv')
            filters: Optional filters to apply
        """
        # Apply filters if provided
        filtered_results = self.results

        if filters:
            if "model_name" in filters:
                filtered_results = [
                    r
                    for r in filtered_results
                    if r.model_name.lower() == filters["model_name"].lower()
                ]

            if "outcome" in filters:
                filtered_results = [
                    r for r in filtered_results if r.outcome.value == filters["outcome"]
                ]

            if "date_range" in filters:
                start_date = datetime.fromisoformat(filters["date_range"]["start"])
                end_date = datetime.fromisoformat(filters["date_range"]["end"])
                filtered_results = [
                    r for r in filtered_results if start_date <= r.timestamp <= end_date
                ]

        if format_type.lower() == "json":
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_results": len(filtered_results),
                    "filters_applied": filters or {},
                },
                "results": [result.to_dict() for result in filtered_results],
            }

            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)

        elif format_type.lower() == "csv":
            import csv

            with open(output_file, "w", newline="") as f:
                if filtered_results:
                    fieldnames = [
                        "attack_id",
                        "model_name",
                        "model_version",
                        "outcome",
                        "score",
                        "timestamp",
                        "tester_id",
                        "notes",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for result in filtered_results:
                        writer.writerow(
                            {
                                "attack_id": result.attack_id,
                                "model_name": result.model_name,
                                "model_version": result.model_version or "",
                                "outcome": result.outcome.value,
                                "score": result.score,
                                "timestamp": result.timestamp.isoformat(),
                                "tester_id": result.tester_id or "",
                                "notes": result.notes or "",
                            }
                        )
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        logger.info(f"Exported {len(filtered_results)} effectiveness results to {output_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall effectiveness tracking statistics."""
        if not self.results:
            return {
                "total_results": 0,
                "unique_attacks_tested": 0,
                "unique_models_tested": 0,
                "date_range": None,
                "outcome_distribution": {},
                "average_effectiveness_score": 0.0,
            }

        # Basic stats
        total_results = len(self.results)
        unique_attacks = len(set(r.attack_id for r in self.results))
        unique_models = len(set(r.model_name for r in self.results))

        # Date range
        timestamps = [r.timestamp for r in self.results]
        earliest = min(timestamps)
        latest = max(timestamps)

        # Outcome distribution
        outcome_counts = defaultdict(int)
        for result in self.results:
            outcome_counts[result.outcome.value] += 1

        outcome_distribution = {
            outcome: count / total_results for outcome, count in outcome_counts.items()
        }

        # Average effectiveness
        average_score = sum(r.score for r in self.results) / total_results

        return {
            "total_results": total_results,
            "unique_attacks_tested": unique_attacks,
            "unique_models_tested": unique_models,
            "date_range": {
                "earliest": earliest.isoformat(),
                "latest": latest.isoformat(),
                "span_days": (latest - earliest).days,
            },
            "outcome_distribution": outcome_distribution,
            "average_effectiveness_score": average_score,
            "models_tested": list(set(r.model_name for r in self.results)),
        }
