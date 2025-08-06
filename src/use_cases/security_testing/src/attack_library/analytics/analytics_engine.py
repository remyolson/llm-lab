"""Analytics engine for comprehensive attack library analysis and reporting."""

import json
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.library import AttackLibrary
from ..core.models import Attack, AttackCategory, AttackSeverity
from .effectiveness_tracker import EffectivenessTracker, TestOutcome
from .tagging_system import TagCategory, TaggingSystem

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsReport:
    """Structured analytics report."""

    title: str
    generated_at: datetime
    data: Dict[str, Any]
    summary: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "data": self.data,
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


class AnalyticsEngine:
    """Comprehensive analytics engine for attack library analysis."""

    def __init__(
        self,
        attack_library: AttackLibrary,
        effectiveness_tracker: Optional[EffectivenessTracker] = None,
        tagging_system: Optional[TaggingSystem] = None,
    ):
        """
        Initialize analytics engine.

        Args:
            attack_library: Attack library to analyze
            effectiveness_tracker: Optional effectiveness tracker
            tagging_system: Optional tagging system
        """
        self.attack_library = attack_library
        self.effectiveness_tracker = effectiveness_tracker or EffectivenessTracker()
        self.tagging_system = tagging_system or TaggingSystem()

    def generate_library_overview(self) -> AnalyticsReport:
        """Generate comprehensive library overview report."""
        attacks = list(self.attack_library.attacks.values())

        if not attacks:
            return AnalyticsReport(
                title="Attack Library Overview",
                generated_at=datetime.now(),
                data={"error": "No attacks found"},
                summary={"total_attacks": 0},
                recommendations=["Add attacks to the library"],
            )

        # Basic statistics
        total_attacks = len(attacks)
        categories = Counter(attack.category for attack in attacks)
        severities = Counter(attack.severity for attack in attacks)
        sophistication_dist = Counter(attack.sophistication for attack in attacks)

        # Content analysis
        content_lengths = [len(attack.content) for attack in attacks]
        avg_length = statistics.mean(content_lengths)
        median_length = statistics.median(content_lengths)

        # Source analysis
        sources = Counter(attack.metadata.source for attack in attacks)

        # Model targeting analysis
        all_targets = []
        for attack in attacks:
            all_targets.extend(attack.target_models)
        target_models = Counter(all_targets)

        # Tag analysis
        all_tags = []
        for attack in attacks:
            all_tags.extend(attack.metadata.tags)
        popular_tags = Counter(all_tags)

        # Verification analysis
        verified_count = sum(1 for attack in attacks if attack.is_verified)
        verification_rate = verified_count / total_attacks if total_attacks > 0 else 0

        # Quality metrics
        quality_scores = []
        for attack in attacks:
            score = 0.5  # Base score

            # Length factor
            if 50 <= len(attack.content) <= 500:
                score += 0.1

            # Verification factor
            if attack.is_verified:
                score += 0.2

            # Tag factor
            if len(attack.metadata.tags) >= 3:
                score += 0.1

            # Effectiveness factor (if available)
            if self.effectiveness_tracker:
                effectiveness = self.effectiveness_tracker.get_attack_effectiveness(attack.id)
                if effectiveness["total_tests"] > 0:
                    score += effectiveness["average_score"] * 0.1

            quality_scores.append(min(1.0, score))

        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0

        # Trend analysis (by creation date)
        creation_dates = []
        for attack in attacks:
            creation_dates.append(attack.metadata.creation_date)

        # Monthly attack creation
        monthly_creation = defaultdict(int)
        for date in creation_dates:
            month_key = date.strftime("%Y-%m")
            monthly_creation[month_key] += 1

        # Data compilation
        data = {
            "basic_statistics": {
                "total_attacks": total_attacks,
                "category_distribution": {cat.value: count for cat, count in categories.items()},
                "severity_distribution": {sev.value: count for sev, count in severities.items()},
                "sophistication_distribution": dict(sophistication_dist),
                "verification_rate": verification_rate,
                "verified_attacks": verified_count,
            },
            "content_analysis": {
                "average_length": avg_length,
                "median_length": median_length,
                "min_length": min(content_lengths) if content_lengths else 0,
                "max_length": max(content_lengths) if content_lengths else 0,
                "length_distribution": self._get_length_distribution(content_lengths),
            },
            "source_analysis": {
                "source_distribution": dict(sources),
                "unique_sources": len(sources),
                "most_common_source": sources.most_common(1)[0] if sources else None,
            },
            "targeting_analysis": {
                "target_model_distribution": dict(target_models),
                "unique_models_targeted": len(target_models),
                "most_targeted_model": target_models.most_common(1)[0] if target_models else None,
                "universal_attacks": len([a for a in attacks if not a.target_models]),
            },
            "tagging_analysis": {
                "popular_tags": dict(popular_tags.most_common(20)),
                "unique_tags": len(popular_tags),
                "average_tags_per_attack": len(all_tags) / total_attacks
                if total_attacks > 0
                else 0,
                "untagged_attacks": len([a for a in attacks if not a.metadata.tags]),
            },
            "quality_metrics": {
                "average_quality_score": avg_quality,
                "quality_distribution": self._get_quality_distribution(quality_scores),
                "high_quality_attacks": len([q for q in quality_scores if q >= 0.8]),
                "low_quality_attacks": len([q for q in quality_scores if q < 0.5]),
            },
            "temporal_analysis": {
                "monthly_creation": dict(monthly_creation),
                "creation_trend": self._analyze_creation_trend(list(monthly_creation.values())),
            },
        }

        # Generate summary
        summary = {
            "total_attacks": total_attacks,
            "most_common_category": categories.most_common(1)[0] if categories else None,
            "most_common_severity": severities.most_common(1)[0] if severities else None,
            "average_quality": round(avg_quality, 3),
            "verification_rate": round(verification_rate, 3),
            "most_popular_tag": popular_tags.most_common(1)[0] if popular_tags else None,
        }

        # Generate recommendations
        recommendations = []

        if verification_rate < 0.5:
            recommendations.append(
                "Increase attack verification rate - less than 50% of attacks are verified"
            )

        if avg_quality < 0.6:
            recommendations.append(
                "Improve overall attack quality - average quality score is below 60%"
            )

        untagged_rate = len([a for a in attacks if not a.metadata.tags]) / total_attacks
        if untagged_rate > 0.2:
            recommendations.append("Add tags to attacks - over 20% of attacks are untagged")

        # Category balance
        min_category_count = min(categories.values()) if categories else 0
        max_category_count = max(categories.values()) if categories else 0
        if max_category_count > 2 * min_category_count:
            recommendations.append(
                "Rebalance attack categories - some categories are over-represented"
            )

        if len(target_models) < 5:
            recommendations.append(
                "Expand model targeting - attacks target fewer than 5 different models"
            )

        return AnalyticsReport(
            title="Attack Library Overview",
            generated_at=datetime.now(),
            data=data,
            summary=summary,
            recommendations=recommendations,
        )

    def generate_effectiveness_report(
        self, time_window_days: Optional[int] = 30
    ) -> AnalyticsReport:
        """Generate effectiveness analysis report."""
        if not self.effectiveness_tracker.results:
            return AnalyticsReport(
                title="Effectiveness Analysis",
                generated_at=datetime.now(),
                data={"error": "No effectiveness data available"},
                summary={},
                recommendations=["Conduct effectiveness testing to generate data"],
            )

        results = self.effectiveness_tracker.results

        # Apply time filter if specified
        if time_window_days:
            cutoff = datetime.now() - timedelta(days=time_window_days)
            results = [r for r in results if r.timestamp >= cutoff]

        if not results:
            return AnalyticsReport(
                title="Effectiveness Analysis",
                generated_at=datetime.now(),
                data={"error": f"No effectiveness data in last {time_window_days} days"},
                summary={},
                recommendations=["Conduct recent effectiveness testing"],
            )

        # Overall effectiveness metrics
        total_tests = len(results)
        success_count = len([r for r in results if r.outcome == TestOutcome.SUCCESS])
        success_rate = success_count / total_tests
        avg_score = statistics.mean(r.score for r in results)

        # Model analysis
        model_stats = {}
        for model in set(r.model_name for r in results):
            model_results = [r for r in results if r.model_name == model]
            model_success = len([r for r in model_results if r.outcome == TestOutcome.SUCCESS])
            model_stats[model] = {
                "total_tests": len(model_results),
                "success_rate": model_success / len(model_results),
                "average_score": statistics.mean(r.score for r in model_results),
                "outcomes": Counter(r.outcome.value for r in model_results),
            }

        # Attack effectiveness ranking
        attack_stats = defaultdict(list)
        for result in results:
            attack_stats[result.attack_id].append(result.score)

        attack_rankings = []
        for attack_id, scores in attack_stats.items():
            attack_rankings.append(
                {
                    "attack_id": attack_id,
                    "average_score": statistics.mean(scores),
                    "test_count": len(scores),
                    "max_score": max(scores),
                    "success_consistency": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                }
            )
        attack_rankings.sort(key=lambda x: x["average_score"], reverse=True)

        # Temporal trends
        daily_stats = defaultdict(lambda: {"tests": 0, "successes": 0, "total_score": 0.0})
        for result in results:
            day_key = result.timestamp.date().isoformat()
            daily_stats[day_key]["tests"] += 1
            if result.outcome == TestOutcome.SUCCESS:
                daily_stats[day_key]["successes"] += 1
            daily_stats[day_key]["total_score"] += result.score

        # Convert to time series
        time_series = {}
        for day, stats in daily_stats.items():
            time_series[day] = {
                "success_rate": stats["successes"] / stats["tests"],
                "average_score": stats["total_score"] / stats["tests"],
                "test_volume": stats["tests"],
            }

        # Outcome distribution
        outcome_dist = Counter(r.outcome.value for r in results)

        data = {
            "overall_metrics": {
                "total_tests": total_tests,
                "overall_success_rate": success_rate,
                "average_effectiveness_score": avg_score,
                "outcome_distribution": dict(outcome_dist),
                "unique_attacks_tested": len(attack_stats),
                "unique_models_tested": len(model_stats),
            },
            "model_analysis": model_stats,
            "attack_rankings": attack_rankings[:20],  # Top 20
            "temporal_analysis": {
                "daily_statistics": time_series,
                "trend_analysis": self._analyze_effectiveness_trend(results),
            },
            "data_quality": {
                "tests_with_notes": len([r for r in results if r.notes]),
                "tests_with_tester_id": len([r for r in results if r.tester_id]),
                "average_response_preview_length": statistics.mean(
                    len(r.response_preview) for r in results
                ),
            },
        }

        # Model vulnerability ranking
        model_vulnerability_ranking = [
            {
                "model": model,
                "vulnerability_score": stats["average_score"],
                "test_count": stats["total_tests"],
            }
            for model, stats in model_stats.items()
        ]
        model_vulnerability_ranking.sort(key=lambda x: x["vulnerability_score"], reverse=True)

        summary = {
            "total_tests": total_tests,
            "overall_success_rate": round(success_rate, 3),
            "most_vulnerable_model": model_vulnerability_ranking[0]
            if model_vulnerability_ranking
            else None,
            "most_effective_attack": attack_rankings[0] if attack_rankings else None,
            "data_coverage_days": (
                max(r.timestamp for r in results) - min(r.timestamp for r in results)
            ).days,
        }

        # Recommendations
        recommendations = []

        if success_rate < 0.3:
            recommendations.append(
                "Low overall success rate - review attack quality or test conditions"
            )

        # Check for model coverage
        library_attacks = len(self.attack_library.attacks)
        tested_attacks = len(attack_stats)
        if tested_attacks / library_attacks < 0.5:
            recommendations.append(
                f"Test coverage is low - only {tested_attacks}/{library_attacks} attacks have been tested"
            )

        # Check for model balance
        if len(model_stats) < 3:
            recommendations.append("Test against more model types for comprehensive analysis")

        # Check for consistency
        inconsistent_attacks = [
            a for a in attack_rankings if a["test_count"] > 3 and a["success_consistency"] > 0.3
        ]
        if len(inconsistent_attacks) > len(attack_rankings) * 0.2:
            recommendations.append(
                "Many attacks show inconsistent results - review test conditions"
            )

        return AnalyticsReport(
            title="Effectiveness Analysis",
            generated_at=datetime.now(),
            data=data,
            summary=summary,
            recommendations=recommendations,
        )

    def generate_model_vulnerability_analysis(self, model_names: List[str]) -> AnalyticsReport:
        """Generate detailed vulnerability analysis for specific models."""
        if not model_names:
            model_names = list(set(r.model_name for r in self.effectiveness_tracker.results))[
                :5
            ]  # Analyze top 5 models by test volume

        model_profiles = {}
        for model in model_names:
            profile = self.effectiveness_tracker.get_model_vulnerability_profile(model)
            model_profiles[model] = profile

        # Cross-model comparison
        comparison = self.effectiveness_tracker.get_comparative_analysis(model_names)

        # Vulnerability patterns
        vulnerability_patterns = self._analyze_vulnerability_patterns(model_profiles)

        data = {
            "model_profiles": model_profiles,
            "comparative_analysis": comparison,
            "vulnerability_patterns": vulnerability_patterns,
            "analysis_scope": {
                "models_analyzed": len(model_names),
                "total_unique_attacks_tested": len(
                    set(
                        r.attack_id
                        for r in self.effectiveness_tracker.results
                        if r.model_name in model_names
                    )
                ),
            },
        }

        # Find most vulnerable model
        most_vulnerable = None
        highest_score = 0
        for model, profile in model_profiles.items():
            if profile["overall_vulnerability_score"] > highest_score:
                highest_score = profile["overall_vulnerability_score"]
                most_vulnerable = model

        summary = {
            "models_analyzed": model_names,
            "most_vulnerable_model": most_vulnerable,
            "highest_vulnerability_score": highest_score,
            "common_vulnerabilities": vulnerability_patterns.get("common_categories", []),
        }

        recommendations = []
        for model, profile in model_profiles.items():
            if profile["overall_vulnerability_score"] > 0.7:
                recommendations.append(
                    f"{model} shows high vulnerability (score: {profile['overall_vulnerability_score']:.3f}) - implement additional safeguards"
                )

        if vulnerability_patterns.get("universal_vulnerabilities"):
            recommendations.append("Address universal vulnerabilities that affect multiple models")

        return AnalyticsReport(
            title="Model Vulnerability Analysis",
            generated_at=datetime.now(),
            data=data,
            summary=summary,
            recommendations=recommendations,
        )

    def generate_attack_evolution_report(self) -> AnalyticsReport:
        """Generate report on attack evolution and trends over time."""
        attacks = list(self.attack_library.attacks.values())

        if not attacks:
            return AnalyticsReport(
                title="Attack Evolution Analysis",
                generated_at=datetime.now(),
                data={"error": "No attacks available"},
                summary={},
                recommendations=["Add attacks to enable trend analysis"],
            )

        # Sort by creation date
        attacks.sort(key=lambda a: a.metadata.creation_date)

        # Monthly evolution
        monthly_stats = defaultdict(
            lambda: {
                "count": 0,
                "categories": Counter(),
                "severities": Counter(),
                "sophistication_sum": 0,
                "sources": Counter(),
                "avg_length": 0,
                "total_length": 0,
            }
        )

        for attack in attacks:
            month_key = attack.metadata.creation_date.strftime("%Y-%m")
            stats = monthly_stats[month_key]
            stats["count"] += 1
            stats["categories"][attack.category] += 1
            stats["severities"][attack.severity] += 1
            stats["sophistication_sum"] += attack.sophistication
            stats["sources"][attack.metadata.source] += 1
            stats["total_length"] += len(attack.content)

        # Calculate averages
        for month, stats in monthly_stats.items():
            if stats["count"] > 0:
                stats["avg_sophistication"] = stats["sophistication_sum"] / stats["count"]
                stats["avg_length"] = stats["total_length"] / stats["count"]

        # Identify trends
        months = sorted(monthly_stats.keys())
        trends = {
            "volume_trend": self._calculate_trend([monthly_stats[m]["count"] for m in months]),
            "sophistication_trend": self._calculate_trend(
                [monthly_stats[m]["avg_sophistication"] for m in months]
            ),
            "length_trend": self._calculate_trend([monthly_stats[m]["avg_length"] for m in months]),
        }

        # Source evolution
        source_evolution = {}
        for month in months:
            source_evolution[month] = dict(monthly_stats[month]["sources"])

        # Category shifts
        category_evolution = {}
        for month in months:
            total_month = monthly_stats[month]["count"]
            category_evolution[month] = {
                cat.value: count / total_month if total_month > 0 else 0
                for cat, count in monthly_stats[month]["categories"].items()
            }

        # Innovation analysis (new techniques/tags)
        innovation_timeline = self._analyze_innovation_timeline(attacks)

        data = {
            "monthly_statistics": {
                month: {
                    "count": stats["count"],
                    "average_sophistication": stats.get("avg_sophistication", 0),
                    "average_length": stats.get("avg_length", 0),
                    "top_category": stats["categories"].most_common(1)[0]
                    if stats["categories"]
                    else None,
                    "top_source": stats["sources"].most_common(1)[0] if stats["sources"] else None,
                }
                for month, stats in monthly_stats.items()
            },
            "trend_analysis": trends,
            "source_evolution": source_evolution,
            "category_evolution": category_evolution,
            "innovation_timeline": innovation_timeline,
            "time_span": {
                "first_attack": attacks[0].metadata.creation_date.isoformat(),
                "latest_attack": attacks[-1].metadata.creation_date.isoformat(),
                "total_months": len(months),
            },
        }

        # Detect significant changes
        recent_months = months[-3:] if len(months) >= 3 else months
        early_months = months[:3] if len(months) >= 3 else months

        recent_avg_sophistication = statistics.mean(
            [monthly_stats[m]["avg_sophistication"] for m in recent_months]
        )
        early_avg_sophistication = statistics.mean(
            [monthly_stats[m]["avg_sophistication"] for m in early_months]
        )

        sophistication_change = recent_avg_sophistication - early_avg_sophistication

        summary = {
            "total_time_span_months": len(months),
            "total_attacks_created": len(attacks),
            "sophistication_evolution": "increasing" if sophistication_change > 0.2 else "stable",
            "most_active_month": max(months, key=lambda m: monthly_stats[m]["count"])
            if months
            else None,
            "innovation_events": len(innovation_timeline.get("new_techniques", [])),
        }

        recommendations = []

        if trends["volume_trend"] < -0.1:
            recommendations.append(
                "Attack creation volume is declining - consider setting creation targets"
            )

        if trends["sophistication_trend"] < 0:
            recommendations.append(
                "Attack sophistication is declining - focus on advanced attack development"
            )

        # Check for source diversity
        recent_sources = set()
        for month in recent_months:
            recent_sources.update(monthly_stats[month]["sources"].keys())

        if len(recent_sources) < 3:
            recommendations.append(
                "Limited source diversity in recent attacks - expand attack sources"
            )

        return AnalyticsReport(
            title="Attack Evolution Analysis",
            generated_at=datetime.now(),
            data=data,
            summary=summary,
            recommendations=recommendations,
        )

    def _get_length_distribution(self, lengths: List[int]) -> Dict[str, int]:
        """Categorize content lengths into distribution buckets."""
        distribution = {"very_short": 0, "short": 0, "medium": 0, "long": 0, "very_long": 0}

        for length in lengths:
            if length < 50:
                distribution["very_short"] += 1
            elif length < 150:
                distribution["short"] += 1
            elif length < 300:
                distribution["medium"] += 1
            elif length < 500:
                distribution["long"] += 1
            else:
                distribution["very_long"] += 1

        return distribution

    def _get_quality_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Categorize quality scores into distribution buckets."""
        distribution = {"low": 0, "medium": 0, "high": 0, "excellent": 0}

        for score in scores:
            if score < 0.5:
                distribution["low"] += 1
            elif score < 0.7:
                distribution["medium"] += 1
            elif score < 0.9:
                distribution["high"] += 1
            else:
                distribution["excellent"] += 1

        return distribution

    def _analyze_creation_trend(self, monthly_counts: List[int]) -> str:
        """Analyze the trend in attack creation."""
        if len(monthly_counts) < 2:
            return "insufficient_data"

        # Simple linear trend
        x = list(range(len(monthly_counts)))
        y = monthly_counts

        # Calculate slope
        n = len(x)
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (
            n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2
        )

        if slope > 1:
            return "increasing"
        elif slope < -1:
            return "decreasing"
        else:
            return "stable"

    def _analyze_effectiveness_trend(self, results: List) -> Dict[str, Any]:
        """Analyze trends in effectiveness testing."""
        if len(results) < 5:
            return {"insufficient_data": True}

        # Sort by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)

        # Weekly groupings
        weekly_success_rates = []
        weekly_avg_scores = []

        current_week_start = sorted_results[0].timestamp.date()
        current_week_results = []

        for result in sorted_results:
            result_date = result.timestamp.date()

            # If more than 7 days since week start, process current week
            if (result_date - current_week_start).days >= 7 and current_week_results:
                success_count = len(
                    [r for r in current_week_results if r.outcome == TestOutcome.SUCCESS]
                )
                success_rate = success_count / len(current_week_results)
                avg_score = statistics.mean(r.score for r in current_week_results)

                weekly_success_rates.append(success_rate)
                weekly_avg_scores.append(avg_score)

                current_week_start = result_date
                current_week_results = [result]
            else:
                current_week_results.append(result)

        # Process final week
        if current_week_results:
            success_count = len(
                [r for r in current_week_results if r.outcome == TestOutcome.SUCCESS]
            )
            success_rate = success_count / len(current_week_results)
            avg_score = statistics.mean(r.score for r in current_week_results)

            weekly_success_rates.append(success_rate)
            weekly_avg_scores.append(avg_score)

        # Calculate trends
        success_rate_trend = self._calculate_trend(weekly_success_rates)
        avg_score_trend = self._calculate_trend(weekly_avg_scores)

        return {
            "weekly_success_rates": weekly_success_rates,
            "weekly_average_scores": weekly_avg_scores,
            "success_rate_trend": success_rate_trend,
            "average_score_trend": avg_score_trend,
            "weeks_analyzed": len(weekly_success_rates),
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend coefficient for a series of values."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = list(range(n))

        # Linear regression slope
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0

    def _analyze_vulnerability_patterns(self, model_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze common vulnerability patterns across models."""
        if not model_profiles:
            return {}

        # Common vulnerable categories
        category_vulnerabilities = defaultdict(list)
        for model, profile in model_profiles.items():
            for category, vuln_data in profile.get("category_vulnerabilities", {}).items():
                category_vulnerabilities[category].append(
                    {
                        "model": model,
                        "score": vuln_data["average_score"],
                        "test_count": vuln_data["test_count"],
                    }
                )

        # Find universally vulnerable categories (high scores across multiple models)
        universal_vulnerabilities = []
        for category, model_scores in category_vulnerabilities.items():
            if len(model_scores) >= 2:  # At least 2 models tested
                avg_score = statistics.mean(data["score"] for data in model_scores)
                if avg_score > 0.6:  # High vulnerability threshold
                    universal_vulnerabilities.append(
                        {
                            "category": category,
                            "average_vulnerability": avg_score,
                            "models_affected": len(model_scores),
                        }
                    )

        # Model-specific vulnerabilities (high in one model, low in others)
        model_specific_vulnerabilities = []
        for category, model_scores in category_vulnerabilities.items():
            if len(model_scores) >= 2:
                scores = [data["score"] for data in model_scores]
                max_score = max(scores)
                min_score = min(scores)

                if max_score > 0.7 and (max_score - min_score) > 0.3:
                    vulnerable_model = next(
                        data["model"] for data in model_scores if data["score"] == max_score
                    )
                    model_specific_vulnerabilities.append(
                        {
                            "category": category,
                            "vulnerable_model": vulnerable_model,
                            "vulnerability_score": max_score,
                            "score_variance": max_score - min_score,
                        }
                    )

        return {
            "universal_vulnerabilities": universal_vulnerabilities,
            "model_specific_vulnerabilities": model_specific_vulnerabilities,
            "category_coverage": len(category_vulnerabilities),
            "most_tested_category": max(
                category_vulnerabilities.keys(),
                key=lambda cat: sum(data["test_count"] for data in category_vulnerabilities[cat]),
            )
            if category_vulnerabilities
            else None,
        }

    def _analyze_innovation_timeline(self, attacks: List[Attack]) -> Dict[str, Any]:
        """Analyze the introduction of new techniques and innovations over time."""
        # Sort by creation date
        sorted_attacks = sorted(attacks, key=lambda a: a.metadata.creation_date)

        # Track introduction of new tags/techniques
        seen_tags = set()
        new_techniques = []

        for attack in sorted_attacks:
            new_tags_in_attack = []
            for tag in attack.metadata.tags:
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    new_tags_in_attack.append(tag)

            if new_tags_in_attack:
                new_techniques.append(
                    {
                        "date": attack.metadata.creation_date.isoformat(),
                        "attack_id": attack.id,
                        "new_techniques": new_tags_in_attack,
                        "attack_title": attack.title,
                    }
                )

        # Track sophistication milestones
        sophistication_milestones = []
        max_sophistication = 0

        for attack in sorted_attacks:
            if attack.sophistication > max_sophistication:
                max_sophistication = attack.sophistication
                sophistication_milestones.append(
                    {
                        "date": attack.metadata.creation_date.isoformat(),
                        "attack_id": attack.id,
                        "sophistication_level": attack.sophistication,
                        "attack_title": attack.title,
                    }
                )

        return {
            "new_techniques": new_techniques,
            "sophistication_milestones": sophistication_milestones,
            "technique_introduction_rate": len(new_techniques) / len(sorted_attacks)
            if sorted_attacks
            else 0,
            "total_unique_techniques": len(seen_tags),
        }

    def export_analytics_dashboard(self, output_dir: Path):
        """Export comprehensive analytics dashboard to files."""
        output_dir.mkdir(exist_ok=True)

        # Generate all reports
        overview = self.generate_library_overview()
        effectiveness = self.generate_effectiveness_report()
        evolution = self.generate_attack_evolution_report()

        # Save individual reports
        reports = {
            "library_overview.json": overview,
            "effectiveness_analysis.json": effectiveness,
            "attack_evolution.json": evolution,
        }

        for filename, report in reports.items():
            with open(output_dir / filename, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        # Generate summary dashboard
        dashboard_summary = {
            "generated_at": datetime.now().isoformat(),
            "reports_included": list(reports.keys()),
            "key_metrics": {
                "total_attacks": overview.summary.get("total_attacks", 0),
                "overall_success_rate": effectiveness.summary.get("overall_success_rate", 0),
                "most_vulnerable_model": effectiveness.summary.get("most_vulnerable_model"),
                "attack_evolution_trend": evolution.summary.get(
                    "sophistication_evolution", "unknown"
                ),
            },
            "top_recommendations": (
                overview.recommendations[:3]
                + effectiveness.recommendations[:3]
                + evolution.recommendations[:3]
            )[:5],  # Top 5 overall
        }

        with open(output_dir / "dashboard_summary.json", "w") as f:
            json.dump(dashboard_summary, f, indent=2, default=str)

        logger.info(f"Exported analytics dashboard to {output_dir}")

        return {
            "output_directory": str(output_dir),
            "reports_generated": list(reports.keys()),
            "summary_file": "dashboard_summary.json",
        }
