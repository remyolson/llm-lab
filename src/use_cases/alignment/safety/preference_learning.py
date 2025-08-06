"""Preference learning system for personalized alignment."""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    REPORT = "report"  # Explicit safety report
    CORRECTION = "correction"  # User provided correction


class PreferenceCategory(Enum):
    """Categories of preferences."""

    TONE = "tone"  # Formal, casual, friendly, etc.
    DETAIL = "detail"  # Brief, detailed, comprehensive
    TECHNICALITY = "technicality"  # Simple, technical, expert
    SAFETY = "safety"  # Conservative, balanced, permissive
    CREATIVITY = "creativity"  # Factual, creative, imaginative
    SPEED = "speed"  # Fast, balanced, thorough


@dataclass
class Feedback:
    """User feedback on a response."""

    feedback_id: str
    user_id: str
    session_id: str
    prompt: str
    response: str
    feedback_type: FeedbackType
    category: Optional[PreferenceCategory] = None
    rating: Optional[float] = None  # 0-1 scale
    comment: Optional[str] = None
    correction: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreferences:
    """Learned preferences for a user."""

    user_id: str
    preference_scores: Dict[PreferenceCategory, Dict[str, float]] = field(default_factory=dict)
    feedback_history: List[str] = field(default_factory=list)  # Feedback IDs
    last_updated: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0

    # Preference weights
    tone_preferences: Dict[str, float] = field(default_factory=dict)
    safety_threshold: float = 0.5  # 0 = very permissive, 1 = very strict
    detail_level: float = 0.5  # 0 = brief, 1 = comprehensive
    technicality_level: float = 0.5  # 0 = simple, 1 = expert
    creativity_level: float = 0.5  # 0 = factual, 1 = creative

    def update_from_feedback(self, feedback: Feedback) -> None:
        """Update preferences based on feedback."""
        self.feedback_history.append(feedback.feedback_id)
        self.interaction_count += 1
        self.last_updated = datetime.now()

        # Update based on feedback type
        if feedback.feedback_type == FeedbackType.POSITIVE:
            self._reinforce_current_settings(feedback)
        elif feedback.feedback_type == FeedbackType.NEGATIVE:
            self._adjust_away_from_current(feedback)
        elif feedback.feedback_type == FeedbackType.REPORT:
            self._increase_safety_threshold()

    def _reinforce_current_settings(self, feedback: Feedback) -> None:
        """Reinforce current preference settings."""
        # Slightly increase current levels
        learning_rate = 0.1

        if feedback.category == PreferenceCategory.TONE:
            # Reinforce current tone
            pass  # Implement tone reinforcement
        elif feedback.category == PreferenceCategory.DETAIL:
            self.detail_level = min(1.0, self.detail_level + learning_rate)
        elif feedback.category == PreferenceCategory.SAFETY:
            # Don't reduce safety based on positive feedback
            pass

    def _adjust_away_from_current(self, feedback: Feedback) -> None:
        """Adjust preferences away from current settings."""
        learning_rate = 0.15

        if feedback.category == PreferenceCategory.DETAIL:
            # User wants different detail level
            if "too brief" in (feedback.comment or "").lower():
                self.detail_level = min(1.0, self.detail_level + learning_rate)
            elif "too long" in (feedback.comment or "").lower():
                self.detail_level = max(0.0, self.detail_level - learning_rate)

    def _increase_safety_threshold(self) -> None:
        """Increase safety threshold after report."""
        self.safety_threshold = min(1.0, self.safety_threshold + 0.2)


class PreferenceLearningSystem:
    """System for learning and applying user preferences."""

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize preference learning system.

        Args:
            storage_path: Path to store learned preferences
        """
        self.storage_path = storage_path or Path("preferences.db")
        self.preferences: Dict[str, UserPreferences] = {}
        self.feedback_store: Dict[str, Feedback] = {}
        self.global_preferences = self._initialize_global_preferences()

        # Load existing preferences
        if self.storage_path.exists():
            self.load_preferences()

    def _initialize_global_preferences(self) -> Dict[str, Any]:
        """Initialize global preference defaults."""
        return {
            "default_tone": "balanced",
            "default_safety": 0.5,
            "default_detail": 0.5,
            "min_interactions_for_learning": 5,
            "preference_decay_days": 30,
        }

    def record_feedback(self, feedback: Feedback) -> None:
        """Record user feedback."""
        # Store feedback
        self.feedback_store[feedback.feedback_id] = feedback

        # Get or create user preferences
        if feedback.user_id not in self.preferences:
            self.preferences[feedback.user_id] = UserPreferences(user_id=feedback.user_id)

        # Update preferences
        user_prefs = self.preferences[feedback.user_id]
        user_prefs.update_from_feedback(feedback)

        # Log for analysis
        logger.info(
            f"Recorded {feedback.feedback_type.value} feedback from user {feedback.user_id}"
        )

    def get_user_preferences(self, user_id: str) -> UserPreferences:
        """Get preferences for a user."""
        if user_id not in self.preferences:
            # Return default preferences
            return UserPreferences(user_id=user_id)

        return self.preferences[user_id]

    def apply_preferences(
        self, response: str, user_id: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Apply user preferences to a response."""
        prefs = self.get_user_preferences(user_id)

        # Apply detail level preference
        if prefs.detail_level < 0.3:
            # User prefers brief responses
            response = self._make_brief(response)
        elif prefs.detail_level > 0.7:
            # User prefers detailed responses
            response = self._make_detailed(response)

        # Apply other preferences...

        return response

    def _make_brief(self, response: str) -> str:
        """Make response more brief."""
        # Simple implementation - in practice would be more sophisticated
        sentences = response.split(". ")
        if len(sentences) > 3:
            # Keep only key sentences
            return ". ".join(sentences[:3]) + "."
        return response

    def _make_detailed(self, response: str) -> str:
        """Make response more detailed."""
        # Add elaboration prompt
        if len(response) < 100:
            response += "\n\nWould you like me to elaborate on any specific aspect?"
        return response

    def get_preference_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user preferences."""
        if user_id not in self.preferences:
            return {"status": "no_data"}

        prefs = self.preferences[user_id]

        # Analyze feedback history
        feedback_analysis = self._analyze_feedback_history(user_id)

        return {
            "user_id": user_id,
            "interaction_count": prefs.interaction_count,
            "last_updated": prefs.last_updated.isoformat(),
            "preferences": {
                "safety_level": prefs.safety_threshold,
                "detail_level": prefs.detail_level,
                "technicality_level": prefs.technicality_level,
                "creativity_level": prefs.creativity_level,
            },
            "feedback_summary": feedback_analysis,
            "recommendations": self._get_recommendations(prefs),
        }

    def _analyze_feedback_history(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's feedback history."""
        prefs = self.preferences.get(user_id)
        if not prefs:
            return {}

        feedback_counts = defaultdict(int)
        category_counts = defaultdict(int)

        for feedback_id in prefs.feedback_history[-50:]:  # Last 50 feedbacks
            if feedback_id in self.feedback_store:
                feedback = self.feedback_store[feedback_id]
                feedback_counts[feedback.feedback_type.value] += 1
                if feedback.category:
                    category_counts[feedback.category.value] += 1

        return {
            "feedback_types": dict(feedback_counts),
            "categories": dict(category_counts),
            "total_feedback": len(prefs.feedback_history),
        }

    def _get_recommendations(self, prefs: UserPreferences) -> List[str]:
        """Get recommendations based on user preferences."""
        recommendations = []

        if prefs.interaction_count < 5:
            recommendations.append("More interactions needed to fully personalize responses")

        if prefs.safety_threshold > 0.8:
            recommendations.append("User prefers very safe content - apply strict filtering")

        if prefs.detail_level > 0.8:
            recommendations.append(
                "User appreciates comprehensive responses - provide examples and context"
            )
        elif prefs.detail_level < 0.2:
            recommendations.append(
                "User prefers concise responses - keep it brief and to the point"
            )

        return recommendations

    def export_preferences(self, user_id: str) -> Dict[str, Any]:
        """Export user preferences for portability."""
        if user_id not in self.preferences:
            return {}

        prefs = self.preferences[user_id]
        return {
            "user_id": user_id,
            "preferences": asdict(prefs),
            "export_date": datetime.now().isoformat(),
            "version": "1.0",
        }

    def import_preferences(self, preference_data: Dict[str, Any]) -> bool:
        """Import user preferences."""
        try:
            user_id = preference_data["user_id"]
            pref_dict = preference_data["preferences"]

            # Create UserPreferences object
            prefs = UserPreferences(user_id=user_id)
            for key, value in pref_dict.items():
                if hasattr(prefs, key):
                    setattr(prefs, key, value)

            self.preferences[user_id] = prefs
            return True

        except Exception as e:
            logger.error(f"Failed to import preferences: {e}")
            return False

    def save_preferences(self) -> None:
        """Save preferences to storage."""
        data = {
            "preferences": {uid: asdict(prefs) for uid, prefs in self.preferences.items()},
            "feedback": {fid: asdict(feedback) for fid, feedback in self.feedback_store.items()},
            "global_preferences": self.global_preferences,
            "saved_at": datetime.now().isoformat(),
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_preferences(self) -> None:
        """Load preferences from storage."""
        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            # Load preferences
            for uid, pref_dict in data.get("preferences", {}).items():
                prefs = UserPreferences(user_id=uid)
                for key, value in pref_dict.items():
                    if hasattr(prefs, key):
                        # Handle datetime conversion
                        if key in ["last_updated"]:
                            value = datetime.fromisoformat(value)
                        setattr(prefs, key, value)
                self.preferences[uid] = prefs

            # Load feedback
            for fid, feedback_dict in data.get("feedback", {}).items():
                feedback = Feedback(
                    feedback_id=fid,
                    user_id=feedback_dict["user_id"],
                    session_id=feedback_dict["session_id"],
                    prompt=feedback_dict["prompt"],
                    response=feedback_dict["response"],
                    feedback_type=FeedbackType(feedback_dict["feedback_type"]),
                    timestamp=datetime.fromisoformat(feedback_dict["timestamp"]),
                )
                self.feedback_store[fid] = feedback

            # Load global preferences
            self.global_preferences.update(data.get("global_preferences", {}))

        except Exception as e:
            logger.error(f"Failed to load preferences: {e}")

    def cleanup_old_data(self, days: int = 90) -> None:
        """Clean up old feedback data."""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Remove old feedback
        old_feedback_ids = [
            fid for fid, feedback in self.feedback_store.items() if feedback.timestamp < cutoff_date
        ]

        for fid in old_feedback_ids:
            del self.feedback_store[fid]

        # Update user preference histories
        for prefs in self.preferences.values():
            prefs.feedback_history = [
                fid for fid in prefs.feedback_history if fid not in old_feedback_ids
            ]

        logger.info(f"Cleaned up {len(old_feedback_ids)} old feedback entries")


# Example preference profiles
class PreferenceProfiles:
    """Pre-defined preference profiles."""

    @staticmethod
    def child_safe() -> UserPreferences:
        """Very safe preferences for children."""
        prefs = UserPreferences(user_id="child_profile")
        prefs.safety_threshold = 0.95
        prefs.detail_level = 0.3
        prefs.technicality_level = 0.1
        prefs.creativity_level = 0.7
        return prefs

    @staticmethod
    def professional() -> UserPreferences:
        """Professional context preferences."""
        prefs = UserPreferences(user_id="professional_profile")
        prefs.safety_threshold = 0.7
        prefs.detail_level = 0.6
        prefs.technicality_level = 0.7
        prefs.creativity_level = 0.3
        return prefs

    @staticmethod
    def researcher() -> UserPreferences:
        """Research context preferences."""
        prefs = UserPreferences(user_id="researcher_profile")
        prefs.safety_threshold = 0.3
        prefs.detail_level = 0.9
        prefs.technicality_level = 0.9
        prefs.creativity_level = 0.5
        return prefs
