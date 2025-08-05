"""Safety filters for content moderation and risk detection."""

import re
import hashlib
import json
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for content."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentCategory(Enum):
    """Categories of potentially problematic content."""
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SEXUAL_CONTENT = "sexual_content"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"
    PERSONAL_INFO = "personal_info"
    MISINFORMATION = "misinformation"
    SPAM = "spam"
    PROFANITY = "profanity"


@dataclass
class SafetyScore:
    """Safety assessment score for content."""
    
    overall_risk: RiskLevel
    category_scores: Dict[ContentCategory, float] = field(default_factory=dict)
    confidence: float = 0.0
    flagged_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_safe(self, threshold: RiskLevel = RiskLevel.MEDIUM) -> bool:
        """Check if content is safe given a risk threshold."""
        risk_order = [RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM, 
                     RiskLevel.HIGH, RiskLevel.CRITICAL]
        return risk_order.index(self.overall_risk) <= risk_order.index(threshold)


class PatternBasedFilter:
    """Filter using regex patterns for different risk categories."""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[ContentCategory, List[re.Pattern]]:
        """Initialize regex patterns for each category."""
        return {
            ContentCategory.VIOLENCE: [
                re.compile(r'\b(kill|murder|assault|attack|hurt|harm|wound|injure)\s+\w+', re.I),
                re.compile(r'\b(weapon|gun|knife|bomb|explosive)\b', re.I),
                re.compile(r'\b(violent|violence|abuse|torture)\b', re.I),
            ],
            ContentCategory.HATE_SPEECH: [
                re.compile(r'\b(hate|discriminate|racist|sexist)\b', re.I),
                re.compile(r'\b(slur|derogatory)\s+\w+', re.I),
            ],
            ContentCategory.HARASSMENT: [
                re.compile(r'\b(harass|bully|threaten|stalk)\b', re.I),
                re.compile(r'\b(dox|doxx|doxxing)\b', re.I),
            ],
            ContentCategory.SELF_HARM: [
                re.compile(r'\b(suicide|self[\s-]harm|cut\s+myself|kill\s+myself)\b', re.I),
                re.compile(r'\b(end\s+my\s+life|hurt\s+myself)\b', re.I),
            ],
            ContentCategory.ILLEGAL_ACTIVITY: [
                re.compile(r'\b(illegal|crime|criminal|steal|theft|fraud)\b', re.I),
                re.compile(r'\b(hack|crack|pirate|bypass\s+security)\b', re.I),
                re.compile(r'\b(drug|narcotic|cocaine|heroin|meth)\b', re.I),
            ],
            ContentCategory.PERSONAL_INFO: [
                re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),  # SSN
                re.compile(r'\b\d{16}\b'),  # Credit card
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
                re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # Phone
            ],
            ContentCategory.PROFANITY: [
                # Add profanity patterns as needed
                re.compile(r'\b(damn|hell|crap)\b', re.I),  # Mild profanity
            ]
        }
        
    def analyze(self, text: str) -> Dict[ContentCategory, List[str]]:
        """Analyze text for problematic patterns."""
        results = {}
        
        for category, patterns in self.patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(text)
                matches.extend(found)
            
            if matches:
                results[category] = matches
                
        return results


class MLBasedFilter:
    """Placeholder for ML-based content filtering."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        # In production, this would load an actual ML model
        self.threshold = 0.5
        
    def predict(self, text: str) -> Dict[ContentCategory, float]:
        """Predict risk scores for each category."""
        # Placeholder implementation
        # In production, this would use a trained model
        
        # Simple heuristic based on text length and complexity
        text_lower = text.lower()
        scores = {}
        
        # Simulate ML predictions
        for category in ContentCategory:
            score = 0.0
            
            # Simple keyword-based scoring
            if category == ContentCategory.VIOLENCE:
                violence_words = ["kill", "hurt", "attack", "weapon"]
                score = sum(1 for word in violence_words if word in text_lower) * 0.3
            elif category == ContentCategory.HATE_SPEECH:
                hate_words = ["hate", "discriminate", "racist"]
                score = sum(1 for word in hate_words if word in text_lower) * 0.4
            # Add more category-specific logic
            
            scores[category] = min(score, 1.0)
            
        return scores


class SafetyFilter:
    """Main safety filter combining multiple detection methods."""
    
    def __init__(self, 
                 use_patterns: bool = True,
                 use_ml: bool = False,
                 risk_thresholds: Optional[Dict[ContentCategory, float]] = None):
        """
        Initialize safety filter.
        
        Args:
            use_patterns: Whether to use pattern-based detection
            use_ml: Whether to use ML-based detection
            risk_thresholds: Custom thresholds for each category
        """
        self.use_patterns = use_patterns
        self.use_ml = use_ml
        
        # Initialize components
        if use_patterns:
            self.pattern_filter = PatternBasedFilter()
        if use_ml:
            self.ml_filter = MLBasedFilter()
            
        # Set risk thresholds
        self.risk_thresholds = risk_thresholds or {
            ContentCategory.VIOLENCE: 0.3,
            ContentCategory.HATE_SPEECH: 0.3,
            ContentCategory.HARASSMENT: 0.4,
            ContentCategory.SELF_HARM: 0.2,
            ContentCategory.ILLEGAL_ACTIVITY: 0.3,
            ContentCategory.PERSONAL_INFO: 0.1,
            ContentCategory.MISINFORMATION: 0.5,
            ContentCategory.SPAM: 0.6,
            ContentCategory.PROFANITY: 0.7,
        }
        
        # Cache for performance
        self._cache: Dict[str, SafetyScore] = {}
        
    def check(self, text: str, use_cache: bool = True) -> SafetyScore:
        """
        Check text for safety issues.
        
        Args:
            text: Text to check
            use_cache: Whether to use cached results
            
        Returns:
            SafetyScore with risk assessment
        """
        # Check cache
        if use_cache:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Initialize scores
        category_scores = {}
        flagged_patterns = []
        
        # Pattern-based detection
        if self.use_patterns:
            pattern_results = self.pattern_filter.analyze(text)
            for category, matches in pattern_results.items():
                # Convert matches to score (0-1)
                score = min(len(matches) / 5.0, 1.0)  # Cap at 5 matches
                category_scores[category] = score
                flagged_patterns.extend(matches[:3])  # Keep first 3 matches
                
        # ML-based detection
        if self.use_ml:
            ml_scores = self.ml_filter.predict(text)
            for category, score in ml_scores.items():
                if category in category_scores:
                    # Combine scores (weighted average)
                    category_scores[category] = (
                        category_scores[category] * 0.4 + score * 0.6
                    )
                else:
                    category_scores[category] = score
                    
        # Determine overall risk
        overall_risk = self._calculate_overall_risk(category_scores)
        
        # Calculate confidence
        confidence = self._calculate_confidence(category_scores, len(flagged_patterns))
        
        # Create safety score
        safety_score = SafetyScore(
            overall_risk=overall_risk,
            category_scores=category_scores,
            confidence=confidence,
            flagged_patterns=flagged_patterns,
            metadata={
                "text_length": len(text),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Cache result
        if use_cache:
            self._cache[cache_key] = safety_score
            
        return safety_score
        
    def _calculate_overall_risk(self, 
                               category_scores: Dict[ContentCategory, float]) -> RiskLevel:
        """Calculate overall risk level from category scores."""
        if not category_scores:
            return RiskLevel.SAFE
            
        # Find highest risk category
        max_score = 0.0
        critical_categories = [
            ContentCategory.VIOLENCE,
            ContentCategory.SELF_HARM,
            ContentCategory.ILLEGAL_ACTIVITY
        ]
        
        for category, score in category_scores.items():
            threshold = self.risk_thresholds.get(category, 0.5)
            
            # Check if score exceeds threshold
            if score >= threshold:
                # Critical categories get higher weight
                if category in critical_categories:
                    max_score = max(max_score, score * 1.5)
                else:
                    max_score = max(max_score, score)
                    
        # Map score to risk level
        if max_score >= 0.8:
            return RiskLevel.CRITICAL
        elif max_score >= 0.6:
            return RiskLevel.HIGH
        elif max_score >= 0.4:
            return RiskLevel.MEDIUM
        elif max_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.SAFE
            
    def _calculate_confidence(self, 
                            category_scores: Dict[ContentCategory, float],
                            pattern_count: int) -> float:
        """Calculate confidence in the safety assessment."""
        # Base confidence on evidence strength
        confidence = 0.5
        
        # Increase confidence based on pattern matches
        if pattern_count > 0:
            confidence += min(pattern_count * 0.1, 0.3)
            
        # Increase confidence based on score consistency
        if category_scores:
            scores = list(category_scores.values())
            avg_score = sum(scores) / len(scores)
            
            # High scores increase confidence
            if avg_score > 0.5:
                confidence += 0.2
            elif avg_score > 0.3:
                confidence += 0.1
                
        return min(confidence, 1.0)
        
    def filter_response(self, 
                       text: str,
                       safety_score: Optional[SafetyScore] = None) -> str:
        """
        Filter response based on safety score.
        
        Args:
            text: Text to filter
            safety_score: Pre-computed safety score (optional)
            
        Returns:
            Filtered text or safety message
        """
        if safety_score is None:
            safety_score = self.check(text)
            
        if safety_score.overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            # Block high-risk content
            return self._get_safety_message(safety_score)
        elif safety_score.overall_risk == RiskLevel.MEDIUM:
            # Redact problematic parts
            return self._redact_content(text, safety_score)
        else:
            # Content is safe
            return text
            
    def _get_safety_message(self, safety_score: SafetyScore) -> str:
        """Get appropriate safety message for blocked content."""
        # Find the highest scoring category
        if not safety_score.category_scores:
            return "This content has been filtered for safety reasons."
            
        highest_category = max(
            safety_score.category_scores.items(),
            key=lambda x: x[1]
        )[0]
        
        messages = {
            ContentCategory.VIOLENCE: 
                "I cannot provide content that could promote violence or harm.",
            ContentCategory.HATE_SPEECH: 
                "I cannot generate content that contains hate speech or discrimination.",
            ContentCategory.SELF_HARM: 
                "I'm concerned about this request. If you're struggling, please reach out to a mental health professional or crisis helpline.",
            ContentCategory.ILLEGAL_ACTIVITY: 
                "I cannot provide guidance on illegal activities.",
            ContentCategory.PERSONAL_INFO: 
                "I cannot share or process personal information for privacy reasons.",
        }
        
        return messages.get(
            highest_category,
            "This content has been filtered for safety reasons."
        )
        
    def _redact_content(self, text: str, safety_score: SafetyScore) -> str:
        """Redact problematic parts of content."""
        redacted = text
        
        # Redact flagged patterns
        for pattern in safety_score.flagged_patterns:
            redacted = redacted.replace(pattern, "[REDACTED]")
            
        return redacted
        
    def clear_cache(self) -> None:
        """Clear the safety check cache."""
        self._cache.clear()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about safety checks."""
        return {
            "cache_size": len(self._cache),
            "risk_thresholds": self.risk_thresholds,
            "features_enabled": {
                "pattern_detection": self.use_patterns,
                "ml_detection": self.use_ml
            }
        }


# Specialized filters for specific use cases
class ChildSafetyFilter(SafetyFilter):
    """Enhanced safety filter for child-safe content."""
    
    def __init__(self):
        super().__init__(
            use_patterns=True,
            use_ml=False,
            risk_thresholds={
                ContentCategory.VIOLENCE: 0.1,
                ContentCategory.SEXUAL_CONTENT: 0.0,
                ContentCategory.PROFANITY: 0.2,
                ContentCategory.SELF_HARM: 0.0,
                # Very strict thresholds for child safety
            }
        )
        

class ProfessionalContentFilter(SafetyFilter):
    """Safety filter for professional/business contexts."""
    
    def __init__(self):
        super().__init__(
            use_patterns=True,
            use_ml=False,
            risk_thresholds={
                ContentCategory.PROFANITY: 0.3,
                ContentCategory.HARASSMENT: 0.2,
                ContentCategory.PERSONAL_INFO: 0.1,
                # Stricter on professionalism
            }
        )