"""Data models for security scanning results and analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from .config import SeverityLevel


class VulnerabilityType(str, Enum):
    """Types of vulnerabilities that can be detected."""

    JAILBREAK = "jailbreak"
    PROMPT_INJECTION = "prompt_injection"
    DATA_LEAKAGE = "data_leakage"
    HARMFUL_CONTENT = "harmful_content"
    BIAS_DISCRIMINATION = "bias_discrimination"
    MISINFORMATION = "misinformation"
    SOCIAL_ENGINEERING = "social_engineering"
    CONTEXT_MANIPULATION = "context_manipulation"
    INPUT_VALIDATION = "input_validation"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INFORMATION_DISCLOSURE = "information_disclosure"


@dataclass
class DetectionResult:
    """Result from a single detection strategy."""

    strategy_name: str
    confidence_score: float
    vulnerability_type: VulnerabilityType
    evidence: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass
class VulnerabilityFinding:
    """A specific vulnerability found in a model response."""

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    vulnerability_type: VulnerabilityType = VulnerabilityType.JAILBREAK
    severity: SeverityLevel = SeverityLevel.MEDIUM
    confidence_score: float = 0.5
    title: str = ""
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    affected_text: Optional[str] = None
    mitigation_suggestions: List[str] = field(default_factory=list)

    # Detection metadata
    detection_strategies: List[str] = field(default_factory=list)
    strategy_scores: Dict[str, float] = field(default_factory=dict)
    pattern_matches: List[str] = field(default_factory=list)

    # Context information
    attack_prompt: Optional[str] = None
    attack_id: Optional[str] = None
    response_excerpt: Optional[str] = None

    # Temporal data
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary for serialization."""
        return {
            "id": self.id,
            "vulnerability_type": self.vulnerability_type.value,
            "severity": self.severity.value,
            "confidence_score": self.confidence_score,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "affected_text": self.affected_text,
            "mitigation_suggestions": self.mitigation_suggestions,
            "detection_strategies": self.detection_strategies,
            "strategy_scores": self.strategy_scores,
            "pattern_matches": self.pattern_matches,
            "attack_prompt": self.attack_prompt,
            "attack_id": self.attack_id,
            "response_excerpt": self.response_excerpt,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class ModelResponse:
    """Container for model response and metadata."""

    content: str
    model_name: str
    response_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Request context
    prompt: Optional[str] = None
    attack_id: Optional[str] = None
    conversation_id: Optional[str] = None

    def get_excerpt(self, max_length: int = 200) -> str:
        """Get truncated excerpt of the response."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."


@dataclass
class ScanResult:
    """Complete result from scanning a model response."""

    scan_id: str = field(default_factory=lambda: str(uuid4())[:8])
    model_name: str = ""
    attack_prompt: str = ""
    response: ModelResponse = field(default_factory=lambda: ModelResponse("", ""))

    # Findings
    vulnerabilities: List[VulnerabilityFinding] = field(default_factory=list)
    overall_risk_score: float = 0.0
    max_severity: SeverityLevel = SeverityLevel.INFO

    # Scan metadata
    scan_timestamp: datetime = field(default_factory=datetime.now)
    scan_duration_ms: float = 0.0
    strategies_used: List[str] = field(default_factory=list)
    config_summary: Dict[str, Any] = field(default_factory=dict)

    # Analysis details
    pattern_matches: Dict[str, List[str]] = field(default_factory=dict)
    sentiment_analysis: Optional[Dict[str, Any]] = None
    contextual_analysis: Optional[Dict[str, Any]] = None

    def add_vulnerability(self, vulnerability: VulnerabilityFinding):
        """Add a vulnerability finding to the scan result."""
        self.vulnerabilities.append(vulnerability)

        # Update overall metrics
        if vulnerability.confidence_score > self.overall_risk_score:
            self.overall_risk_score = vulnerability.confidence_score

        # Update max severity (assuming enum values are ordered)
        severity_order = list(SeverityLevel)
        if severity_order.index(vulnerability.severity) < severity_order.index(self.max_severity):
            self.max_severity = vulnerability.severity

    def get_vulnerabilities_by_type(
        self, vuln_type: VulnerabilityType
    ) -> List[VulnerabilityFinding]:
        """Get all vulnerabilities of a specific type."""
        return [v for v in self.vulnerabilities if v.vulnerability_type == vuln_type]

    def get_vulnerabilities_by_severity(
        self, severity: SeverityLevel
    ) -> List[VulnerabilityFinding]:
        """Get all vulnerabilities of a specific severity."""
        return [v for v in self.vulnerabilities if v.severity == severity]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the scan results."""
        severity_counts = {}
        type_counts = {}

        for vuln in self.vulnerabilities:
            # Count by severity
            severity_counts[vuln.severity.value] = severity_counts.get(vuln.severity.value, 0) + 1
            # Count by type
            type_counts[vuln.vulnerability_type.value] = (
                type_counts.get(vuln.vulnerability_type.value, 0) + 1
            )

        return {
            "scan_id": self.scan_id,
            "model_name": self.model_name,
            "total_vulnerabilities": len(self.vulnerabilities),
            "overall_risk_score": self.overall_risk_score,
            "max_severity": self.max_severity.value,
            "severity_distribution": severity_counts,
            "vulnerability_types": type_counts,
            "scan_duration_ms": self.scan_duration_ms,
            "strategies_used": self.strategies_used,
            "scan_timestamp": self.scan_timestamp.isoformat(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert scan result to dictionary for serialization."""
        return {
            "scan_id": self.scan_id,
            "model_name": self.model_name,
            "attack_prompt": self.attack_prompt,
            "response": {
                "content": self.response.content,
                "model_name": self.response.model_name,
                "response_time_ms": self.response.response_time_ms,
                "timestamp": self.response.timestamp.isoformat(),
                "metadata": self.response.metadata,
            },
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "overall_risk_score": self.overall_risk_score,
            "max_severity": self.max_severity.value,
            "scan_timestamp": self.scan_timestamp.isoformat(),
            "scan_duration_ms": self.scan_duration_ms,
            "strategies_used": self.strategies_used,
            "config_summary": self.config_summary,
            "pattern_matches": self.pattern_matches,
            "sentiment_analysis": self.sentiment_analysis,
            "contextual_analysis": self.contextual_analysis,
            "summary": self.get_summary(),
        }


@dataclass
class BatchScanResult:
    """Result from scanning multiple prompts/responses."""

    batch_id: str = field(default_factory=lambda: str(uuid4())[:8])
    model_name: str = ""
    scan_results: List[ScanResult] = field(default_factory=list)

    # Batch metadata
    total_scans: int = 0
    successful_scans: int = 0
    failed_scans: int = 0
    batch_start_time: datetime = field(default_factory=datetime.now)
    batch_duration_ms: float = 0.0

    # Aggregate statistics
    total_vulnerabilities: int = 0
    average_risk_score: float = 0.0
    vulnerability_distribution: Dict[str, int] = field(default_factory=dict)
    severity_distribution: Dict[str, int] = field(default_factory=dict)

    def add_scan_result(self, scan_result: ScanResult):
        """Add a scan result to the batch."""
        self.scan_results.append(scan_result)
        self.successful_scans += 1
        self.total_scans += 1

        # Update aggregate statistics
        self._update_statistics()

    def add_failed_scan(self, error_info: Dict[str, Any]):
        """Record a failed scan."""
        self.failed_scans += 1
        self.total_scans += 1

    def _update_statistics(self):
        """Update aggregate statistics from scan results."""
        if not self.scan_results:
            return

        # Reset counters
        self.total_vulnerabilities = 0
        vuln_dist = {}
        sev_dist = {}
        risk_scores = []

        for scan_result in self.scan_results:
            self.total_vulnerabilities += len(scan_result.vulnerabilities)
            risk_scores.append(scan_result.overall_risk_score)

            for vuln in scan_result.vulnerabilities:
                # Count vulnerability types
                vuln_type = vuln.vulnerability_type.value
                vuln_dist[vuln_type] = vuln_dist.get(vuln_type, 0) + 1

                # Count severity levels
                severity = vuln.severity.value
                sev_dist[severity] = sev_dist.get(severity, 0) + 1

        self.vulnerability_distribution = vuln_dist
        self.severity_distribution = sev_dist
        self.average_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0

    def get_high_risk_scans(self, threshold: float = 0.7) -> List[ScanResult]:
        """Get scan results with risk score above threshold."""
        return [scan for scan in self.scan_results if scan.overall_risk_score >= threshold]

    def get_summary(self) -> Dict[str, Any]:
        """Get batch scan summary."""
        return {
            "batch_id": self.batch_id,
            "model_name": self.model_name,
            "total_scans": self.total_scans,
            "successful_scans": self.successful_scans,
            "failed_scans": self.failed_scans,
            "success_rate": self.successful_scans / max(1, self.total_scans),
            "batch_duration_ms": self.batch_duration_ms,
            "total_vulnerabilities": self.total_vulnerabilities,
            "average_risk_score": self.average_risk_score,
            "vulnerability_distribution": self.vulnerability_distribution,
            "severity_distribution": self.severity_distribution,
            "high_risk_scans": len(self.get_high_risk_scans()),
            "batch_start_time": self.batch_start_time.isoformat(),
        }
