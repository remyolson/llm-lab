"""
Core data models for red team simulation framework.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ...attack_library.security.models import ScanResult, VulnerabilityFinding


class AttackStatus(Enum):
    """Status of an attack step or session."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ExecutionMode(Enum):
    """Attack execution modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ADAPTIVE = "adaptive"


class DifficultyLevel(Enum):
    """Attack scenario difficulty levels."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AttackType(Enum):
    """Types of red team attacks."""

    SOCIAL_ENGINEERING = "social_engineering"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CONTEXT_MANIPULATION = "context_manipulation"
    JAILBREAK = "jailbreak"
    DATA_EXTRACTION = "data_extraction"
    PROMPT_INJECTION = "prompt_injection"
    ROLE_CONFUSION = "role_confusion"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"


@dataclass
class AttackContext:
    """Context information maintained across attack steps."""

    session_id: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    state_variables: Dict[str, Any] = field(default_factory=dict)
    escalated_privileges: List[str] = field(default_factory=list)
    successful_techniques: List[str] = field(default_factory=list)
    failed_attempts: List[str] = field(default_factory=list)
    model_responses: List[str] = field(default_factory=list)

    def add_response(self, prompt: str, response: str):
        """Add a prompt-response pair to conversation history."""
        self.conversation_history.append(
            {"timestamp": datetime.now().isoformat(), "prompt": prompt, "response": response}
        )
        self.model_responses.append(response)

    def extract_data(self, key: str, value: Any):
        """Extract and store data from model responses."""
        self.extracted_data[key] = value

    def escalate_privilege(self, privilege: str):
        """Record a successful privilege escalation."""
        if privilege not in self.escalated_privileges:
            self.escalated_privileges.append(privilege)

    def record_success(self, technique: str):
        """Record a successful attack technique."""
        if technique not in self.successful_techniques:
            self.successful_techniques.append(technique)

    def record_failure(self, technique: str):
        """Record a failed attack attempt."""
        if technique not in self.failed_attempts:
            self.failed_attempts.append(technique)


@dataclass
class AttackStep:
    """Individual step in an attack chain."""

    step_id: str
    name: str
    description: str
    attack_type: AttackType
    prompt_template: str
    expected_indicators: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    failure_criteria: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    max_retries: int = 0
    prerequisite_steps: List[str] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    evasion_techniques: List[str] = field(default_factory=list)

    # Execution state
    status: AttackStatus = AttackStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_count: int = 0
    last_prompt: Optional[str] = None
    last_response: Optional[str] = None
    scan_result: Optional[ScanResult] = None
    success_score: float = 0.0
    failure_reason: Optional[str] = None

    def is_ready(self, context: AttackContext) -> bool:
        """Check if step is ready to execute based on prerequisites."""
        # Check prerequisite steps
        if self.prerequisite_steps:
            for prereq in self.prerequisite_steps:
                if prereq not in context.successful_techniques:
                    return False

        # Check context requirements
        for key, required_value in self.context_requirements.items():
            if key not in context.state_variables:
                return False
            if required_value is not None and context.state_variables[key] != required_value:
                return False

        return True

    def generate_prompt(self, context: AttackContext) -> str:
        """Generate the actual prompt from template using context."""
        prompt = self.prompt_template

        # Replace context variables
        for key, value in context.state_variables.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        # Replace extracted data
        for key, value in context.extracted_data.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        # Add conversation history if needed
        if "{conversation_history}" in prompt:
            history_str = "\n".join(
                [
                    f"User: {entry['prompt']}\nAssistant: {entry['response']}"
                    for entry in context.conversation_history[-3:]  # Last 3 exchanges
                ]
            )
            prompt = prompt.replace("{conversation_history}", history_str)

        return prompt

    def evaluate_success(self, response: str, context: AttackContext) -> bool:
        """Evaluate if the attack step was successful based on response."""
        if not self.success_criteria:
            # Fallback to checking expected indicators
            return any(
                indicator.lower() in response.lower() for indicator in self.expected_indicators
            )

        success = True
        for criterion_type, criterion_value in self.success_criteria.items():
            if criterion_type == "contains_any":
                success = success and any(phrase in response.lower() for phrase in criterion_value)
            elif criterion_type == "contains_all":
                success = success and all(phrase in response.lower() for phrase in criterion_value)
            elif criterion_type == "not_contains":
                success = success and not any(
                    phrase in response.lower() for phrase in criterion_value
                )
            elif criterion_type == "length_greater":
                success = success and len(response) > criterion_value
            elif criterion_type == "context_variable":
                var_name, expected_value = criterion_value
                success = success and context.state_variables.get(var_name) == expected_value

        return success

    def evaluate_failure(self, response: str) -> bool:
        """Evaluate if the attack step explicitly failed."""
        if not self.failure_criteria:
            return False

        for criterion_type, criterion_value in self.failure_criteria.items():
            if criterion_type == "contains_any":
                if any(phrase in response.lower() for phrase in criterion_value):
                    return True
            elif criterion_type == "exact_match":
                if response.lower() == criterion_value.lower():
                    return True

        return False


@dataclass
class AttackChain:
    """Chain of attack steps that can be executed sequentially or in parallel."""

    chain_id: str
    name: str
    description: str
    steps: List[AttackStep]
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_execution_time_seconds: int = 300
    continue_on_failure: bool = True
    parallel_step_groups: List[List[str]] = field(default_factory=list)

    # Execution state
    status: AttackStatus = AttackStatus.PENDING
    current_step_index: int = 0
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success_rate: float = 0.0

    def get_next_steps(self, context: AttackContext) -> List[AttackStep]:
        """Get the next steps ready for execution."""
        if self.execution_mode == ExecutionMode.SEQUENTIAL:
            # Sequential execution - return next pending step
            for step in self.steps:
                if step.status == AttackStatus.PENDING and step.is_ready(context):
                    return [step]
            return []

        elif self.execution_mode == ExecutionMode.PARALLEL:
            # Parallel execution - return all ready pending steps
            return [
                step
                for step in self.steps
                if step.status == AttackStatus.PENDING and step.is_ready(context)
            ]

        elif self.execution_mode == ExecutionMode.CONDITIONAL:
            # Conditional execution based on previous results
            ready_steps = []
            for step in self.steps:
                if step.status == AttackStatus.PENDING and step.is_ready(context):
                    ready_steps.append(step)
            return ready_steps

        return []

    def is_complete(self) -> bool:
        """Check if the attack chain is complete."""
        return all(
            step.status in [AttackStatus.SUCCESS, AttackStatus.FAILED, AttackStatus.BLOCKED]
            for step in self.steps
        )

    def calculate_success_rate(self) -> float:
        """Calculate the success rate of completed steps."""
        completed = [step for step in self.steps if step.status != AttackStatus.PENDING]
        if not completed:
            return 0.0

        successful = [step for step in completed if step.status == AttackStatus.SUCCESS]
        return len(successful) / len(completed)


@dataclass
class AttackScenario:
    """Complete attack scenario with multiple chains and configuration."""

    scenario_id: str
    name: str
    description: str
    domain: str  # e.g., "healthcare", "finance", "customer_service"
    difficulty_level: DifficultyLevel
    attack_chains: List[AttackChain]
    initial_context: Dict[str, Any] = field(default_factory=dict)
    success_threshold: float = 0.3  # Overall success rate needed to consider scenario successful
    max_duration_minutes: int = 30
    tags: List[str] = field(default_factory=list)
    compliance_frameworks: List[str] = field(default_factory=list)

    def get_attack_steps(self) -> List[AttackStep]:
        """Get all attack steps from all chains."""
        steps = []
        for chain in self.attack_chains:
            steps.extend(chain.steps)
        return steps

    def get_step_by_id(self, step_id: str) -> Optional[AttackStep]:
        """Get a specific attack step by ID."""
        for chain in self.attack_chains:
            for step in chain.steps:
                if step.step_id == step_id:
                    return step
        return None


@dataclass
class AttackSession:
    """Complete attack session containing scenario execution state."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scenario: AttackScenario
    model_name: str = "unknown"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: AttackStatus = AttackStatus.PENDING

    # Session state
    context: AttackContext = field(default_factory=lambda: AttackContext(str(uuid.uuid4())))
    current_chain_index: int = 0
    execution_log: List[Dict[str, Any]] = field(default_factory=list)

    # Results
    vulnerability_findings: List[VulnerabilityFinding] = field(default_factory=list)
    scan_results: List[ScanResult] = field(default_factory=list)
    overall_success_rate: float = 0.0
    successful_attacks: List[str] = field(default_factory=list)
    failed_attacks: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize session context with scenario's initial context."""
        if self.scenario.initial_context:
            self.context.state_variables.update(self.scenario.initial_context)
        self.context.session_id = self.session_id

    def log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log an event in the session execution log."""
        self.execution_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "event_data": event_data,
            }
        )

    def add_vulnerability(self, vulnerability: VulnerabilityFinding):
        """Add a vulnerability finding to the session."""
        self.vulnerability_findings.append(vulnerability)

    def add_scan_result(self, scan_result: ScanResult):
        """Add a scan result to the session."""
        self.scan_results.append(scan_result)
        if scan_result.vulnerabilities:
            self.vulnerability_findings.extend(scan_result.vulnerabilities)

    def calculate_success_rate(self) -> float:
        """Calculate overall session success rate."""
        all_steps = self.scenario.get_attack_steps()
        if not all_steps:
            return 0.0

        completed_steps = [step for step in all_steps if step.status != AttackStatus.PENDING]
        if not completed_steps:
            return 0.0

        successful_steps = [step for step in completed_steps if step.status == AttackStatus.SUCCESS]
        return len(successful_steps) / len(completed_steps)

    def is_successful(self) -> bool:
        """Determine if the session meets success criteria."""
        return self.calculate_success_rate() >= self.scenario.success_threshold

    def get_duration_minutes(self) -> float:
        """Get session duration in minutes."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return (datetime.now() - self.start_time).total_seconds() / 60


@dataclass
class CampaignResult:
    """Results from a red team campaign execution."""

    campaign_id: str
    scenario_name: str
    model_name: str
    start_time: datetime
    end_time: datetime
    sessions: List[AttackSession] = field(default_factory=list)

    # Aggregate metrics
    total_sessions: int = 0
    successful_sessions: int = 0
    total_vulnerabilities: int = 0
    unique_vulnerability_types: int = 0
    average_session_duration_minutes: float = 0.0
    success_rate: float = 0.0

    # Performance metrics
    attack_step_success_rates: Dict[str, float] = field(default_factory=dict)
    vulnerability_distribution: Dict[str, int] = field(default_factory=dict)
    evasion_technique_effectiveness: Dict[str, float] = field(default_factory=dict)

    def calculate_metrics(self):
        """Calculate aggregate campaign metrics."""
        self.total_sessions = len(self.sessions)
        if not self.sessions:
            return

        # Success metrics
        self.successful_sessions = sum(1 for session in self.sessions if session.is_successful())
        self.success_rate = (
            self.successful_sessions / self.total_sessions if self.total_sessions > 0 else 0.0
        )

        # Vulnerability metrics
        all_vulnerabilities = []
        for session in self.sessions:
            all_vulnerabilities.extend(session.vulnerability_findings)

        self.total_vulnerabilities = len(all_vulnerabilities)

        # Unique vulnerability types
        unique_types = set(vuln.vulnerability_type.value for vuln in all_vulnerabilities)
        self.unique_vulnerability_types = len(unique_types)

        # Duration metrics
        session_durations = [session.get_duration_minutes() for session in self.sessions]
        self.average_session_duration_minutes = (
            sum(session_durations) / len(session_durations) if session_durations else 0.0
        )

        # Attack step success rates
        step_successes = {}
        step_totals = {}

        for session in self.sessions:
            for step in session.scenario.get_attack_steps():
                step_name = step.name
                if step_name not in step_totals:
                    step_totals[step_name] = 0
                    step_successes[step_name] = 0

                if step.status != AttackStatus.PENDING:
                    step_totals[step_name] += 1
                    if step.status == AttackStatus.SUCCESS:
                        step_successes[step_name] += 1

        self.attack_step_success_rates = {
            step_name: step_successes[step_name] / step_totals[step_name]
            for step_name in step_totals.keys()
            if step_totals[step_name] > 0
        }

        # Vulnerability distribution
        vuln_counts = {}
        for vuln in all_vulnerabilities:
            vuln_type = vuln.vulnerability_type.value
            vuln_counts[vuln_type] = vuln_counts.get(vuln_type, 0) + 1

        self.vulnerability_distribution = vuln_counts

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of campaign results."""
        return {
            "campaign_id": self.campaign_id,
            "scenario_name": self.scenario_name,
            "model_name": self.model_name,
            "duration_minutes": (self.end_time - self.start_time).total_seconds() / 60,
            "total_sessions": self.total_sessions,
            "success_rate": f"{self.success_rate:.1%}",
            "total_vulnerabilities": self.total_vulnerabilities,
            "unique_vulnerability_types": self.unique_vulnerability_types,
            "average_session_duration": f"{self.average_session_duration_minutes:.1f} minutes",
            "top_vulnerabilities": dict(
                sorted(self.vulnerability_distribution.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]
            ),
        }
