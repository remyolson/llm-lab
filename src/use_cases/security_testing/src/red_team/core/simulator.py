"""
Core RedTeamSimulator class for orchestrating multi-step attack campaigns.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

from ...attack_library.core.library import AttackLibrary
from ...attack_library.security.config import ScanConfig
from ...attack_library.security.scanner import SecurityScanner
from .models import (
    AttackChain,
    AttackContext,
    AttackScenario,
    AttackSession,
    AttackStatus,
    AttackStep,
    CampaignResult,
    ExecutionMode,
)

logger = logging.getLogger(__name__)


class RedTeamSimulator:
    """
    Core red team simulation engine for orchestrating multi-step attack campaigns.

    The RedTeamSimulator provides a unified interface for executing sophisticated
    attack scenarios, managing session state, and coordinating with the underlying
    security scanning infrastructure.
    """

    def __init__(
        self,
        attack_library: AttackLibrary,
        security_scanner: Optional[SecurityScanner] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RedTeamSimulator.

        Args:
            attack_library: Library of attacks for vulnerability scanning
            security_scanner: Optional pre-configured security scanner
            config: Simulation configuration options
        """
        self.attack_library = attack_library
        self.config = config or self._get_default_config()

        # Initialize security scanner for vulnerability assessment
        if security_scanner:
            self.security_scanner = security_scanner
        else:
            scan_config = ScanConfig.create_quick_scan_config()
            self.security_scanner = SecurityScanner(attack_library, scan_config)

        # Session management
        self._active_sessions: Dict[str, AttackSession] = {}
        self._session_history: List[AttackSession] = []

        # Performance tracking
        self._campaign_history: List[CampaignResult] = []
        self._performance_metrics = {
            "total_campaigns": 0,
            "total_sessions": 0,
            "average_success_rate": 0.0,
            "total_vulnerabilities_found": 0,
        }

        # Plugin system for custom attack modules
        self._custom_modules: Dict[str, Any] = {}
        self._attack_plugins: Dict[str, Callable] = {}

        logger.info(f"RedTeamSimulator initialized with {len(attack_library.attacks)} base attacks")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the simulator."""
        return {
            "max_concurrent_sessions": 5,
            "default_session_timeout_minutes": 30,
            "enable_vulnerability_scanning": True,
            "enable_response_analysis": True,
            "enable_adaptive_execution": True,
            "session_state_persistence": True,
            "debugging_enabled": False,
            "rate_limiting": {"requests_per_second": 2, "burst_size": 5},
            "rollback_support": True,
            "checkpoint_frequency": 10,  # Every 10 steps
        }

    async def run_campaign(
        self,
        model_interface: Callable[[str], str],
        scenario: Union[AttackScenario, str],
        model_name: str = "unknown",
        num_sessions: int = 1,
        campaign_id: Optional[str] = None,
    ) -> CampaignResult:
        """
        Execute a complete red team campaign with multiple sessions.

        Args:
            model_interface: Function to call the target model
            scenario: Attack scenario to execute (object or scenario name)
            model_name: Name/identifier of the target model
            num_sessions: Number of parallel sessions to run
            campaign_id: Optional campaign identifier

        Returns:
            CampaignResult containing all session results and analytics
        """
        campaign_id = campaign_id or str(uuid.uuid4())
        start_time = datetime.now()

        # Resolve scenario
        if isinstance(scenario, str):
            # Load from registry (to be implemented in subtask 6.2)
            from ..scenarios.registry import ScenarioRegistry

            scenario = ScenarioRegistry.get_scenario(scenario)
            if not scenario:
                raise ValueError(f"Unknown scenario: {scenario}")

        logger.info(
            f"Starting campaign {campaign_id} with scenario '{scenario.name}' "
            f"({num_sessions} sessions, model: {model_name})"
        )

        # Create campaign result container
        campaign_result = CampaignResult(
            campaign_id=campaign_id,
            scenario_name=scenario.name,
            model_name=model_name,
            start_time=start_time,
            end_time=start_time,  # Will be updated
        )

        # Execute sessions
        session_tasks = []
        for session_num in range(num_sessions):
            session_task = asyncio.create_task(
                self._run_session(
                    model_interface=model_interface,
                    scenario=scenario,
                    model_name=model_name,
                    session_id=f"{campaign_id}_session_{session_num + 1}",
                )
            )
            session_tasks.append(session_task)

        # Wait for all sessions to complete
        try:
            session_results = await asyncio.gather(*session_tasks, return_exceptions=True)

            # Process session results
            for i, result in enumerate(session_results):
                if isinstance(result, Exception):
                    logger.error(f"Session {i + 1} failed: {result}")
                    # Create a failed session result
                    failed_session = AttackSession(
                        session_id=f"{campaign_id}_session_{i + 1}",
                        scenario=scenario,
                        model_name=model_name,
                        status=AttackStatus.FAILED,
                    )
                    failed_session.end_time = datetime.now()
                    campaign_result.sessions.append(failed_session)
                else:
                    campaign_result.sessions.append(result)

        except Exception as e:
            logger.error(f"Campaign {campaign_id} failed: {e}")
            raise

        # Finalize campaign
        campaign_result.end_time = datetime.now()
        campaign_result.calculate_metrics()

        # Update performance metrics
        self._update_performance_metrics(campaign_result)
        self._campaign_history.append(campaign_result)

        logger.info(
            f"Campaign {campaign_id} completed: {campaign_result.success_rate:.1%} success rate, "
            f"{campaign_result.total_vulnerabilities} vulnerabilities found"
        )

        return campaign_result

    async def _run_session(
        self,
        model_interface: Callable[[str], str],
        scenario: AttackScenario,
        model_name: str,
        session_id: str,
    ) -> AttackSession:
        """Execute a single attack session."""
        session = AttackSession(session_id=session_id, scenario=scenario, model_name=model_name)

        # Register active session
        self._active_sessions[session_id] = session
        session.log_event("session_started", {"scenario": scenario.name, "model": model_name})

        try:
            session.status = AttackStatus.IN_PROGRESS
            logger.info(f"Starting session {session_id} with scenario '{scenario.name}'")

            # Execute attack chains in sequence
            for chain_idx, attack_chain in enumerate(scenario.attack_chains):
                if self._should_stop_session(session):
                    break

                session.current_chain_index = chain_idx
                session.log_event(
                    "chain_started",
                    {"chain_id": attack_chain.chain_id, "chain_name": attack_chain.name},
                )

                # Execute attack chain
                await self._execute_attack_chain(
                    session=session, attack_chain=attack_chain, model_interface=model_interface
                )

                session.log_event(
                    "chain_completed",
                    {
                        "chain_id": attack_chain.chain_id,
                        "success_rate": attack_chain.calculate_success_rate(),
                    },
                )

            # Calculate final results
            session.overall_success_rate = session.calculate_success_rate()
            session.status = (
                AttackStatus.SUCCESS if session.is_successful() else AttackStatus.FAILED
            )

        except Exception as e:
            logger.error(f"Session {session_id} failed: {e}")
            session.status = AttackStatus.FAILED
            session.log_event("session_error", {"error": str(e)})

        finally:
            session.end_time = datetime.now()
            self._active_sessions.pop(session_id, None)
            self._session_history.append(session)

            session.log_event(
                "session_completed",
                {
                    "status": session.status.value,
                    "success_rate": session.overall_success_rate,
                    "duration_minutes": session.get_duration_minutes(),
                },
            )

        return session

    async def _execute_attack_chain(
        self,
        session: AttackSession,
        attack_chain: AttackChain,
        model_interface: Callable[[str], str],
    ):
        """Execute an attack chain with proper orchestration."""
        attack_chain.status = AttackStatus.IN_PROGRESS
        attack_chain.start_time = datetime.now()

        try:
            # Execute based on execution mode
            if attack_chain.execution_mode == ExecutionMode.SEQUENTIAL:
                await self._execute_chain_sequential(session, attack_chain, model_interface)
            elif attack_chain.execution_mode == ExecutionMode.PARALLEL:
                await self._execute_chain_parallel(session, attack_chain, model_interface)
            elif attack_chain.execution_mode in [ExecutionMode.CONDITIONAL, ExecutionMode.ADAPTIVE]:
                await self._execute_chain_adaptive(session, attack_chain, model_interface)

            # Update chain status
            attack_chain.success_rate = attack_chain.calculate_success_rate()
            attack_chain.status = (
                AttackStatus.SUCCESS if attack_chain.success_rate > 0 else AttackStatus.FAILED
            )

        except Exception as e:
            logger.error(f"Attack chain {attack_chain.chain_id} failed: {e}")
            attack_chain.status = AttackStatus.FAILED
            raise

        finally:
            attack_chain.end_time = datetime.now()

    async def _execute_chain_sequential(
        self,
        session: AttackSession,
        attack_chain: AttackChain,
        model_interface: Callable[[str], str],
    ):
        """Execute attack chain steps sequentially."""
        for step in attack_chain.steps:
            if not step.is_ready(session.context):
                logger.warning(f"Step {step.step_id} not ready, skipping")
                continue

            # Execute step
            await self._execute_attack_step(session, step, model_interface)

            # Check if we should continue
            if step.status == AttackStatus.FAILED and not attack_chain.continue_on_failure:
                logger.info(f"Stopping chain due to step failure: {step.step_id}")
                break

            # Add successful step to completed list
            if step.status == AttackStatus.SUCCESS:
                attack_chain.completed_steps.append(step.step_id)
            else:
                attack_chain.failed_steps.append(step.step_id)

    async def _execute_chain_parallel(
        self,
        session: AttackSession,
        attack_chain: AttackChain,
        model_interface: Callable[[str], str],
    ):
        """Execute attack chain steps in parallel."""
        ready_steps = [step for step in attack_chain.steps if step.is_ready(session.context)]

        if not ready_steps:
            logger.warning(
                f"No ready steps found for parallel execution in chain {attack_chain.chain_id}"
            )
            return

        # Create tasks for parallel execution
        step_tasks = [
            asyncio.create_task(self._execute_attack_step(session, step, model_interface))
            for step in ready_steps
        ]

        # Wait for all steps to complete
        await asyncio.gather(*step_tasks, return_exceptions=True)

        # Update chain state
        for step in ready_steps:
            if step.status == AttackStatus.SUCCESS:
                attack_chain.completed_steps.append(step.step_id)
            else:
                attack_chain.failed_steps.append(step.step_id)

    async def _execute_chain_adaptive(
        self,
        session: AttackSession,
        attack_chain: AttackChain,
        model_interface: Callable[[str], str],
    ):
        """Execute attack chain with adaptive logic based on results."""
        max_iterations = len(attack_chain.steps) * 2  # Prevent infinite loops
        iteration = 0

        while not attack_chain.is_complete() and iteration < max_iterations:
            iteration += 1

            # Get next ready steps
            next_steps = attack_chain.get_next_steps(session.context)
            if not next_steps:
                break

            # Execute the next batch of steps
            for step in next_steps:
                await self._execute_attack_step(session, step, model_interface)

                # Update chain state
                if step.status == AttackStatus.SUCCESS:
                    attack_chain.completed_steps.append(step.step_id)
                    # Adaptive: Update context based on success
                    session.context.record_success(step.name)
                else:
                    attack_chain.failed_steps.append(step.step_id)
                    session.context.record_failure(step.name)

            # Adaptive: Modify strategy based on results
            if self.config.get("enable_adaptive_execution", True):
                await self._adapt_strategy(session, attack_chain, next_steps)

    async def _execute_attack_step(
        self, session: AttackSession, step: AttackStep, model_interface: Callable[[str], str]
    ) -> bool:
        """Execute a single attack step."""
        step.status = AttackStatus.IN_PROGRESS
        step.start_time = datetime.now()
        step.execution_count += 1

        try:
            session.log_event(
                "step_started",
                {
                    "step_id": step.step_id,
                    "step_name": step.name,
                    "attack_type": step.attack_type.value,
                },
            )

            # Generate prompt with context
            prompt = step.generate_prompt(session.context)
            step.last_prompt = prompt

            logger.debug(f"Executing step {step.step_id} with prompt: {prompt[:100]}...")

            # Apply rate limiting
            await self._apply_rate_limiting()

            # Get model response
            response = await asyncio.get_event_loop().run_in_executor(None, model_interface, prompt)
            step.last_response = response

            # Update session context
            session.context.add_response(prompt, response)

            # Evaluate success
            success = step.evaluate_success(response, session.context)
            explicit_failure = step.evaluate_failure(response)

            if explicit_failure:
                step.status = AttackStatus.BLOCKED
                step.failure_reason = "Explicit failure criteria met"
            elif success:
                step.status = AttackStatus.SUCCESS
                step.success_score = 1.0
                # Record successful technique
                session.context.record_success(step.name)
            else:
                # Check if we should retry
                if step.execution_count < step.max_retries + 1:
                    logger.info(
                        f"Step {step.step_id} failed, retrying ({step.execution_count}/{step.max_retries + 1})"
                    )
                    step.status = AttackStatus.PENDING  # Reset for retry
                    return await self._execute_attack_step(session, step, model_interface)
                else:
                    step.status = AttackStatus.FAILED
                    step.failure_reason = "Success criteria not met"
                    session.context.record_failure(step.name)

            # Perform vulnerability scanning if enabled
            if self.config.get("enable_vulnerability_scanning", True):
                await self._perform_vulnerability_scan(session, step, prompt, response)

            session.log_event(
                "step_completed",
                {
                    "step_id": step.step_id,
                    "status": step.status.value,
                    "success_score": step.success_score,
                    "response_length": len(response),
                },
            )

            return step.status == AttackStatus.SUCCESS

        except asyncio.TimeoutError:
            step.status = AttackStatus.TIMEOUT
            step.failure_reason = "Step execution timed out"
            logger.warning(f"Step {step.step_id} timed out")
            return False

        except Exception as e:
            step.status = AttackStatus.FAILED
            step.failure_reason = f"Execution error: {str(e)}"
            logger.error(f"Step {step.step_id} failed: {e}")
            return False

        finally:
            step.end_time = datetime.now()

    async def _perform_vulnerability_scan(
        self, session: AttackSession, step: AttackStep, prompt: str, response: str
    ):
        """Perform vulnerability scanning on the step's response."""
        try:
            # Use the security scanner to assess the response
            scan_result = await self.security_scanner.assess_vulnerability(
                response=response, attack_prompt=prompt, attack_id=step.step_id
            )

            # Store scan result
            step.scan_result = scan_result
            session.add_scan_result(scan_result)

            # Update step success score based on vulnerabilities found
            if scan_result.vulnerabilities:
                # Higher vulnerability confidence increases step success score
                max_confidence = max(vuln.confidence_score for vuln in scan_result.vulnerabilities)
                step.success_score = max(step.success_score, max_confidence)

                logger.info(
                    f"Step {step.step_id} found {len(scan_result.vulnerabilities)} vulnerabilities"
                )

        except Exception as e:
            logger.error(f"Vulnerability scan failed for step {step.step_id}: {e}")

    async def _adapt_strategy(
        self, session: AttackSession, attack_chain: AttackChain, executed_steps: List[AttackStep]
    ):
        """Adapt attack strategy based on execution results."""
        # Analyze success/failure patterns
        recent_successes = [step for step in executed_steps if step.status == AttackStatus.SUCCESS]
        recent_failures = [
            step
            for step in executed_steps
            if step.status in [AttackStatus.FAILED, AttackStatus.BLOCKED]
        ]

        # Update context based on patterns
        if len(recent_successes) > len(recent_failures):
            # Successful pattern - continue with similar techniques
            session.context.state_variables["strategy_confidence"] = "high"
            for step in recent_successes:
                for technique in step.evasion_techniques:
                    session.context.successful_techniques.append(technique)
        else:
            # Failing pattern - need to adapt
            session.context.state_variables["strategy_confidence"] = "low"

            # Try different evasion techniques for remaining steps
            remaining_steps = [
                step for step in attack_chain.steps if step.status == AttackStatus.PENDING
            ]
            if remaining_steps and hasattr(self, "_evasion_engine"):
                # Apply evasion techniques (to be implemented in subtask 6.3)
                pass

    async def _apply_rate_limiting(self):
        """Apply rate limiting between requests."""
        if "rate_limiting" in self.config:
            rate_config = self.config["rate_limiting"]
            delay = 1.0 / rate_config.get("requests_per_second", 1)
            await asyncio.sleep(delay)

    def _should_stop_session(self, session: AttackSession) -> bool:
        """Check if session should be stopped based on time limits or other criteria."""
        max_duration = timedelta(minutes=session.scenario.max_duration_minutes)
        current_duration = datetime.now() - session.start_time

        if current_duration > max_duration:
            logger.warning(f"Session {session.session_id} exceeded max duration")
            return True

        return False

    def _update_performance_metrics(self, campaign_result: CampaignResult):
        """Update internal performance metrics."""
        self._performance_metrics["total_campaigns"] += 1
        self._performance_metrics["total_sessions"] += len(campaign_result.sessions)
        self._performance_metrics["total_vulnerabilities_found"] += (
            campaign_result.total_vulnerabilities
        )

        # Update average success rate
        total_success = self._performance_metrics["average_success_rate"] * (
            self._performance_metrics["total_campaigns"] - 1
        )
        total_success += campaign_result.success_rate
        self._performance_metrics["average_success_rate"] = (
            total_success / self._performance_metrics["total_campaigns"]
        )

    # Session management methods

    def get_active_sessions(self) -> Dict[str, AttackSession]:
        """Get currently active sessions."""
        return self._active_sessions.copy()

    def get_session(self, session_id: str) -> Optional[AttackSession]:
        """Get a specific session by ID."""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        # Search in history
        for session in self._session_history:
            if session.session_id == session_id:
                return session

        return None

    async def pause_session(self, session_id: str) -> bool:
        """Pause an active session."""
        if session_id not in self._active_sessions:
            return False

        session = self._active_sessions[session_id]
        session.log_event("session_paused", {"timestamp": datetime.now().isoformat()})
        # Implementation would involve pausing execution loops
        return True

    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        if session_id not in self._active_sessions:
            return False

        session = self._active_sessions[session_id]
        session.log_event("session_resumed", {"timestamp": datetime.now().isoformat()})
        # Implementation would involve resuming execution
        return True

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an active session."""
        if session_id not in self._active_sessions:
            return False

        session = self._active_sessions[session_id]
        session.status = AttackStatus.CANCELLED
        session.end_time = datetime.now()
        session.log_event("session_cancelled", {"timestamp": datetime.now().isoformat()})

        # Move to history
        self._session_history.append(session)
        del self._active_sessions[session_id]

        return True

    # Plugin system methods

    def register_attack_plugin(self, name: str, plugin: Callable):
        """Register a custom attack plugin."""
        self._attack_plugins[name] = plugin
        logger.info(f"Registered attack plugin: {name}")

    def load_custom_module(self, name: str, module: Any):
        """Load a custom attack module."""
        self._custom_modules[name] = module
        logger.info(f"Loaded custom module: {name}")

    # Analytics and reporting

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        return self._performance_metrics.copy()

    def get_campaign_history(self) -> List[CampaignResult]:
        """Get history of completed campaigns."""
        return self._campaign_history.copy()

    def export_session_data(
        self, session_id: str, format_type: str = "json"
    ) -> Optional[Dict[str, Any]]:
        """Export session data in specified format."""
        session = self.get_session(session_id)
        if not session:
            return None

        if format_type == "json":
            return {
                "session_id": session.session_id,
                "scenario_name": session.scenario.name,
                "model_name": session.model_name,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "status": session.status.value,
                "success_rate": session.overall_success_rate,
                "duration_minutes": session.get_duration_minutes(),
                "vulnerabilities_found": len(session.vulnerability_findings),
                "execution_log": session.execution_log,
                "context_state": {
                    "extracted_data": session.context.extracted_data,
                    "successful_techniques": session.context.successful_techniques,
                    "failed_attempts": session.context.failed_attempts,
                    "escalated_privileges": session.context.escalated_privileges,
                },
            }

        return None
