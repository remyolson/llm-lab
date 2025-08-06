"""
Core workflow orchestration engine for automated and manual red team operations.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.models import AttackScenario, AttackSession, AttackStatus, CampaignResult
from ..core.simulator import RedTeamSimulator

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of red team workflows."""

    AUTOMATED = "automated"
    MANUAL = "manual"
    HYBRID = "hybrid"
    SCHEDULED = "scheduled"
    CONTINUOUS = "continuous"
    TRIGGERED = "triggered"


class WorkflowState(Enum):
    """Workflow execution states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowCheckpoint:
    """Checkpoint for workflow state persistence."""

    def __init__(
        self,
        workflow_id: str,
        state: WorkflowState,
        progress: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ):
        self.workflow_id = workflow_id
        self.state = state
        self.progress = progress
        self.timestamp = timestamp or datetime.now()
        self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "progress": self.progress,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowCheckpoint":
        """Create checkpoint from dictionary."""
        checkpoint = cls(
            workflow_id=data["workflow_id"],
            state=WorkflowState(data["state"]),
            progress=data["progress"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
        checkpoint.metadata = data.get("metadata", {})
        return checkpoint


class WorkflowEngine:
    """
    Main workflow orchestration engine for red team operations.

    Manages automated campaigns, manual operations, and hybrid workflows
    with support for scheduling, triggering, and continuous testing.
    """

    def __init__(
        self, red_team_simulator: RedTeamSimulator, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the workflow engine.

        Args:
            red_team_simulator: Core simulator for attack execution
            config: Workflow engine configuration
        """
        self.simulator = red_team_simulator
        self.config = config or self._get_default_config()

        # Workflow management
        self._active_workflows: Dict[str, "WorkflowExecutor"] = {}
        self._workflow_history: List[Dict[str, Any]] = []
        self._scheduled_workflows: List[Dict[str, Any]] = []

        # Checkpoint management
        self._checkpoints: Dict[str, List[WorkflowCheckpoint]] = {}
        self._checkpoint_dir = Path(self.config.get("checkpoint_dir", ".red_team_checkpoints"))
        self._checkpoint_dir.mkdir(exist_ok=True)

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Scheduler task
        self._scheduler_task: Optional[asyncio.Task] = None

        logger.info(
            "WorkflowEngine initialized with checkpoint directory: %s", self._checkpoint_dir
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default workflow engine configuration."""
        return {
            "max_concurrent_workflows": 3,
            "checkpoint_interval_minutes": 5,
            "checkpoint_dir": ".red_team_checkpoints",
            "enable_auto_checkpoint": True,
            "enable_scheduler": True,
            "scheduler_interval_seconds": 60,
            "workflow_timeout_hours": 24,
            "enable_notifications": True,
            "enable_result_correlation": True,
            "correlation_window_hours": 24,
        }

    async def create_workflow(
        self,
        name: str,
        workflow_type: WorkflowType,
        scenarios: List[Union[AttackScenario, str]],
        model_interface: Callable[[str], str],
        model_name: str = "unknown",
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create and optionally start a new workflow.

        Args:
            name: Workflow name
            workflow_type: Type of workflow
            scenarios: List of attack scenarios to execute
            model_interface: Function to call the target model
            model_name: Name of the target model
            config: Workflow-specific configuration

        Returns:
            Workflow ID
        """
        workflow_id = str(uuid.uuid4())

        # Create workflow executor
        executor = WorkflowExecutor(
            workflow_id=workflow_id,
            name=name,
            workflow_type=workflow_type,
            scenarios=scenarios,
            model_interface=model_interface,
            model_name=model_name,
            simulator=self.simulator,
            config=config or {},
        )

        # Register workflow
        self._active_workflows[workflow_id] = executor

        # Register event handlers
        executor.on_state_change(self._handle_workflow_state_change)
        executor.on_progress(self._handle_workflow_progress)

        logger.info("Created workflow %s (%s) with %d scenarios", name, workflow_id, len(scenarios))

        # Auto-start if configured
        if workflow_type in [WorkflowType.AUTOMATED, WorkflowType.CONTINUOUS]:
            await self.start_workflow(workflow_id)

        return workflow_id

    async def start_workflow(self, workflow_id: str) -> bool:
        """Start or resume a workflow."""
        if workflow_id not in self._active_workflows:
            logger.error("Workflow %s not found", workflow_id)
            return False

        executor = self._active_workflows[workflow_id]

        # Check concurrent workflow limit
        running_count = sum(
            1 for w in self._active_workflows.values() if w.state == WorkflowState.RUNNING
        )
        if running_count >= self.config.get("max_concurrent_workflows", 3):
            logger.warning(
                "Maximum concurrent workflows reached (%d). Queuing workflow %s",
                running_count,
                workflow_id,
            )
            # Queue the workflow
            executor.state = WorkflowState.IDLE
            return False

        # Start the workflow
        await executor.start()

        # Start checkpoint monitoring if enabled
        if self.config.get("enable_auto_checkpoint", True):
            asyncio.create_task(self._monitor_checkpoints(workflow_id))

        return True

    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        if workflow_id not in self._active_workflows:
            return False

        executor = self._active_workflows[workflow_id]
        await executor.pause()

        # Create checkpoint
        await self._create_checkpoint(workflow_id)

        return True

    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        if workflow_id not in self._active_workflows:
            # Try to restore from checkpoint
            checkpoint = await self._load_latest_checkpoint(workflow_id)
            if checkpoint:
                return await self._restore_from_checkpoint(checkpoint)
            return False

        executor = self._active_workflows[workflow_id]
        await executor.resume()

        return True

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        if workflow_id not in self._active_workflows:
            return False

        executor = self._active_workflows[workflow_id]
        await executor.cancel()

        # Move to history
        self._workflow_history.append(
            {
                "workflow_id": workflow_id,
                "name": executor.name,
                "type": executor.workflow_type.value,
                "state": "cancelled",
                "end_time": datetime.now().isoformat(),
            }
        )

        # Clean up
        del self._active_workflows[workflow_id]

        return True

    async def schedule_workflow(
        self,
        name: str,
        workflow_type: WorkflowType,
        scenarios: List[Union[AttackScenario, str]],
        model_interface: Callable[[str], str],
        model_name: str,
        schedule: Dict[str, Any],
    ) -> str:
        """
        Schedule a workflow for future or recurring execution.

        Args:
            name: Workflow name
            workflow_type: Type of workflow
            scenarios: Attack scenarios
            model_interface: Model interface function
            model_name: Model name
            schedule: Schedule configuration (cron, interval, or specific time)

        Returns:
            Schedule ID
        """
        schedule_id = str(uuid.uuid4())

        scheduled_workflow = {
            "schedule_id": schedule_id,
            "name": name,
            "workflow_type": workflow_type.value,
            "scenarios": scenarios,
            "model_name": model_name,
            "schedule": schedule,
            "created_at": datetime.now().isoformat(),
            "next_run": self._calculate_next_run(schedule),
            "enabled": True,
        }

        self._scheduled_workflows.append(scheduled_workflow)

        # Start scheduler if not running
        if self.config.get("enable_scheduler", True) and not self._scheduler_task:
            self._scheduler_task = asyncio.create_task(self._run_scheduler())

        logger.info("Scheduled workflow %s (%s) with schedule: %s", name, schedule_id, schedule)

        return schedule_id

    def _calculate_next_run(self, schedule: Dict[str, Any]) -> Optional[str]:
        """Calculate next run time based on schedule configuration."""
        if "interval_minutes" in schedule:
            next_run = datetime.now() + timedelta(minutes=schedule["interval_minutes"])
            return next_run.isoformat()
        elif "daily_at" in schedule:
            # Parse time string (e.g., "14:30")
            time_parts = schedule["daily_at"].split(":")
            hour = int(time_parts[0])
            minute = int(time_parts[1]) if len(time_parts) > 1 else 0

            next_run = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= datetime.now():
                next_run += timedelta(days=1)
            return next_run.isoformat()
        elif "specific_time" in schedule:
            return schedule["specific_time"]

        return None

    async def _run_scheduler(self):
        """Run the workflow scheduler."""
        logger.info("Workflow scheduler started")

        while self.config.get("enable_scheduler", True):
            try:
                current_time = datetime.now()

                for scheduled in self._scheduled_workflows:
                    if not scheduled.get("enabled", True):
                        continue

                    next_run_str = scheduled.get("next_run")
                    if not next_run_str:
                        continue

                    next_run = datetime.fromisoformat(next_run_str)

                    if current_time >= next_run:
                        # Time to run this workflow
                        logger.info("Executing scheduled workflow: %s", scheduled["name"])

                        # Create and start workflow
                        # Note: model_interface needs to be stored or recreated
                        # This is a simplified version
                        workflow_id = await self.create_workflow(
                            name=scheduled["name"],
                            workflow_type=WorkflowType(scheduled["workflow_type"]),
                            scenarios=scheduled["scenarios"],
                            model_interface=lambda x: x,  # Placeholder
                            model_name=scheduled["model_name"],
                        )

                        # Update next run time
                        scheduled["last_run"] = current_time.isoformat()
                        scheduled["next_run"] = self._calculate_next_run(scheduled["schedule"])

                # Sleep until next check
                await asyncio.sleep(self.config.get("scheduler_interval_seconds", 60))

            except Exception as e:
                logger.error("Scheduler error: %s", e)
                await asyncio.sleep(60)  # Retry after error

    async def _monitor_checkpoints(self, workflow_id: str):
        """Monitor and create periodic checkpoints for a workflow."""
        interval = self.config.get("checkpoint_interval_minutes", 5) * 60

        while workflow_id in self._active_workflows:
            await asyncio.sleep(interval)

            if workflow_id not in self._active_workflows:
                break

            executor = self._active_workflows[workflow_id]
            if executor.state == WorkflowState.RUNNING:
                await self._create_checkpoint(workflow_id)

    async def _create_checkpoint(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Create a checkpoint for a workflow."""
        if workflow_id not in self._active_workflows:
            return None

        executor = self._active_workflows[workflow_id]

        # Get workflow progress
        progress = await executor.get_progress()

        # Create checkpoint
        checkpoint = WorkflowCheckpoint(
            workflow_id=workflow_id, state=executor.state, progress=progress
        )

        # Store checkpoint
        if workflow_id not in self._checkpoints:
            self._checkpoints[workflow_id] = []
        self._checkpoints[workflow_id].append(checkpoint)

        # Save to disk
        checkpoint_file = self._checkpoint_dir / f"{workflow_id}_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.debug("Created checkpoint for workflow %s", workflow_id)

        return checkpoint

    async def _load_latest_checkpoint(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Load the latest checkpoint for a workflow."""
        checkpoint_file = self._checkpoint_dir / f"{workflow_id}_checkpoint.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)

            return WorkflowCheckpoint.from_dict(data)

        except Exception as e:
            logger.error("Failed to load checkpoint for %s: %s", workflow_id, e)
            return None

    async def _restore_from_checkpoint(self, checkpoint: WorkflowCheckpoint) -> bool:
        """Restore a workflow from a checkpoint."""
        # This would require recreating the workflow executor
        # with the saved state - simplified for now
        logger.info("Restoring workflow %s from checkpoint", checkpoint.workflow_id)
        return True

    def _handle_workflow_state_change(
        self, workflow_id: str, old_state: WorkflowState, new_state: WorkflowState
    ):
        """Handle workflow state change events."""
        logger.info(
            "Workflow %s state changed: %s -> %s", workflow_id, old_state.value, new_state.value
        )

        # Trigger registered event handlers
        for handler in self._event_handlers.get("state_change", []):
            handler(workflow_id, old_state, new_state)

        # Handle completion
        if new_state in [WorkflowState.COMPLETED, WorkflowState.FAILED]:
            self._handle_workflow_completion(workflow_id)

    def _handle_workflow_progress(self, workflow_id: str, progress: Dict[str, Any]):
        """Handle workflow progress updates."""
        # Trigger registered event handlers
        for handler in self._event_handlers.get("progress", []):
            handler(workflow_id, progress)

    def _handle_workflow_completion(self, workflow_id: str):
        """Handle workflow completion."""
        if workflow_id not in self._active_workflows:
            return

        executor = self._active_workflows[workflow_id]

        # Move to history
        self._workflow_history.append(
            {
                "workflow_id": workflow_id,
                "name": executor.name,
                "type": executor.workflow_type.value,
                "state": executor.state.value,
                "end_time": datetime.now().isoformat(),
                "results": executor.get_results(),
            }
        )

        # Clean up checkpoints
        checkpoint_file = self._checkpoint_dir / f"{workflow_id}_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        # Remove from active workflows
        del self._active_workflows[workflow_id]

        logger.info("Workflow %s completed and moved to history", workflow_id)

    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active workflows."""
        return {
            wf_id: {
                "name": executor.name,
                "type": executor.workflow_type.value,
                "state": executor.state.value,
                "progress": executor.get_progress(),
            }
            for wf_id, executor in self._active_workflows.items()
        }

    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        return self._workflow_history.copy()

    def get_scheduled_workflows(self) -> List[Dict[str, Any]]:
        """Get scheduled workflows."""
        return self._scheduled_workflows.copy()


class WorkflowExecutor:
    """
    Executor for individual workflow instances.

    Manages the execution of a specific workflow including state management,
    progress tracking, and result collection.
    """

    def __init__(
        self,
        workflow_id: str,
        name: str,
        workflow_type: WorkflowType,
        scenarios: List[Union[AttackScenario, str]],
        model_interface: Callable[[str], str],
        model_name: str,
        simulator: RedTeamSimulator,
        config: Dict[str, Any],
    ):
        self.workflow_id = workflow_id
        self.name = name
        self.workflow_type = workflow_type
        self.scenarios = scenarios
        self.model_interface = model_interface
        self.model_name = model_name
        self.simulator = simulator
        self.config = config

        # State management
        self.state = WorkflowState.IDLE
        self._state_lock = asyncio.Lock()

        # Progress tracking
        self.current_scenario_index = 0
        self.completed_scenarios = []
        self.failed_scenarios = []
        self.campaign_results: List[CampaignResult] = []

        # Execution control
        self._execution_task: Optional[asyncio.Task] = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default

        # Event callbacks
        self._state_change_callbacks: List[Callable] = []
        self._progress_callbacks: List[Callable] = []

        # Timing
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    async def start(self):
        """Start workflow execution."""
        async with self._state_lock:
            if self.state != WorkflowState.IDLE:
                logger.warning(
                    "Cannot start workflow %s in state %s", self.workflow_id, self.state.value
                )
                return

            self._set_state(WorkflowState.RUNNING)
            self.start_time = datetime.now()

        # Start execution
        self._execution_task = asyncio.create_task(self._execute())

    async def pause(self):
        """Pause workflow execution."""
        async with self._state_lock:
            if self.state != WorkflowState.RUNNING:
                return

            self._pause_event.clear()
            self._set_state(WorkflowState.PAUSED)

    async def resume(self):
        """Resume workflow execution."""
        async with self._state_lock:
            if self.state != WorkflowState.PAUSED:
                return

            self._pause_event.set()
            self._set_state(WorkflowState.RUNNING)

    async def cancel(self):
        """Cancel workflow execution."""
        async with self._state_lock:
            self._set_state(WorkflowState.CANCELLED)

            if self._execution_task:
                self._execution_task.cancel()

    async def _execute(self):
        """Execute the workflow."""
        try:
            # Execute based on workflow type
            if self.workflow_type == WorkflowType.AUTOMATED:
                await self._execute_automated()
            elif self.workflow_type == WorkflowType.CONTINUOUS:
                await self._execute_continuous()
            elif self.workflow_type == WorkflowType.MANUAL:
                await self._execute_manual()
            elif self.workflow_type == WorkflowType.HYBRID:
                await self._execute_hybrid()
            else:
                await self._execute_automated()  # Default

            # Mark as completed
            async with self._state_lock:
                self._set_state(WorkflowState.COMPLETED)
                self.end_time = datetime.now()

        except asyncio.CancelledError:
            logger.info("Workflow %s cancelled", self.workflow_id)
            raise

        except Exception as e:
            logger.error("Workflow %s failed: %s", self.workflow_id, e)
            async with self._state_lock:
                self._set_state(WorkflowState.FAILED)
                self.end_time = datetime.now()

    async def _execute_automated(self):
        """Execute automated workflow."""
        for i, scenario in enumerate(self.scenarios):
            # Check for pause
            await self._pause_event.wait()

            # Check for cancellation
            if self.state == WorkflowState.CANCELLED:
                break

            self.current_scenario_index = i
            self._notify_progress()

            # Run campaign for scenario
            campaign_result = await self.simulator.run_campaign(
                model_interface=self.model_interface,
                scenario=scenario,
                model_name=self.model_name,
                num_sessions=self.config.get("sessions_per_scenario", 1),
                campaign_id=f"{self.workflow_id}_scenario_{i}",
            )

            self.campaign_results.append(campaign_result)

            # Track completion
            if campaign_result.success_rate > 0:
                self.completed_scenarios.append(scenario)
            else:
                self.failed_scenarios.append(scenario)

    async def _execute_continuous(self):
        """Execute continuous testing workflow."""
        iteration = 0
        max_iterations = self.config.get("max_iterations", 100)

        while iteration < max_iterations:
            # Check for pause
            await self._pause_event.wait()

            # Check for cancellation
            if self.state == WorkflowState.CANCELLED:
                break

            # Run all scenarios
            for scenario in self.scenarios:
                campaign_result = await self.simulator.run_campaign(
                    model_interface=self.model_interface,
                    scenario=scenario,
                    model_name=self.model_name,
                    num_sessions=1,
                    campaign_id=f"{self.workflow_id}_iter_{iteration}",
                )

                self.campaign_results.append(campaign_result)

            iteration += 1

            # Sleep between iterations
            await asyncio.sleep(self.config.get("iteration_delay_seconds", 60))

    async def _execute_manual(self):
        """Execute manual workflow (placeholder for manual interface)."""
        # Manual workflows would be controlled through the ManualRedTeamInterface
        # This is a simplified placeholder
        logger.info("Manual workflow %s ready for interaction", self.workflow_id)

        # Wait for manual completion signal
        while self.state == WorkflowState.RUNNING:
            await asyncio.sleep(1)

    async def _execute_hybrid(self):
        """Execute hybrid workflow combining automated and manual steps."""
        # Execute automated parts
        await self._execute_automated()

        # Then switch to manual mode for review/adjustment
        logger.info(
            "Workflow %s automated phase complete, switching to manual review", self.workflow_id
        )

        # Wait for manual completion
        while self.state == WorkflowState.RUNNING:
            await asyncio.sleep(1)

    def _set_state(self, new_state: WorkflowState):
        """Set workflow state and trigger callbacks."""
        old_state = self.state
        self.state = new_state

        # Trigger callbacks
        for callback in self._state_change_callbacks:
            callback(self.workflow_id, old_state, new_state)

    def _notify_progress(self):
        """Notify progress callbacks."""
        progress = self.get_progress()

        for callback in self._progress_callbacks:
            callback(self.workflow_id, progress)

    def on_state_change(self, callback: Callable):
        """Register state change callback."""
        self._state_change_callbacks.append(callback)

    def on_progress(self, callback: Callable):
        """Register progress callback."""
        self._progress_callbacks.append(callback)

    def get_progress(self) -> Dict[str, Any]:
        """Get current workflow progress."""
        total_scenarios = len(self.scenarios)
        completed = len(self.completed_scenarios)
        failed = len(self.failed_scenarios)

        return {
            "current_scenario_index": self.current_scenario_index,
            "total_scenarios": total_scenarios,
            "completed_scenarios": completed,
            "failed_scenarios": failed,
            "progress_percentage": (completed + failed) / total_scenarios * 100
            if total_scenarios > 0
            else 0,
            "state": self.state.value,
            "campaign_results_count": len(self.campaign_results),
        }

    async def get_progress_async(self) -> Dict[str, Any]:
        """Get current workflow progress (async version for checkpointing)."""
        return self.get_progress()

    def get_results(self) -> Dict[str, Any]:
        """Get workflow execution results."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "type": self.workflow_type.value,
            "state": self.state.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "scenarios_executed": len(self.completed_scenarios) + len(self.failed_scenarios),
            "scenarios_successful": len(self.completed_scenarios),
            "scenarios_failed": len(self.failed_scenarios),
            "campaign_results": [
                {
                    "campaign_id": cr.campaign_id,
                    "scenario_name": cr.scenario_name,
                    "success_rate": cr.success_rate,
                    "vulnerabilities_found": cr.total_vulnerabilities,
                }
                for cr in self.campaign_results
            ],
        }
