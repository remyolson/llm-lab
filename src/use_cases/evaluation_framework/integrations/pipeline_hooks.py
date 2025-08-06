"""
Fine-Tuning Pipeline Integration Hooks

This module provides integration hooks for automatic evaluation triggers,
progress callbacks, automatic report generation, model registry integration,
deployment triggers, and notification systems.

Example:
    hooks = FineTuningHooks()

    # Register hooks
    hooks.register_hook(
        HookType.POST_TRAINING,
        lambda model: run_evaluation(model)
    )

    # Integrate with pipeline
    integration = PipelineIntegration(hooks)
    integration.attach_to_trainer(trainer)
"""

import asyncio
import inspect
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..analysis.cost_benefit import CostBenefitAnalyzer

# Import evaluation components
from ..benchmark_runner import AutoBenchmarkRunner, BenchmarkConfig, ComparisonResult
from ..reporting.report_generator import ReportConfig, ReportContent, ReportFormat, ReportGenerator

# Import fine-tuning components
try:
    from ...fine_tuning.monitoring.structured_logger import StructuredLogger
    from ...fine_tuning.trainers.base_trainer import BaseTrainer

    FINE_TUNING_AVAILABLE = True
except ImportError:
    FINE_TUNING_AVAILABLE = False
    BaseTrainer = None

# Import notification libraries
try:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import requests

    WEBHOOK_AVAILABLE = True
except ImportError:
    WEBHOOK_AVAILABLE = False

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Types of hooks in the fine-tuning pipeline."""

    PRE_TRAINING = "pre_training"
    POST_EPOCH = "post_epoch"
    POST_TRAINING = "post_training"
    PRE_EVALUATION = "pre_evaluation"
    POST_EVALUATION = "post_evaluation"
    ON_CHECKPOINT = "on_checkpoint"
    ON_IMPROVEMENT = "on_improvement"
    ON_REGRESSION = "on_regression"
    ON_ERROR = "on_error"
    ON_COMPLETION = "on_completion"


class NotificationType(Enum):
    """Types of notifications."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    CONSOLE = "console"
    FILE = "file"


@dataclass
class HookConfig:
    """Configuration for pipeline hooks."""

    auto_evaluate: bool = True
    auto_generate_report: bool = True
    auto_deploy: bool = False
    deployment_threshold: float = 10.0  # Minimum improvement % for auto-deploy
    notification_on_completion: bool = True
    notification_on_error: bool = True
    notification_on_regression: bool = True
    notification_channels: List[NotificationType] = field(
        default_factory=lambda: [NotificationType.CONSOLE]
    )
    benchmark_config: Optional[BenchmarkConfig] = None
    report_config: Optional[ReportConfig] = None
    model_registry_enabled: bool = True
    model_registry_path: str = "./model_registry"
    async_execution: bool = True
    max_parallel_hooks: int = 3
    retry_on_failure: bool = True
    max_retries: int = 3

    # Notification settings
    email_config: Dict[str, Any] = field(default_factory=dict)
    webhook_url: Optional[str] = None
    slack_webhook: Optional[str] = None
    teams_webhook: Optional[str] = None


@dataclass
class HookResult:
    """Result from hook execution."""

    hook_type: HookType
    success: bool
    timestamp: datetime
    duration_seconds: float
    result_data: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FineTuningHooks:
    """Hook system for fine-tuning pipeline integration."""

    def __init__(self, config: Optional[HookConfig] = None):
        """Initialize hooks system.

        Args:
            config: Hook configuration
        """
        self.config = config or HookConfig()
        self.hooks: Dict[HookType, List[Callable]] = {hook_type: [] for hook_type in HookType}
        self.hook_history: List[HookResult] = []
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_hooks)

        # Initialize components
        self.benchmark_runner = None
        self.report_generator = None
        self.cost_analyzer = None
        self.logger = StructuredLogger("pipeline_hooks")

        # Initialize default hooks
        self._init_default_hooks()

    def _init_default_hooks(self):
        """Initialize default hooks based on configuration."""
        if self.config.auto_evaluate:
            self.register_hook(HookType.POST_TRAINING, self._auto_evaluate_hook)

        if self.config.auto_generate_report:
            self.register_hook(HookType.POST_EVALUATION, self._auto_report_hook)

        if self.config.notification_on_completion:
            self.register_hook(HookType.ON_COMPLETION, self._notification_hook)

        if self.config.notification_on_error:
            self.register_hook(HookType.ON_ERROR, self._error_notification_hook)

        if self.config.notification_on_regression:
            self.register_hook(HookType.ON_REGRESSION, self._regression_notification_hook)

    def register_hook(self, hook_type: HookType, callback: Callable, priority: int = 0):
        """Register a hook callback.

        Args:
            hook_type: Type of hook
            callback: Callback function
            priority: Execution priority (higher = earlier)
        """
        # Store with priority
        self.hooks[hook_type].append((priority, callback))

        # Sort by priority
        self.hooks[hook_type].sort(key=lambda x: x[0], reverse=True)

        logger.info(f"Registered hook for {hook_type.value} with priority {priority}")

    def unregister_hook(self, hook_type: HookType, callback: Callable):
        """Unregister a hook callback.

        Args:
            hook_type: Type of hook
            callback: Callback to remove
        """
        self.hooks[hook_type] = [(p, c) for p, c in self.hooks[hook_type] if c != callback]

    async def trigger_hook_async(self, hook_type: HookType, **kwargs) -> List[HookResult]:
        """Trigger hooks asynchronously.

        Args:
            hook_type: Type of hook to trigger
            **kwargs: Arguments to pass to callbacks

        Returns:
            List of hook results
        """
        results = []
        callbacks = [callback for _, callback in self.hooks[hook_type]]

        if not callbacks:
            return results

        logger.info(f"Triggering {len(callbacks)} hooks for {hook_type.value}")

        # Execute hooks asynchronously
        tasks = []
        for callback in callbacks:
            if asyncio.iscoroutinefunction(callback):
                task = callback(**kwargs)
            else:
                task = asyncio.get_event_loop().run_in_executor(self.executor, callback, **kwargs)
            tasks.append(task)

        # Wait for all hooks to complete
        hook_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for callback, result in zip(callbacks, hook_results):
            start_time = datetime.now()

            if isinstance(result, Exception):
                hook_result = HookResult(
                    hook_type=hook_type,
                    success=False,
                    timestamp=start_time,
                    duration_seconds=0,
                    error_message=str(result),
                )

                if self.config.retry_on_failure:
                    # Retry logic
                    for retry in range(self.config.max_retries):
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                result = await callback(**kwargs)
                            else:
                                result = await asyncio.get_event_loop().run_in_executor(
                                    self.executor, callback, **kwargs
                                )
                            hook_result.success = True
                            hook_result.result_data = result
                            break
                        except Exception as e:
                            logger.warning(f"Retry {retry + 1} failed: {e}")
            else:
                hook_result = HookResult(
                    hook_type=hook_type,
                    success=True,
                    timestamp=start_time,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    result_data=result,
                )

            results.append(hook_result)
            self.hook_history.append(hook_result)

        return results

    def trigger_hook(self, hook_type: HookType, **kwargs) -> List[HookResult]:
        """Trigger hooks synchronously.

        Args:
            hook_type: Type of hook to trigger
            **kwargs: Arguments to pass to callbacks

        Returns:
            List of hook results
        """
        if self.config.async_execution:
            # Run async version in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.trigger_hook_async(hook_type, **kwargs))
            finally:
                loop.close()

        # Synchronous execution
        results = []
        callbacks = [callback for _, callback in self.hooks[hook_type]]

        for callback in callbacks:
            start_time = datetime.now()

            try:
                result = callback(**kwargs)
                hook_result = HookResult(
                    hook_type=hook_type,
                    success=True,
                    timestamp=start_time,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    result_data=result,
                )
            except Exception as e:
                logger.error(f"Hook execution failed: {e}")
                hook_result = HookResult(
                    hook_type=hook_type,
                    success=False,
                    timestamp=start_time,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    error_message=str(e),
                )

            results.append(hook_result)
            self.hook_history.append(hook_result)

        return results

    def _auto_evaluate_hook(self, **kwargs) -> ComparisonResult:
        """Automatic evaluation hook.

        Args:
            **kwargs: Hook arguments including model paths

        Returns:
            Comparison results
        """
        logger.info("Running automatic evaluation")

        # Extract model paths
        base_model = kwargs.get("base_model")
        fine_tuned_model = kwargs.get("fine_tuned_model")

        if not base_model or not fine_tuned_model:
            raise ValueError("Base and fine-tuned model paths required")

        # Initialize benchmark runner if needed
        if not self.benchmark_runner:
            self.benchmark_runner = AutoBenchmarkRunner(config=self.config.benchmark_config)

        # Run evaluation
        comparison = self.benchmark_runner.evaluate_fine_tuning(
            base_model=base_model, fine_tuned_model=fine_tuned_model
        )

        # Check for regressions
        if comparison.regressions:
            self.trigger_hook(HookType.ON_REGRESSION, comparison=comparison)

        # Check for improvements
        avg_improvement = self._calculate_avg_improvement(comparison)
        if avg_improvement > self.config.deployment_threshold:
            self.trigger_hook(
                HookType.ON_IMPROVEMENT, comparison=comparison, improvement=avg_improvement
            )

        return comparison

    def _auto_report_hook(self, **kwargs) -> str:
        """Automatic report generation hook.

        Args:
            **kwargs: Hook arguments including comparison results

        Returns:
            Report file path
        """
        logger.info("Generating automatic report")

        comparison = kwargs.get("comparison")
        if not comparison:
            # Try to get from hook history
            for result in reversed(self.hook_history):
                if result.hook_type == HookType.POST_TRAINING and result.result_data:
                    comparison = result.result_data
                    break

        if not comparison:
            raise ValueError("No comparison results available for report")

        # Initialize report generator if needed
        if not self.report_generator:
            self.report_generator = ReportGenerator(config=self.config.report_config)

        # Run cost analysis if available
        cost_analysis = None
        if self.cost_analyzer:
            cost_analysis = self.cost_analyzer.analyze_fine_tuning(
                comparison_result=comparison, training_hours=kwargs.get("training_hours", 1.0)
            )

        # Create report content
        content = ReportContent(
            comparison_result=comparison,
            cost_analysis=cost_analysis,
            metadata=kwargs.get("metadata", {}),
        )

        # Generate report
        report = self.report_generator.generate_report(content, format=ReportFormat.HTML)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"evaluation_report_{timestamp}.html"
        self.report_generator.save_report(report, report_path)

        logger.info(f"Report saved to {report_path}")

        return report_path

    def _notification_hook(self, **kwargs):
        """Send completion notification.

        Args:
            **kwargs: Hook arguments
        """
        message = self._format_notification_message("Fine-Tuning Evaluation Complete", kwargs)

        self._send_notifications(message, NotificationType.CONSOLE)

    def _error_notification_hook(self, **kwargs):
        """Send error notification.

        Args:
            **kwargs: Hook arguments including error information
        """
        error = kwargs.get("error", "Unknown error")
        message = f"Error in fine-tuning pipeline: {error}"

        self._send_notifications(message, NotificationType.CONSOLE, priority="high")

    def _regression_notification_hook(self, **kwargs):
        """Send regression notification.

        Args:
            **kwargs: Hook arguments including regression information
        """
        comparison = kwargs.get("comparison")
        if comparison and comparison.regressions:
            message = f"⚠️ {len(comparison.regressions)} regressions detected:\n"
            for reg in comparison.regressions[:5]:  # Show first 5
                message += f"  - {reg['benchmark']}: {reg['regression_pct']:.1f}%\n"

            self._send_notifications(message, NotificationType.CONSOLE, priority="high")

    def _send_notifications(
        self,
        message: str,
        default_type: NotificationType = NotificationType.CONSOLE,
        priority: str = "normal",
    ):
        """Send notifications through configured channels.

        Args:
            message: Notification message
            default_type: Default notification type
            priority: Message priority
        """
        channels = self.config.notification_channels or [default_type]

        for channel in channels:
            try:
                if channel == NotificationType.CONSOLE:
                    print(f"\n{'=' * 50}")
                    print(f"[{priority.upper()}] {message}")
                    print(f"{'=' * 50}\n")

                elif channel == NotificationType.EMAIL and EMAIL_AVAILABLE:
                    self._send_email(message, priority)

                elif channel == NotificationType.WEBHOOK and WEBHOOK_AVAILABLE:
                    self._send_webhook(message, priority)

                elif channel == NotificationType.SLACK and self.config.slack_webhook:
                    self._send_slack(message, priority)

                elif channel == NotificationType.TEAMS and self.config.teams_webhook:
                    self._send_teams(message, priority)

                elif channel == NotificationType.FILE:
                    self._log_to_file(message, priority)

            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification: {e}")

    def _send_email(self, message: str, priority: str):
        """Send email notification.

        Args:
            message: Email message
            priority: Message priority
        """
        if not self.config.email_config:
            return

        smtp_server = self.config.email_config.get("smtp_server")
        smtp_port = self.config.email_config.get("smtp_port", 587)
        sender = self.config.email_config.get("sender")
        password = self.config.email_config.get("password")
        recipients = self.config.email_config.get("recipients", [])

        if not all([smtp_server, sender, password, recipients]):
            logger.warning("Email configuration incomplete")
            return

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = f"[{priority.upper()}] Fine-Tuning Evaluation Notification"

        msg.attach(MIMEText(message, "plain"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)

    def _send_webhook(self, message: str, priority: str):
        """Send webhook notification.

        Args:
            message: Webhook message
            priority: Message priority
        """
        if not self.config.webhook_url:
            return

        payload = {
            "message": message,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "source": "fine_tuning_evaluation",
        }

        response = requests.post(self.config.webhook_url, json=payload, timeout=10)
        response.raise_for_status()

    def _send_slack(self, message: str, priority: str):
        """Send Slack notification.

        Args:
            message: Slack message
            priority: Message priority
        """
        if not self.config.slack_webhook or not WEBHOOK_AVAILABLE:
            return

        emoji = {
            "high": ":warning:",
            "normal": ":information_source:",
            "low": ":white_check_mark:",
        }.get(priority, ":information_source:")

        payload = {"text": f"{emoji} {message}"}

        response = requests.post(self.config.slack_webhook, json=payload, timeout=10)
        response.raise_for_status()

    def _send_teams(self, message: str, priority: str):
        """Send Microsoft Teams notification.

        Args:
            message: Teams message
            priority: Message priority
        """
        if not self.config.teams_webhook or not WEBHOOK_AVAILABLE:
            return

        color = {"high": "FF0000", "normal": "0078D4", "low": "00FF00"}.get(priority, "0078D4")

        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": "Fine-Tuning Evaluation Notification",
            "sections": [{"activityTitle": f"Priority: {priority.upper()}", "text": message}],
        }

        response = requests.post(self.config.teams_webhook, json=payload, timeout=10)
        response.raise_for_status()

    def _log_to_file(self, message: str, priority: str):
        """Log notification to file.

        Args:
            message: Log message
            priority: Message priority
        """
        log_file = Path("evaluation_notifications.log")

        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] [{priority.upper()}] {message}\n")

    def _calculate_avg_improvement(self, comparison: ComparisonResult) -> float:
        """Calculate average improvement from comparison.

        Args:
            comparison: Comparison results

        Returns:
            Average improvement percentage
        """
        improvements = []
        for benchmark, imp in comparison.improvements.items():
            if isinstance(imp, dict) and "improvement_pct" in imp:
                improvements.append(imp["improvement_pct"])

        return sum(improvements) / len(improvements) if improvements else 0

    def _format_notification_message(self, title: str, data: Dict[str, Any]) -> str:
        """Format notification message.

        Args:
            title: Message title
            data: Message data

        Returns:
            Formatted message
        """
        message = f"{title}\n\n"

        # Add key metrics if available
        if "comparison" in data:
            comparison = data["comparison"]
            avg_improvement = self._calculate_avg_improvement(comparison)
            message += f"Average Improvement: {avg_improvement:.1f}%\n"
            message += f"Regressions: {len(comparison.regressions)}\n"

        # Add timing information
        if "duration" in data:
            message += f"Duration: {data['duration']:.1f} seconds\n"

        # Add report path if available
        if "report_path" in data:
            message += f"Report: {data['report_path']}\n"

        return message

    def register_model(self, model_path: str, metadata: Dict[str, Any]):
        """Register model in model registry.

        Args:
            model_path: Path to model
            metadata: Model metadata
        """
        if not self.config.model_registry_enabled:
            return

        registry_path = Path(self.config.model_registry_path)
        registry_path.mkdir(parents=True, exist_ok=True)

        # Create registry entry
        model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        entry = {
            "model_id": model_id,
            "model_path": str(model_path),
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata,
        }

        # Save to registry
        registry_file = registry_path / f"{model_id}.json"
        with open(registry_file, "w") as f:
            json.dump(entry, f, indent=2)

        logger.info(f"Model registered: {model_id}")

        return model_id

    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)

        if self.benchmark_runner:
            self.benchmark_runner.cleanup()

        self.logger.close()


class PipelineIntegration:
    """Integration with fine-tuning pipeline."""

    def __init__(self, hooks: FineTuningHooks):
        """Initialize pipeline integration.

        Args:
            hooks: Hook system
        """
        self.hooks = hooks
        self.attached_trainers = []

    def attach_to_trainer(self, trainer):
        """Attach hooks to a trainer.

        Args:
            trainer: Trainer instance
        """
        if not FINE_TUNING_AVAILABLE:
            logger.warning("Fine-tuning module not available")
            return

        if not isinstance(trainer, BaseTrainer):
            logger.warning(f"Trainer must be instance of BaseTrainer, got {type(trainer)}")
            return

        # Monkey-patch trainer methods
        original_train = trainer.train
        original_evaluate = trainer.evaluate if hasattr(trainer, "evaluate") else None

        def wrapped_train(*args, **kwargs):
            """Wrapped training method."""
            # Pre-training hook
            self.hooks.trigger_hook(
                HookType.PRE_TRAINING, trainer=trainer, args=args, kwargs=kwargs
            )

            try:
                # Run original training
                result = original_train(*args, **kwargs)

                # Post-training hook
                self.hooks.trigger_hook(
                    HookType.POST_TRAINING,
                    trainer=trainer,
                    result=result,
                    base_model=kwargs.get("base_model"),
                    fine_tuned_model=trainer.output_dir,
                )

                # Completion hook
                self.hooks.trigger_hook(HookType.ON_COMPLETION, trainer=trainer, result=result)

                return result

            except Exception as e:
                # Error hook
                self.hooks.trigger_hook(HookType.ON_ERROR, trainer=trainer, error=e)
                raise

        def wrapped_evaluate(*args, **kwargs):
            """Wrapped evaluation method."""
            # Pre-evaluation hook
            self.hooks.trigger_hook(
                HookType.PRE_EVALUATION, trainer=trainer, args=args, kwargs=kwargs
            )

            # Run original evaluation
            result = original_evaluate(*args, **kwargs) if original_evaluate else None

            # Post-evaluation hook
            self.hooks.trigger_hook(HookType.POST_EVALUATION, trainer=trainer, result=result)

            return result

        # Replace methods
        trainer.train = wrapped_train
        if original_evaluate:
            trainer.evaluate = wrapped_evaluate

        # Track attached trainer
        self.attached_trainers.append(trainer)

        logger.info(f"Hooks attached to trainer: {type(trainer).__name__}")

    def detach_from_trainer(self, trainer):
        """Detach hooks from a trainer.

        Args:
            trainer: Trainer instance
        """
        if trainer in self.attached_trainers:
            self.attached_trainers.remove(trainer)
            logger.info(f"Hooks detached from trainer: {type(trainer).__name__}")


# Example usage
if __name__ == "__main__":
    # Create hook configuration
    config = HookConfig(
        auto_evaluate=True,
        auto_generate_report=True,
        notification_on_completion=True,
        notification_channels=[NotificationType.CONSOLE],
    )

    # Create hooks
    hooks = FineTuningHooks(config)

    # Add custom hook
    def custom_hook(**kwargs):
        print("Custom hook executed!")
        return "Custom result"

    hooks.register_hook(HookType.POST_TRAINING, custom_hook)

    # Simulate pipeline execution
    print("Simulating fine-tuning pipeline...")

    # Trigger pre-training
    hooks.trigger_hook(HookType.PRE_TRAINING, model="gpt2")

    # Simulate training completion
    hooks.trigger_hook(
        HookType.POST_TRAINING, base_model="gpt2", fine_tuned_model="./fine_tuned/gpt2-custom"
    )

    # Cleanup
    hooks.cleanup()

    print("Pipeline simulation complete!")
