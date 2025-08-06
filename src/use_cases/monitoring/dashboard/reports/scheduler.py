"""
Report Scheduler

Provides cron-like scheduling for automated report generation with
configurable intervals, email delivery, and failure handling.
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import schedule

from .delivery import EmailDelivery
from .generator import ReportGenerator


class ScheduleFrequency(Enum):
    """Report generation frequencies."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class ScheduledReport:
    """Configuration for a scheduled report."""

    id: str
    name: str
    template: str
    frequency: ScheduleFrequency
    time: str  # Format: "HH:MM" for daily/weekly, "0 */4 * * *" for cron
    output_format: str = "pdf"
    email_recipients: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    failure_count: int = 0
    max_failures: int = 3
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class ReportScheduler:
    """Automated report scheduling system."""

    def __init__(
        self,
        report_generator: ReportGenerator,
        email_delivery: Optional[EmailDelivery] = None,
        config_file: Optional[str] = None,
    ):
        self.report_generator = report_generator
        self.email_delivery = email_delivery
        self.config_file = Path(config_file) if config_file else Path("scheduled_reports.json")

        self.scheduled_reports: Dict[str, ScheduledReport] = {}
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False

        self.logger = logging.getLogger(__name__)

        # Load existing scheduled reports
        self._load_scheduled_reports()

        # Initialize scheduler
        self._setup_scheduler()

    def add_scheduled_report(self, report_config: Dict[str, Any]) -> str:
        """Add a new scheduled report."""
        try:
            # Generate unique ID
            report_id = f"report_{int(time.time())}_{hash(report_config.get('name', ''))}"

            # Create scheduled report
            scheduled_report = ScheduledReport(
                id=report_id,
                name=report_config["name"],
                template=report_config["template"],
                frequency=ScheduleFrequency(report_config["frequency"]),
                time=report_config["time"],
                output_format=report_config.get("output_format", "pdf"),
                email_recipients=report_config.get("email_recipients"),
                filters=report_config.get("filters"),
                enabled=report_config.get("enabled", True),
            )

            # Calculate next run time
            scheduled_report.next_run = self._calculate_next_run(scheduled_report)

            # Store the scheduled report
            self.scheduled_reports[report_id] = scheduled_report

            # Save to file
            self._save_scheduled_reports()

            # Update scheduler
            self._schedule_report(scheduled_report)

            self.logger.info(f"Added scheduled report: {report_config['name']}")
            return report_id

        except Exception as e:
            self.logger.error(f"Failed to add scheduled report: {e}")
            raise

    def remove_scheduled_report(self, report_id: str) -> bool:
        """Remove a scheduled report."""
        if report_id not in self.scheduled_reports:
            return False

        try:
            # Remove from scheduler
            schedule.clear(f"report_{report_id}")

            # Remove from memory
            del self.scheduled_reports[report_id]

            # Save changes
            self._save_scheduled_reports()

            self.logger.info(f"Removed scheduled report: {report_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to remove scheduled report {report_id}: {e}")
            return False

    def update_scheduled_report(self, report_id: str, updates: Dict[str, Any]) -> bool:
        """Update a scheduled report configuration."""
        if report_id not in self.scheduled_reports:
            return False

        try:
            scheduled_report = self.scheduled_reports[report_id]

            # Update fields
            for key, value in updates.items():
                if hasattr(scheduled_report, key):
                    if key == "frequency":
                        setattr(scheduled_report, key, ScheduleFrequency(value))
                    else:
                        setattr(scheduled_report, key, value)

            # Recalculate next run if time or frequency changed
            if "time" in updates or "frequency" in updates:
                scheduled_report.next_run = self._calculate_next_run(scheduled_report)
                # Reschedule
                schedule.clear(f"report_{report_id}")
                self._schedule_report(scheduled_report)

            # Save changes
            self._save_scheduled_reports()

            self.logger.info(f"Updated scheduled report: {report_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update scheduled report {report_id}: {e}")
            return False

    def get_scheduled_reports(self) -> List[Dict[str, Any]]:
        """Get all scheduled reports."""
        return [asdict(report) for report in self.scheduled_reports.values()]

    def get_scheduled_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific scheduled report."""
        if report_id in self.scheduled_reports:
            return asdict(self.scheduled_reports[report_id])
        return None

    def start_scheduler(self):
        """Start the report scheduler."""
        if self.running:
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        self.logger.info("Report scheduler started")

    def stop_scheduler(self):
        """Stop the report scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        self.logger.info("Report scheduler stopped")

    def run_report_now(self, report_id: str) -> Dict[str, Any]:
        """Run a scheduled report immediately."""
        if report_id not in self.scheduled_reports:
            return {"status": "error", "message": "Report not found"}

        scheduled_report = self.scheduled_reports[report_id]
        return self._execute_report(scheduled_report)

    def _load_scheduled_reports(self):
        """Load scheduled reports from config file."""
        if not self.config_file.exists():
            return

        try:
            with open(self.config_file) as f:
                data = json.load(f)

            for report_data in data.get("scheduled_reports", []):
                # Convert frequency back to enum
                report_data["frequency"] = ScheduleFrequency(report_data["frequency"])
                scheduled_report = ScheduledReport(**report_data)
                self.scheduled_reports[scheduled_report.id] = scheduled_report

            self.logger.info(f"Loaded {len(self.scheduled_reports)} scheduled reports")

        except Exception as e:
            self.logger.error(f"Failed to load scheduled reports: {e}")

    def _save_scheduled_reports(self):
        """Save scheduled reports to config file."""
        try:
            # Convert to serializable format
            data = {"scheduled_reports": []}

            for report in self.scheduled_reports.values():
                report_dict = asdict(report)
                report_dict["frequency"] = report.frequency.value
                data["scheduled_reports"].append(report_dict)

            # Save to file
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save scheduled reports: {e}")

    def _setup_scheduler(self):
        """Set up all scheduled reports."""
        for scheduled_report in self.scheduled_reports.values():
            if scheduled_report.enabled:
                self._schedule_report(scheduled_report)

    def _schedule_report(self, scheduled_report: ScheduledReport):
        """Schedule a single report."""
        try:
            job_tag = f"report_{scheduled_report.id}"

            if scheduled_report.frequency == ScheduleFrequency.HOURLY:
                schedule.every().hour.at(f":{scheduled_report.time.split(':')[1]}").do(
                    self._execute_report, scheduled_report
                ).tag(job_tag)

            elif scheduled_report.frequency == ScheduleFrequency.DAILY:
                schedule.every().day.at(scheduled_report.time).do(
                    self._execute_report, scheduled_report
                ).tag(job_tag)

            elif scheduled_report.frequency == ScheduleFrequency.WEEKLY:
                # Assume format "Monday 09:00"
                parts = scheduled_report.time.split()
                if len(parts) == 2:
                    day, time = parts
                    getattr(schedule.every(), day.lower()).at(time).do(
                        self._execute_report, scheduled_report
                    ).tag(job_tag)

            elif scheduled_report.frequency == ScheduleFrequency.MONTHLY:
                # For monthly, we'll use a custom check in the scheduler loop
                pass

            self.logger.info(f"Scheduled report: {scheduled_report.name}")

        except Exception as e:
            self.logger.error(f"Failed to schedule report {scheduled_report.id}: {e}")

    def _run_scheduler(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Run pending scheduled jobs
                schedule.run_pending()

                # Check for monthly jobs
                self._check_monthly_jobs()

                # Sleep for a minute
                time.sleep(60)

            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(60)

    def _check_monthly_jobs(self):
        """Check and run monthly jobs."""
        now = datetime.utcnow()

        for scheduled_report in self.scheduled_reports.values():
            if scheduled_report.frequency == ScheduleFrequency.MONTHLY and scheduled_report.enabled:
                # Check if it's time to run
                if self._should_run_monthly(scheduled_report, now):
                    self._execute_report(scheduled_report)

    def _should_run_monthly(self, scheduled_report: ScheduledReport, now: datetime) -> bool:
        """Check if a monthly report should run."""
        try:
            # Parse time (format: "1 09:00" for 1st day of month at 09:00)
            parts = scheduled_report.time.split()
            if len(parts) != 2:
                return False

            day, time = parts
            day = int(day)
            hour, minute = map(int, time.split(":"))

            # Check if it's the right day and time
            if now.day == day and now.hour == hour and now.minute == minute:
                # Check if we haven't run this month
                if scheduled_report.last_run:
                    last_run = datetime.fromisoformat(scheduled_report.last_run)
                    if last_run.year == now.year and last_run.month == now.month:
                        return False

                return True

        except (ValueError, AttributeError):
            pass

        return False

    def _execute_report(self, scheduled_report: ScheduledReport) -> Dict[str, Any]:
        """Execute a scheduled report."""
        try:
            self.logger.info(f"Executing scheduled report: {scheduled_report.name}")

            # Calculate date range based on frequency
            date_range = self._get_date_range_for_frequency(scheduled_report.frequency)

            # Generate report
            result = self.report_generator.generate_report(
                template=scheduled_report.template,
                output_format=scheduled_report.output_format,
                date_range=date_range,
                filters=scheduled_report.filters,
            )

            if result.get("status") == "completed":
                # Send email if configured
                if (
                    scheduled_report.email_recipients
                    and self.email_delivery
                    and "file_path" in result
                ):
                    email_result = self.email_delivery.send_report_email(
                        recipients=scheduled_report.email_recipients,
                        subject=f"Scheduled Report: {scheduled_report.name}",
                        report_path=result["file_path"],
                        template_name=scheduled_report.template,
                    )

                    result["email_sent"] = email_result.get("status") == "sent"

                # Update last run
                scheduled_report.last_run = datetime.utcnow().isoformat()
                scheduled_report.next_run = self._calculate_next_run(scheduled_report)
                scheduled_report.failure_count = 0  # Reset failure count

                self.logger.info(f"Successfully executed report: {scheduled_report.name}")

            else:
                # Handle failure
                scheduled_report.failure_count += 1
                self.logger.error(
                    f"Report execution failed: {result.get('error', 'Unknown error')}"
                )

                # Disable if too many failures
                if scheduled_report.failure_count >= scheduled_report.max_failures:
                    scheduled_report.enabled = False
                    self.logger.warning(
                        f"Disabled report after {scheduled_report.max_failures} failures: {scheduled_report.name}"
                    )

            # Save changes
            self._save_scheduled_reports()

            return result

        except Exception as e:
            self.logger.error(f"Failed to execute scheduled report {scheduled_report.id}: {e}")

            # Update failure count
            scheduled_report.failure_count += 1
            if scheduled_report.failure_count >= scheduled_report.max_failures:
                scheduled_report.enabled = False

            self._save_scheduled_reports()

            return {"status": "failed", "error": str(e)}

    def _get_date_range_for_frequency(self, frequency: ScheduleFrequency) -> tuple:
        """Get appropriate date range for report frequency."""
        end_date = datetime.utcnow()

        if frequency == ScheduleFrequency.HOURLY:
            start_date = end_date - timedelta(hours=1)
        elif frequency == ScheduleFrequency.DAILY:
            start_date = end_date - timedelta(days=1)
        elif frequency == ScheduleFrequency.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        elif frequency == ScheduleFrequency.MONTHLY:
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=1)  # Default to daily

        return (start_date, end_date)

    def _calculate_next_run(self, scheduled_report: ScheduledReport) -> str:
        """Calculate next run time for a scheduled report."""
        try:
            now = datetime.utcnow()

            if scheduled_report.frequency == ScheduleFrequency.HOURLY:
                # Next hour at specified minute
                minute = int(scheduled_report.time.split(":")[1])
                next_run = now.replace(minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(hours=1)

            elif scheduled_report.frequency == ScheduleFrequency.DAILY:
                # Next day at specified time
                hour, minute = map(int, scheduled_report.time.split(":"))
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)

            elif scheduled_report.frequency == ScheduleFrequency.WEEKLY:
                # Next week at specified day and time
                parts = scheduled_report.time.split()
                if len(parts) == 2:
                    day_name, time_str = parts
                    hour, minute = map(int, time_str.split(":"))

                    days_ahead = [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                    ].index(day_name.lower()) - now.weekday()

                    if days_ahead <= 0:  # Target day already happened this week
                        days_ahead += 7

                    next_run = now + timedelta(days=days_ahead)
                    next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    next_run = now + timedelta(days=7)

            elif scheduled_report.frequency == ScheduleFrequency.MONTHLY:
                # Next month at specified day and time
                parts = scheduled_report.time.split()
                if len(parts) == 2:
                    day, time_str = parts
                    day = int(day)
                    hour, minute = map(int, time_str.split(":"))

                    # Start with next month
                    if now.month == 12:
                        next_run = now.replace(
                            year=now.year + 1,
                            month=1,
                            day=day,
                            hour=hour,
                            minute=minute,
                            second=0,
                            microsecond=0,
                        )
                    else:
                        next_run = now.replace(
                            month=now.month + 1,
                            day=day,
                            hour=hour,
                            minute=minute,
                            second=0,
                            microsecond=0,
                        )
                else:
                    next_run = now + timedelta(days=30)

            else:
                next_run = now + timedelta(days=1)

            return next_run.isoformat()

        except Exception as e:
            self.logger.error(f"Failed to calculate next run time: {e}")
            return (datetime.utcnow() + timedelta(days=1)).isoformat()


def create_report_scheduler(
    report_generator: ReportGenerator,
    email_delivery: Optional[EmailDelivery] = None,
    config_file: Optional[str] = None,
) -> ReportScheduler:
    """Factory function to create a report scheduler."""
    return ReportScheduler(report_generator, email_delivery, config_file)
