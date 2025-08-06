"""
Dashboard Configuration Management

Handles all configuration settings for the monitoring dashboard including
database connections, API settings, security configurations, and feature flags.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    url: str = "sqlite:///monitoring_dashboard.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class APIConfig:
    """API configuration settings."""

    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False
    reload: bool = False
    workers: int = 1
    timeout: int = 30
    max_request_size: int = 16 * 1024 * 1024  # 16MB


@dataclass
class SecurityConfig:
    """Security and authentication settings."""

    secret_key: str = field(default_factory=lambda: os.urandom(32).hex())
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    jwt_expiration: int = 86400  # 24 hours
    bcrypt_rounds: int = 12
    require_https: bool = False
    csrf_protection: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and alerting settings."""

    refresh_interval: int = 30  # seconds
    data_retention_days: int = 90
    alert_check_interval: int = 300  # 5 minutes
    max_chart_points: int = 1000
    websocket_timeout: int = 60
    background_sync_interval: int = 60


@dataclass
class ReportConfig:
    """Report generation settings."""

    output_dir: str = "reports"
    temp_dir: str = "temp"
    max_report_size: int = 50 * 1024 * 1024  # 50MB
    chart_width: int = 800
    chart_height: int = 600
    pdf_page_size: str = "A4"
    email_enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""


@dataclass
class DashboardConfig:
    """Main dashboard configuration."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)

    # Feature flags
    enable_auth: bool = True
    enable_reports: bool = True
    enable_alerts: bool = True
    enable_export: bool = True
    enable_websockets: bool = True

    # Environment
    environment: str = "development"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "DashboardConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Database settings
        if db_url := os.getenv("DATABASE_URL"):
            config.database.url = db_url
        config.database.echo = os.getenv("DB_ECHO", "false").lower() == "true"

        # API settings
        config.api.host = os.getenv("DASHBOARD_HOST", config.api.host)
        config.api.port = int(os.getenv("DASHBOARD_PORT", str(config.api.port)))
        config.api.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Security settings
        if secret_key := os.getenv("SECRET_KEY"):
            config.security.secret_key = secret_key
        config.security.require_https = os.getenv("REQUIRE_HTTPS", "false").lower() == "true"

        # Monitoring settings
        config.monitoring.refresh_interval = int(
            os.getenv("REFRESH_INTERVAL", str(config.monitoring.refresh_interval))
        )
        config.monitoring.data_retention_days = int(
            os.getenv("DATA_RETENTION_DAYS", str(config.monitoring.data_retention_days))
        )

        # Report settings
        config.reports.output_dir = os.getenv("REPORTS_DIR", config.reports.output_dir)
        config.reports.email_enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
        if smtp_server := os.getenv("SMTP_SERVER"):
            config.reports.smtp_server = smtp_server
        if smtp_username := os.getenv("SMTP_USERNAME"):
            config.reports.smtp_username = smtp_username
        if smtp_password := os.getenv("SMTP_PASSWORD"):
            config.reports.smtp_password = smtp_password

        # Feature flags
        config.enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
        config.enable_reports = os.getenv("ENABLE_REPORTS", "true").lower() == "true"
        config.enable_alerts = os.getenv("ENABLE_ALERTS", "true").lower() == "true"
        config.enable_export = os.getenv("ENABLE_EXPORT", "true").lower() == "true"
        config.enable_websockets = os.getenv("ENABLE_WEBSOCKETS", "true").lower() == "true"

        # Environment
        config.environment = os.getenv("ENVIRONMENT", config.environment)
        config.log_level = os.getenv("LOG_LEVEL", config.log_level)

        return config

    @classmethod
    def from_file(cls, config_path: str) -> "DashboardConfig":
        """Load configuration from JSON file."""
        import json

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file) as f:
            config_data = json.load(f)

        config = cls()

        # Update configuration from file
        if "database" in config_data:
            for key, value in config_data["database"].items():
                if hasattr(config.database, key):
                    setattr(config.database, key, value)

        if "api" in config_data:
            for key, value in config_data["api"].items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)

        if "security" in config_data:
            for key, value in config_data["security"].items():
                if hasattr(config.security, key):
                    setattr(config.security, key, value)

        if "monitoring" in config_data:
            for key, value in config_data["monitoring"].items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)

        if "reports" in config_data:
            for key, value in config_data["reports"].items():
                if hasattr(config.reports, key):
                    setattr(config.reports, key, value)

        # Feature flags and environment
        for key in [
            "enable_auth",
            "enable_reports",
            "enable_alerts",
            "enable_export",
            "enable_websockets",
            "environment",
            "log_level",
        ]:
            if key in config_data:
                setattr(config, key, config_data[key])

        return config

    def to_dict(self) -> Dict[str | Any]:
        """Convert configuration to dictionary."""
        return {
            "database": self.database.__dict__,
            "api": self.api.__dict__,
            "security": {
                k: v for k, v in self.security.__dict__.items() if k != "secret_key"
            },  # Exclude secret
            "monitoring": self.monitoring.__dict__,
            "reports": {
                k: v for k, v in self.reports.__dict__.items() if "password" not in k.lower()
            },  # Exclude passwords
            "enable_auth": self.enable_auth,
            "enable_reports": self.enable_reports,
            "enable_alerts": self.enable_alerts,
            "enable_export": self.enable_export,
            "enable_websockets": self.enable_websockets,
            "environment": self.environment,
            "log_level": self.log_level,
        }

    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []

        # Validate database URL
        if not self.database.url:
            errors.append("Database URL is required")

        # Validate API settings
        if not (1 <= self.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")

        # Validate monitoring settings
        if self.monitoring.refresh_interval < 1:
            errors.append("Refresh interval must be at least 1 second")

        if self.monitoring.data_retention_days < 1:
            errors.append("Data retention must be at least 1 day")

        # Validate report settings
        if self.reports.email_enabled:
            if not self.reports.smtp_server:
                errors.append("SMTP server is required when email is enabled")
            if not self.reports.smtp_username:
                errors.append("SMTP username is required when email is enabled")

        if errors:
            raise ValueError("Configuration validation failed: " + "; ".join(errors))

        return True

    def create_directories(self):
        """Create necessary directories."""
        directories = [self.reports.output_dir, self.reports.temp_dir, "logs", "data"]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = DashboardConfig.from_env()

# Configuration validation on import
try:
    config.validate()
    config.create_directories()
except Exception as e:
    print(f"Warning: Configuration validation failed: {e}")
    print("Using default configuration values")


def get_config() -> DashboardConfig:
    """Get the global configuration instance."""
    return config


def update_config(new_config: DashboardConfig):
    """Update the global configuration instance."""
    global config
    config = new_config
    config.validate()
    config.create_directories()
