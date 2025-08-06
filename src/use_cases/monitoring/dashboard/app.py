"""
Main Flask Application for LLM Monitoring Dashboard

This module creates and configures the Flask application with all necessary
blueprints, extensions, and middleware for the monitoring dashboard.
"""

import logging
import os
from datetime import datetime
from typing import Optional

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.middleware.proxy_fix import ProxyFix

from .api import create_api_blueprint
from .components import create_components_blueprint
from .config import get_config
from .data_service import init_data_service
from .models import init_database


def create_app(config_override: Optional[dict] = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config_override: Optional configuration overrides

    Returns:
        Configured Flask application
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Load configuration
    config = get_config()

    # Apply configuration overrides
    if config_override:
        for key, value in config_override.items():
            setattr(config, key, value)

    # Configure Flask app
    app.config["SECRET_KEY"] = config.security.secret_key
    app.config["DEBUG"] = config.api.debug
    app.config["SQLALCHEMY_DATABASE_URI"] = config.database.url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_size": config.database.pool_size,
        "max_overflow": config.database.max_overflow,
        "pool_timeout": config.database.pool_timeout,
        "pool_recycle": config.database.pool_recycle,
        "echo": config.database.echo,
    }

    # Configure logging
    setup_logging(app, config)

    # Trust proxy headers if behind reverse proxy
    if config.security.require_https:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # Initialize database and data services
    initialize_data_services(app, config)

    # Initialize extensions
    initialize_extensions(app, config)

    # Register blueprints
    register_blueprints(app, config)

    # Register error handlers
    register_error_handlers(app)

    # Add global template variables
    setup_template_globals(app, config)

    # Health check endpoint
    @app.route("/health")
    def health_check():
        """Health check endpoint for load balancers."""
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "api_version": "v1",
            }
        )

    # Main dashboard route
    @app.route("/")
    def dashboard():
        """Main dashboard page."""
        return render_template(
            "dashboard.html", config=config.to_dict(), page_title="LLM Monitoring Dashboard"
        )

    # API documentation route
    @app.route("/api/docs")
    def api_docs():
        """API documentation page."""
        return render_template("api_docs.html", page_title="API Documentation")

    app.logger.info(f"Dashboard application created with config: {config.environment}")
    return app


def setup_logging(app: Flask, config) -> None:
    """Configure application logging."""
    if not app.debug and not app.testing:
        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # File handler
        file_handler = logging.FileHandler("logs/dashboard.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]")
        )
        file_handler.setLevel(getattr(logging, config.log_level))
        app.logger.addHandler(file_handler)

        # Set app logger level
        app.logger.setLevel(getattr(logging, config.log_level))
        app.logger.info("Dashboard startup")


def initialize_data_services(app: Flask, config) -> None:
    """Initialize database and data services."""
    try:
        # Initialize database
        db_manager = init_database(config.database.url, config.database.echo)
        app.logger.info("Database initialized successfully")

        # Initialize data service
        data_service = init_data_service(db_manager)

        # Start data collection and real-time updates
        data_service.start()
        app.logger.info("Data services started successfully")

        # Store references for cleanup
        app.db_manager = db_manager
        app.data_service = data_service

    except Exception as e:
        app.logger.error(f"Failed to initialize data services: {e}")
        # Continue without data services for now
        pass


def initialize_extensions(app: Flask, config) -> None:
    """Initialize Flask extensions."""
    # CORS for API access
    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/ws/*": {"origins": "*"}})

    # SocketIO for real-time updates (if enabled)
    if config.enable_websockets:
        socketio = SocketIO(
            app,
            cors_allowed_origins="*",
            async_mode="threading",
            ping_timeout=config.monitoring.websocket_timeout,
        )

        @socketio.on("connect")
        def handle_connect():
            app.logger.info(f"Client connected: {request.sid}")
            emit("status", {"message": "Connected to dashboard"})

        @socketio.on("disconnect")
        def handle_disconnect():
            app.logger.info(f"Client disconnected: {request.sid}")

        @socketio.on("subscribe_metrics")
        def handle_subscribe_metrics(data):
            app.logger.info(f"Client subscribed to metrics: {data}")
            # Join room for specific metric updates
            if "metric_type" in data:
                from flask_socketio import join_room

                join_room(f"metrics_{data['metric_type']}")
                emit(
                    "subscription_confirmed",
                    {"metric_type": data["metric_type"], "status": "subscribed"},
                )

        app.socketio = socketio


def register_blueprints(app: Flask, config) -> None:
    """Register application blueprints."""
    # API blueprint
    api_bp = create_api_blueprint(config)
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    # Components blueprint for UI components
    components_bp = create_components_blueprint(config)
    app.register_blueprint(components_bp, url_prefix="/components")


def register_error_handlers(app: Flask) -> None:
    """Register error handlers for the application."""

    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return render_template("error.html", error="Page not found", error_code=404), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Server Error: {error}")
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal server error"}), 500
        return render_template("error.html", error="Internal server error", error_code=500), 500

    @app.errorhandler(403)
    def forbidden_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Forbidden"}), 403
        return render_template("error.html", error="Access forbidden", error_code=403), 403

    @app.errorhandler(429)
    def ratelimit_handler(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Rate limit exceeded"}), 429
        return render_template("error.html", error="Rate limit exceeded", error_code=429), 429


def setup_template_globals(app: Flask, config) -> None:
    """Set up global template variables and functions."""

    @app.template_global()
    def dashboard_config():
        """Get dashboard configuration for templates."""
        return config.to_dict()

    @app.template_global()
    def current_time():
        """Get current timestamp for templates."""
        return datetime.utcnow().isoformat()

    @app.template_filter("datetime")
    def datetime_filter(timestamp):
        """Format datetime for display."""
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                return timestamp
        else:
            dt = timestamp
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @app.template_filter("duration")
    def duration_filter(seconds):
        """Format duration in seconds to human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    @app.template_filter("filesize")
    def filesize_filter(bytes_size):
        """Format file size in bytes to human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"


if __name__ == "__main__":
    """Run the dashboard application directly."""
    app = create_app()
    config = get_config()

    # Run with or without SocketIO
    if config.enable_websockets and hasattr(app, "socketio"):
        app.socketio.run(app, host=config.api.host, port=config.api.port, debug=config.api.debug)
    else:
        app.run(host=config.api.host, port=config.api.port, debug=config.api.debug)
