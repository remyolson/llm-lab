#!/usr/bin/env python3
"""
Dashboard Startup Script

Simple script to run the LLM Monitoring Dashboard with proper configuration
and error handling.
"""

import sys
import argparse
import logging
from pathlib import Path

# Use relative imports - sys.path manipulation removed

from src.use_cases.monitoring.dashboard import create_app
from src.use_cases.monitoring.dashboard.config import get_config, update_config, DashboardConfig


def setup_logging(level='INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/dashboard.log')
        ]
    )


def main():
    """Main entry point for the dashboard."""
    parser = argparse.ArgumentParser(description='LLM Monitoring Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8050, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = get_config()
        
        # Override with command line arguments
        config.api.host = args.host
        config.api.port = args.port
        config.api.debug = args.debug
        config.log_level = args.log_level
        
        # Load from config file if provided
        if args.config:
            config = DashboardConfig.from_file(args.config)
            # Still override with command line args
            config.api.host = args.host
            config.api.port = args.port
            if args.debug:
                config.api.debug = True
        
        # Update global config
        update_config(config)
        
        # Validate configuration
        config.validate()
        
        logger.info(f"Starting dashboard on {config.api.host}:{config.api.port}")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Debug mode: {config.api.debug}")
        logger.info(f"Features enabled: Auth={config.enable_auth}, "
                   f"Reports={config.enable_reports}, Alerts={config.enable_alerts}, "
                   f"WebSockets={config.enable_websockets}")
        
        # Create Flask app
        app = create_app()
        
        # Run the application
        if config.enable_websockets and hasattr(app, 'socketio'):
            logger.info("Starting with WebSocket support")
            app.socketio.run(
                app,
                host=config.api.host,
                port=config.api.port,
                debug=config.api.debug,
                use_reloader=False  # Disable reloader to avoid duplicate runs
            )
        else:
            logger.info("Starting without WebSocket support")
            app.run(
                host=config.api.host,
                port=config.api.port,
                debug=config.api.debug,
                use_reloader=False
            )
            
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()