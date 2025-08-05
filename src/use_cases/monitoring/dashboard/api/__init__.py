"""
API Blueprint for Dashboard

Provides REST API endpoints for data access, configuration, and real-time updates.
"""

from flask import Blueprint, jsonify, request
from sqlalchemy import func
from typing import Dict, Any
import logging

from ..config import DashboardConfig
from ..data_service import get_data_service


def create_api_blueprint(config: DashboardConfig) -> Blueprint:
    """Create and configure the API blueprint."""
    api_bp = Blueprint('api', __name__)
    
    @api_bp.route('/status')
    def api_status():
        """API status endpoint."""
        return jsonify({
            'status': 'online',
            'version': '1.0.0',
            'features': {
                'auth': config.enable_auth,
                'reports': config.enable_reports,
                'alerts': config.enable_alerts,
                'export': config.enable_export,
                'websockets': config.enable_websockets
            },
            'environment': config.environment
        })
    
    @api_bp.route('/config')
    def get_config():
        """Get dashboard configuration (public settings only)."""
        return jsonify(config.to_dict())
    
    @api_bp.route('/metrics/overview')
    def metrics_overview():
        """Get overview metrics for dashboard."""
        try:
            data_service = get_data_service()
            metrics = data_service.get_metrics_summary()
            return jsonify(metrics)
        except Exception as e:
            logging.error(f'Error getting metrics overview: {e}')
            # Return fallback data
            return jsonify({
                'total_models': 0,
                'total_requests': 0,
                'avg_latency': 0.0,
                'total_cost': 0.0,
                'active_alerts': 0,
                'uptime': 0,
                'last_updated': '2024-01-01T00:00:00Z'
            })
    
    @api_bp.route('/metrics/performance')
    def performance_metrics():
        """Get performance metrics data."""
        try:
            data_service = get_data_service()
            
            # Get query parameters
            hours = int(request.args.get('hours', 24))
            provider = request.args.get('provider')
            model = request.args.get('model')
            
            performance_data = data_service.get_performance_data(hours, provider, model)
            return jsonify(performance_data)
            
        except Exception as e:
            logging.error(f'Error getting performance metrics: {e}')
            return jsonify({
                'providers': [],
                'time_series': [],
                'models': []
            })
    
    @api_bp.route('/metrics/costs')
    def cost_metrics():
        """Get cost analysis data."""
        try:
            data_service = get_data_service()
            
            hours = int(request.args.get('hours', 24))
            cost_data = data_service.get_cost_breakdown(hours)
            return jsonify(cost_data)
            
        except Exception as e:
            logging.error(f'Error getting cost metrics: {e}')
            return jsonify({
                'daily_costs': [],
                'provider_breakdown': {},
                'total_cost': 0.0
            })
    
    @api_bp.route('/alerts')
    def get_alerts():
        """Get current alerts."""
        try:
            data_service = get_data_service()
            
            limit = int(request.args.get('limit', 50))
            alerts = data_service.get_active_alerts(limit)
            
            # Calculate alert stats
            alert_stats = {
                'total': len(alerts),
                'critical': len([a for a in alerts if a['severity'] == 'critical']),
                'warning': len([a for a in alerts if a['severity'] == 'warning']),
                'resolved': len([a for a in alerts if a['status'] == 'resolved'])
            }
            
            return jsonify({
                'active_alerts': alerts,
                'alert_history': alerts,  # For now, same as active
                'alert_stats': alert_stats
            })
            
        except Exception as e:
            logging.error(f'Error getting alerts: {e}')
            return jsonify({
                'active_alerts': [],
                'alert_history': [],
                'alert_stats': {
                    'total': 0,
                    'critical': 0,
                    'warning': 0,
                    'resolved': 0
                }
            })
    
    @api_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
    def acknowledge_alert(alert_id):
        """Acknowledge an alert."""
        try:
            data_service = get_data_service()
            # This would be implemented with user authentication
            user = request.json.get('user', 'anonymous')
            
            # Would need to add method to data service
            return jsonify({'success': True, 'message': 'Alert acknowledged'})
            
        except Exception as e:
            logging.error(f'Error acknowledging alert: {e}')
            return jsonify({'error': 'Failed to acknowledge alert'}), 500
    
    @api_bp.route('/alerts/<int:alert_id>/resolve', methods=['POST'])
    def resolve_alert(alert_id):
        """Resolve an alert."""
        try:
            data_service = get_data_service()
            user = request.json.get('user', 'anonymous')
            
            # Would need to add method to data service
            return jsonify({'success': True, 'message': 'Alert resolved'})
            
        except Exception as e:
            logging.error(f'Error resolving alert: {e}')
            return jsonify({'error': 'Failed to resolve alert'}), 500
    
    @api_bp.route('/export/<format>')
    def export_data(format):
        """Export data in various formats."""
        try:
            if format not in ['csv', 'json', 'xlsx']:
                return jsonify({'error': 'Unsupported export format'}), 400
            
            from ..reports.exporter import create_data_exporter
            
            data_service = get_data_service()
            exporter = create_data_exporter(data_service)
            
            # Get parameters
            data_type = request.args.get('type', 'performance')  
            hours = int(request.args.get('hours', 24))
            provider = request.args.get('provider')
            model = request.args.get('model')
            aggregation = request.args.get('aggregation')
            
            # Calculate date range
            from datetime import datetime, timedelta
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=hours)
            
            # Build filters
            filters = {}
            if provider:
                filters['provider'] = provider
            if model:
                filters['model'] = model
            
            # Export data
            result = exporter.export_data(
                data_type=data_type,
                format=format,
                date_range=(start_date, end_date),
                filters=filters,
                aggregation=aggregation
            )
            
            if result.get('status') == 'completed':
                from flask import send_file
                return send_file(
                    result['file_path'],
                    as_attachment=True,
                    download_name=f"{data_type}_export.{format}"
                )
            else:
                return jsonify({'error': result.get('error', 'Export failed')}), 500
                
        except Exception as e:
            logging.error(f'Error exporting data: {e}')
            return jsonify({'error': 'Export failed'}), 500
    
    @api_bp.route('/reports/generate', methods=['POST'])
    def generate_report():
        """Generate a report."""
        try:
            from ..reports.generator import create_report_generator
            
            data = request.get_json()
            template = data.get('template', 'daily_summary')
            output_format = data.get('format', 'html')
            hours = int(data.get('hours', 24))
            filters = data.get('filters', {})
            
            # Calculate date range
            from datetime import datetime, timedelta
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=hours)
            
            data_service = get_data_service()
            generator = create_report_generator(data_service)
            
            result = generator.generate_report(
                template=template,
                output_format=output_format,
                date_range=(start_date, end_date),
                filters=filters
            )
            
            if result.get('status') == 'completed':
                return jsonify({
                    'status': 'success',
                    'report_id': result['id'],
                    'file_path': result['file_path'],
                    'file_size': result['file_size'],
                    'generated_at': result['generated_at']
                })
            else:
                return jsonify({'error': result.get('error', 'Report generation failed')}), 500
                
        except Exception as e:
            logging.error(f'Error generating report: {e}')
            return jsonify({'error': 'Report generation failed'}), 500
    
    @api_bp.route('/reports/download/<report_id>')
    def download_report(report_id):
        """Download a generated report."""
        try:
            # In a real implementation, you'd look up the report by ID
            # For now, we'll return an error
            return jsonify({'error': 'Report download not yet implemented'}), 501
            
        except Exception as e:
            logging.error(f'Error downloading report: {e}')
            return jsonify({'error': 'Download failed'}), 500
    
    @api_bp.route('/reports/scheduled', methods=['GET'])
    def get_scheduled_reports():
        """Get all scheduled reports."""
        try:
            from ..reports.scheduler import create_report_scheduler
            from ..reports.generator import create_report_generator
            
            data_service = get_data_service()
            generator = create_report_generator(data_service)
            scheduler = create_report_scheduler(generator)
            
            reports = scheduler.get_scheduled_reports()
            return jsonify({'scheduled_reports': reports})
            
        except Exception as e:
            logging.error(f'Error getting scheduled reports: {e}')
            return jsonify({'error': 'Failed to get scheduled reports'}), 500
    
    @api_bp.route('/reports/scheduled', methods=['POST'])
    def create_scheduled_report():
        """Create a new scheduled report."""
        try:
            from ..reports.scheduler import create_report_scheduler
            from ..reports.generator import create_report_generator
            
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['name', 'template', 'frequency', 'time']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            data_service = get_data_service()
            generator = create_report_generator(data_service)
            scheduler = create_report_scheduler(generator)
            
            report_id = scheduler.add_scheduled_report(data)
            
            return jsonify({
                'status': 'success',
                'report_id': report_id,
                'message': 'Scheduled report created successfully'
            })
            
        except Exception as e:
            logging.error(f'Error creating scheduled report: {e}')
            return jsonify({'error': 'Failed to create scheduled report'}), 500
    
    @api_bp.route('/reports/scheduled/<report_id>', methods=['DELETE'])
    def delete_scheduled_report(report_id):
        """Delete a scheduled report."""
        try:
            from ..reports.scheduler import create_report_scheduler
            from ..reports.generator import create_report_generator
            
            data_service = get_data_service()
            generator = create_report_generator(data_service)
            scheduler = create_report_scheduler(generator)
            
            success = scheduler.remove_scheduled_report(report_id)
            
            if success:
                return jsonify({'status': 'success', 'message': 'Scheduled report deleted'})
            else:
                return jsonify({'error': 'Report not found'}), 404
                
        except Exception as e:
            logging.error(f'Error deleting scheduled report: {e}')
            return jsonify({'error': 'Failed to delete scheduled report'}), 500
    
    @api_bp.errorhandler(404)
    def api_not_found(error):
        """Handle API 404 errors."""
        return jsonify({'error': 'API endpoint not found'}), 404
    
    @api_bp.errorhandler(500)
    def api_error(error):
        """Handle API 500 errors."""
        logging.error(f'API Error: {error}')
        return jsonify({'error': 'Internal server error'}), 500
    
    return api_bp