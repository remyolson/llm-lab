"""
Dashboard Components Blueprint

Provides reusable UI components for the monitoring dashboard including
charts, widgets, and interactive elements.
"""

from flask import Blueprint, render_template, jsonify, request
from typing import Dict, Any
import logging
import json

from ..config import DashboardConfig
from ..data_service import get_data_service


def create_components_blueprint(config: DashboardConfig) -> Blueprint:
    """Create and configure the components blueprint."""
    components_bp = Blueprint('components', __name__)
    
    @components_bp.route('/chart/<chart_type>')
    def render_chart(chart_type):
        """Render a chart component."""
        try:
            # Get chart configuration from query parameters
            chart_config = {
                'chart_type': chart_type,
                'title': request.args.get('title', ''),
                'height': request.args.get('height', '400'),
                'width': request.args.get('width', '100%'),
                'refresh_interval': int(request.args.get('refresh', 30)),
                'time_range': request.args.get('time_range', '24h'),
                'providers': request.args.getlist('providers'),
                'models': request.args.getlist('models')
            }
            
            return render_template(f'components/charts/{chart_type}.html', 
                                 config=chart_config,
                                 dashboard_config=config.to_dict())
        except Exception as e:
            logging.error(f'Error rendering chart {chart_type}: {e}')
            return render_template('components/error.html', 
                                 error=f'Chart {chart_type} not available'), 404
    
    @components_bp.route('/widget/<widget_type>')
    def render_widget(widget_type):
        """Render a widget component."""
        try:
            widget_config = {
                'widget_type': widget_type,
                'title': request.args.get('title', ''),
                'size': request.args.get('size', 'medium'),
                'refresh_interval': int(request.args.get('refresh', 30))
            }
            
            return render_template(f'components/widgets/{widget_type}.html',
                                 config=widget_config,
                                 dashboard_config=config.to_dict())
        except Exception as e:
            logging.error(f'Error rendering widget {widget_type}: {e}')
            return render_template('components/error.html',
                                 error=f'Widget {widget_type} not available'), 404
    
    @components_bp.route('/chart-data/<chart_type>')
    def get_chart_data(chart_type):
        """Get data for a specific chart type."""
        try:
            data_service = get_data_service()
            
            # Get parameters
            hours = int(request.args.get('hours', 24))
            provider = request.args.get('provider')
            model = request.args.get('model')
            
            if chart_type == 'performance-comparison':
                data = data_service.get_performance_data(hours, provider, model)
                return jsonify(_format_performance_chart_data(data))
            
            elif chart_type == 'cost-trends':
                data = data_service.get_cost_breakdown(hours)
                return jsonify(_format_cost_chart_data(data))
            
            elif chart_type == 'model-comparison':
                data = data_service.get_metrics_summary()
                return jsonify(_format_model_comparison_data(data))
            
            elif chart_type == 'alert-timeline':
                alerts = data_service.get_active_alerts(100)
                return jsonify(_format_alert_timeline_data(alerts))
            
            elif chart_type == 'latency-distribution':
                data = data_service.get_performance_data(hours, provider, model)
                return jsonify(_format_latency_distribution_data(data))
            
            elif chart_type == 'success-rate':
                data = data_service.get_performance_data(hours, provider, model)
                return jsonify(_format_success_rate_data(data))
            
            else:
                return jsonify({'error': 'Unknown chart type'}), 400
                
        except Exception as e:
            logging.error(f'Error getting chart data for {chart_type}: {e}')
            return jsonify({'error': 'Failed to load chart data'}), 500
    
    return components_bp


def _format_performance_chart_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format performance data for Chart.js."""
    return {
        'type': 'line',
        'data': {
            'labels': [item['timestamp'] for item in data.get('time_series', [])],
            'datasets': [
                {
                    'label': 'Average Latency (ms)',
                    'data': [item['avg_latency'] * 1000 for item in data.get('time_series', [])],
                    'borderColor': 'rgb(54, 162, 235)',
                    'backgroundColor': 'rgba(54, 162, 235, 0.1)',
                    'tension': 0.4,
                    'yAxisID': 'y'
                },
                {
                    'label': 'Success Rate (%)',
                    'data': [item['success_rate'] for item in data.get('time_series', [])],
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.1)',
                    'tension': 0.4,
                    'yAxisID': 'y1'
                }
            ]
        },
        'options': {
            'responsive': True,
            'interaction': {'intersect': False},
            'scales': {
                'x': {'display': True, 'title': {'display': True, 'text': 'Time'}},
                'y': {'type': 'linear', 'display': True, 'position': 'left', 'title': {'display': True, 'text': 'Latency (ms)'}},
                'y1': {'type': 'linear', 'display': True, 'position': 'right', 'title': {'display': True, 'text': 'Success Rate (%)'}, 'grid': {'drawOnChartArea': False}}
            },
            'plugins': {
                'legend': {'display': True},
                'tooltip': {'mode': 'index', 'intersect': False}
            }
        }
    }


def _format_cost_chart_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format cost data for Chart.js."""
    daily_costs = data.get('daily_costs', [])
    provider_breakdown = data.get('provider_breakdown', {})
    
    return {
        'type': 'bar',
        'data': {
            'labels': [item['date'] for item in daily_costs],
            'datasets': [
                {
                    'label': 'Daily Cost ($)',
                    'data': [item['cost'] for item in daily_costs],
                    'backgroundColor': 'rgba(255, 99, 132, 0.6)',
                    'borderColor': 'rgb(255, 99, 132)',
                    'borderWidth': 1
                }
            ]
        },
        'options': {
            'responsive': True,
            'scales': {
                'x': {'title': {'display': True, 'text': 'Date'}},
                'y': {'title': {'display': True, 'text': 'Cost ($)'}, 'beginAtZero': True}
            },
            'plugins': {
                'legend': {'display': True},
                'tooltip': {
                    'callbacks': {
                        'label': 'function(context) { return "$" + context.parsed.y.toFixed(2); }'
                    }
                }
            }
        },
        'provider_breakdown': {
            'type': 'doughnut',
            'data': {
                'labels': list(provider_breakdown.keys()),
                'datasets': [{
                    'data': list(provider_breakdown.values()),
                    'backgroundColor': [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)', 
                        'rgba(255, 205, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ]
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {'position': 'bottom'},
                    'tooltip': {
                        'callbacks': {
                            'label': 'function(context) { return context.label + ": $" + context.parsed.toFixed(2); }'
                        }
                    }
                }
            }
        }
    }


def _format_model_comparison_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format model comparison data for Chart.js."""
    models = data.get('models', [])
    
    return {
        'type': 'radar',
        'data': {
            'labels': ['Latency', 'Success Rate', 'Cost Efficiency', 'Throughput', 'Reliability'],
            'datasets': [
                {
                    'label': model['name'],
                    'data': [
                        100 - (model.get('avg_latency', 1) * 100),  # Lower latency = higher score
                        model.get('success_rate', 0),
                        100 - (model.get('cost_per_token', 0.001) * 10000),  # Lower cost = higher score
                        min(model.get('requests_per_hour', 0) / 10, 100),  # Normalized throughput
                        model.get('uptime_percentage', 0)
                    ],
                    'borderColor': f'hsl({hash(model["name"]) % 360}, 70%, 50%)',
                    'backgroundColor': f'hsla({hash(model["name"]) % 360}, 70%, 50%, 0.2)',
                    'pointBackgroundColor': f'hsl({hash(model["name"]) % 360}, 70%, 50%)'
                } for model in models[:5]  # Limit to 5 models for readability
            ]
        },
        'options': {
            'responsive': True,
            'scales': {
                'r': {
                    'beginAtZero': True,
                    'max': 100,
                    'ticks': {'stepSize': 20}
                }
            },
            'plugins': {
                'legend': {'position': 'bottom'}
            }
        }
    }


def _format_alert_timeline_data(alerts: list) -> Dict[str, Any]:
    """Format alert data for timeline visualization."""
    severity_colors = {
        'critical': 'rgb(255, 99, 132)',
        'warning': 'rgb(255, 205, 86)',
        'info': 'rgb(54, 162, 235)'
    }
    
    # Group alerts by date
    alert_dates = {}
    for alert in alerts:
        date = alert['created_at'][:10]  # Extract date part
        if date not in alert_dates:
            alert_dates[date] = {'critical': 0, 'warning': 0, 'info': 0}
        alert_dates[date][alert['severity']] += 1
    
    sorted_dates = sorted(alert_dates.keys())
    
    return {
        'type': 'bar',
        'data': {
            'labels': sorted_dates,
            'datasets': [
                {
                    'label': 'Critical',
                    'data': [alert_dates[date]['critical'] for date in sorted_dates],
                    'backgroundColor': severity_colors['critical']
                },
                {
                    'label': 'Warning', 
                    'data': [alert_dates[date]['warning'] for date in sorted_dates],
                    'backgroundColor': severity_colors['warning']
                },
                {
                    'label': 'Info',
                    'data': [alert_dates[date]['info'] for date in sorted_dates],
                    'backgroundColor': severity_colors['info']
                }
            ]
        },
        'options': {
            'responsive': True,
            'scales': {
                'x': {'title': {'display': True, 'text': 'Date'}, 'stacked': True},
                'y': {'title': {'display': True, 'text': 'Alert Count'}, 'stacked': True, 'beginAtZero': True}
            },
            'plugins': {
                'legend': {'display': True}
            }
        }
    }


def _format_latency_distribution_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format latency distribution data for Chart.js."""
    time_series = data.get('time_series', [])
    latencies = [item['avg_latency'] * 1000 for item in time_series]  # Convert to ms
    
    # Create histogram bins
    if latencies:
        min_lat, max_lat = min(latencies), max(latencies)
        bin_size = (max_lat - min_lat) / 10 if max_lat > min_lat else 1
        bins = [min_lat + i * bin_size for i in range(11)]
        
        # Count values in each bin
        bin_counts = [0] * 10
        for lat in latencies:
            bin_idx = min(int((lat - min_lat) / bin_size), 9)
            bin_counts[bin_idx] += 1
    else:
        bins = list(range(11))
        bin_counts = [0] * 10
    
    return {
        'type': 'bar',
        'data': {
            'labels': [f'{bins[i]:.1f}-{bins[i+1]:.1f}ms' for i in range(10)],
            'datasets': [{
                'label': 'Request Count',
                'data': bin_counts,
                'backgroundColor': 'rgba(75, 192, 192, 0.6)',
                'borderColor': 'rgb(75, 192, 192)',
                'borderWidth': 1
            }]
        },
        'options': {
            'responsive': True,
            'scales': {
                'x': {'title': {'display': True, 'text': 'Latency Range'}},
                'y': {'title': {'display': True, 'text': 'Request Count'}, 'beginAtZero': True}
            },
            'plugins': {
                'legend': {'display': True}
            }
        }
    }


def _format_success_rate_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format success rate data for Chart.js."""
    time_series = data.get('time_series', [])
    
    return {
        'type': 'line',
        'data': {
            'labels': [item['timestamp'] for item in time_series],
            'datasets': [{
                'label': 'Success Rate (%)',
                'data': [item['success_rate'] for item in time_series],
                'borderColor': 'rgb(75, 192, 192)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'fill': True,
                'tension': 0.4
            }]
        },
        'options': {
            'responsive': True,
            'scales': {
                'x': {'title': {'display': True, 'text': 'Time'}},
                'y': {'title': {'display': True, 'text': 'Success Rate (%)'}, 'min': 0, 'max': 100}
            },
            'plugins': {
                'legend': {'display': True}
            }
        }
    }