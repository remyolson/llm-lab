"""
Email Delivery System

Provides email delivery functionality for automated report distribution
with support for multiple recipients, attachments, and delivery tracking.
"""

import smtplib
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import json

from jinja2 import Template


class EmailDelivery:
    """Email delivery system for reports."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        """
        Initialize email delivery system.
        
        Args:
            smtp_config: SMTP configuration dict with keys:
                - host: SMTP server host
                - port: SMTP server port
                - username: SMTP username
                - password: SMTP password
                - use_tls: Whether to use TLS
                - from_email: From email address
                - from_name: From name (optional)
        """
        self.smtp_config = smtp_config
        self.logger = logging.getLogger(__name__)
        
        # Email templates
        self.templates = {
            'report_email': self._get_report_email_template(),
            'alert_email': self._get_alert_email_template(),
            'summary_email': self._get_summary_email_template()
        }
    
    def send_report_email(
        self,
        recipients: List[str],
        subject: str,
        report_path: str,
        template_name: str = "daily_summary",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a report via email.
        
        Args:
            recipients: List of email addresses
            subject: Email subject
            report_path: Path to the report file
            template_name: Name of the report template
            additional_context: Additional context for email template
            
        Returns:
            Dict with delivery status and metadata
        """
        try:
            # Prepare email content
            context = {
                'template_name': template_name,
                'report_filename': Path(report_path).name,
                'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'file_size': self._format_file_size(Path(report_path).stat().st_size),
                **(additional_context or {})
            }
            
            html_body = self.templates['report_email'].render(**context)
            text_body = self._html_to_text(html_body)
            
            # Send email with attachment
            result = self._send_email(
                recipients=recipients,
                subject=subject,
                html_body=html_body,
                text_body=text_body,
                attachments=[report_path]
            )
            
            self.logger.info(f"Report email sent to {len(recipients)} recipients")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to send report email: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def send_alert_email(
        self,
        recipients: List[str],
        alert_data: Dict[str, Any],
        additional_alerts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Send an alert notification email.
        
        Args:
            recipients: List of email addresses
            alert_data: Primary alert data
            additional_alerts: Additional related alerts
            
        Returns:
            Dict with delivery status and metadata
        """
        try:
            severity = alert_data.get('severity', 'info')
            subject = f"[{severity.upper()}] {alert_data.get('title', 'System Alert')}"
            
            context = {
                'alert': alert_data,
                'additional_alerts': additional_alerts or [],
                'severity_color': self._get_severity_color(severity),
                'alert_count': 1 + len(additional_alerts or [])
            }
            
            html_body = self.templates['alert_email'].render(**context)
            text_body = self._html_to_text(html_body)
            
            result = self._send_email(
                recipients=recipients,
                subject=subject,
                html_body=html_body,
                text_body=text_body
            )
            
            self.logger.info(f"Alert email sent to {len(recipients)} recipients")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to send alert email: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def send_summary_email(
        self,
        recipients: List[str],
        summary_data: Dict[str, Any],
        period: str = "daily"
    ) -> Dict[str, Any]:
        """
        Send a summary email.
        
        Args:
            recipients: List of email addresses
            summary_data: Summary data to include
            period: Summary period (daily, weekly, monthly)
            
        Returns:
            Dict with delivery status and metadata
        """
        try:
            subject = f"{period.title()} Monitoring Summary - {datetime.utcnow().strftime('%Y-%m-%d')}"
            
            context = {
                'period': period,
                'summary': summary_data,
                'metrics': summary_data.get('metrics', {}),
                'alerts': summary_data.get('alerts', []),
                'trends': summary_data.get('trends', {}),
                'recommendations': summary_data.get('recommendations', [])
            }
            
            html_body = self.templates['summary_email'].render(**context)
            text_body = self._html_to_text(html_body)
            
            result = self._send_email(
                recipients=recipients,
                subject=subject,
                html_body=html_body,
                text_body=text_body
            )
            
            self.logger.info(f"Summary email sent to {len(recipients)} recipients")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to send summary email: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test SMTP connection."""
        try:
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            
            if self.smtp_config.get('use_tls', True):
                server.starttls()
            
            if self.smtp_config.get('username') and self.smtp_config.get('password'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            server.quit()
            
            return {
                'status': 'success',
                'message': 'SMTP connection successful',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"SMTP connection test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _send_email(
        self,
        recipients: List[str],
        subject: str,
        html_body: str,
        text_body: str,
        attachments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Send email via SMTP."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.smtp_config.get('from_name', 'LLM Monitor')} <{self.smtp_config['from_email']}>"
            msg['To'] = ', '.join(recipients)
            
            # Add text and HTML parts
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Add attachments
            if attachments:
                for attachment_path in attachments:
                    self._add_attachment(msg, attachment_path)
            
            # Send email
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            
            if self.smtp_config.get('use_tls', True):
                server.starttls()
            
            if self.smtp_config.get('username') and self.smtp_config.get('password'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            return {
                'status': 'sent',
                'recipients': recipients,
                'subject': subject,
                'timestamp': datetime.utcnow().isoformat(),
                'attachments': len(attachments) if attachments else 0
            }
            
        except Exception as e:
            raise Exception(f"Email sending failed: {e}")
    
    def _add_attachment(self, msg: MIMEMultipart, file_path: str):
        """Add file attachment to email."""
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.warning(f"Attachment file not found: {file_path}")
                return
            
            with open(path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {path.name}'
            )
            
            msg.attach(part)
            
        except Exception as e:
            self.logger.error(f"Failed to add attachment {file_path}: {e}")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color for alert severity."""
        colors = {
            'critical': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8'
        }
        return colors.get(severity, '#6c757d')
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        # Simple HTML to text conversion
        import re
        
        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', html)
        
        # Replace common HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _get_report_email_template(self) -> Template:
        """Get report email template."""
        template_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Monitoring Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background-color: #007bff; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; }
        .footer { background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #6c757d; }
        .attachment-info { background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 15px 0; }
        .button { display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0; }
        .alert { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .alert-info { background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Monitoring Report</h1>
            <p>{{ template_name.replace('_', ' ').title() }}</p>
        </div>
        
        <div class="content">
            <p>Hello,</p>
            
            <p>Your monitoring report has been generated and is attached to this email.</p>
            
            <div class="attachment-info">
                <h3>📎 Report Details</h3>
                <ul>
                    <li><strong>Report Type:</strong> {{ template_name.replace('_', ' ').title() }}</li>
                    <li><strong>Generated:</strong> {{ generated_at }}</li>
                    <li><strong>File Name:</strong> {{ report_filename }}</li>
                    <li><strong>File Size:</strong> {{ file_size }}</li>
                </ul>
            </div>
            
            {% if additional_context %}
            <div class="alert alert-info">
                <h4>Additional Information</h4>
                {% for key, value in additional_context.items() %}
                <p><strong>{{ key.replace('_', ' ').title() }}:</strong> {{ value }}</p>
                {% endfor %}
            </div>
            {% endif %}
            
            <p>If you have any questions about this report, please contact your system administrator.</p>
        </div>
        
        <div class="footer">
            <p>This is an automated message from the LLM Monitoring System.</p>
            <p>Generated at {{ generated_at }}</p>
        </div>
    </div>
</body>
</html>
        """
        return Template(template_html)
    
    def _get_alert_email_template(self) -> Template:
        """Get alert email template."""
        template_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>System Alert</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background-color: {{ severity_color }}; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; }
        .footer { background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #6c757d; }
        .alert-details { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid {{ severity_color }}; }
        .additional-alerts { margin-top: 20px; }
        .alert-item { background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 3px solid #ffc107; }
        .timestamp { color: #6c757d; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 System Alert</h1>
            <p>{{ alert.severity.upper() }} - {{ alert.title }}</p>
        </div>
        
        <div class="content">
            <div class="alert-details">
                <h3>Alert Details</h3>
                <p><strong>Message:</strong> {{ alert.message }}</p>
                <p><strong>Severity:</strong> {{ alert.severity.upper() }}</p>
                {% if alert.provider %}
                <p><strong>Provider:</strong> {{ alert.provider }}</p>
                {% endif %}
                {% if alert.model %}
                <p><strong>Model:</strong> {{ alert.model }}</p>
                {% endif %}
                {% if alert.current_value and alert.threshold_value %}
                <p><strong>Current Value:</strong> {{ alert.current_value }}</p>
                <p><strong>Threshold:</strong> {{ alert.threshold_value }}</p>
                {% endif %}
                <p class="timestamp"><strong>Created:</strong> {{ alert.created_at }}</p>
            </div>
            
            {% if additional_alerts %}
            <div class="additional-alerts">
                <h3>Related Alerts ({{ additional_alerts|length }})</h3>
                {% for related_alert in additional_alerts %}
                <div class="alert-item">
                    <strong>{{ related_alert.title }}</strong><br>
                    {{ related_alert.message }}<br>
                    <span class="timestamp">{{ related_alert.created_at }}</span>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <p><strong>Action Required:</strong> Please review the monitoring dashboard and take appropriate action if necessary.</p>
        </div>
        
        <div class="footer">
            <p>This is an automated alert from the LLM Monitoring System.</p>
            <p>Total alerts: {{ alert_count }}</p>
        </div>
    </div>
</body>
</html>
        """
        return Template(template_html)
    
    def _get_summary_email_template(self) -> Template:
        """Get summary email template."""
        template_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ period.title() }} Summary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 700px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background-color: #28a745; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; }
        .footer { background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #6c757d; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { color: #6c757d; font-size: 12px; margin-top: 5px; }
        .trends { background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 15px 0; }
        .trend-item { margin: 5px 0; }
        .trend-up { color: #28a745; }
        .trend-down { color: #dc3545; }
        .trend-stable { color: #6c757d; }
        .recommendations { background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 15px 0; }
        .recommendation { margin: 8px 0; padding: 8px; background-color: white; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 {{ period.title() }} Summary</h1>
            <p>LLM Monitoring System Report</p>
        </div>
        
        <div class="content">
            <h2>Key Metrics</h2>
            <div class="metrics-grid">
                {% for metric, value in metrics.items() %}
                <div class="metric-card">
                    <div class="metric-value">{{ value }}</div>
                    <div class="metric-label">{{ metric.replace('_', ' ').title() }}</div>
                </div>
                {% endfor %}
            </div>
            
            {% if trends %}
            <div class="trends">
                <h3>Performance Trends</h3>
                {% for trend, direction in trends.items() %}
                <div class="trend-item">
                    <span>{{ trend.replace('_', ' ').title() }}:</span>
                    <span class="trend-{{ direction }}">
                        {% if direction == 'improving' %}📈 Improving
                        {% elif direction == 'degrading' %}📉 Degrading
                        {% else %}➡️ Stable{% endif %}
                    </span>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if alerts %}
            <h3>Recent Alerts ({{ alerts|length }})</h3>
            {% for alert in alerts[:5] %}
            <div class="alert-item" style="background-color: {% if alert.severity == 'critical' %}#f8d7da{% elif alert.severity == 'warning' %}#fff3cd{% else %}#d1ecf1{% endif %}; padding: 10px; margin: 5px 0; border-radius: 3px;">
                <strong>{{ alert.title }}</strong> ({{ alert.severity }})<br>
                <span style="font-size: 12px; color: #6c757d;">{{ alert.created_at }}</span>
            </div>
            {% endfor %}
            {% endif %}
            
            {% if recommendations %}
            <div class="recommendations">
                <h3>💡 Recommendations</h3>
                {% for recommendation in recommendations %}
                <div class="recommendation">{{ recommendation }}</div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>This is an automated {{ period }} summary from the LLM Monitoring System.</p>
        </div>
    </div>
</body>
</html>
        """
        return Template(template_html)


def create_email_delivery(smtp_config: Dict[str, Any]) -> EmailDelivery:
    """Factory function to create email delivery system."""
    return EmailDelivery(smtp_config)