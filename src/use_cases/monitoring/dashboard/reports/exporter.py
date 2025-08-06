"""
Data Exporter

Provides data export functionality in various formats (CSV, JSON, Excel)
with filtering, aggregation, and formatting capabilities.
"""

import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available. Excel export will be limited.")


class DataExporter:
    """Data export system for monitoring data."""

    def __init__(self, data_service=None, output_dir: Optional[str] = None):
        self.data_service = data_service
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "exports"
        self.output_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def export_data(
        self,
        data_type: str,
        format: str = "csv",
        date_range: Optional[tuple] = None,
        filters: Optional[Dict[str, Any]] = None,
        aggregation: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str | Any]:
        """
        Export monitoring data in specified format.

        Args:
            data_type: Type of data to export (metrics, performance, costs, alerts)
            format: Export format (csv, json, xlsx)
            date_range: (start_date, end_date) tuple
            filters: Data filters to apply
            aggregation: Aggregation method (hourly, daily, weekly)
            filename: Custom filename (optional)

        Returns:
            Dict with export metadata and file path
        """
        try:
            # Gather data
            data = self._gather_export_data(data_type, date_range, filters)

            if not data:
                return {
                    "status": "error",
                    "message": "No data available for export",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # Apply aggregation if specified
            if aggregation:
                data = self._apply_aggregation(data, aggregation, data_type)

            # Format data for export
            formatted_data = self._format_for_export(data, data_type)

            # Generate filename
            if not filename:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"{data_type}_export_{timestamp}.{format}"

            output_path = self.output_dir / filename

            # Export based on format
            if format.lower() == "csv":
                self._export_csv(formatted_data, output_path)
            elif format.lower() == "json":
                self._export_json(formatted_data, output_path)
            elif format.lower() == "xlsx":
                self._export_xlsx(formatted_data, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            export_metadata = {
                "status": "completed",
                "data_type": data_type,
                "format": format,
                "file_path": str(output_path),
                "file_size": output_path.stat().st_size,
                "record_count": len(formatted_data) if isinstance(formatted_data, list) else 1,
                "date_range": date_range,
                "filters": filters,
                "aggregation": aggregation,
                "exported_at": datetime.utcnow().isoformat(),
            }

            self.logger.info(f"Data exported: {output_path}")
            return export_metadata

        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            return {"status": "failed", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def export_custom_query(
        self, query_config: Dict[str, Any], format: str = "csv", filename: Optional[str] = None
    ) -> Dict[str | Any]:
        """
        Export data based on custom query configuration.

        Args:
            query_config: Custom query configuration
            format: Export format
            filename: Custom filename

        Returns:
            Export metadata dict
        """
        try:
            # Build query from config
            data_sources = query_config.get("data_sources", [])
            joins = query_config.get("joins", [])
            filters = query_config.get("filters", {})
            fields = query_config.get("fields", [])

            # Gather data from multiple sources
            combined_data = []
            for source in data_sources:
                source_data = self._gather_export_data(
                    source["type"], source.get("date_range"), source.get("filters", {})
                )

                if source_data:
                    # Add source identifier
                    if isinstance(source_data, list):
                        for item in source_data:
                            item["_source"] = source["type"]
                        combined_data.extend(source_data)
                    else:
                        source_data["_source"] = source["type"]
                        combined_data.append(source_data)

            # Apply field selection
            if fields:
                combined_data = self._select_fields(combined_data, fields)

            # Export combined data
            return self.export_data(data_type="custom", format=format, filename=filename)

        except Exception as e:
            self.logger.error(f"Custom query export failed: {e}")
            return {"status": "failed", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def get_export_templates(self) -> List[Dict[str | Any]]:
        """Get predefined export templates."""
        return [
            {
                "name": "daily_metrics",
                "description": "Daily metrics summary",
                "data_type": "metrics",
                "aggregation": "daily",
                "default_format": "csv",
            },
            {
                "name": "performance_analysis",
                "description": "Performance analysis with trends",
                "data_type": "performance",
                "aggregation": "hourly",
                "default_format": "xlsx",
            },
            {
                "name": "cost_breakdown",
                "description": "Detailed cost breakdown by provider",
                "data_type": "costs",
                "aggregation": "daily",
                "default_format": "csv",
            },
            {
                "name": "alert_history",
                "description": "Complete alert history",
                "data_type": "alerts",
                "default_format": "json",
            },
            {
                "name": "model_comparison",
                "description": "Model performance comparison",
                "data_type": "performance",
                "filters": {"include_models": True},
                "default_format": "xlsx",
            },
        ]

    def _gather_export_data(
        self,
        data_type: str,
        date_range: Optional[tuple] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict | Dict[str | Any]]:
        """Gather data for export based on type."""
        if not self.data_service:
            return self._get_mock_export_data(data_type, date_range)

        try:
            # Calculate hours for data service calls
            if date_range:
                start_date, end_date = date_range
                hours = int((end_date - start_date).total_seconds() / 3600)
            else:
                hours = 24

            # Extract filter parameters
            provider = filters.get("provider") if filters else None
            model = filters.get("model") if filters else None

            if data_type == "metrics":
                return self.data_service.get_metrics_summary()

            elif data_type == "performance":
                performance_data = self.data_service.get_performance_data(hours, provider, model)
                return performance_data.get("time_series", [])

            elif data_type == "costs":
                cost_data = self.data_service.get_cost_breakdown(hours)
                return {
                    "daily_costs": cost_data.get("daily_costs", []),
                    "provider_breakdown": cost_data.get("provider_breakdown", {}),
                    "total_cost": cost_data.get("total_cost", 0),
                }

            elif data_type == "alerts":
                limit = filters.get("limit", 1000) if filters else 1000
                return self.data_service.get_active_alerts(limit)

            else:
                return []

        except Exception as e:
            self.logger.error(f"Data gathering failed for {data_type}: {e}")
            return self._get_mock_export_data(data_type, date_range)

    def _get_mock_export_data(
        self, data_type: str, date_range: Optional[tuple] = None
    ) -> List[Dict | Dict[str | Any]]:
        """Generate mock data for export testing."""
        if date_range:
            start_date, end_date = date_range
        else:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=1)

        if data_type == "metrics":
            return {
                "total_models": 5,
                "total_requests": 12345,
                "avg_latency": 0.456,
                "total_cost": 123.45,
                "active_alerts": 2,
                "uptime": 99.8,
                "success_rate": 98.5,
            }

        elif data_type == "performance":
            data = []
            current_time = start_date
            while current_time <= end_date:
                data.append(
                    {
                        "timestamp": current_time.isoformat(),
                        "provider": "OpenAI",
                        "model": "gpt-4o-mini",
                        "requests_count": 100 + (hash(str(current_time)) % 50),
                        "avg_latency": 0.4 + (hash(str(current_time)) % 100) / 1000,
                        "success_rate": 95 + (hash(str(current_time)) % 5),
                        "total_cost": 1.0 + (hash(str(current_time)) % 10) / 10,
                        "tokens_processed": 5000 + (hash(str(current_time)) % 1000),
                    }
                )
                current_time += timedelta(hours=1)
            return data

        elif data_type == "costs":
            daily_costs = []
            for i in range(7):
                date = start_date + timedelta(days=i)
                daily_costs.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "cost": 15.0 + i * 2.5,
                        "requests": 1000 + i * 100,
                        "tokens": 50000 + i * 5000,
                    }
                )

            return {
                "daily_costs": daily_costs,
                "provider_breakdown": {"OpenAI": 65.5, "Anthropic": 45.2, "Google": 23.8},
                "total_cost": 134.5,
            }

        elif data_type == "alerts":
            return [
                {
                    "id": 1,
                    "severity": "warning",
                    "title": "High latency detected",
                    "message": "Average response time exceeded threshold",
                    "provider": "OpenAI",
                    "model": "gpt-4o-mini",
                    "created_at": (end_date - timedelta(hours=2)).isoformat(),
                    "status": "active",
                    "current_value": 1.2,
                    "threshold_value": 1.0,
                },
                {
                    "id": 2,
                    "severity": "critical",
                    "title": "Cost threshold breached",
                    "message": "Daily cost limit exceeded",
                    "provider": "Anthropic",
                    "model": "claude-3-opus",
                    "created_at": (end_date - timedelta(hours=1)).isoformat(),
                    "status": "active",
                    "current_value": 75.0,
                    "threshold_value": 50.0,
                },
            ]

        return []

    def _apply_aggregation(
        self, data: List[Dict, Dict[str, Any]], aggregation: str, data_type: str
    ) -> List[Dict | Dict[str | Any]]:
        """Apply aggregation to time series data."""
        if not isinstance(data, list):
            return data

        if aggregation not in ["hourly", "daily", "weekly"]:
            return data

        # Group data by time period
        grouped_data = {}

        for item in data:
            if "timestamp" not in item:
                continue

            try:
                timestamp = datetime.fromisoformat(item["timestamp"])

                # Create grouping key based on aggregation
                if aggregation == "hourly":
                    key = timestamp.strftime("%Y-%m-%d %H:00:00")
                elif aggregation == "daily":
                    key = timestamp.strftime("%Y-%m-%d")
                elif aggregation == "weekly":
                    # Week starting Monday
                    week_start = timestamp - timedelta(days=timestamp.weekday())
                    key = week_start.strftime("%Y-%m-%d")

                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(item)

            except (ValueError, KeyError):
                continue

        # Aggregate grouped data
        aggregated_data = []
        for key, items in grouped_data.items():
            if not items:
                continue

            aggregated_item = {"timestamp": key, "period": aggregation, "record_count": len(items)}

            # Aggregate numeric fields
            numeric_fields = [
                "requests_count",
                "avg_latency",
                "success_rate",
                "total_cost",
                "tokens_processed",
            ]

            for field in numeric_fields:
                values = [item.get(field, 0) for item in items if item.get(field) is not None]
                if values:
                    if field in ["requests_count", "total_cost", "tokens_processed"]:
                        aggregated_item[field] = sum(values)
                    else:  # avg_latency, success_rate
                        aggregated_item[field] = sum(values) / len(values)

            # Keep first non-aggregated fields
            for field in ["provider", "model"]:
                if field in items[0]:
                    aggregated_item[field] = items[0][field]

            aggregated_data.append(aggregated_item)

        return sorted(aggregated_data, key=lambda x: x["timestamp"])

    def _format_for_export(
        self, data: List[Dict, Dict[str, Any]], data_type: str
    ) -> List[Dict[str | Any]]:
        """Format data for export."""
        if isinstance(data, dict):
            if data_type == "costs" and "daily_costs" in data:
                # Flatten cost data
                formatted_data = []

                # Add daily costs
                for daily_cost in data.get("daily_costs", []):
                    formatted_data.append(
                        {
                            "type": "daily_cost",
                            "date": daily_cost.get("date"),
                            "cost": daily_cost.get("cost"),
                            "requests": daily_cost.get("requests", 0),
                            "tokens": daily_cost.get("tokens", 0),
                        }
                    )

                # Add provider breakdown
                for provider, cost in data.get("provider_breakdown", {}).items():
                    formatted_data.append(
                        {
                            "type": "provider_cost",
                            "provider": provider,
                            "cost": cost,
                            "date": "",
                            "requests": 0,
                            "tokens": 0,
                        }
                    )

                return formatted_data
            else:
                # Convert single dict to list
                return [data]

        elif isinstance(data, list):
            # Format timestamps and clean up data
            formatted_data = []
            for item in data:
                formatted_item = item.copy()

                # Format timestamps
                if "created_at" in formatted_item:
                    try:
                        dt = datetime.fromisoformat(
                            formatted_item["created_at"].replace("Z", "+00:00")
                        )
                        formatted_item["created_at"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass

                if "timestamp" in formatted_item:
                    try:
                        dt = datetime.fromisoformat(formatted_item["timestamp"])
                        formatted_item["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass

                # Round numeric values
                for key, value in formatted_item.items():
                    if isinstance(value, float):
                        formatted_item[key] = round(value, 4)

                formatted_data.append(formatted_item)

            return formatted_data

        return []

    def _select_fields(self, data: List[Dict], fields: List[str]) -> List[Dict]:
        """Select specific fields from data."""
        if not data or not fields:
            return data

        selected_data = []
        for item in data:
            selected_item = {}
            for field in fields:
                if field in item:
                    selected_item[field] = item[field]
            selected_data.append(selected_item)

        return selected_data

    def _export_csv(self, data: List[Dict[str, Any]], output_path: Path):
        """Export data as CSV."""
        if not data:
            # Create empty file
            output_path.touch()
            return

        # Get all unique keys for headers
        headers = set()
        for item in data:
            headers.update(item.keys())

        headers = sorted(list(headers))

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)

    def _export_json(self, data: List[Dict, Dict[str, Any]], output_path: Path):
        """Export data as JSON."""
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "record_count": len(data) if isinstance(data, list) else 1,
            "data": data,
        }

        with open(output_path, "w", encoding="utf-8") as jsonfile:
            json.dump(export_data, jsonfile, indent=2, default=str)

    def _export_xlsx(self, data: List[Dict[str, Any]], output_path: Path):
        """Export data as Excel file."""
        if not PANDAS_AVAILABLE:
            # Fallback to CSV with xlsx extension
            self.logger.warning("Pandas not available. Exporting as CSV with .xlsx extension")
            csv_path = output_path.with_suffix(".csv")
            self._export_csv(data, csv_path)
            return

        try:
            df = pd.DataFrame(data)

            # Format columns
            for col in df.columns:
                if df[col].dtype == "object":
                    # Try to convert timestamp columns
                    if "timestamp" in col.lower() or "created_at" in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except:
                            pass

            # Write to Excel
            with pd.ExcelWriter(
                output_path,
                engine="openpyxl" if "openpyxl" in str(pd.__version__) else "xlsxwriter",
            ) as writer:
                df.to_excel(writer, sheet_name="Data", index=False)

                # Add summary sheet if multiple sheets would be useful
                if len(data) > 0:
                    summary_data = {
                        "Total Records": len(data),
                        "Export Date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "Columns": len(df.columns),
                    }

                    summary_df = pd.DataFrame([summary_data])
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)

        except Exception as e:
            self.logger.error(f"Excel export failed: {e}")
            # Fallback to CSV
            csv_path = output_path.with_suffix(".csv")
            self._export_csv(data, csv_path)


def create_data_exporter(data_service=None, output_dir: Optional[str] = None) -> DataExporter:
    """Factory function to create a data exporter."""
    return DataExporter(data_service, output_dir)
