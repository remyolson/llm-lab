"""
Multi-Format Data Export System

This module provides comprehensive data export capabilities for evaluation results,
supporting multiple formats including JSON, CSV, Excel, Parquet, HDF5, and SQL.

Example:
    exporter = DataExporter()

    # Export evaluation results
    exporter.export_results(
        comparison_results,
        format="excel",
        output_path="evaluation_results.xlsx"
    )

    # Export with custom transformation
    exporter.export_with_transform(
        data,
        format="parquet",
        transform_fn=lambda x: flatten_nested_data(x)
    )
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import csv
import json
import logging
import pickle
import sqlite3
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Import evaluation components
from ..ab_testing.ab_testing import ABTestResults
from ..analysis.cost_benefit import CostAnalysis, ROIAnalysis
from ..benchmark_runner import BenchmarkResult, ComparisonResult
from ..plugins.metric_plugin import MetricResult
from ..reporting.report_generator import ReportContent

# Optional imports for advanced formats
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import xlsxwriter

    EXCEL_ADVANCED_AVAILABLE = True
except ImportError:
    EXCEL_ADVANCED_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    SQL = "sql"
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"


@dataclass
class ExportConfig:
    """Configuration for data export."""

    format: ExportFormat
    output_path: Optional[str] = None
    include_metadata: bool = True
    include_timestamps: bool = True
    compression: Optional[str] = None  # 'gzip', 'bz2', 'xz', 'zip'
    flatten_nested: bool = False
    decimal_places: int = 4
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Format-specific options
    csv_delimiter: str = ","
    csv_quoting: int = csv.QUOTE_MINIMAL
    excel_sheet_names: Optional[List[str]] = None
    excel_index: bool = False
    json_indent: int = 2
    sql_table_name: str = "evaluation_results"
    sql_if_exists: str = "replace"  # 'fail', 'replace', 'append'
    parquet_engine: str = "pyarrow"
    hdf5_key: str = "data"


class DataTransformer:
    """Transform data for export."""

    @staticmethod
    def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str | Any]:
        """Flatten nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(DataTransformer.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to indexed dict
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(
                            DataTransformer.flatten_dict(
                                item, f"{new_key}{sep}{i}", sep=sep
                            ).items()
                        )
                    else:
                        items.append((f"{new_key}{sep}{i}", item))
            else:
                items.append((new_key, v))

        return dict(items)

    @staticmethod
    def dataclass_to_dict(obj: Any) -> Any:
        """Convert dataclass to dictionary recursively.

        Args:
            obj: Object to convert

        Returns:
            Dictionary representation
        """
        if is_dataclass(obj):
            return {k: DataTransformer.dataclass_to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [DataTransformer.dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: DataTransformer.dataclass_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    @staticmethod
    def normalize_data(data: Any) -> Dict | List[Dict]:
        """Normalize data to dictionary format.

        Args:
            data: Data to normalize

        Returns:
            Normalized data
        """
        # Handle various data types
        if isinstance(
            data,
            (
                ComparisonResult,
                BenchmarkResult,
                ABTestResults,
                MetricResult,
                CostAnalysis,
                ROIAnalysis,
                ReportContent,
            ),
        ):
            # Convert known types to dict
            if hasattr(data, "to_dict"):
                return data.to_dict()
            else:
                return DataTransformer.dataclass_to_dict(data)

        elif is_dataclass(data):
            return DataTransformer.dataclass_to_dict(data)

        elif isinstance(data, pd.DataFrame):
            return data.to_dict("records")

        elif isinstance(data, list):
            return [DataTransformer.normalize_data(item) for item in data]

        elif isinstance(data, dict):
            return {k: DataTransformer.normalize_data(v) for k, v in data.items()}

        else:
            return data

    @staticmethod
    def prepare_for_tabular(data: Dict | List[Dict]) -> pd.DataFrame:
        """Prepare data for tabular export.

        Args:
            data: Data to prepare

        Returns:
            DataFrame
        """
        # Normalize data first
        normalized = DataTransformer.normalize_data(data)

        # Convert to DataFrame
        if isinstance(normalized, dict):
            # Single record
            df = pd.DataFrame([normalized])
        elif isinstance(normalized, list):
            # Multiple records
            df = pd.DataFrame(normalized)
        else:
            df = pd.DataFrame([{"value": normalized}])

        # Handle nested columns
        for col in df.columns:
            if df[col].dtype == "object":
                # Check if column contains dicts or lists
                first_val = df[col].iloc[0] if len(df) > 0 else None
                if isinstance(first_val, (dict, list)):
                    # Convert to string representation
                    df[col] = df[col].apply(json.dumps)

        return df


class DataExporter:
    """Multi-format data exporter."""

    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize exporter.

        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig(format=ExportFormat.JSON)
        self.transformer = DataTransformer()

    def export(
        self,
        data: Any,
        format: Optional[ExportFormat] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str | bytes | None:
        """Export data in specified format.

        Args:
            data: Data to export
            format: Export format
            output_path: Output file path
            **kwargs: Format-specific options

        Returns:
            Exported data (if no output_path) or None
        """
        format = format or self.config.format
        output_path = output_path or self.config.output_path

        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Select export method
        export_methods = {
            ExportFormat.JSON: self._export_json,
            ExportFormat.CSV: self._export_csv,
            ExportFormat.EXCEL: self._export_excel,
            ExportFormat.PARQUET: self._export_parquet,
            ExportFormat.HDF5: self._export_hdf5,
            ExportFormat.PICKLE: self._export_pickle,
            ExportFormat.SQL: self._export_sql,
            ExportFormat.MARKDOWN: self._export_markdown,
            ExportFormat.HTML: self._export_html,
            ExportFormat.LATEX: self._export_latex,
        }

        export_method = export_methods.get(format)
        if not export_method:
            raise ValueError(f"Unsupported format: {format}")

        # Export data
        result = export_method(data, output_path)

        logger.info(f"Data exported in {format.value} format")
        if output_path:
            logger.info(f"Saved to: {output_path}")

        return result

    def _export_json(self, data: Any, output_path: Optional[str] = None) -> Optional[str]:
        """Export data as JSON.

        Args:
            data: Data to export
            output_path: Output file path

        Returns:
            JSON string if no output_path
        """
        # Normalize data
        normalized = self.transformer.normalize_data(data)

        # Add metadata if configured
        if self.config.include_metadata:
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "export_format": "json",
                    "data_type": type(data).__name__,
                },
                "data": normalized,
            }
        else:
            export_data = normalized

        # Convert to JSON
        json_str = json.dumps(export_data, indent=self.config.json_indent, default=str)

        if output_path:
            with open(output_path, "w") as f:
                f.write(json_str)
            return None
        else:
            return json_str

    def _export_csv(self, data: Any, output_path: Optional[str] = None) -> Optional[str]:
        """Export data as CSV.

        Args:
            data: Data to export
            output_path: Output file path

        Returns:
            CSV string if no output_path
        """
        # Prepare DataFrame
        df = self.transformer.prepare_for_tabular(data)

        # Flatten if configured
        if self.config.flatten_nested:
            # Flatten any remaining nested structures
            flat_data = []
            for _, row in df.iterrows():
                flat_row = self.transformer.flatten_dict(row.to_dict())
                flat_data.append(flat_row)
            df = pd.DataFrame(flat_data)

        # Round numerical columns
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(self.config.decimal_places)

        # Export
        if output_path:
            df.to_csv(
                output_path,
                index=False,
                sep=self.config.csv_delimiter,
                quoting=self.config.csv_quoting,
            )
            return None
        else:
            buffer = StringIO()
            df.to_csv(
                buffer, index=False, sep=self.config.csv_delimiter, quoting=self.config.csv_quoting
            )
            return buffer.getvalue()

    def _export_excel(self, data: Any, output_path: Optional[str] = None) -> Optional[bytes]:
        """Export data as Excel.

        Args:
            data: Data to export
            output_path: Output file path

        Returns:
            Excel bytes if no output_path
        """
        # Handle different data structures
        if isinstance(data, dict):
            # Multiple sheets
            sheets_data = {}

            for key, value in data.items():
                df = self.transformer.prepare_for_tabular(value)
                sheets_data[str(key)[:31]] = df  # Excel sheet name limit
        else:
            # Single sheet
            df = self.transformer.prepare_for_tabular(data)
            sheets_data = {"Data": df}

        # Create Excel file
        if output_path:
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                for sheet_name, sheet_df in sheets_data.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=self.config.excel_index)

                    # Format if xlsxwriter available
                    if EXCEL_ADVANCED_AVAILABLE:
                        workbook = writer.book
                        worksheet = writer.sheets[sheet_name]

                        # Add formatting
                        header_format = workbook.add_format(
                            {"bold": True, "bg_color": "#D7E4BD", "border": 1}
                        )

                        # Format header row
                        for col_num, value in enumerate(sheet_df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
            return None
        else:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                for sheet_name, sheet_df in sheets_data.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=self.config.excel_index)
            return buffer.getvalue()

    def _export_parquet(self, data: Any, output_path: Optional[str] = None) -> Optional[bytes]:
        """Export data as Parquet.

        Args:
            data: Data to export
            output_path: Output file path

        Returns:
            Parquet bytes if no output_path
        """
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow is required for Parquet export")

        # Prepare DataFrame
        df = self.transformer.prepare_for_tabular(data)

        # Convert to Parquet
        if output_path:
            df.to_parquet(
                output_path, engine=self.config.parquet_engine, compression=self.config.compression
            )
            return None
        else:
            buffer = BytesIO()
            df.to_parquet(
                buffer, engine=self.config.parquet_engine, compression=self.config.compression
            )
            return buffer.getvalue()

    def _export_hdf5(self, data: Any, output_path: Optional[str] = None) -> Optional[bytes]:
        """Export data as HDF5.

        Args:
            data: Data to export
            output_path: Output file path

        Returns:
            HDF5 bytes if no output_path
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 export")

        # Prepare DataFrame
        df = self.transformer.prepare_for_tabular(data)

        # Export to HDF5
        if output_path:
            df.to_hdf(
                output_path,
                key=self.config.hdf5_key,
                mode="w",
                complevel=9 if self.config.compression else 0,
            )
            return None
        else:
            # HDF5 doesn't support in-memory easily, save to temp
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                df.to_hdf(
                    tmp.name,
                    key=self.config.hdf5_key,
                    mode="w",
                    complevel=9 if self.config.compression else 0,
                )
                tmp.seek(0)
                return tmp.read()

    def _export_pickle(self, data: Any, output_path: Optional[str] = None) -> Optional[bytes]:
        """Export data as Pickle.

        Args:
            data: Data to export
            output_path: Output file path

        Returns:
            Pickle bytes if no output_path
        """
        # Normalize data
        normalized = self.transformer.normalize_data(data)

        if output_path:
            with open(output_path, "wb") as f:
                pickle.dump(normalized, f)
            return None
        else:
            buffer = BytesIO()
            pickle.dump(normalized, buffer)
            return buffer.getvalue()

    def _export_sql(self, data: Any, output_path: Optional[str] = None) -> Optional[str]:
        """Export data to SQL database.

        Args:
            data: Data to export
            output_path: Database path (SQLite)

        Returns:
            SQL statements if no output_path
        """
        # Prepare DataFrame
        df = self.transformer.prepare_for_tabular(data)

        if output_path:
            # Create SQLite database
            conn = sqlite3.connect(output_path)
            df.to_sql(
                self.config.sql_table_name, conn, if_exists=self.config.sql_if_exists, index=False
            )
            conn.close()
            return None
        else:
            # Generate SQL statements
            sql_statements = []

            # CREATE TABLE statement
            columns = []
            for col, dtype in zip(df.columns, df.dtypes):
                if dtype == "object":
                    sql_type = "TEXT"
                elif dtype == "int64":
                    sql_type = "INTEGER"
                elif dtype == "float64":
                    sql_type = "REAL"
                else:
                    sql_type = "TEXT"
                columns.append(f"{col} {sql_type}")

            create_stmt = f"CREATE TABLE {self.config.sql_table_name} ({', '.join(columns)});"
            sql_statements.append(create_stmt)

            # INSERT statements
            for _, row in df.iterrows():
                values = []
                for val in row.values:
                    if pd.isna(val):
                        values.append("NULL")
                    elif isinstance(val, str):
                        values.append(f"'{val}'")
                    else:
                        values.append(str(val))

                insert_stmt = (
                    f"INSERT INTO {self.config.sql_table_name} VALUES ({', '.join(values)});"
                )
                sql_statements.append(insert_stmt)

            return "\n".join(sql_statements)

    def _export_markdown(self, data: Any, output_path: Optional[str] = None) -> Optional[str]:
        """Export data as Markdown table.

        Args:
            data: Data to export
            output_path: Output file path

        Returns:
            Markdown string if no output_path
        """
        # Prepare DataFrame
        df = self.transformer.prepare_for_tabular(data)

        # Round numerical columns
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(self.config.decimal_places)

        # Convert to Markdown
        markdown = df.to_markdown(index=False)

        # Add metadata if configured
        if self.config.include_metadata:
            metadata = f"""
# Data Export

- **Export Date**: {datetime.now().strftime(self.config.date_format)}
- **Records**: {len(df)}
- **Columns**: {len(df.columns)}

## Data

{markdown}
"""
            markdown = metadata

        if output_path:
            with open(output_path, "w") as f:
                f.write(markdown)
            return None
        else:
            return markdown

    def _export_html(self, data: Any, output_path: Optional[str] = None) -> Optional[str]:
        """Export data as HTML table.

        Args:
            data: Data to export
            output_path: Output file path

        Returns:
            HTML string if no output_path
        """
        # Prepare DataFrame
        df = self.transformer.prepare_for_tabular(data)

        # Round numerical columns
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(self.config.decimal_places)

        # Convert to HTML
        html = df.to_html(index=False, classes="table table-striped")

        # Add full HTML structure
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Export</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>Data Export</h1>
    <p>Export Date: {datetime.now().strftime(self.config.date_format)}</p>
    <p>Records: {len(df)}</p>
    {html}
</body>
</html>
"""

        if output_path:
            with open(output_path, "w") as f:
                f.write(full_html)
            return None
        else:
            return full_html

    def _export_latex(self, data: Any, output_path: Optional[str] = None) -> Optional[str]:
        """Export data as LaTeX table.

        Args:
            data: Data to export
            output_path: Output file path

        Returns:
            LaTeX string if no output_path
        """
        # Prepare DataFrame
        df = self.transformer.prepare_for_tabular(data)

        # Round numerical columns
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(self.config.decimal_places)

        # Convert to LaTeX
        latex = df.to_latex(index=False, escape=True)

        # Add document structure
        full_latex = f"""
\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\begin{{document}}

\\section{{Data Export}}

Export Date: {datetime.now().strftime(self.config.date_format)}

Records: {len(df)}

{latex}

\\end{{document}}
"""

        if output_path:
            with open(output_path, "w") as f:
                f.write(full_latex)
            return None
        else:
            return full_latex

    def export_with_transform(
        self,
        data: Any,
        transform_fn: Callable,
        format: Optional[ExportFormat] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str | bytes | None:
        """Export data with custom transformation.

        Args:
            data: Data to export
            transform_fn: Transformation function
            format: Export format
            output_path: Output file path
            **kwargs: Format-specific options

        Returns:
            Exported data
        """
        # Apply transformation
        transformed_data = transform_fn(data)

        # Export transformed data
        return self.export(transformed_data, format, output_path, **kwargs)

    def export_batch(
        self,
        data_dict: Dict[str, Any],
        format: Optional[ExportFormat] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str | str | bytes | None]:
        """Export multiple datasets.

        Args:
            data_dict: Dictionary of datasets
            format: Export format
            output_dir: Output directory
            **kwargs: Format-specific options

        Returns:
            Dictionary of exported data
        """
        format = format or self.config.format
        results = {}

        for name, data in data_dict.items():
            if output_dir:
                # Generate output path
                extension = self._get_extension(format)
                output_path = Path(output_dir) / f"{name}.{extension}"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                result = self.export(data, format, str(output_path), **kwargs)
            else:
                result = self.export(data, format, None, **kwargs)

            results[name] = result

        return results

    def _get_extension(self, format: ExportFormat) -> str:
        """Get file extension for format.

        Args:
            format: Export format

        Returns:
            File extension
        """
        extensions = {
            ExportFormat.JSON: "json",
            ExportFormat.CSV: "csv",
            ExportFormat.EXCEL: "xlsx",
            ExportFormat.PARQUET: "parquet",
            ExportFormat.HDF5: "h5",
            ExportFormat.PICKLE: "pkl",
            ExportFormat.SQL: "db",
            ExportFormat.MARKDOWN: "md",
            ExportFormat.HTML: "html",
            ExportFormat.LATEX: "tex",
        }
        return extensions.get(format, "dat")


# Convenience functions
def export_comparison_results(
    comparison: ComparisonResult,
    format: ExportFormat = ExportFormat.EXCEL,
    output_path: Optional[str] = None,
) -> str | bytes | None:
    """Export comparison results.

    Args:
        comparison: Comparison results
        format: Export format
        output_path: Output file path

    Returns:
        Exported data
    """
    exporter = DataExporter()

    # Prepare structured data
    data = {
        "summary": {
            "base_model": comparison.base_result.model_version.model_path,
            "fine_tuned_model": comparison.fine_tuned_result.model_version.model_path,
            "improvements": comparison.improvements,
            "regressions": comparison.regressions,
        },
        "base_results": comparison.base_result.to_dict(),
        "fine_tuned_results": comparison.fine_tuned_result.to_dict(),
        "statistical_analysis": comparison.statistical_analysis,
    }

    return exporter.export(data, format, output_path)


def export_ab_test_results(
    results: ABTestResults,
    format: ExportFormat = ExportFormat.JSON,
    output_path: Optional[str] = None,
) -> str | bytes | None:
    """Export A/B test results.

    Args:
        results: A/B test results
        format: Export format
        output_path: Output file path

    Returns:
        Exported data
    """
    exporter = DataExporter()
    return exporter.export(results, format, output_path)


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        "evaluation_results": [
            {
                "model": "gpt2",
                "benchmark": "hellaswag",
                "score": 0.456,
                "samples": 1000,
                "duration": 120.5,
            },
            {
                "model": "gpt2-finetuned",
                "benchmark": "hellaswag",
                "score": 0.523,
                "samples": 1000,
                "duration": 125.3,
            },
        ],
        "metadata": {
            "date": datetime.now().isoformat(),
            "evaluator": "AutoBenchmarkRunner",
            "version": "1.0.0",
        },
    }

    # Create exporter
    exporter = DataExporter()

    # Export to different formats
    print("Exporting to JSON...")
    json_data = exporter.export(sample_data, ExportFormat.JSON)
    print(f"JSON length: {len(json_data)} characters")

    print("\nExporting to CSV...")
    csv_data = exporter.export(sample_data["evaluation_results"], ExportFormat.CSV)
    print("CSV preview:")
    print(csv_data[:200])

    print("\nExporting to Markdown...")
    markdown_data = exporter.export(sample_data["evaluation_results"], ExportFormat.MARKDOWN)
    print("Markdown preview:")
    print(markdown_data[:300])

    # Export with transformation
    def custom_transform(data):
        # Extract only scores
        return [{"model": r["model"], "score": r["score"]} for r in data["evaluation_results"]]

    print("\nExporting with custom transformation...")
    transformed_data = exporter.export_with_transform(
        sample_data, custom_transform, ExportFormat.CSV
    )
    print("Transformed CSV:")
    print(transformed_data)

    # Batch export
    batch_data = {"results": sample_data["evaluation_results"], "metadata": sample_data["metadata"]}

    print("\nBatch export to multiple files...")
    batch_results = exporter.export_batch(batch_data, format=ExportFormat.JSON)
    print(f"Exported {len(batch_results)} datasets")

    print("\nData export examples completed successfully!")
