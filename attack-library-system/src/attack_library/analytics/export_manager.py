"""Export manager for attack library data in multiple formats."""

import csv
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..core.library import AttackLibrary
from ..core.models import Attack, AttackCategory, AttackSeverity
from .analytics_engine import AnalyticsEngine
from .effectiveness_tracker import EffectivenessTracker
from .tagging_system import TaggingSystem

logger = logging.getLogger(__name__)


class ExportManager:
    """Manager for exporting attack library data in various formats."""

    SUPPORTED_FORMATS = ["json", "csv", "xml", "yaml", "markdown", "txt", "jsonl", "excel"]

    def __init__(
        self,
        attack_library: AttackLibrary,
        effectiveness_tracker: Optional[EffectivenessTracker] = None,
        tagging_system: Optional[TaggingSystem] = None,
        analytics_engine: Optional[AnalyticsEngine] = None,
    ):
        """
        Initialize export manager.

        Args:
            attack_library: Attack library to export from
            effectiveness_tracker: Optional effectiveness data
            tagging_system: Optional tagging system
            analytics_engine: Optional analytics engine
        """
        self.attack_library = attack_library
        self.effectiveness_tracker = effectiveness_tracker
        self.tagging_system = tagging_system
        self.analytics_engine = analytics_engine

    def export_attacks(
        self,
        output_path: Path,
        format_type: str,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_analytics: bool = False,
    ) -> Dict[str, Any]:
        """
        Export attacks in specified format.

        Args:
            output_path: Output file path
            format_type: Export format
            filters: Optional filters to apply
            include_metadata: Include metadata in export
            include_analytics: Include analytics data

        Returns:
            Export summary
        """
        if format_type.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format_type}. Supported: {self.SUPPORTED_FORMATS}"
            )

        # Get attacks with filters
        attacks = self._filter_attacks(filters or {})

        if not attacks:
            logger.warning("No attacks match the specified filters")
            return {"status": "warning", "message": "No attacks to export", "count": 0}

        # Prepare export data
        export_data = self._prepare_export_data(
            attacks, include_metadata=include_metadata, include_analytics=include_analytics
        )

        # Export in specified format
        format_lower = format_type.lower()

        if format_lower == "json":
            self._export_json(export_data, output_path)
        elif format_lower == "csv":
            self._export_csv(attacks, output_path)
        elif format_lower == "xml":
            self._export_xml(export_data, output_path)
        elif format_lower == "yaml":
            self._export_yaml(export_data, output_path)
        elif format_lower == "markdown":
            self._export_markdown(attacks, output_path)
        elif format_lower == "txt":
            self._export_txt(attacks, output_path)
        elif format_lower == "jsonl":
            self._export_jsonl(attacks, output_path)
        elif format_lower == "excel":
            self._export_excel(attacks, output_path)

        logger.info(f"Exported {len(attacks)} attacks to {output_path} in {format_type} format")

        return {
            "status": "success",
            "format": format_type,
            "output_path": str(output_path),
            "attacks_exported": len(attacks),
            "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "export_timestamp": datetime.now().isoformat(),
        }

    def _filter_attacks(self, filters: Dict[str, Any]) -> List[Attack]:
        """Apply filters to attack list."""
        attacks = list(self.attack_library.attacks.values())

        if not filters:
            return attacks

        filtered = attacks

        # Category filter
        if "category" in filters:
            category_filter = filters["category"]
            if isinstance(category_filter, str):
                category_filter = [category_filter]

            filtered = [a for a in filtered if a.category.value in category_filter]

        # Severity filter
        if "severity" in filters:
            severity_filter = filters["severity"]
            if isinstance(severity_filter, str):
                severity_filter = [severity_filter]

            filtered = [a for a in filtered if a.severity.value in severity_filter]

        # Sophistication filter
        if "sophistication" in filters:
            soph_filter = filters["sophistication"]
            if isinstance(soph_filter, dict):
                min_soph = soph_filter.get("min", 1)
                max_soph = soph_filter.get("max", 5)
                filtered = [a for a in filtered if min_soph <= a.sophistication <= max_soph]
            else:
                filtered = [a for a in filtered if a.sophistication == soph_filter]

        # Tag filter
        if "tags" in filters:
            tag_filter = filters["tags"]
            if isinstance(tag_filter, str):
                tag_filter = [tag_filter]

            filtered = [a for a in filtered if any(tag in a.metadata.tags for tag in tag_filter)]

        # Source filter
        if "source" in filters:
            source_filter = filters["source"]
            if isinstance(source_filter, str):
                source_filter = [source_filter]

            filtered = [a for a in filtered if a.metadata.source in source_filter]

        # Date range filter
        if "date_range" in filters:
            date_range = filters["date_range"]
            start_date = (
                datetime.fromisoformat(date_range["start"]) if "start" in date_range else None
            )
            end_date = datetime.fromisoformat(date_range["end"]) if "end" in date_range else None

            if start_date:
                filtered = [a for a in filtered if a.metadata.creation_date >= start_date]
            if end_date:
                filtered = [a for a in filtered if a.metadata.creation_date <= end_date]

        # Verification filter
        if "verified" in filters:
            verified = filters["verified"]
            filtered = [a for a in filtered if a.is_verified == verified]

        # Text search filter
        if "search" in filters:
            search_text = filters["search"].lower()
            filtered = [
                a
                for a in filtered
                if (
                    search_text in a.title.lower()
                    or search_text in a.content.lower()
                    or any(search_text in tag.lower() for tag in a.metadata.tags)
                )
            ]

        return filtered

    def _prepare_export_data(
        self, attacks: List[Attack], include_metadata: bool = True, include_analytics: bool = False
    ) -> Dict[str, Any]:
        """Prepare comprehensive export data."""
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_attacks": len(attacks),
                "export_version": "1.0",
                "library_source": "attack-library-system",
            },
            "attacks": [],
        }

        for attack in attacks:
            attack_data = {
                "id": attack.id,
                "title": attack.title,
                "content": attack.content,
                "category": attack.category.value,
                "severity": attack.severity.value,
                "sophistication": attack.sophistication,
                "target_models": attack.target_models,
                "is_verified": attack.is_verified,
            }

            if include_metadata:
                attack_data["metadata"] = {
                    "source": attack.metadata.source,
                    "creation_date": attack.metadata.creation_date.isoformat(),
                    "last_updated": attack.metadata.last_updated.isoformat(),
                    "tags": list(attack.metadata.tags),
                    "language": attack.metadata.language,
                    "effectiveness_score": attack.metadata.effectiveness_score,
                    "success_rate": attack.metadata.success_rate,
                    "references": attack.metadata.references,
                }

                if attack.parent_id:
                    attack_data["parent_id"] = attack.parent_id
                if attack.variant_type:
                    attack_data["variant_type"] = attack.variant_type

            # Add effectiveness data if available
            if include_analytics and self.effectiveness_tracker:
                effectiveness = self.effectiveness_tracker.get_attack_effectiveness(attack.id)
                if effectiveness["total_tests"] > 0:
                    attack_data["effectiveness_analytics"] = effectiveness

            export_data["attacks"].append(attack_data)

        # Add library statistics
        if include_metadata:
            library_stats = self.attack_library.get_statistics()
            export_data["library_statistics"] = library_stats

        # Add analytics summary
        if include_analytics and self.analytics_engine:
            try:
                overview = self.analytics_engine.generate_library_overview()
                export_data["analytics_summary"] = overview.summary
            except Exception as e:
                logger.warning(f"Failed to generate analytics summary: {e}")

        return export_data

    def _export_json(self, data: Dict[str, Any], output_path: Path):
        """Export to JSON format."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _export_csv(self, attacks: List[Attack], output_path: Path):
        """Export to CSV format."""
        if not attacks:
            return

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "id",
                "title",
                "content",
                "category",
                "severity",
                "sophistication",
                "target_models",
                "is_verified",
                "source",
                "creation_date",
                "tags",
                "effectiveness_score",
                "success_rate",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for attack in attacks:
                writer.writerow(
                    {
                        "id": attack.id,
                        "title": attack.title,
                        "content": attack.content.replace("\n", "\\n"),  # Escape newlines
                        "category": attack.category.value,
                        "severity": attack.severity.value,
                        "sophistication": attack.sophistication,
                        "target_models": ";".join(attack.target_models),
                        "is_verified": attack.is_verified,
                        "source": attack.metadata.source,
                        "creation_date": attack.metadata.creation_date.isoformat(),
                        "tags": ";".join(sorted(attack.metadata.tags)),
                        "effectiveness_score": attack.metadata.effectiveness_score or "",
                        "success_rate": attack.metadata.success_rate or "",
                    }
                )

    def _export_xml(self, data: Dict[str, Any], output_path: Path):
        """Export to XML format."""
        root = ET.Element("attack_library")

        # Add metadata
        metadata_elem = ET.SubElement(root, "metadata")
        for key, value in data["metadata"].items():
            elem = ET.SubElement(metadata_elem, key)
            elem.text = str(value)

        # Add attacks
        attacks_elem = ET.SubElement(root, "attacks")

        for attack_data in data["attacks"]:
            attack_elem = ET.SubElement(attacks_elem, "attack")
            attack_elem.set("id", attack_data["id"])

            for key, value in attack_data.items():
                if key == "id":
                    continue

                elem = ET.SubElement(attack_elem, key)

                if isinstance(value, list):
                    for item in value:
                        item_elem = ET.SubElement(elem, "item")
                        item_elem.text = str(item)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        sub_elem = ET.SubElement(elem, sub_key)
                        sub_elem.text = str(sub_value)
                else:
                    elem.text = str(value)

        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def _export_yaml(self, data: Dict[str, Any], output_path: Path):
        """Export to YAML format."""
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _export_markdown(self, attacks: List[Attack], output_path: Path):
        """Export to Markdown format."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Attack Library Export\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Attacks: {len(attacks)}\n\n")

            # Group by category
            by_category = {}
            for attack in attacks:
                category = attack.category.value
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(attack)

            for category, category_attacks in by_category.items():
                f.write(f"## {category.title()} Attacks ({len(category_attacks)})\n\n")

                for attack in category_attacks:
                    f.write(f"### {attack.title}\n\n")
                    f.write(f"**ID:** `{attack.id}`\n")
                    f.write(f"**Severity:** {attack.severity.value}\n")
                    f.write(f"**Sophistication:** {attack.sophistication}/5\n")

                    if attack.target_models:
                        f.write(f"**Target Models:** {', '.join(attack.target_models)}\n")

                    if attack.metadata.tags:
                        f.write(f"**Tags:** {', '.join(sorted(attack.metadata.tags))}\n")

                    f.write(f"**Verified:** {'Yes' if attack.is_verified else 'No'}\n\n")

                    f.write("**Content:**\n")
                    f.write("```\n")
                    f.write(attack.content)
                    f.write("\n```\n\n")

                    f.write("---\n\n")

    def _export_txt(self, attacks: List[Attack], output_path: Path):
        """Export to plain text format."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("ATTACK LIBRARY EXPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Attacks: {len(attacks)}\n\n")

            for i, attack in enumerate(attacks, 1):
                f.write(f"[{i:03d}] {attack.title}\n")
                f.write("-" * 50 + "\n")
                f.write(f"ID: {attack.id}\n")
                f.write(f"Category: {attack.category.value}\n")
                f.write(f"Severity: {attack.severity.value}\n")
                f.write(f"Sophistication: {attack.sophistication}/5\n")
                f.write(f"Verified: {'Yes' if attack.is_verified else 'No'}\n")

                if attack.target_models:
                    f.write(f"Target Models: {', '.join(attack.target_models)}\n")

                if attack.metadata.tags:
                    f.write(f"Tags: {', '.join(sorted(attack.metadata.tags))}\n")

                f.write(f"Source: {attack.metadata.source}\n")
                f.write(f"Created: {attack.metadata.creation_date.strftime('%Y-%m-%d')}\n")
                f.write("\nContent:\n")
                f.write(attack.content)
                f.write("\n\n" + "=" * 50 + "\n\n")

    def _export_jsonl(self, attacks: List[Attack], output_path: Path):
        """Export to JSON Lines format (one JSON object per line)."""
        with open(output_path, "w", encoding="utf-8") as f:
            for attack in attacks:
                attack_data = {
                    "id": attack.id,
                    "title": attack.title,
                    "content": attack.content,
                    "category": attack.category.value,
                    "severity": attack.severity.value,
                    "sophistication": attack.sophistication,
                    "target_models": attack.target_models,
                    "is_verified": attack.is_verified,
                    "source": attack.metadata.source,
                    "creation_date": attack.metadata.creation_date.isoformat(),
                    "tags": list(attack.metadata.tags),
                    "effectiveness_score": attack.metadata.effectiveness_score,
                    "success_rate": attack.metadata.success_rate,
                }
                f.write(json.dumps(attack_data, ensure_ascii=False, default=str) + "\n")

    def _export_excel(self, attacks: List[Attack], output_path: Path):
        """Export to Excel format."""
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is required for Excel export")
            raise ImportError(
                "pandas is required for Excel export. Install with: pip install pandas openpyxl"
            )

        # Prepare data for DataFrame
        data_rows = []
        for attack in attacks:
            row = {
                "ID": attack.id,
                "Title": attack.title,
                "Content": attack.content,
                "Category": attack.category.value,
                "Severity": attack.severity.value,
                "Sophistication": attack.sophistication,
                "Target Models": "; ".join(attack.target_models),
                "Verified": attack.is_verified,
                "Source": attack.metadata.source,
                "Creation Date": attack.metadata.creation_date.strftime("%Y-%m-%d"),
                "Tags": "; ".join(sorted(attack.metadata.tags)),
                "Effectiveness Score": attack.metadata.effectiveness_score or "",
                "Success Rate": attack.metadata.success_rate or "",
                "Language": attack.metadata.language,
                "References": "; ".join(attack.metadata.references)
                if attack.metadata.references
                else "",
            }
            data_rows.append(row)

        # Create DataFrame and export
        df = pd.DataFrame(data_rows)

        # Use ExcelWriter for better formatting
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Attacks", index=False)

            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets["Attacks"]

            # Auto-adjust column widths
            for col in worksheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                worksheet.column_dimensions[column].width = adjusted_width

    def export_effectiveness_data(
        self, output_path: Path, format_type: str = "json", filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Export effectiveness tracking data."""
        if not self.effectiveness_tracker:
            raise ValueError("No effectiveness tracker available")

        self.effectiveness_tracker.export_effectiveness_data(output_path, format_type, filters)

        return {
            "status": "success",
            "format": format_type,
            "output_path": str(output_path),
            "export_timestamp": datetime.now().isoformat(),
        }

    def export_tag_system(self, output_path: Path, format_type: str = "json") -> Dict[str, Any]:
        """Export tagging system data."""
        if not self.tagging_system:
            raise ValueError("No tagging system available")

        self.tagging_system.export_tag_system(output_path, format_type)

        return {
            "status": "success",
            "format": format_type,
            "output_path": str(output_path),
            "export_timestamp": datetime.now().isoformat(),
        }

    def export_complete_dataset(
        self, output_dir: Path, formats: List[str] = None, include_analytics: bool = True
    ) -> Dict[str, Any]:
        """Export complete dataset in multiple formats."""
        if formats is None:
            formats = ["json", "csv", "markdown"]

        output_dir.mkdir(exist_ok=True)
        export_results = {}

        # Export attacks in each format
        for fmt in formats:
            if fmt in self.SUPPORTED_FORMATS:
                output_path = output_dir / f"attacks.{fmt}"
                try:
                    result = self.export_attacks(
                        output_path, fmt, include_analytics=include_analytics
                    )
                    export_results[fmt] = result
                except Exception as e:
                    logger.error(f"Failed to export in {fmt} format: {e}")
                    export_results[fmt] = {"status": "error", "error": str(e)}

        # Export effectiveness data if available
        if self.effectiveness_tracker:
            try:
                effectiveness_path = output_dir / "effectiveness_data.json"
                self.export_effectiveness_data(effectiveness_path, "json")
                export_results["effectiveness"] = {
                    "status": "success",
                    "path": str(effectiveness_path),
                }
            except Exception as e:
                logger.error(f"Failed to export effectiveness data: {e}")
                export_results["effectiveness"] = {"status": "error", "error": str(e)}

        # Export tag system if available
        if self.tagging_system:
            try:
                tags_path = output_dir / "tag_system.json"
                self.export_tag_system(tags_path, "json")
                export_results["tag_system"] = {"status": "success", "path": str(tags_path)}
            except Exception as e:
                logger.error(f"Failed to export tag system: {e}")
                export_results["tag_system"] = {"status": "error", "error": str(e)}

        # Export analytics dashboard if available
        if include_analytics and self.analytics_engine:
            try:
                analytics_dir = output_dir / "analytics"
                dashboard_result = self.analytics_engine.export_analytics_dashboard(analytics_dir)
                export_results["analytics"] = dashboard_result
            except Exception as e:
                logger.error(f"Failed to export analytics: {e}")
                export_results["analytics"] = {"status": "error", "error": str(e)}

        # Create export summary
        summary = {
            "export_timestamp": datetime.now().isoformat(),
            "output_directory": str(output_dir),
            "formats_exported": formats,
            "total_files_created": sum(
                1
                for result in export_results.values()
                if isinstance(result, dict) and result.get("status") == "success"
            ),
            "export_results": export_results,
        }

        # Save export summary
        with open(output_dir / "export_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Complete dataset export finished. Results in {output_dir}")

        return summary

    def get_export_statistics(self) -> Dict[str, Any]:
        """Get statistics about exportable data."""
        total_attacks = len(self.attack_library.attacks)

        stats = {
            "total_attacks": total_attacks,
            "supported_formats": self.SUPPORTED_FORMATS,
            "library_statistics": self.attack_library.get_statistics() if total_attacks > 0 else {},
            "data_sources": {
                "attack_library": True,
                "effectiveness_tracker": self.effectiveness_tracker is not None,
                "tagging_system": self.tagging_system is not None,
                "analytics_engine": self.analytics_engine is not None,
            },
        }

        if self.effectiveness_tracker:
            stats["effectiveness_statistics"] = self.effectiveness_tracker.get_statistics()

        if self.tagging_system:
            stats["tagging_statistics"] = self.tagging_system.get_tag_statistics()

        return stats
