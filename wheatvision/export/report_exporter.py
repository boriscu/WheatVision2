"""Report exporter for metrics and comparison reports."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from wheatvision.config.constants import ExportFormat
from wheatvision.config.models import MetricsReport
from wheatvision.export.base_exporter import BaseExporter


class ReportExporter(BaseExporter):
    """
    Exports metrics reports to JSON and CSV formats.
    
    Creates structured reports for analysis and archival.
    """

    def export(
        self,
        data: MetricsReport,
        output_path: Path,
        format: ExportFormat = ExportFormat.JSON,
    ) -> Path:
        """
        Export a single metrics report.
        
        Args:
            data: MetricsReport to export.
            output_path: Output file path.
            format: Export format (JSON or CSV).
            
        Returns:
            Path to the exported file.
        """
        output_path = Path(output_path)
        self._ensure_parent_exists(output_path)

        if format == ExportFormat.JSON:
            return self._export_as_json(data, output_path)
        elif format == ExportFormat.CSV:
            return self._export_as_csv(data, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_comparison(
        self,
        sam_report: MetricsReport,
        sam2_report: MetricsReport,
        comparison: Dict[str, Any],
        output_path: Path,
        format: ExportFormat = ExportFormat.JSON,
    ) -> Path:
        """
        Export a comparison report between SAM and SAM2.
        
        Args:
            sam_report: SAM metrics report.
            sam2_report: SAM2 metrics report.
            comparison: Comparison dictionary.
            output_path: Output file path.
            format: Export format.
            
        Returns:
            Path to the exported file.
        """
        output_path = Path(output_path)
        self._ensure_parent_exists(output_path)

        report_data = {
            "generated_at": datetime.now().isoformat(),
            "sam_metrics": self._report_to_dict(sam_report),
            "sam2_metrics": self._report_to_dict(sam2_report),
            "comparison": comparison,
        }

        if format == ExportFormat.JSON:
            output_path = output_path.with_suffix(".json")
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2)
        elif format == ExportFormat.CSV:
            output_path = output_path.with_suffix(".csv")
            df = self._comparison_to_dataframe(sam_report, sam2_report, comparison)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_path

    def _export_as_json(self, report: MetricsReport, path: Path) -> Path:
        """Export report as JSON."""
        output_path = path.with_suffix(".json")

        data = {
            "generated_at": datetime.now().isoformat(),
            "metrics": self._report_to_dict(report),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def _export_as_csv(self, report: MetricsReport, path: Path) -> Path:
        """Export report as CSV."""
        output_path = path.with_suffix(".csv")

        data = self._report_to_dict(report)
        flat_data = self._flatten_dict(data)

        df = pd.DataFrame([flat_data])
        df.to_csv(output_path, index=False)

        return output_path

    def _report_to_dict(self, report: MetricsReport) -> Dict[str, Any]:
        """Convert MetricsReport to dictionary."""
        return {
            "model_name": report.model_name,
            "speed": {
                "total_frames": report.speed_metrics.total_frames,
                "model_load_time_ms": report.speed_metrics.model_load_time_ms,
                "total_processing_time_ms": report.speed_metrics.total_processing_time_ms,
                "fps": report.speed_metrics.fps,
                "avg_time_per_frame_ms": report.speed_metrics.avg_time_per_frame_ms,
                "min_time_per_frame_ms": report.speed_metrics.min_time_per_frame_ms,
                "max_time_per_frame_ms": report.speed_metrics.max_time_per_frame_ms,
            },
            "accuracy": {
                "avg_masks_per_frame": report.accuracy_metrics.avg_masks_per_frame,
                "mask_count_std": report.accuracy_metrics.mask_count_std,
                "temporal_consistency_score": report.accuracy_metrics.temporal_consistency_score,
                "coverage_ratio": report.accuracy_metrics.coverage_ratio,
            },
        }

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "_",
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _comparison_to_dataframe(
        self,
        sam_report: MetricsReport,
        sam2_report: MetricsReport,
        comparison: Dict[str, Any],
    ) -> pd.DataFrame:
        """Convert comparison to DataFrame."""
        rows = []

        sam_dict = self._flatten_dict(self._report_to_dict(sam_report), "sam")
        sam2_dict = self._flatten_dict(self._report_to_dict(sam2_report), "sam2")

        combined = {**sam_dict, **sam2_dict}

        if "overall_summary" in comparison:
            for k, v in comparison["overall_summary"].items():
                combined[f"comparison_{k}"] = v

        rows.append(combined)

        return pd.DataFrame(rows)

    def get_supported_formats(self) -> List[str]:
        """Get supported formats."""
        return [ExportFormat.JSON.value, ExportFormat.CSV.value]
