"""Unified metrics calculator combining all metric types."""

from typing import List, Optional

from wheatvision.config.models import (
    AccuracyMetricsReport,
    MetricsReport,
    SegmentationResult,
    SpeedMetricsReport,
)
from wheatvision.metrics.speed_metrics import SpeedMetrics
from wheatvision.metrics.accuracy_metrics import AccuracyMetrics


class MetricsCalculator:
    """
    Unified calculator for all segmentation metrics.
    
    Combines speed and accuracy metrics into a single report
    for easy comparison between models.
    """

    def __init__(self) -> None:
        """Initialize the metrics calculator."""
        self._speed_metrics = SpeedMetrics()
        self._accuracy_metrics = AccuracyMetrics()

    def calculate_all(
        self,
        results: List[SegmentationResult],
        model_name: str,
        model_load_time_ms: float = 0.0,
        roi_area: Optional[int] = None,
    ) -> MetricsReport:
        """
        Calculate all metrics for a set of segmentation results.
        
        Args:
            results: List of segmentation results.
            model_name: Name of the model being evaluated.
            model_load_time_ms: Time to load the model.
            roi_area: Optional ROI area for coverage calculation.
            
        Returns:
            Complete MetricsReport with speed and accuracy metrics.
        """
        speed = self._speed_metrics.calculate(
            results,
            model_name=model_name,
            model_load_time_ms=model_load_time_ms,
        )

        accuracy = self._accuracy_metrics.calculate(
            results,
            model_name=model_name,
            roi_area=roi_area,
        )

        return MetricsReport(
            model_name=model_name,
            speed_metrics=speed,
            accuracy_metrics=accuracy,
        )

    def compare_models(
        self,
        sam_report: MetricsReport,
        sam2_report: MetricsReport,
    ) -> dict:
        """
        Generate a comparison report between SAM and SAM2.
        
        Args:
            sam_report: MetricsReport for SAM model.
            sam2_report: MetricsReport for SAM2 model.
            
        Returns:
            Dictionary with detailed comparison.
        """
        speed_comparison = self._speed_metrics.get_comparison(
            sam_report.speed_metrics,
            sam2_report.speed_metrics,
        )

        accuracy_comparison = self._accuracy_metrics.get_comparison(
            sam_report.accuracy_metrics,
            sam2_report.accuracy_metrics,
        )

        sam_total_time = sam_report.speed_metrics.total_processing_time_ms
        sam2_total_time = sam2_report.speed_metrics.total_processing_time_ms

        winner_speed = (
            sam_report.model_name if sam_total_time < sam2_total_time else sam2_report.model_name
        )

        sam_consistency = sam_report.accuracy_metrics.temporal_consistency_score
        sam2_consistency = sam2_report.accuracy_metrics.temporal_consistency_score

        winner_consistency = (
            sam_report.model_name if sam_consistency > sam2_consistency else sam2_report.model_name
        )

        return {
            "speed_comparison": speed_comparison,
            "accuracy_comparison": accuracy_comparison,
            "overall_summary": {
                "faster_model": winner_speed,
                "more_consistent_model": winner_consistency,
                "sam_fps": sam_report.speed_metrics.fps,
                "sam2_fps": sam2_report.speed_metrics.fps,
                "sam_consistency": sam_consistency,
                "sam2_consistency": sam2_consistency,
            },
        }

    def format_report_text(self, report: MetricsReport) -> str:
        """
        Format a metrics report as human-readable text.
        
        Args:
            report: MetricsReport to format.
            
        Returns:
            Formatted text string.
        """
        lines = [
            f"=== {report.model_name} Metrics ===",
            "",
            "Speed Metrics:",
            f"  Total Frames: {report.speed_metrics.total_frames}",
            f"  Total Time: {report.speed_metrics.total_processing_time_ms:.2f} ms",
            f"  FPS: {report.speed_metrics.fps:.2f}",
            f"  Avg Time/Frame: {report.speed_metrics.avg_time_per_frame_ms:.2f} ms",
            f"  Min Time/Frame: {report.speed_metrics.min_time_per_frame_ms:.2f} ms",
            f"  Max Time/Frame: {report.speed_metrics.max_time_per_frame_ms:.2f} ms",
            "",
            "Accuracy Metrics:",
            f"  Avg Masks/Frame: {report.accuracy_metrics.avg_masks_per_frame:.2f}",
            f"  Mask Count Std: {report.accuracy_metrics.mask_count_std:.2f}",
            f"  Temporal Consistency: {report.accuracy_metrics.temporal_consistency_score:.4f}",
            f"  Coverage Ratio: {report.accuracy_metrics.coverage_ratio:.4f}",
        ]
        return "\n".join(lines)

    def format_comparison_text(
        self,
        sam_report: MetricsReport,
        sam2_report: MetricsReport,
    ) -> str:
        """
        Format a comparison report as human-readable text.
        
        Args:
            sam_report: SAM metrics.
            sam2_report: SAM2 metrics.
            
        Returns:
            Formatted comparison text.
        """
        comparison = self.compare_models(sam_report, sam2_report)
        summary = comparison["overall_summary"]

        lines = [
            "=== SAM vs SAM2 Comparison ===",
            "",
            "Speed:",
            f"  SAM FPS: {summary['sam_fps']:.2f}",
            f"  SAM2 FPS: {summary['sam2_fps']:.2f}",
            f"  Faster Model: {summary['faster_model']}",
            "",
            "Consistency:",
            f"  SAM: {summary['sam_consistency']:.4f}",
            f"  SAM2: {summary['sam2_consistency']:.4f}",
            f"  More Consistent: {summary['more_consistent_model']}",
        ]
        return "\n".join(lines)
