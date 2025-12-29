"""Speed metrics for measuring segmentation performance."""

from typing import List

import numpy as np

from wheatvision.config.models import SegmentationResult, SpeedMetricsReport
from wheatvision.metrics.base_metric import BaseMetric


class SpeedMetrics(BaseMetric):
    """
    Calculates speed-related metrics for segmentation performance.
    
    Measures:
    - Total processing time
    - Frames per second (FPS)
    - Per-frame timing statistics (avg, min, max)
    """

    def calculate(
        self,
        results: List[SegmentationResult],
        model_name: str = "Unknown",
        model_load_time_ms: float = 0.0,
    ) -> SpeedMetricsReport:
        """
        Calculate speed metrics from segmentation results.
        
        Args:
            results: List of segmentation results with timing info.
            model_name: Name of the model being evaluated.
            model_load_time_ms: Time taken to load the model.
            
        Returns:
            SpeedMetricsReport with computed metrics.
        """
        if not results:
            return SpeedMetricsReport(
                model_name=model_name,
                total_frames=0,
                model_load_time_ms=model_load_time_ms,
                total_processing_time_ms=0.0,
                fps=0.0,
                avg_time_per_frame_ms=0.0,
                min_time_per_frame_ms=0.0,
                max_time_per_frame_ms=0.0,
            )

        frame_times = [r.processing_time_ms for r in results]
        total_time = sum(frame_times)
        total_frames = len(results)

        fps = (total_frames / total_time) * 1000 if total_time > 0 else 0.0

        return SpeedMetricsReport(
            model_name=model_name,
            total_frames=total_frames,
            model_load_time_ms=model_load_time_ms,
            total_processing_time_ms=total_time,
            fps=fps,
            avg_time_per_frame_ms=float(np.mean(frame_times)),
            min_time_per_frame_ms=float(np.min(frame_times)),
            max_time_per_frame_ms=float(np.max(frame_times)),
        )

    def get_comparison(
        self,
        report1: SpeedMetricsReport,
        report2: SpeedMetricsReport,
    ) -> dict:
        """
        Compare speed metrics between two models.
        
        Args:
            report1: First model's speed metrics.
            report2: Second model's speed metrics.
            
        Returns:
            Dictionary with comparison statistics.
        """
        fps_ratio = report1.fps / report2.fps if report2.fps > 0 else float("inf")
        time_ratio = (
            report1.avg_time_per_frame_ms / report2.avg_time_per_frame_ms
            if report2.avg_time_per_frame_ms > 0
            else float("inf")
        )

        return {
            "model1_name": report1.model_name,
            "model2_name": report2.model_name,
            "fps_ratio": fps_ratio,
            "fps_difference": report1.fps - report2.fps,
            "time_ratio": time_ratio,
            "faster_model": report1.model_name if report1.fps > report2.fps else report2.model_name,
            "speedup_factor": max(fps_ratio, 1 / fps_ratio) if fps_ratio > 0 else 0.0,
        }

    def get_name(self) -> str:
        """Get the metric name."""
        return "Speed Metrics"
