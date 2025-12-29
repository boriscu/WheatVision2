"""Accuracy metrics for evaluating segmentation quality."""

from typing import List, Optional

import numpy as np

from wheatvision.config.models import AccuracyMetricsReport, SegmentationResult
from wheatvision.metrics.base_metric import BaseMetric


class AccuracyMetrics(BaseMetric):
    """
    Calculates accuracy-related metrics for segmentation quality.
    
    Since we don't have ground truth labels, these metrics focus on:
    - Temporal consistency (how stable are masks across frames)
    - Detection stability (mask count variance)
    - Coverage ratio (how much of the ROI is covered)
    """

    def calculate(
        self,
        results: List[SegmentationResult],
        model_name: str = "Unknown",
        roi_area: Optional[int] = None,
    ) -> AccuracyMetricsReport:
        """
        Calculate accuracy metrics from segmentation results.
        
        Args:
            results: List of segmentation results.
            model_name: Name of the model being evaluated.
            roi_area: Optional ROI area for coverage calculation.
            
        Returns:
            AccuracyMetricsReport with computed metrics.
        """
        if not results:
            return AccuracyMetricsReport(
                model_name=model_name,
                total_frames=0,
                avg_masks_per_frame=0.0,
                mask_count_std=0.0,
                temporal_consistency_score=0.0,
                coverage_ratio=0.0,
            )

        mask_counts = [len(r.masks) for r in results]
        avg_masks = float(np.mean(mask_counts))
        mask_std = float(np.std(mask_counts))

        temporal_consistency = self._calculate_temporal_consistency(results)

        coverage = self._calculate_coverage(results, roi_area)

        return AccuracyMetricsReport(
            model_name=model_name,
            total_frames=len(results),
            avg_masks_per_frame=avg_masks,
            mask_count_std=mask_std,
            temporal_consistency_score=temporal_consistency,
            coverage_ratio=coverage,
        )

    def _calculate_temporal_consistency(
        self,
        results: List[SegmentationResult],
    ) -> float:
        """
        Calculate temporal consistency between consecutive frames.
        
        Measures how much the combined mask overlaps between frames.
        Higher values indicate more stable segmentation.
        
        Args:
            results: List of segmentation results.
            
        Returns:
            Average IoU between consecutive frames (0-1).
        """
        if len(results) < 2:
            return 1.0

        ious = []

        for i in range(len(results) - 1):
            mask1 = self._get_combined_mask(results[i])
            mask2 = self._get_combined_mask(results[i + 1])

            if mask1 is None or mask2 is None:
                continue

            if mask1.shape != mask2.shape:
                continue

            iou = self._calculate_iou(mask1, mask2)
            ious.append(iou)

        return float(np.mean(ious)) if ious else 0.0

    def _get_combined_mask(
        self,
        result: SegmentationResult,
    ) -> Optional[np.ndarray]:
        """
        Combine all masks in a result into a single binary mask.
        
        Args:
            result: Segmentation result.
            
        Returns:
            Combined binary mask, or None if no masks.
        """
        if not result.masks:
            return None

        combined = np.zeros_like(result.masks[0], dtype=np.uint8)
        for mask in result.masks:
            combined = np.maximum(combined, (mask > 0).astype(np.uint8))

        return combined

    def _calculate_iou(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> float:
        """
        Calculate Intersection over Union between two masks.
        
        Args:
            mask1: First binary mask.
            mask2: Second binary mask.
            
        Returns:
            IoU value (0-1).
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        return float(intersection / union) if union > 0 else 0.0

    def _calculate_coverage(
        self,
        results: List[SegmentationResult],
        roi_area: Optional[int],
    ) -> float:
        """
        Calculate average coverage of masks relative to ROI.
        
        Args:
            results: List of segmentation results.
            roi_area: Area of the region of interest.
            
        Returns:
            Average coverage ratio (0-1).
        """
        if roi_area is None or roi_area == 0:
            return 0.0

        coverages = []

        for result in results:
            combined = self._get_combined_mask(result)
            if combined is not None:
                mask_area = (combined > 0).sum()
                coverages.append(float(mask_area) / roi_area)
            else:
                coverages.append(0.0)

        return float(np.mean(coverages))

    def get_comparison(
        self,
        report1: AccuracyMetricsReport,
        report2: AccuracyMetricsReport,
    ) -> dict:
        """
        Compare accuracy metrics between two models.
        
        Args:
            report1: First model's accuracy metrics.
            report2: Second model's accuracy metrics.
            
        Returns:
            Dictionary with comparison statistics.
        """
        consistency_diff = (
            report1.temporal_consistency_score - report2.temporal_consistency_score
        )
        more_consistent = (
            report1.model_name
            if report1.temporal_consistency_score > report2.temporal_consistency_score
            else report2.model_name
        )

        stability_ratio = (
            report1.mask_count_std / report2.mask_count_std
            if report2.mask_count_std > 0
            else float("inf")
        )

        return {
            "model1_name": report1.model_name,
            "model2_name": report2.model_name,
            "consistency_difference": consistency_diff,
            "more_consistent_model": more_consistent,
            "stability_ratio": stability_ratio,
            "coverage_difference": report1.coverage_ratio - report2.coverage_ratio,
        }

    def get_name(self) -> str:
        """Get the metric name."""
        return "Accuracy Metrics"
