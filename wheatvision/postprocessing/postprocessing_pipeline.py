"""Postprocessing pipeline combining filtering and refinement."""

from typing import List

import numpy as np

from wheatvision.config.models import SegmentationResult
from wheatvision.config.settings import PostprocessingSettings, get_postprocessing_settings
from wheatvision.postprocessing.wheat_ear_filter import WheatEarFilter
from wheatvision.postprocessing.mask_refiner import MaskRefiner


class PostprocessingPipeline:
    """
    Combines postprocessing steps into a unified pipeline.
    
    Applies mask refinement followed by wheat ear filtering
    to produce clean, relevant segmentation results.
    """

    def __init__(
        self,
        settings: PostprocessingSettings | None = None,
        enable_refinement: bool = True,
        enable_filtering: bool = True,
    ) -> None:
        """
        Initialize the postprocessing pipeline.
        
        Args:
            settings: Postprocessing settings. If None, loads from environment.
            enable_refinement: Whether to apply mask refinement.
            enable_filtering: Whether to apply wheat ear filtering.
        """
        self._settings = settings or get_postprocessing_settings()
        self._mask_refiner = MaskRefiner()
        self._wheat_ear_filter = WheatEarFilter(self._settings)
        self._enable_refinement = enable_refinement
        self._enable_filtering = enable_filtering

    def process_result(
        self,
        result: SegmentationResult,
        image_shape: tuple[int, int],
    ) -> SegmentationResult:
        """
        Process a single segmentation result.
        
        Args:
            result: Segmentation result with masks and scores.
            image_shape: (height, width) of the original image.
            
        Returns:
            New SegmentationResult with processed masks.
        """
        masks = result.masks
        scores = result.scores

        if self._enable_refinement:
            masks, scores = self._mask_refiner.process(masks, scores, image_shape)

        if self._enable_filtering:
            masks, scores = self._wheat_ear_filter.process(masks, scores, image_shape)

        return SegmentationResult(
            frame_index=result.frame_index,
            masks=masks,
            scores=scores,
            processing_time_ms=result.processing_time_ms,
        )

    def process_results(
        self,
        results: List[SegmentationResult],
        image_shape: tuple[int, int],
    ) -> List[SegmentationResult]:
        """
        Process multiple segmentation results.
        
        Args:
            results: List of segmentation results.
            image_shape: (height, width) of the frames.
            
        Returns:
            List of processed SegmentationResult objects.
        """
        return [self.process_result(r, image_shape) for r in results]

    def get_filter_statistics(
        self,
        original_results: List[SegmentationResult],
        processed_results: List[SegmentationResult],
    ) -> dict:
        """
        Calculate statistics about filtering effectiveness.
        
        Args:
            original_results: Results before postprocessing.
            processed_results: Results after postprocessing.
            
        Returns:
            Dictionary with filtering statistics.
        """
        original_mask_count = sum(len(r.masks) for r in original_results)
        processed_mask_count = sum(len(r.masks) for r in processed_results)

        if original_mask_count > 0:
            retention_rate = processed_mask_count / original_mask_count
        else:
            retention_rate = 0.0

        return {
            "original_mask_count": original_mask_count,
            "processed_mask_count": processed_mask_count,
            "masks_removed": original_mask_count - processed_mask_count,
            "retention_rate": retention_rate,
        }

    def set_refinement_enabled(self, enabled: bool) -> None:
        """Enable or disable mask refinement."""
        self._enable_refinement = enabled

    def set_filtering_enabled(self, enabled: bool) -> None:
        """Enable or disable wheat ear filtering."""
        self._enable_filtering = enabled

    def update_filter_settings(
        self,
        min_aspect: float | None = None,
        max_aspect: float | None = None,
        min_area_ratio: float | None = None,
        max_area_ratio: float | None = None,
    ) -> None:
        """
        Update wheat ear filter settings at runtime.
        
        Args:
            min_aspect: New minimum aspect ratio (None to keep current).
            max_aspect: New maximum aspect ratio (None to keep current).
            min_area_ratio: New minimum area ratio (None to keep current).
            max_area_ratio: New maximum area ratio (None to keep current).
        """
        if min_aspect is not None:
            self._settings.min_aspect = min_aspect
        if max_aspect is not None:
            self._settings.max_aspect = max_aspect
        if min_area_ratio is not None:
            self._settings.min_area_ratio = min_area_ratio
        if max_area_ratio is not None:
            self._settings.max_area_ratio = max_area_ratio

        self._wheat_ear_filter = WheatEarFilter(self._settings)
