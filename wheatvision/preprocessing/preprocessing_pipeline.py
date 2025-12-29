"""Preprocessing pipeline combining multiple preprocessing steps."""

from typing import List, Optional

import numpy as np

from wheatvision.config.models import PreprocessingResult, BoundingBox
from wheatvision.config.settings import PreprocessingSettings, get_preprocessing_settings
from wheatvision.preprocessing.background_remover import BackgroundRemover
from wheatvision.preprocessing.roi_detector import ROIDetector


class PreprocessingPipeline:
    """
    Combines preprocessing steps into a unified pipeline.
    
    Runs background removal and ROI detection in sequence, providing
    all intermediate results for visualization and debugging.
    """

    def __init__(self, settings: PreprocessingSettings | None = None) -> None:
        """
        Initialize the preprocessing pipeline.
        
        Args:
            settings: Preprocessing settings. If None, loads from environment.
        """
        self._settings = settings or get_preprocessing_settings()
        self._background_remover = BackgroundRemover(self._settings)
        self._roi_detector = ROIDetector(self._settings)
        self._roi_padding_ratio = 0.05

    def process(
        self,
        image: np.ndarray,
        expand_roi: bool = True,
    ) -> PreprocessingResult:
        """
        Run the full preprocessing pipeline on an image.
        
        Steps:
        1. Background removal (creates foreground mask)
        2. ROI detection (finds bounding box of plant region)
        3. Optional ROI expansion for margin
        
        Args:
            image: Input RGB image.
            expand_roi: Whether to add padding to the detected ROI.
            
        Returns:
            PreprocessingResult containing original image, processed image,
            foreground mask, and detected ROI.
        """
        foreground_mask = self._background_remover.process(image)

        roi = self._roi_detector.detect_roi(foreground_mask)

        if roi is not None and expand_roi:
            roi = self._roi_detector.expand_roi(
                roi,
                image.shape,
                self._roi_padding_ratio,
            )

        processed_frame = self._background_remover.apply_mask(image, foreground_mask)

        return PreprocessingResult(
            original_frame=image,
            processed_frame=processed_frame,
            foreground_mask=foreground_mask,
            roi_bbox=roi,
        )

    def process_batch(
        self,
        images: List[np.ndarray],
        expand_roi: bool = True,
    ) -> List[PreprocessingResult]:
        """
        Process multiple images through the pipeline.
        
        Args:
            images: List of RGB images.
            expand_roi: Whether to add padding to detected ROIs.
            
        Returns:
            List of PreprocessingResult for each image.
        """
        return [self.process(image, expand_roi) for image in images]

    def get_combined_roi(
        self,
        results: List[PreprocessingResult],
    ) -> Optional[BoundingBox]:
        """
        Compute a single ROI that encompasses all detected ROIs.
        
        Useful for video processing where we want a consistent ROI
        across all frames.
        
        Args:
            results: List of preprocessing results.
            
        Returns:
            Combined bounding box, or None if no ROIs were detected.
        """
        valid_rois = [r.roi_bbox for r in results if r.roi_bbox is not None]

        if not valid_rois:
            return None

        x_min = min(roi.x_min for roi in valid_rois)
        y_min = min(roi.y_min for roi in valid_rois)
        x_max = max(roi.x_max for roi in valid_rois)
        y_max = max(roi.y_max for roi in valid_rois)

        return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def get_visualization(
        self,
        result: PreprocessingResult,
    ) -> np.ndarray:
        """
        Create a multi-panel visualization of preprocessing results.
        
        Layout:
        [Original] [Foreground Mask] [Processed with ROI]
        
        Args:
            result: Preprocessing result to visualize.
            
        Returns:
            Combined visualization image.
        """
        import cv2

        original = result.original_frame
        mask_rgb = cv2.cvtColor(result.foreground_mask, cv2.COLOR_GRAY2RGB)

        processed_with_roi = self._roi_detector.get_visualization(
            result.processed_frame,
            result.roi_bbox,
        )

        return np.hstack([original, mask_rgb, processed_with_roi])

    def set_roi_padding(self, padding_ratio: float) -> None:
        """
        Update the ROI padding ratio.
        
        Args:
            padding_ratio: New padding ratio (0.0-1.0).
        """
        self._roi_padding_ratio = padding_ratio

    def update_background_hsv(
        self,
        hsv_low: tuple[int, int, int],
        hsv_high: tuple[int, int, int],
    ) -> None:
        """
        Update background HSV detection range.
        
        Args:
            hsv_low: Lower HSV bound.
            hsv_high: Upper HSV bound.
        """
        self._background_remover.update_hsv_range(hsv_low, hsv_high)
