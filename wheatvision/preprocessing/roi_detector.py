"""Region of Interest (ROI) detection for wheat plant images."""

from typing import List, Optional

import cv2
import numpy as np

from wheatvision.config.models import BoundingBox
from wheatvision.config.settings import PreprocessingSettings, get_preprocessing_settings
from wheatvision.preprocessing.base_preprocessor import BasePreprocessor


class ROIDetector(BasePreprocessor):
    """
    Detects the region of interest containing wheat plants.
    
    Uses contour analysis on a foreground mask to find the bounding
    region of the wheat plants. This helps focus segmentation on
    relevant areas and exclude empty background regions.
    """

    def __init__(self, settings: PreprocessingSettings | None = None) -> None:
        """
        Initialize the ROI detector.
        
        Args:
            settings: Preprocessing settings. If None, loads from environment.
        """
        self._settings = settings or get_preprocessing_settings()
        self._min_area_ratio = self._settings.roi_min_area_ratio

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process is not typically used directly for ROI detection.
        
        For ROI detection, use detect_roi() with a foreground mask instead.
        This method returns the image unchanged.
        
        Args:
            image: Input image.
            
        Returns:
            Unchanged image.
        """
        return image

    def detect_roi(self, foreground_mask: np.ndarray) -> Optional[BoundingBox]:
        """
        Find the bounding box containing all foreground content.
        
        Args:
            foreground_mask: Binary mask where 255 = foreground.
            
        Returns:
            BoundingBox of the detected ROI, or None if no ROI found.
        """
        contours, _ = cv2.findContours(
            foreground_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        image_area = foreground_mask.shape[0] * foreground_mask.shape[1]
        min_area = image_area * self._min_area_ratio

        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not valid_contours:
            return None

        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        return BoundingBox(
            x_min=x,
            y_min=y,
            x_max=x + w,
            y_max=y + h,
        )

    def detect_multiple_rois(
        self,
        foreground_mask: np.ndarray,
        max_rois: int = 10,
    ) -> List[BoundingBox]:
        """
        Find multiple distinct regions of interest.
        
        Useful when multiple wheat plants or plant groups are present
        in the image and should be processed separately.
        
        Args:
            foreground_mask: Binary mask where 255 = foreground.
            max_rois: Maximum number of ROIs to return.
            
        Returns:
            List of BoundingBox objects, sorted by area (largest first).
        """
        contours, _ = cv2.findContours(
            foreground_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return []

        image_area = foreground_mask.shape[0] * foreground_mask.shape[1]
        min_area = image_area * self._min_area_ratio

        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        valid_contours.sort(key=cv2.contourArea, reverse=True)
        valid_contours = valid_contours[:max_rois]

        rois = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            rois.append(BoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h))

        return rois

    def expand_roi(
        self,
        roi: BoundingBox,
        image_shape: tuple[int, int],
        padding_ratio: float = 0.1,
    ) -> BoundingBox:
        """
        Expand an ROI by a percentage to add margin.
        
        Args:
            roi: Original bounding box.
            image_shape: (height, width) of the image.
            padding_ratio: Percentage to expand (0.1 = 10% on each side).
            
        Returns:
            Expanded bounding box, clipped to image bounds.
        """
        height, width = image_shape[:2]

        pad_x = int(roi.width * padding_ratio)
        pad_y = int(roi.height * padding_ratio)

        return BoundingBox(
            x_min=max(0, roi.x_min - pad_x),
            y_min=max(0, roi.y_min - pad_y),
            x_max=min(width, roi.x_max + pad_x),
            y_max=min(height, roi.y_max + pad_y),
        )

    def crop_to_roi(self, image: np.ndarray, roi: BoundingBox) -> np.ndarray:
        """
        Crop an image to the specified ROI.
        
        Args:
            image: Input image.
            roi: Bounding box to crop to.
            
        Returns:
            Cropped image.
        """
        return image[roi.y_min : roi.y_max, roi.x_min : roi.x_max]

    def get_visualization(
        self,
        image: np.ndarray,
        roi: Optional[BoundingBox],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3,
    ) -> np.ndarray:
        """
        Create a visualization showing the detected ROI.
        
        Args:
            image: Original image.
            roi: Detected bounding box.
            color: RGB color for the rectangle.
            thickness: Line thickness in pixels.
            
        Returns:
            Image with ROI rectangle drawn.
        """
        vis = image.copy()
        if roi is not None:
            cv2.rectangle(
                vis,
                (roi.x_min, roi.y_min),
                (roi.x_max, roi.y_max),
                color,
                thickness,
            )
        return vis

    def get_name(self) -> str:
        """Get the preprocessor name."""
        return "ROI Detector"
