"""Wheat ear filter for removing non-ear segmentation results."""

from typing import List, Tuple

import cv2
import numpy as np

from wheatvision.config.models import BoundingBox, MaskProperties
from wheatvision.config.settings import PostprocessingSettings, get_postprocessing_settings
from wheatvision.postprocessing.base_postprocessor import BasePostprocessor


class WheatEarFilter(BasePostprocessor):
    """
    Filters segmentation masks based on wheat ear characteristics.
    
    Wheat ears have distinctive geometric properties:
    - Elongated shape (aspect ratio typically 2-10)
    - Specific size relative to image (not too small, not too large)
    - High solidity (compact, without holes)
    
    This filter removes masks that don't match these characteristics.
    """

    def __init__(self, settings: PostprocessingSettings | None = None) -> None:
        """
        Initialize the wheat ear filter.
        
        Args:
            settings: Postprocessing settings. If None, loads from environment.
        """
        self._settings = settings or get_postprocessing_settings()

    def process(
        self,
        masks: List[np.ndarray],
        scores: List[float],
        image_shape: tuple[int, int],
    ) -> tuple[List[np.ndarray], List[float]]:
        """
        Filter masks to keep only those matching wheat ear characteristics.
        
        Args:
            masks: List of binary masks.
            scores: Corresponding confidence scores.
            image_shape: (height, width) of the image.
            
        Returns:
            Tuple of (filtered_masks, filtered_scores).
        """
        filtered_masks = []
        filtered_scores = []

        image_area = image_shape[0] * image_shape[1]

        for mask, score in zip(masks, scores):
            properties = self._compute_mask_properties(mask)

            if properties is None:
                continue

            if self._is_valid_wheat_ear(properties, image_area):
                filtered_masks.append(mask)
                filtered_scores.append(score)

        return filtered_masks, filtered_scores

    def _compute_mask_properties(self, mask: np.ndarray) -> MaskProperties | None:
        """
        Extract geometric properties from a binary mask.
        
        Args:
            mask: Binary mask array.
            
        Returns:
            MaskProperties object, or None if mask is empty/invalid.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 10:
            return None

        perimeter = cv2.arcLength(largest_contour, closed=True)

        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = max(w, h) / max(min(w, h), 1)

        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)

        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = x + w / 2, y + h / 2

        return MaskProperties(
            area=int(area),
            perimeter=float(perimeter),
            aspect_ratio=float(aspect_ratio),
            solidity=float(solidity),
            bounding_box=BoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h),
            centroid=(float(cx), float(cy)),
        )

    def _is_valid_wheat_ear(
        self,
        properties: MaskProperties,
        image_area: int,
    ) -> bool:
        """
        Check if mask properties match wheat ear characteristics.
        
        Args:
            properties: Computed mask properties.
            image_area: Total image area in pixels.
            
        Returns:
            True if the mask likely represents a wheat ear.
        """
        min_aspect = self._settings.min_aspect
        max_aspect = self._settings.max_aspect
        if not (min_aspect <= properties.aspect_ratio <= max_aspect):
            return False

        area_ratio = properties.area / image_area
        if not (self._settings.min_area_ratio <= area_ratio <= self._settings.max_area_ratio):
            return False

        if properties.solidity < 0.5:
            return False

        return True

    def filter_by_custom_criteria(
        self,
        masks: List[np.ndarray],
        scores: List[float],
        min_aspect: float,
        max_aspect: float,
        min_area: int,
        max_area: int,
    ) -> tuple[List[np.ndarray], List[float]]:
        """
        Filter masks using custom criteria instead of settings.
        
        Args:
            masks: List of binary masks.
            scores: Corresponding confidence scores.
            min_aspect: Minimum aspect ratio.
            max_aspect: Maximum aspect ratio.
            min_area: Minimum mask area in pixels.
            max_area: Maximum mask area in pixels.
            
        Returns:
            Tuple of (filtered_masks, filtered_scores).
        """
        filtered_masks = []
        filtered_scores = []

        for mask, score in zip(masks, scores):
            properties = self._compute_mask_properties(mask)

            if properties is None:
                continue

            if not (min_aspect <= properties.aspect_ratio <= max_aspect):
                continue

            if not (min_area <= properties.area <= max_area):
                continue

            filtered_masks.append(mask)
            filtered_scores.append(score)

        return filtered_masks, filtered_scores

    def get_mask_properties_batch(
        self,
        masks: List[np.ndarray],
    ) -> List[MaskProperties | None]:
        """
        Compute properties for all masks (for analysis/visualization).
        
        Args:
            masks: List of binary masks.
            
        Returns:
            List of MaskProperties (None for invalid masks).
        """
        return [self._compute_mask_properties(mask) for mask in masks]

    def get_name(self) -> str:
        """Get the postprocessor name."""
        return "Wheat Ear Filter"
