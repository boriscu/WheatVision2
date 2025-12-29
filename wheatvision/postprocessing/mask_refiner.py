"""Mask refinement for improving segmentation quality."""

from typing import List

import cv2
import numpy as np

from wheatvision.postprocessing.base_postprocessor import BasePostprocessor


class MaskRefiner(BasePostprocessor):
    """
    Refines segmentation masks through morphological operations.
    
    Cleans up mask boundaries, fills small holes, and removes
    small disconnected regions.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        min_component_ratio: float = 0.1,
    ) -> None:
        """
        Initialize the mask refiner.
        
        Args:
            kernel_size: Size of morphological kernel.
            min_component_ratio: Minimum connected component size relative
                                 to largest component (smaller ones removed).
        """
        self._kernel_size = kernel_size
        self._min_component_ratio = min_component_ratio
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

    def process(
        self,
        masks: List[np.ndarray],
        scores: List[float],
        image_shape: tuple[int, int],
    ) -> tuple[List[np.ndarray], List[float]]:
        """
        Refine all masks in the list.
        
        Args:
            masks: List of binary masks.
            scores: Corresponding confidence scores.
            image_shape: (height, width) of the image.
            
        Returns:
            Tuple of (refined_masks, scores) - scores unchanged.
        """
        refined_masks = []

        for mask in masks:
            refined = self.refine_mask(mask)
            if refined.any():
                refined_masks.append(refined)

        valid_scores = scores[: len(refined_masks)]

        return refined_masks, list(valid_scores)

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply refinement operations to a single mask.
        
        Steps:
        1. Morphological closing (fill small holes)
        2. Morphological opening (remove small protrusions)
        3. Remove small connected components
        
        Args:
            mask: Binary mask array.
            
        Returns:
            Refined binary mask.
        """
        mask_uint8 = mask.astype(np.uint8)

        refined = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, self._kernel)

        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, self._kernel)

        refined = self._remove_small_components(refined)

        return refined

    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove connected components smaller than threshold.
        
        Keeps only components that are at least min_component_ratio
        of the size of the largest component.
        
        Args:
            mask: Binary mask.
            
        Returns:
            Mask with small components removed.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        if num_labels <= 1:
            return mask

        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = areas.max()
        min_area = max_area * self._min_component_ratio

        result = np.zeros_like(mask)

        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
                result[labels == label_id] = 255

        return result

    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill all holes in a mask.
        
        Args:
            mask: Binary mask.
            
        Returns:
            Mask with holes filled.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

        return filled

    def smooth_boundaries(
        self,
        mask: np.ndarray,
        sigma: float = 2.0,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Smooth mask boundaries using Gaussian blur.
        
        Args:
            mask: Binary mask.
            sigma: Gaussian blur sigma.
            threshold: Threshold for re-binarizing after blur.
            
        Returns:
            Mask with smoothed boundaries.
        """
        kernel_size = int(sigma * 4) | 1

        blurred = cv2.GaussianBlur(
            mask.astype(np.float32),
            (kernel_size, kernel_size),
            sigma,
        )

        smoothed = (blurred > threshold * 255).astype(np.uint8) * 255

        return smoothed

    def get_name(self) -> str:
        """Get the postprocessor name."""
        return "Mask Refiner"
