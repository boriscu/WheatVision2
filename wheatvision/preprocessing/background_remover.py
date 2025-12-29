"""Background removal for white wall backgrounds."""

from typing import Tuple

import cv2
import numpy as np

from wheatvision.config.settings import PreprocessingSettings, get_preprocessing_settings
from wheatvision.preprocessing.base_preprocessor import BasePreprocessor


class BackgroundRemover(BasePreprocessor):
    """
    Removes white/light backgrounds from wheat plant images.
    
    Uses HSV color space thresholding to detect white wall backgrounds
    and creates a foreground mask. Detects pixels with low saturation
    and high value (white/gray) as background.
    """

    def __init__(self, settings: PreprocessingSettings | None = None) -> None:
        """
        Initialize the background remover.
        
        Args:
            settings: Preprocessing settings. If None, loads from environment.
        """
        self._settings = settings or get_preprocessing_settings()
        self._update_thresholds()
        
        self._open_kernel_size = 5
        self._close_kernel_size = 11
        self._open_iterations = 2
        self._close_iterations = 2

    def _update_thresholds(self) -> None:
        """Update HSV thresholds from settings."""
        hsv_low = self._settings.bg_hsv_low
        hsv_high = self._settings.bg_hsv_high
        
        if isinstance(hsv_low, str):
            hsv_low = tuple(int(x) for x in hsv_low.split(","))
        if isinstance(hsv_high, str):
            hsv_high = tuple(int(x) for x in hsv_high.split(","))
            
        self._hsv_low = np.array(hsv_low)
        self._hsv_high = np.array(hsv_high)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Detect background and create a foreground mask.
        
        The algorithm:
        1. Convert to HSV color space
        2. Detect white/gray pixels (low saturation, high value) as background
        3. Invert to get foreground mask
        4. Apply morphological operations to clean up
        
        Args:
            image: Input RGB image.
            
        Returns:
            Binary mask where 255 = foreground (plant), 0 = background.
        """
        if image.shape[2] == 3 and image.dtype == np.uint8:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr = image
            
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        background_mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)

        foreground_mask = cv2.bitwise_not(background_mask)

        foreground_mask = self._refine_mask(foreground_mask)

        return foreground_mask

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the mask.
        
        Removes small noise and fills small holes in the foreground mask.
        
        Args:
            mask: Binary foreground mask.
            
        Returns:
            Refined binary mask.
        """
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self._open_kernel_size, self._open_kernel_size)
        )
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self._close_kernel_size, self._close_kernel_size)
        )
        
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, open_kernel, 
            iterations=self._open_iterations
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, close_kernel, 
            iterations=self._close_iterations
        )

        return mask

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply a foreground mask to an image, zeroing out background.
        
        Args:
            image: Input RGB image.
            mask: Binary foreground mask (255 = keep, 0 = remove).
            
        Returns:
            Masked image with background set to black.
        """
        return cv2.bitwise_and(image, image, mask=mask)

    def get_visualization(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create a visualization showing the detected foreground.
        
        Args:
            image: Original RGB image.
            mask: Foreground mask.
            
        Returns:
            Side-by-side visualization of original and masked image.
        """
        masked = self.apply_mask(image, mask)
        return np.hstack([image, masked])

    def get_name(self) -> str:
        """Get the preprocessor name."""
        return "Background Remover"

    def update_hsv_range(
        self,
        hsv_low: Tuple[int, int, int],
        hsv_high: Tuple[int, int, int],
    ) -> None:
        """
        Update the HSV range for background detection.
        
        Args:
            hsv_low: Lower HSV bound (H, S, V).
            hsv_high: Upper HSV bound (H, S, V).
        """
        self._hsv_low = np.array(hsv_low)
        self._hsv_high = np.array(hsv_high)

    def update_morphology_params(
        self,
        open_kernel: int = 5,
        close_kernel: int = 11,
        open_iterations: int = 2,
        close_iterations: int = 2,
    ) -> None:
        """
        Update morphological operation parameters.
        
        Args:
            open_kernel: Kernel size for opening operation.
            close_kernel: Kernel size for closing operation.
            open_iterations: Number of opening iterations.
            close_iterations: Number of closing iterations.
        """
        self._open_kernel_size = open_kernel
        self._close_kernel_size = close_kernel
        self._open_iterations = open_iterations
        self._close_iterations = close_iterations
