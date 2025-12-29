"""Abstract base class for postprocessing operations."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BasePostprocessor(ABC):
    """
    Abstract base class for mask postprocessing operations.
    
    Postprocessors filter or refine segmentation masks after
    the initial segmentation to improve result quality.
    """

    @abstractmethod
    def process(
        self,
        masks: List[np.ndarray],
        scores: List[float],
        image_shape: tuple[int, int],
    ) -> tuple[List[np.ndarray], List[float]]:
        """
        Process a list of masks and their associated scores.
        
        Args:
            masks: List of binary mask arrays.
            scores: List of confidence scores corresponding to masks.
            image_shape: (height, width) of the original image.
            
        Returns:
            Tuple of (filtered_masks, filtered_scores).
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this postprocessing step.
        
        Returns:
            Human-readable name for the postprocessor.
        """
        pass
