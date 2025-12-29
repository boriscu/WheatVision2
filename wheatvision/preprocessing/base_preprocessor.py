"""Abstract base class for preprocessing operations."""

from abc import ABC, abstractmethod

import numpy as np


class BasePreprocessor(ABC):
    """
    Abstract base class for image preprocessing operations.
    
    All preprocessors implement a common interface for processing 
    individual frames. This enables easy composition in pipelines.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the preprocessing operation to an image.
        
        Args:
            image: Input image as RGB numpy array.
            
        Returns:
            Processed image or mask as numpy array.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this preprocessing step.
        
        Returns:
            Human-readable name for the preprocessor.
        """
        pass
