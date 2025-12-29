"""Abstract base class for segmentation engines."""

from abc import ABC, abstractmethod
from typing import List, Optional

from wheatvision.config.models import BoundingBox, FrameData, SegmentationResult


class BaseSegmentationEngine(ABC):
    """
    Abstract base class for segmentation model wrappers.
    
    Defines the common interface that all segmentation engines must
    implement. This enables swapping between SAM and SAM2 engines
    while maintaining consistent API.
    """

    def __init__(self) -> None:
        """Initialize the engine in unloaded state."""
        self._is_loaded = False
        self._model_load_time_ms: float = 0.0

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the segmentation model into memory.
        
        Should set self._is_loaded = True after successful loading.
        Should record model load time in self._model_load_time_ms.
        
        Raises:
            FileNotFoundError: If model checkpoint is not found.
            RuntimeError: If model cannot be loaded.
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Unload the model and free GPU/memory resources.
        
        Should set self._is_loaded = False after unloading.
        """
        pass

    @abstractmethod
    def segment_frame(
        self,
        frame: FrameData,
        roi: Optional[BoundingBox] = None,
    ) -> SegmentationResult:
        """
        Segment a single frame.
        
        Args:
            frame: The frame to segment.
            roi: Optional region of interest to restrict segmentation.
            
        Returns:
            SegmentationResult containing detected masks.
            
        Raises:
            RuntimeError: If model is not loaded.
        """
        pass

    @abstractmethod
    def segment_frames(
        self,
        frames: List[FrameData],
        roi: Optional[BoundingBox] = None,
    ) -> List[SegmentationResult]:
        """
        Segment multiple frames.
        
        For SAM: Each frame is processed independently.
        For SAM2: First frame initializes, subsequent frames use propagation.
        
        Args:
            frames: List of frames to segment.
            roi: Optional region of interest for all frames.
            
        Returns:
            List of SegmentationResult for each frame.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of this segmentation model.
        
        Returns:
            Human-readable model name (e.g., "SAM-ViT-H", "SAM2-Hiera-S").
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._is_loaded

    @property
    def model_load_time_ms(self) -> float:
        """Get the time taken to load the model in milliseconds."""
        return self._model_load_time_ms

    def _ensure_loaded(self) -> None:
        """
        Ensure the model is loaded before performing operations.
        
        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError(
                f"{self.get_model_name()} model is not loaded. "
                "Call load_model() first."
            )
