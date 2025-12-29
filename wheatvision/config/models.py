"""Pydantic data models for WheatVision2."""

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class NumpyArrayModel(BaseModel):
    """Base model that allows numpy arrays as fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BoundingBox(BaseModel):
    """Represents a bounding box in image coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        """Calculate box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Calculate box height."""
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        """Calculate box area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        """Get the center point of the bounding box."""
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Return as (x_min, y_min, x_max, y_max) tuple."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_xywh(self) -> tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x_min, self.y_min, self.width, self.height)


class FrameData(NumpyArrayModel):
    """Container for a single video frame with metadata."""

    frame_index: int
    image: np.ndarray
    timestamp_ms: float = 0.0

    @field_validator("image", mode="before")
    @classmethod
    def _validate_image(cls, value: np.ndarray) -> np.ndarray:
        """Validate that image is a proper numpy array."""
        if not isinstance(value, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if len(value.shape) not in (2, 3):
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")
        return value

    @property
    def height(self) -> int:
        """Get image height."""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """Get image width."""
        return self.image.shape[1]

    @property
    def shape(self) -> tuple[int, ...]:
        """Get image shape."""
        return self.image.shape


class MaskProperties(BaseModel):
    """Geometric properties extracted from a segmentation mask."""

    area: int
    perimeter: float
    aspect_ratio: float
    solidity: float
    bounding_box: BoundingBox
    centroid: tuple[float, float]


class SegmentationResult(NumpyArrayModel):
    """Result of segmenting a single frame."""

    frame_index: int
    masks: List[np.ndarray]
    scores: List[float]
    processing_time_ms: float

    @property
    def mask_count(self) -> int:
        """Get the number of masks detected."""
        return len(self.masks)

    def get_combined_mask(self) -> np.ndarray:
        """Create a single mask combining all detected masks."""
        if not self.masks:
            raise ValueError("No masks to combine")
        combined = np.zeros_like(self.masks[0], dtype=np.uint8)
        for i, mask in enumerate(self.masks):
            combined[mask > 0] = i + 1
        return combined


class PreprocessingResult(NumpyArrayModel):
    """Result of preprocessing a single frame."""

    original_frame: np.ndarray
    processed_frame: np.ndarray
    foreground_mask: np.ndarray
    roi_bbox: Optional[BoundingBox] = None


class SpeedMetricsReport(BaseModel):
    """Report containing speed-related metrics."""

    model_name: str
    total_frames: int
    model_load_time_ms: float
    total_processing_time_ms: float
    fps: float
    avg_time_per_frame_ms: float
    min_time_per_frame_ms: float
    max_time_per_frame_ms: float


class AccuracyMetricsReport(BaseModel):
    """Report containing accuracy and consistency metrics."""

    model_name: str
    total_frames: int
    avg_masks_per_frame: float
    mask_count_std: float
    temporal_consistency_score: float
    coverage_ratio: float


class MetricsReport(BaseModel):
    """Combined metrics report for a segmentation run."""

    model_name: str
    speed_metrics: SpeedMetricsReport
    accuracy_metrics: AccuracyMetricsReport
