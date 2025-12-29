"""Video writer for creating output videos with segmentation overlays."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from wheatvision.config.models import FrameData, SegmentationResult


class VideoWriter:
    """
    Writes frames to video files with optional segmentation overlays.
    
    Supports creating visualization videos showing the segmentation results
    overlaid on the original frames.
    """

    def __init__(
        self,
        output_path: Path | str,
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: str = "mp4v",
    ) -> None:
        """
        Initialize the video writer.
        
        Args:
            output_path: Path for the output video file.
            fps: Frames per second for the output video.
            frame_size: Optional (width, height) tuple. If None, uses first frame size.
            codec: FourCC codec identifier.
        """
        self._output_path = Path(output_path)
        self._fps = fps
        self._frame_size = frame_size
        self._codec = codec
        self._writer: Optional[cv2.VideoWriter] = None
        self._is_open = False

    def _ensure_writer(self, frame: np.ndarray) -> None:
        """Create the video writer if not already open."""
        if self._writer is not None and self._is_open:
            return

        if self._frame_size is None:
            height, width = frame.shape[:2]
            self._frame_size = (width, height)

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._writer = cv2.VideoWriter(
            str(self._output_path),
            fourcc,
            self._fps,
            self._frame_size,
        )
        self._is_open = True

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to the video.
        
        Args:
            frame: RGB image array to write.
        """
        self._ensure_writer(frame)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self._frame_size:
            frame_bgr = cv2.resize(frame_bgr, self._frame_size)

        self._writer.write(frame_bgr)

    def write_frame_with_overlay(
        self,
        frame: FrameData,
        result: SegmentationResult,
        alpha: float = 0.5,
        colors: Optional[List[Tuple[int, int, int]]] = None,
    ) -> None:
        """
        Write a frame with segmentation masks overlaid.
        
        Args:
            frame: The original frame data.
            result: Segmentation result containing masks.
            alpha: Transparency of the overlay (0-1).
            colors: Optional list of RGB colors for each mask.
        """
        overlay = self._create_overlay(frame.image, result.masks, alpha, colors)
        self.write_frame(overlay)

    def _create_overlay(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        alpha: float = 0.5,
        colors: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Create an image with colored mask overlays.
        
        Args:
            image: Original RGB image.
            masks: List of binary masks.
            alpha: Overlay transparency.
            colors: Optional list of colors; uses default palette if None.
            
        Returns:
            Image with overlaid masks.
        """
        if colors is None:
            colors = self._get_default_colors(len(masks))

        overlay = image.copy()

        for mask, color in zip(masks, colors):
            mask_bool = mask > 0
            overlay[mask_bool] = (
                (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
            ).astype(np.uint8)

        return overlay

    def _get_default_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate a list of distinct colors for mask visualization."""
        default_palette = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 255, 128),    # Spring green
            (255, 128, 128),  # Light red
        ]

        colors = []
        for i in range(n):
            colors.append(default_palette[i % len(default_palette)])
        return colors

    def write_frames(self, frames: List[FrameData]) -> None:
        """
        Write multiple frames to the video.
        
        Args:
            frames: List of frames to write.
        """
        for frame in frames:
            self.write_frame(frame.image)

    def write_results(
        self,
        frames: List[FrameData],
        results: List[SegmentationResult],
        alpha: float = 0.5,
    ) -> None:
        """
        Write frames with segmentation overlays to the video.
        
        Args:
            frames: List of original frames.
            results: List of corresponding segmentation results.
            alpha: Overlay transparency.
        """
        for frame, result in zip(frames, results):
            self.write_frame_with_overlay(frame, result, alpha)

    def close(self) -> None:
        """Release the video writer resources."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._is_open = False

    def __enter__(self) -> "VideoWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()
