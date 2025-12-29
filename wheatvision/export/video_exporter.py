"""Video exporter for creating output videos with overlays."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from wheatvision.config.models import FrameData, SegmentationResult
from wheatvision.export.base_exporter import BaseExporter


class VideoExporter(BaseExporter):
    """
    Exports segmentation results as video files.
    
    Creates videos showing masks overlaid on original frames
    for visual verification of segmentation quality.
    """

    def __init__(
        self,
        fps: float = 10.0,
        codec: str = "mp4v",
    ) -> None:
        """
        Initialize the video exporter.
        
        Args:
            fps: Frames per second for output video (default 10 to match typical processing).
            codec: FourCC codec for video encoding.
        """
        self._fps = fps
        self._codec = codec

    def export(
        self,
        data: Tuple[List[FrameData], List[SegmentationResult]],
        output_path: Path,
    ) -> Path:
        """
        Export frames and segmentation results as a video.
        
        Args:
            data: Tuple of (frames, results).
            output_path: Path for output video.
            
        Returns:
            Path to the exported video.
        """
        frames, results = data
        return self.export_with_overlay(frames, results, output_path)

    def export_with_overlay(
        self,
        frames: List[FrameData],
        results: List[SegmentationResult],
        output_path: Path,
        alpha: float = 0.5,
        fps: float | None = None,
    ) -> Path:
        """
        Create video with mask overlays.
        
        Args:
            frames: Original frames.
            results: Segmentation results.
            output_path: Output video path.
            alpha: Overlay transparency.
            fps: Output FPS. If None, uses instance default.
            
        Returns:
            Path to the video.
        """
        output_path = Path(output_path).with_suffix(".mp4")
        self._ensure_parent_exists(output_path)

        if not frames:
            raise ValueError("No frames to export")

        output_fps = fps if fps is not None else self._fps
        
        height, width = frames[0].height, frames[0].width
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            output_fps,
            (width, height),
        )

        colors = self._get_color_palette(50)

        try:
            for frame, result in zip(frames, results):
                overlay = self._create_overlay(frame.image, result.masks, colors, alpha)
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                writer.write(overlay_bgr)
        finally:
            writer.release()

        return output_path

    def export_side_by_side(
        self,
        frames: List[FrameData],
        results: List[SegmentationResult],
        output_path: Path,
        alpha: float = 0.5,
    ) -> Path:
        """
        Create side-by-side video (original | segmented).
        
        Args:
            frames: Original frames.
            results: Segmentation results.
            output_path: Output video path.
            alpha: Overlay transparency.
            
        Returns:
            Path to the video.
        """
        output_path = Path(output_path).with_suffix(".mp4")
        self._ensure_parent_exists(output_path)

        if not frames:
            raise ValueError("No frames to export")

        height, width = frames[0].height, frames[0].width
        combined_width = width * 2

        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self._fps,
            (combined_width, height),
        )

        colors = self._get_color_palette(50)

        try:
            for frame, result in zip(frames, results):
                overlay = self._create_overlay(frame.image, result.masks, colors, alpha)
                combined = np.hstack([frame.image, overlay])
                combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                writer.write(combined_bgr)
        finally:
            writer.release()

        return output_path

    def _create_overlay(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        colors: List[Tuple[int, int, int]],
        alpha: float,
    ) -> np.ndarray:
        """Create image with colored mask overlays."""
        overlay = image.copy()

        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            mask_bool = mask > 0
            overlay[mask_bool] = (
                (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
            ).astype(np.uint8)

        return overlay

    def _get_color_palette(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization."""
        colors = []
        for i in range(n):
            hue = (i * 137.508) % 360
            rgb = self._hsv_to_rgb(hue / 360, 0.8, 0.9)
            colors.append(rgb)
        return colors

    def _hsv_to_rgb(
        self,
        h: float,
        s: float,
        v: float,
    ) -> Tuple[int, int, int]:
        """Convert HSV to RGB."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def get_supported_formats(self) -> List[str]:
        """Get supported formats."""
        return ["mp4"]
