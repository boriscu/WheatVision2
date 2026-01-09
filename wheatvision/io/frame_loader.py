"""Frame loader for extracting frames from video files."""

from pathlib import Path
from typing import Generator, List, Optional

import cv2
import numpy as np

from wheatvision.config.models import FrameData


class FrameLoader:
    """
    Loads frames from video files or image sequences.
    
    Provides both batch loading and streaming interfaces for efficient
    memory usage with large videos.
    """

    def __init__(
        self,
        max_frames: Optional[int] = None,
        target_fps: float = 1.0,
    ) -> None:
        """
        Initialize the frame loader.
        
        Args:
            max_frames: Optional limit on number of frames to load.
                        If None, loads all frames.
            target_fps: Target frames per second to extract from video.
                        Defaults to 1.0 (1 frame per second), so an 11-second
                        video yields 11 frames.
        """
        self._max_frames = max_frames
        self._target_fps = target_fps

    def load_video(self, video_path: Path | str) -> List[FrameData]:
        """
        Load all frames from a video file into memory.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            List of FrameData objects containing each frame.
            
        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If video cannot be opened.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames = list(self.stream_video(video_path))
        return frames

    def stream_video(self, video_path: Path | str) -> Generator[FrameData, None, None]:
        """
        Stream frames from a video file one at a time.
        
        This is more memory-efficient than load_video for large files.
        
        Args:
            video_path: Path to the video file.
            
        Yields:
            FrameData objects for each frame.
            
        Raises:
            ValueError: If video cannot be opened.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame skip to achieve target FPS
        skip = max(1, int(round(video_fps / self._target_fps))) if video_fps > 0 else 1
        
        source_frame_index = 0
        output_frame_index = 0

        try:
            while True:
                if self._max_frames and output_frame_index >= self._max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if source_frame_index % skip == 0:
                    timestamp_ms = (source_frame_index / video_fps) * 1000 if video_fps > 0 else 0.0
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    yield FrameData(
                        frame_index=output_frame_index,
                        image=frame_rgb,
                        timestamp_ms=timestamp_ms,
                    )
                    output_frame_index += 1
                
                source_frame_index += 1
        finally:
            cap.release()

    def load_image(self, image_path: Path | str) -> FrameData:
        """
        Load a single image file as a FrameData object.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            FrameData object containing the image.
            
        Raises:
            FileNotFoundError: If image file does not exist.
            ValueError: If image cannot be loaded.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return FrameData(
            frame_index=0,
            image=image_rgb,
            timestamp_ms=0.0,
        )

    def load_images(self, image_paths: List[Path | str]) -> List[FrameData]:
        """
        Load multiple images as a sequence of frames.
        
        Args:
            image_paths: List of paths to image files.
            
        Returns:
            List of FrameData objects.
        """
        frames = []
        for idx, path in enumerate(image_paths):
            frame_data = self.load_image(path)
            frame_data.frame_index = idx
            frames.append(frame_data)

            if self._max_frames and idx >= self._max_frames - 1:
                break

        return frames

    def get_video_info(self, video_path: Path | str) -> dict:
        """
        Get metadata about a video file without loading frames.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Dictionary with video metadata including:
            - frame_count: Total number of frames
            - fps: Frames per second
            - width: Frame width in pixels
            - height: Frame height in pixels
            - duration_ms: Video duration in milliseconds
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_ms = (frame_count / fps) * 1000 if fps > 0 else 0.0

            return {
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_ms": duration_ms,
            }
        finally:
            cap.release()
