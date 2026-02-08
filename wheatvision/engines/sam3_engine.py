"""SAM3 segmentation engine using Promptable Concept Segmentation."""

import sys
import tempfile
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from wheatvision.config.models import BoundingBox, FrameData, SegmentationResult
from wheatvision.config.settings import SAM3Settings, get_sam3_settings
from wheatvision.engines.base_engine import BaseSegmentationEngine
from wheatvision.utils import get_logger

_logger = get_logger(__name__)


class SAM3Engine(BaseSegmentationEngine):
    """
    Segmentation engine using SAM3's Promptable Concept Segmentation.
    
    This engine leverages SAM3's ability to:
    1. Segment based on text prompts (open-vocabulary)
    2. Track objects consistently through video
    3. Handle fine structures using a unified detector-tracker architecture
    
    Key advantage: Uses text prompts like "wheat ear" for targeted segmentation.
    """

    def __init__(self, settings: SAM3Settings | None = None) -> None:
        """
        Initialize the SAM3 engine.
        
        Args:
            settings: SAM3 settings. If None, loads from environment.
        """
        super().__init__()
        self._settings = settings or get_sam3_settings()
        
        self._video_predictor: Optional[Any] = None
        self._image_model: Optional[Any] = None
        self._image_processor: Optional[Any] = None
        self._session_id: Optional[str] = None
        self._temp_dir: Optional[Path] = None
        
        # Default text prompt for wheat ear segmentation
        self._text_prompt = "wheat ear"
        
        # SAM3 detection parameters
        self._score_threshold = 0.5
        
        _logger.info(f"SAM3Engine initialized with device: {self._settings.device}")

    def _validate_repo_path(self) -> None:
        """Validate that SAM3 repository exists."""
        repo_path = Path(self._settings.repo)
        if not repo_path.exists():
            raise RuntimeError(
                f"SAM3 repository not found: {repo_path.resolve()}. "
                f"Please clone: git clone https://github.com/facebookresearch/sam3.git {repo_path}"
            )
        
        # Add to path if needed
        repo_str = str(repo_path.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def load_model(self) -> None:
        """Load the SAM3 video predictor model."""
        if self._is_loaded:
            _logger.debug("SAM3 model already loaded")
            return
        
        self._validate_repo_path()
        
        _logger.info("Loading SAM3 model...")
        start_time = time.perf_counter()
        
        try:
            from sam3.model_builder import build_sam3_video_predictor, build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # Build SAM3 video predictor (downloads from HuggingFace by default)
            self._video_predictor = build_sam3_video_predictor()
            
            # Build image model for single frame processing
            self._image_model = build_sam3_image_model(device=self._settings.device)
            self._image_processor = Sam3Processor(self._image_model)
            
            self._is_loaded = True
            self._model_load_time_ms = (time.perf_counter() - start_time) * 1000
            _logger.info(f"SAM3 model loaded in {self._model_load_time_ms:.0f}ms")
            
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import SAM3. Please install: pip install -e {self._settings.repo}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {e}") from e

    def unload_model(self) -> None:
        """Unload the SAM3 model and free memory."""
        self._cleanup_temp_files()
        
        if self._video_predictor is not None:
            del self._video_predictor
            self._video_predictor = None
            
        if self._image_model is not None:
            del self._image_model
            self._image_model = None
            
        if self._image_processor is not None:
            del self._image_processor
            self._image_processor = None
            
        self._session_id = None
        self._is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        _logger.info("SAM3 model unloaded")

    def set_text_prompt(self, prompt: str) -> None:
        """
        Set the text prompt for concept segmentation.
        
        Args:
            prompt: Text describing what to segment (e.g., "wheat ear", "plant leaf")
        """
        self._text_prompt = prompt
        _logger.info(f"SAM3 text prompt set to: {prompt}")

    def segment_frame(
        self,
        frame: FrameData,
        roi: Optional[BoundingBox] = None,
    ) -> SegmentationResult:
        """
        Segment a single frame using SAM3 image model with text prompt.
        
        Args:
            frame: The frame to segment.
            roi: Optional region of interest to restrict segmentation.
            
        Returns:
            SegmentationResult with detected masks.
        """
        self._ensure_loaded()
        
        start_time = time.perf_counter()
        
        from PIL import Image as PILImage
        
        image = frame.image
        if roi is not None:
            image = image[roi.y_min:roi.y_max, roi.x_min:roi.x_max].copy()
        
        # Convert to PIL
        pil_image = PILImage.fromarray(image)
        
        # Set image and get inference state
        inference_state = self._image_processor.set_image(pil_image)
        
        # Query with text prompt
        output = self._image_processor.set_text_prompt(
            state=inference_state,
            prompt=self._text_prompt,
        )
        
        raw_masks = output.get("masks", [])
        raw_scores = output.get("scores", [])
        
        # Convert to our format
        masks = []
        scores = []
        
        for i, mask in enumerate(raw_masks):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask.squeeze()
            masks.append(mask.astype(bool))
            scores.append(float(raw_scores[i]) if i < len(raw_scores) else 1.0)
        
        if roi is not None:
            masks = self._restore_mask_positions(masks, roi, frame.image.shape[:2])
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        return SegmentationResult(
            frame_index=frame.frame_index,
            masks=masks,
            scores=scores,
            processing_time_ms=processing_time_ms,
        )

    def _restore_mask_positions(
        self,
        masks: List[np.ndarray],
        roi: BoundingBox,
        original_shape: tuple,
    ) -> List[np.ndarray]:
        """Restore cropped masks to full image position."""
        restored = []
        h, w = original_shape
        for mask in masks:
            full_mask = np.zeros((h, w), dtype=bool)
            full_mask[roi.y_min:roi.y_max, roi.x_min:roi.x_max] = mask
            restored.append(full_mask)
        return restored

    def _prepare_video_frames(self, frames: List[FrameData]) -> Path:
        """
        Prepare frames as JPEG files for SAM3 video predictor.
        
        Args:
            frames: List of FrameData objects.
            
        Returns:
            Path to temporary directory containing JPEG files.
        """
        self._temp_dir = Path(tempfile.mkdtemp(prefix="sam3_"))
        
        for i, frame in enumerate(frames):
            frame_path = self._temp_dir / f"{i:06d}.jpg"
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), bgr_frame)
        
        return self._temp_dir

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary video frame files."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def segment_frames(
        self,
        frames: List[FrameData],
        roi: Optional[BoundingBox] = None,
    ) -> List[SegmentationResult]:
        """
        Segment multiple frames using SAM3 video predictor with text prompts.
        
        Uses SAM3's request-based video API for tracking.
        
        Args:
            frames: List of frames to segment.
            roi: Optional region of interest (applied to reference frame).
            
        Returns:
            List of SegmentationResults, one per frame.
        """
        if not frames:
            return []
        
        if len(frames) == 1:
            return [self.segment_frame(frames[0], roi)]
        
        self._ensure_loaded()
        
        _logger.info(f"Processing {len(frames)} frames with SAM3 video predictor")
        start_time = time.perf_counter()
        
        try:
            # Prepare frames as JPEG folder
            video_path = str(self._prepare_video_frames(frames))
            
            # Start session
            response = self._video_predictor.handle_request(
                request={
                    "type": "start_session",
                    "resource_path": video_path,
                }
            )
            self._session_id = response["session_id"]
            
            # Add text prompt at frame 0
            response = self._video_predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": self._session_id,
                    "frame_index": 0,
                    "text": self._text_prompt,
                }
            )
            
            # Process outputs
            outputs = response.get("outputs", {})
            
            results = []
            for frame_idx, frame in enumerate(frames):
                frame_start = time.perf_counter()
                frame_output = outputs.get(frame_idx, {})
                
                raw_masks = frame_output.get("masks", [])
                raw_scores = frame_output.get("scores", [])
                
                # Convert masks
                masks = []
                scores = []
                
                for i, mask in enumerate(raw_masks):
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    masks.append(mask.astype(bool))
                    scores.append(float(raw_scores[i]) if i < len(raw_scores) else 1.0)
                
                frame_time = (time.perf_counter() - frame_start) * 1000
                
                results.append(SegmentationResult(
                    frame_index=frame.frame_index,
                    masks=masks,
                    scores=scores,
                    processing_time_ms=frame_time,
                ))
            
            # End session
            self._video_predictor.handle_request(
                request={
                    "type": "end_session",
                    "session_id": self._session_id,
                }
            )
            self._session_id = None
            
            total_time = (time.perf_counter() - start_time) * 1000
            _logger.info(f"SAM3 video processing complete: {len(results)} frames in {total_time:.0f}ms")
            return results
            
        finally:
            self._cleanup_temp_files()

    def get_model_name(self) -> str:
        """Get the model name."""
        return f"SAM3-PCS-{self._text_prompt}"
