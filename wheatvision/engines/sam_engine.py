"""SAM (Segment Anything Model) engine using Automatic Mask Generator."""

import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch

from wheatvision.config.models import BoundingBox, FrameData, SegmentationResult
from wheatvision.config.settings import SAMSettings, get_sam_settings
from wheatvision.engines.base_engine import BaseSegmentationEngine
from wheatvision.utils import get_logger

_logger = get_logger("engines.sam")


class SAMEngine(BaseSegmentationEngine):
    """
    Segmentation engine using the original SAM model with AMG.
    
    Processes each frame independently using Automatic Mask Generator
    for high-quality oversegmentation of wheat ears.
    """

    def __init__(self, settings: SAMSettings | None = None) -> None:
        """
        Initialize the SAM engine.
        
        Args:
            settings: SAM settings. If None, loads from environment.
        """
        super().__init__()
        self._settings = settings or get_sam_settings()
        self._model: Optional[Any] = None
        self._amg: Optional[Any] = None
        
        self._points_per_side: int = 32
        self._pred_iou_thresh: float = 0.86
        self._stability_score_thresh: float = 0.92
        self._min_mask_region_area: int = 100

    def load_model(self) -> None:
        """
        Load the SAM model from checkpoint.
        
        Raises:
            FileNotFoundError: If checkpoint or repository not found.
        """
        _logger.info("Loading SAM model...")
        start_time = time.perf_counter()

        repo_path = Path(self._settings.repo).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"SAM repository not found: {repo_path}. "
                "Please clone: git clone https://github.com/facebookresearch/segment-anything.git external/sam_repo"
            )

        checkpoint_path = Path(self._settings.checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}. "
                "Please download the checkpoint."
            )

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        self._model = sam_model_registry[self._settings.model_type.value](
            checkpoint=str(checkpoint_path)
        )
        self._model.to(device=self._settings.device)
        
        self._amg = SamAutomaticMaskGenerator(
            self._model,
            points_per_side=self._points_per_side,
            pred_iou_thresh=self._pred_iou_thresh,
            stability_score_thresh=self._stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=self._min_mask_region_area,
        )

        self._is_loaded = True
        self._model_load_time_ms = (time.perf_counter() - start_time) * 1000
        _logger.info(f"SAM model loaded in {self._model_load_time_ms:.0f}ms")

    def unload_model(self) -> None:
        """Unload the model and free resources."""
        if self._model is not None:
            del self._amg
            del self._model
            self._amg = None
            self._model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._is_loaded = False

    def segment_frame(
        self,
        frame: FrameData,
        roi: Optional[BoundingBox] = None,
    ) -> SegmentationResult:
        """
        Segment a single frame using Automatic Mask Generator.
        
        Args:
            frame: Frame to segment.
            roi: Optional ROI to restrict segmentation.
            
        Returns:
            SegmentationResult with detected masks.
        """
        self._ensure_loaded()

        start_time = time.perf_counter()

        if roi is not None:
            image = frame.image[roi.y_min:roi.y_max, roi.x_min:roi.x_max].copy()
        else:
            image = frame.image

        masks, scores = self._run_amg(image)

        if roi is not None:
            masks = self._restore_mask_positions(masks, roi, frame.image.shape[:2])

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return SegmentationResult(
            frame_index=frame.frame_index,
            masks=masks,
            scores=scores,
            processing_time_ms=processing_time_ms,
        )

    def segment_frames(
        self,
        frames: List[FrameData],
        roi: Optional[BoundingBox] = None,
    ) -> List[SegmentationResult]:
        """
        Segment multiple frames independently.
        
        Each frame is processed separately using SAM's AMG.
        
        Args:
            frames: List of frames to segment.
            roi: Optional ROI to apply to all frames.
            
        Returns:
            List of SegmentationResult for each frame.
        """
        self._ensure_loaded()

        results = []
        for i, frame in enumerate(frames):
            _logger.debug(f"Segmenting frame {i+1}/{len(frames)}")
            result = self.segment_frame(frame, roi)
            _logger.debug(f"Frame {i+1}: {len(result.masks)} masks in {result.processing_time_ms:.1f}ms")
            results.append(result)

        _logger.info(f"Completed {len(frames)} frames")
        return results

    @torch.inference_mode()
    def _run_amg(
        self,
        image_rgb: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Run Automatic Mask Generator on an image.
        
        Args:
            image_rgb: RGB image to segment.
            
        Returns:
            Tuple of (masks, scores).
        """
        proposals = self._amg.generate(image_rgb)

        proposals.sort(
            key=lambda r: (r.get("area", 0), r.get("predicted_iou", 0.0)),
            reverse=True,
        )

        masks = []
        scores = []
        for proposal in proposals[:500]:
            mask = proposal["segmentation"]
            mask_uint8 = mask.astype(np.uint8) * 255
            masks.append(mask_uint8)
            scores.append(float(proposal.get("predicted_iou", 0.9)))

        return masks, scores

    def _restore_mask_positions(
        self,
        masks: List[np.ndarray],
        roi: BoundingBox,
        full_shape: Tuple[int, int],
    ) -> List[np.ndarray]:
        """
        Restore cropped masks to full image coordinates.
        
        Args:
            masks: Masks from cropped region.
            roi: The ROI used for cropping.
            full_shape: (height, width) of full image.
            
        Returns:
            Masks positioned in full image.
        """
        restored = []
        for mask in masks:
            full_mask = np.zeros(full_shape, dtype=np.uint8)
            h, w = mask.shape[:2]
            full_mask[roi.y_min:roi.y_min+h, roi.x_min:roi.x_min+w] = mask
            restored.append(full_mask)
        return restored

    def set_amg_parameters(
        self,
        points_per_side: Optional[int] = None,
        pred_iou_thresh: Optional[float] = None,
        stability_score_thresh: Optional[float] = None,
        min_mask_region_area: Optional[int] = None,
    ) -> None:
        """
        Update AMG parameters and rebuild the generator.
        
        Args:
            points_per_side: Grid density per side.
            pred_iou_thresh: Predicted IoU threshold.
            stability_score_thresh: Stability score threshold.
            min_mask_region_area: Minimum mask area in pixels.
        """
        if points_per_side is not None:
            self._points_per_side = points_per_side
        if pred_iou_thresh is not None:
            self._pred_iou_thresh = pred_iou_thresh
        if stability_score_thresh is not None:
            self._stability_score_thresh = stability_score_thresh
        if min_mask_region_area is not None:
            self._min_mask_region_area = min_mask_region_area

        if self._model is not None:
            from segment_anything import SamAutomaticMaskGenerator
            self._amg = SamAutomaticMaskGenerator(
                self._model,
                points_per_side=self._points_per_side,
                pred_iou_thresh=self._pred_iou_thresh,
                stability_score_thresh=self._stability_score_thresh,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=self._min_mask_region_area,
            )

    def get_model_name(self) -> str:
        """Get the model name."""
        return f"SAM-{self._settings.model_type.value.upper()}-AMG"
