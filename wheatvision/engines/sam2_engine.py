"""SAM2 engine for video segmentation with Automatic Mask Generator."""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from wheatvision.config.models import BoundingBox, FrameData, SegmentationResult
from wheatvision.config.settings import SAM2Settings, get_sam2_settings
from wheatvision.engines.base_engine import BaseSegmentationEngine


class SAM2Engine(BaseSegmentationEngine):
    """
    Segmentation engine using SAM2's Automatic Mask Generator.
    
    Uses the AMG approach for high-quality automatic oversegmentation,
    which is more effective than simple point grids for detecting
    individual wheat ears.
    """

    def __init__(self, settings: SAM2Settings | None = None) -> None:
        """
        Initialize the SAM2 engine.
        
        Args:
            settings: SAM2 settings. If None, loads from environment.
        """
        super().__init__()
        self._settings = settings or get_sam2_settings()
        self._image_predictor: Optional[Any] = None
        self._amg_class: Optional[Any] = None
        
        self._points_per_side: int = 48
        self._pred_iou_thresh: float = 0.75
        self._stability_score_thresh: float = 0.90
        self._box_nms_thresh: float = 0.7
        self._min_mask_region_area: int = 80
        self._crop_n_layers: int = 1
        self._downscale_long_side: int = 1024

    def load_model(self) -> None:
        """
        Load the SAM2 model and Automatic Mask Generator.
        
        Raises:
            FileNotFoundError: If checkpoint or config not found.
        """
        start_time = time.perf_counter()

        repo_path = Path(self._settings.repo).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"SAM2 repository not found: {repo_path}. "
                "Please clone: git clone https://github.com/facebookresearch/sam2.git external/sam2_repo"
            )

        checkpoint_path = Path(self._settings.ckpt).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {checkpoint_path}. "
                "Please download checkpoints."
            )

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        config_name = str(self._settings.cfg)
        
        if '/sam2/configs/' in config_name:
            config_name = 'configs/' + config_name.split('/sam2/configs/')[-1]
        elif config_name.startswith('configs/'):
            pass
        elif '/' not in config_name:
            config_name = f"configs/sam2.1/{config_name}"
        
        model = build_sam2(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=self._settings.device,
        )
        
        self._image_predictor = SAM2ImagePredictor(model)
        self._amg_class = SAM2AutomaticMaskGenerator

        self._is_loaded = True
        self._model_load_time_ms = (time.perf_counter() - start_time) * 1000

    def unload_model(self) -> None:
        """Unload the model and free resources."""
        if self._image_predictor is not None:
            del self._image_predictor
            self._image_predictor = None

        self._amg_class = None

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
            roi: Optional ROI to restrict segmentation area.
            
        Returns:
            SegmentationResult with detected masks.
        """
        self._ensure_loaded()

        start_time = time.perf_counter()

        working_image, scale = self._prepare_working_image(frame.image)

        if roi is not None:
            scaled_roi = BoundingBox(
                x_min=int(roi.x_min * scale),
                y_min=int(roi.y_min * scale),
                x_max=int(roi.x_max * scale),
                y_max=int(roi.y_max * scale),
            )
            cropped_image = working_image[
                scaled_roi.y_min:scaled_roi.y_max,
                scaled_roi.x_min:scaled_roi.x_max
            ]
        else:
            cropped_image = working_image
            scaled_roi = None

        masks, scores = self._run_amg(cropped_image)

        if scaled_roi is not None:
            masks = self._restore_mask_positions(
                masks, scaled_roi, working_image.shape[:2]
            )

        if scale != 1.0:
            masks = self._upsample_masks(masks, frame.image.shape[:2])

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
        Segment multiple frames.
        
        For SAM2 AMG mode, each frame is processed independently
        with the automatic mask generator.
        
        Args:
            frames: List of frames to segment.
            roi: Optional ROI for all frames.
            
        Returns:
            List of SegmentationResult for each frame.
        """
        self._ensure_loaded()

        results = []
        for frame in frames:
            result = self.segment_frame(frame, roi)
            results.append(result)

        return results

    def _prepare_working_image(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Optionally downscale image for faster processing.
        
        Args:
            image: Original RGB image.
            
        Returns:
            Tuple of (working_image, scale_factor).
        """
        h, w = image.shape[:2]
        max_side = max(h, w)

        if max_side > self._downscale_long_side:
            scale = self._downscale_long_side / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            working = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return working, scale
        
        return image, 1.0

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
        amg = self._amg_class(
            self._image_predictor.model,
            points_per_side=self._points_per_side,
            points_per_batch=64,
            pred_iou_thresh=self._pred_iou_thresh,
            stability_score_thresh=self._stability_score_thresh,
            mask_threshold=0.0,
            box_nms_thresh=self._box_nms_thresh,
            crop_n_layers=self._crop_n_layers,
            crop_overlap_ratio=0.2,
            min_mask_region_area=self._min_mask_region_area,
            output_mode="binary_mask",
            multimask_output=True,
        )

        proposals = amg.generate(image_rgb)

        if not proposals:
            amg_relaxed = self._amg_class(
                self._image_predictor.model,
                points_per_side=64,
                points_per_batch=64,
                pred_iou_thresh=0.70,
                stability_score_thresh=0.88,
                mask_threshold=0.0,
                box_nms_thresh=self._box_nms_thresh,
                crop_n_layers=1,
                crop_overlap_ratio=0.2,
                min_mask_region_area=max(20, self._min_mask_region_area // 2),
                output_mode="binary_mask",
                multimask_output=True,
            )
            proposals = amg_relaxed.generate(image_rgb)

        proposals.sort(
            key=lambda r: (r.get("area", 0), r.get("predicted_iou", 0.0)),
            reverse=True,
        )

        masks = []
        scores = []
        for proposal in proposals[:500]:
            mask = proposal["segmentation"]
            if isinstance(mask, np.ndarray):
                mask_uint8 = (mask > 0).astype(np.uint8) * 255
            else:
                mask_uint8 = (np.array(mask) > 0).astype(np.uint8) * 255
            
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
            full_mask[roi.y_min:roi.y_max, roi.x_min:roi.x_max] = mask
            restored.append(full_mask)
        return restored

    def _upsample_masks(
        self,
        masks: List[np.ndarray],
        original_shape: Tuple[int, int],
    ) -> List[np.ndarray]:
        """
        Upsample masks to original image size.
        
        Args:
            masks: Downscaled masks.
            original_shape: (height, width) of original image.
            
        Returns:
            Upsampled masks.
        """
        h, w = original_shape
        upsampled = []
        for mask in masks:
            up = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            upsampled.append(up)
        return upsampled

    def set_amg_parameters(
        self,
        points_per_side: Optional[int] = None,
        pred_iou_thresh: Optional[float] = None,
        stability_score_thresh: Optional[float] = None,
        min_mask_region_area: Optional[int] = None,
    ) -> None:
        """
        Update AMG parameters at runtime.
        
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

    def get_model_name(self) -> str:
        """Get the model name."""
        return "SAM2-AMG"
