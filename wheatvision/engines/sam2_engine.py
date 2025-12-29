"""SAM2 engine for video segmentation with propagation."""

import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from wheatvision.config.models import BoundingBox, FrameData, SegmentationResult
from wheatvision.config.settings import SAM2Settings, get_sam2_settings
from wheatvision.engines.base_engine import BaseSegmentationEngine
from wheatvision.utils import get_logger

_logger = get_logger("engines.sam2")


class SAM2Engine(BaseSegmentationEngine):
    """
    Segmentation engine using SAM2's video propagation.
    
    This engine:
    1. Runs AMG on the first (or sharpest) frame to get initial masks
    2. Uses SAM2's video predictor to propagate masks through all frames
    
    This is the key advantage over SAM - track objects consistently through video.
    """

    def __init__(self, settings: SAM2Settings | None = None) -> None:
        """
        Initialize the SAM2 engine.
        
        Args:
            settings: SAM2 settings. If None, loads from environment.
        """
        super().__init__()
        self._settings = settings or get_sam2_settings()
        
        self._video_predictor: Optional[Any] = None
        self._image_predictor: Optional[Any] = None
        self._amg_class: Optional[Any] = None
        
        self._inference_state: Optional[Any] = None
        self._temp_dir: Optional[TemporaryDirectory] = None
        
        self._points_per_side: int = 32
        self._pred_iou_thresh: float = 0.80
        self._stability_score_thresh: float = 0.90
        self._min_mask_region_area: int = 100
        self._max_objects: int = 50
        
        self._merge_masks: bool = False  # Disabled - was causing display issues

    def load_model(self) -> None:
        """
        Load SAM2 video predictor and image predictor for AMG.
        
        Raises:
            FileNotFoundError: If checkpoint not found.
        """
        _logger.info("Loading SAM2 model...")
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

        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        config_name = str(self._settings.cfg)
        if '/sam2/configs/' in config_name:
            config_name = 'configs/' + config_name.split('/sam2/configs/')[-1]
        elif config_name.startswith('configs/'):
            pass
        elif '/' not in config_name:
            config_name = f"configs/sam2.1/{config_name}"

        _logger.debug(f"Using config: {config_name}")
        _logger.debug(f"Using checkpoint: {checkpoint_path}")

        self._video_predictor = build_sam2_video_predictor(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=self._settings.device,
        )

        model = build_sam2(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=self._settings.device,
        )
        self._image_predictor = SAM2ImagePredictor(model)
        self._amg_class = SAM2AutomaticMaskGenerator

        self._is_loaded = True
        self._model_load_time_ms = (time.perf_counter() - start_time) * 1000
        _logger.info(f"SAM2 model loaded in {self._model_load_time_ms:.0f}ms")

    def unload_model(self) -> None:
        """Unload the model and free resources."""
        self._cleanup_temp()
        
        if self._video_predictor is not None:
            del self._video_predictor
            self._video_predictor = None
        
        if self._image_predictor is not None:
            del self._image_predictor
            self._image_predictor = None

        self._amg_class = None
        self._inference_state = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False

    def segment_frame(
        self,
        frame: FrameData,
        roi: Optional[BoundingBox] = None,
    ) -> SegmentationResult:
        """
        Segment a single frame using AMG.
        
        For single frames, falls back to AMG mode.
        For video propagation, use segment_frames instead.
        """
        self._ensure_loaded()
        
        start_time = time.perf_counter()
        
        image = frame.image
        if roi is not None:
            image = image[roi.y_min:roi.y_max, roi.x_min:roi.x_max].copy()
        
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
        Segment video frames using SAM2 propagation.
        
        Workflow:
        1. Find the sharpest frame as reference
        2. Run AMG on reference frame to get initial masks
        3. Initialize video predictor with all frames
        4. Add initial masks to reference frame
        5. Propagate masks through all frames
        
        Args:
            frames: List of frames to segment.
            roi: Optional ROI (applied to reference frame for AMG).
            
        Returns:
            List of SegmentationResult for each frame.
        """
        self._ensure_loaded()
        
        if len(frames) == 0:
            return []
        
        if len(frames) == 1:
            return [self.segment_frame(frames[0], roi)]
        
        total_start = time.perf_counter()
        
        # IMPORTANT: Use frame 0 as reference because propagate_in_video only
        # propagates FORWARD from the reference frame. Using any other frame
        # would leave earlier frames without masks.
        seq_ref_idx = 0
        _logger.info(f"Using frame 0 as reference (propagation only goes forward)")
        
        _logger.info("Running AMG on reference frame...")
        ref_frame = frames[seq_ref_idx]
        ref_image = ref_frame.image
        if roi is not None:
            ref_image = ref_image[roi.y_min:roi.y_max, roi.x_min:roi.x_max].copy()
        
        initial_masks, initial_scores = self._run_amg(ref_image)
        
        if roi is not None:
            initial_masks = self._restore_mask_positions(
                initial_masks, roi, ref_frame.image.shape[:2]
            )
        
        _logger.info(f"Found {len(initial_masks)} masks on reference frame")
        
        if len(initial_masks) == 0:
            _logger.warning("No masks found, returning empty results")
            return [
                SegmentationResult(
                    frame_index=f.frame_index,
                    masks=[],
                    scores=[],
                    processing_time_ms=0.0,
                )
                for f in frames
            ]
        
        masks_to_track = initial_masks[:self._max_objects]
        scores_to_track = initial_scores[:self._max_objects]
        _logger.info(f"Tracking {len(masks_to_track)} objects through {len(frames)} frames")
        
        _logger.info("Initializing video predictor...")
        self._init_video(frames)
        
        for obj_id, mask in enumerate(masks_to_track):
            mask_bool = (mask > 0).astype(np.uint8)
            self._add_mask(seq_ref_idx, mask_bool, obj_id)
        
        _logger.info("Propagating masks through video...")
        propagated = self._propagate()
        
        _logger.info(f"Propagation returned {len(propagated)} frame results")
        
        results = []
        for seq_idx, frame in enumerate(frames):
            frame_masks = []
            frame_scores = []
            
            if seq_idx in propagated:
                for obj_id in sorted(propagated[seq_idx].keys()):
                    mask = propagated[seq_idx][obj_id]
                    mask_uint8 = (mask > 0).astype(np.uint8) * 255
                    frame_masks.append(mask_uint8)
                    
                    if obj_id < len(scores_to_track):
                        frame_scores.append(scores_to_track[obj_id])
                    else:
                        frame_scores.append(0.9)
            
            elapsed = (time.perf_counter() - total_start) * 1000
            per_frame_ms = elapsed / max(1, seq_idx + 1)
            
            if self._merge_masks and frame_masks:
                _logger.debug(f"Merging {len(frame_masks)} masks for frame {seq_idx}")
                merged = self._combine_masks(frame_masks)
                frame_masks = [merged]
                frame_scores = [sum(frame_scores) / len(frame_scores)] if frame_scores else [1.0]
            
            results.append(SegmentationResult(
                frame_index=frame.frame_index,
                masks=frame_masks,
                scores=frame_scores,
                processing_time_ms=per_frame_ms,
            ))
        
        total_time = (time.perf_counter() - total_start) * 1000
        _logger.info(f"Video propagation complete in {total_time:.0f}ms")
        
        self._cleanup_temp()
        
        return results

    def _init_video(self, frames: List[FrameData]) -> None:
        """
        Write frames to temp dir and init video predictor.
        
        IMPORTANT: Frame filenames must be sequential (00000.jpg, 00001.jpg, ...)
        regardless of the original frame.frame_index values.
        """
        self._cleanup_temp()
        
        self._temp_dir = TemporaryDirectory(prefix="wheatvision_sam2_")
        video_dir = self._temp_dir.name
        
        for seq_idx, frame in enumerate(frames):
            path = os.path.join(video_dir, f"{seq_idx:05d}.jpg")
            bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
        
        _logger.debug(f"Wrote {len(frames)} frames to {video_dir}")
        self._inference_state = self._video_predictor.init_state(video_path=video_dir)

    def _add_mask(self, seq_frame_idx: int, mask: np.ndarray, obj_id: int) -> None:
        """
        Add a mask for tracking.
        
        Args:
            seq_frame_idx: Sequential frame index (0, 1, 2, ...) NOT frame.frame_index
            mask: Binary mask
            obj_id: Object ID
        """
        if self._inference_state is None:
            raise RuntimeError("Video not initialized")
        
        _logger.debug(f"Adding mask for obj_id={obj_id} at seq_frame_idx={seq_frame_idx}")
        self._video_predictor.add_new_mask(
            inference_state=self._inference_state,
            frame_idx=seq_frame_idx,
            obj_id=obj_id,
            mask=mask,
        )

    def _propagate(self) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Propagate masks through all frames.
        
        Returns:
            Dict mapping sequential frame index -> {obj_id -> binary mask}
        """
        if self._inference_state is None:
            raise RuntimeError("Video not initialized")
        
        results = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self._video_predictor.propagate_in_video(
            self._inference_state
        ):
            frame_masks = {}
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                frame_masks[obj_id] = mask
            results[out_frame_idx] = frame_masks
            _logger.debug(f"Propagated frame {out_frame_idx}: {len(frame_masks)} masks")
        
        return results

    def _cleanup_temp(self) -> None:
        """Clean up temporary directory."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None
        self._inference_state = None

    def _find_sharpest_frame_seq_idx(self, frames: List[FrameData]) -> int:
        """
        Find the sharpest frame using Laplacian variance.
        
        Returns:
            Sequential index (0, 1, 2, ...) into the frames list, NOT frame.frame_index
        """
        best_seq_idx = 0
        best_score = 0.0
        
        for seq_idx, frame in enumerate(frames):
            gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_seq_idx = seq_idx
        
        return best_seq_idx

    @torch.inference_mode()
    def _run_amg(self, image_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """Run Automatic Mask Generator on an image."""
        amg = self._amg_class(
            self._image_predictor.model,
            points_per_side=self._points_per_side,
            points_per_batch=64,
            pred_iou_thresh=self._pred_iou_thresh,
            stability_score_thresh=self._stability_score_thresh,
            mask_threshold=0.0,
            box_nms_thresh=0.7,
            crop_n_layers=1,
            crop_overlap_ratio=0.2,
            min_mask_region_area=self._min_mask_region_area,
            output_mode="binary_mask",
            multimask_output=True,
        )

        proposals = amg.generate(image_rgb)

        proposals.sort(
            key=lambda r: (r.get("area", 0), r.get("predicted_iou", 0.0)),
            reverse=True,
        )

        masks = []
        scores = []
        for proposal in proposals[:200]:
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
        """Restore cropped masks to full image coordinates."""
        restored = []
        for mask in masks:
            full_mask = np.zeros(full_shape, dtype=np.uint8)
            h, w = mask.shape[:2]
            full_mask[roi.y_min:roi.y_min+h, roi.x_min:roi.x_min+w] = mask
            restored.append(full_mask)
        return restored

    def set_propagation_parameters(
        self,
        max_objects: Optional[int] = None,
        points_per_side: Optional[int] = None,
        pred_iou_thresh: Optional[float] = None,
        stability_score_thresh: Optional[float] = None,
        min_mask_region_area: Optional[int] = None,
    ) -> None:
        """Update propagation/AMG parameters."""
        if max_objects is not None:
            self._max_objects = max_objects
        if points_per_side is not None:
            self._points_per_side = points_per_side
        if pred_iou_thresh is not None:
            self._pred_iou_thresh = pred_iou_thresh
        if stability_score_thresh is not None:
            self._stability_score_thresh = stability_score_thresh
        if min_mask_region_area is not None:
            self._min_mask_region_area = min_mask_region_area

    def set_merge_masks(self, merge: bool) -> None:
        """
        Set whether to merge all masks into one semantic mask.
        
        Args:
            merge: If True, all instance masks are combined into one.
        """
        self._merge_masks = merge

    def _combine_masks(
        self,
        masks: List[np.ndarray],
    ) -> np.ndarray:
        """
        Combine all instance masks into a single semantic mask.
        
        Args:
            masks: List of instance masks.
            
        Returns:
            Single combined mask with all instances merged.
        """
        if not masks:
            return np.zeros((100, 100), dtype=np.uint8)
        
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined = np.maximum(combined, mask)
        return combined

    def get_model_name(self) -> str:
        """Get the model name."""
        suffix = "-Semantic" if self._merge_masks else "-Instance"
        return f"SAM2-VideoTrack{suffix}"

