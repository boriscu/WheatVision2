"""End-to-end segmentation pipeline combining all processing steps."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from wheatvision.config.constants import SegmentationModel
from wheatvision.config.models import (
    BoundingBox,
    FrameData,
    MetricsReport,
    PreprocessingResult,
    SegmentationResult,
)
from wheatvision.engines import BaseSegmentationEngine, SegmentationEngineFactory
from wheatvision.io import FrameLoader
from wheatvision.metrics import MetricsCalculator
from wheatvision.postprocessing import PostprocessingPipeline
from wheatvision.preprocessing import PreprocessingPipeline


class SegmentationPipeline:
    """
    End-to-end pipeline for video segmentation.
    
    Orchestrates the complete workflow:
    1. Load video frames
    2. Preprocess (background removal, ROI detection)
    3. Segment (SAM or SAM2)
    4. Postprocess (wheat ear filtering)
    5. Calculate metrics
    """

    def __init__(
        self,
        model_type: SegmentationModel,
        enable_preprocessing: bool = True,
        enable_postprocessing: bool = True,
    ) -> None:
        """
        Initialize the segmentation pipeline.
        
        Args:
            model_type: Which segmentation model to use (SAM or SAM2).
            enable_preprocessing: Whether to run preprocessing.
            enable_postprocessing: Whether to run postprocessing.
        """
        self._model_type = model_type
        self._enable_preprocessing = enable_preprocessing
        self._enable_postprocessing = enable_postprocessing

        self._frame_loader = FrameLoader()
        self._preprocessing = PreprocessingPipeline()
        self._postprocessing = PostprocessingPipeline()
        self._metrics_calculator = MetricsCalculator()

        self._engine: Optional[BaseSegmentationEngine] = None
        self._is_engine_loaded = False

    def load_engine(self) -> None:
        """
        Load the segmentation model.
        
        Creates and loads the appropriate engine based on model_type.
        """
        if self._engine is not None:
            self._engine.unload_model()

        self._engine = SegmentationEngineFactory.create(self._model_type)
        self._engine.load_model()
        self._is_engine_loaded = True

    def unload_engine(self) -> None:
        """Unload the segmentation model to free resources."""
        if self._engine is not None:
            self._engine.unload_model()
            self._engine = None
            self._is_engine_loaded = False

    def process_video(
        self,
        video_path: Path | str,
        max_frames: Optional[int] = None,
    ) -> Tuple[
        List[FrameData],
        List[PreprocessingResult],
        List[SegmentationResult],
        MetricsReport,
    ]:
        """
        Process a complete video through the pipeline.
        
        Args:
            video_path: Path to the input video.
            max_frames: Optional limit on frames to process.
            
        Returns:
            Tuple of:
            - Original frames
            - Preprocessing results
            - Segmentation results (postprocessed)
            - Metrics report
        """
        self._ensure_engine_loaded()

        self._frame_loader = FrameLoader(max_frames=max_frames)
        frames = self._frame_loader.load_video(video_path)

        preprocessing_results = self._run_preprocessing(frames)

        roi = self._get_combined_roi(preprocessing_results)

        segmentation_results = self._run_segmentation(frames, roi)

        if self._enable_postprocessing and frames:
            image_shape = (frames[0].height, frames[0].width)
            segmentation_results = self._postprocessing.process_results(
                segmentation_results, image_shape
            )

        roi_area = roi.area if roi else None
        metrics = self._metrics_calculator.calculate_all(
            segmentation_results,
            model_name=self._engine.get_model_name(),
            model_load_time_ms=self._engine.model_load_time_ms,
            roi_area=roi_area,
        )

        return frames, preprocessing_results, segmentation_results, metrics

    def process_frames(
        self,
        frames: List[FrameData],
    ) -> Tuple[
        List[PreprocessingResult],
        List[SegmentationResult],
        MetricsReport,
    ]:
        """
        Process a list of frames through the pipeline.
        
        Args:
            frames: List of frames to process.
            
        Returns:
            Tuple of (preprocessing_results, segmentation_results, metrics).
        """
        self._ensure_engine_loaded()

        preprocessing_results = self._run_preprocessing(frames)

        roi = self._get_combined_roi(preprocessing_results)

        segmentation_results = self._run_segmentation(frames, roi)

        if self._enable_postprocessing and frames:
            image_shape = (frames[0].height, frames[0].width)
            segmentation_results = self._postprocessing.process_results(
                segmentation_results, image_shape
            )

        roi_area = roi.area if roi else None
        metrics = self._metrics_calculator.calculate_all(
            segmentation_results,
            model_name=self._engine.get_model_name(),
            model_load_time_ms=self._engine.model_load_time_ms,
            roi_area=roi_area,
        )

        return preprocessing_results, segmentation_results, metrics

    def _run_preprocessing(
        self,
        frames: List[FrameData],
    ) -> List[PreprocessingResult]:
        """Run preprocessing on all frames."""
        if not self._enable_preprocessing:
            return [
                PreprocessingResult(
                    original_frame=f.image,
                    processed_frame=f.image,
                    foreground_mask=np.ones_like(f.image[:, :, 0]) * 255,
                    roi_bbox=None,
                )
                for f in frames
            ]

        return [self._preprocessing.process(f.image) for f in frames]

    def _run_segmentation(
        self,
        frames: List[FrameData],
        roi: Optional[BoundingBox],
    ) -> List[SegmentationResult]:
        """Run segmentation on all frames."""
        return self._engine.segment_frames(frames, roi)

    def _get_combined_roi(
        self,
        preprocessing_results: List[PreprocessingResult],
    ) -> Optional[BoundingBox]:
        """Get a combined ROI from preprocessing results."""
        if not self._enable_preprocessing:
            return None

        return self._preprocessing.get_combined_roi(preprocessing_results)

    def _ensure_engine_loaded(self) -> None:
        """Ensure segmentation engine is loaded."""
        if not self._is_engine_loaded or self._engine is None:
            self.load_engine()

    def get_model_name(self) -> str:
        """Get the name of the current model."""
        if self._engine is not None:
            return self._engine.get_model_name()
        return self._model_type.value

    def set_preprocessing_enabled(self, enabled: bool) -> None:
        """Enable or disable preprocessing."""
        self._enable_preprocessing = enabled

    def set_postprocessing_enabled(self, enabled: bool) -> None:
        """Enable or disable postprocessing."""
        self._enable_postprocessing = enabled

    def update_preprocessing_settings(
        self,
        hsv_low: Optional[tuple[int, int, int]] = None,
        hsv_high: Optional[tuple[int, int, int]] = None,
    ) -> None:
        """Update preprocessing HSV settings."""
        if hsv_low is not None and hsv_high is not None:
            self._preprocessing.update_background_hsv(hsv_low, hsv_high)

    def update_postprocessing_settings(
        self,
        min_aspect: Optional[float] = None,
        max_aspect: Optional[float] = None,
        min_area_ratio: Optional[float] = None,
        max_area_ratio: Optional[float] = None,
    ) -> None:
        """Update postprocessing filter settings."""
        self._postprocessing.update_filter_settings(
            min_aspect=min_aspect,
            max_aspect=max_aspect,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
