"""Shared application state for WheatVision2 UI."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from wheatvision.config.models import FrameData, MetricsReport, PreprocessingResult, SegmentationResult
from wheatvision.export import MaskExporter, ReportExporter, VideoExporter
from wheatvision.metrics import MetricsCalculator
from wheatvision.pipeline import SegmentationPipeline


@dataclass
class AppState:
    """
    Shared state between UI tabs.
    
    Holds pipeline instances, results, and exporters that need
    to be shared across different tabs.
    """
    # Pipelines
    sam_pipeline: Optional[SegmentationPipeline] = None
    sam2_pipeline: Optional[SegmentationPipeline] = None
    
    # Results: (frames, preprocess_results, seg_results, metrics)
    sam_results: Optional[Tuple[
        List[FrameData], 
        List[PreprocessingResult], 
        List[SegmentationResult], 
        MetricsReport
    ]] = None
    
    sam2_results: Optional[Tuple[
        List[FrameData], 
        List[PreprocessingResult], 
        List[SegmentationResult], 
        MetricsReport
    ]] = None
    
    # Exporters
    metrics_calculator: MetricsCalculator = field(default_factory=MetricsCalculator)
    mask_exporter: MaskExporter = field(default_factory=MaskExporter)
    video_exporter: VideoExporter = field(default_factory=VideoExporter)
    report_exporter: ReportExporter = field(default_factory=ReportExporter)
