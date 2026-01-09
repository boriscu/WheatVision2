"""SAM2 segmentation tab for WheatVision2 UI."""

import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np

from wheatvision.config.constants import ExportFormat, SegmentationModel
from wheatvision.config.models import SegmentationResult
from wheatvision.pipeline import SegmentationPipeline
from wheatvision.ui.state import AppState
from wheatvision.ui.utils import create_preprocess_gallery, create_result_gallery


class SAM2Tab:
    """SAM2 segmentation tab component."""

    def __init__(self, state: AppState) -> None:
        """Initialize with shared state."""
        self._state = state

    def build(self) -> None:
        """Build the SAM2 segmentation tab UI."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                sam2_video = gr.Video(label="Upload Video")

                gr.Markdown("### Preprocessing Settings")
                sam2_enable_preprocess = gr.Checkbox(
                    label="Enable Background Removal",
                    value=True,
                )
                with gr.Row():
                    sam2_hsv_h_low = gr.Slider(0, 180, value=0, label="H Low")
                    sam2_hsv_h_high = gr.Slider(0, 180, value=180, label="H High")
                with gr.Row():
                    sam2_hsv_s_low = gr.Slider(0, 255, value=0, label="S Low")
                    sam2_hsv_s_high = gr.Slider(0, 255, value=30, label="S High")
                with gr.Row():
                    sam2_hsv_v_low = gr.Slider(0, 255, value=200, label="V Low")
                    sam2_hsv_v_high = gr.Slider(0, 255, value=255, label="V High")

                gr.Markdown("### Postprocessing Settings")
                sam2_enable_postprocess = gr.Checkbox(
                    label="Enable Wheat Ear Filtering",
                    value=True,
                )
                with gr.Row():
                    sam2_min_aspect = gr.Slider(1, 10, value=2, label="Min Aspect")
                    sam2_max_aspect = gr.Slider(1, 20, value=10, label="Max Aspect")

                gr.Markdown("### Processing")
                sam2_max_frames = gr.Slider(
                    1, 100, value=10, step=1,
                    label="Max Frames to Process",
                )
                sam2_run_btn = gr.Button("Run SAM2 Segmentation", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                sam2_status = gr.Textbox(label="Status", interactive=False)
                sam2_metrics = gr.Textbox(
                    label="Metrics",
                    lines=10,
                    interactive=False,
                )

                with gr.Tabs():
                    with gr.Tab("Preprocessing"):
                        sam2_preprocess_gallery = gr.Gallery(
                            label="Preprocessing Steps",
                            columns=3,
                            height=400,
                        )

                    with gr.Tab("Segmentation"):
                        sam2_result_gallery = gr.Gallery(
                            label="Segmented Frames",
                            columns=3,
                            height=400,
                        )

                with gr.Row():
                    sam2_export_masks_btn = gr.Button("Export Masks (PNG)")
                    sam2_export_video_btn = gr.Button("Export Video")
                    sam2_export_metrics_btn = gr.Button("Export Metrics")

                sam2_export_output = gr.File(label="Exported Files")

        # Event handlers
        sam2_run_btn.click(
            fn=self._run_pipeline,
            inputs=[
                sam2_video, sam2_enable_preprocess, sam2_enable_postprocess,
                sam2_hsv_h_low, sam2_hsv_s_low, sam2_hsv_v_low,
                sam2_hsv_h_high, sam2_hsv_s_high, sam2_hsv_v_high,
                sam2_min_aspect, sam2_max_aspect, sam2_max_frames,
            ],
            outputs=[sam2_status, sam2_metrics, sam2_preprocess_gallery, sam2_result_gallery],
        )

        sam2_export_masks_btn.click(
            fn=self._export_masks,
            outputs=[sam2_export_output],
        )
        sam2_export_video_btn.click(
            fn=self._export_video,
            outputs=[sam2_export_output],
        )
        sam2_export_metrics_btn.click(
            fn=self._export_metrics,
            outputs=[sam2_export_output],
        )

    def _run_pipeline(
        self,
        video_path: str,
        enable_preprocess: bool,
        enable_postprocess: bool,
        h_low: int, s_low: int, v_low: int,
        h_high: int, s_high: int, v_high: int,
        min_aspect: float, max_aspect: float,
        max_frames: int,
    ) -> Tuple[str, str, List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]]]:
        """Run SAM2 pipeline and return results for display."""
        try:
            if not video_path:
                return "Error: No video uploaded", "", [], []

            if self._state.sam2_pipeline is None:
                self._state.sam2_pipeline = SegmentationPipeline(
                    SegmentationModel.SAM2,
                    enable_preprocessing=enable_preprocess,
                    enable_postprocessing=enable_postprocess,
                )

            self._state.sam2_pipeline.set_preprocessing_enabled(enable_preprocess)
            self._state.sam2_pipeline.set_postprocessing_enabled(enable_postprocess)

            if enable_preprocess:
                self._state.sam2_pipeline.update_preprocessing_settings(
                    hsv_low=(h_low, s_low, v_low),
                    hsv_high=(h_high, s_high, v_high),
                )

            if enable_postprocess:
                self._state.sam2_pipeline.update_postprocessing_settings(
                    min_aspect=min_aspect,
                    max_aspect=max_aspect,
                )

            frames, preprocess_results, seg_results, metrics = self._state.sam2_pipeline.process_video(
                video_path, max_frames=int(max_frames)
            )

            self._state.sam2_results = (frames, preprocess_results, seg_results, metrics)

            preprocess_images = create_preprocess_gallery(frames, preprocess_results)
            result_images = create_result_gallery(frames, seg_results)
            metrics_text = self._state.metrics_calculator.format_report_text(metrics)

            return (
                f"✓ Processed {len(frames)} frames successfully",
                metrics_text,
                preprocess_images,
                result_images,
            )

        except Exception as e:
            return f"Error: {str(e)}", "", [], []

    def _export_masks(self) -> Optional[str]:
        """Export SAM2 masks as ZIP."""
        if self._state.sam2_results is None:
            return None
        _, _, results, _ = self._state.sam2_results
        return self._export_masks_zip(results, "sam2")

    def _export_masks_zip(self, results: List[SegmentationResult], prefix: str) -> str:
        """Export masks as a ZIP file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / f"{prefix}_masks"
            paths = self._state.mask_exporter.export_batch(results, output_dir, ExportFormat.PNG)
            
            export_dir = Path("exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            zip_path = export_dir / f"{prefix}_masks.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for p in paths:
                    zipf.write(p, Path(p).name)
            
            return str(zip_path)

    def _export_video(self) -> Optional[str]:
        """Export SAM2 video."""
        if self._state.sam2_results is None:
            return None
        frames, _, results, _ = self._state.sam2_results
        output_path = Path("exports/sam2_result.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._state.video_exporter.export_with_overlay(frames, results, output_path)
        return str(output_path)

    def _export_metrics(self) -> Optional[str]:
        """Export SAM2 metrics."""
        if self._state.sam2_results is None:
            return None
        _, _, _, metrics = self._state.sam2_results
        output_path = Path("exports/sam2_metrics")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._state.report_exporter.export(metrics, output_path, ExportFormat.JSON)
        return str(output_path.with_suffix(".json"))
