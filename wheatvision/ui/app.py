"""Main Gradio application for WheatVision2."""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from wheatvision.config.constants import ExportFormat, SegmentationModel
from wheatvision.config.models import FrameData, MetricsReport, PreprocessingResult, SegmentationResult
from wheatvision.config.settings import get_ui_settings
from wheatvision.export import MaskExporter, ReportExporter, VideoExporter
from wheatvision.io import FrameLoader
from wheatvision.metrics import MetricsCalculator
from wheatvision.pipeline import SegmentationPipeline
from wheatvision.utils import setup_logging, get_logger

_logger = get_logger("ui")


class WheatVisionApp:
    """
    Main Gradio application for SAM vs SAM2 comparison.
    
    Provides a tabbed interface for:
    - SAM: Frame-by-frame segmentation
    - SAM2: Video propagation segmentation
    - Comparison: Side-by-side metrics comparison
    """

    def __init__(self) -> None:
        """Initialize the application."""
        self._sam_pipeline: Optional[SegmentationPipeline] = None
        self._sam2_pipeline: Optional[SegmentationPipeline] = None
        self._metrics_calculator = MetricsCalculator()
        self._mask_exporter = MaskExporter()
        self._video_exporter = VideoExporter()
        self._report_exporter = ReportExporter()

        self._sam_results: Optional[Tuple[List[FrameData], List[PreprocessingResult], List[SegmentationResult], MetricsReport]] = None
        self._sam2_results: Optional[Tuple[List[FrameData], List[PreprocessingResult], List[SegmentationResult], MetricsReport]] = None

    def create_interface(self) -> gr.Blocks:
        """
        Create the main Gradio interface.
        
        Returns:
            Gradio Blocks application.
        """
        with gr.Blocks(
            title="WheatVision2 - SAM vs SAM2 Comparison",
            theme=gr.themes.Soft(),
        ) as app:
            gr.Markdown("# 🌾 WheatVision2 - Segmentation Model Comparison")
            gr.Markdown(
                "Compare SAM (frame-by-frame) vs SAM2 (video propagation) "
                "for wheat ear segmentation."
            )

            with gr.Tabs():
                with gr.Tab("SAM (Frame-by-Frame)"):
                    self._build_sam_tab()

                with gr.Tab("SAM2 (Video Propagation)"):
                    self._build_sam2_tab()

                with gr.Tab("Comparison"):
                    self._build_comparison_tab()

        return app

    def _build_sam_tab(self) -> None:
        """Build the SAM segmentation tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                sam_video = gr.Video(label="Upload Video")

                gr.Markdown("### Preprocessing Settings")
                sam_enable_preprocess = gr.Checkbox(
                    label="Enable Background Removal",
                    value=True,
                )
                with gr.Row():
                    sam_hsv_h_low = gr.Slider(0, 180, value=0, label="H Low")
                    sam_hsv_h_high = gr.Slider(0, 180, value=180, label="H High")
                with gr.Row():
                    sam_hsv_s_low = gr.Slider(0, 255, value=0, label="S Low")
                    sam_hsv_s_high = gr.Slider(0, 255, value=30, label="S High")
                with gr.Row():
                    sam_hsv_v_low = gr.Slider(0, 255, value=200, label="V Low")
                    sam_hsv_v_high = gr.Slider(0, 255, value=255, label="V High")

                gr.Markdown("### Postprocessing Settings")
                sam_enable_postprocess = gr.Checkbox(
                    label="Enable Wheat Ear Filtering",
                    value=True,
                )
                with gr.Row():
                    sam_min_aspect = gr.Slider(1, 10, value=2, label="Min Aspect")
                    sam_max_aspect = gr.Slider(1, 20, value=10, label="Max Aspect")

                gr.Markdown("### Processing")
                sam_max_frames = gr.Slider(
                    1, 100, value=10, step=1,
                    label="Max Frames to Process",
                )
                sam_run_btn = gr.Button("Run SAM Segmentation", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                sam_status = gr.Textbox(label="Status", interactive=False)
                sam_metrics = gr.Textbox(
                    label="Metrics",
                    lines=10,
                    interactive=False,
                )

                with gr.Tabs():
                    with gr.Tab("Preprocessing"):
                        sam_preprocess_gallery = gr.Gallery(
                            label="Preprocessing Steps",
                            columns=3,
                            height=400,
                        )

                    with gr.Tab("Segmentation"):
                        sam_result_gallery = gr.Gallery(
                            label="Segmented Frames",
                            columns=3,
                            height=400,
                        )

                with gr.Row():
                    sam_export_masks_btn = gr.Button("Export Masks (PNG)")
                    sam_export_video_btn = gr.Button("Export Video")
                    sam_export_metrics_btn = gr.Button("Export Metrics")

                sam_export_output = gr.File(label="Exported Files")

        sam_run_btn.click(
            fn=self._run_sam_pipeline,
            inputs=[
                sam_video, sam_enable_preprocess, sam_enable_postprocess,
                sam_hsv_h_low, sam_hsv_s_low, sam_hsv_v_low,
                sam_hsv_h_high, sam_hsv_s_high, sam_hsv_v_high,
                sam_min_aspect, sam_max_aspect, sam_max_frames,
            ],
            outputs=[sam_status, sam_metrics, sam_preprocess_gallery, sam_result_gallery],
        )

        sam_export_masks_btn.click(
            fn=self._export_sam_masks,
            outputs=[sam_export_output],
        )
        sam_export_video_btn.click(
            fn=self._export_sam_video,
            outputs=[sam_export_output],
        )
        sam_export_metrics_btn.click(
            fn=self._export_sam_metrics,
            outputs=[sam_export_output],
        )

    def _build_sam2_tab(self) -> None:
        """Build the SAM2 segmentation tab."""
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

        sam2_run_btn.click(
            fn=self._run_sam2_pipeline,
            inputs=[
                sam2_video, sam2_enable_preprocess, sam2_enable_postprocess,
                sam2_hsv_h_low, sam2_hsv_s_low, sam2_hsv_v_low,
                sam2_hsv_h_high, sam2_hsv_s_high, sam2_hsv_v_high,
                sam2_min_aspect, sam2_max_aspect, sam2_max_frames,
            ],
            outputs=[sam2_status, sam2_metrics, sam2_preprocess_gallery, sam2_result_gallery],
        )

        sam2_export_masks_btn.click(
            fn=self._export_sam2_masks,
            outputs=[sam2_export_output],
        )
        sam2_export_video_btn.click(
            fn=self._export_sam2_video,
            outputs=[sam2_export_output],
        )
        sam2_export_metrics_btn.click(
            fn=self._export_sam2_metrics,
            outputs=[sam2_export_output],
        )

    def _build_comparison_tab(self) -> None:
        """Build the comparison tab."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### SAM vs SAM2 Comparison")
                gr.Markdown(
                    "Run both SAM and SAM2 on the same video, "
                    "then click 'Compare Results' to see metrics comparison."
                )
                compare_btn = gr.Button("Compare Results", variant="primary")
                comparison_text = gr.Textbox(
                    label="Comparison Results",
                    lines=15,
                    interactive=False,
                )
                export_comparison_btn = gr.Button("Export Comparison Report")
                comparison_file = gr.File(label="Comparison Report")

        compare_btn.click(
            fn=self._compare_results,
            outputs=[comparison_text],
        )
        export_comparison_btn.click(
            fn=self._export_comparison,
            outputs=[comparison_file],
        )

    def _run_sam_pipeline(
        self,
        video_path: str,
        enable_preprocess: bool,
        enable_postprocess: bool,
        h_low: int, s_low: int, v_low: int,
        h_high: int, s_high: int, v_high: int,
        min_aspect: float, max_aspect: float,
        max_frames: int,
    ) -> Tuple[str, str, List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]]]:
        """Run SAM pipeline and return results for display."""
        try:
            if not video_path:
                return "Error: No video uploaded", "", [], []

            if self._sam_pipeline is None:
                self._sam_pipeline = SegmentationPipeline(
                    SegmentationModel.SAM,
                    enable_preprocessing=enable_preprocess,
                    enable_postprocessing=enable_postprocess,
                )

            self._sam_pipeline.set_preprocessing_enabled(enable_preprocess)
            self._sam_pipeline.set_postprocessing_enabled(enable_postprocess)

            if enable_preprocess:
                self._sam_pipeline.update_preprocessing_settings(
                    hsv_low=(h_low, s_low, v_low),
                    hsv_high=(h_high, s_high, v_high),
                )

            if enable_postprocess:
                self._sam_pipeline.update_postprocessing_settings(
                    min_aspect=min_aspect,
                    max_aspect=max_aspect,
                )

            frames, preprocess_results, seg_results, metrics = self._sam_pipeline.process_video(
                video_path, max_frames=int(max_frames)
            )

            self._sam_results = (frames, preprocess_results, seg_results, metrics)

            preprocess_images = self._create_preprocess_gallery(frames, preprocess_results)
            result_images = self._create_result_gallery(frames, seg_results)
            metrics_text = self._metrics_calculator.format_report_text(metrics)

            return (
                f"✓ Processed {len(frames)} frames successfully",
                metrics_text,
                preprocess_images,
                result_images,
            )

        except Exception as e:
            return f"Error: {str(e)}", "", [], []

    def _run_sam2_pipeline(
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

            if self._sam2_pipeline is None:
                self._sam2_pipeline = SegmentationPipeline(
                    SegmentationModel.SAM2,
                    enable_preprocessing=enable_preprocess,
                    enable_postprocessing=enable_postprocess,
                )

            self._sam2_pipeline.set_preprocessing_enabled(enable_preprocess)
            self._sam2_pipeline.set_postprocessing_enabled(enable_postprocess)

            if enable_preprocess:
                self._sam2_pipeline.update_preprocessing_settings(
                    hsv_low=(h_low, s_low, v_low),
                    hsv_high=(h_high, s_high, v_high),
                )

            if enable_postprocess:
                self._sam2_pipeline.update_postprocessing_settings(
                    min_aspect=min_aspect,
                    max_aspect=max_aspect,
                )

            frames, preprocess_results, seg_results, metrics = self._sam2_pipeline.process_video(
                video_path, max_frames=int(max_frames)
            )

            self._sam2_results = (frames, preprocess_results, seg_results, metrics)

            preprocess_images = self._create_preprocess_gallery(frames, preprocess_results)
            result_images = self._create_result_gallery(frames, seg_results)
            metrics_text = self._metrics_calculator.format_report_text(metrics)

            return (
                f"✓ Processed {len(frames)} frames successfully",
                metrics_text,
                preprocess_images,
                result_images,
            )

        except Exception as e:
            return f"Error: {str(e)}", "", [], []

    def _create_preprocess_gallery(
        self,
        frames: List[FrameData],
        results: List[PreprocessingResult],
    ) -> List[Tuple[np.ndarray, str]]:
        """Create gallery images for preprocessing visualization."""
        images = []
        for i, (frame, result) in enumerate(zip(frames, results)):
            mask_rgb = cv2.cvtColor(result.foreground_mask, cv2.COLOR_GRAY2RGB)

            vis = np.hstack([frame.image, mask_rgb])
            images.append((vis, f"Frame {i}: Original | Mask"))
        return images

    def _create_result_gallery(
        self,
        frames: List[FrameData],
        results: List[SegmentationResult],
    ) -> List[Tuple[np.ndarray, str]]:
        """Create gallery images for segmentation results."""
        images = []

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        for i, (frame, result) in enumerate(zip(frames, results)):
            overlay = frame.image.copy()
            for j, mask in enumerate(result.masks):
                color = colors[j % len(colors)]
                mask_bool = mask > 0
                overlay[mask_bool] = (
                    0.5 * overlay[mask_bool] + 0.5 * np.array(color)
                ).astype(np.uint8)

            images.append((
                overlay,
                f"Frame {i}: {len(result.masks)} masks ({result.processing_time_ms:.1f}ms)"
            ))

        return images

    def _compare_results(self) -> str:
        """Compare SAM and SAM2 results."""
        if self._sam_results is None:
            return "Please run SAM segmentation first."
        if self._sam2_results is None:
            return "Please run SAM2 segmentation first."

        sam_metrics = self._sam_results[3]
        sam2_metrics = self._sam2_results[3]

        return self._metrics_calculator.format_comparison_text(sam_metrics, sam2_metrics)

    def _export_sam_masks(self) -> Optional[str]:
        """Export SAM masks."""
        if self._sam_results is None:
            return None
        _, _, results, _ = self._sam_results
        return self._export_masks(results, "sam")

    def _export_sam2_masks(self) -> Optional[str]:
        """Export SAM2 masks."""
        if self._sam2_results is None:
            return None
        _, _, results, _ = self._sam2_results
        return self._export_masks(results, "sam2")

    def _export_masks(
        self,
        results: List[SegmentationResult],
        prefix: str,
    ) -> str:
        """Export masks to temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / f"{prefix}_masks"
            paths = self._mask_exporter.export_batch(results, output_dir, ExportFormat.PNG)
            
            import shutil
            export_dir = Path("exports") / f"{prefix}_masks"
            export_dir.mkdir(parents=True, exist_ok=True)
            for p in paths:
                shutil.copy(p, export_dir)
            
            return str(export_dir)

    def _export_sam_video(self) -> Optional[str]:
        """Export SAM video."""
        if self._sam_results is None:
            return None
        frames, _, results, _ = self._sam_results
        output_path = Path("exports/sam_result.mp4")
        self._video_exporter.export_with_overlay(frames, results, output_path)
        return str(output_path)

    def _export_sam2_video(self) -> Optional[str]:
        """Export SAM2 video."""
        if self._sam2_results is None:
            return None
        frames, _, results, _ = self._sam2_results
        output_path = Path("exports/sam2_result.mp4")
        self._video_exporter.export_with_overlay(frames, results, output_path)
        return str(output_path)

    def _export_sam_metrics(self) -> Optional[str]:
        """Export SAM metrics."""
        if self._sam_results is None:
            return None
        _, _, _, metrics = self._sam_results
        output_path = Path("exports/sam_metrics")
        self._report_exporter.export(metrics, output_path, ExportFormat.JSON)
        return str(output_path.with_suffix(".json"))

    def _export_sam2_metrics(self) -> Optional[str]:
        """Export SAM2 metrics."""
        if self._sam2_results is None:
            return None
        _, _, _, metrics = self._sam2_results
        output_path = Path("exports/sam2_metrics")
        self._report_exporter.export(metrics, output_path, ExportFormat.JSON)
        return str(output_path.with_suffix(".json"))

    def _export_comparison(self) -> Optional[str]:
        """Export comparison report."""
        if self._sam_results is None or self._sam2_results is None:
            return None

        sam_metrics = self._sam_results[3]
        sam2_metrics = self._sam2_results[3]
        comparison = self._metrics_calculator.compare_models(sam_metrics, sam2_metrics)

        output_path = Path("exports/comparison_report")
        self._report_exporter.export_comparison(
            sam_metrics, sam2_metrics, comparison, output_path, ExportFormat.JSON
        )
        return str(output_path.with_suffix(".json"))


def launch_app() -> None:
    """Launch the Gradio application."""
    setup_logging(level=logging.INFO)
    _logger.info("Starting WheatVision2 application...")
    
    settings = get_ui_settings()
    app = WheatVisionApp()
    interface = app.create_interface()
    
    _logger.info(f"Launching on http://{settings.host}:{settings.port}")
    interface.launch(
        server_name=settings.host,
        server_port=settings.port,
        share=False,
    )


if __name__ == "__main__":
    launch_app()
