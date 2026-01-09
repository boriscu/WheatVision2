"""Main Gradio application for WheatVision2."""

import logging
import tempfile
import shutil
import zipfile

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
from wheatvision.metrics import MetricsCalculator, AccuracyMetrics
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

                with gr.Tab("Ground Truth Evaluation"):
                    self._build_ground_truth_tab()

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
                    1, 100, value=12, step=1,
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

    def _build_ground_truth_tab(self) -> None:
        """Build the ground truth evaluation tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Ground Truth Management")
                
                gt_status = gr.Textbox(
                    label="Ground Truth Status",
                    value=self._get_ground_truth_status(),
                    interactive=False,
                    lines=3,
                )
                refresh_gt_btn = gr.Button("Refresh Ground Truth List")
                
                gr.Markdown("### Upload Ground Truth Masks")
                gt_upload = gr.File(
                    label="Upload PNG Masks",
                    file_count="multiple",
                    file_types=[".png"],
                )
                upload_btn = gr.Button("Save to Ground Truth Folder")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                calculate_metrics_btn = gr.Button(
                    "🔍 Calculate Metrics vs Ground Truth",
                    variant="primary",
                )

            with gr.Column(scale=3):
                gr.Markdown("### 📊 Comparison Summary")
                
                # Summary table at the top
                summary_html = gr.HTML(
                    label="Summary Comparison",
                    value="<p><i>Run segmentation and click 'Calculate Metrics' to see comparison.</i></p>",
                )
                
                gr.Markdown("---")
                gr.Markdown("### 📋 Per-Frame Detailed Metrics")
                
                metrics_table = gr.Dataframe(
                    headers=["Frame", "Model", "Masks", "Avg Size (px)", "Coverage %", "IoU", "Dice", "Precision", "Recall"],
                    label="Per-Frame Metrics",
                    wrap=True,
                )

        gr.Markdown("---")
        gr.Markdown("### 🎨 Visual Comparison Overlays")
        gr.Markdown("🟢 **Green** = Prediction only (FP) | 🔴 **Red** = Ground Truth only (FN) | 🟡 **Yellow** = Overlap (TP)")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Ground Truth Masks")
                gt_gallery = gr.Gallery(
                    label="Ground Truth",
                    columns=6,
                    height=200,
                )
            
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### SAM vs Ground Truth")
                sam_overlay_gallery = gr.Gallery(
                    label="SAM Overlay",
                    columns=6,
                    height=200,
                )
            with gr.Column():
                gr.Markdown("#### SAM2 vs Ground Truth")
                sam2_overlay_gallery = gr.Gallery(
                    label="SAM2 Overlay",
                    columns=6,
                    height=200,
                )

        # Event handlers
        refresh_gt_btn.click(
            fn=self._get_ground_truth_status,
            outputs=[gt_status],
        )
        
        upload_btn.click(
            fn=self._upload_ground_truth,
            inputs=[gt_upload],
            outputs=[upload_status, gt_status],
        )
        
        calculate_metrics_btn.click(
            fn=self._calculate_ground_truth_metrics,
            outputs=[summary_html, metrics_table, gt_gallery, sam_overlay_gallery, sam2_overlay_gallery],
        )

    def _get_ground_truth_status(self) -> str:
        """Get status of ground truth folder."""
        gt_dir = Path("groundtruth")
        if not gt_dir.exists():
            return "Ground truth folder not found. It will be created on first upload."
        
        png_files = sorted(gt_dir.glob("*.png"))
        if not png_files:
            return "Ground truth folder exists but contains no PNG files."
        
        return f"Found {len(png_files)} ground truth masks:\n" + "\n".join(
            f"  - {f.name}" for f in png_files[:10]
        ) + ("\n  ..." if len(png_files) > 10 else "")

    def _upload_ground_truth(
        self,
        files: List[str],
    ) -> Tuple[str, str]:
        """Upload ground truth files to the groundtruth folder."""
        if not files:
            return "No files selected.", self._get_ground_truth_status()
        
        gt_dir = Path("groundtruth")
        gt_dir.mkdir(exist_ok=True)
        
        uploaded = []
        for file_path in files:
            src = Path(file_path)
            dest = gt_dir / src.name
            shutil.copy(src, dest)
            uploaded.append(src.name)
        
        return f"Uploaded {len(uploaded)} files: {', '.join(uploaded)}", self._get_ground_truth_status()

    def _calculate_ground_truth_metrics(
        self,
    ) -> Tuple[str, List[List], List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]]]:
        """Calculate metrics comparing SAM/SAM2 results to ground truth."""
        gt_dir = Path("groundtruth")
        
        empty_html = "<p style='color: red;'>Error: Ground truth not available.</p>"
        if not gt_dir.exists():
            return empty_html, [], [], [], []
        
        gt_files = sorted(gt_dir.glob("*.png"))
        if not gt_files:
            return "<p style='color: red;'>Error: No ground truth PNG files found.</p>", [], [], [], []
        
        gt_images = []
        for gt_file in gt_files:
            img = cv2.imread(str(gt_file))
            if img is not None:
                gt_images.append((gt_file.name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        
        if not gt_images:
            return "<p style='color: red;'>Error: Could not load any ground truth images.</p>", [], [], [], []
        
        gallery_images = [(img, name) for name, img in gt_images]
        
        accuracy_metrics = AccuracyMetrics()
        
        all_rows = []
        sam_summary = None
        sam2_summary = None
        sam_overlays = []
        sam2_overlays = []
        
        if self._sam_results is not None:
            sam_summary, sam_rows, sam_overlays = self._compare_with_ground_truth_extended(
                "SAM", self._sam_results, gt_images, accuracy_metrics
            )
            all_rows.extend(sam_rows)
        
        if self._sam2_results is not None:
            sam2_summary, sam2_rows, sam2_overlays = self._compare_with_ground_truth_extended(
                "SAM2", self._sam2_results, gt_images, accuracy_metrics
            )
            all_rows.extend(sam2_rows)
        
        html = self._build_summary_html(len(gt_images), sam_summary, sam2_summary)
        
        return html, all_rows, gallery_images, sam_overlays, sam2_overlays

    def _compare_with_ground_truth_extended(
        self,
        model_name: str,
        results: Tuple[List[FrameData], List[PreprocessingResult], List[SegmentationResult], MetricsReport],
        gt_images: List[Tuple[str, np.ndarray]],
        accuracy_metrics: AccuracyMetrics,
    ) -> Tuple[dict, List[List], List[Tuple[np.ndarray, str]]]:
        """Compare segmentation results with ground truth masks, returning extended metrics."""
        frames, _, seg_results, _ = results
        
        comparisons = min(len(seg_results), len(gt_images))
        
        all_metrics = []
        all_mask_counts = []
        all_avg_sizes = []
        all_coverages = []
        rows = []
        overlay_images = []
        
        for i in range(comparisons):
            seg_result = seg_results[i]
            gt_name, gt_img = gt_images[i]
            
            # Count masks and calculate average size
            mask_count = len(seg_result.masks)
            all_mask_counts.append(mask_count)
            
            total_mask_pixels = 0
            if seg_result.masks:
                pred_mask = np.zeros_like(seg_result.masks[0], dtype=np.uint8)
                for mask in seg_result.masks:
                    mask_pixels = np.sum(mask > 0)
                    total_mask_pixels += mask_pixels
                    pred_mask[mask > 0] = 255
                avg_size = total_mask_pixels / mask_count if mask_count > 0 else 0
            else:
                if frames:
                    pred_mask = np.zeros((frames[0].height, frames[0].width), dtype=np.uint8)
                else:
                    pred_mask = np.zeros((gt_img.shape[0], gt_img.shape[1]), dtype=np.uint8)
                avg_size = 0
            
            all_avg_sizes.append(avg_size)
            
            total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
            coverage_pct = (total_mask_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            all_coverages.append(coverage_pct)
            
            overlay = self._create_comparison_overlay(pred_mask, gt_img)
            overlay_images.append((overlay, f"Frame {i}: {mask_count} masks"))
            
            try:
                metrics = accuracy_metrics.calculate_against_ground_truth(pred_mask, gt_img)
                all_metrics.append(metrics)
                
                rows.append([
                    f"Frame {i}",
                    model_name,
                    mask_count,
                    f"{avg_size:.0f}",
                    f"{coverage_pct:.1f}",
                    f"{metrics['iou']:.4f}",
                    f"{metrics['dice']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                ])
            except ValueError:
                rows.append([f"Frame {i}", model_name, mask_count, f"{avg_size:.0f}", f"{coverage_pct:.1f}", "Error", "Error", "Error", "Error"])
        
        summary = {}
        if all_metrics:
            summary = {
                "total_masks": sum(all_mask_counts),
                "avg_masks_per_frame": np.mean(all_mask_counts),
                "avg_object_size": np.mean(all_avg_sizes),
                "avg_coverage": np.mean(all_coverages),
                "avg_iou": np.mean([m['iou'] for m in all_metrics]),
                "avg_dice": np.mean([m['dice'] for m in all_metrics]),
                "avg_precision": np.mean([m['precision'] for m in all_metrics]),
                "avg_recall": np.mean([m['recall'] for m in all_metrics]),
            }
        
        return summary, rows, overlay_images

    def _build_summary_html(
        self,
        num_gt_frames: int,
        sam_summary: Optional[dict],
        sam2_summary: Optional[dict],
    ) -> str:
        """Build HTML summary comparison table."""
        html = f"""
        <div style="font-family: Arial, sans-serif;">
            <h3>📊 Evaluation Summary ({num_gt_frames} frames)</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <thead>
                    <tr style="background-color: #f0f0f0;">
                        <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Metric</th>
                        <th style="padding: 10px; border: 1px solid #ddd; text-align: center;">SAM</th>
                        <th style="padding: 10px; border: 1px solid #ddd; text-align: center;">SAM2</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        def get_val(summary, key, fmt=".2f"):
            if summary is None:
                return "<i>N/A</i>"
            return f"{summary.get(key, 0):{fmt}}"
        
        def get_val_int(summary, key):
            if summary is None:
                return "<i>N/A</i>"
            return f"{int(summary.get(key, 0))}"
        
        # Highlight best values
        def compare_higher(s1, s2, key):
            if s1 is None and s2 is None:
                return "", ""
            if s1 is None:
                return "", "background-color: #d4edda;"
            if s2 is None:
                return "background-color: #d4edda;", ""
            if s1.get(key, 0) > s2.get(key, 0):
                return "background-color: #d4edda;", ""
            elif s2.get(key, 0) > s1.get(key, 0):
                return "", "background-color: #d4edda;"
            return "", ""
        
        metrics_config = [
            ("🔢 Total Objects Found", "total_masks", "int", True),
            ("📊 Avg Objects/Frame", "avg_masks_per_frame", ".1f", False),
            ("📐 Avg Object Size (px)", "avg_object_size", ".0f", False),
            ("📏 Avg Coverage (%)", "avg_coverage", ".1f", False),
            ("🎯 Avg IoU", "avg_iou", ".4f", True),
            ("🎯 Avg Dice", "avg_dice", ".4f", True),
            ("✅ Avg Precision", "avg_precision", ".4f", True),
            ("📥 Avg Recall", "avg_recall", ".4f", True),
        ]
        
        for label, key, fmt, highlight in metrics_config:
            style1, style2 = compare_higher(sam_summary, sam2_summary, key) if highlight else ("", "")
            if fmt == "int":
                val1, val2 = get_val_int(sam_summary, key), get_val_int(sam2_summary, key)
            else:
                val1, val2 = get_val(sam_summary, key, fmt), get_val(sam2_summary, key, fmt)
            
            html += f"""
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>{label}</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: center; {style1}">{val1}</td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: center; {style2}">{val2}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            <p style="font-size: 12px; color: #666; margin-top: 10px;">
                <span style="background-color: #d4edda; padding: 2px 6px;">Green</span> = Better value
            </p>
        </div>
        """
        
        return html

    def _create_comparison_overlay(
        self,
        pred_mask: np.ndarray,
        gt_img: np.ndarray,
    ) -> np.ndarray:
        """
        Create overlay showing prediction vs ground truth.
        
        Colors:
        - Green: Prediction only (false positive)
        - Red: Ground truth only (false negative)
        - Yellow: Overlap (true positive)
        - Black: Background (true negative)
        """
        pred_binary = pred_mask > 0
        
        if len(gt_img.shape) == 3:
            gt_binary = np.any(gt_img > 0, axis=2)
        else:
            gt_binary = gt_img > 0
        
        height, width = pred_binary.shape
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        overlap = np.logical_and(pred_binary, gt_binary)
        overlay[overlap] = [255, 255, 0]  # Yellow
        
        pred_only = np.logical_and(pred_binary, ~gt_binary)
        overlay[pred_only] = [0, 255, 0]  # Green
        
        gt_only = np.logical_and(~pred_binary, gt_binary)
        overlay[gt_only] = [255, 0, 0]  # Red
        
        return overlay

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
        import cv2
        images = []

        color = (0, 255, 0)  

        for i, (frame, result) in enumerate(zip(frames, results)):
            overlay = frame.image.copy()
            mask_pixel_count = 0
            
            for mask in result.masks:
                mask_bool = mask > 0
                mask_pixel_count += np.sum(mask_bool)
                
                overlay[mask_bool] = (
                    0.5 * overlay[mask_bool] + 0.5 * np.array(color)
                ).astype(np.uint8)
                
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color, 2)

            images.append((
                overlay,
                f"Frame {i}: {len(result.masks)} masks, {mask_pixel_count} px"
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
        """Export masks as a ZIP file for single download."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / f"{prefix}_masks"
            paths = self._mask_exporter.export_batch(results, output_dir, ExportFormat.PNG)
            
            export_dir = Path("exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            zip_path = export_dir / f"{prefix}_masks.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for p in paths:
                    zipf.write(p, Path(p).name)
            
            return str(zip_path)

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
