"""Ground truth evaluation tab for WheatVision2 UI."""

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from wheatvision.config.models import FrameData, MetricsReport, PreprocessingResult, SegmentationResult
from wheatvision.metrics import AccuracyMetrics
from wheatvision.ui.state import AppState
from wheatvision.ui.utils import create_comparison_overlay


class GroundTruthTab:
    """Ground truth evaluation tab component."""

    def __init__(self, state: AppState) -> None:
        """Initialize with shared state."""
        self._state = state

    def build(self) -> None:
        """Build the ground truth evaluation tab UI."""
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

    def _upload_ground_truth(self, files: List[str]) -> Tuple[str, str]:
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
        
        if self._state.sam_results is not None:
            sam_summary, sam_rows, sam_overlays = self._compare_with_ground_truth_extended(
                "SAM", self._state.sam_results, gt_images, accuracy_metrics
            )
            all_rows.extend(sam_rows)
        
        if self._state.sam2_results is not None:
            sam2_summary, sam2_rows, sam2_overlays = self._compare_with_ground_truth_extended(
                "SAM2", self._state.sam2_results, gt_images, accuracy_metrics
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
        """Compare segmentation results with ground truth masks."""
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
            
            overlay = create_comparison_overlay(pred_mask, gt_img)
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
