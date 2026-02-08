"""Comparison tab for WheatVision2 UI."""

from pathlib import Path
from typing import Optional

import gradio as gr

from wheatvision.config.constants import ExportFormat
from wheatvision.ui.state import AppState


class ComparisonTab:
    """SAM vs SAM2 vs SAM3 comparison tab component."""

    def __init__(self, state: AppState) -> None:
        """Initialize with shared state."""
        self._state = state

    def build(self) -> None:
        """Build the comparison tab UI."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### SAM vs SAM2 vs SAM3 Comparison")
                gr.Markdown(
                    "Run SAM, SAM2, and/or SAM3 on the same video, "
                    "then click 'Compare Results' to see metrics comparison."
                )
                compare_btn = gr.Button("Compare Results", variant="primary")
                comparison_text = gr.Textbox(
                    label="Comparison Results",
                    lines=20,
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

    def _compare_results(self) -> str:
        """Compare SAM, SAM2, and SAM3 results."""
        results = []
        models = []
        
        if self._state.sam_results is not None:
            results.append(("SAM", self._state.sam_results[3]))
            models.append("SAM")
        if self._state.sam2_results is not None:
            results.append(("SAM2", self._state.sam2_results[3]))
            models.append("SAM2")
        if self._state.sam3_results is not None:
            results.append(("SAM3", self._state.sam3_results[3]))
            models.append("SAM3")
        
        if len(results) == 0:
            return "Please run at least one segmentation model first."
        
        if len(results) == 1:
            return f"Only {models[0]} results available. Run at least one more model for comparison."
        
        # Build comparison text
        lines = [f"=== Model Comparison ({', '.join(models)}) ===", ""]
        
        # Speed comparison
        lines.append("Speed Metrics:")
        for name, metrics in results:
            lines.append(f"  {name}:")
            lines.append(f"    FPS: {metrics.speed_metrics.fps:.2f}")
            lines.append(f"    Total Time: {metrics.speed_metrics.total_processing_time_ms:.2f} ms")
            lines.append(f"    Avg Time/Frame: {metrics.speed_metrics.avg_time_per_frame_ms:.2f} ms")
        
        # Find fastest
        fastest = min(results, key=lambda x: x[1].speed_metrics.total_processing_time_ms)
        lines.append(f"  → Fastest: {fastest[0]}")
        lines.append("")
        
        # Accuracy comparison
        lines.append("Accuracy Metrics:")
        for name, metrics in results:
            lines.append(f"  {name}:")
            lines.append(f"    Avg Masks/Frame: {metrics.accuracy_metrics.avg_masks_per_frame:.2f}")
            lines.append(f"    Temporal Consistency: {metrics.accuracy_metrics.temporal_consistency_score:.4f}")
            lines.append(f"    Coverage Ratio: {metrics.accuracy_metrics.coverage_ratio:.4f}")
        
        # Find most consistent
        most_consistent = max(results, key=lambda x: x[1].accuracy_metrics.temporal_consistency_score)
        lines.append(f"  → Most Consistent: {most_consistent[0]}")
        
        return "\n".join(lines)

    def _export_comparison(self) -> Optional[str]:
        """Export comparison report."""
        results = []
        
        if self._state.sam_results is not None:
            results.append(("sam", self._state.sam_results[3]))
        if self._state.sam2_results is not None:
            results.append(("sam2", self._state.sam2_results[3]))
        if self._state.sam3_results is not None:
            results.append(("sam3", self._state.sam3_results[3]))
        
        if len(results) < 2:
            return None
        
        output_path = Path("exports/comparison_report")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For now, export as two-way comparison if we have SAM and SAM2
        # TODO: Extend report exporter for three-way comparison
        if self._state.sam_results is not None and self._state.sam2_results is not None:
            sam_metrics = self._state.sam_results[3]
            sam2_metrics = self._state.sam2_results[3]
            comparison = self._state.metrics_calculator.compare_models(sam_metrics, sam2_metrics)
            self._state.report_exporter.export_comparison(
                sam_metrics, sam2_metrics, comparison, output_path, ExportFormat.JSON
            )
            return str(output_path.with_suffix(".json"))
        
        return None

