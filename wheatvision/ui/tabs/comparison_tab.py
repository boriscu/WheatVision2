"""Comparison tab for WheatVision2 UI."""

from pathlib import Path
from typing import Optional

import gradio as gr

from wheatvision.config.constants import ExportFormat
from wheatvision.ui.state import AppState


class ComparisonTab:
    """SAM vs SAM2 comparison tab component."""

    def __init__(self, state: AppState) -> None:
        """Initialize with shared state."""
        self._state = state

    def build(self) -> None:
        """Build the comparison tab UI."""
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

    def _compare_results(self) -> str:
        """Compare SAM and SAM2 results."""
        if self._state.sam_results is None:
            return "Please run SAM segmentation first."
        if self._state.sam2_results is None:
            return "Please run SAM2 segmentation first."

        sam_metrics = self._state.sam_results[3]
        sam2_metrics = self._state.sam2_results[3]

        return self._state.metrics_calculator.format_comparison_text(sam_metrics, sam2_metrics)

    def _export_comparison(self) -> Optional[str]:
        """Export comparison report."""
        if self._state.sam_results is None or self._state.sam2_results is None:
            return None
        
        sam_metrics = self._state.sam_results[3]
        sam2_metrics = self._state.sam2_results[3]
        comparison = self._state.metrics_calculator.compare_models(sam_metrics, sam2_metrics)
        
        output_path = Path("exports/comparison_report")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._state.report_exporter.export_comparison(
            sam_metrics, sam2_metrics, comparison, output_path, ExportFormat.JSON
        )
        return str(output_path.with_suffix(".json"))
