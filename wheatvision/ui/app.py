"""Main Gradio application for WheatVision2."""

import gradio as gr

from wheatvision.ui.state import AppState
from wheatvision.ui.tabs import SAMTab, SAM2Tab, ComparisonTab, GroundTruthTab
from wheatvision.utils import setup_logging, get_logger

_logger = get_logger("ui")


class WheatVisionApp:
    """
    Main Gradio application for SAM vs SAM2 comparison.
    
    Provides a tabbed interface for:
    - SAM: Frame-by-frame segmentation
    - SAM2: Video propagation segmentation
    - Comparison: Side-by-side metrics comparison
    - Ground Truth: Evaluation against ground truth masks
    """

    def __init__(self) -> None:
        """Initialize the application."""
        self._state = AppState()
        
        # Initialize tab components with shared state
        self._sam_tab = SAMTab(self._state)
        self._sam2_tab = SAM2Tab(self._state)
        self._comparison_tab = ComparisonTab(self._state)
        self._ground_truth_tab = GroundTruthTab(self._state)

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
                    self._sam_tab.build()

                with gr.Tab("SAM2 (Video Propagation)"):
                    self._sam2_tab.build()

                with gr.Tab("Comparison"):
                    self._comparison_tab.build()

                with gr.Tab("Ground Truth Evaluation"):
                    self._ground_truth_tab.build()

        return app


def launch_app() -> None:
    """Launch the Gradio application."""
    setup_logging()
    _logger.info("Initializing WheatVision2 UI...")
    
    app = WheatVisionApp()
    interface = app.create_interface()
    
    _logger.info("Launching on http://127.0.0.1:7860")
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )


if __name__ == "__main__":
    launch_app()
