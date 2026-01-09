"""Mask exporter for saving segmentation masks."""

from pathlib import Path
from typing import List

import cv2
import numpy as np

from wheatvision.config.constants import ExportFormat
from wheatvision.config.models import SegmentationResult
from wheatvision.export.base_exporter import BaseExporter


class MaskExporter(BaseExporter):
    """
    Exports segmentation masks to image or numpy formats.
    
    Supports:
    - PNG: Individual masks as grayscale images
    - NPY: Numpy arrays for programmatic use
    """

    def export(
        self,
        data: SegmentationResult,
        output_path: Path,
        format: ExportFormat = ExportFormat.PNG,
    ) -> Path:
        """
        Export masks from a single segmentation result.
        
        Args:
            data: Segmentation result with masks.
            output_path: Base path for output (frame index added).
            format: Export format (PNG or NPY).
            
        Returns:
            Path to the exported file or directory.
        """
        output_path = Path(output_path)
        self._ensure_parent_exists(output_path)

        if format == ExportFormat.PNG:
            return self._export_as_png(data, output_path)
        elif format == ExportFormat.NPY:
            return self._export_as_npy(data, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_batch(
        self,
        results: List[SegmentationResult],
        output_dir: Path,
        format: ExportFormat = ExportFormat.PNG,
    ) -> List[Path]:
        """
        Export masks from multiple segmentation results.
        
        Args:
            results: List of segmentation results.
            output_dir: Directory for output files.
            format: Export format.
            
        Returns:
            List of paths to exported files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for result in results:
            filename = f"frame_{result.frame_index:05d}"
            output_path = output_dir / filename
            path = self.export(result, output_path, format)
            paths.append(path)

        return paths

    def _export_as_png(
        self,
        result: SegmentationResult,
        base_path: Path,
    ) -> Path:
        """
        Export masks as a binary PNG image.
        
        Creates a combined black/white mask where all segmented areas are white (255)
        and background is black (0).
        
        Args:
            result: Segmentation result.
            base_path: Base path (extension added).
            
        Returns:
            Path to the PNG file.
        """
        output_path = base_path.with_suffix(".png")

        if not result.masks:
            binary_mask = np.zeros((1, 1), dtype=np.uint8)
        else:
            combined = np.zeros_like(result.masks[0], dtype=np.uint8)
            for mask in result.masks:
                combined[mask > 0] = 255 
            binary_mask = combined

        cv2.imwrite(str(output_path), binary_mask)
        return output_path

    def _export_as_npy(
        self,
        result: SegmentationResult,
        base_path: Path,
    ) -> Path:
        """
        Export masks as numpy array.
        
        Saves a dictionary containing masks and scores.
        
        Args:
            result: Segmentation result.
            base_path: Base path (extension added).
            
        Returns:
            Path to the NPY file.
        """
        output_path = base_path.with_suffix(".npz")

        np.savez(
            output_path,
            masks=np.array(result.masks) if result.masks else np.array([]),
            scores=np.array(result.scores),
            frame_index=result.frame_index,
            processing_time_ms=result.processing_time_ms,
        )

        return output_path

    def export_combined_visualization(
        self,
        result: SegmentationResult,
        original_image: np.ndarray,
        output_path: Path,
        alpha: float = 0.5,
    ) -> Path:
        """
        Export visualization of masks overlaid on original image.
        
        Args:
            result: Segmentation result.
            original_image: Original RGB image.
            output_path: Path for output image.
            alpha: Overlay transparency.
            
        Returns:
            Path to the visualization image.
        """
        output_path = Path(output_path).with_suffix(".png")
        self._ensure_parent_exists(output_path)

        overlay = original_image.copy()

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

        for i, mask in enumerate(result.masks):
            color = colors[i % len(colors)]
            mask_bool = mask > 0
            overlay[mask_bool] = (
                (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
            ).astype(np.uint8)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), overlay_bgr)

        return output_path

    def get_supported_formats(self) -> List[str]:
        """Get supported formats."""
        return [ExportFormat.PNG.value, ExportFormat.NPY.value]
