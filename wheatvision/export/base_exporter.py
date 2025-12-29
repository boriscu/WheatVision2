"""Abstract base class for exporters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List


class BaseExporter(ABC):
    """
    Abstract base class for result exporters.
    
    Exporters handle saving segmentation results, metrics,
    and visualizations to various file formats.
    """

    @abstractmethod
    def export(self, data: Any, output_path: Path) -> Path:
        """
        Export data to a file.
        
        Args:
            data: Data to export.
            output_path: Path for the output file.
            
        Returns:
            Path to the exported file.
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of file extensions (e.g., ["png", "npy"]).
        """
        pass

    def _ensure_parent_exists(self, path: Path) -> None:
        """Create parent directories if they don't exist."""
        path.parent.mkdir(parents=True, exist_ok=True)
