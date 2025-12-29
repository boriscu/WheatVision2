"""Abstract base class for metrics calculation."""

from abc import ABC, abstractmethod
from typing import Any, List

from wheatvision.config.models import SegmentationResult


class BaseMetric(ABC):
    """
    Abstract base class for segmentation metrics.
    
    All metrics implement a common interface for calculating
    performance or quality measurements from segmentation results.
    """

    @abstractmethod
    def calculate(
        self,
        results: List[SegmentationResult],
        **kwargs: Any,
    ) -> Any:
        """
        Calculate the metric from segmentation results.
        
        Args:
            results: List of segmentation results.
            **kwargs: Additional metric-specific parameters.
            
        Returns:
            Metric report or value.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this metric.
        
        Returns:
            Human-readable metric name.
        """
        pass
