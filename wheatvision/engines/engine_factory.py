"""Factory for creating segmentation engines."""

from wheatvision.config.constants import SegmentationModel
from wheatvision.config.settings import get_sam_settings, get_sam2_settings
from wheatvision.engines.base_engine import BaseSegmentationEngine
from wheatvision.engines.sam_engine import SAMEngine
from wheatvision.engines.sam2_engine import SAM2Engine


class SegmentationEngineFactory:
    """
    Factory class for creating segmentation engine instances.
    
    Encapsulates the logic for instantiating the appropriate engine
    based on the requested model type, handling all configuration.
    """

    @staticmethod
    def create(model_type: SegmentationModel) -> BaseSegmentationEngine:
        """
        Create a segmentation engine of the specified type.
        
        Args:
            model_type: The type of segmentation model to create.
            
        Returns:
            Configured segmentation engine instance.
            
        Raises:
            ValueError: If model_type is not recognized.
        """
        if model_type == SegmentationModel.SAM:
            return SAMEngine(get_sam_settings())
        elif model_type == SegmentationModel.SAM2:
            return SAM2Engine(get_sam2_settings())
        else:
            raise ValueError(f"Unknown segmentation model type: {model_type}")

    @staticmethod
    def create_sam() -> SAMEngine:
        """
        Create a SAM engine with default settings.
        
        Returns:
            Configured SAMEngine instance.
        """
        return SAMEngine(get_sam_settings())

    @staticmethod
    def create_sam2() -> SAM2Engine:
        """
        Create a SAM2 engine with default settings.
        
        Returns:
            Configured SAM2Engine instance.
        """
        return SAM2Engine(get_sam2_settings())

    @staticmethod
    def get_available_models() -> list[SegmentationModel]:
        """
        Get list of available segmentation models.
        
        Returns:
            List of supported SegmentationModel values.
        """
        return [SegmentationModel.SAM, SegmentationModel.SAM2]
