"""Enum constants and type definitions for WheatVision2."""

from enum import Enum


class SegmentationModel(str, Enum):
    """Available segmentation model types."""

    SAM = "sam"
    SAM2 = "sam2"


class ExportFormat(str, Enum):
    """Supported export formats for results."""

    PNG = "png"
    NPY = "npy"
    MP4 = "mp4"
    JSON = "json"
    CSV = "csv"


class PreprocessingStep(str, Enum):
    """Preprocessing step identifiers."""

    BACKGROUND_REMOVAL = "background_removal"
    ROI_DETECTION = "roi_detection"


class PostprocessingStep(str, Enum):
    """Postprocessing step identifiers."""

    WHEAT_EAR_FILTER = "wheat_ear_filter"
    MASK_REFINEMENT = "mask_refinement"


class SAMModelType(str, Enum):
    """SAM model architecture variants."""

    VIT_H = "vit_h"
    VIT_L = "vit_l"
    VIT_B = "vit_b"
