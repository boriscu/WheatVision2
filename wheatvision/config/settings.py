"""Pydantic settings classes for WheatVision2 configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from wheatvision.config.constants import SAMModelType


class SAMSettings(BaseSettings):
    """Configuration for the original SAM model."""

    model_config = SettingsConfigDict(
        env_prefix="WHEATVISION_SAM_",
        env_file=".env",
        extra="ignore",
    )

    repo: Path = Path("external/sam_repo")
    checkpoint: Path = Path("external/sam_repo/checkpoints/sam_vit_h.pth")
    model_type: SAMModelType = SAMModelType.VIT_H
    device: str = "cuda"

    @field_validator("repo", "checkpoint", mode="before")
    @classmethod
    def _convert_to_path(cls, value: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(value) if isinstance(value, str) else value


class SAM2Settings(BaseSettings):
    """Configuration for SAM2 video segmentation model."""

    model_config = SettingsConfigDict(
        env_prefix="WHEATVISION_SAM2_",
        env_file=".env",
        extra="ignore",
    )

    repo: Path = Path("external/sam2_repo")
    cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    ckpt: Path = Path("external/sam2_repo/checkpoints/sam2.1_hiera_small.pt")
    device: str = "cuda"

    @field_validator("repo", "ckpt", mode="before")
    @classmethod
    def _convert_to_path(cls, value: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(value) if isinstance(value, str) else value


class PreprocessingSettings(BaseSettings):
    """Configuration for image preprocessing."""

    model_config = SettingsConfigDict(
        env_prefix="WHEATVISION_",
        env_file=".env",
        extra="ignore",
    )

    bg_hsv_low: Union[str, Tuple[int, int, int]] = (0, 0, 200)
    bg_hsv_high: Union[str, Tuple[int, int, int]] = (180, 30, 255)
    roi_min_area_ratio: float = 0.01

    @field_validator("bg_hsv_low", "bg_hsv_high", mode="before")
    @classmethod
    def _parse_hsv_tuple(cls, value: Any) -> Tuple[int, int, int]:
        """Parse comma-separated HSV values from environment."""
        if isinstance(value, str):
            parts = [int(x.strip()) for x in value.split(",")]
            return (parts[0], parts[1], parts[2])
        if isinstance(value, (list, tuple)):
            return (int(value[0]), int(value[1]), int(value[2]))
        return value


class PostprocessingSettings(BaseSettings):
    """Configuration for segmentation postprocessing and filtering."""

    model_config = SettingsConfigDict(
        env_prefix="WHEATVISION_EAR_",
        env_file=".env",
        extra="ignore",
    )

    min_aspect: float = 2.0
    max_aspect: float = 10.0
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.1


class UISettings(BaseSettings):
    """Configuration for the Gradio UI."""

    model_config = SettingsConfigDict(
        env_prefix="WHEATVISION_UI_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 7860


@lru_cache
def get_sam_settings() -> SAMSettings:
    """Get cached SAM settings instance."""
    return SAMSettings()


@lru_cache
def get_sam2_settings() -> SAM2Settings:
    """Get cached SAM2 settings instance."""
    return SAM2Settings()


@lru_cache
def get_preprocessing_settings() -> PreprocessingSettings:
    """Get cached preprocessing settings instance."""
    return PreprocessingSettings()


@lru_cache
def get_postprocessing_settings() -> PostprocessingSettings:
    """Get cached postprocessing settings instance."""
    return PostprocessingSettings()


@lru_cache
def get_ui_settings() -> UISettings:
    """Get cached UI settings instance."""
    return UISettings()
