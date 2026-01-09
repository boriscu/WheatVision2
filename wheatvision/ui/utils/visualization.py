"""Visualization utilities for WheatVision2 UI."""

from typing import List, Tuple

import cv2
import numpy as np

from wheatvision.config.models import FrameData, PreprocessingResult, SegmentationResult


def create_preprocess_gallery(
    frames: List[FrameData],
    results: List[PreprocessingResult],
) -> List[Tuple[np.ndarray, str]]:
    """Create gallery images for preprocessing visualization."""
    images = []
    for i, (frame, result) in enumerate(zip(frames, results)):
        mask_rgb = cv2.cvtColor(result.foreground_mask, cv2.COLOR_GRAY2RGB)
        vis = np.hstack([frame.image, mask_rgb])
        images.append((vis, f"Frame {i}: Original | Mask"))
    return images


def create_result_gallery(
    frames: List[FrameData],
    results: List[SegmentationResult],
) -> List[Tuple[np.ndarray, str]]:
    """Create gallery images for segmentation results."""
    images = []
    color = (0, 255, 0)

    for i, (frame, result) in enumerate(zip(frames, results)):
        overlay = frame.image.copy()
        mask_pixel_count = 0
        
        for mask in result.masks:
            mask_bool = mask > 0
            mask_pixel_count += np.sum(mask_bool)
            
            overlay[mask_bool] = (
                0.5 * overlay[mask_bool] + 0.5 * np.array(color)
            ).astype(np.uint8)
            
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

        images.append((
            overlay,
            f"Frame {i}: {len(result.masks)} masks, {mask_pixel_count} px"
        ))

    return images


def create_comparison_overlay(
    pred_mask: np.ndarray,
    gt_img: np.ndarray,
) -> np.ndarray:
    """
    Create overlay showing prediction vs ground truth.
    
    Colors:
    - Green: Prediction only (false positive)
    - Red: Ground truth only (false negative)
    - Yellow: Overlap (true positive)
    - Black: Background (true negative)
    """
    # Convert prediction to binary
    pred_binary = pred_mask > 0
    
    # Convert ground truth to binary (any non-black pixel is foreground)
    if len(gt_img.shape) == 3:
        gt_binary = np.any(gt_img > 0, axis=2)
    else:
        gt_binary = gt_img > 0
    
    # Create RGB overlay
    height, width = pred_binary.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Overlap (yellow) - both prediction and ground truth
    overlap = np.logical_and(pred_binary, gt_binary)
    overlay[overlap] = [255, 255, 0]  # Yellow
    
    # Prediction only (green) - false positive
    pred_only = np.logical_and(pred_binary, ~gt_binary)
    overlay[pred_only] = [0, 255, 0]  # Green
    
    # Ground truth only (red) - false negative
    gt_only = np.logical_and(~pred_binary, gt_binary)
    overlay[gt_only] = [255, 0, 0]  # Red
    
    return overlay
