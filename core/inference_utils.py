"""Utility functions for temporal inference in validation workflows."""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional


def create_temporal_image(current_frame_path: Path, all_video_frames: Optional[List[Path]] = None) -> Optional[np.ndarray]:
    """
    Create temporal image from prev/current/next frames.
    
    Args:
        current_frame_path: Path to the current frame
        all_video_frames: Optional pre-sorted list of all frames in video. 
                         If None, will scan the directory.
    
    Returns:
        np.ndarray: 3-channel RGB image where R=prev, G=current, B=next (all grayscale)
                   Returns None if frame cannot be loaded
    """
    current_frame_path = Path(current_frame_path)
    
    # Get all frames in this video if not provided
    if all_video_frames is None:
        video_frames_dir = current_frame_path.parent
        all_video_frames = sorted(
            list(video_frames_dir.glob('*.jpg')) + 
            list(video_frames_dir.glob('*.png'))
        )
    
    # Find current frame index
    try:
        current_idx = all_video_frames.index(current_frame_path)
    except ValueError:
        # Frame not in list, use current frame for all channels
        current_img = cv2.imread(str(current_frame_path), cv2.IMREAD_GRAYSCALE)
        if current_img is None:
            return None
        return np.stack([current_img, current_img, current_img], axis=2)
    
    # Get previous frame (or duplicate current if at start)
    if current_idx > 0:
        prev_frame_path = all_video_frames[current_idx - 1]
    else:
        prev_frame_path = current_frame_path
    
    # Get next frame (or duplicate current if at end)
    if current_idx < len(all_video_frames) - 1:
        next_frame_path = all_video_frames[current_idx + 1]
    else:
        next_frame_path = current_frame_path
    
    # Load frames as grayscale
    prev_img = cv2.imread(str(prev_frame_path), cv2.IMREAD_GRAYSCALE)
    current_img = cv2.imread(str(current_frame_path), cv2.IMREAD_GRAYSCALE)
    next_img = cv2.imread(str(next_frame_path), cv2.IMREAD_GRAYSCALE)
    
    if prev_img is None or current_img is None or next_img is None:
        return None
    
    # Ensure all frames have the same dimensions
    h, w = current_img.shape
    if prev_img.shape != (h, w):
        prev_img = cv2.resize(prev_img, (w, h))
    if next_img.shape != (h, w):
        next_img = cv2.resize(next_img, (w, h))
    
    # Stack as RGB channels: R=prev, G=current, B=next
    temporal_img = np.stack([prev_img, current_img, next_img], axis=2)
    
    return temporal_img
