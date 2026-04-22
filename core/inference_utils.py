"""
Utility functions for BeeHaveSquE inference
Extracted to avoid GUI dependencies in validation worker
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from core.instance_tracker import Detection


def create_temporal_image(current_frame_path: Path, all_video_frames: Optional[List[Path]] = None) -> Optional[np.ndarray]:
    """
    Create temporal image from prev/current/next frames for BeeHaveSquE inference
    
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


def run_beehavesque_soho(
    model,
    temporal_img: np.ndarray,
    sahi_params: dict,
    edge_filter: int = 50,
    device: str = None
) -> List[Detection]:
    """
    Run BeeHaveSquE SOHO (Sliced Overlapping Heuristic Optimization) inference
    
    Args:
        model: YOLO model instance
        temporal_img: Temporal image (prev/current/next as RGB)
        sahi_params: Dict with keys: slice_height, slice_width, overlap_height_ratio, overlap_width_ratio
        edge_filter: Pixels to filter from slice edges (default: 50)
        device: Device to run inference on ('cpu' or 'cuda' or 'cuda:0', etc.). If None, uses model's current device.
    
    Returns:
        List of Detection objects with masks and bounding boxes
    """
    import torch
    
    # Determine device to use
    if device is None:
        # Try to infer from model or default to CPU to avoid device errors in background threads
        try:
            if hasattr(model, 'device'):
                device = str(model.device)
            else:
                device = 'cpu'  # Safest default for background workers
        except:
            device = 'cpu'
    
    # For validation workers and background threads, always use CPU or cuda:0 explicitly
    # to avoid "Invalid device id" errors with multi-GPU systems or thread contexts
    if device != 'cpu' and 'cuda' in device:
        # Normalize CUDA device specification
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        elif ':' not in device:
            # Ensure we use cuda:0 explicitly to avoid device index issues
            device = 'cuda:0'
    
    print(f"Running BeeHaveSquE SOHO inference on device: {device}")
    orig_h, orig_w = temporal_img.shape[:2]
    
    # Padding
    pad_size = 100
    padded_img = cv2.copyMakeBorder(
        temporal_img,
        pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_CONSTANT,
        value=0
    )
    padded_h, padded_w = padded_img.shape[:2]
    
    # Generate slice positions
    slice_h = sahi_params['slice_height']
    slice_w = sahi_params['slice_width']
    overlap_h = sahi_params['overlap_height_ratio']
    overlap_w = sahi_params['overlap_width_ratio']
    
    stride_h = int(slice_h * (1 - overlap_h))
    stride_w = int(slice_w * (1 - overlap_w))
    
    slice_positions = []
    y = 0
    while y < padded_h:
        x = 0
        while x < padded_w:
            y2 = min(y + slice_h, padded_h)
            x2 = min(x + slice_w, padded_w)
            y1 = max(0, y2 - slice_h)
            x1 = max(0, x2 - slice_w)
            slice_positions.append((x1, y1, x2, y2))
            if x2 >= padded_w:
                break
            x += stride_w
        if y2 >= padded_h:
            break
        y += stride_h
    
    # Process each slice
    all_detections = []
    for x1, y1, x2, y2 in slice_positions:
        slice_img = padded_img[y1:y2, x1:x2]
        
        # Run model with explicit device to avoid device ID errors in background threads
        results = model.predict(
            source=slice_img,
            conf=0.5,
            iou=0.5,
            verbose=False,
            retina_masks=True,
            device=device  # Explicitly specify device to avoid "Invalid device id" errors
        )
        
        if not results or len(results) == 0:
            continue
        
        result = results[0]
        
        # Convert to Detection objects
        slice_detections = yolo_results_to_detections(result, model)
        
        # Filter detections near slice edges
        slice_h_actual = y2 - y1
        slice_w_actual = x2 - x1
        
        for detection in slice_detections:
            bbox = detection.bbox  # [x1_local, y1_local, x2_local, y2_local]
            
            # Check if detection is too close to any edge
            is_too_close = (
                bbox[0] < edge_filter or 
                bbox[2] > slice_w_actual - edge_filter or
                bbox[1] < edge_filter or 
                bbox[3] > slice_h_actual - edge_filter
            )
            
            if is_too_close:
                continue
            
            # Transform coordinates from slice to padded image space
            detection.bbox = [
                bbox[0] + x1,
                bbox[1] + y1,
                bbox[2] + x1,
                bbox[3] + y1
            ]
            
            # Transform mask to padded image coordinates
            full_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = detection.mask
            detection.mask = full_mask
            
            all_detections.append(detection)
    
    # Remove padding offset from final detections
    final_detections = []
    for detection in all_detections:
        bbox = detection.bbox
        
        # Transform bbox back to original image coordinates
        bbox_orig = [
            max(0, bbox[0] - pad_size),
            max(0, bbox[1] - pad_size),
            min(orig_w, bbox[2] - pad_size),
            min(orig_h, bbox[3] - pad_size)
        ]
        
        # Crop mask to original image region
        mask_orig = detection.mask[pad_size:pad_size+orig_h, pad_size:pad_size+orig_w]
        
        # Skip if mask is empty after cropping
        if np.sum(mask_orig > 0) == 0:
            continue
        
        detection.bbox = bbox_orig
        detection.mask = mask_orig
        final_detections.append(detection)
    
    # Apply NMS to remove duplicate detections from overlapping slices
    final_detections = mask_nms(final_detections, iou_threshold=0.5)
    
    return final_detections


def mask_nms(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """
    Apply Non-Maximum Suppression using mask IoU
    
    Args:
        detections: List of Detection objects
        iou_threshold: IoU threshold for considering detections as duplicates
    
    Returns:
        List of Detection objects after NMS
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda d: d.confidence if d.confidence is not None else 0.5, reverse=True)
    
    keep = []
    suppress = set()
    
    for i, det_i in enumerate(sorted_dets):
        if i in suppress:
            continue
        
        keep.append(det_i)
        
        # Compare with remaining detections
        for j in range(i + 1, len(sorted_dets)):
            if j in suppress:
                continue
            
            det_j = sorted_dets[j]
            
            # Compute mask IoU
            iou = compute_mask_iou(det_i.mask, det_j.mask)
            
            if iou > iou_threshold:
                suppress.add(j)
    
    print(f"  NMS: {len(detections)} -> {len(keep)} detections (removed {len(detections) - len(keep)} duplicates)")
    
    return keep


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two masks
    
    Args:
        mask1: First mask (2D array)
        mask2: Second mask (2D array)
    
    Returns:
        IoU value between 0 and 1
    """
    if mask1 is None or mask2 is None:
        return 0.0
    
    # Ensure 2D
    if len(mask1.shape) > 2:
        mask1 = mask1.squeeze()
    if len(mask2.shape) > 2:
        mask2 = mask2.squeeze()
    
    # Check shape compatibility
    if mask1.shape != mask2.shape:
        return 0.0
    
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def yolo_results_to_detections(yolo_result, model) -> List[Detection]:
    """
    Convert YOLO results to Detection objects
    
    Args:
        yolo_result: Single YOLO result object
        model: YOLO model (for class names)
    
    Returns:
        List of Detection objects
    """
    detections = []
    if yolo_result.masks is not None:
        masks = yolo_result.masks.data.cpu().numpy()
        boxes = yolo_result.boxes.xyxy.cpu().numpy()
        confidences = yolo_result.boxes.conf.cpu().numpy()
        class_ids = yolo_result.boxes.cls.cpu().numpy() if yolo_result.boxes.cls is not None else None
        
        # Get original image shape from result
        orig_shape = yolo_result.orig_shape  # (height, width)
        
        for i in range(len(masks)):
            # With retina_masks=True, masks are already at original image size
            mask = masks[i]
            
            # Convert to uint8 (0-255 range)
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            # Ensure 2D
            if len(mask.shape) == 3:
                mask = mask.squeeze()
            
            # Get bbox
            bbox = boxes[i].tolist()  # [x1, y1, x2, y2]
            
            # Get confidence
            confidence = float(confidences[i])
            
            # Get class
            class_id = int(class_ids[i]) if class_ids is not None else 0
            class_name = model.names[class_id] if hasattr(model, 'names') else f"class_{class_id}"
            
            # Create Detection object
            detection = Detection(
                bbox=bbox,
                mask=mask,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name
            )
            
            detections.append(detection)
    
    return detections
