"""
Helper functions for frame-level validation metrics
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def compute_bbox_iou(bbox1: Tuple[float, float, float, float], 
                    bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate IoU between two bounding boxes in (center_x, center_y, width, height) format
    
    Args:
        bbox1: (center_x, center_y, width, height)
        bbox2: (center_x, center_y, width, height)
        
    Returns:
        IoU score between 0 and 1
    """
    # Convert from center format to corner format (x1, y1, x2, y2)
    def center_to_corners(bbox):
        cx, cy, w, h = bbox
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = center_to_corners(bbox1)
    x1_2, y1_2, x2_2, y2_2 = center_to_corners(bbox2)
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    bbox1_area = bbox1[2] * bbox1[3]  # width * height
    bbox2_area = bbox2[2] * bbox2[3]  # width * height
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate IoU between two binary masks
    
    Args:
        mask1: Binary mask (H, W) with values 0 or 255 (or 0/1)
        mask2: Binary mask (H, W) with values 0 or 255 (or 0/1)
        
    Returns:
        IoU score between 0 and 1
    """
    # Ensure masks are same size
    if mask1.shape != mask2.shape:
        return 0.0
    
    # Binarize masks
    mask1_bool = mask1 > 0
    mask2_bool = mask2 > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def distance_to_mask(point: Tuple[float, float], mask: np.ndarray) -> float:
    """
    Calculate minimum distance from a point to nearest True pixel in mask
    
    Args:
        point: (x, y) coordinates
        mask: Binary mask (H, W) with values > 0 indicating foreground
        
    Returns:
        Minimum distance in pixels, or np.inf if mask is empty
    """
    # Find all foreground pixels
    foreground_pixels = np.argwhere(mask > 0)
    
    if len(foreground_pixels) == 0:
        return np.inf
    
    # Convert to (x, y) format - argwhere returns (row, col) = (y, x)
    foreground_coords = foreground_pixels[:, ::-1]  # Swap to (x, y)
    
    # Calculate distances to all foreground pixels
    point_array = np.array([point])
    distances = cdist(point_array, foreground_coords, metric='euclidean')
    
    # Return minimum distance
    min_distance = distances.min()
    return float(min_distance)


def point_in_chamber(point: Tuple[float, float], 
                    chamber_masks: Dict[int, np.ndarray]) -> Optional[int]:
    """
    Determine which chamber instance contains a point
    
    Args:
        point: (x, y) coordinates
        chamber_masks: Dictionary mapping chamber_id to binary mask
        
    Returns:
        chamber_id if point is in a chamber, None otherwise
    """
    x, y = point
    x_int, y_int = int(round(x)), int(round(y))
    
    for chamber_id, mask in chamber_masks.items():
        # Check bounds
        if 0 <= y_int < mask.shape[0] and 0 <= x_int < mask.shape[1]:
            if mask[y_int, x_int] > 0:
                return chamber_id
    
    return None


def distance_between_masks(mask1: np.ndarray, mask2: np.ndarray, 
                          method: str = 'contour') -> float:
    """
    Calculate minimum pixel-to-pixel distance between two binary masks
    
    Args:
        mask1: Binary mask (H, W) with values > 0 indicating foreground
        mask2: Binary mask (H, W) with values > 0 indicating foreground
        method: Distance calculation method
            - 'contour': Use contour points only (fast, accurate for edges)
            - 'downsample': Downsample masks before calculation (fast, approximate)
            - 'full': Use all pixels (slow, exact)
            - 'bbox_filter': Pre-filter with bbox, then use contours (fastest)
        
    Returns:
        Minimum distance in pixels, or np.inf if either mask is empty
    """
    import cv2
    
    if method == 'bbox_filter':
        # First check if bounding boxes are close enough to warrant mask calculation
        bbox1 = bbox_from_mask(mask1)
        bbox2 = bbox_from_mask(mask2)
        
        # Calculate bbox center-to-center distance
        centroid_dist = np.sqrt((bbox1[0] - bbox2[0])**2 + (bbox1[1] - bbox2[1])**2)
        
        # Max possible bbox dimension (conservative estimate)
        max_dim = max(bbox1[2], bbox1[3], bbox2[2], bbox2[3])
        
        # If centroids are very far apart, use centroid distance as approximation
        if centroid_dist > max_dim * 3:
            # Return approximate distance (centroid distance minus half of max dimensions)
            approx_dist = centroid_dist - (bbox1[2] + bbox1[3] + bbox2[2] + bbox2[3]) / 4
            return max(0.0, float(approx_dist))
        
        # Otherwise fall through to contour method
        method = 'contour'
    
    if method == 'contour':
        # Use contour points only - much faster than all pixels
        contours1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours1 or not contours2:
            return np.inf
        
        # Get largest contour from each mask
        contour1 = max(contours1, key=cv2.contourArea)
        contour2 = max(contours2, key=cv2.contourArea)
        
        # Reshape contours to (N, 2)
        coords1 = contour1.reshape(-1, 2)
        coords2 = contour2.reshape(-1, 2)
        
    elif method == 'downsample':
        # Downsample masks by factor of 4 to reduce pixel count
        mask1_small = mask1[::4, ::4]
        mask2_small = mask2[::4, ::4]
        
        pixels1 = np.argwhere(mask1_small > 0)
        pixels2 = np.argwhere(mask2_small > 0)
        
        if len(pixels1) == 0 or len(pixels2) == 0:
            return np.inf
        
        # Convert to (x, y) and scale back up
        coords1 = pixels1[:, ::-1] * 4
        coords2 = pixels2[:, ::-1] * 4
        
    else:  # method == 'full'
        # Find all foreground pixels in both masks
        pixels1 = np.argwhere(mask1 > 0)
        pixels2 = np.argwhere(mask2 > 0)
        
        if len(pixels1) == 0 or len(pixels2) == 0:
            return np.inf
        
        # Convert to (x, y) format - argwhere returns (row, col) = (y, x)
        coords1 = pixels1[:, ::-1]
        coords2 = pixels2[:, ::-1]
    
    # Calculate pairwise distances and find minimum
    distances = cdist(coords1, coords2, metric='euclidean')
    min_distance = distances.min()
    
    return float(min_distance)


def match_by_hungarian(cost_matrix: np.ndarray, 
                      threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match predictions to ground truth using Hungarian algorithm with IoU threshold
    
    Args:
        cost_matrix: Cost matrix where cost_matrix[i, j] = IoU between pred i and GT j
                    (Higher values = better match)
        threshold: Minimum IoU threshold for a valid match
        
    Returns:
        Tuple of:
            - matched_pairs: List of (pred_idx, gt_idx) tuples
            - unmatched_preds: List of prediction indices with no match (FP)
            - unmatched_gts: List of GT indices with no match (FN)
    """
    if cost_matrix.size == 0:
        # No predictions or no ground truth
        n_preds = cost_matrix.shape[0] if len(cost_matrix.shape) > 0 else 0
        n_gts = cost_matrix.shape[1] if len(cost_matrix.shape) > 1 else 0
        return [], list(range(n_preds)), list(range(n_gts))
    
    # Hungarian algorithm minimizes cost, so negate IoU values
    cost_matrix_neg = -cost_matrix
    
    # Run Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix_neg)
    
    # Filter matches by threshold
    matched_pairs = []
    unmatched_pred_set = set(range(cost_matrix.shape[0]))
    unmatched_gt_set = set(range(cost_matrix.shape[1]))
    
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = cost_matrix[pred_idx, gt_idx]
        if iou >= threshold:
            matched_pairs.append((pred_idx, gt_idx))
            unmatched_pred_set.discard(pred_idx)
            unmatched_gt_set.discard(gt_idx)
    
    unmatched_preds = sorted(list(unmatched_pred_set))
    unmatched_gts = sorted(list(unmatched_gt_set))
    
    return matched_pairs, unmatched_preds, unmatched_gts


def bbox_from_mask(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Extract bounding box from a binary mask in (center_x, center_y, width, height) format
    
    Args:
        mask: Binary mask (H, W) with values > 0 indicating foreground
        
    Returns:
        (center_x, center_y, width, height) or (0, 0, 0, 0) if mask is empty
    """
    # Find foreground pixels
    coords = np.argwhere(mask > 0)
    
    if len(coords) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    # coords is in (row, col) = (y, x) format
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]
    
    # Get bounding box
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    
    # Convert to center format
    width = float(x_max - x_min + 1)
    height = float(y_max - y_min + 1)
    center_x = float(x_min + width / 2)
    center_y = float(y_min + height / 2)
    
    return (center_x, center_y, width, height)


def extract_instance_masks(combined_mask: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Extract individual instance masks from a combined instance segmentation mask
    
    Args:
        combined_mask: Instance mask where each unique value > 0 represents a different instance
        
    Returns:
        Dictionary mapping instance_id to binary mask
    """
    instance_masks = {}
    
    # Get unique instance IDs (excluding background = 0)
    instance_ids = np.unique(combined_mask)
    instance_ids = instance_ids[instance_ids > 0]
    
    for instance_id in instance_ids:
        # Create binary mask for this instance
        instance_mask = (combined_mask == instance_id).astype(np.uint8) * 255
        instance_masks[int(instance_id)] = instance_mask
    
    return instance_masks


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the centroid (center of mass) of a binary mask
    
    Args:
        mask: Binary mask (H, W) with values > 0 indicating foreground
        
    Returns:
        (centroid_x, centroid_y) or (0, 0) if mask is empty
    """
    import cv2
    
    # Find all foreground pixels
    coords = np.argwhere(mask > 0)
    
    if len(coords) == 0:
        return (0.0, 0.0)
    
    # Calculate moments for centroid
    moments = cv2.moments(mask.astype(np.uint8))
    
    if moments['m00'] == 0:
        # Fallback to simple average if moments fail
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]
        centroid_x = float(np.mean(x_coords))
        centroid_y = float(np.mean(y_coords))
    else:
        centroid_x = float(moments['m10'] / moments['m00'])
        centroid_y = float(moments['m01'] / moments['m00'])
    
    return (centroid_x, centroid_y)


def mask_to_simplified_polygon(mask: np.ndarray, epsilon_percent: float = 2.0) -> List[List[float]]:
    """
    Convert a binary mask to a simplified polygon contour
    
    Args:
        mask: Binary mask (H, W) with values > 0 indicating foreground
        epsilon_percent: Approximation accuracy as percentage of perimeter (lower = more accurate)
        
    Returns:
        List of [x, y] coordinate pairs, or empty list if no contours found
    """
    import cv2
    
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify polygon using Douglas-Peucker algorithm
    epsilon = (epsilon_percent / 100.0) * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of [x, y] pairs
    polygon = simplified.reshape(-1, 2).tolist()
    
    return polygon


def polygon_to_string(polygon: List[List[float]]) -> str:
    """
    Convert polygon to space-separated string format for CSV storage
    
    Args:
        polygon: List of [x, y] coordinate pairs
        
    Returns:
        String in format "x1 y1 x2 y2 x3 y3 ..."
    """
    if not polygon:
        return ""
    
    coords = []
    for point in polygon:
        coords.append(f"{point[0]:.2f}")
        coords.append(f"{point[1]:.2f}")
    
    return " ".join(coords)
