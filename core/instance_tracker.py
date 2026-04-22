"""
Instance tracking for maintaining consistent IDs across video frames.
Uses ByteTrack-inspired matching with IoU-based association.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    """Represents a detection in the current frame"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None  # Binary mask (H, W)
    confidence: Optional[float] = None  # Detection confidence (None for manual)
    source: str = 'unknown'  # 'yolo', 'sam2', 'manual', 'propagated'
    class_id: int = 0  # Class ID (0 for bee)
    class_name: str = 'bee'  # Class name
    instance_id: Optional[int] = None  # Track ID (if assigned by tracker)
    gt_id: Optional[Any] = None  # Ground truth ID (for validation with GT detections)


@dataclass
class Track:
    """Represents a tracked instance across frames"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None
    last_seen_frame: int = 0
    source_history: List[str] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    frames_lost: int = 0  # Consecutive frames without match
    
    def update(self, detection: Detection, frame_idx: int):
        """Update track with new detection"""
        self.bbox = detection.bbox
        self.mask = detection.mask
        self.last_seen_frame = frame_idx
        self.source_history.append(detection.source)
        if detection.confidence is not None:
            self.confidence_history.append(detection.confidence)
        self.frames_lost = 0
        
        # Keep history bounded
        if len(self.source_history) > 30:
            self.source_history = self.source_history[-30:]
        if len(self.confidence_history) > 30:
            self.confidence_history = self.confidence_history[-30:]


class InstanceTracker:
    """
    ByteTrack-inspired tracker for maintaining instance IDs across frames.
    
    Uses two-stage matching:
    1. High-confidence detections with high IoU threshold
    2. Low-confidence detections with low IoU threshold
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tracker with configuration.
        
        Args:
            config: Tracking configuration dictionary
        """
        # Default configuration
        self.config = {
            'conf_threshold_high': 0.5,  # Confidence for high-conf detection pool
            'conf_threshold_low': 0.1,   # Confidence for low-conf detection pool
            'iou_threshold_high': 0.15,  # High-confidence match IoU threshold (lowered for better matching)
            'iou_threshold_low': 0.1,    # Low-confidence match IoU threshold (lowered for better matching)
            'max_frames_lost': 10,       # Frames before track deletion (increased to be more forgiving)
            'min_detection_conf': 0.25,  # Minimum YOLO confidence
            'use_mask_iou': True,        # Use mask IoU vs box IoU
            'match_strategy': 'hungarian',  # 'hungarian' or 'greedy'
            'preserve_manual_ids': True, # Don't reassign manual annotations
            'use_centroid_distance': True,  # Use centroid distance in addition to IoU
            'max_centroid_distance': 600,  # Maximum centroid distance (pixels) to consider a match
            'distance_weight': 0.5,  # Weight for distance vs IoU (0=only IoU, 1=only distance)
        }
        
        if config:
            self.config.update(config)
        
        # Tracking state
        self.active_tracks: Dict[int, Track] = {}
        self.lost_tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.current_frame = 0
        self.frame_history = deque(maxlen=30)
    
    def reset(self):
        """Reset tracker state (e.g., when switching videos)"""
        self.active_tracks = {}
        self.lost_tracks = {}
        self.next_track_id = 1
        self.current_frame = 0
        self.frame_history.clear()
    
    def set_next_track_id(self, next_id: int):
        """Set the next track ID (for continuing existing videos)"""
        self.next_track_id = max(next_id, self.next_track_id)
    
    def match_detections_to_tracks(
        self,
        detections: List[Detection],
        frame_idx: int,
        use_mask_iou: Optional[bool] = None
    ) -> List[Tuple[Detection, int]]:
        """
        Match detections to existing tracks using two-stage matching.
        
        Args:
            detections: List of Detection objects from current frame
            frame_idx: Current frame index
            use_mask_iou: Override config to use mask IoU
            
        Returns:
            List of (Detection, track_id) tuples with assigned IDs
        """
        self.current_frame = frame_idx
        
        if use_mask_iou is None:
            use_mask_iou = self.config['use_mask_iou']
        
        # Separate high and low confidence detections
        high_conf_dets = []
        low_conf_dets = []
        
        for det in detections:
            # Assign default confidence based on source
            if det.confidence is None:
                if det.source == 'manual':
                    det.confidence = 1.0
                elif det.source == 'propagated':
                    det.confidence = 0.7
                else:
                    det.confidence = 0.5
            
            if det.confidence >= self.config['conf_threshold_high']:
                high_conf_dets.append(det)
            else:
                low_conf_dets.append(det)
        
        # Combine active and recently lost tracks
        all_tracks = list(self.active_tracks.values()) + list(self.lost_tracks.values())
        
        print(f"Tracker matching at frame {frame_idx}:")
        print(f"  Active tracks: {list(self.active_tracks.keys())}")
        print(f"  Lost tracks: {list(self.lost_tracks.keys())}")
        print(f"  Total tracks to match: {len(all_tracks)}")
        print(f"  Detections: {len(detections)} (high-conf: {len(high_conf_dets)}, low-conf: {len(low_conf_dets)})")
        
        matched_pairs = []
        unmatched_dets = []
        unmatched_tracks = set(range(len(all_tracks)))
        
        # Stage 1: Match high-confidence detections
        if high_conf_dets and all_tracks:
            print(f"  Stage 1: Matching {len(high_conf_dets)} high-confidence detections (IoU threshold={self.config['iou_threshold_high']})...")
            matches_1, unmatched_d1, unmatched_t1 = self._match_stage(
                high_conf_dets, all_tracks, 
                self.config['iou_threshold_high'],
                use_mask_iou
            )
            matched_pairs.extend(matches_1)
            unmatched_tracks = unmatched_t1
            
            # Stage 2: Match low-confidence detections to remaining tracks
            if low_conf_dets and unmatched_t1:
                print(f"  Stage 2: Matching {len(low_conf_dets)} low-confidence detections to {len(unmatched_t1)} remaining tracks (IoU threshold={self.config['iou_threshold_low']})...")
                remaining_tracks = [all_tracks[i] for i in unmatched_t1]
                matches_2, unmatched_d2, _ = self._match_stage(
                    low_conf_dets, remaining_tracks,
                    self.config['iou_threshold_low'],
                    use_mask_iou
                )
                matched_pairs.extend(matches_2)
                unmatched_dets = [high_conf_dets[i] for i in unmatched_d1] + \
                                 [low_conf_dets[i] for i in unmatched_d2]
            else:
                unmatched_dets = [high_conf_dets[i] for i in unmatched_d1] + low_conf_dets
        else:
            unmatched_dets = high_conf_dets + low_conf_dets
        
        # Update matched tracks
        result = []
        matched_track_ids = []
        for det, track_idx in matched_pairs:
            track = all_tracks[track_idx]
            track.update(det, frame_idx)
            
            # Move lost track back to active
            if track.track_id in self.lost_tracks:
                del self.lost_tracks[track.track_id]
                self.active_tracks[track.track_id] = track
                print(f"  Reactivated lost track ID {track.track_id}")
            
            result.append((det, track.track_id))
            matched_track_ids.append(track.track_id)
        
        # Create new tracks for unmatched detections
        new_track_ids = []
        for det in unmatched_dets:
            new_track = Track(
                track_id=self.next_track_id,
                bbox=det.bbox,
                mask=det.mask,
                last_seen_frame=frame_idx
            )
            new_track.update(det, frame_idx)
            self.active_tracks[self.next_track_id] = new_track
            result.append((det, self.next_track_id))
            new_track_ids.append(self.next_track_id)
            self.next_track_id += 1
        
        # Handle unmatched tracks (increment lost counter)
        lost_track_ids = []
        for track_idx in unmatched_tracks:
            track = all_tracks[track_idx]
            track.frames_lost += 1
            
            # Move to lost tracks if newly lost
            if track.track_id in self.active_tracks:
                del self.active_tracks[track.track_id]
                self.lost_tracks[track.track_id] = track
                lost_track_ids.append(track.track_id)
        
        # Remove tracks lost too long
        to_remove = []
        for track_id, track in self.lost_tracks.items():
            if track.frames_lost > self.config['max_frames_lost']:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.lost_tracks[track_id]
        
        print(f"  Result: Matched {len(matched_track_ids)} tracks {matched_track_ids}, Created {len(new_track_ids)} new tracks {new_track_ids}")
        print(f"  Lost {len(lost_track_ids)} tracks {lost_track_ids}, Removed {len(to_remove)} old tracks {to_remove}")
        
        return result
    
    def _match_stage(
        self,
        detections: List[Detection],
        tracks: List[Track],
        iou_threshold: float,
        use_mask_iou: bool
    ) -> Tuple[List[Tuple[Detection, int]], List[int], set]:
        """
        Perform one stage of matching using Hungarian algorithm.
        
        Returns:
            matched_pairs: List of (Detection, track_index) tuples
            unmatched_detections: Indices of unmatched detections
            unmatched_tracks: Set of unmatched track indices
        """
        if not detections or not tracks:
            return [], list(range(len(detections))), set(range(len(tracks)))
        
        # Compute cost matrix (combined IoU + centroid distance if enabled)
        if self.config['use_centroid_distance']:
            cost_matrix = self._compute_combined_cost_matrix(detections, tracks, use_mask_iou)
            iou_matrix = self._compute_iou_matrix(detections, tracks, use_mask_iou)
        else:
            iou_matrix = self._compute_iou_matrix(detections, tracks, use_mask_iou)
            cost_matrix = 1.0 - iou_matrix
        
        # Apply Hungarian matching
        if self.config['match_strategy'] == 'hungarian':
            matched, unmatched_d, unmatched_t = self._hungarian_matching(
                cost_matrix, iou_matrix, iou_threshold
            )
        else:
            matched, unmatched_d, unmatched_t = self._greedy_matching(
                iou_matrix, iou_threshold
            )
        
        # Convert track indices back to original indices
        matched_pairs = [(detections[d], t) for d, t in matched]
        
        return matched_pairs, unmatched_d, unmatched_t
    
    def _compute_iou_matrix(
        self,
        detections: List[Detection],
        tracks: List[Track],
        use_mask_iou: bool
    ) -> np.ndarray:
        """
        Compute IoU matrix between detections and tracks.
        
        Returns:
            iou_matrix: (N_detections, N_tracks) array of IoU values
        """
        n_det = len(detections)
        n_track = len(tracks)
        iou_matrix = np.zeros((n_det, n_track), dtype=np.float32)
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                if use_mask_iou and det.mask is not None and track.mask is not None:
                    iou_matrix[i, j] = self._mask_iou(det.mask, track.mask)
                else:
                    iou_matrix[i, j] = self._box_iou(det.bbox, track.bbox)
        
        return iou_matrix
    
    def _compute_combined_cost_matrix(
        self,
        detections: List[Detection],
        tracks: List[Track],
        use_mask_iou: bool
    ) -> np.ndarray:
        """
        Compute combined cost matrix using IoU and centroid distance.
        Lower cost = better match.
        
        Returns:
            cost_matrix: (N_detections, N_tracks) array of costs
        """
        n_det = len(detections)
        n_track = len(tracks)
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(detections, tracks, use_mask_iou)
        
        if not self.config['use_centroid_distance']:
            # Standard IoU-only cost
            return 1.0 - iou_matrix
        
        # Compute centroid distances
        distance_matrix = np.zeros((n_det, n_track), dtype=np.float32)
        max_dist = self.config['max_centroid_distance']
        
        for i, det in enumerate(detections):
            det_center = self._get_bbox_center(det.bbox)
            for j, track in enumerate(tracks):
                track_center = self._get_bbox_center(track.bbox)
                distance = np.linalg.norm(det_center - track_center)
                
                # Normalize distance (0 = same position, 1 = max_centroid_distance away)
                distance_matrix[i, j] = min(distance / max_dist, 1.0)
        
        # Combine IoU and distance costs
        # cost_iou: 0 (perfect overlap) to 1 (no overlap)
        # cost_distance: 0 (same position) to 1 (far away)
        weight = self.config['distance_weight']
        cost_iou = 1.0 - iou_matrix
        cost_distance = distance_matrix
        
        # Combined cost: weighted average
        combined_cost = (1 - weight) * cost_iou + weight * cost_distance
        
        return combined_cost
    
    def _hungarian_matching(
        self,
        cost_matrix: np.ndarray,
        iou_matrix: np.ndarray,
        iou_threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], set]:
        """
        Perform Hungarian algorithm matching using cost matrix, filter by IoU.
        
        Args:
            cost_matrix: Cost matrix to minimize (lower = better match)
            iou_matrix: IoU matrix for threshold filtering
            iou_threshold: Minimum IoU to accept a match
        
        Returns:
            matches: List of (detection_idx, track_idx) tuples
            unmatched_detections: List of detection indices
            unmatched_tracks: Set of track indices
        """
        # Run Hungarian algorithm on cost matrix
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter by IoU threshold
        matches = []
        matched_dets = set()
        matched_tracks = set()
        
        for d, t in zip(det_indices, track_indices):
            iou_value = iou_matrix[d, t]
            cost_value = cost_matrix[d, t]
            
            # Accept match if IoU is above threshold
            if iou_value >= iou_threshold:
                matches.append((d, t))
                matched_dets.add(d)
                matched_tracks.add(t)
                print(f"  Matched detection {d} to track {t} with IoU={iou_value:.3f}, cost={cost_value:.3f}")
            else:
                print(f"  Rejected match: detection {d} to track {t} with IoU={iou_value:.3f}, cost={cost_value:.3f} (IoU below threshold {iou_threshold})")
        
        # Find unmatched
        unmatched_dets = [i for i in range(cost_matrix.shape[0]) if i not in matched_dets]
        unmatched_tracks = set(range(cost_matrix.shape[1])) - matched_tracks
        
        return matches, unmatched_dets, unmatched_tracks
    
    def _greedy_matching(
        self,
        iou_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], set]:
        """
        Perform greedy matching (highest IoU first).
        
        Returns:
            matches: List of (detection_idx, track_idx) tuples
            unmatched_detections: List of detection indices
            unmatched_tracks: Set of track indices
        """
        matches = []
        matched_dets = set()
        matched_tracks = set()
        
        # Get all (det, track, iou) tuples sorted by IoU
        candidates = []
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                if iou_matrix[i, j] >= threshold:
                    candidates.append((i, j, iou_matrix[i, j]))
        
        # Sort by IoU descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy matching
        for d, t, iou in candidates:
            if d not in matched_dets and t not in matched_tracks:
                matches.append((d, t))
                matched_dets.add(d)
                matched_tracks.add(t)
        
        # Find unmatched
        unmatched_dets = [i for i in range(iou_matrix.shape[0]) if i not in matched_dets]
        unmatched_tracks = set(range(iou_matrix.shape[1])) - matched_tracks
        
        return matches, unmatched_dets, unmatched_tracks
    
    @staticmethod
    def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes [x1, y1, x2, y2].
        """
        # Intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def _get_bbox_center(bbox: np.ndarray) -> np.ndarray:
        """
        Get the center point of a bounding box [x1, y1, x2, y2].
        
        Returns:
            center: [cx, cy] numpy array
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return np.array([cx, cy], dtype=np.float32)
    
    @staticmethod
    def _mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute IoU between two binary masks.
        """
        # Ensure same shape
        if mask1.shape != mask2.shape:
            return 0.0
        
        # Convert to binary
        mask1_bin = (mask1 > 0).astype(bool)
        mask2_bin = (mask2 > 0).astype(bool)
        
        # Intersection and union
        intersection = np.logical_and(mask1_bin, mask2_bin).sum()
        union = np.logical_or(mask1_bin, mask2_bin).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    def get_track_info(self, track_id: int) -> Optional[Track]:
        """Get information about a specific track"""
        if track_id in self.active_tracks:
            return self.active_tracks[track_id]
        if track_id in self.lost_tracks:
            return self.lost_tracks[track_id]
        return None
    
    def get_active_track_ids(self) -> List[int]:
        """Get list of currently active track IDs"""
        return list(self.active_tracks.keys())
