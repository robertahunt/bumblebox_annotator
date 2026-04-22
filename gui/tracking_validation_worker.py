"""
Worker thread for tracking algorithm validation
"""

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import csv
import cv2
import os
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import copy

from core.instance_tracker import InstanceTracker, Detection, Track


class SimpleIoUTracker:
    """Simple baseline tracker using greedy IoU matching"""
    
    def __init__(self, iou_threshold=0.5, use_mask_iou=True):
        self.iou_threshold = iou_threshold
        self.use_mask_iou = use_mask_iou
        self.tracks = {}
        self.next_track_id = 1
    
    def reset(self):
        self.tracks = {}
        self.next_track_id = 1
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update with new detections, return detections with IDs"""
        if not self.tracks:
            # First frame - assign sequential IDs
            for det in detections:
                det.instance_id = self.next_track_id
                self.tracks[self.next_track_id] = det
                self.next_track_id += 1
            return detections
        
        # Match detections to existing tracks
        if not detections:
            return []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        track_ids = list(self.tracks.keys())
        
        for d_idx, det in enumerate(detections):
            for t_idx, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                if self.use_mask_iou and det.mask is not None and track.mask is not None:
                    iou = self._mask_iou(det.mask, track.mask)
                else:
                    iou = self._bbox_iou(det.bbox, track.bbox)
                iou_matrix[d_idx, t_idx] = iou
        
        # Greedy matching
        matched_detections = []
        used_tracks = set()
        
        # Sort by highest IoU
        flat_ious = [(iou_matrix[d, t], d, t) for d in range(len(detections)) 
                     for t in range(len(track_ids))]
        flat_ious.sort(reverse=True)
        
        matched_dets = set()
        for iou, d_idx, t_idx in flat_ious:
            if d_idx in matched_dets or t_idx in used_tracks:
                continue
            if iou >= self.iou_threshold:
                track_id = track_ids[t_idx]
                detections[d_idx].instance_id = track_id
                self.tracks[track_id] = detections[d_idx]
                matched_dets.add(d_idx)
                used_tracks.add(t_idx)
        
        # New tracks for unmatched
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                det.instance_id = self.next_track_id
                self.tracks[self.next_track_id] = det
                self.next_track_id += 1
        
        return detections
    
    def _bbox_iou(self, bbox1, bbox2):
        """Compute bbox IoU"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _mask_iou(self, mask1, mask2):
        """Compute mask IoU"""
        # Ensure same shape
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        return intersection / union if union > 0 else 0.0


class CentroidTracker:
    """Simple centroid distance tracker with track timeout"""
    
    def __init__(self, max_distance=600, max_frames_missing=1):
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing  # Remove tracks not seen for N frames
        self.tracks = {}  # track_id -> centroid
        self.frames_since_update = {}  # track_id -> frames without match
        self.next_track_id = 1
    
    def reset(self):
        self.tracks = {}
        self.frames_since_update = {}
        self.next_track_id = 1
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update with new detections"""
        if not self.tracks:
            for det in detections:
                det.instance_id = self.next_track_id
                centroid = self._get_centroid(det.bbox)
                self.tracks[self.next_track_id] = centroid
                self.frames_since_update[self.next_track_id] = 0
                self.next_track_id += 1
            return detections
        
        if not detections:
            # Increment age for all tracks
            for track_id in list(self.frames_since_update.keys()):
                self.frames_since_update[track_id] += 1
            # Remove old tracks
            self._remove_old_tracks()
            return []
        
        # Compute distance matrix
        dist_matrix = np.zeros((len(detections), len(self.tracks)))
        track_ids = list(self.tracks.keys())
        
        for d_idx, det in enumerate(detections):
            det_centroid = self._get_centroid(det.bbox)
            for t_idx, track_id in enumerate(track_ids):
                track_centroid = self.tracks[track_id]
                dist = np.linalg.norm(np.array(det_centroid) - np.array(track_centroid))
                dist_matrix[d_idx, t_idx] = dist
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        matched_dets = set()
        matched_track_ids = set()
        
        for d_idx, t_idx in zip(row_ind, col_ind):
            if dist_matrix[d_idx, t_idx] <= self.max_distance:
                track_id = track_ids[t_idx]
                detections[d_idx].instance_id = track_id
                self.tracks[track_id] = self._get_centroid(detections[d_idx].bbox)
                self.frames_since_update[track_id] = 0  # Reset age
                matched_dets.add(d_idx)
                matched_track_ids.add(track_id)
        
        # Age unmatched tracks
        for track_id in self.tracks.keys():
            if track_id not in matched_track_ids:
                self.frames_since_update[track_id] += 1
        
        # Remove old tracks
        self._remove_old_tracks()
        
        # New tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                det.instance_id = self.next_track_id
                self.tracks[self.next_track_id] = self._get_centroid(det.bbox)
                self.frames_since_update[self.next_track_id] = 0
                self.next_track_id += 1
        
        return detections
    
    def _remove_old_tracks(self):
        """Remove tracks that haven't been matched for too long"""
        tracks_to_remove = [
            track_id for track_id, age in self.frames_since_update.items()
            if age > self.max_frames_missing
        ]
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            del self.frames_since_update[track_id]
    
    def _get_centroid(self, bbox):
        """Get centroid from bbox"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class TrackingValidationWorker(QThread):
    """Worker thread for tracking validation"""
    
    # Signals
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)  # current, total
    log_message = pyqtSignal(str)
    metrics_updated = pyqtSignal(dict)  # Algorithm metrics
    validation_complete = pyqtSignal(str)  # results_path
    validation_failed = pyqtSignal(str)  # error_msg
    
    def __init__(self, main_window, config):
        super().__init__()
        self.main_window = main_window
        self.config = config
        self.should_stop = False
    
    def stop(self):
        """Request worker to stop"""
        self.should_stop = True
    
    def run(self):
        """Run tracking validation"""
        try:
            use_ground_truth = self.config.get('use_ground_truth', False)
            
            self.log_message.emit("=== Tracking Algorithm Validation ===")
            if use_ground_truth:
                self.log_message.emit("Detection Source: Ground Truth Annotations (perfect detections)")
            else:
                self.log_message.emit(f"Detection Source: Model ({Path(self.config['model_path']).name})")
            self.log_message.emit(f"Sequences: {len(self.config['sequences'])}")
            self.log_message.emit(f"Algorithms: {', '.join(self.config['algorithms'].keys())}")
            self.log_message.emit("")
            
            # Load model only if not using ground truth
            model = None
            if not use_ground_truth:
                self.status_updated.emit("Loading detection model...")
                model = YOLO(self.config['model_path'])
            
            # Create results folder with subfolder for detection source
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detection_source = "ground_truth" if use_ground_truth else "model"
            results_folder = self.main_window.project_path / "tracking_validation" / detection_source / timestamp
            results_folder.mkdir(parents=True, exist_ok=True)
            
            # Save config
            config_file = results_folder / "config.json"
            config_to_save = copy.deepcopy(self.config)
            config_to_save['sequences'] = [
                {'sequence_id': s.sequence_id, 'video_id': s.video_id, 
                 'start_frame': s.start_frame, 'end_frame': s.end_frame}
                for s in self.config['sequences']
            ]
            with open(config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            
            # Results storage
            algorithm_results = {algo_name: {'sequences': []} for algo_name in self.config['algorithms'].keys()}
            
            # Process each sequence
            total_sequences = len(self.config['sequences'])
            for seq_idx, sequence in enumerate(self.config['sequences'], 1):
                if self.should_stop:
                    break
                
                self.log_message.emit(
                    f"=== Sequence {seq_idx}/{total_sequences}: "
                    f"{sequence.video_id} frames {sequence.start_frame}-{sequence.end_frame} ==="
                )
                self.status_updated.emit(
                    f"Processing sequence {seq_idx}/{total_sequences}: {sequence.video_id}"
                )
                self.progress_updated.emit(seq_idx, total_sequences)
                
                # Process this sequence with each algorithm
                for algo_name, algo_params in self.config['algorithms'].items():
                    if self.should_stop:
                        break
                    
                    self.log_message.emit(f"  Testing {algo_name}...")
                    
                    metrics = self._process_sequence_with_algorithm(
                        sequence, model, algo_name, algo_params, results_folder
                    )
                    
                    algorithm_results[algo_name]['sequences'].append(metrics)
                    
                    self.log_message.emit(f"    MOTA: {metrics.get('mota', 0):.2%} | "
                                         f"IDF1: {metrics.get('idf1', 0):.2%} | "
                                         f"ID Switches: {metrics.get('id_switches', 0)}")
                
                # Update overall metrics
                self._update_aggregate_metrics(algorithm_results)
                self.metrics_updated.emit(self._get_current_averages(algorithm_results))
            
            if self.should_stop:
                self.log_message.emit("\n⚠️ Validation stopped by user")
                return
            
            # Save final results
            self._save_final_results(algorithm_results, results_folder)
            
            self.log_message.emit("\n✓ Validation complete!")
            self.log_message.emit(f"Results saved to: {results_folder}")
            self.validation_complete.emit(str(results_folder))
            
        except Exception as e:
            import traceback
            error_msg = f"Validation failed: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(f"\n❌ {error_msg}")
            self.validation_failed.emit(error_msg)
    
    def _process_sequence_with_algorithm(self, sequence, model, algo_name, algo_params, results_folder):
        """Process one sequence with one algorithm"""
        # Initialize tracker
        if algo_name == 'bytetrack':
            tracker = InstanceTracker(config=algo_params)
        elif algo_name == 'simple_iou':
            tracker = SimpleIoUTracker(
                iou_threshold=algo_params['iou_threshold'],
                use_mask_iou=True
            )
        elif algo_name == 'centroid':
            tracker = CentroidTracker(
                max_distance=algo_params['max_distance'],
                max_frames_missing=algo_params.get('max_frames_missing', 1)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        
        # Reset tracker
        if hasattr(tracker, 'reset'):
            tracker.reset()
        
        # Create visualization folder if needed
        viz_folder = None
        if self.config.get('save_visualizations', False):
            viz_folder = results_folder / 'visualizations' / f"{sequence.sequence_id}_{algo_name}"
            viz_folder.mkdir(parents=True, exist_ok=True)
        
        # Load ground truth for all frames  
        gt_frames = {}
        for frame_idx in sequence.frame_range:
            annotations = self.main_window.annotation_manager.load_frame_annotations(
                self.main_window.project_path, sequence.video_id, frame_idx
            )
            gt_frames[frame_idx] = annotations if annotations else []
            if annotations:
                gt_ids = [ann.get('mask_id', ann.get('instance_id', 'N/A')) for ann in annotations]
                self.log_message.emit(f"  Loaded {len(annotations)} GT annotations for frame {frame_idx}, IDs: {gt_ids}")
                # Debug: Log GT mask shapes on first frame
                if frame_idx == sequence.start_frame:
                    for ann in annotations:
                        if ann.get('mask') is not None:
                            self.log_message.emit(f"    GT mask shape: {ann['mask'].shape}")
                            break  # Just log one for debugging
            else:
                self.log_message.emit(f"  WARNING: No GT annotations found for frame {frame_idx}")
        
        # Process frames
        id_mapping = {}  # Maps predicted_id -> gt_id (established in first frame)
        frame_metrics = []
        prev_frame_positions = {}  # Maps gt_id -> centroid position for tracking lines
        
        for frame_idx in sequence.frame_range:
            # Load frame image
            frame_path = self.main_window.project_manager.get_frame_path(sequence.video_id, frame_idx)
            if not frame_path.exists():
                self.log_message.emit(f"    Frame {frame_idx} not found, skipping")
                continue
            
            frame = cv2.imread(str(frame_path))
            
            # Debug: Log frame shape on first frame
            if frame_idx == sequence.start_frame:
                self.log_message.emit(f"    Frame shape: {frame.shape}")
            
            # Get detections
            use_ground_truth = self.config.get('use_ground_truth', False)
            if use_ground_truth:
                # Use ground truth annotations as perfect detections
                gt_annotations = gt_frames[frame_idx]
                detections = self._gt_to_detections(gt_annotations)
                self.log_message.emit(f"    Using {len(detections)} ground truth detections")
            else:
                # Run detection model
                results = model(
                    frame, 
                    conf=self.config['min_confidence'], 
                    iou=self.config['nms_iou_threshold'],
                    verbose=False
                )
                
                # Debug: Log YOLO orig_shape on first frame
                if frame_idx == sequence.start_frame and len(results) > 0:
                    self.log_message.emit(f"    YOLO orig_shape: {results[0].orig_shape}")
                
                # Convert to Detection objects
                detections = self._yolo_to_detections(results[0])
                self.log_message.emit(f"    Detected {len(detections)} objects")
            
            # Apply tracking
            if algo_name == 'bytetrack':
                # Use InstanceTracker's match_detections_to_tracks method
                # Note: match_detections_to_tracks returns List[Tuple[Detection, track_id]]
                tracker_result = tracker.match_detections_to_tracks(detections, frame_idx)
                
                # Replace detections with tracked ones and set their instance_ids
                detections = []
                for tracked_det, track_id in tracker_result:
                    tracked_det.instance_id = track_id
                    detections.append(tracked_det)
                
                self.log_message.emit(f"    Tracked {len(detections)} objects with IDs: {[d.instance_id for d in detections]}")
            else:
                # Simple trackers return detections directly with IDs set
                detections = tracker.update(detections)
                self.log_message.emit(f"    Tracked {len(detections)} objects")
            
            # Match to ground truth and calculate metrics
            gt_annotations = gt_frames[frame_idx]
            
            if frame_idx == sequence.start_frame:
                # First frame - establish ID mapping
                self.log_message.emit(f"    Establishing ID mapping in first frame...")
                self.log_message.emit(f"    Detections have IDs: {[d.instance_id for d in detections]}")
                id_mapping = self._establish_id_mapping(detections, gt_annotations)
                self.log_message.emit(f"    ID mapping: {id_mapping}")
            
            # Calculate frame metrics
            # Make a copy of id_mapping before metrics (for visualization to detect switches)
            id_mapping_before_metrics = id_mapping.copy()
            
            metrics = self._calculate_frame_metrics(
                detections, gt_annotations, id_mapping
            )
            frame_metrics.append(metrics)
            
            # Save visualization if enabled (use mapping from before metrics update)
            if viz_folder is not None:
                self._save_frame_visualization(
                    frame, detections, gt_annotations, id_mapping_before_metrics, 
                    prev_frame_positions, frame_idx, viz_folder, metrics
                )
            
            # Update previous frame positions for next iteration
            current_frame_positions = {}
            for gt in gt_annotations:
                gt_id = gt.get('mask_id', gt.get('instance_id'))
                if gt_id is not None:
                    centroid = self._get_centroid(gt.get('mask'), gt.get('bbox'))
                    if centroid is not None:
                        current_frame_positions[gt_id] = centroid
            prev_frame_positions = current_frame_positions
        
        # Aggregate sequence metrics
        return self._aggregate_sequence_metrics(frame_metrics, sequence)
    
    def _yolo_to_detections(self, yolo_result) -> List[Detection]:
        """Convert YOLO results to Detection objects"""
        detections = []
        
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return detections
        
        boxes = yolo_result.boxes.xyxy.cpu().numpy()
        confidences = yolo_result.boxes.conf.cpu().numpy()
        
        # Check for masks
        has_masks = yolo_result.masks is not None
        
        for idx in range(len(boxes)):
            bbox = boxes[idx]
            conf = float(confidences[idx])
            
            mask = None
            if has_masks:
                mask = yolo_result.masks.data[idx].cpu().numpy()
                # Resize mask to frame size if needed
                if mask.shape[:2] != yolo_result.orig_shape[:2]:
                    # Debug log on first detection
                    if idx == 0:
                        print(f"[MASK RESIZE] Mask shape: {mask.shape}, target: {yolo_result.orig_shape}, resizing to ({yolo_result.orig_shape[1]}, {yolo_result.orig_shape[0]})")
                    mask = cv2.resize(mask, (yolo_result.orig_shape[1], yolo_result.orig_shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
                    if idx == 0:
                        print(f"[MASK RESIZE] After resize: {mask.shape}")
                mask = (mask > 0.5).astype(np.uint8)
            
            det = Detection(
                bbox=bbox,
                mask=mask,
                confidence=conf,
                source='yolo',
                instance_id=None
            )
            detections.append(det)
        
        return detections
    
    def _gt_to_detections(self, gt_annotations) -> List[Detection]:
        """Convert ground truth annotations to Detection objects"""
        detections = []
        
        for gt in gt_annotations:
            mask = gt.get('mask')
            bbox = gt.get('bbox')
            
            # Get the original GT ID
            gt_id = gt.get('mask_id', gt.get('instance_id'))
            
            # Convert bbox to xyxy format
            # Annotations are always stored as [x, y, width, height]
            if bbox is not None:
                if isinstance(bbox, list) and len(bbox) == 4:
                    # Convert [x,y,w,h] to [x1,y1,x2,y2]
                    bbox = np.array([
                        bbox[0],           # x1 = x
                        bbox[1],           # y1 = y  
                        bbox[0] + bbox[2], # x2 = x + width
                        bbox[1] + bbox[3]  # y2 = y + height
                    ], dtype=np.float32)
                else:
                    bbox = np.array(bbox, dtype=np.float32)
            elif mask is not None:
                # Generate bbox from mask
                bbox = self._mask_to_bbox(mask)
            else:
                continue  # Skip if no mask or bbox
            
            det = Detection(
                bbox=bbox,
                mask=mask,
                confidence=1.0,  # Perfect confidence for GT
                source='ground_truth',
                instance_id=None,
                gt_id=gt_id  # Store original GT ID for matching
            )
            detections.append(det)
        
        return detections
    
    def _establish_id_mapping(self, detections, gt_annotations):
        """Establish mapping from predicted IDs to ground truth IDs in first frame"""
        id_mapping = {}
        
        if not detections or not gt_annotations:
            print("WARNING: No detections or GT annotations to map!")
            return id_mapping
        
        print(f"  Establishing ID mapping: {len(detections)} detections, {len(gt_annotations)} GT")
        print(f"  Detection IDs: {[d.instance_id for d in detections]}")
        print(f"  GT IDs: {[gt.get('mask_id', gt.get('instance_id', 'N/A')) for gt in gt_annotations]}")
        
        # Check if detections have stored GT IDs (ground truth mode)
        if detections and detections[0].gt_id is not None:
            print("  Using stored GT IDs for mapping (ground truth mode)")
            for det in detections:
                if det.instance_id is not None and det.gt_id is not None:
                    id_mapping[det.instance_id] = det.gt_id
                    print(f"    Mapped predicted ID {det.instance_id} -> GT ID {det.gt_id}")
            print(f"  Established {len(id_mapping)} ID mappings: {id_mapping}")
            return id_mapping
        
        # Model-based detection - use IoU matching
        # Build IoU matrix
        iou_matrix = np.zeros((len(detections), len(gt_annotations)))
        
        for d_idx, det in enumerate(detections):
            for g_idx, gt in enumerate(gt_annotations):
                gt_mask = gt.get('mask')
                if gt_mask is not None and det.mask is not None:
                    iou = self._mask_iou(det.mask, gt_mask)
                else:
                    gt_bbox = gt.get('bbox')
                    if gt_bbox is not None:
                        # Annotations are stored as [x, y, width, height], convert to [x1,y1,x2,y2]
                        if isinstance(gt_bbox, list) and len(gt_bbox) == 4:
                            gt_bbox_xyxy = [
                                gt_bbox[0],           # x1 = x
                                gt_bbox[1],           # y1 = y
                                gt_bbox[0] + gt_bbox[2],  # x2 = x + width
                                gt_bbox[1] + gt_bbox[3]   # y2 = y + height
                            ]
                        else:
                            gt_bbox_xyxy = gt_bbox
                        iou = self._bbox_iou(det.bbox, gt_bbox_xyxy)
                    elif gt_mask is not None:
                        gt_bbox_extracted = self._mask_to_bbox(gt_mask)
                        if gt_bbox_extracted is not None:
                            iou = self._bbox_iou(det.bbox, gt_bbox_extracted)
                        else:
                            iou = 0.0
                    else:
                        iou = 0.0
                iou_matrix[d_idx, g_idx] = iou
        
        print(f"  IoU matrix max values: {[np.max(iou_matrix[i, :]) for i in range(len(detections))]}")
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximize IoU
        
        for d_idx, g_idx in zip(row_ind, col_ind):
            if iou_matrix[d_idx, g_idx] >= self.config['gt_iou_threshold']:
                pred_id = detections[d_idx].instance_id
                # Use mask_id or instance_id from ground truth annotations
                gt_id = gt_annotations[g_idx].get('mask_id', gt_annotations[g_idx].get('instance_id', g_idx))
                
                if pred_id is not None:  # Only map if detection has an ID
                    id_mapping[pred_id] = gt_id
                    print(f"    Mapped predicted ID {pred_id} -> GT ID {gt_id} (IoU={iou_matrix[d_idx, g_idx]:.3f})")
                else:
                    print(f"    WARNING: Detection {d_idx} has no instance_id! Cannot map.")
        
        print(f"  Established {len(id_mapping)} ID mappings: {id_mapping}")
        return id_mapping
    
    def _calculate_frame_metrics(self, detections, gt_annotations, id_mapping):
        """Calculate metrics for one frame"""
        metrics = {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'id_switches': 0,
            'total_iou': 0.0,
            'matched_count': 0,
        }
        
        if not gt_annotations:
            metrics['fp'] = len(detections)
            return metrics
        
        print(f"    Calculating metrics: {len(detections)} detections, {len(gt_annotations)} GT, ID mapping: {id_mapping}")
        
        # Classify all detections using shared logic
        detection_status = self._classify_detections(detections, gt_annotations, id_mapping)
        
        # Count metrics from classification results
        matched_gt_indices = set()
        for det in detections:
            status, matched_gt_id, matched_gt_idx, iou = detection_status[id(det)]
            
            if status == 'TP':
                metrics['tp'] += 1
                metrics['total_iou'] += iou
                metrics['matched_count'] += 1
                if matched_gt_idx is not None:
                    matched_gt_indices.add(matched_gt_idx)
                print(f"      Det ID {det.instance_id} -> GT {matched_gt_id}: TP (IoU={iou:.3f})")
            elif status == 'ID_SWITCH':
                metrics['id_switches'] += 1
                metrics['total_iou'] += iou
                metrics['matched_count'] += 1
                if matched_gt_idx is not None:
                    matched_gt_indices.add(matched_gt_idx)
                print(f"      Det ID {det.instance_id} -> ID SWITCH (GT {matched_gt_id}, IoU={iou:.3f})")
            else:  # FP
                metrics['fp'] += 1
                print(f"      Det ID {det.instance_id} -> FP")
        
        # Count false negatives (unmatched ground truth)
        metrics['fn'] = len(gt_annotations) - len(matched_gt_indices)
        print(f"    Metrics: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, ID_switches={metrics['id_switches']}")
        
        return metrics
    
    def _classify_detections(self, detections, gt_annotations, id_mapping):
        """Classify all detections as TP, FP, or ID_SWITCH
        
        Returns:
            dict: Maps detection id -> (status, matched_gt_id, matched_gt_idx, iou)
                  status is 'TP', 'FP', or 'ID_SWITCH'
                  matched_gt_id is the GT ID if matched, None otherwise
                  matched_gt_idx is the GT index if matched, None otherwise
                  iou is the IoU value if matched, 0.0 otherwise
        
        Note: This function modifies id_mapping in-place as switches are detected
        """
        detection_status = {}
        matched_gt_indices = set()
        current_frame_gt_to_tracker = {}  # Track which GTs are matched in THIS frame
        
        # Build reverse mapping: GT_ID -> Tracker_ID from historical id_mapping
        gt_to_last_tracker = {}
        for tracker_id, gt_id in id_mapping.items():
            gt_to_last_tracker[gt_id] = tracker_id
        
        # Process existing tracks FIRST to avoid spurious ID switches
        # New detections might have high IoU with a GT that an existing track should match
        existing_tracks = [d for d in detections if d.instance_id in id_mapping]
        new_tracks = [d for d in detections if d.instance_id not in id_mapping]
        sorted_detections = existing_tracks + new_tracks
        
        for det in sorted_detections:
            if det.instance_id is None:
                detection_status[id(det)] = ('FP', None, None, 0.0)
                continue
            
            # Find best matching GT for this detection
            best_match_iou = 0.0
            best_match_gt_id = None
            best_match_gt_idx = None
            
            for gt_idx, gt in enumerate(gt_annotations):
                if gt_idx in matched_gt_indices:
                    continue  # Already matched
                
                gt_id = gt.get('mask_id', gt.get('instance_id'))
                iou = self._calculate_iou(det, gt)
                
                if iou >= self.config['gt_iou_threshold'] and iou > best_match_iou:
                    best_match_iou = iou
                    best_match_gt_id = gt_id
                    best_match_gt_idx = gt_idx
            
            if best_match_gt_id is None:
                # No matching GT - false positive
                detection_status[id(det)] = ('FP', None, None, 0.0)
                continue
            
            # Determine if this is an ID switch
            is_id_switch = False
            
            # Check 1: Tracker changed which GT it's following
            if det.instance_id in id_mapping:
                previously_tracked_gt = id_mapping[det.instance_id]
                if best_match_gt_id != previously_tracked_gt:
                    is_id_switch = True
            else:
                # Check 2: For NEW tracks only - GT changed which tracker is following it (fragmentation)
                # Only applies to new tracks; existing tracks are handled by Check 1
                if best_match_gt_id in gt_to_last_tracker:
                    last_tracker_for_this_gt = gt_to_last_tracker[best_match_gt_id]
                    if last_tracker_for_this_gt != det.instance_id:
                        is_id_switch = True
            
            # Check 3: Multiple trackers trying to follow same GT in current frame
            if best_match_gt_id in current_frame_gt_to_tracker:
                is_id_switch = True
            
            # Classify detection
            if is_id_switch:
                detection_status[id(det)] = ('ID_SWITCH', best_match_gt_id, best_match_gt_idx, best_match_iou)
            else:
                detection_status[id(det)] = ('TP', best_match_gt_id, best_match_gt_idx, best_match_iou)
            
            # Update tracking state
            matched_gt_indices.add(best_match_gt_idx)
            id_mapping[det.instance_id] = best_match_gt_id
            current_frame_gt_to_tracker[best_match_gt_id] = det.instance_id
            gt_to_last_tracker[best_match_gt_id] = det.instance_id
        
        return detection_status
    
    def _find_gt_by_id(self, target_gt_id, gt_annotations):
        """Find ground truth annotation by ID, return (index, annotation) or None"""
        for g_idx, gt in enumerate(gt_annotations):
            gt_id = gt.get('mask_id', gt.get('instance_id', g_idx))
            if gt_id == target_gt_id:
                return (g_idx, gt)
        return None
    
    def _find_best_gt_match(self, det, gt_annotations, exclude_idx, matched_gt_indices):
        """Find best matching GT for detection (excluding specific index and already matched)
        
        Returns (gt_index, iou) if good match found, None otherwise
        """
        best_iou = 0.0
        best_idx = None
        
        for gt_idx, gt in enumerate(gt_annotations):
            # Skip the GT we already tried and already matched GTs
            if gt_idx == exclude_idx or gt_idx in matched_gt_indices:
                continue
            
            # Calculate IoU
            iou = self._calculate_iou(det, gt)
            
            if iou >= self.config['gt_iou_threshold'] and iou > best_iou:
                best_iou = iou
                best_idx = gt_idx
        
        if best_idx is not None:
            return (best_idx, best_iou)
        return None
    
    def _calculate_iou_from_gt(self, det, gt):
        """Calculate IoU between detection and ground truth (handles bbox conversion)"""
        gt_mask = gt.get('mask')
        gt_bbox = gt.get('bbox')
        
        # Try mask IoU first
        if det.mask is not None and gt_mask is not None:
            return self._mask_iou(det.mask, gt_mask)
        
        # Fall back to bbox IoU
        if gt_bbox is not None and det.bbox is not None:
            gt_bbox_xyxy = self._convert_bbox_to_xyxy(gt_bbox)
            return self._bbox_iou(det.bbox, gt_bbox_xyxy)
        
        # Extract bbox from mask if needed
        if gt_mask is not None and det.bbox is not None:
            gt_bbox_extracted = self._mask_to_bbox(gt_mask)
            if gt_bbox_extracted is not None:
                return self._bbox_iou(det.bbox, gt_bbox_extracted)
        
        return 0.0
    
    def _convert_bbox_to_xyxy(self, bbox):
        """Convert bbox from [x,y,w,h] to [x1,y1,x2,y2] format"""
        if isinstance(bbox, list) and len(bbox) == 4:
            return [
                bbox[0],           # x1 = x
                bbox[1],           # y1 = y
                bbox[0] + bbox[2], # x2 = x + width
                bbox[1] + bbox[3]  # y2 = y + height
            ]
        return bbox
    
    def _aggregate_sequence_metrics(self, frame_metrics, sequence):
        """Aggregate frame metrics into sequence metrics"""
        total_tp = sum(m['tp'] for m in frame_metrics)
        total_fp = sum(m['fp'] for m in frame_metrics)
        total_fn = sum(m['fn'] for m in frame_metrics)
        total_id_switches = sum(m['id_switches'] for m in frame_metrics)
        total_iou = sum(m['total_iou'] for m in frame_metrics)
        total_matched = sum(m['matched_count'] for m in frame_metrics)
        
        # TP = all detection matches (IoU > threshold, regardless of ID)
        # IDTP = identity true positives (IoU match + correct ID)
        # Relationship: TP = IDTP + ID_SWITCH
        total_idtp = total_tp  # Rename for clarity: total_tp from frames means correct ID
        total_tp_detection = total_idtp + total_id_switches  # All IoU matches
        
        # Total ground truth = all matched + not matched
        total_gt = total_tp_detection + total_fn
        
        # Total detections = matched + false positives
        total_detections = total_tp_detection + total_fp
        
        # MOTA - penalizes FP, FN, and ID switches relative to total GT
        mota = 0.0
        if total_gt > 0:
            mota = 1.0 - (total_fp + total_fn + total_id_switches) / total_gt
        
        # MOTP - average IoU over all matches
        motp = total_iou / total_matched if total_matched > 0 else 0.0
        
        # Precision/Recall - detection quality (all IoU matches count)
        precision = total_tp_detection / total_detections if total_detections > 0 else 0.0
        recall = total_tp_detection / total_gt if total_gt > 0 else 0.0
        
        # IDF1 - identity preservation metric (only correct IDs count as IDTP)
        # Formula: 2*IDTP / (num_gt + num_detections)
        idf1 = 2 * total_idtp / (total_gt + total_detections) if (total_gt + total_detections) > 0 else 0.0
        
        return {
            'sequence_id': sequence.sequence_id,
            'video_id': sequence.video_id,
            'start_frame': sequence.start_frame,
            'end_frame': sequence.end_frame,
            'mota': mota,
            'motp': motp,
            'idf1': idf1,
            'precision': precision,
            'recall': recall,
            'id_switches': total_id_switches,
            'tp': total_tp_detection,  # Detection TP (IoU match, any ID) - should be same across algorithms with GT detections
            'idtp': total_idtp,  # Identity TP (IoU match + correct ID) - varies by tracker quality
            'fp': total_fp,
            'fn': total_fn,
        }
    
    def _update_aggregate_metrics(self, algorithm_results):
        """Update aggregate metrics for all algorithms"""
        pass  # Metrics are calculated per sequence and averaged later
    
    def _get_current_averages(self, algorithm_results):
        """Get current average metrics across all sequences"""
        averages = {}
        
        for algo_name, results in algorithm_results.items():
            if not results['sequences']:
                continue
            
            sequences = results['sequences']
            averages[algo_name] = {
                'mota': np.mean([s['mota'] for s in sequences]),
                'motp': np.mean([s['motp'] for s in sequences]),
                'idf1': np.mean([s['idf1'] for s in sequences]),
                'precision': np.mean([s['precision'] for s in sequences]),
                'recall': np.mean([s['recall'] for s in sequences]),
                'id_switches': sum([s['id_switches'] for s in sequences]),
            }
        
        return averages
    
    def _save_final_results(self, algorithm_results, results_folder):
        """Save final results to files"""
        # Save per-algorithm CSVs
        for algo_name, results in algorithm_results.items():
            csv_path = results_folder / f"{algo_name}_results.csv"
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'sequence_id', 'video_id', 'start_frame', 'end_frame',
                    'mota', 'motp', 'idf1', 'precision', 'recall', 
                    'id_switches', 'tp', 'idtp', 'fp', 'fn'
                ])
                writer.writeheader()
                writer.writerows(results['sequences'])
        
        # Save summary JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'algorithms': {}
        }
        
        for algo_name, results in algorithm_results.items():
            if not results['sequences']:
                continue
            
            sequences = results['sequences']
            summary['algorithms'][algo_name] = {
                'num_sequences': len(sequences),
                'average_mota': float(np.mean([s['mota'] for s in sequences])),
                'average_motp': float(np.mean([s['motp'] for s in sequences])),
                'average_idf1': float(np.mean([s['idf1'] for s in sequences])),
                'average_precision': float(np.mean([s['precision'] for s in sequences])),
                'average_recall': float(np.mean([s['recall'] for s in sequences])),
                'total_id_switches': sum([s['id_switches'] for s in sequences]),
            }
        
        summary_path = results_folder / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save comparison plot
        self._save_comparison_plot(algorithm_results, results_folder)
    
    def _save_comparison_plot(self, algorithm_results, results_folder):
        """Save comparison plot"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend for worker thread
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Tracking Algorithm Comparison', fontsize=16)
        
        algorithms = list(algorithm_results.keys())
        metrics_to_plot = [
            ('mota', 'MOTA', axes[0, 0]),
            ('idf1', 'IDF1', axes[0, 1]),
            ('precision', 'Precision', axes[1, 0]),
            ('recall', 'Recall', axes[1, 1]),
        ]
        
        for metric_key, metric_label, ax in metrics_to_plot:
            values = []
            for algo_name in algorithms:
                sequences = algorithm_results[algo_name]['sequences']
                if sequences:
                    avg = np.mean([s[metric_key] for s in sequences])
                    values.append(avg)
                else:
                    values.append(0)
            
            ax.bar(algorithms, values)
            ax.set_ylabel(metric_label)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_title(metric_label)
        
        try:
            plt.tight_layout()
        except Exception:
            pass  # Ignore tight_layout warnings
        plt.savefig(results_folder / 'comparison.png', dpi=150)
        plt.close()
    
    def _bbox_iou(self, bbox1, bbox2):
        """Compute bbox IoU"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _mask_iou(self, mask1, mask2):
        """Compute mask IoU"""
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def _mask_to_bbox(self, mask):
        """Convert mask to bbox"""
        if mask is None:
            return None
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)
    
    def _get_centroid(self, mask, bbox):
        """Get centroid of mask or bbox"""
        if mask is not None:
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                y, x = coords.mean(axis=0)
                return (int(x), int(y))
        
        if bbox is not None:
            if isinstance(bbox, list):
                bbox = np.array(bbox)
            
            # Annotations are stored as [x, y, width, height]
            if len(bbox) == 4:
                # Assume [x,y,w,h] format for loaded annotations
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
                return (int(cx), int(cy))
        
        return None
    
    def _save_frame_visualization(self, frame, detections, gt_annotations, id_mapping, 
                                  prev_frame_positions, frame_idx, viz_folder, metrics):
        """Save visualization of tracking for one frame"""
        # Create a copy of the frame for drawing
        viz_frame = frame.copy()
        
        # Use a copy of id_mapping so we can update it without affecting the original
        id_mapping = id_mapping.copy()
        
        # Classify detections using the same shared logic as metrics calculation
        detection_status = self._classify_detections(detections, gt_annotations, id_mapping)
        
        # Extract matched GT IDs for FN detection
        matched_gt_ids = set()
        for det in detections:
            status, matched_gt_id, matched_gt_idx, iou = detection_status[id(det)]
            if matched_gt_id is not None:
                matched_gt_ids.add(matched_gt_id)
        
        # Draw ground truth annotations first (FN in red, matched ones will be overlaid)
        for gt in gt_annotations:
            gt_id = gt.get('mask_id', gt.get('instance_id'))
            is_matched = gt_id in matched_gt_ids
            
            if not is_matched:
                # False Negative - red, thick
                color = (0, 0, 255)
                thickness = 3
                
                # Draw mask contour or bbox
                if gt.get('mask') is not None:
                    contours, _ = cv2.findContours(gt['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(viz_frame, contours, -1, color, thickness)
                elif gt.get('bbox') is not None:
                    # Annotations are stored as [x, y, width, height]
                    bbox = gt['bbox']
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    cv2.rectangle(viz_frame, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)
            
            # Draw tracking line if this is not the first frame
            if gt_id in prev_frame_positions:
                prev_pos = prev_frame_positions[gt_id]
                curr_pos = self._get_centroid(gt.get('mask'), gt.get('bbox'))
                
                if curr_pos is not None:
                    # Check if tracked correctly (is there a TP detection with this gt_id?)
                    tracked_correctly = any(
                        status == 'TP' and matched_id == gt_id 
                        for status, matched_id, _, _ in detection_status.values()
                    )
                    
                    # Green line if tracked correctly, red if not
                    line_color = (0, 255, 0) if tracked_correctly else (0, 0, 255)
                    cv2.line(viz_frame, prev_pos, curr_pos, line_color, 2)
                    cv2.circle(viz_frame, curr_pos, 3, line_color, -1)
        
        # Draw detections with status colors
        for det in detections:
            status, matched_gt_id, _, _ = detection_status.get(id(det), ('FP', None, None, 0.0))
            
            # Determine color and thickness based on status
            if status == 'TP':
                color = (0, 255, 0)  # Green - correctly matched
                thickness = 2
            elif status == 'ID_SWITCH':
                color = (0, 165, 255)  # Orange - matched but wrong ID
                thickness = 3
            else:  # FP
                color = (255, 0, 255)  # Pink - no match
                thickness = 3
            
            # Draw detection mask or bbox
            if det.mask is not None:
                # Draw segmentation mask contours
                contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(viz_frame, contours, -1, color, thickness)
            else:
                # Fall back to bounding box if no mask
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(viz_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Add detection ID label at centroid (or top-left of bbox as fallback)
            if det.instance_id is not None:
                # Get centroid position from mask or bbox
                centroid = self._get_centroid(det.mask, det.bbox)
                if centroid is not None:
                    label_pos = centroid
                else:
                    x1, y1, x2, y2 = det.bbox
                    label_pos = (int(x1), int(y1) - 10)
                
                label_color = color
                cv2.putText(viz_frame, f"ID:{det.instance_id}", label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
        
        # Add legend
        self._draw_legend(viz_frame, metrics)
        
        # Save visualization
        viz_path = viz_folder / f"frame_{frame_idx:06d}.png"
        
        # Ensure frame is uint8
        if viz_frame.dtype != np.uint8:
            viz_frame = viz_frame.astype(np.uint8)
        
        # Check if frame is valid
        if viz_frame.size == 0:
            print(f"Warning: Empty frame, cannot save visualization to {viz_path}")
            return
        
        # Try to save with explicit parameters
        try:
            success = cv2.imwrite(
                str(viz_path), 
                viz_frame,
                [cv2.IMWRITE_PNG_COMPRESSION, 3]  # Lower compression for faster/safer writes
            )
            if not success:
                # Try alternative path with shorter name
                alt_path = viz_folder / f"f{frame_idx}.png"
                print(f"Warning: Failed to save to {viz_path.name}, trying shorter path...")
                success = cv2.imwrite(str(alt_path), viz_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not success:
                    print(f"Error: Could not save visualization. Frame shape: {viz_frame.shape}, dtype: {viz_frame.dtype}")
                    print(f"  Path: {viz_path}")
                    print(f"  Directory exists: {viz_path.parent.exists()}")
                    print(f"  Directory writable: {os.access(viz_path.parent, os.W_OK)}")
        except Exception as e:
            print(f"Exception saving visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_iou(self, detection, gt_annotation):
        """Calculate IoU between detection and ground truth annotation"""
        return self._calculate_iou_from_gt(detection, gt_annotation)
    
    def _draw_legend(self, frame, metrics):
        """Draw a comprehensive legend on the frame"""
        # Legend background
        legend_x = 10
        legend_y = 10
        legend_width = 300
        legend_height = 200
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (255, 255, 255), 2)
        
        # Title
        y_pos = legend_y + 25
        cv2.putText(frame, "Legend", (legend_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos += 30
        
        # Detection box colors
        box_x = legend_x + 15
        
        # Green box - Correctly matched (TP)
        cv2.rectangle(frame, (box_x, y_pos - 10), (box_x + 20, y_pos + 10), 
                     (0, 255, 0), 2)
        cv2.putText(frame, "Correct Match", (box_x + 30, y_pos + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 25
        
        # Red box - False Negative (GT not matched)
        cv2.rectangle(frame, (box_x, y_pos - 10), (box_x + 20, y_pos + 10), 
                     (0, 0, 255), 3)
        cv2.putText(frame, "GT Not Matched (FN)", (box_x + 30, y_pos + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 25
        
        # Pink box - False Positive (Pred not matched)
        cv2.rectangle(frame, (box_x, y_pos - 10), (box_x + 20, y_pos + 10), 
                     (255, 0, 255), 3)
        cv2.putText(frame, "Pred Not Matched (FP)", (box_x + 30, y_pos + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 25
        
        # Orange box - ID Switch (matched but wrong ID)
        cv2.rectangle(frame, (box_x, y_pos - 10), (box_x + 20, y_pos + 10), 
                     (0, 165, 255), 3)
        cv2.putText(frame, "Wrong Track ID", (box_x + 30, y_pos + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 30
        
        # Tracking indicators
        # Green line - Correct tracking
        line_start_x = box_x + 5
        line_end_x = box_x + 15
        cv2.line(frame, (line_start_x, y_pos), (line_end_x, y_pos), (0, 255, 0), 2)
        cv2.circle(frame, (line_end_x, y_pos), 3, (0, 255, 0), -1)
        cv2.putText(frame, "Correct Track", (box_x + 30, y_pos + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 25
        
        # Red line - Lost/incorrect tracking
        cv2.line(frame, (line_start_x, y_pos), (line_end_x, y_pos), (0, 0, 255), 2)
        cv2.circle(frame, (line_end_x, y_pos), 3, (0, 0, 255), -1)
        cv2.putText(frame, "Lost/Wrong Track", (box_x + 30, y_pos + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 30
        
        # Metrics
        cv2.putText(frame, f"TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}", 
                   (legend_x + 10, y_pos + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
