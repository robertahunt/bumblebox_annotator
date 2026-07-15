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

from core.aruco_tracking_identity import ArucoTrackingIdentityManager
from core.instance_tracker import InstanceTracker, Detection, Track
from utils.validation_metrics import mask_to_simplified_polygon, polygon_to_string


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
        self._external_aruco_index = None
        self._external_aruco_path = None
        self._external_aruco_video_ids = None
    
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
            self.log_message.emit(
                f"ArUco tracking guidance: {'Enabled' if self.config.get('enable_aruco', False) else 'Disabled'}"
            )
            if self.config.get('enable_aruco', False):
                aruco_source = self.config.get('aruco_source', 'builtin')
                if aruco_source == 'external':
                    self.log_message.emit(
                        f"ArUco source: external CSV ({self.config.get('external_aruco_path', '')})"
                    )
                    self._load_external_aruco_index()
                else:
                    self.log_message.emit("ArUco source: built-in detector")
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

        aruco_identity = None
        use_aruco_tracking = self.config.get('enable_aruco', False)
        requested_aruco_source = self.config.get('aruco_source', 'builtin')
        aruco_source = requested_aruco_source
        external_aruco_available = False
        if use_aruco_tracking:
            aruco_identity = ArucoTrackingIdentityManager(
                log_callback=self.log_message.emit,
                verbose_output=False
            )
            if requested_aruco_source == 'external':
                external_aruco_available = self._external_aruco_available_for_video(sequence.video_id)
                if not external_aruco_available:
                    aruco_source = 'builtin'
                    self.log_message.emit(
                        f"    No external ArUco CSV rows found for {sequence.video_id}; "
                        "falling back to built-in detector"
                    )
                else:
                    self.log_message.emit(f"    Using external ArUco CSV rows for {sequence.video_id}")
        
        video_aruco_tracking = self._load_video_aruco_tracking(sequence.video_id)

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
        processed_frames = []

        def merge_validation_tracks(source_id: int, target_id: int, aruco_code: str):
            """Retroactively apply ArUco re-identification to processed validation frames."""
            for processed in processed_frames:
                for det in processed['detections']:
                    if det.instance_id == source_id:
                        det.instance_id = target_id
                if source_id in processed.get('aruco_detections', {}):
                    processed['aruco_detections'][target_id] = processed['aruco_detections'].pop(source_id)
            self.log_message.emit(
                f"    ArUco re-identification: merged track {source_id} into {target_id} "
                f"(ArUco {aruco_code})"
            )
        
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
                detections = self._gt_to_detections(gt_annotations, frame.shape)
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

            aruco_detections = {}
            aruco_match_stats = {}
            if aruco_identity is not None:
                if aruco_source == 'external':
                    external_markers = self._get_external_aruco_records(
                        sequence.video_id, frame_idx, frame_path
                    )
                    marker_codes, aruco_detections, aruco_match_stats = self._match_external_aruco_to_bees(
                        external_markers, detections, frame_idx=frame_idx
                    )
                    if external_markers and len(aruco_detections) != len(external_markers):
                        self._log_external_aruco_match_stats(
                            frame_idx, external_markers, detections, aruco_match_stats
                        )
                    aruco_identity.apply_bee_aruco_codes(
                        bee_detections=detections,
                        frame_count=frame_idx,
                        marker_codes=marker_codes,
                        merge_callback=merge_validation_tracks
                    )
                    aruco_detections = self._remap_aruco_detections_after_identity_update(
                        aruco_detections,
                        detections,
                        aruco_identity,
                    )
                else:
                    aruco_detections = aruco_identity.detect_bee_aruco_codes(
                        frame=frame,
                        bee_detections=detections,
                        frame_count=frame_idx,
                        merge_callback=merge_validation_tracks
                    )

            processed_frames.append({
                'frame_idx': frame_idx,
                'frame_path': frame_path,
                'detections': detections,
                'gt_annotations': gt_frames[frame_idx],
                'aruco_detections': aruco_detections,
                'aruco_match_stats': aruco_match_stats if aruco_source == 'external' else {},
            })

        mapping_initialized = False
        sequence_detection_rows = []
        sequence_aruco_summary = defaultdict(int)
        sequence_aruco_summary['aruco_requested_source'] = requested_aruco_source if use_aruco_tracking else 'disabled'
        sequence_aruco_summary['aruco_detection_source'] = aruco_source if use_aruco_tracking else 'disabled'
        sequence_aruco_summary['external_aruco_available'] = bool(external_aruco_available)
        sequence_aruco_summary['external_aruco_fallback_to_builtin'] = (
            requested_aruco_source == 'external' and aruco_source == 'builtin'
        )
        for processed in processed_frames:
            frame_idx = processed['frame_idx']
            detections = processed['detections']
            gt_annotations = processed['gt_annotations']
            aruco_detections = processed.get('aruco_detections', {})

            if not mapping_initialized:
                self.log_message.emit(f"    Establishing ID mapping in first evaluated frame...")
                self.log_message.emit(f"    Detections have IDs: {[d.instance_id for d in detections]}")
                id_mapping = self._establish_id_mapping(detections, gt_annotations)
                self.log_message.emit(f"    ID mapping: {id_mapping}")
                mapping_initialized = True

            id_mapping_before_metrics = id_mapping.copy()
            detection_status_for_export = self._classify_detections(
                detections, gt_annotations, id_mapping_before_metrics.copy()
            )

            metrics = self._calculate_frame_metrics(
                detections, gt_annotations, id_mapping
            )
            frame_metrics.append(metrics)

            rows, aruco_summary = self._build_detection_export_rows(
                sequence=sequence,
                algo_name=algo_name,
                frame_idx=frame_idx,
                detections=detections,
                gt_annotations=gt_annotations,
                detection_status=detection_status_for_export,
                aruco_detections=aruco_detections,
                aruco_identity=aruco_identity,
                video_aruco_tracking=video_aruco_tracking
            )
            if self.config.get('export_bee_detections', False):
                sequence_detection_rows.extend(rows)
            for key, value in aruco_summary.items():
                sequence_aruco_summary[key] += value
            if aruco_source == 'external':
                match_stats = processed.get('aruco_match_stats', {})
                sequence_aruco_summary['num_external_csv_aruco_rows'] += match_stats.get('csv_aruco_rows', 0)
                sequence_aruco_summary['num_external_csv_noid_rows'] += match_stats.get('csv_noid_rows', 0)
                sequence_aruco_summary['num_external_csv_aruco_or_noid_rows'] += match_stats.get('csv_rows', 0)
                sequence_aruco_summary['num_external_csv_linked_aruco_rows'] += match_stats.get('linked_aruco_rows', 0)
                sequence_aruco_summary['num_external_csv_linked_noid_rows'] += match_stats.get('linked_noid_rows', 0)
                sequence_aruco_summary['num_external_csv_aruco_detections'] += aruco_summary.get(
                    'num_aruco_detections', 0
                )
                sequence_aruco_summary['num_external_csv_noid_detections'] += aruco_summary.get(
                    'num_noid_detections', 0
                )
                sequence_aruco_summary['num_external_csv_aruco_or_noid_detections'] += aruco_summary.get(
                    'num_aruco_or_noid_detections', 0
                )
            elif aruco_source == 'builtin':
                sequence_aruco_summary['num_builtin_aruco_detections'] += aruco_summary.get(
                    'num_aruco_detections', 0
                )
                sequence_aruco_summary['num_builtin_noid_detections'] += aruco_summary.get(
                    'num_noid_detections', 0
                )
                sequence_aruco_summary['num_builtin_aruco_or_noid_detections'] += aruco_summary.get(
                    'num_aruco_or_noid_detections', 0
                )

            if viz_folder is not None:
                frame = cv2.imread(str(processed['frame_path']))
                if frame is not None:
                    aruco_assignments = (
                        aruco_identity.bee_to_aruco.copy()
                        if aruco_identity is not None
                        else {}
                    )
                    self._save_frame_visualization(
                        frame, detections, gt_annotations, id_mapping_before_metrics,
                        prev_frame_positions, frame_idx, viz_folder, metrics,
                        aruco_assignments
                    )

            current_frame_positions = {}
            for gt in gt_annotations:
                gt_id = gt.get('mask_id', gt.get('instance_id'))
                if gt_id is not None:
                    centroid = self._get_centroid(gt.get('mask'), gt.get('bbox'))
                    if centroid is not None:
                        current_frame_positions[gt_id] = centroid
            prev_frame_positions = current_frame_positions
        
        if self.config.get('export_bee_detections', False) and sequence_detection_rows:
            self._append_bee_detection_rows(results_folder, algo_name, sequence_detection_rows)

        # Aggregate sequence metrics
        return self._aggregate_sequence_metrics(frame_metrics, sequence, sequence_aruco_summary)
    
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
                mask = ((mask > 0.5).astype(np.uint8)) * 255
            elif self.config.get('enable_aruco', False):
                frame_height, frame_width = yolo_result.orig_shape[:2]
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                mask[y1:y2, x1:x2] = 255
            
            det = Detection(
                bbox=bbox,
                mask=mask,
                confidence=conf,
                source='yolo',
                instance_id=None
            )
            detections.append(det)
        
        return detections
    
    def _gt_to_detections(self, gt_annotations, frame_shape=None) -> List[Detection]:
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

            if mask is None and self.config.get('enable_aruco', False) and frame_shape is not None:
                frame_height, frame_width = frame_shape[:2]
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                mask[y1:y2, x1:x2] = 255
            elif mask is not None and mask.max() <= 1:
                mask = (mask.astype(np.uint8)) * 255
            
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
            'num_bee_detections': len(detections),
            'num_gt_bees': len(gt_annotations),
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

    def _load_external_aruco_index(self):
        """Load external ArUco detections from a CSV file or folder of CSV files."""
        external_path = self.config.get('external_aruco_path', '')
        if not external_path:
            return self._empty_external_aruco_index()

        target_video_ids = self._external_aruco_target_video_ids()
        if (
            self._external_aruco_index is not None
            and self._external_aruco_path == external_path
            and self._external_aruco_video_ids == target_video_ids
        ):
            return self._external_aruco_index

        path = Path(external_path)
        if path.is_file():
            csv_paths = [path]
        elif path.is_dir():
            csv_paths = self._find_external_aruco_csv_paths(path, target_video_ids)
        else:
            self.log_message.emit(f"  WARNING: External ArUco path does not exist: {path}")
            self._external_aruco_index = self._empty_external_aruco_index()
            self._external_aruco_path = external_path
            self._external_aruco_video_ids = target_video_ids
            return self._external_aruco_index

        index = self._empty_external_aruco_index()
        skipped_rows = 0
        skipped_files = 0

        for csv_path in csv_paths:
            loaded_from_file = 0
            try:
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        skipped_files += 1
                        continue

                    for row in reader:
                        record = self._external_aruco_record_from_row(row, csv_path)
                        if not record:
                            skipped_rows += 1
                            continue
                        self._add_external_aruco_record(index, record)
                        loaded_from_file += 1
            except Exception as e:
                skipped_files += 1
                self.log_message.emit(f"  WARNING: Could not read external ArUco CSV {csv_path}: {e}")

            if loaded_from_file:
                index['total_records'] += loaded_from_file

        self._external_aruco_index = index
        self._external_aruco_path = external_path
        self._external_aruco_video_ids = target_video_ids
        self.log_message.emit(
            f"  Loaded {index['total_records']} external ArUco detection rows "
            f"from {len(csv_paths)} CSV file(s)"
        )
        if skipped_files or skipped_rows:
            self.log_message.emit(
                f"  External ArUco skipped {skipped_files} file(s), {skipped_rows} row(s)"
            )
        return index

    def _external_aruco_target_video_ids(self):
        """Return selected validation video IDs for targeted external CSV discovery."""
        video_ids = {
            getattr(sequence, 'video_id', None)
            for sequence in self.config.get('sequences', [])
        }
        return frozenset(video_id for video_id in video_ids if video_id)

    def _find_external_aruco_csv_paths(self, folder, target_video_ids):
        """Find likely external ArUco CSVs without opening unrelated files."""
        if not target_video_ids:
            return sorted(folder.rglob('*.csv'))

        if len(target_video_ids) > 20:
            csv_paths = [
                csv_path for csv_path in folder.rglob('*.csv')
                if any(video_id in csv_path.name for video_id in target_video_ids)
            ]
        else:
            csv_paths = set()
            for video_id in sorted(target_video_ids):
                csv_paths.update(folder.rglob(f"*{video_id}*.csv"))
            csv_paths = sorted(csv_paths)

        self.log_message.emit(
            f"  External ArUco CSV search: {len(csv_paths)} candidate file(s) "
            f"for {len(target_video_ids)} selected video(s)"
        )
        return csv_paths

    def _empty_external_aruco_index(self):
        """Return the empty index structure used for external ArUco detections."""
        return {
            'by_frame': defaultdict(list),
            'by_stem': defaultdict(list),
            'videos': set(),
            'total_records': 0,
        }

    def _external_aruco_record_from_row(self, row, csv_path):
        """Normalize one external ArUco CSV row into a small marker record."""
        code = self._row_value(row, ('tag_id', 'ID', 'aruco_id', 'aruco_code', 'marker_id'))
        if code.upper() == 'X' or csv_path.stem.lower().endswith('_noid'):
            code = 'noID'
        if code == "":
            return None

        x_value = self._row_value(
            row,
            ('center_x', 'centroidX', 'aruco_centroidX', 'centerX', 'x')
        )
        y_value = self._row_value(
            row,
            ('center_y', 'centroidY', 'aruco_centroidY', 'centerY', 'y')
        )
        instance_id = self._parse_optional_int(
            self._row_value(row, ('instance_id', 'bee_id', 'mask_id', 'track_id'))
        )

        if (x_value == "" or y_value == "") and instance_id is None:
            return None

        x = self._parse_optional_float(x_value)
        y = self._parse_optional_float(y_value)
        if (x_value != "" or y_value != "") and (x is None or y is None):
            return None

        frame_number = self._parse_optional_int(
            self._row_value(row, ('frame', 'frame_number', 'frame_idx', 'frame_index', 'image_index'))
        )
        path_value = self._row_value(
            row,
            ('image_path', 'relative_image_path', 'filename', 'file', 'source_csv')
        )
        video_id = self._row_value(row, ('video_id', 'video', 'source_video'))
        image_stem = ""
        if path_value:
            image_stem = Path(path_value).stem
            if not video_id:
                video_id = self._video_id_from_external_stem(image_stem)

        source_csv_value = self._row_value(row, ('source_csv', 'source_csv_path'))
        if not video_id and source_csv_value:
            video_id = self._video_id_from_external_stem(Path(source_csv_value).stem)
        if not video_id:
            video_id = self._video_id_from_external_stem(csv_path.stem)

        return {
            'code': str(code).strip(),
            'x': x,
            'y': y,
            'instance_id': instance_id,
            'frame_number': frame_number,
            'video_id': video_id,
            'image_stem': image_stem,
            'source_csv': str(csv_path),
        }

    def _add_external_aruco_record(self, index, record):
        """Add one normalized external ArUco record to all useful lookup keys."""
        frame_number = record.get('frame_number')
        video_id = record.get('video_id') or ""
        image_stem = record.get('image_stem') or ""

        if frame_number is not None:
            if video_id:
                index['by_frame'][(video_id, frame_number)].append(record)
            else:
                index['by_frame'][("", frame_number)].append(record)

        if image_stem:
            index['by_stem'][image_stem].append(record)
            if video_id:
                index['by_stem'][f"{video_id}/{image_stem}"].append(record)

        if video_id:
            index['videos'].add(video_id)

    def _external_aruco_available_for_video(self, video_id):
        """Return True when the external CSV index has rows for this video."""
        if not video_id:
            return False
        index = self._load_external_aruco_index()
        return video_id in index.get('videos', set())

    def _get_external_aruco_records(self, video_id, frame_idx, frame_path):
        """Return external ArUco records for one validation frame."""
        index = self._load_external_aruco_index()
        frame_stem = frame_path.stem
        candidate_records = []
        candidate_records.extend(index['by_frame'].get((video_id, frame_idx), []))
        if not candidate_records:
            candidate_records.extend(index['by_frame'].get(("", frame_idx), []))

        candidate_records.extend(index['by_stem'].get(f"{video_id}/{frame_stem}", []))
        candidate_records.extend(index['by_stem'].get(frame_stem, []))

        deduped = []
        seen = set()
        for record in candidate_records:
            if not self._external_aruco_record_matches_frame(
                record, video_id, frame_idx, frame_stem
            ):
                continue
            key = (
                record.get('code'),
                record.get('x'),
                record.get('y'),
                record.get('instance_id'),
                record.get('source_csv')
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def _remap_aruco_detections_after_identity_update(
        self,
        aruco_detections,
        detections,
        aruco_identity
    ):
        """Keep external CSV detections distinct from persistent ArUco assignments.

        External CSV rows are first linked to the track ID present before ArUco
        identity correction. If a real-code detection triggers a track merge,
        the Detection object is rewritten to the persistent target ID. In that
        one case, move the raw detection to the target ID so it still exports on
        the visible row. Do not add other applied/persistent ArUco assignments:
        those belong in aruco_code, not aruco_detected.
        """
        current_ids = {det.instance_id for det in detections if det.instance_id is not None}
        remapped = {}

        for instance_id, code in aruco_detections.items():
            if instance_id in current_ids:
                remapped[instance_id] = code
                continue

            if code == 'noID' or aruco_identity is None:
                continue

            target_id = aruco_identity.aruco_to_bee.get(str(code))
            if target_id in current_ids:
                remapped[target_id] = code

        return remapped

    def _external_aruco_record_matches_frame(self, record, video_id, frame_idx, frame_stem):
        """Return True when a normalized external CSV row belongs to this frame."""
        record_video_id = record.get('video_id') or ""
        if record_video_id and video_id and record_video_id != video_id:
            return False

        record_frame = record.get('frame_number')
        if record_frame is not None:
            return int(record_frame) == int(frame_idx)

        image_stem = record.get('image_stem') or ""
        if image_stem:
            return image_stem == frame_stem

        return False

    def _match_external_aruco_to_bees(self, marker_records, detections, frame_idx=None):
        """Match external marker centers to tracked bee detections."""
        markers_by_instance = defaultdict(list)
        stats = {
            'csv_rows': len(marker_records),
            'csv_aruco_rows': sum(
                1 for marker in marker_records
                if marker.get('code') and marker.get('code') != 'noID'
            ),
            'csv_noid_rows': sum(
                1 for marker in marker_records
                if marker.get('code') == 'noID'
            ),
            'matched_mask': 0,
            'matched_bbox': 0,
            'matched_instance_id': 0,
            'linked_aruco_rows': 0,
            'linked_noid_rows': 0,
            'unmatched': 0,
            'ambiguous': 0,
            'multi_code_instances': 0,
        }

        for marker in marker_records:
            code = marker.get('code')
            if not code:
                continue

            matched_ids = []
            match_source = None
            x = marker.get('x')
            y = marker.get('y')
            if x is not None and y is not None:
                mask_matched_ids = []
                bbox_matched_ids = []
                for det in detections:
                    if det.instance_id is None:
                        continue
                    match_type = self._point_detection_match_type(x, y, det)
                    if match_type == 'mask':
                        mask_matched_ids.append(det.instance_id)
                    elif match_type == 'bbox':
                        bbox_matched_ids.append(det.instance_id)
                matched_ids = mask_matched_ids if mask_matched_ids else bbox_matched_ids
                match_source = 'mask' if mask_matched_ids else 'bbox'
            elif marker.get('instance_id') is not None:
                matched_ids.append(marker['instance_id'])
                match_source = 'instance_id'

            unique_matched_ids = set(matched_ids)
            if len(unique_matched_ids) == 0:
                stats['unmatched'] += 1
                continue
            if len(unique_matched_ids) != 1:
                stats['ambiguous'] += 1
                continue

            if match_source == 'mask':
                stats['matched_mask'] += 1
            elif match_source == 'bbox':
                stats['matched_bbox'] += 1
            elif match_source == 'instance_id':
                stats['matched_instance_id'] += 1

            if code == 'noID':
                stats['linked_noid_rows'] += 1
            else:
                stats['linked_aruco_rows'] += 1

            markers_by_instance[matched_ids[0]].append(str(code))

        marker_codes = {}
        detected_codes = {}
        for instance_id, codes in markers_by_instance.items():
            unique_codes = sorted(set(codes))
            real_codes = [code for code in unique_codes if code != 'noID']
            if len(real_codes) == 1:
                detected_codes[instance_id] = real_codes[0]
                marker_codes[instance_id] = real_codes[0]
            elif len(real_codes) == 0 and unique_codes == ['noID']:
                detected_codes[instance_id] = 'noID'
            else:
                stats['multi_code_instances'] += 1
        stats['exported_detections'] = len(detected_codes)
        return marker_codes, detected_codes, stats

    def _log_external_aruco_match_stats(self, frame_idx, marker_records, detections, stats):
        """Log why external ArUco rows were not linked to bee detections."""
        matched_total = (
            stats.get('matched_mask', 0) +
            stats.get('matched_bbox', 0) +
            stats.get('matched_instance_id', 0)
        )
        self.log_message.emit(
            f"    Frame {frame_idx}: external ArUco rows={stats.get('csv_rows', 0)}, "
            f"linked={matched_total}, exported={stats.get('exported_detections', 0)}, "
            f"mask={stats.get('matched_mask', 0)}, bbox={stats.get('matched_bbox', 0)}, "
            f"unmatched={stats.get('unmatched', 0)}, ambiguous={stats.get('ambiguous', 0)}, "
            f"multi-code bees={stats.get('multi_code_instances', 0)}"
        )

        if stats.get('unmatched', 0) and detections:
            samples = []
            for marker in marker_records[:3]:
                x = marker.get('x')
                y = marker.get('y')
                if x is None or y is None:
                    continue
                nearest = self._nearest_detection_distance(x, y, detections)
                if nearest is not None:
                    bee_id, distance = nearest
                    samples.append(
                        f"{marker.get('code')}@({x:.1f},{y:.1f})->ID:{bee_id} {distance:.1f}px"
                    )
            if samples:
                self.log_message.emit("      Nearest bee samples: " + "; ".join(samples))

    def _nearest_detection_distance(self, x, y, detections):
        """Return nearest detection ID and bbox-center distance to a point."""
        nearest = None
        for det in detections:
            if det.instance_id is None or det.bbox is None:
                continue
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            distance = float(np.linalg.norm(np.array([x, y]) - np.array([cx, cy])))
            if nearest is None or distance < nearest[1]:
                nearest = (det.instance_id, distance)
        return nearest

    def _point_detection_match_type(self, x, y, det):
        """Return whether a marker center matches a detection by mask or bbox."""
        x1, y1, x2, y2 = det.bbox
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return None

        if det.mask is not None:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= yi < det.mask.shape[0] and 0 <= xi < det.mask.shape[1]:
                if det.mask[yi, xi] > 0:
                    return 'mask'

        return 'bbox'

    def _row_value(self, row, candidates):
        """Get a CSV row value using case-insensitive candidate column names."""
        lower_to_key = {str(key).lower(): key for key in row.keys()}
        for candidate in candidates:
            key = lower_to_key.get(str(candidate).lower())
            if key is not None:
                value = row.get(key, "")
                if value is None:
                    return ""
                value = str(value).strip()
                if value.lower() in {'nan', 'none', 'null'}:
                    return ""
                return value
        return ""

    def _parse_optional_float(self, value):
        if value == "" or value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_optional_int(self, value):
        if value == "" or value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _video_id_from_external_stem(self, stem):
        """Infer a likely video ID from an external filename stem."""
        if not stem:
            return ""
        lower_stem = stem.lower()
        for suffix in ('_raw', '_noid', '_aruco_detections', '_detections'):
            if lower_stem.endswith(suffix):
                return stem[:-len(suffix)]
        return stem

    def _load_video_aruco_tracking(self, video_id):
        """Load optional video-level GT mapping as bee instance ID -> ArUco code."""
        video_annotations_path = (
            self.main_window.project_path / 'annotations' / 'json' /
            video_id / 'video_annotations.json'
        )
        if not video_annotations_path.exists():
            return {}

        try:
            with open(video_annotations_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.log_message.emit(
                f"  WARNING: Could not load video-level ArUco tracking for {video_id}: {e}"
            )
            return {}

        # Stored on disk as ArUco code -> bee instance ID.
        aruco_tracking = data.get('aruco_tracking', {})
        return {str(gt_id): str(aruco_code) for aruco_code, gt_id in aruco_tracking.items()}

    def _get_gt_aruco_code(self, gt_annotation, gt_id=None, video_aruco_tracking=None):
        """Return the GT ArUco code attached to a matched bee annotation, if available."""
        if gt_id is not None and video_aruco_tracking:
            aruco_code = video_aruco_tracking.get(str(gt_id), "")
            if aruco_code:
                return aruco_code

        if gt_annotation:
            marker = gt_annotation.get('marker')
            if isinstance(marker, dict) and marker.get('type') == 'aruco':
                marker_id = marker.get('id', marker.get('marker_id'))
                if marker_id is not None:
                    return str(marker_id)

            for key in ('aruco_code', 'aruco_id', 'marker_id'):
                if gt_annotation.get(key) is not None:
                    return str(gt_annotation[key])

        return ""

    def _aruco_tracking_status(self, tracking_status, aruco_code, gt_aruco_code):
        """Classify tracked ArUco identity against the matched GT bee identity."""
        if tracking_status == 'FP':
            return 'unmatched_detection'
        if not gt_aruco_code:
            return 'no_gt_aruco'
        if aruco_code and aruco_code == gt_aruco_code:
            return 'true_positive_aruco_tracked'
        if aruco_code and aruco_code != gt_aruco_code:
            return 'false_positive_aruco_tracked'
        return 'false_negative_aruco_tracked'

    def _mask_polygon_string(self, mask) -> str:
        """Convert a binary mask to the same polygon string format used by batch export."""
        if mask is None or not np.any(mask > 0):
            return ""

        polygon = mask_to_simplified_polygon(mask.astype(np.uint8), epsilon_percent=2.0)
        return polygon_to_string(polygon) if polygon else ""

    def _build_detection_export_rows(
        self,
        sequence,
        algo_name,
        frame_idx,
        detections,
        gt_annotations,
        detection_status,
        aruco_detections,
        aruco_identity,
        video_aruco_tracking
    ):
        """Build detailed per-bee rows and ArUco tracking summary counts for one frame."""
        rows = []
        aruco_summary = defaultdict(int)

        for det in detections:
            tracking_status, matched_gt_id, matched_gt_idx, iou = detection_status.get(
                id(det), ('FP', None, None, 0.0)
            )
            matched_gt = (
                gt_annotations[matched_gt_idx]
                if matched_gt_idx is not None and 0 <= matched_gt_idx < len(gt_annotations)
                else None
            )
            gt_aruco_code = self._get_gt_aruco_code(
                matched_gt,
                gt_id=matched_gt_id,
                video_aruco_tracking=video_aruco_tracking
            )
            aruco_code = ""
            if aruco_identity is not None and det.instance_id is not None:
                aruco_code = aruco_identity.bee_to_aruco.get(det.instance_id, "")
            aruco_detected = ""
            if det.instance_id is not None:
                aruco_detected = aruco_detections.get(det.instance_id, "")

            if aruco_detected:
                aruco_summary['num_aruco_or_noid_detections'] += 1
                if aruco_detected == 'noID':
                    aruco_summary['num_noid_detections'] += 1
                else:
                    aruco_summary['num_aruco_detections'] += 1

            aruco_status = self._aruco_tracking_status(
                tracking_status=tracking_status,
                aruco_code=aruco_code,
                gt_aruco_code=gt_aruco_code
            )
            if aruco_status in {
                'true_positive_aruco_tracked',
                'false_positive_aruco_tracked',
                'false_negative_aruco_tracked'
            }:
                aruco_summary[f'num_{aruco_status}'] += 1

            bbox_x, bbox_y, bbox_x2, bbox_y2 = det.bbox
            bbox_width = bbox_x2 - bbox_x
            bbox_height = bbox_y2 - bbox_y
            if det.mask is not None:
                centroid = self._get_centroid(det.mask, det.bbox)
                centroid_x, centroid_y = centroid if centroid is not None else ("", "")
            else:
                centroid_x = (bbox_x + bbox_x2) / 2
                centroid_y = (bbox_y + bbox_y2) / 2

            rows.append({
                'video_id': sequence.video_id,
                'sequence_id': sequence.sequence_id,
                'algorithm': algo_name,
                'frame_number': frame_idx,
                'bee_id': det.instance_id if det.instance_id is not None else "",
                'aruco_code': aruco_code,
                'aruco_detected': aruco_detected,
                'gt_aruco_code': gt_aruco_code,
                'aruco_tracking_status': aruco_status,
                'tracking_status': tracking_status,
                'matched_gt_id': matched_gt_id if matched_gt_id is not None else "",
                'match_iou': f"{iou:.4f}" if iou else "",
                'bbox_x': f"{bbox_x:.2f}",
                'bbox_y': f"{bbox_y:.2f}",
                'bbox_width': f"{bbox_width:.2f}",
                'bbox_height': f"{bbox_height:.2f}",
                'confidence': f"{det.confidence:.4f}",
                'centroid_x': f"{centroid_x:.2f}" if centroid_x != "" else "",
                'centroid_y': f"{centroid_y:.2f}" if centroid_y != "" else "",
                'pred_polygon': self._mask_polygon_string(det.mask)
            })

        return rows, aruco_summary

    def _bee_detection_export_fieldnames(self):
        """Column order for detailed validation bee detection CSVs."""
        return [
            'video_id', 'sequence_id', 'algorithm', 'frame_number',
            'bee_id', 'aruco_code', 'aruco_detected', 'gt_aruco_code',
            'aruco_tracking_status', 'tracking_status', 'matched_gt_id', 'match_iou',
            'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'confidence',
            'centroid_x', 'centroid_y', 'pred_polygon'
        ]

    def _append_bee_detection_rows(self, results_folder, algo_name, rows):
        """Append detailed per-bee detection rows for an algorithm."""
        csv_path = results_folder / f"{algo_name}_bee_detections.csv"
        write_header = not csv_path.exists()

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._bee_detection_export_fieldnames())
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    
    def _aggregate_sequence_metrics(self, frame_metrics, sequence, aruco_summary=None):
        """Aggregate frame metrics into sequence metrics"""
        aruco_summary = aruco_summary or {}
        total_tp = sum(m['tp'] for m in frame_metrics)
        total_fp = sum(m['fp'] for m in frame_metrics)
        total_fn = sum(m['fn'] for m in frame_metrics)
        total_id_switches = sum(m['id_switches'] for m in frame_metrics)
        total_iou = sum(m['total_iou'] for m in frame_metrics)
        total_matched = sum(m['matched_count'] for m in frame_metrics)
        num_bee_detections = sum(m.get('num_bee_detections', 0) for m in frame_metrics)
        num_gt_bees = sum(m.get('num_gt_bees', 0) for m in frame_metrics)
        
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
            'num_bee_detections': int(num_bee_detections),
            'num_gt_bees': int(num_gt_bees),
            'num_aruco_detections': int(aruco_summary.get('num_aruco_detections', 0)),
            'num_noid_detections': int(aruco_summary.get('num_noid_detections', 0)),
            'num_aruco_or_noid_detections': int(aruco_summary.get('num_aruco_or_noid_detections', 0)),
            'num_external_csv_aruco_rows': int(aruco_summary.get('num_external_csv_aruco_rows', 0)),
            'num_external_csv_noid_rows': int(aruco_summary.get('num_external_csv_noid_rows', 0)),
            'num_external_csv_aruco_or_noid_rows': int(aruco_summary.get('num_external_csv_aruco_or_noid_rows', 0)),
            'num_external_csv_linked_aruco_rows': int(aruco_summary.get('num_external_csv_linked_aruco_rows', 0)),
            'num_external_csv_linked_noid_rows': int(aruco_summary.get('num_external_csv_linked_noid_rows', 0)),
            'num_external_csv_aruco_detections': int(aruco_summary.get('num_external_csv_aruco_detections', 0)),
            'num_external_csv_noid_detections': int(aruco_summary.get('num_external_csv_noid_detections', 0)),
            'num_external_csv_aruco_or_noid_detections': int(
                aruco_summary.get('num_external_csv_aruco_or_noid_detections', 0)
            ),
            'num_builtin_aruco_detections': int(aruco_summary.get('num_builtin_aruco_detections', 0)),
            'num_builtin_noid_detections': int(aruco_summary.get('num_builtin_noid_detections', 0)),
            'num_builtin_aruco_or_noid_detections': int(
                aruco_summary.get('num_builtin_aruco_or_noid_detections', 0)
            ),
            'num_true_positive_aruco_tracked': int(aruco_summary.get('num_true_positive_aruco_tracked', 0)),
            'num_false_positive_aruco_tracked': int(aruco_summary.get('num_false_positive_aruco_tracked', 0)),
            'num_false_negative_aruco_tracked': int(aruco_summary.get('num_false_negative_aruco_tracked', 0)),
            'aruco_requested_source': aruco_summary.get('aruco_requested_source', 'disabled'),
            'aruco_detection_source': aruco_summary.get('aruco_detection_source', 'disabled'),
            'external_aruco_available': bool(aruco_summary.get('external_aruco_available', False)),
            'external_aruco_fallback_to_builtin': bool(
                aruco_summary.get('external_aruco_fallback_to_builtin', False)
            ),
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
                'num_bee_detections': sum([s.get('num_bee_detections', 0) for s in sequences]),
                'num_gt_bees': sum([s.get('num_gt_bees', 0) for s in sequences]),
                'num_aruco_detections': sum([s.get('num_aruco_detections', 0) for s in sequences]),
                'num_noid_detections': sum([s.get('num_noid_detections', 0) for s in sequences]),
                'num_aruco_or_noid_detections': sum([s.get('num_aruco_or_noid_detections', 0) for s in sequences]),
                'num_external_csv_aruco_rows': sum([s.get('num_external_csv_aruco_rows', 0) for s in sequences]),
                'num_external_csv_noid_rows': sum([s.get('num_external_csv_noid_rows', 0) for s in sequences]),
                'num_external_csv_aruco_or_noid_rows': sum([
                    s.get('num_external_csv_aruco_or_noid_rows', 0) for s in sequences
                ]),
                'num_external_csv_linked_aruco_rows': sum([
                    s.get('num_external_csv_linked_aruco_rows', 0) for s in sequences
                ]),
                'num_external_csv_linked_noid_rows': sum([
                    s.get('num_external_csv_linked_noid_rows', 0) for s in sequences
                ]),
                'num_external_csv_aruco_detections': sum([
                    s.get('num_external_csv_aruco_detections', 0) for s in sequences
                ]),
                'num_external_csv_noid_detections': sum([
                    s.get('num_external_csv_noid_detections', 0) for s in sequences
                ]),
                'num_external_csv_aruco_or_noid_detections': sum([
                    s.get('num_external_csv_aruco_or_noid_detections', 0) for s in sequences
                ]),
                'num_builtin_aruco_detections': sum([s.get('num_builtin_aruco_detections', 0) for s in sequences]),
                'num_builtin_noid_detections': sum([s.get('num_builtin_noid_detections', 0) for s in sequences]),
                'num_builtin_aruco_or_noid_detections': sum([
                    s.get('num_builtin_aruco_or_noid_detections', 0) for s in sequences
                ]),
                'num_true_positive_aruco_tracked': sum([s.get('num_true_positive_aruco_tracked', 0) for s in sequences]),
                'num_false_positive_aruco_tracked': sum([s.get('num_false_positive_aruco_tracked', 0) for s in sequences]),
                'num_false_negative_aruco_tracked': sum([s.get('num_false_negative_aruco_tracked', 0) for s in sequences]),
                'num_sequences_external_aruco_used': sum([
                    s.get('aruco_detection_source') == 'external' for s in sequences
                ]),
                'num_sequences_external_aruco_missing': sum([
                    s.get('external_aruco_fallback_to_builtin', False) for s in sequences
                ]),
                'external_aruco_used_video_ids': sorted({
                    s.get('video_id') for s in sequences
                    if s.get('aruco_detection_source') == 'external'
                }),
                'external_aruco_missing_video_ids': sorted({
                    s.get('video_id') for s in sequences
                    if s.get('external_aruco_fallback_to_builtin', False)
                }),
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
                    'id_switches', 'tp', 'idtp', 'fp', 'fn',
                    'num_bee_detections',
                    'num_gt_bees',
                    'num_aruco_detections',
                    'num_noid_detections',
                    'num_aruco_or_noid_detections',
                    'num_external_csv_aruco_rows',
                    'num_external_csv_noid_rows',
                    'num_external_csv_aruco_or_noid_rows',
                    'num_external_csv_linked_aruco_rows',
                    'num_external_csv_linked_noid_rows',
                    'num_external_csv_aruco_detections',
                    'num_external_csv_noid_detections',
                    'num_external_csv_aruco_or_noid_detections',
                    'num_builtin_aruco_detections',
                    'num_builtin_noid_detections',
                    'num_builtin_aruco_or_noid_detections',
                    'num_true_positive_aruco_tracked',
                    'num_false_positive_aruco_tracked',
                    'num_false_negative_aruco_tracked',
                    'aruco_requested_source',
                    'aruco_detection_source',
                    'external_aruco_available',
                    'external_aruco_fallback_to_builtin'
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
                'num_bee_detections': sum([s.get('num_bee_detections', 0) for s in sequences]),
                'num_gt_bees': sum([s.get('num_gt_bees', 0) for s in sequences]),
                'num_aruco_detections': sum([s.get('num_aruco_detections', 0) for s in sequences]),
                'num_noid_detections': sum([s.get('num_noid_detections', 0) for s in sequences]),
                'num_aruco_or_noid_detections': sum([s.get('num_aruco_or_noid_detections', 0) for s in sequences]),
                'num_external_csv_aruco_rows': sum([s.get('num_external_csv_aruco_rows', 0) for s in sequences]),
                'num_external_csv_noid_rows': sum([s.get('num_external_csv_noid_rows', 0) for s in sequences]),
                'num_external_csv_aruco_or_noid_rows': sum([
                    s.get('num_external_csv_aruco_or_noid_rows', 0) for s in sequences
                ]),
                'num_external_csv_linked_aruco_rows': sum([
                    s.get('num_external_csv_linked_aruco_rows', 0) for s in sequences
                ]),
                'num_external_csv_linked_noid_rows': sum([
                    s.get('num_external_csv_linked_noid_rows', 0) for s in sequences
                ]),
                'num_external_csv_aruco_detections': sum([
                    s.get('num_external_csv_aruco_detections', 0) for s in sequences
                ]),
                'num_external_csv_noid_detections': sum([
                    s.get('num_external_csv_noid_detections', 0) for s in sequences
                ]),
                'num_external_csv_aruco_or_noid_detections': sum([
                    s.get('num_external_csv_aruco_or_noid_detections', 0) for s in sequences
                ]),
                'num_builtin_aruco_detections': sum([s.get('num_builtin_aruco_detections', 0) for s in sequences]),
                'num_builtin_noid_detections': sum([s.get('num_builtin_noid_detections', 0) for s in sequences]),
                'num_builtin_aruco_or_noid_detections': sum([
                    s.get('num_builtin_aruco_or_noid_detections', 0) for s in sequences
                ]),
                'num_true_positive_aruco_tracked': sum([s.get('num_true_positive_aruco_tracked', 0) for s in sequences]),
                'num_false_positive_aruco_tracked': sum([s.get('num_false_positive_aruco_tracked', 0) for s in sequences]),
                'num_false_negative_aruco_tracked': sum([s.get('num_false_negative_aruco_tracked', 0) for s in sequences]),
                'num_sequences_external_aruco_used': sum([
                    s.get('aruco_detection_source') == 'external' for s in sequences
                ]),
                'num_sequences_external_aruco_missing': sum([
                    s.get('external_aruco_fallback_to_builtin', False) for s in sequences
                ]),
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
                                  prev_frame_positions, frame_idx, viz_folder, metrics,
                                  aruco_assignments=None):
        """Save visualization of tracking for one frame"""
        # Create a copy of the frame for drawing
        viz_frame = frame.copy()
        aruco_assignments = aruco_assignments or {}
        
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

                pred_polygon = mask_to_simplified_polygon(det.mask.astype(np.uint8), epsilon_percent=2.0)
                if pred_polygon and len(pred_polygon) >= 3:
                    polygon_points = np.array(pred_polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(viz_frame, [polygon_points], True, (255, 255, 0), 2)
                    for point in pred_polygon:
                        cv2.circle(viz_frame, (int(point[0]), int(point[1])), 3, (255, 255, 0), -1)
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
                aruco_code = aruco_assignments.get(det.instance_id, "")
                label = f"ID:{det.instance_id}"
                if aruco_code:
                    label += f" A:{aruco_code}"
                cv2.putText(viz_frame, label, label_pos,
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
