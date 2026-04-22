"""
Background worker for BeeHaveSquE pipeline validation
"""

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import cv2
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from core.inference_utils import create_temporal_image, run_beehavesque_soho
from core.instance_tracker import InstanceTracker


class ValidationWorker(QThread):
    """Worker thread for running BeeHaveSquE pipeline validation"""
    
    # Signals
    video_started = pyqtSignal(str, int, int, int)  # video_id, num_frames, video_num, total_videos
    frame_processed = pyqtSignal(str, int, int, dict)  # video_id, frame_idx, total, metrics
    video_completed = pyqtSignal(str, dict, int, int)  # video_id, metrics, video_num, total_videos
    validation_complete = pyqtSignal(dict, str)  # overall_metrics, results_path
    validation_failed = pyqtSignal(str)  # error_msg
    
    def __init__(self, main_window, validation_config, validation_videos):
        """
        Initialize validation worker
        
        Args:
            main_window: Reference to main window for accessing methods
            validation_config: Configuration dict from dialog
            validation_videos: List of video IDs to validate
        """
        super().__init__()
        self.main_window = main_window
        self.config = validation_config
        self.validation_videos = validation_videos
        self.should_stop = False
        
        # Track current video's frames and tracker state
        self.current_video_frames = []
        self.current_tracker = None
        self.previous_frame_detections = []
        
    def stop(self):
        """Request worker to stop"""
        self.should_stop = True
    
    def run(self):
        """Run validation in background thread"""
        try:
            print(f"\n=== ValidationWorker starting ===")
            print(f"Videos to validate: {len(self.validation_videos)}")
            for i, vid in enumerate(self.validation_videos, 1):
                print(f"  {i}. {vid}")
            
            print(f"\nSOHO Inference Parameters:")
            print(f"  Slice size: {self.config.get('slice_size', 640)}x{self.config.get('slice_size', 640)} px")
            print(f"  Overlap ratio: {self.config.get('overlap_ratio', 0.5):.2f} ({int(self.config.get('overlap_ratio', 0.5)*100)}%)")
            print(f"  Edge filter: {self.config.get('edge_filter', 50)} px")
            print(f"\nValidation Parameters:")
            print(f"  IoU threshold: {self.config.get('iou_threshold', 0.5):.2f}")
            print(f"  Confidence threshold: {self.config.get('conf_threshold', 0.5):.2f}")
            print(f"  ID switch IoU: {self.config.get('id_switch_iou', 0.3):.2f}")
            
            # Create results folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = Path(self.main_window.yolo_beehavesque_toolbar.model_path)
            results_folder = model_path.parent / f"validation_results_{timestamp}"
            results_folder.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_file = results_folder / "config.json"
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Track overall metrics
            all_video_metrics = []
            overall_stats = {
                'total_tp': 0,
                'total_fp': 0,
                'total_fn': 0,
                'total_id_switches': 0,
                'total_gt_instances': 0,
                'total_frames': 0
            }
            
            # Process each validation video
            for video_num, video_id in enumerate(self.validation_videos, 1):
                if self.should_stop:
                    return
                
                try:
                    # Run validation on this video
                    video_metrics, video_stats = self._validate_video(
                        video_id, video_num, len(self.validation_videos), results_folder
                    )
                    
                    if video_metrics:
                        all_video_metrics.append({
                            'video_id': video_id,
                            **video_metrics
                        })
                        
                        # Accumulate stats
                        for key in overall_stats:
                            if key in video_stats:
                                overall_stats[key] += video_stats[key]
                        
                        self.video_completed.emit(video_id, video_metrics, video_num, len(self.validation_videos))
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Error validating video {video_id}: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    # Continue with next video instead of failing completely
                    continue
            
            if self.should_stop:
                return
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(overall_stats)
            
            # Save results
            results = {
                'config': self.config,
                'overall': overall_metrics,
                'per_video': all_video_metrics,
                'timestamp': timestamp
            }
            
            results_file = results_folder / "metrics.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save summary text
            summary_file = results_folder / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write("BeeHaveSquE Pipeline Validation Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Videos validated: {len(all_video_metrics)}\n\n")
                f.write("Overall Metrics:\n")
                f.write(f"  MOTA: {overall_metrics.get('mota', 0):.3f}\n")
                f.write(f"  IDF1: {overall_metrics.get('idf1', 0):.3f}\n")
                f.write(f"  Precision: {overall_metrics.get('precision', 0):.3f}\n")
                f.write(f"  Recall: {overall_metrics.get('recall', 0):.3f}\n")
                f.write(f"  ID Switches: {overall_metrics.get('id_switches', 0)}\n")
                f.write(f"  Total Frames: {overall_metrics.get('total_frames', 0)}\n\n")
                f.write("Per-Video Results:\n")
                for vm in all_video_metrics:
                    f.write(f"\n  {vm['video_id']}:\n")
                    f.write(f"    MOTA: {vm.get('mota', 0):.3f}\n")
                    f.write(f"    IDF1: {vm.get('idf1', 0):.3f}\n")
                    f.write(f"    ID Switches: {vm.get('id_switches', 0)}\n")
            
            self.validation_complete.emit(overall_metrics, str(results_folder))
            
        except Exception as e:
            import traceback
            error_msg = f"Validation failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.validation_failed.emit(str(e))
    
    def _validate_video(self, video_id: str, video_num: int, total_videos: int, results_folder: Path) -> Tuple[Dict, Dict]:
        """
        Validate a single video
        
        Returns:
            (video_metrics, video_stats)
        """
        # Get validation frames for this video
        val_frames = self._get_validation_frames(video_id)
        
        if not val_frames:
            print(f"No validation frames found for video {video_id}")
            return None, None
        
        print(f"\n=== Validating video {video_num}/{total_videos}: {video_id} ({len(val_frames)} frames) ===")
        self.video_started.emit(video_id, len(val_frames), video_num, total_videos)
        
        # Create video-specific results folder
        video_folder = results_folder / video_id
        video_folder.mkdir(exist_ok=True)
        
        # Setup video-specific state
        self._setup_video_state(video_id, val_frames[0])
        
        # === STEP 1: Run inference on first frame and establish ID mapping ===
        first_frame_idx = val_frames[0]
        
        # Run BeeHaveSquE SOHO on first frame (raw detections without IDs)
        raw_predictions_first = self._run_inference_on_frame(first_frame_idx)
        
        if not raw_predictions_first:
            print(f"No predictions on first frame of {video_id}")
            return None, None
        
        # Initialize tracker with first frame detections to assign track IDs
        matched_detections = self.current_tracker.match_detections_to_tracks(
            raw_predictions_first,
            first_frame_idx
        )
        
        # Convert to Detection objects with assigned IDs
        predictions_first = []
        for detection, track_id in matched_detections:
            detection.instance_id = track_id
            predictions_first.append(detection)
        
        print(f"First frame: {len(predictions_first)} predictions with tracker IDs: {[p.instance_id for p in predictions_first]}")
        
        # Load ground truth for first frame
        gt_first = self._load_ground_truth(video_id, first_frame_idx)
        
        if not gt_first:
            print(f"No ground truth on first frame of {video_id}")
            return None, None
        
        print(f"First frame: {len(gt_first)} ground truth instances with IDs: {[gt.get('mask_id', gt.get('instance_id', '?')) for gt in gt_first]}")
        
        # Establish ID mapping: tracker_id -> ground_truth_id
        id_mapping = self._match_predictions_to_gt(predictions_first, gt_first)
        
        print(f"Initial ID mapping for {video_id}: {id_mapping}")
        
        # Store all predictions for later evaluation
        all_predictions = {first_frame_idx: predictions_first}
        all_ground_truth = {first_frame_idx: gt_first}
        
        # === STEP 2: Autonomous propagation through remaining frames ===
        for i, frame_idx in enumerate(val_frames[1:], 1):
            if self.should_stop:
                return None, None
            
            # Propagate autonomously (no GT matching!)
            predictions = self._propagate_to_frame(frame_idx)
            
            # Load GT for evaluation later
            gt = self._load_ground_truth(video_id, frame_idx)
            
            if predictions:
                all_predictions[frame_idx] = predictions
            if gt:
                all_ground_truth[frame_idx] = gt
            
            # Emit progress (but don't calculate full metrics yet - too slow)
            progress_pct = int((video_num - 1 + i / len(val_frames)) / total_videos * 100)
            self.frame_processed.emit(video_id, i, len(val_frames), {})
        
        # === STEP 3: Evaluate all predictions against GT ===
        video_metrics, video_stats = self._evaluate_predictions(
            all_predictions, all_ground_truth, id_mapping,
            video_id, video_folder
        )
        
        # === STEP 4: Generate visualizations for first and last frames ===
        if self.config.get('save_visualizations', True):
            print(f"Generating visualizations for {video_id}...")
            
            # First frame visualization
            first_frame_idx = val_frames[0]
            self._generate_comparison_visualization(
                first_frame_idx,
                all_predictions.get(first_frame_idx, []),
                all_ground_truth.get(first_frame_idx, []),
                id_mapping,
                video_folder / "frame_first_comparison.jpg",
                f"{video_id} - First Frame"
            )
            
            # Last frame visualization
            last_frame_idx = val_frames[-1]
            self._generate_comparison_visualization(
                last_frame_idx,
                all_predictions.get(last_frame_idx, []),
                all_ground_truth.get(last_frame_idx, []),
                id_mapping,
                video_folder / "frame_last_comparison.jpg",
                f"{video_id} - Last Frame"
            )
        
        return video_metrics, video_stats
    
    def _setup_video_state(self, video_id: str, first_frame_idx: int):
        """Setup state for processing a new video"""
        # Get all frames in this video's directory for temporal image creation
        frames_dir = self.main_window.project_path / 'frames' / video_id
        self.current_video_frames = sorted(
            list(frames_dir.glob('*.jpg')) + list(frames_dir.glob('*.png'))
        )
        
        # Reset tracker for this video
        self.current_tracker = InstanceTracker()
        self.previous_frame_detections = []
    
    def _get_validation_frames(self, video_id: str) -> List[int]:
        """Get list of validation frame indices (video-local) for this video with GT annotations"""
        val_frames = []
        
        print(f"  Searching for validation frames in {video_id}...")
        
        # Check for annotations in PNG+JSON format
        png_annotations_dir = self.main_window.project_path / 'annotations/png' / video_id
        json_annotations_dir = self.main_window.project_path / 'annotations/json' / video_id
        
        # Collect all annotation files (need both PNG and JSON)
        annotation_files = set()
        
        # Check PNG+JSON format (need both files)
        if png_annotations_dir.exists() and json_annotations_dir.exists():
            png_files = list(png_annotations_dir.glob('frame_*.png'))
            json_files = list(json_annotations_dir.glob('frame_*.json'))
            
            # Only count frames that have both PNG and JSON
            png_frames = {int(f.stem.split('_')[1]) for f in png_files}
            json_frames = {int(f.stem.split('_')[1]) for f in json_files}
            annotation_files = png_frames & json_frames
            
            print(f"  Found {len(annotation_files)} PNG+JSON annotation pairs")
        else:
            print(f"  Annotation directories not found for {video_id}")
        
        if not annotation_files:
            print(f"  No annotation files found for {video_id}")
            return []
        
        # Now verify each frame actually has annotations (not empty)
        for frame_idx in sorted(annotation_files):
            # Load and check if it has annotations
            annotations = self.main_window.annotation_manager.load_frame_annotations(
                self.main_window.project_path, video_id, frame_idx
            )
            if annotations and len(annotations) > 0:
                val_frames.append(frame_idx)
        
        print(f"  Found {len(val_frames)} validation frames with annotations for {video_id}")
        return sorted(val_frames)
    
    def _run_inference_on_frame(self, frame_idx: int) -> List:
        """Run BeeHaveSquE SOHO inference on a frame (frame_idx is video-local) - returns list of Detection objects"""
        try:
            # Get frame path from current video frames
            frame_path = self.current_video_frames[frame_idx]
            
            # Create temporal image
            temporal_img = create_temporal_image(
                Path(frame_path), 
                self.current_video_frames
            )
            
            if temporal_img is None:
                print(f"Failed to create temporal image for frame {frame_idx}")
                return []
            
            # Get model from toolbar
            model = self.main_window.yolo_beehavesque_toolbar.get_model()
            
            # Build SAHI params from config
            sahi_params = {
                'slice_height': self.config.get('slice_size', 640),
                'slice_width': self.config.get('slice_size', 640),
                'overlap_height_ratio': self.config.get('overlap_ratio', 0.5),
                'overlap_width_ratio': self.config.get('overlap_ratio', 0.5),
            }
            
            # Run SOHO inference with config parameters
            # Use CPU for validation to avoid "Invalid device id" errors in background QThread
            detections = run_beehavesque_soho(
                model,
                temporal_img,
                sahi_params,
                edge_filter=self.config.get('edge_filter', 50),
                device='cpu'  # Force CPU in background worker to avoid CUDA threading issues
            )
            
            return detections
            
        except Exception as e:
            print(f"Error running inference on frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _load_ground_truth(self, video_id: str, frame_idx: int) -> List:
        """Load ground truth annotations for a frame (frame_idx is video-local)"""
        annotations = self.main_window.annotation_manager.load_frame_annotations(
            self.main_window.project_path, video_id, frame_idx
        )
        return annotations if annotations else []
    
    def _match_predictions_to_gt(self, predictions: List, ground_truth: List) -> Dict[int, int]:
        """
        Match predictions to ground truth using Hungarian algorithm
        
        Returns:
            id_mapping: {predicted_id: ground_truth_id}
        """
        if not predictions or not ground_truth:
            return {}
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truth)))
        
        print(f"\n  Computing IoU for {len(predictions)} predictions x {len(ground_truth)} GT:")
        
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                # Handle both Detection objects and dict annotations
                pred_mask = pred.mask if hasattr(pred, 'mask') else pred['mask']
                gt_mask = gt.mask if hasattr(gt, 'mask') else gt['mask']
                
                # Debug mask properties
                if i == 0 and j == 0:
                    print(f"  Mask format check (pred[0] vs gt[0]):")
                    print(f"    Pred mask: shape={pred_mask.shape}, dtype={pred_mask.dtype}, min={pred_mask.min()}, max={pred_mask.max()}, nonzero={np.count_nonzero(pred_mask)}")
                    print(f"    GT mask:   shape={gt_mask.shape}, dtype={gt_mask.dtype}, min={gt_mask.min()}, max={gt_mask.max()}, nonzero={np.count_nonzero(gt_mask)}")
                
                iou_matrix[i, j] = self._compute_mask_iou(pred_mask, gt_mask)
        
        print(f"\n  IoU Matrix ({len(predictions)} predictions x {len(ground_truth)} GT):")
        print(f"  Prediction IDs: {[p.instance_id for p in predictions]}")
        print(f"  GT IDs: {[gt.get('mask_id', gt.get('instance_id', '?')) for gt in ground_truth]}")
        
        # Print IoU matrix with prediction and GT IDs as labels
        for i, pred in enumerate(predictions):
            pred_id = pred.instance_id
            ious = [f"{iou_matrix[i, j]:.3f}" for j in range(len(ground_truth))]
            print(f"  Pred ID {pred_id}: {ious}")
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximize IoU
        
        print(f"\n  Hungarian matches:")
        # Build mapping (only for matches above threshold)
        id_mapping = {}
        for pred_idx, gt_idx in zip(row_ind, col_ind):
            iou = iou_matrix[pred_idx, gt_idx]
            
            # Get predicted ID (tracker-assigned)
            pred = predictions[pred_idx]
            pred_id = pred.instance_id
            
            # Get GT ID
            gt = ground_truth[gt_idx]
            if hasattr(gt, 'instance_id'):
                gt_id = gt.instance_id
            else:
                gt_id = gt.get('mask_id', gt.get('instance_id', gt_idx + 1))
            
            if iou >= self.config['iou_threshold']:
                id_mapping[pred_id] = gt_id
                print(f"    ✓ Tracker ID {pred_id} -> GT ID {gt_id} (IoU={iou:.3f})")
            else:
                print(f"    ✗ Tracker ID {pred_id} -> GT ID {gt_id} (IoU={iou:.3f}) - below threshold {self.config['iou_threshold']:.3f}")
        
        return id_mapping
    
    def _propagate_to_frame(self, target_frame_idx: int) -> List:
        """Propagate to next frame autonomously using ByteTrack matching"""
        try:
            # Run inference on target frame
            detections = self._run_inference_on_frame(target_frame_idx)
            
            if not detections:
                return []
            
            # Match to previous frame using tracker
            if self.current_tracker is None:
                # Initialize tracker if needed
                self.current_tracker = InstanceTracker()
            
            # Match detections to existing tracks
            matched_detections = self.current_tracker.match_detections_to_tracks(
                detections,
                target_frame_idx
            )
            
            # Convert back to Detection objects with assigned IDs
            result_detections = []
            for detection, track_id in matched_detections:
                # Store the track ID as instance_id in the detection
                detection.instance_id = track_id
                result_detections.append(detection)
            
            # Store for next iteration
            self.previous_frame_detections = result_detections
            
            return result_detections
            
        except Exception as e:
            print(f"Error propagating to frame {target_frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two masks"""
        if mask1 is None or mask2 is None:
            return 0.0
        
        # Ensure both masks are 2D
        if len(mask1.shape) > 2:
            mask1 = mask1.squeeze()
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()
        
        # Check if shapes match
        if mask1.shape != mask2.shape:
            print(f"    WARNING: Mask shape mismatch! mask1: {mask1.shape}, mask2: {mask2.shape}")
            return 0.0
        
        # Compute intersection and union
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    def _evaluate_predictions(self, all_predictions: Dict, all_ground_truth: Dict,
                             id_mapping: Dict, video_id: str, video_folder: Path) -> Tuple[Dict, Dict]:
        """
        Evaluate predictions against ground truth
        
        Returns:
            (video_metrics, video_stats)
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        id_switches = 0
        
        frame_metrics = []
        
        for frame_idx in sorted(all_predictions.keys()):
            predictions = all_predictions.get(frame_idx, [])
            ground_truth = all_ground_truth.get(frame_idx, [])
            
            # Evaluate this frame
            frame_tp, frame_fp, frame_fn, frame_switches = self._evaluate_frame(
                predictions, ground_truth, id_mapping
            )
            
            total_tp += frame_tp
            total_fp += frame_fp
            total_fn += frame_fn
            id_switches += frame_switches
            
            # Calculate frame-level metrics
            precision = frame_tp / (frame_tp + frame_fp) if (frame_tp + frame_fp) > 0 else 0
            recall = frame_tp / (frame_tp + frame_fn) if (frame_tp + frame_fn) > 0 else 0
            
            frame_metrics.append({
                'frame_idx': frame_idx,
                'tp': frame_tp,
                'fp': frame_fp,
                'fn': frame_fn,
                'precision': precision,
                'recall': recall
            })
        
        # Calculate video-level metrics
        total_detections = total_tp + total_fp
        total_gt = total_tp + total_fn
        
        precision = total_tp / total_detections if total_detections > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        
        # MOTA = 1 - (FN + FP + ID_switches) / total_GT
        mota = 1 - (total_fn + total_fp + id_switches) / total_gt if total_gt > 0 else 0
        
        # IDF1 (simplified - proper calculation would track ID matches over time)
        idf1 = 2 * total_tp / (total_gt + total_detections) if (total_gt + total_detections) > 0 else 0
        
        video_metrics = {
            'mota': mota,
            'idf1': idf1,
            'precision': precision,
            'recall': recall,
            'id_switches': id_switches,
            'total_frames': len(all_predictions)
        }
        
        video_stats = {
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_id_switches': id_switches,
            'total_gt_instances': total_gt,
            'total_frames': len(all_predictions)
        }
        
        # Save per-frame metrics
        metrics_file = video_folder / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'video_metrics': video_metrics,
                'frame_metrics': frame_metrics
            }, f, indent=2)
        
        return video_metrics, video_stats
    
    def _evaluate_frame(self, predictions: List, ground_truth: List, id_mapping: Dict) -> Tuple[int, int, int, int]:
        """
        Evaluate a single frame
        
        Returns:
            (tp, fp, fn, id_switches)
        """
        tp = 0
        fp = 0
        fn = 0
        id_switches = 0
        
        # Track which GT instances have been matched
        matched_gt_indices = set()
        
        # For each prediction, check if it matches the expected GT instance
        for pred in predictions:
            # Get prediction ID and mask
            if hasattr(pred, 'instance_id'):
                pred_id = pred.instance_id
                pred_mask = pred.mask
            else:
                pred_id = pred.get('mask_id', pred.get('instance_id', 0))
                pred_mask = pred['mask']
            
            expected_gt_id = id_mapping.get(pred_id)
            
            if expected_gt_id is None:
                # New track not in initial mapping - count as FP
                fp += 1
                continue
            
            # Find the GT instance with this ID
            gt_idx = None
            for idx, gt in enumerate(ground_truth):
                if hasattr(gt, 'instance_id'):
                    gt_id = gt.instance_id
                else:
                    gt_id = gt.get('mask_id', gt.get('instance_id', 0))
                
                if gt_id == expected_gt_id:
                    gt_idx = idx
                    break
            
            if gt_idx is None:
                # Expected GT instance no longer exists (left scene) - not FP or FN
                continue
            
            # Check IoU with expected GT instance
            gt_mask = ground_truth[gt_idx].mask if hasattr(ground_truth[gt_idx], 'mask') else ground_truth[gt_idx]['mask']
            iou = self._compute_mask_iou(pred_mask, gt_mask)
            
            if iou >= self.config['iou_threshold']:
                tp += 1
                matched_gt_indices.add(gt_idx)
            else:
                # Check if it matches a different GT instance (ID switch)
                best_match_iou = 0
                best_match_idx = None
                for idx, gt in enumerate(ground_truth):
                    if idx not in matched_gt_indices:
                        gt_mask_alt = gt.mask if hasattr(gt, 'mask') else gt['mask']
                        iou_alt = self._compute_mask_iou(pred_mask, gt_mask_alt)
                        if iou_alt > best_match_iou:
                            best_match_iou = iou_alt
                            best_match_idx = idx
                
                if best_match_iou >= self.config['id_switch_iou']:
                    # ID switch detected
                    id_switches += 1
                    matched_gt_indices.add(best_match_idx)
                else:
                    # Poor tracking - count as FP
                    fp += 1
        
        # Count unmatched GT instances as FN
        fn = len(ground_truth) - len(matched_gt_indices)
        
        return tp, fp, fn, id_switches
    
    def _calculate_overall_metrics(self, overall_stats: Dict) -> Dict:
        """Calculate overall metrics from accumulated stats"""
        total_tp = overall_stats['total_tp']
        total_fp = overall_stats['total_fp']
        total_fn = overall_stats['total_fn']
        id_switches = overall_stats['total_id_switches']
        total_gt = overall_stats['total_gt_instances']
        total_detections = total_tp + total_fp
        
        precision = total_tp / total_detections if total_detections > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        mota = 1 - (total_fn + total_fp + id_switches) / total_gt if total_gt > 0 else 0
        idf1 = 2 * total_tp / (total_gt + total_detections) if (total_gt + total_detections) > 0 else 0
        
        return {
            'mota': mota,
            'idf1': idf1,
            'precision': precision,
            'recall': recall,
            'id_switches': id_switches,
            'total_frames': overall_stats['total_frames'],
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }
    
    def _generate_comparison_visualization(
        self, 
        frame_idx: int,
        predictions: List,
        ground_truth: List,
        id_mapping: Dict,
        output_path: Path,
        title: str
    ):
        """Generate side-by-side comparison of ground truth and predictions"""
        try:
            import cv2
            import matplotlib.pyplot as plt
            from matplotlib import patches, colors as mcolors
            
            # Load the frame image (frame_idx is video-local)
            frame_path = self.current_video_frames[frame_idx]
            frame_img = cv2.imread(str(frame_path))
            if frame_img is None:
                print(f"Could not load frame: {frame_path}")
                return
            
            # Convert BGR to RGB for matplotlib
            frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            
            # Determine which GT instances were matched
            matched_gt_ids = set(id_mapping.values())
            
            # Count matched vs unmatched
            gt_ids = []
            for gt in ground_truth:
                gt_id = gt.instance_id if hasattr(gt, 'instance_id') else gt.get('mask_id', gt.get('instance_id', 0))
                gt_ids.append(gt_id)
            
            matched_count = sum(1 for gt_id in gt_ids if gt_id in matched_gt_ids)
            unmatched_count = len(gt_ids) - matched_count
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # === LEFT: Ground Truth ===
            gt_img = frame_img_rgb.copy().astype(np.float32)
            
            # Generate distinct colors for each GT instance
            colormap = plt.cm.get_cmap('tab20')  # Use tab20 for distinct colors
            
            for i, gt in enumerate(ground_truth):
                mask = gt.mask if hasattr(gt, 'mask') else gt['mask']
                gt_id = gt.instance_id if hasattr(gt, 'instance_id') else gt.get('mask_id', gt.get('instance_id', 0))
                
                # Check if this GT was matched
                is_matched = gt_id in matched_gt_ids
                
                if is_matched:
                    # Use distinct color for matched instances
                    color_idx = gt_id % 20
                    color = np.array(colormap(color_idx)[:3]) * 255  # RGB, 0-255 range
                    contour_color = color
                else:
                    # Use orange/yellow for unmatched instances (False Negatives)
                    color = np.array([255, 165, 0])  # Orange
                    contour_color = np.array([255, 100, 0])  # Darker orange
                
                # Blend the color onto the image
                mask_bool = mask > 0
                gt_img[mask_bool] = gt_img[mask_bool] * 0.4 + color * 0.6  # 60% color, 40% original
                
                # Draw contour around mask - thicker for unmatched
                contours, _ = cv2.findContours(
                    (mask > 0).astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                thickness = 5 if not is_matched else 3
                for contour in contours:
                    cv2.drawContours(gt_img, [contour], -1, contour_color.tolist(), thickness)
            
            ax1.imshow(gt_img.astype(np.uint8))
            gt_title = f'Ground Truth ({len(ground_truth)} instances)\n'
            gt_title += f'Matched: {matched_count} | Unmatched (FN): {unmatched_count}'
            ax1.set_title(gt_title, fontsize=16, fontweight='bold')
            ax1.axis('off')
            
            # Add labels with backgrounds
            for gt in ground_truth:
                mask = gt.mask if hasattr(gt, 'mask') else gt['mask']
                gt_id = gt.instance_id if hasattr(gt, 'instance_id') else gt.get('mask_id', gt.get('instance_id', 0))
                is_matched = gt_id in matched_gt_ids
                
                y_coords, x_coords = np.where(mask > 0)
                if len(y_coords) > 0:
                    center_y, center_x = y_coords.mean(), x_coords.mean()
                    
                    # Use different background for unmatched
                    bg_color = 'black' if is_matched else 'orange'
                    
                    ax1.text(center_x, center_y, str(gt_id), 
                            color='white', fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='circle', facecolor=bg_color, alpha=0.3, edgecolor='white', linewidth=2),
                            ha='center', va='center')
            
            # === RIGHT: Predictions ===
            pred_img = frame_img_rgb.copy().astype(np.float32)
            
            # Count correct vs incorrect predictions
            correct_count = 0
            fp_count = 0
            
            for i, pred in enumerate(predictions):
                mask = pred.mask if hasattr(pred, 'mask') else pred['mask']
                pred_id = pred.instance_id if hasattr(pred, 'instance_id') else pred.get('mask_id', pred.get('instance_id', 0))
                
                # Determine if this is correctly matched
                expected_gt_id = id_mapping.get(pred_id)
                is_correct = False
                if expected_gt_id is not None:
                    # Check if GT with this ID exists and matches
                    for gt in ground_truth:
                        gt_id = gt.instance_id if hasattr(gt, 'instance_id') else gt.get('mask_id', gt.get('instance_id', 0))
                        if gt_id == expected_gt_id:
                            gt_mask = gt.mask if hasattr(gt, 'mask') else gt['mask']
                            iou = self._compute_mask_iou(mask, gt_mask)
                            if iou >= self.config['iou_threshold']:
                                is_correct = True
                                correct_count += 1
                            break
                
                if not is_correct:
                    fp_count += 1
                
                # Color based on correctness
                if is_correct:
                    # Use same color as GT for correct matches
                    color_idx = expected_gt_id % 20
                    color = np.array(colormap(color_idx)[:3]) * 255  # Green tint for correct
                    contour_color = np.array([0, 255, 0])  # Green contour
                else:
                    # Red for incorrect/FP
                    color = np.array([255, 100, 100])  # Red
                    contour_color = np.array([255, 0, 0])  # Red contour
                
                # Blend the color onto the image
                mask_bool = mask > 0
                pred_img[mask_bool] = pred_img[mask_bool] * 0.4 + color * 0.6
                
                # Draw contour - thicker for incorrect
                contours, _ = cv2.findContours(
                    (mask > 0).astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                thickness = 5 if not is_correct else 3
                for contour in contours:
                    cv2.drawContours(pred_img, [contour], -1, contour_color.tolist(), thickness)
            
            ax2.imshow(pred_img.astype(np.uint8))
            pred_title = f'Predictions ({len(predictions)} instances)\n'
            pred_title += f'Correct: {correct_count} | False Positives: {fp_count}'
            ax2.set_title(pred_title, fontsize=16, fontweight='bold')
            ax2.axis('off')
            
            # Add labels with mapping info - only show GT ID or FP
            for pred in predictions:
                mask = pred.mask if hasattr(pred, 'mask') else pred['mask']
                pred_id = pred.instance_id if hasattr(pred, 'instance_id') else pred.get('mask_id', pred.get('instance_id', 0))
                expected_gt_id = id_mapping.get(pred_id)
                
                # Check correctness
                is_correct = False
                if expected_gt_id is not None:
                    for gt in ground_truth:
                        gt_id = gt.instance_id if hasattr(gt, 'instance_id') else gt.get('mask_id', gt.get('instance_id', 0))
                        if gt_id == expected_gt_id:
                            gt_mask = gt.mask if hasattr(gt, 'mask') else gt['mask']
                            iou = self._compute_mask_iou(mask, gt_mask)
                            if iou >= self.config['iou_threshold']:
                                is_correct = True
                            break
                
                y_coords, x_coords = np.where(mask > 0)
                if len(y_coords) > 0:
                    center_y, center_x = y_coords.mean(), x_coords.mean()
                    
                    # Format label - just show GT ID it matched to, or FP
                    if is_correct and expected_gt_id is not None:
                        label = str(expected_gt_id)
                        bg_color = 'green'
                    else:
                        label = "FP"
                        bg_color = 'red'
                    
                    ax2.text(center_x, center_y, label,
                            color='white', fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', 
                                    facecolor=bg_color, 
                                    alpha=0.4,
                                    edgecolor='white',
                                    linewidth=2),
                            ha='center', va='center')
            
            # Overall title
            fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
            
            # Add legend with better styling
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', edgecolor='white', linewidth=2, label='True Positive (Correct Match)'),
                Patch(facecolor='red', edgecolor='white', linewidth=2, label='False Positive'),
                Patch(facecolor='orange', edgecolor='white', linewidth=2, label='False Negative (GT Unmatched)')
            ]
            ax2.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.9)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Saved visualization: {output_path.name}")
            
        except Exception as e:
            print(f"  Error generating visualization: {e}")
            import traceback
            traceback.print_exc()
