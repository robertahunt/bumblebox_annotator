"""
Background worker for frame-level validation analysis
"""

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import csv
import cv2
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO

from utils.validation_metrics import (
    compute_bbox_iou, compute_mask_iou, distance_to_mask,
    distance_between_masks, point_in_chamber, match_by_hungarian,
    bbox_from_mask, mask_centroid, mask_to_polygons_string,
    extract_instance_masks, closest_points_between_masks
)


class FrameLevelValidationWorker(QThread):
    """Worker thread for frame-level validation analysis"""
    
    # Signals
    status_updated = pyqtSignal(str)  # Status message
    stats_updated = pyqtSignal(int, int, int, int)  # videos_done, total_videos, frames_done, total_frames
    match_stats_updated = pyqtSignal(int, int, int)  # matches, fps, fns
    time_remaining_updated = pyqtSignal(str)  # time_remaining as formatted string
    log_message = pyqtSignal(str)  # Log message
    analysis_complete = pyqtSignal(str)  # results_path
    analysis_failed = pyqtSignal(str)  # error_msg
    
    def __init__(self, main_window, config):
        """
        Initialize worker
        
        Args:
            main_window: Reference to main window
            config: Configuration dict from dialog
        """
        super().__init__()
        self.main_window = main_window
        self.config = config
        self.should_stop = False
        
        # Statistics
        self.total_matches = 0
        self.total_fps = 0
        self.total_fns = 0
        self.map_eval_frames = []
        
        # Timing
        self.start_time = None

    def _polygon_epsilon(self) -> float:
        """Return contour simplification epsilon percentage for CSV polygons."""
        return 0.01 if self.config.get('high_resolution_polygons', False) else 2.0
        
    def stop(self):
        """Request worker to stop"""
        self.should_stop = True
    
    def run(self):
        """Run validation analysis in background thread"""
        try:
            bee_model_type = self.config.get('bee_model_type', 'bbox')
            
            self.log_message.emit("=== Frame-Level Validation Analysis ===")
            self.log_message.emit(f"Bee Model: {Path(self.config['bbox_model_path']).name}")
            self.log_message.emit(f"Bee Model Type: {bee_model_type}")
            if bee_model_type == 'bbox_sam2' and self.config.get('sam2_checkpoint_path'):
                self.log_message.emit(f"SAM2 Checkpoint: {Path(self.config['sam2_checkpoint_path']).name}")
            if bee_model_type == 'bbox_instance_focused' and self.config.get('instance_focused_model_path'):
                self.log_message.emit(f"Instance-Focused Model: {Path(self.config['instance_focused_model_path']).name}")
                self.log_message.emit(f"Instance crop margin: {self.config.get('instance_crop_margin', 50)} px")
            if self.config['hive_model_path']:
                self.log_message.emit(f"Hive Model: {Path(self.config['hive_model_path']).name}")
            if self.config.get('pollen_model_path'):
                self.log_message.emit(f"Pollen Model: {Path(self.config['pollen_model_path']).name}")
            if self.config['chamber_model_path']:
                self.log_message.emit(f"Chamber Model: {Path(self.config['chamber_model_path']).name}")
            self.log_message.emit(f"Confidence threshold: {self.config['conf_threshold']:.2f}")
            self.log_message.emit(f"IoU threshold: {self.config['iou_threshold']:.2f}")
            
            # Debug mode indicator
            if self.config.get('debug_mode', False):
                self.log_message.emit("⚠️  DEBUG MODE: Only processing first frame")
            
            self.log_message.emit("")
            
            # Load models
            self.status_updated.emit("Loading models...")
            bbox_model = YOLO(self.config['bbox_model_path'])
            second_stage_model = None
            if bee_model_type == 'bbox_sam2':
                from core.sam2_integrator import SAM2Integrator
                second_stage_model = SAM2Integrator(self.config['sam2_checkpoint_path'])
            elif bee_model_type == 'bbox_instance_focused':
                second_stage_model = YOLO(self.config['instance_focused_model_path'])
            hive_model = YOLO(self.config['hive_model_path']) if self.config['hive_model_path'] else None
            pollen_model = YOLO(self.config['pollen_model_path']) if self.config.get('pollen_model_path') else None
            chamber_model = YOLO(self.config['chamber_model_path']) if self.config['chamber_model_path'] else None
            
            # Create results folder in frame_level_validation subfolder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_folder = self.main_window.project_path / "frame_level_validation" / timestamp
            results_folder.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_file = results_folder / "config.json"
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Find all validation frames
            self.status_updated.emit("Finding validation frames...")
            validation_frames = self._find_validation_frames()
            
            if not validation_frames:
                self.analysis_failed.emit("No validation frames with bee annotations found in project.")
                return
            
            total_frames = sum(len(frames) for frames in validation_frames.values())
            self.log_message.emit(f"Found {len(validation_frames)} videos with {total_frames} validation frames")
            self.log_message.emit("")
            
            # Prepare CSV files
            bee_csv_path = results_folder / "bee_detections.csv"
            hive_csv_path = results_folder / "hive_detections.csv"
            pollen_csv_path = results_folder / "pollen_detections.csv"
            chamber_csv_path = results_folder / "chamber_detections.csv"
            summary_csv_path = results_folder / "frame_summary.csv"
            
            bee_csv = open(bee_csv_path, 'w', newline='')
            bee_writer = csv.writer(bee_csv)
            bee_writer.writerow([
                'video_id', 'frame_idx', 'match_status',
                'pred_center_x', 'pred_center_y', 'pred_width', 'pred_height', 'pred_confidence',
                'gt_center_x', 'gt_center_y', 'gt_width', 'gt_height', 'gt_instance_id',
                'iou', 'pred_chamber_id', 'gt_chamber_id', 
                'pred_hive_distance', 'gt_hive_distance',
                'pred_pollen_distance', 'gt_pollen_distance',
                'pred_nearest_bee_distance', 'gt_nearest_bee_distance',
                'pred_avg_chamber_bee_distance', 'gt_avg_chamber_bee_distance',
                'pred_polygon', 'gt_polygon'
            ])
            
            summary_csv = open(summary_csv_path, 'w', newline='')
            summary_writer = csv.writer(summary_csv)
            summary_writer.writerow([
                'video_id', 'frame_idx', 'chamber_id',
                'bumblebox_number', 'datetime',
                'pred_bee_count', 'gt_bee_count',
                'pred_hive_pixels', 'gt_hive_pixels',
                'pred_pollen_count', 'gt_pollen_count',
                'pred_pollen_pixels', 'gt_pollen_pixels',
                'pred_avg_hive_distance', 'gt_avg_hive_distance',
                'pred_avg_pollen_distance', 'gt_avg_pollen_distance',
                'pred_avg_nearest_bee_distance', 'gt_avg_nearest_bee_distance',
                'pred_avg_chamber_bee_distance', 'gt_avg_chamber_bee_distance'
            ])
            
            hive_csv = None
            hive_writer = None
            if hive_model:
                hive_csv = open(hive_csv_path, 'w', newline='')
                hive_writer = csv.writer(hive_csv)
                hive_writer.writerow([
                    'video_id', 'frame_idx', 'chamber_id',
                    'match_status',
                    'pred_hive_id', 'gt_hive_id',
                    'pred_hive_pixels', 'gt_hive_pixels', 'hive_iou',
                    'pred_polygon', 'gt_polygon'
                ])

            pollen_csv = None
            pollen_writer = None
            if pollen_model:
                pollen_csv = open(pollen_csv_path, 'w', newline='')
                pollen_writer = csv.writer(pollen_csv)
                pollen_writer.writerow([
                    'video_id', 'frame_idx', 'chamber_id',
                    'match_status',
                    'pred_pollen_id', 'gt_pollen_id',
                    'pred_pollen_pixels', 'gt_pollen_pixels',
                    'pollen_iou',
                    'pred_polygon', 'gt_polygon'
                ])
            
            chamber_csv = None
            chamber_writer = None
            if chamber_model:
                chamber_csv = open(chamber_csv_path, 'w', newline='')
                chamber_writer = csv.writer(chamber_csv)
                chamber_writer.writerow([
                    'video_id', 'frame_idx', 'match_status',
                    'pred_chamber_id', 'gt_chamber_id',
                    'pred_area_pixels', 'gt_area_pixels', 'chamber_iou',
                    'pred_polygon', 'gt_polygon'
                ])
            
            # Process each video
            frames_processed = 0
            self.start_time = datetime.now()  # Start timing
            
            for video_num, (video_id, frame_indices) in enumerate(validation_frames.items(), 1):
                if self.should_stop:
                    break
                
                self.log_message.emit(f"=== Video {video_num}/{len(validation_frames)}: {video_id} ({len(frame_indices)} frames) ===")
                
                # Load video-level annotations (hive and chamber)
                video_annotations, _ = self.main_window.annotation_manager.load_video_annotations(
                    self.main_window.project_path, video_id
                )
                
                # Load COCO data for this video
                coco_data = self._load_coco_data_for_video(video_id)
                if not coco_data:
                    self.log_message.emit(f"  Warning: No COCO data found for {video_id}, skipping")
                    continue
                
                if video_annotations:
                    hive_count = sum(1 for a in video_annotations if a.get('category') == 'hive')
                    chamber_count = sum(1 for a in video_annotations if a.get('category') == 'chamber')
                    self.log_message.emit(f"  Found video-level annotations: {hive_count} hive, {chamber_count} chambers")
                else:
                    self.log_message.emit(f"  No video-level annotations found for {video_id}")
                
                # Process each frame
                for frame_idx in frame_indices:
                    if self.should_stop:
                        break
                    
                    self.status_updated.emit(f"Processing {video_id} frame {frame_idx}...")
                    
                    # Process this frame
                    self._process_frame(
                        video_id, frame_idx,
                        bbox_model, second_stage_model, hive_model, pollen_model, chamber_model,
                        bee_writer, hive_writer, pollen_writer, chamber_writer,
                        summary_writer,
                        results_folder,
                        video_annotations,
                        coco_data
                    )
                    
                    frames_processed += 1
                    self.stats_updated.emit(video_num, len(validation_frames), frames_processed, total_frames)
                    self.match_stats_updated.emit(self.total_matches, self.total_fps, self.total_fns)
                    
                    # Calculate and emit time remaining
                    if frames_processed > 0 and self.start_time is not None:
                        elapsed = (datetime.now() - self.start_time).total_seconds()
                        avg_time_per_frame = elapsed / frames_processed
                        frames_remaining = total_frames - frames_processed
                        seconds_remaining = avg_time_per_frame * frames_remaining
                        
                        # Format time remaining
                        if seconds_remaining < 60:
                            time_str = f"{int(seconds_remaining)}s"
                        elif seconds_remaining < 3600:
                            minutes = int(seconds_remaining / 60)
                            seconds = int(seconds_remaining % 60)
                            time_str = f"{minutes}m {seconds}s"
                        else:
                            hours = int(seconds_remaining / 3600)
                            minutes = int((seconds_remaining % 3600) / 60)
                            time_str = f"{hours}h {minutes}m"
                        
                        self.time_remaining_updated.emit(time_str)
                    
                    # Exit early if debug mode
                    if self.config.get('debug_mode', False):
                        self.log_message.emit("Debug mode: Stopping after first frame")
                        break
                
                self.log_message.emit(f"Completed {video_id}")
                
                # Exit early if debug mode (after first video with frames)
                if self.config.get('debug_mode', False):
                    break
            
            # Close CSV files
            bee_csv.close()
            summary_csv.close()
            if hive_csv:
                hive_csv.close()
            if pollen_csv:
                pollen_csv.close()
            if chamber_csv:
                chamber_csv.close()
            
            if self.should_stop:
                self.log_message.emit("Analysis cancelled by user")
                return
            
            # Generate summary
            self._generate_summary(results_folder, len(validation_frames), total_frames)

            # Generate residual-uncertainty plot
            self.status_updated.emit("Generating residual-uncertainty plot …")
            try:
                from utils.plot_validation import generate_residual_plot
                generate_residual_plot(
                    bee_csv=results_folder / "bee_detections.csv",
                    summary_csv=results_folder / "frame_summary.csv",
                    output_dir=results_folder,
                )
                self.log_message.emit("  Residual-uncertainty plot saved.")
            except Exception as plot_err:
                self.log_message.emit(f"  Warning: plot generation skipped ({plot_err})")

            self.analysis_complete.emit(str(results_folder))
            
        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.analysis_failed.emit(str(e))
    
    def _find_validation_frames(self) -> Dict[str, List[int]]:
        """Find all validation frames with bee annotations from COCO exports"""
        validation_frames = {}
        
        # Get all videos in 'val' split
        val_videos = self.main_window.project_manager.get_videos_by_split('val')
        
        # Check COCO export directory
        coco_dir = self.main_window.project_path / 'annotations' / 'coco' / 'val'
        
        if not coco_dir.exists():
            self.log_message.emit("Warning: COCO validation annotations not found. Please export COCO format first.")
            return {}
        
        for video_id in val_videos:
            # Load COCO JSON for this video
            coco_file = coco_dir / f'{video_id}.json'
            
            if not coco_file.exists():
                continue
            
            try:
                with open(coco_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Build image_id to frame_idx mapping
                image_to_frame = {}
                for img in coco_data['images']:
                    # Extract frame index from file_name (e.g., "frames/video_name/frame_000123.jpg")
                    file_name = img['file_name']
                    frame_str = Path(file_name).stem  # e.g., "frame_000123"
                    frame_idx = int(frame_str.split('_')[1])  # Extract 123
                    image_to_frame[img['id']] = frame_idx
                
                # Find frames with bee annotations (category_id=1)
                frames_with_bees = set()
                for ann in coco_data['annotations']:
                    if ann['category_id'] == 1:  # bee category
                        image_id = ann['image_id']
                        if image_id in image_to_frame:
                            frames_with_bees.add(image_to_frame[image_id])
                
                if frames_with_bees:
                    validation_frames[video_id] = sorted(list(frames_with_bees))
                    
            except Exception as e:
                self.log_message.emit(f"Warning: Failed to load COCO data for {video_id}: {str(e)}")
                continue
        
        return validation_frames
    
    def _load_coco_data_for_video(self, video_id: str) -> Optional[Dict]:
        """Load COCO JSON data for a specific video"""
        coco_dir = self.main_window.project_path / 'annotations' / 'coco' / 'val'
        coco_file = coco_dir / f'{video_id}.json'
        
        if not coco_file.exists():
            return None
        
        try:
            with open(coco_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log_message.emit(f"Error loading COCO data for {video_id}: {str(e)}")
            return None
    
    def _process_frame(self, video_id: str, frame_idx: int,
                      bbox_model, second_stage_model, hive_model, pollen_model, chamber_model,
                      bee_writer, hive_writer, pollen_writer, chamber_writer,
                      summary_writer,
                      results_folder: Path,
                      video_annotations: List[Dict],
                      coco_data: Dict):
        """Process a single validation frame"""
        try:
            # Load frame image
            frame_path = self.main_window.project_manager.get_frame_path(video_id, frame_idx)
            
            if not frame_path.exists():
                self.log_message.emit(f"  Warning: Frame image not found: {frame_path}")
                return
            
            frame = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run bee bbox predictions
            pred_bees = self._predict_bees(frame_rgb, bbox_model, second_stage_model)
            gt_bees = self._extract_gt_bees_from_coco(coco_data, frame_idx, frame.shape[:2])
            self._record_map_eval_frame(pred_bees, gt_bees)
            
            # Keep hive instances for CSV output, and combined masks for existing distance metrics.
            gt_hive_instances = self._extract_gt_hive_instances(video_annotations, frame.shape[:2])
            gt_hive_mask = self._combine_instance_masks(gt_hive_instances, frame.shape[:2])
            pred_hive_instances = self._predict_hive_instances(frame_rgb, hive_model) if hive_model else []
            pred_hive_mask = self._combine_instance_masks(pred_hive_instances, frame.shape[:2]) if hive_model else None

            # Extract GT pollen balls from frame-level COCO annotations and run pollen prediction if available
            gt_pollen_balls = self._extract_gt_pollen_from_coco(coco_data, frame_idx, frame.shape[:2])
            pred_pollen_balls = self._predict_pollen_balls(frame_rgb, pollen_model) if pollen_model else []
            
            # Always extract GT chamber masks for chamber ID (even if no chamber model)
            gt_chamber_masks = self._extract_gt_chambers(video_annotations, frame.shape[:2])
            # Run chamber prediction if model available
            pred_chamber_masks = self._predict_chambers(frame_rgb, chamber_model) if chamber_model else None
            
            # Match bees and get matching info for visualization
            matched_pairs, unmatched_preds, unmatched_gts, pred_nearest_connections, gt_nearest_connections = self._match_and_write_bees(
                video_id, frame_idx, pred_bees, gt_bees,
                pred_hive_mask, gt_hive_mask,
                pred_pollen_balls, gt_pollen_balls,
                pred_chamber_masks, gt_chamber_masks,
                bee_writer
            )
            
            # Save visualization if enabled
            if self.config.get('save_visualizations', False):
                self._save_visualization(
                    frame, video_id, frame_idx,
                    pred_bees, gt_bees,
                    matched_pairs, unmatched_preds, unmatched_gts,
                    results_folder,
                    pred_hive_mask, gt_hive_mask,
                    pred_pollen_balls, gt_pollen_balls,
                    pred_chamber_masks, gt_chamber_masks,
                    pred_nearest_connections, gt_nearest_connections
                )
            
            # Write hive results
            if hive_writer and pred_hive_mask is not None:
                self._write_hive_results(
                    video_id, frame_idx,
                    pred_hive_instances, gt_hive_instances,
                    hive_writer, pred_chamber_masks, gt_chamber_masks
                )

            # Write pollen results
            if pollen_writer is not None:
                self._write_pollen_results(
                    video_id, frame_idx,
                    pred_pollen_balls, gt_pollen_balls,
                    pollen_writer, pred_chamber_masks, gt_chamber_masks
                )
            
            # Write chamber results
            if chamber_writer and pred_chamber_masks is not None:
                self._write_chamber_results(video_id, frame_idx, pred_chamber_masks, gt_chamber_masks, chamber_writer)
            
            # Write frame summary
            self._write_frame_summary(
                video_id, frame_idx,
                pred_bees, gt_bees,
                pred_hive_mask, gt_hive_mask,
                pred_pollen_balls, gt_pollen_balls,
                summary_writer,
                pred_chamber_masks, gt_chamber_masks
            )
            
        except Exception as e:
            import traceback
            self.log_message.emit(f"  Error processing {video_id} frame {frame_idx}: {str(e)}")
            traceback.print_exc()
    
    def _bee_predictions_are_masks(self) -> bool:
        """Return True for bee detection modes that produce segmentation masks."""
        return self.config.get('bee_model_type', 'bbox') in {
            'segmentation',
            'bbox_sam2',
            'bbox_instance_focused',
        }

    def _record_map_eval_frame(self, pred_bees: List[Dict], gt_bees: List[Dict]) -> None:
        """Store frame-level detections for COCO-style AP calculation."""
        use_masks = self._bee_predictions_are_masks()
        scores = [float(pred.get('confidence', 0.0)) for pred in pred_bees]
        bbox_iou_matrix = np.zeros((len(pred_bees), len(gt_bees)), dtype=float)
        mask_iou_matrix = np.zeros((len(pred_bees), len(gt_bees)), dtype=float)
        missing_pred_masks = 0
        missing_gt_masks = 0
        mask_eval_pred_indices = list(range(len(pred_bees)))
        mask_eval_gt_indices = list(range(len(gt_bees)))
        ignored_bbox_only_predictions = 0

        if use_masks:
            missing_pred_masks = sum(1 for pred in pred_bees if pred.get('mask') is None)
            missing_gt_masks = sum(1 for gt in gt_bees if gt.get('mask') is None)
            mask_eval_gt_indices = [
                gt_idx for gt_idx, gt in enumerate(gt_bees)
                if gt.get('mask') is not None
            ]

        for pred_idx, pred in enumerate(pred_bees):
            for gt_idx, gt in enumerate(gt_bees):
                bbox_iou_matrix[pred_idx, gt_idx] = compute_bbox_iou(pred['bbox'], gt['bbox'])
                if use_masks:
                    pred_mask = pred.get('mask')
                    gt_mask = gt.get('mask')
                    if pred_mask is not None and gt_mask is not None:
                        mask_iou_matrix[pred_idx, gt_idx] = compute_mask_iou(pred_mask, gt_mask)
                    else:
                        mask_iou_matrix[pred_idx, gt_idx] = 0.0

        if use_masks:
            mask_gt_set = set(mask_eval_gt_indices)
            bbox_only_gt_indices = [
                gt_idx for gt_idx in range(len(gt_bees))
                if gt_idx not in mask_gt_set
            ]
            mask_eval_pred_indices = []
            for pred_idx, pred in enumerate(pred_bees):
                if pred.get('mask') is None:
                    continue

                best_mask_bbox_iou = (
                    float(np.max(bbox_iou_matrix[pred_idx, mask_eval_gt_indices]))
                    if mask_eval_gt_indices else 0.0
                )
                best_bbox_only_iou = (
                    float(np.max(bbox_iou_matrix[pred_idx, bbox_only_gt_indices]))
                    if bbox_only_gt_indices else 0.0
                )

                # If a prediction clearly belongs to a bbox-only GT bee, leave it out of
                # mask AP. There is no segmentation target for that bee to evaluate.
                if best_bbox_only_iou >= 0.5 and best_bbox_only_iou >= best_mask_bbox_iou:
                    ignored_bbox_only_predictions += 1
                    continue

                mask_eval_pred_indices.append(pred_idx)

        if use_masks:
            eval_iou_matrix = mask_iou_matrix[np.ix_(mask_eval_pred_indices, mask_eval_gt_indices)]
            eval_scores = [scores[pred_idx] for pred_idx in mask_eval_pred_indices]
            eval_num_gts = len(mask_eval_gt_indices)
            eval_num_preds = len(mask_eval_pred_indices)
        else:
            eval_iou_matrix = bbox_iou_matrix
            eval_scores = scores
            eval_num_gts = len(gt_bees)
            eval_num_preds = len(pred_bees)

        self.map_eval_frames.append({
            'scores': eval_scores,
            'iou_matrix': eval_iou_matrix,
            'bbox_iou_matrix': bbox_iou_matrix,
            'mask_iou_matrix': mask_iou_matrix,
            'bbox_scores': scores,
            'num_gts': eval_num_gts,
            'num_preds': eval_num_preds,
            'bbox_num_gts': len(gt_bees),
            'bbox_num_preds': len(pred_bees),
            'missing_pred_masks': missing_pred_masks,
            'missing_gt_masks': missing_gt_masks,
            'mask_evaluable_gts': len(mask_eval_gt_indices) if use_masks else 0,
            'mask_ignored_bbox_only_preds': ignored_bbox_only_predictions,
        })

    def _compute_average_precision(self, iou_threshold: float, iou_matrix_key: str = 'iou_matrix',
                                   scores_key: str = 'scores',
                                   num_gts_key: str = 'num_gts') -> float:
        """Compute one-class AP for the stored validation detections."""
        total_gts = sum(frame[num_gts_key] for frame in self.map_eval_frames)
        if total_gts == 0:
            return 0.0

        detections = []
        for frame_idx, frame in enumerate(self.map_eval_frames):
            for pred_idx, score in enumerate(frame[scores_key]):
                detections.append((score, frame_idx, pred_idx))

        if not detections:
            return 0.0

        detections.sort(key=lambda item: item[0], reverse=True)
        matched_gts = {
            frame_idx: set()
            for frame_idx, frame in enumerate(self.map_eval_frames)
            if frame[num_gts_key] > 0
        }

        tp = np.zeros(len(detections), dtype=float)
        fp = np.zeros(len(detections), dtype=float)

        for det_idx, (_, frame_idx, pred_idx) in enumerate(detections):
            frame = self.map_eval_frames[frame_idx]
            num_gts = frame[num_gts_key]
            if num_gts == 0:
                fp[det_idx] = 1.0
                continue

            ious = frame[iou_matrix_key][pred_idx]
            best_gt_idx = -1
            best_iou = -1.0
            for gt_idx, iou in enumerate(ious):
                if gt_idx in matched_gts[frame_idx]:
                    continue
                if iou > best_iou:
                    best_iou = float(iou)
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                tp[det_idx] = 1.0
                matched_gts[frame_idx].add(best_gt_idx)
            else:
                fp[det_idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / total_gts
        precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(float).eps)

        # 101-point interpolation, matching the COCO mAP convention.
        ap = 0.0
        for recall_threshold in np.linspace(0, 1, 101):
            precisions = precision[recall >= recall_threshold]
            ap += float(np.max(precisions)) if precisions.size else 0.0
        return ap / 101.0

    def _compute_best_iou_stats(self, iou_matrix_key: str = 'iou_matrix',
                                num_preds_key: str = 'num_preds',
                                num_gts_key: str = 'num_gts') -> Dict[str, float]:
        """Summarize best available IoU per prediction/GT for diagnostics."""
        best_pred_ious = []
        best_gt_ious = []
        total_preds = 0
        total_gts = 0

        for frame in self.map_eval_frames:
            iou_matrix = frame[iou_matrix_key]
            total_preds += frame.get(num_preds_key, iou_matrix.shape[0])
            total_gts += frame.get(num_gts_key, iou_matrix.shape[1])

            if iou_matrix.size == 0:
                continue
            if iou_matrix.shape[1] > 0:
                best_pred_ious.extend(np.max(iou_matrix, axis=1).tolist())
            if iou_matrix.shape[0] > 0:
                best_gt_ious.extend(np.max(iou_matrix, axis=0).tolist())

        def summarize(values):
            if not values:
                return {'mean': 0.0, 'median': 0.0, 'pct_ge_050': 0.0}
            arr = np.asarray(values, dtype=float)
            return {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'pct_ge_050': float(np.mean(arr >= 0.5)),
            }

        pred_stats = summarize(best_pred_ious)
        gt_stats = summarize(best_gt_ious)
        return {
            'total_preds': total_preds,
            'total_gts': total_gts,
            'pred_best_iou_mean': pred_stats['mean'],
            'pred_best_iou_median': pred_stats['median'],
            'pred_best_iou_pct_ge_050': pred_stats['pct_ge_050'],
            'gt_best_iou_mean': gt_stats['mean'],
            'gt_best_iou_median': gt_stats['median'],
            'gt_best_iou_pct_ge_050': gt_stats['pct_ge_050'],
        }

    def _compute_map_metrics(self) -> Dict[str, float]:
        """Compute mAP@0.5 and mAP@0.5:0.95 for bee detections."""
        thresholds = np.arange(0.5, 0.96, 0.05)
        ap_by_threshold = {
            round(float(threshold), 2): self._compute_average_precision(float(threshold), 'iou_matrix')
            for threshold in thresholds
        }
        metrics = {
            'mAP@0.5': ap_by_threshold[0.5],
            'mAP@0.5:0.95': float(np.mean(list(ap_by_threshold.values()))) if ap_by_threshold else 0.0,
            'AP_by_threshold': ap_by_threshold,
            'IoU_type': 'segmentation' if self._bee_predictions_are_masks() else 'bbox',
            'IoU_diagnostics': self._compute_best_iou_stats('iou_matrix'),
            'missing_pred_masks': sum(frame.get('missing_pred_masks', 0) for frame in self.map_eval_frames),
            'missing_gt_masks': sum(frame.get('missing_gt_masks', 0) for frame in self.map_eval_frames),
        }
        if self._bee_predictions_are_masks():
            bbox_ap_by_threshold = {
                round(float(threshold), 2): self._compute_average_precision(
                    float(threshold),
                    'bbox_iou_matrix',
                    scores_key='bbox_scores',
                    num_gts_key='bbox_num_gts',
                )
                for threshold in thresholds
            }
            metrics['bbox_mAP@0.5'] = bbox_ap_by_threshold[0.5]
            metrics['bbox_mAP@0.5:0.95'] = (
                float(np.mean(list(bbox_ap_by_threshold.values()))) if bbox_ap_by_threshold else 0.0
            )
            metrics['bbox_AP_by_threshold'] = bbox_ap_by_threshold
            metrics['bbox_IoU_diagnostics'] = self._compute_best_iou_stats(
                'bbox_iou_matrix',
                num_preds_key='bbox_num_preds',
                num_gts_key='bbox_num_gts',
            )
            metrics['mask_evaluable_gts'] = sum(
                frame.get('mask_evaluable_gts', 0) for frame in self.map_eval_frames
            )
            metrics['mask_ignored_bbox_only_preds'] = sum(
                frame.get('mask_ignored_bbox_only_preds', 0) for frame in self.map_eval_frames
            )
        return metrics

    def _predict_bees(self, frame: np.ndarray, model, second_stage_model=None) -> List[Dict]:
        """Run YOLO prediction for bees (bbox or segmentation based on config)"""
        bee_model_type = self.config.get('bee_model_type', 'bbox')

        if bee_model_type == 'bbox_sam2':
            return self._predict_bees_bbox_sam2(frame, model, second_stage_model)
        if bee_model_type == 'bbox_instance_focused':
            return self._predict_bees_bbox_instance_focused(frame, model, second_stage_model)
        
        results = model.predict(
            frame,
            conf=self.config['conf_threshold'],
            retina_masks=True if bee_model_type == 'segmentation' else False,
            verbose=False
        )
        
        bees = []
        
        if bee_model_type == 'segmentation':
            # Segmentation model - extract masks, centroids, polygons, and bboxes
            if len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes
                
                for i in range(len(masks)):
                    # Filter by class ID - only include bee class (class_id=0 for bees)
                    cls_id = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
                    
                    # Only include class 0 (bee) predictions, skip others
                    if cls_id != 0:
                        continue
                    
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    # Get mask (with retina_masks=True, already at original image size)
                    mask = (masks[i] > 0.5).astype(np.uint8) * 255
                    
                    # Calculate centroid
                    centroid_x, centroid_y = mask_centroid(mask)
                    
                    # Extract bounding box from mask
                    center_x, center_y, width, height = bbox_from_mask(mask)
                    
                    # Simplify polygon
                    polygon = mask_to_polygons_string(mask, epsilon_percent=self._polygon_epsilon())
                    
                    bees.append({
                        'bbox': (center_x, center_y, width, height),
                        'centroid': (centroid_x, centroid_y),
                        'polygon': polygon,
                        'mask': mask,
                        'confidence': conf,
                        'is_segmentation': True
                    })
        else:
            # Bounding box model - original behavior
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    # Filter by class ID - only include bee class (class_id=0 for bees)
                    cls_id = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
                    
                    # Only include class 0 (bee) predictions, skip others
                    if cls_id != 0:
                        continue
                    
                    # Get bbox in xyxy format
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    # Convert to center format
                    x1, y1, x2, y2 = xyxy
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width / 2
                    center_y = y1 + height / 2
                    
                    bees.append({
                        'bbox': (center_x, center_y, width, height),
                        'centroid': (center_x, center_y),  # For bbox, centroid = bbox center
                        'polygon': None,
                        'mask': None,
                        'confidence': conf,
                        'is_segmentation': False
                    })
        
        return bees

    def _iter_bbox_detections(self, frame: np.ndarray, model):
        """Yield first-stage class-0 bbox detections as xyxy plus confidence."""
        results = model.predict(
            frame,
            conf=self.config['conf_threshold'],
            retina_masks=False,
            verbose=False
        )

        if len(results) == 0 or results[0].boxes is None:
            return

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
            if cls_id != 0:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            yield xyxy, conf

    def _bee_from_mask(self, mask: np.ndarray, confidence: float) -> Optional[Dict]:
        """Convert a binary bee mask to the standard prediction dictionary."""
        if mask is None or not np.any(mask > 0):
            return None

        mask = (mask > 0).astype(np.uint8) * 255
        centroid_x, centroid_y = mask_centroid(mask)
        center_x, center_y, width, height = bbox_from_mask(mask)
        polygon = mask_to_polygons_string(mask, epsilon_percent=self._polygon_epsilon())

        return {
            'bbox': (center_x, center_y, width, height),
            'centroid': (centroid_x, centroid_y),
            'polygon': polygon,
            'mask': mask,
            'confidence': confidence,
            'is_segmentation': True
        }

    def _predict_bees_bbox_sam2(self, frame: np.ndarray, bbox_model, sam2_model) -> List[Dict]:
        """Run bbox YOLO, then prompt SAM2 with each bbox to produce bee masks."""
        if sam2_model is None:
            raise RuntimeError("BBox + SAM2 selected, but no SAM2 model was loaded.")

        h, w = frame.shape[:2]
        bees = []
        for xyxy, conf in self._iter_bbox_detections(frame, bbox_model):
            x1, y1, x2, y2 = xyxy
            x1 = float(np.clip(x1, 0, w - 1))
            y1 = float(np.clip(y1, 0, h - 1))
            x2 = float(np.clip(x2, 0, w - 1))
            y2 = float(np.clip(y2, 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            mask = sam2_model.predict_with_box(frame, x1, y1, x2, y2)
            bee = self._bee_from_mask(mask, conf)
            if bee is not None:
                bees.append(bee)

        return bees

    def _predict_bees_bbox_instance_focused(self, frame: np.ndarray, bbox_model, instance_model) -> List[Dict]:
        """Run bbox YOLO, then segment each bbox crop with an instance-focused YOLO model."""
        if instance_model is None:
            raise RuntimeError("BBox + Instance-Focused YOLO selected, but no instance-focused model was loaded.")

        h, w = frame.shape[:2]
        crop_margin = int(self.config.get('instance_crop_margin', 50))
        target_size = 640
        bees = []

        for xyxy, bbox_conf in self._iter_bbox_detections(frame, bbox_model):
            x1, y1, x2, y2 = xyxy
            crop_x1 = max(0, int(np.floor(x1)) - crop_margin)
            crop_y1 = max(0, int(np.floor(y1)) - crop_margin)
            crop_x2 = min(w, int(np.ceil(x2)) + crop_margin + 1)
            crop_y2 = min(h, int(np.ceil(y2)) + crop_margin + 1)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue

            orig_crop_h, orig_crop_w = crop.shape[:2]
            crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            crop_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR)

            results = instance_model.predict(
                source=crop_bgr,
                conf=self.config['conf_threshold'],
                iou=0.45,
                augment=False,
                retina_masks=True,
                verbose=False,
            )

            if not results or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
                continue

            result = results[0]
            best_pred_idx = 0
            if result.boxes is not None and result.boxes.conf is not None:
                confs = result.boxes.conf.cpu().numpy()
                best_pred_idx = int(np.argmax(confs))
                confidence = float(confs[best_pred_idx])
            else:
                confidence = bbox_conf

            pred_mask = result.masks[best_pred_idx].data[0].cpu().numpy()
            pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
            pred_mask_resized = cv2.resize(
                pred_mask_binary,
                (orig_crop_w, orig_crop_h),
                interpolation=cv2.INTER_NEAREST
            )

            contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                pred_mask_clean = np.zeros_like(pred_mask_resized)
                cv2.drawContours(pred_mask_clean, [largest_contour], 0, 255, -1)
            else:
                pred_mask_clean = pred_mask_resized

            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = pred_mask_clean
            bee = self._bee_from_mask(full_mask, confidence)
            if bee is not None:
                bees.append(bee)

        return bees
    
    def _segmentation_to_mask(self, segmentation, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Convert COCO polygon segmentation to a binary mask.

        Supports polygon lists and COCO RLE dicts.
        Returns None only when segmentation data is missing/invalid.
        """
        h, w = frame_shape
        if segmentation is None:
            return None

        # Handle COCO RLE dict format.
        if isinstance(segmentation, dict):
            try:
                from pycocotools import mask as mask_utils
                decoded = mask_utils.decode(segmentation)
                if decoded is None:
                    return None

                if decoded.ndim == 3:
                    # If multiple components are decoded, union them.
                    decoded = np.any(decoded > 0, axis=2).astype(np.uint8)
                else:
                    decoded = (decoded > 0).astype(np.uint8)

                if decoded.shape[:2] != (h, w):
                    decoded = cv2.resize(decoded, (w, h), interpolation=cv2.INTER_NEAREST)

                mask = (decoded > 0).astype(np.uint8) * 255
                return mask if np.any(mask > 0) else None
            except Exception:
                return None

        # COCO polygon format can be list[list[float]] or occasionally a flat list[float].
        polygons = []
        if isinstance(segmentation, list):
            if len(segmentation) > 0 and isinstance(segmentation[0], (int, float, np.number)):
                polygons = [segmentation]  # flat polygon list -> wrap once
            else:
                polygons = segmentation
        if not polygons:
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            if not isinstance(poly, (list, tuple, np.ndarray)):
                continue

            poly_arr = np.asarray(poly, dtype=np.float32).reshape(-1)
            # Need at least 3 xy points.
            if poly_arr.size < 6:
                continue

            # Drop trailing value if odd-length malformed polygon.
            if poly_arr.size % 2 != 0:
                poly_arr = poly_arr[:-1]
                if poly_arr.size < 6:
                    continue

            pts = poly_arr.reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            pts_i = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts_i], 255)

        if np.any(mask > 0):
            return mask
        return None

    def _extract_gt_bees_from_coco(self, coco_data: Dict, frame_idx: int,
                                   frame_shape: Tuple[int, int]) -> List[Dict]:
        """Extract ground truth bee bounding boxes from COCO data"""
        # Build image_id to frame_idx mapping
        frame_to_image = {}
        for img in coco_data['images']:
            # Extract frame index from file_name (e.g., "frames/video_name/frame_000123.jpg")
            file_name = img['file_name']
            frame_str = Path(file_name).stem  # e.g., "frame_000123"
            img_frame_idx = int(frame_str.split('_')[1])  # Extract 123
            frame_to_image[img_frame_idx] = img['id']
        
        # Find the image_id for this frame
        if frame_idx not in frame_to_image:
            return []
        
        image_id = frame_to_image[frame_idx]
        
        # Build category mapping to get category names
        category_id_to_name = {}
        if 'categories' in coco_data:
            for cat in coco_data['categories']:
                category_id_to_name[cat['id']] = cat['name']
        
        # Extract bee annotations for this image
        bees = []
        missing_seg_mask_count = 0
        use_seg = self._bee_predictions_are_masks()
        for ann in coco_data['annotations']:
            # Only extract bee annotations (category_id=1), explicitly exclude hive (2) and chamber (3)
            if ann['image_id'] == image_id:
                category_id = ann.get('category_id', -1)
                category_name = category_id_to_name.get(category_id, '')
                
                # Multiple checks to ensure we only get bees:
                # 1. category_id must be 1
                # 2. category_id must NOT be 2 (hive) or 3 (chamber)
                # 3. category name (if available) must not be 'hive' or 'chamber'
                if category_id == 1 and category_name not in ['hive', 'chamber']:
                    # COCO bbox format is [x, y, width, height]
                    # Convert to center format [cx, cy, w, h]
                    x, y, w, h = ann['bbox']
                    cx = x + w / 2
                    cy = y + h / 2
                    
                    # Get instance ID if available
                    instance_id = ann.get('track_id', ann.get('instance_id', -1))
                    
                    mask = None
                    polygon = None

                    if use_seg:
                        mask = self._segmentation_to_mask(ann.get('segmentation'), frame_shape)
                        if mask is not None:
                            # For segmentation mode, align GT bbox/centroid to mask geometry.
                            cx, cy, w, h = bbox_from_mask(mask)
                            polygon = mask_to_polygons_string(mask, epsilon_percent=self._polygon_epsilon())
                        else:
                            missing_seg_mask_count += 1

                    bees.append({
                        'bbox': [cx, cy, w, h],
                        'centroid': (cx, cy),
                        'polygon': polygon,
                        'mask': mask,
                        'is_segmentation': mask is not None,
                        'instance_id': instance_id
                    })

        if self.config.get('debug_mode', False) and use_seg and missing_seg_mask_count > 0:
            self.log_message.emit(
                f"  Debug: {missing_seg_mask_count}/{len(bees)} GT bees had segmentation but no decodable mask in frame {frame_idx}"
            )
        
        return bees
    
    def _extract_gt_bees(self, annotations: List[Dict]) -> List[Dict]:
        """Extract ground truth bee bounding boxes"""
        bees = []
        for ann in annotations:
            # Get category information
            category = ann.get('category', 'bee')
            category_id = ann.get('category_id', 1)
            
            # Explicitly skip hive (category_id=2) and chamber (category_id=3) annotations
            if category in ['hive', 'chamber'] or category_id in [2, 3]:
                continue
            
            # Only process bee annotations
            if category != 'bee' and category_id != 1:
                continue
            
            # Get bbox from mask
            if 'mask' in ann and ann['mask'] is not None:
                bbox = bbox_from_mask(ann['mask'])
                # Try multiple keys for instance ID
                instance_id = ann.get('instance_id', ann.get('mask_id', ann.get('id', -1)))
                bees.append({
                    'bbox': bbox,
                    'centroid': (bbox[0], bbox[1]),
                    'instance_id': instance_id
                })
        
        return bees
    
    def _combine_instance_masks(self, instances: List[Dict], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Combine instance dict masks into one binary mask."""
        h, w = frame_shape
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for instance in instances:
            mask = instance.get('mask')
            if mask is None:
                continue
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.uint8) * 255)

        return combined_mask

    def _predict_hive_instances(self, frame: np.ndarray, model) -> List[Dict]:
        """Run YOLO segmentation prediction for hive instances."""
        if model is None:
            return []
        
        results = model.predict(
            frame,
            conf=self.config['conf_threshold'],
            retina_masks=True,  # Return masks at original image size
            verbose=False
        )
        
        hive_instances = []
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes

            for i, mask_data in enumerate(masks):
                # With retina_masks=True, masks are already at original image size
                mask = (mask_data > 0.5).astype(np.uint8) * 255
                hive_instances.append({
                    'hive_id': i + 1,
                    'mask': mask,
                    'polygon': mask_to_polygons_string(mask, epsilon_percent=self._polygon_epsilon()),
                    'confidence': float(boxes.conf[i].cpu().numpy()) if boxes is not None and boxes.conf is not None else None
                })

        return hive_instances

    def _predict_hive(self, frame: np.ndarray, model) -> Optional[np.ndarray]:
        """Run YOLO segmentation prediction for hive and return one combined mask."""
        if model is None:
            return None
        return self._combine_instance_masks(self._predict_hive_instances(frame, model), frame.shape[:2])

    def _predict_pollen_balls(self, frame: np.ndarray, model) -> List[Dict]:
        """Run YOLO segmentation prediction for pollen balls as separate instances."""
        if model is None:
            return []

        results = model.predict(
            frame,
            conf=self.config['conf_threshold'],
            retina_masks=True,
            verbose=False
        )

        pollen_balls = []
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes

            for i, mask_data in enumerate(masks):
                cls_id = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
                if cls_id != 0:
                    continue

                conf = float(boxes.conf[i].cpu().numpy())
                mask = (mask_data > 0.5).astype(np.uint8) * 255
                centroid_x, centroid_y = mask_centroid(mask)
                bbox = bbox_from_mask(mask)
                polygon = mask_to_polygons_string(mask, epsilon_percent=self._polygon_epsilon())

                pollen_balls.append({
                    'bbox': bbox,
                    'centroid': (centroid_x, centroid_y),
                    'polygon': polygon,
                    'mask': mask,
                    'confidence': conf,
                    'pollen_id': i + 1
                })

        return pollen_balls
    
    def _extract_gt_hive_instances(self, video_annotations: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """Extract ground truth hive instances from video-level annotations."""
        h, w = frame_shape
        hive_instances = []

        for ann in video_annotations:
            if ann.get('category', '') == 'hive' or ann.get('category_id', -1) == 2:
                if 'mask' in ann and ann['mask'] is not None:
                    # Resize mask to frame size if necessary
                    mask = ann['mask']
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0).astype(np.uint8) * 255
                    hive_instances.append({
                        'hive_id': ann.get('instance_id', ann.get('mask_id', ann.get('id', len(hive_instances) + 1))),
                        'mask': mask,
                        'polygon': mask_to_polygons_string(mask, epsilon_percent=self._polygon_epsilon())
                    })

        return hive_instances

    def _extract_gt_hive(self, video_annotations: List[Dict], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract combined ground truth hive mask from video-level annotations."""
        return self._combine_instance_masks(self._extract_gt_hive_instances(video_annotations, frame_shape), frame_shape)

    def _extract_gt_pollen_from_coco(self, coco_data: Dict, frame_idx: int,
                                     frame_shape: Tuple[int, int]) -> List[Dict]:
        """Extract ground truth pollen ball masks from COCO data for one frame."""
        frame_to_image = {}
        for img in coco_data['images']:
            file_name = img['file_name']
            frame_str = Path(file_name).stem
            img_frame_idx = int(frame_str.split('_')[1])
            frame_to_image[img_frame_idx] = img['id']

        if frame_idx not in frame_to_image:
            return []

        image_id = frame_to_image[frame_idx]

        category_id_to_name = {}
        if 'categories' in coco_data:
            for cat in coco_data['categories']:
                category_id_to_name[cat['id']] = cat['name']

        pollen_balls = []
        for ann in coco_data['annotations']:
            if ann.get('image_id') != image_id:
                continue

            category_id = ann.get('category_id', -1)
            category_name = category_id_to_name.get(category_id, '')
            if category_name != 'pollen' and category_id != 4:
                continue

            mask = self._segmentation_to_mask(ann.get('segmentation'), frame_shape)
            if mask is None:
                continue

            cx, cy, bw, bh = bbox_from_mask(mask)
            pollen_balls.append({
                'bbox': (cx, cy, bw, bh),
                'centroid': mask_centroid(mask),
                'polygon': mask_to_polygons_string(mask, epsilon_percent=self._polygon_epsilon()),
                'mask': mask,
                'confidence': None,
                'pollen_id': ann.get('id', len(pollen_balls) + 1)
            })

        return pollen_balls
    
    def _predict_chambers(self, frame: np.ndarray, model) -> Optional[Dict[int, np.ndarray]]:
        """Run YOLO segmentation prediction for chambers"""
        if model is None:
            return None
        
        results = model.predict(
            frame,
            conf=self.config['conf_threshold'],
            retina_masks=True,  # Return masks at original image size
            verbose=False
        )
        
        chamber_masks = {}
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            
            # Store masks with their x-coordinates for sorting
            chambers_with_coords = []
            for i, mask in enumerate(masks):
                # With retina_masks=True, masks are already at original image size
                chamber_mask = (mask > 0.5).astype(np.uint8) * 255
                
                # Calculate centroid x-coordinate
                cx, _, _, _ = bbox_from_mask(chamber_mask)
                chambers_with_coords.append((cx, chamber_mask))
            
            # Sort chambers left to right by x-coordinate
            chambers_with_coords.sort(key=lambda x: x[0])
            
            # Assign IDs 1, 2, 3... from left to right
            for i, (_, chamber_mask) in enumerate(chambers_with_coords):
                chamber_masks[i + 1] = chamber_mask  # 1-indexed
        
        return chamber_masks
    
    def _extract_gt_chambers(self, video_annotations: List[Dict], frame_shape: Tuple[int, int]) -> Optional[Dict[int, np.ndarray]]:
        """Extract ground truth chamber masks from video-level annotations"""
        h, w = frame_shape
        chamber_list = []
        
        for ann in video_annotations:
            if ann.get('category', '') == 'chamber' or ann.get('category_id', -1) == 3:
                if 'mask' in ann and ann['mask'] is not None:
                    mask = ann['mask']
                    # Resize mask to frame size if necessary
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Calculate centroid x-coordinate for sorting
                    cx, _, _, _ = bbox_from_mask(mask)
                    chamber_list.append((cx, mask))
        
        # Sort chambers left to right by x-coordinate
        chamber_list.sort(key=lambda x: x[0])
        
        # Assign IDs 1, 2, 3... from left to right
        chamber_masks = {}
        for i, (_, mask) in enumerate(chamber_list):
            chamber_masks[i + 1] = mask  # 1-indexed
        
        return chamber_masks
    
    def _calculate_nearest_bee_distance(self, bee_idx: int, all_bees: List[Dict],
                                       chamber_id: Optional[int] = None,
                                       chamber_masks = None) -> Optional[float]:
        """Calculate distance from bee at bee_idx to its nearest neighbor bee.
        
        Uses contour/pixel distance in segmentation mode (same utility as predictions),
        otherwise centroid-to-centroid distance for bbox mode.
        If chamber_id and chamber_masks are provided, only considers bees in the same chamber.
        Returns None if no other bees are available.
        """
        conn = self._nearest_bee_connection(bee_idx, all_bees, chamber_id, chamber_masks)
        return conn[2] if conn is not None else None

    def _nearest_bee_connection(self, bee_idx: int, all_bees: List[Dict],
                               chamber_id: Optional[int] = None,
                               chamber_masks=None) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """Get connection points to nearest bee for visualization.

        Segmentation mode: nearest contour points.
        BBox mode: centroid-to-centroid points.
        """
        if len(all_bees) <= 1:
            return None

        current_bee = all_bees[bee_idx]
        current_center = current_bee.get('centroid', (current_bee['bbox'][0], current_bee['bbox'][1]))
        seg_mode = self._bee_predictions_are_masks()
        use_seg = (
            seg_mode
            and current_bee.get('is_segmentation', False)
            and current_bee.get('mask') is not None
        )

        # In segmentation mode, require a valid mask for this bee.
        # If missing, return no nearest-connection instead of centroid fallback.
        if seg_mode and not use_seg:
            return None

        distance_method = self.config.get('distance_method', 'contour')

        best = None
        best_dist = np.inf
        for i, other_bee in enumerate(all_bees):
            if i == bee_idx:
                continue

            if chamber_id is not None and chamber_masks is not None:
                other_center = other_bee.get('centroid', (other_bee['bbox'][0], other_bee['bbox'][1]))
                if point_in_chamber(other_center, chamber_masks) != chamber_id:
                    continue

            if use_seg:
                # In segmentation mode, only compare against other bees with valid masks.
                if not (other_bee.get('is_segmentation', False) and other_bee.get('mask') is not None):
                    continue
                # Respect configured distance method for nearest-neighbor selection.
                dist = distance_between_masks(
                    current_bee['mask'], other_bee['mask'], method=distance_method
                )
                # Always draw true contour-to-contour nearest endpoints.
                p1, p2, _ = closest_points_between_masks(
                    current_bee['mask'], other_bee['mask'], method='contour'
                )
            else:
                other_center = other_bee.get('centroid', (other_bee['bbox'][0], other_bee['bbox'][1]))
                p1, p2 = current_center, other_center
                dist = float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

            if dist < best_dist and np.isfinite(dist) and p1 is not None and p2 is not None:
                best_dist = dist
                best = (p1, p2, dist)

        return best

    def _calculate_hive_distance(self, bee: Dict, hive_mask: Optional[np.ndarray]) -> Optional[float]:
        """Calculate bee-to-hive distance with model-conditional behavior.

        Segmentation bees: nearest-pixel-to-nearest-pixel using distance_between_masks.
        BBox bees: centroid-to-mask distance via distance_to_mask.
        """
        if hive_mask is None or not np.any(hive_mask > 0):
            return None

        is_seg_bee = (
            self._bee_predictions_are_masks()
            and bee.get('is_segmentation', False)
            and bee.get('mask') is not None
        )

        if is_seg_bee:
            # Use contour mode for nearest-pixel distance to match requested behavior.
            return distance_between_masks(bee['mask'], hive_mask, method='contour')

        bee_center = bee.get('centroid', (bee['bbox'][0], bee['bbox'][1]))
        return distance_to_mask(bee_center, hive_mask)

    def _mask_to_chamber(self, mask: Optional[np.ndarray], chamber_id: Optional[int], chamber_masks) -> Optional[np.ndarray]:
        """Restrict a binary mask to one chamber when chamber information is available."""
        if mask is None or not np.any(mask > 0):
            return None

        if chamber_id is None or not chamber_masks:
            return mask

        chamber_mask = chamber_masks.get(chamber_id)
        if chamber_mask is None:
            return mask

        masked = mask.copy()
        masked[chamber_mask == 0] = 0
        return masked if np.any(masked > 0) else None

    def _calculate_chamber_hive_distance(self, bee: Dict, hive_mask: Optional[np.ndarray],
                                         chamber_id: Optional[int] = None,
                                         chamber_masks=None) -> Optional[float]:
        """Calculate bee-to-hive distance using only hive pixels in the bee's chamber."""
        chamber_hive_mask = self._mask_to_chamber(hive_mask, chamber_id, chamber_masks)
        return self._calculate_hive_distance(bee, chamber_hive_mask)

    def _nearest_hive_connection(self, bee: Dict, hive_mask: Optional[np.ndarray],
                                 chamber_id: Optional[int] = None,
                                 chamber_masks=None) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """Get nearest hive connection points for a bee within its chamber."""
        chamber_hive_mask = self._mask_to_chamber(hive_mask, chamber_id, chamber_masks)
        if chamber_hive_mask is None or not np.any(chamber_hive_mask > 0):
            return None

        is_seg_bee = (
            self._bee_predictions_are_masks()
            and bee.get('is_segmentation', False)
            and bee.get('mask') is not None
        )

        if is_seg_bee:
            p1, p2, dist = closest_points_between_masks(bee['mask'], chamber_hive_mask, method='contour')
        else:
            bee_center = bee.get('centroid', (bee['bbox'][0], bee['bbox'][1]))
            p1 = bee_center
            p2, dist = self._closest_point_to_mask(bee_center, chamber_hive_mask)

        if p1 is None or p2 is None or not np.isfinite(dist):
            return None
        return p1, p2, dist

    def _mask_in_chamber(self, mask: np.ndarray, chamber_id: Optional[int], chamber_masks) -> bool:
        """Return True when a mask belongs to the requested chamber.

        Uses mask/chamber overlap first, with centroid containment as a fallback.
        """
        if chamber_id is None or not chamber_masks:
            return True

        chamber_mask = chamber_masks.get(chamber_id)
        if chamber_mask is None:
            return True

        if mask is not None and np.any(mask > 0):
            return bool(np.logical_and(mask > 0, chamber_mask > 0).any())

        return False

    def _closest_point_to_mask(self, point: Tuple[float, float],
                               mask: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """Return nearest foreground pixel in mask to point and its distance."""
        if mask is None or not np.any(mask > 0):
            return None, np.inf

        pixels = np.argwhere(mask > 0)
        coords = pixels[:, ::-1].astype(np.float32)
        point_arr = np.asarray(point, dtype=np.float32)
        dists = np.sqrt(np.sum((coords - point_arr) ** 2, axis=1))
        idx = int(np.argmin(dists))
        return (float(coords[idx, 0]), float(coords[idx, 1])), float(dists[idx])

    def _nearest_pollen_connection(self, bee: Dict, pollen_balls: List[Dict],
                                   chamber_id: Optional[int] = None,
                                   chamber_masks=None) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """Get nearest pollen ball connection points for a bee within its chamber."""
        if not pollen_balls:
            return None

        bee_center = bee.get('centroid', (bee['bbox'][0], bee['bbox'][1]))
        is_seg_bee = (
            self._bee_predictions_are_masks()
            and bee.get('is_segmentation', False)
            and bee.get('mask') is not None
        )
        distance_method = self.config.get('distance_method', 'contour')

        best = None
        best_dist = np.inf
        for pollen in pollen_balls:
            pollen_mask = pollen.get('mask')
            if pollen_mask is None or not np.any(pollen_mask > 0):
                continue

            if not self._mask_in_chamber(pollen_mask, chamber_id, chamber_masks):
                continue

            if is_seg_bee:
                dist = distance_between_masks(bee['mask'], pollen_mask, method=distance_method)
                p1, p2, _ = closest_points_between_masks(bee['mask'], pollen_mask, method='contour')
            else:
                p1 = bee_center
                p2, dist = self._closest_point_to_mask(bee_center, pollen_mask)

            if dist < best_dist and np.isfinite(dist) and p1 is not None and p2 is not None:
                best_dist = dist
                best = (p1, p2, dist)

        return best

    def _calculate_pollen_distance(self, bee: Dict, pollen_balls: List[Dict],
                                   chamber_id: Optional[int] = None,
                                   chamber_masks=None) -> Optional[float]:
        """Calculate distance from bee to nearest pollen ball in its chamber."""
        conn = self._nearest_pollen_connection(bee, pollen_balls, chamber_id, chamber_masks)
        return conn[2] if conn is not None else None
    
    def _calculate_avg_chamber_bee_distance(self, bee_idx: int, all_bees: List[Dict], 
                                           chamber_id: Optional[int] = None,
                                           chamber_masks = None) -> Optional[float]:
        """Calculate average distance from bee to other bees in the same chamber.
        
        Uses centroids if available (segmentation), otherwise bbox centers.
        """
        if chamber_id is None or not chamber_masks:
            return None  # No chamber information
        
        if len(all_bees) <= 1:
            return None  # No other bees
        
        # Use centroid if available, otherwise bbox center
        bee = all_bees[bee_idx]
        bee_center = bee.get('centroid', (bee['bbox'][0], bee['bbox'][1]))
        chamber_distances = []
        
        for i, other_bee in enumerate(all_bees):
            if i == bee_idx:
                continue  # Skip self
            
            # Check if other bee is in the same chamber
            other_center = other_bee.get('centroid', (other_bee['bbox'][0], other_bee['bbox'][1]))
            other_chamber_id = point_in_chamber(other_center, chamber_masks)
            
            if other_chamber_id == chamber_id:
                # Calculate distance to this chamber mate (using centroid)
                other_center = other_bee.get('centroid', (other_bee['bbox'][0], other_bee['bbox'][1]))
                dist = np.sqrt((bee_center[0] - other_center[0])**2 + (bee_center[1] - other_center[1])**2)
                chamber_distances.append(dist)
        
        # Return average distance to chamber mates
        if chamber_distances:
            return np.mean(chamber_distances)
        return None  # No other bees in same chamber
    
    def _match_and_write_bees(self, video_id: str, frame_idx: int,
                              pred_bees: List[Dict], gt_bees: List[Dict],
                              pred_hive_mask, gt_hive_mask,
                              pred_pollen_balls: List[Dict], gt_pollen_balls: List[Dict],
                              pred_chamber_masks, gt_chamber_masks,
                              writer):
        """Match predictions to GT and write to CSV. Returns matching info for visualization."""
        # Build IoU cost matrix
        n_preds = len(pred_bees)
        n_gts = len(gt_bees)
        
        if n_preds == 0 and n_gts == 0:
            return [], [], [], {}, {}  # Nothing to match

        # Cache nearest-bee connection per index so distance and visualization share one computation.
        pred_nearest_connections = {}
        gt_nearest_connections = {}
        
        cost_matrix = np.zeros((n_preds, n_gts))
        for i, pred in enumerate(pred_bees):
            for j, gt in enumerate(gt_bees):
                cost_matrix[i, j] = compute_bbox_iou(pred['bbox'], gt['bbox'])
        
        # Match using Hungarian algorithm
        matched_pairs, unmatched_preds, unmatched_gts = match_by_hungarian(
            cost_matrix, self.config['iou_threshold']
        )
        
        # Update statistics
        self.total_matches += len(matched_pairs)
        self.total_fps += len(unmatched_preds)
        self.total_fns += len(unmatched_gts)
        
        # Write matched pairs
        for pred_idx, gt_idx in matched_pairs:
            pred = pred_bees[pred_idx]
            gt = gt_bees[gt_idx]
            iou = cost_matrix[pred_idx, gt_idx]
            
            # Get chamber IDs
            pred_center = pred.get('centroid', (pred['bbox'][0], pred['bbox'][1]))
            gt_center = gt.get('centroid', (gt['bbox'][0], gt['bbox'][1]))
            pred_chamber_id = point_in_chamber(pred_center, pred_chamber_masks) if pred_chamber_masks else None
            gt_chamber_id = point_in_chamber(gt_center, gt_chamber_masks) if gt_chamber_masks else None
            
            # Get hive distances
            pred_hive_dist = self._calculate_chamber_hive_distance(
                pred, pred_hive_mask, pred_chamber_id, pred_chamber_masks
            )
            gt_hive_dist = self._calculate_chamber_hive_distance(
                gt, gt_hive_mask, gt_chamber_id, gt_chamber_masks
            )

            # Get nearest pollen distances (chamber-restricted)
            pred_pollen_dist = self._calculate_pollen_distance(
                pred, pred_pollen_balls, pred_chamber_id, pred_chamber_masks
            )
            gt_pollen_dist = self._calculate_pollen_distance(
                gt, gt_pollen_balls, gt_chamber_id, gt_chamber_masks
            )
            
            # Get nearest bee distances (chamber-restricted)
            pred_conn = self._nearest_bee_connection(pred_idx, pred_bees, pred_chamber_id, pred_chamber_masks)
            gt_conn = self._nearest_bee_connection(gt_idx, gt_bees, gt_chamber_id, gt_chamber_masks)
            pred_nearest_connections[pred_idx] = pred_conn
            gt_nearest_connections[gt_idx] = gt_conn
            pred_nearest_dist = pred_conn[2] if pred_conn is not None else None
            gt_nearest_dist = gt_conn[2] if gt_conn is not None else None
            
            # Get average chamber bee distances
            pred_chamber_dist = self._calculate_avg_chamber_bee_distance(pred_idx, pred_bees, pred_chamber_id, pred_chamber_masks)
            gt_chamber_dist = self._calculate_avg_chamber_bee_distance(gt_idx, gt_bees, gt_chamber_id, gt_chamber_masks)
            
            writer.writerow([
                video_id, frame_idx, 'matched',
                pred['bbox'][0], pred['bbox'][1], pred['bbox'][2], pred['bbox'][3], pred['confidence'],
                gt['bbox'][0], gt['bbox'][1], gt['bbox'][2], gt['bbox'][3], gt['instance_id'],
                iou,
                pred_chamber_id if pred_chamber_id is not None else '',
                gt_chamber_id if gt_chamber_id is not None else '',
                pred_hive_dist if pred_hive_dist is not None and pred_hive_dist != np.inf else '',
                gt_hive_dist if gt_hive_dist is not None and gt_hive_dist != np.inf else '',
                pred_pollen_dist if pred_pollen_dist is not None and pred_pollen_dist != np.inf else '',
                gt_pollen_dist if gt_pollen_dist is not None and gt_pollen_dist != np.inf else '',
                pred_nearest_dist if pred_nearest_dist is not None else '',
                gt_nearest_dist if gt_nearest_dist is not None else '',
                pred_chamber_dist if pred_chamber_dist is not None else '',
                gt_chamber_dist if gt_chamber_dist is not None else '',
                pred.get('polygon') or '',
                gt.get('polygon') or ''
            ])
        
        # Write false positives (unmatched predictions)
        for pred_idx in unmatched_preds:
            pred = pred_bees[pred_idx]
            pred_center = pred.get('centroid', (pred['bbox'][0], pred['bbox'][1]))
            pred_chamber_id = point_in_chamber(pred_center, pred_chamber_masks) if pred_chamber_masks else None
            
            pred_hive_dist = self._calculate_chamber_hive_distance(
                pred, pred_hive_mask, pred_chamber_id, pred_chamber_masks
            )
            pred_pollen_dist = self._calculate_pollen_distance(
                pred, pred_pollen_balls, pred_chamber_id, pred_chamber_masks
            )
            
            # Get nearest bee distance (chamber-restricted)
            pred_conn = self._nearest_bee_connection(pred_idx, pred_bees, pred_chamber_id, pred_chamber_masks)
            pred_nearest_connections[pred_idx] = pred_conn
            pred_nearest_dist = pred_conn[2] if pred_conn is not None else None
            
            # Get average chamber bee distance
            pred_chamber_dist = self._calculate_avg_chamber_bee_distance(pred_idx, pred_bees, pred_chamber_id, pred_chamber_masks)
            
            writer.writerow([
                video_id, frame_idx, 'false_positive',
                pred['bbox'][0], pred['bbox'][1], pred['bbox'][2], pred['bbox'][3], pred['confidence'],
                '', '', '', '', '',  # No GT
                '',  # No IoU
                pred_chamber_id if pred_chamber_id is not None else '',
                '',  # No GT chamber
                pred_hive_dist if pred_hive_dist is not None and pred_hive_dist != np.inf else '',
                '',  # No GT hive distance
                pred_pollen_dist if pred_pollen_dist is not None and pred_pollen_dist != np.inf else '',
                '',  # No GT pollen distance
                pred_nearest_dist if pred_nearest_dist is not None else '',
                '',  # No GT bee distance
                pred_chamber_dist if pred_chamber_dist is not None else '',
                '',  # No GT chamber bee distance
                pred.get('polygon') or '',
                ''
            ])
        
        # Write false negatives (unmatched GT)
        for gt_idx in unmatched_gts:
            gt = gt_bees[gt_idx]
            gt_center = gt.get('centroid', (gt['bbox'][0], gt['bbox'][1]))
            gt_chamber_id = point_in_chamber(gt_center, gt_chamber_masks) if gt_chamber_masks else None
            
            gt_hive_dist = self._calculate_chamber_hive_distance(
                gt, gt_hive_mask, gt_chamber_id, gt_chamber_masks
            )
            gt_pollen_dist = self._calculate_pollen_distance(
                gt, gt_pollen_balls, gt_chamber_id, gt_chamber_masks
            )
            
            # Get nearest bee distance (chamber-restricted)
            gt_conn = self._nearest_bee_connection(gt_idx, gt_bees, gt_chamber_id, gt_chamber_masks)
            gt_nearest_connections[gt_idx] = gt_conn
            gt_nearest_dist = gt_conn[2] if gt_conn is not None else None
            
            # Get average chamber bee distance
            gt_chamber_dist = self._calculate_avg_chamber_bee_distance(gt_idx, gt_bees, gt_chamber_id, gt_chamber_masks)
            
            writer.writerow([
                video_id, frame_idx, 'false_negative',
                '', '', '', '', '',  # No prediction
                gt['bbox'][0], gt['bbox'][1], gt['bbox'][2], gt['bbox'][3], gt['instance_id'],
                '',  # No IoU
                '',  # No pred chamber
                gt_chamber_id if gt_chamber_id is not None else '',
                '',  # No pred hive distance
                gt_hive_dist if gt_hive_dist is not None and gt_hive_dist != np.inf else '',
                '',  # No pred pollen distance
                gt_pollen_dist if gt_pollen_dist is not None and gt_pollen_dist != np.inf else '',
                '',  # No pred bee distance
                gt_nearest_dist if gt_nearest_dist is not None else '',
                '',  # No pred chamber bee distance
                gt_chamber_dist if gt_chamber_dist is not None else '',
                '',
                gt.get('polygon') or ''
            ])
        
        return matched_pairs, unmatched_preds, unmatched_gts, pred_nearest_connections, gt_nearest_connections
    
    def _save_visualization(self, frame: np.ndarray, video_id: str, frame_idx: int,
                           pred_bees: List[Dict], gt_bees: List[Dict],
                           matched_pairs: List[Tuple[int, int]],
                           unmatched_preds: List[int],
                           unmatched_gts: List[int],
                           results_folder: Path,
                           pred_hive_mask=None, gt_hive_mask=None,
                           pred_pollen_balls: Optional[List[Dict]] = None,
                           gt_pollen_balls: Optional[List[Dict]] = None,
                           pred_chamber_masks=None, gt_chamber_masks=None,
                           pred_nearest_connections: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float], float]]] = None,
                           gt_nearest_connections: Optional[Dict[int, Tuple[Tuple[float, float], Tuple[float, float], float]]]=None):
        """Save visualization with color-coded bee boxes and hive/chamber outlines"""
        try:
            # Create visualizations folder
            vis_folder = results_folder / "visualizations" / video_id
            vis_folder.mkdir(parents=True, exist_ok=True)
            
            # Create single image visualization
            vis_img = frame.copy()
            
            # Draw hive masks
            gt_hive_color = (0, 255, 255)   # Yellow in BGR — GT hive
            pred_hive_color = (0, 165, 255) # Orange in BGR — predicted hive
            
            if gt_hive_mask is not None and np.any(gt_hive_mask > 0):
                contours, _ = cv2.findContours(gt_hive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, gt_hive_color, 3)
            
            if pred_hive_mask is not None and np.any(pred_hive_mask > 0):
                contours, _ = cv2.findContours(pred_hive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, pred_hive_color, 3)

            # Draw pollen ball masks
            gt_pollen_color = (0, 215, 255)     # Gold in BGR — GT pollen
            pred_pollen_color = (255, 0, 255)   # Magenta in BGR — predicted pollen

            def _draw_pollen_balls(pollen_balls, color, label_prefix):
                for pollen in pollen_balls or []:
                    mask = pollen.get('mask')
                    if mask is None or not np.any(mask > 0):
                        continue
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_img, contours, -1, color, 2)
                    centroid = pollen.get('centroid')
                    if centroid:
                        cx, cy = int(centroid[0]), int(centroid[1])
                        cv2.circle(vis_img, (cx, cy), 3, color, -1)
                        cv2.putText(vis_img, label_prefix, (cx + 5, cy - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            _draw_pollen_balls(gt_pollen_balls, gt_pollen_color, "GT P")
            _draw_pollen_balls(pred_pollen_balls, pred_pollen_color, "P")
            
            # Draw chamber masks
            gt_chamber_color = (255, 0, 0)    # Blue in BGR — GT chamber
            pred_chamber_color = (255, 255, 0) # Cyan in BGR — predicted chamber
            
            if gt_chamber_masks:
                for chamber_id, mask in gt_chamber_masks.items():
                    if np.any(mask > 0):
                        mask_uint8 = mask.astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis_img, contours, -1, gt_chamber_color, 3)
            
            if pred_chamber_masks:
                for chamber_id, mask in pred_chamber_masks.items():
                    if np.any(mask > 0):
                        mask_uint8 = mask.astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis_img, contours, -1, pred_chamber_color, 3)
                        
                        # Draw chamber ID label at bottom-center of chamber
                        cx, cy, w, h = bbox_from_mask(mask_uint8)
                        label_x = int(cx)
                        label_y = int(cy + h/2 - 10)  # Bottom of chamber
                        cv2.putText(vis_img, f"C{chamber_id}", (label_x, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_chamber_color, 3)
            
            # Helper to draw a single bee prediction (segmentation mask or bbox)
            def _draw_bee(vis_img, bee, color, label_prefix, label_suffix=""):
                """Draw bee as segmentation mask (semi-transparent + contour) or bbox."""
                if bee.get('is_segmentation', False) and bee.get('mask') is not None:
                    mask = bee['mask']
                    if np.any(mask > 0):
                        # Semi-transparent filled overlay
                        overlay = vis_img.copy()
                        overlay[mask > 0] = color
                        cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
                        
                        # Contour outline
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(vis_img, contours, -1, color, 2)
                        
                        # Centroid marker
                        centroid = bee.get('centroid')
                        if centroid:
                            cv2.circle(vis_img, (int(centroid[0]), int(centroid[1])), 4, color, -1)
                        
                        # Label at centroid
                        label_x = int(centroid[0]) + 8 if centroid else int(bee['bbox'][0])
                        label_y = int(centroid[1]) - 8 if centroid else int(bee['bbox'][1] - bee['bbox'][3]/2 - 5)
                        label = f"{label_prefix}{label_suffix}"
                        cv2.putText(vis_img, label, (label_x, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    # Bounding box drawing (original behavior)
                    cx, cy, bw, bh = bee['bbox']
                    x1, y1 = int(cx - bw/2), int(cy - bh/2)
                    x2, y2 = int(cx + bw/2), int(cy + bh/2)
                    
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"{label_prefix}{label_suffix}"
                    cv2.putText(vis_img, label, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw false negatives (GT with no match) in RED
            for gt_idx in unmatched_gts:
                gt = gt_bees[gt_idx]
                instance_id = gt.get('instance_id', -1)
                _draw_bee(vis_img, gt, (0, 0, 255), f"FN:{instance_id}")
            
            # Draw false positives (prediction with no GT match) in PINK
            for pred_idx in unmatched_preds:
                pred = pred_bees[pred_idx]
                conf = pred.get('confidence', 0)
                _draw_bee(vis_img, pred, (203, 192, 255), f"FP (", f"{conf:.2f})")
            
            # Draw matched pairs (correct predictions) in GREEN
            for pred_idx, gt_idx in matched_pairs:
                pred = pred_bees[pred_idx]
                gt = gt_bees[gt_idx]
                instance_id = gt.get('instance_id', -1)
                conf = pred.get('confidence', 0)
                _draw_bee(vis_img, pred, (0, 255, 0), f"ID:{instance_id} (", f"{conf:.2f})")

            # Visualize nearest-bee connections for segmentation mode.
            if self._bee_predictions_are_masks():
                def _draw_nearest_lines(bees, chamber_masks, line_color, conn_map):
                    for i, bee in enumerate(bees):
                        conn = conn_map.get(i) if conn_map is not None else None
                        if conn is None:
                            # Fallback for robustness if visualization is called without precomputed cache.
                            bee_center = bee.get('centroid', (bee['bbox'][0], bee['bbox'][1]))
                            cid = point_in_chamber(bee_center, chamber_masks) if chamber_masks else None
                            conn = self._nearest_bee_connection(i, bees, cid, chamber_masks)
                        if conn is None:
                            continue
                        p1, p2, _ = conn
                        a = (int(round(p1[0])), int(round(p1[1])))
                        b = (int(round(p2[0])), int(round(p2[1])))
                        cv2.line(vis_img, a, b, line_color, 1, cv2.LINE_AA)
                        cv2.circle(vis_img, a, 3, line_color, -1)
                        cv2.circle(vis_img, b, 3, line_color, -1)

                # Pred and GT nearest links use distinct colors for readability.
                _draw_nearest_lines(pred_bees, pred_chamber_masks, (0, 255, 0), pred_nearest_connections)    # green
                _draw_nearest_lines(gt_bees, gt_chamber_masks, (0, 140, 255), gt_nearest_connections)      # orange

            # Visualize nearest hive connections. Lines are chamber-restricted.
            def _draw_hive_lines(bees, hive_mask, chamber_masks, line_color):
                if hive_mask is None or not np.any(hive_mask > 0):
                    return
                for bee in bees:
                    bee_center = bee.get('centroid', (bee['bbox'][0], bee['bbox'][1]))
                    cid = point_in_chamber(bee_center, chamber_masks) if chamber_masks else None
                    conn = self._nearest_hive_connection(bee, hive_mask, cid, chamber_masks)
                    if conn is None:
                        continue
                    p1, p2, _ = conn
                    a = (int(round(p1[0])), int(round(p1[1])))
                    b = (int(round(p2[0])), int(round(p2[1])))
                    cv2.line(vis_img, a, b, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.line(vis_img, a, b, line_color, 2, cv2.LINE_AA)
                    cv2.circle(vis_img, a, 4, (0, 0, 0), -1)
                    cv2.circle(vis_img, a, 3, line_color, -1)
                    cv2.circle(vis_img, b, 4, (0, 0, 0), -1)
                    cv2.circle(vis_img, b, 3, line_color, -1)

            _draw_hive_lines(pred_bees, pred_hive_mask, pred_chamber_masks, pred_hive_color)
            _draw_hive_lines(gt_bees, gt_hive_mask, gt_chamber_masks, gt_hive_color)

            # Visualize nearest pollen-ball connections.
            def _draw_pollen_lines(bees, pollen_balls, chamber_masks, line_color):
                if not pollen_balls:
                    return
                for bee in bees:
                    bee_center = bee.get('centroid', (bee['bbox'][0], bee['bbox'][1]))
                    cid = point_in_chamber(bee_center, chamber_masks) if chamber_masks else None
                    conn = self._nearest_pollen_connection(bee, pollen_balls, cid, chamber_masks)
                    if conn is None:
                        continue
                    p1, p2, _ = conn
                    a = (int(round(p1[0])), int(round(p1[1])))
                    b = (int(round(p2[0])), int(round(p2[1])))
                    cv2.line(vis_img, a, b, line_color, 1, cv2.LINE_AA)
                    cv2.circle(vis_img, a, 3, line_color, -1)
                    cv2.circle(vis_img, b, 3, line_color, -1)

            _draw_pollen_lines(pred_bees, pred_pollen_balls, pred_chamber_masks, (255, 0, 255))
            _draw_pollen_lines(gt_bees, gt_pollen_balls, gt_chamber_masks, (0, 215, 255))
            
            # Add legend
            legend_y = 30
            cv2.putText(vis_img, "Bees:", (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, "Green = Matched", (10, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_img, "Red = False Negative", (10, legend_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(vis_img, "Pink = False Positive", (10, legend_y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (203, 192, 255), 2)
            
            # Add hive/chamber legend if applicable
            has_hive = (gt_hive_mask is not None and np.any(gt_hive_mask > 0)) or \
                       (pred_hive_mask is not None and np.any(pred_hive_mask > 0))
            has_chamber = (gt_chamber_masks and any(np.any(m > 0) for m in gt_chamber_masks.values())) or \
                          (pred_chamber_masks and any(np.any(m > 0) for m in pred_chamber_masks.values()))
            has_pollen = bool(gt_pollen_balls) or bool(pred_pollen_balls)
            
            legend_offset = 120
            if has_hive:
                cv2.putText(vis_img, "Hive — Yellow = GT,  Orange = Pred/links", (10, legend_y + legend_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                legend_offset += 25
            
            if has_chamber:
                cv2.putText(vis_img, "Chamber — Blue = GT,  Cyan = Pred", (10, legend_y + legend_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
                legend_offset += 25

            if has_pollen:
                cv2.putText(vis_img, "Pollen — Gold = GT,  Magenta = Pred/links", (10, legend_y + legend_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, pred_pollen_color, 2)
                legend_offset += 25

            if self._bee_predictions_are_masks():
                cv2.putText(vis_img, "Nearest-bee links: Green=Pred, Orange=GT", (10, legend_y + legend_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            
            # Add statistics at bottom
            h = vis_img.shape[0]
            stats_text = f"Matched: {len(matched_pairs)} | FP: {len(unmatched_preds)} | FN: {len(unmatched_gts)}"
            cv2.putText(vis_img, stats_text, (10, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save image
            vis_path = vis_folder / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(vis_path), vis_img)
            
        except Exception as e:
            self.log_message.emit(f"  Warning: Failed to save visualization: {str(e)}")
    
    def _hive_instance_chamber_id(self, hive: Dict, chamber_masks) -> int:
        """Return the chamber id containing the hive instance, or -1 when unavailable."""
        if not chamber_masks:
            return -1

        mask = hive.get('mask')
        if mask is None:
            return -1

        best_chamber_id = -1
        best_overlap = 0
        for chamber_id, chamber_mask in chamber_masks.items():
            overlap = int(np.logical_and(mask > 0, chamber_mask > 0).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_chamber_id = chamber_id

        if best_chamber_id != -1:
            return best_chamber_id

        centroid = mask_centroid(mask)
        chamber_id = point_in_chamber(centroid, chamber_masks)
        return chamber_id if chamber_id is not None else -1

    def _write_hive_results(self, video_id: str, frame_idx: int,
                            pred_hives: List[Dict], gt_hives: List[Dict],
                            writer, pred_chamber_masks=None, gt_chamber_masks=None):
        """Write one hive CSV row per matched, predicted, or ground-truth hive instance."""
        if not pred_hives and not gt_hives:
            return

        cost_matrix = np.zeros((len(pred_hives), len(gt_hives)))
        for i, pred_hive in enumerate(pred_hives):
            for j, gt_hive in enumerate(gt_hives):
                cost_matrix[i, j] = compute_mask_iou(pred_hive['mask'], gt_hive['mask'])

        matched_pairs, unmatched_preds, unmatched_gts = match_by_hungarian(
            cost_matrix, self.config['iou_threshold']
        )

        for pred_idx, gt_idx in matched_pairs:
            pred_hive = pred_hives[pred_idx]
            gt_hive = gt_hives[gt_idx]
            pred_mask = pred_hive['mask']
            gt_mask = gt_hive['mask']
            pred_chamber_id = self._hive_instance_chamber_id(pred_hive, pred_chamber_masks)
            gt_chamber_id = self._hive_instance_chamber_id(gt_hive, gt_chamber_masks or pred_chamber_masks)
            chamber_id = pred_chamber_id if pred_chamber_id != -1 else gt_chamber_id

            writer.writerow([
                video_id, frame_idx, chamber_id,
                'matched',
                pred_hive.get('hive_id', pred_idx + 1), gt_hive.get('hive_id', gt_idx + 1),
                int(np.sum(pred_mask > 0)), int(np.sum(gt_mask > 0)), cost_matrix[pred_idx, gt_idx],
                pred_hive.get('polygon', ''),
                gt_hive.get('polygon', '')
            ])

        for pred_idx in unmatched_preds:
            pred_hive = pred_hives[pred_idx]
            pred_mask = pred_hive['mask']

            writer.writerow([
                video_id, frame_idx, self._hive_instance_chamber_id(pred_hive, pred_chamber_masks),
                'false_positive',
                pred_hive.get('hive_id', pred_idx + 1), '',
                int(np.sum(pred_mask > 0)), '', '',
                pred_hive.get('polygon', ''), ''
            ])

        for gt_idx in unmatched_gts:
            gt_hive = gt_hives[gt_idx]
            gt_mask = gt_hive['mask']

            writer.writerow([
                video_id, frame_idx, self._hive_instance_chamber_id(gt_hive, gt_chamber_masks or pred_chamber_masks),
                'false_negative',
                '', gt_hive.get('hive_id', gt_idx + 1),
                '', int(np.sum(gt_mask > 0)), '',
                '', gt_hive.get('polygon', '')
            ])

    def _write_pollen_results(self, video_id: str, frame_idx: int,
                              pred_pollen_balls: List[Dict], gt_pollen_balls: List[Dict],
                              writer, pred_chamber_masks=None, gt_chamber_masks=None):
        """Write one pollen CSV row per matched, predicted, or ground-truth pollen instance."""
        if not pred_pollen_balls and not gt_pollen_balls:
            return

        cost_matrix = np.zeros((len(pred_pollen_balls), len(gt_pollen_balls)))
        for i, pred_pollen in enumerate(pred_pollen_balls):
            for j, gt_pollen in enumerate(gt_pollen_balls):
                cost_matrix[i, j] = compute_mask_iou(pred_pollen['mask'], gt_pollen['mask'])

        matched_pairs, unmatched_preds, unmatched_gts = match_by_hungarian(
            cost_matrix, self.config['iou_threshold']
        )

        for pred_idx, gt_idx in matched_pairs:
            pred_pollen = pred_pollen_balls[pred_idx]
            gt_pollen = gt_pollen_balls[gt_idx]
            pred_mask = pred_pollen['mask']
            gt_mask = gt_pollen['mask']
            pred_chamber_id = self._hive_instance_chamber_id(pred_pollen, pred_chamber_masks)
            gt_chamber_id = self._hive_instance_chamber_id(gt_pollen, gt_chamber_masks or pred_chamber_masks)
            chamber_id = pred_chamber_id if pred_chamber_id != -1 else gt_chamber_id

            writer.writerow([
                video_id, frame_idx, chamber_id,
                'matched',
                pred_pollen.get('pollen_id', pred_idx + 1), gt_pollen.get('pollen_id', gt_idx + 1),
                int(np.sum(pred_mask > 0)), int(np.sum(gt_mask > 0)),
                cost_matrix[pred_idx, gt_idx],
                pred_pollen.get('polygon', ''),
                gt_pollen.get('polygon', '')
            ])

        for pred_idx in unmatched_preds:
            pred_pollen = pred_pollen_balls[pred_idx]
            pred_mask = pred_pollen['mask']

            writer.writerow([
                video_id, frame_idx, self._hive_instance_chamber_id(pred_pollen, pred_chamber_masks),
                'false_positive',
                pred_pollen.get('pollen_id', pred_idx + 1), '',
                int(np.sum(pred_mask > 0)), '', '',
                pred_pollen.get('polygon', ''), ''
            ])

        for gt_idx in unmatched_gts:
            gt_pollen = gt_pollen_balls[gt_idx]
            gt_mask = gt_pollen['mask']

            writer.writerow([
                video_id, frame_idx, self._hive_instance_chamber_id(gt_pollen, gt_chamber_masks or pred_chamber_masks),
                'false_negative',
                '', gt_pollen.get('pollen_id', gt_idx + 1),
                '', int(np.sum(gt_mask > 0)), '',
                '', gt_pollen.get('polygon', '')
            ])
    
    def _write_chamber_results(self, video_id: str, frame_idx: int,
                               pred_masks, gt_masks, writer):
        """Write chamber analysis to CSV"""
        if not pred_masks and not gt_masks:
            return
        
        pred_masks = pred_masks or {}
        gt_masks = gt_masks or {}
        
        # Build IoU cost matrix
        pred_ids = list(pred_masks.keys())
        gt_ids = list(gt_masks.keys())
        
        if not pred_ids and not gt_ids:
            return
        
        cost_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        for i, pred_id in enumerate(pred_ids):
            for j, gt_id in enumerate(gt_ids):
                cost_matrix[i, j] = compute_mask_iou(pred_masks[pred_id], gt_masks[gt_id])
        
        # Match chambers
        matched_pairs, unmatched_preds, unmatched_gts = match_by_hungarian(
            cost_matrix, self.config['iou_threshold']
        )
        
        # Write matched chambers
        for pred_idx, gt_idx in matched_pairs:
            pred_id = pred_ids[pred_idx]
            gt_id = gt_ids[gt_idx]
            iou = cost_matrix[pred_idx, gt_idx]
            pred_area = int(np.sum(pred_masks[pred_id] > 0))
            gt_area = int(np.sum(gt_masks[gt_id] > 0))
            
            writer.writerow([
                video_id, frame_idx, 'matched',
                pred_id, gt_id,
                pred_area, gt_area, iou,
                mask_to_polygons_string(pred_masks[pred_id], epsilon_percent=self._polygon_epsilon()),
                mask_to_polygons_string(gt_masks[gt_id], epsilon_percent=self._polygon_epsilon())
            ])
        
        # Write unmatched predictions (FP)
        for pred_idx in unmatched_preds:
            pred_id = pred_ids[pred_idx]
            pred_area = int(np.sum(pred_masks[pred_id] > 0))
            
            writer.writerow([
                video_id, frame_idx, 'false_positive',
                pred_id, '',
                pred_area, '', '',
                mask_to_polygons_string(pred_masks[pred_id], epsilon_percent=self._polygon_epsilon()),
                ''
            ])
        
        # Write unmatched GT (FN)
        for gt_idx in unmatched_gts:
            gt_id = gt_ids[gt_idx]
            gt_area = int(np.sum(gt_masks[gt_id] > 0))
            
            writer.writerow([
                video_id, frame_idx, 'false_negative',
                '', gt_id,
                '', gt_area, '',
                '',
                mask_to_polygons_string(gt_masks[gt_id], epsilon_percent=self._polygon_epsilon())
            ])
    
    def _parse_bumblebox_video_name(self, video_id: str):
        """Parse bumblebox video name to extract number and datetime.
        
        Format: bumblebox-XX_YYYY-MM-DD_HH_MM_SS
        Returns: (bumblebox_number, datetime_str) or ('', '') if not matching
        Datetime returned in ISO 8601 format: YYYY-MM-DD HH:MM:SS
        """
        import re
        match = re.match(r'bumblebox-(\d+)_(.+)', video_id)
        if match:
            bumblebox_num = match.group(1)
            datetime_raw = match.group(2)
            
            # Convert from YYYY-MM-DD_HH_MM_SS to YYYY-MM-DD HH:MM:SS
            # Replace first underscore with space, then remaining underscores with colons
            parts = datetime_raw.split('_', 1)
            if len(parts) == 2:
                date_part = parts[0]
                time_part = parts[1].replace('_', ':')
                datetime_formatted = f"{date_part} {time_part}"
                return bumblebox_num, datetime_formatted
            
            return bumblebox_num, datetime_raw
        return '', ''
    
    def _write_frame_summary(self, video_id: str, frame_idx: int,
                            pred_bees: List[Dict], gt_bees: List[Dict],
                            pred_hive_mask, gt_hive_mask,
                            pred_pollen_balls: List[Dict], gt_pollen_balls: List[Dict],
                            writer, pred_chamber_masks=None, gt_chamber_masks=None):
        """Write per-frame summary statistics to CSV, split by chamber if available"""
        
        # Parse video name for bumblebox videos
        bumblebox_number, datetime_str = self._parse_bumblebox_video_name(video_id)
        
        # Helper function to calculate statistics for a list of bees
        def pollen_pixels(pollen_balls):
            return int(sum(np.sum(pollen.get('mask') > 0) for pollen in pollen_balls if pollen.get('mask') is not None))

        def filter_pollen_to_chamber(pollen_balls, chamber_id, chamber_masks):
            if not chamber_masks:
                return pollen_balls
            return [
                pollen for pollen in pollen_balls
                if self._mask_in_chamber(pollen.get('mask'), chamber_id, chamber_masks)
            ]

        def calculate_bee_stats(bees, hive_mask, pollen_balls, chamber_masks=None, is_chamber_filtered=False,
                                fixed_chamber_id=None):
            count = len(bees)
            avg_hive_dist = None
            avg_pollen_dist = None
            avg_bee_dist = None
            avg_chamber_bee_dist = None
            
            # Calculate average hive distance
            if count > 0 and hive_mask is not None and np.any(hive_mask > 0):
                hive_dists = []
                for bee in bees:
                    bee_center = bee.get('centroid', (bee['bbox'][0], bee['bbox'][1]))
                    cid = fixed_chamber_id if fixed_chamber_id is not None else (
                        point_in_chamber(bee_center, chamber_masks) if chamber_masks else None
                    )
                    dist = self._calculate_chamber_hive_distance(bee, hive_mask, cid, chamber_masks)
                    if dist is not None and dist != np.inf:
                        hive_dists.append(dist)
                if hive_dists:
                    avg_hive_dist = np.mean(hive_dists)

            # Calculate average nearest pollen distance
            if count > 0 and pollen_balls:
                pollen_dists = []
                for bee in bees:
                    bee_center = bee.get('centroid', (bee['bbox'][0], bee['bbox'][1]))
                    cid = None if is_chamber_filtered else (point_in_chamber(bee_center, chamber_masks) if chamber_masks else None)
                    dist = self._calculate_pollen_distance(bee, pollen_balls, cid, chamber_masks)
                    if dist is not None and dist != np.inf:
                        pollen_dists.append(dist)
                if pollen_dists:
                    avg_pollen_dist = np.mean(pollen_dists)
            
            # Calculate average nearest bee distance (to nearest bee in same chamber)
            if count > 1 and chamber_masks:
                bee_dists = []
                for i in range(count):
                    bee_center = bees[i].get('centroid', (bees[i]['bbox'][0], bees[i]['bbox'][1]))
                    cid = point_in_chamber(bee_center, chamber_masks)
                    dist = self._calculate_nearest_bee_distance(i, bees, cid, chamber_masks)
                    if dist is not None:
                        bee_dists.append(dist)
                if bee_dists:
                    avg_bee_dist = np.mean(bee_dists)
            
            # Calculate average chamber bee distance
            if count > 1:
                if is_chamber_filtered:
                    # Bees are already filtered to one chamber, so avg chamber bee distance
                    # is the average distance between bees in this filtered list
                    chamber_dists = []
                    for i in range(count):
                        # Calculate average distance from this bee to other bees in list
                        bee_center = bees[i].get('centroid', (bees[i]['bbox'][0], bees[i]['bbox'][1]))
                        dists_to_others = []
                        for j in range(count):
                            if i != j:
                                other_center = bees[j].get('centroid', (bees[j]['bbox'][0], bees[j]['bbox'][1]))
                                dist = np.sqrt((bee_center[0] - other_center[0])**2 + (bee_center[1] - other_center[1])**2)
                                dists_to_others.append(dist)
                        if dists_to_others:
                            chamber_dists.append(np.mean(dists_to_others))
                    if chamber_dists:
                        avg_chamber_bee_dist = np.mean(chamber_dists)
                elif chamber_masks:
                    # For whole-frame: calculate avg chamber bee distance for each bee
                    chamber_dists = []
                    for i in range(count):
                        bee_center = bees[i].get('centroid', (bees[i]['bbox'][0], bees[i]['bbox'][1]))
                        chamber_id = point_in_chamber(bee_center, chamber_masks)
                        dist = self._calculate_avg_chamber_bee_distance(i, bees, chamber_id, chamber_masks)
                        if dist is not None:
                            chamber_dists.append(dist)
                    if chamber_dists:
                        avg_chamber_bee_dist = np.mean(chamber_dists)
            
            return count, avg_hive_dist, avg_pollen_dist, avg_bee_dist, avg_chamber_bee_dist
        
        # If we have chamber masks, write per-chamber rows
        if pred_chamber_masks and len(pred_chamber_masks) > 0:
            for chamber_id in range(1, len(pred_chamber_masks) + 1):
                # Filter pred bees to this chamber
                pred_chamber_bees = [
                    bee for bee in pred_bees
                    if point_in_chamber(bee.get('centroid', (bee['bbox'][0], bee['bbox'][1])), pred_chamber_masks) == chamber_id
                ]
                
                # Filter gt bees to this chamber
                gt_chamber_bees = []
                if gt_chamber_masks:
                    gt_chamber_bees = [
                        bee for bee in gt_bees
                        if point_in_chamber(bee.get('centroid', (bee['bbox'][0], bee['bbox'][1])), gt_chamber_masks) == chamber_id
                    ]
                else:
                    # If no GT chambers, use pred chambers for GT bees too
                    gt_chamber_bees = [
                        bee for bee in gt_bees
                        if point_in_chamber(bee.get('centroid', (bee['bbox'][0], bee['bbox'][1])), pred_chamber_masks) == chamber_id
                    ]

                pred_chamber_pollen = filter_pollen_to_chamber(pred_pollen_balls, chamber_id, pred_chamber_masks)
                gt_chamber_pollen = filter_pollen_to_chamber(
                    gt_pollen_balls, chamber_id, gt_chamber_masks or pred_chamber_masks
                )
                
                # Calculate statistics for this chamber
                pred_count, pred_avg_hive, pred_avg_pollen, pred_avg_bee, pred_avg_chamber = calculate_bee_stats(
                    pred_chamber_bees, pred_hive_mask, pred_chamber_pollen, pred_chamber_masks,
                    is_chamber_filtered=True, fixed_chamber_id=chamber_id)
                gt_count, gt_avg_hive, gt_avg_pollen, gt_avg_bee, gt_avg_chamber = calculate_bee_stats(
                    gt_chamber_bees, gt_hive_mask, gt_chamber_pollen, gt_chamber_masks or pred_chamber_masks,
                    is_chamber_filtered=True, fixed_chamber_id=chamber_id)
                
                # Compute hive pixels for this chamber
                chamber_mask = pred_chamber_masks[chamber_id]
                pred_hive_px = int(np.sum(pred_hive_mask[chamber_mask > 0] > 0)) if pred_hive_mask is not None else 0
                gt_hive_px = int(np.sum(gt_hive_mask[chamber_mask > 0] > 0)) if gt_hive_mask is not None else ''
                pred_pollen_px = pollen_pixels(pred_chamber_pollen)
                gt_pollen_px = pollen_pixels(gt_chamber_pollen)
                
                # Write chamber row
                writer.writerow([
                    video_id, frame_idx, chamber_id,
                    bumblebox_number, datetime_str,
                    pred_count, gt_count,
                    pred_hive_px,
                    gt_hive_px if gt_hive_px != '' else '',
                    len(pred_chamber_pollen), len(gt_chamber_pollen),
                    pred_pollen_px, gt_pollen_px,
                    pred_avg_hive if pred_avg_hive is not None else '',
                    gt_avg_hive if gt_avg_hive is not None else '',
                    pred_avg_pollen if pred_avg_pollen is not None else '',
                    gt_avg_pollen if gt_avg_pollen is not None else '',
                    pred_avg_bee if pred_avg_bee is not None else '',
                    gt_avg_bee if gt_avg_bee is not None else '',
                    pred_avg_chamber if pred_avg_chamber is not None else '',
                    gt_avg_chamber if gt_avg_chamber is not None else ''
                ])
        
        # Always write whole-frame summary row
        pred_count, pred_avg_hive, pred_avg_pollen, pred_avg_bee, pred_avg_chamber = calculate_bee_stats(
            pred_bees, pred_hive_mask, pred_pollen_balls, pred_chamber_masks, is_chamber_filtered=False)
        gt_count, gt_avg_hive, gt_avg_pollen, gt_avg_bee, gt_avg_chamber = calculate_bee_stats(
            gt_bees, gt_hive_mask, gt_pollen_balls, gt_chamber_masks, is_chamber_filtered=False)
        
        # Compute whole-frame hive pixels
        pred_hive_px = int(np.sum(pred_hive_mask > 0)) if pred_hive_mask is not None else 0
        gt_hive_px = int(np.sum(gt_hive_mask > 0)) if gt_hive_mask is not None else ''
        pred_pollen_px = pollen_pixels(pred_pollen_balls)
        gt_pollen_px = pollen_pixels(gt_pollen_balls)
        
        writer.writerow([
            video_id, frame_idx, -1,  # chamber_id -1 = whole-frame aggregate
            bumblebox_number, datetime_str,
            pred_count, gt_count,
            pred_hive_px,
            gt_hive_px if gt_hive_px != '' else '',
            len(pred_pollen_balls), len(gt_pollen_balls),
            pred_pollen_px, gt_pollen_px,
            pred_avg_hive if pred_avg_hive is not None else '',
            gt_avg_hive if gt_avg_hive is not None else '',
            pred_avg_pollen if pred_avg_pollen is not None else '',
            gt_avg_pollen if gt_avg_pollen is not None else '',
            pred_avg_bee if pred_avg_bee is not None else '',
            gt_avg_bee if gt_avg_bee is not None else '',
            pred_avg_chamber if pred_avg_chamber is not None else '',
            gt_avg_chamber if gt_avg_chamber is not None else ''
        ])
    
    def _generate_summary(self, results_folder: Path, num_videos: int, num_frames: int):
        """Generate human-readable summary file"""
        summary_path = results_folder / "summary.txt"
        map_metrics = self._compute_map_metrics()
        map_csv_path = results_folder / "map_metrics.csv"

        with open(map_csv_path, 'w', newline='') as map_csv:
            writer = csv.writer(map_csv)
            writer.writerow(['metric', 'iou_type', 'value'])
            writer.writerow(['mAP@0.5', map_metrics['IoU_type'], map_metrics['mAP@0.5']])
            writer.writerow(['mAP@0.5:0.95', map_metrics['IoU_type'], map_metrics['mAP@0.5:0.95']])
            for threshold, ap in map_metrics['AP_by_threshold'].items():
                writer.writerow([f'AP@{threshold:.2f}', map_metrics['IoU_type'], ap])
            if 'bbox_AP_by_threshold' in map_metrics:
                writer.writerow(['mAP@0.5', 'bbox_diagnostic', map_metrics['bbox_mAP@0.5']])
                writer.writerow(['mAP@0.5:0.95', 'bbox_diagnostic', map_metrics['bbox_mAP@0.5:0.95']])
                for threshold, ap in map_metrics['bbox_AP_by_threshold'].items():
                    writer.writerow([f'AP@{threshold:.2f}', 'bbox_diagnostic', ap])

            for key, value in map_metrics['IoU_diagnostics'].items():
                writer.writerow([key, map_metrics['IoU_type'], value])
            if 'bbox_IoU_diagnostics' in map_metrics:
                for key, value in map_metrics['bbox_IoU_diagnostics'].items():
                    writer.writerow([key, 'bbox_diagnostic', value])
            writer.writerow(['missing_pred_masks', map_metrics['IoU_type'], map_metrics['missing_pred_masks']])
            writer.writerow(['missing_gt_masks', map_metrics['IoU_type'], map_metrics['missing_gt_masks']])
            if 'mask_evaluable_gts' in map_metrics:
                writer.writerow(['mask_evaluable_gts', map_metrics['IoU_type'], map_metrics['mask_evaluable_gts']])
                writer.writerow([
                    'mask_ignored_bbox_only_predictions',
                    map_metrics['IoU_type'],
                    map_metrics['mask_ignored_bbox_only_preds']
                ])
        
        with open(summary_path, 'w') as f:
            f.write("Frame-Level Validation Analysis Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Note debug mode if enabled
            if self.config.get('debug_mode', False):
                f.write("Mode: DEBUG (first frame only)\n")
            
            f.write(f"Videos analyzed: {num_videos}\n")
            f.write(f"Frames analyzed: {num_frames}\n\n")
            
            f.write("Bee Detection Results:\n")
            f.write(f"  Matched (True Positives): {self.total_matches}\n")
            f.write(f"  False Positives: {self.total_fps}\n")
            f.write(f"  False Negatives: {self.total_fns}\n")
            f.write(f"  mAP@0.5 ({map_metrics['IoU_type']} IoU): {map_metrics['mAP@0.5']:.4f}\n")
            f.write(f"  mAP@0.5:0.95 ({map_metrics['IoU_type']} IoU): {map_metrics['mAP@0.5:0.95']:.4f}\n")
            iou_diag = map_metrics['IoU_diagnostics']
            f.write(
                f"  Mean best {map_metrics['IoU_type']} IoU per prediction: "
                f"{iou_diag['pred_best_iou_mean']:.4f}\n"
            )
            f.write(
                f"  Median best {map_metrics['IoU_type']} IoU per prediction: "
                f"{iou_diag['pred_best_iou_median']:.4f}\n"
            )
            f.write(
                f"  GT bees with best {map_metrics['IoU_type']} IoU >= 0.5: "
                f"{iou_diag['gt_best_iou_pct_ge_050'] * 100:.1f}%\n"
            )
            if map_metrics['IoU_type'] == 'segmentation':
                if 'mask_evaluable_gts' in map_metrics:
                    f.write(f"  GT bees with usable masks: {map_metrics['mask_evaluable_gts']}\n")
                f.write(f"  Missing predicted masks: {map_metrics['missing_pred_masks']}\n")
                f.write(f"  Missing GT masks: {map_metrics['missing_gt_masks']}\n")
                if 'mask_ignored_bbox_only_preds' in map_metrics:
                    f.write(
                        f"  BBox-only GT predictions ignored for segmentation AP: "
                        f"{map_metrics['mask_ignored_bbox_only_preds']}\n"
                    )
            if 'bbox_mAP@0.5' in map_metrics:
                bbox_diag = map_metrics['bbox_IoU_diagnostics']
                f.write(f"  Diagnostic bbox mAP@0.5: {map_metrics['bbox_mAP@0.5']:.4f}\n")
                f.write(f"  Diagnostic bbox mAP@0.5:0.95: {map_metrics['bbox_mAP@0.5:0.95']:.4f}\n")
                f.write(f"  Diagnostic mean best bbox IoU per prediction: {bbox_diag['pred_best_iou_mean']:.4f}\n")
                f.write(f"  Diagnostic GT bees with best bbox IoU >= 0.5: {bbox_diag['gt_best_iou_pct_ge_050'] * 100:.1f}%\n")
            
            total_gt = self.total_matches + self.total_fns
            total_pred = self.total_matches + self.total_fps
            
            if total_gt > 0:
                recall = self.total_matches / total_gt
                f.write(f"  Recall: {recall:.3f}\n")
            
            if total_pred > 0:
                precision = self.total_matches / total_pred
                f.write(f"  Precision: {precision:.3f}\n")
            
            if total_gt > 0 and total_pred > 0:
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f.write(f"  F1 Score: {f1:.3f}\n")
            
            f.write("\n")
            f.write("Output Files:\n")
            f.write("  - bee_detections.csv: Detailed bee-level predictions and matches\n")
            f.write("  - frame_summary.csv: Per-frame statistics (bee counts, average distances)\n")
            f.write("  - map_metrics.csv: Bee mAP@0.5, mAP@0.5:0.95, and per-threshold AP\n")
            if self.config['hive_model_path']:
                f.write("  - hive_detections.csv: Hive segmentation analysis\n")
            if self.config.get('pollen_model_path'):
                f.write("  - pollen_detections.csv: Pollen segmentation analysis\n")
            if self.config['chamber_model_path']:
                f.write("  - chamber_detections.csv: Chamber segmentation analysis\n")
            if self.config.get('save_visualizations', False):
                f.write("  - visualizations/: Images with color-coded boxes, hive/chamber/pollen outlines, and distance links\n")
        
        self.log_message.emit(f"\nSummary:")
        self.log_message.emit(f"  Matched: {self.total_matches}")
        self.log_message.emit(f"  False Positives: {self.total_fps}")
        self.log_message.emit(f"  False Negatives: {self.total_fns}")
        self.log_message.emit(f"  mAP@0.5 ({map_metrics['IoU_type']} IoU): {map_metrics['mAP@0.5']:.4f}")
        self.log_message.emit(f"  mAP@0.5:0.95 ({map_metrics['IoU_type']} IoU): {map_metrics['mAP@0.5:0.95']:.4f}")
        iou_diag = map_metrics['IoU_diagnostics']
        self.log_message.emit(
            f"  Mean best {map_metrics['IoU_type']} IoU per prediction: "
            f"{iou_diag['pred_best_iou_mean']:.4f}"
        )
        self.log_message.emit(
            f"  GT bees with best {map_metrics['IoU_type']} IoU >= 0.5: "
            f"{iou_diag['gt_best_iou_pct_ge_050'] * 100:.1f}%"
        )
        if map_metrics['IoU_type'] == 'segmentation':
            if 'mask_evaluable_gts' in map_metrics:
                self.log_message.emit(f"  GT bees with usable masks: {map_metrics['mask_evaluable_gts']}")
            self.log_message.emit(f"  Missing predicted masks: {map_metrics['missing_pred_masks']}")
            self.log_message.emit(f"  Missing GT masks: {map_metrics['missing_gt_masks']}")
            if 'mask_ignored_bbox_only_preds' in map_metrics:
                self.log_message.emit(
                    f"  BBox-only GT predictions ignored for segmentation AP: "
                    f"{map_metrics['mask_ignored_bbox_only_preds']}"
                )
        if 'bbox_mAP@0.5' in map_metrics:
            bbox_diag = map_metrics['bbox_IoU_diagnostics']
            self.log_message.emit(f"  Diagnostic bbox mAP@0.5: {map_metrics['bbox_mAP@0.5']:.4f}")
            self.log_message.emit(f"  Diagnostic bbox mAP@0.5:0.95: {map_metrics['bbox_mAP@0.5:0.95']:.4f}")
            self.log_message.emit(f"  Diagnostic mean best bbox IoU per prediction: {bbox_diag['pred_best_iou_mean']:.4f}")
