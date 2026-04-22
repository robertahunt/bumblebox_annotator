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
    point_in_chamber, match_by_hungarian, bbox_from_mask,
    extract_instance_masks
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
        
        # Timing
        self.start_time = None
        
    def stop(self):
        """Request worker to stop"""
        self.should_stop = True
    
    def run(self):
        """Run validation analysis in background thread"""
        try:
            self.log_message.emit("=== Frame-Level Validation Analysis ===")
            self.log_message.emit(f"BBox Model: {Path(self.config['bbox_model_path']).name}")
            if self.config['hive_model_path']:
                self.log_message.emit(f"Hive Model: {Path(self.config['hive_model_path']).name}")
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
            hive_model = YOLO(self.config['hive_model_path']) if self.config['hive_model_path'] else None
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
                'pred_nearest_bee_distance', 'gt_nearest_bee_distance',
                'pred_avg_chamber_bee_distance', 'gt_avg_chamber_bee_distance'
            ])
            
            summary_csv = open(summary_csv_path, 'w', newline='')
            summary_writer = csv.writer(summary_csv)
            summary_writer.writerow([
                'video_id', 'frame_idx', 'chamber_id',
                'bumblebox_number', 'datetime',
                'pred_bee_count', 'gt_bee_count',
                'pred_avg_hive_distance', 'gt_avg_hive_distance',
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
                    'pred_hive_pixels', 'gt_hive_pixels', 'hive_iou'
                ])
            
            chamber_csv = None
            chamber_writer = None
            if chamber_model:
                chamber_csv = open(chamber_csv_path, 'w', newline='')
                chamber_writer = csv.writer(chamber_csv)
                chamber_writer.writerow([
                    'video_id', 'frame_idx', 'match_status',
                    'pred_chamber_id', 'gt_chamber_id',
                    'pred_area_pixels', 'gt_area_pixels', 'chamber_iou'
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
                        bbox_model, hive_model, chamber_model,
                        bee_writer, hive_writer, chamber_writer,
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
            if chamber_csv:
                chamber_csv.close()
            
            if self.should_stop:
                self.log_message.emit("Analysis cancelled by user")
                return
            
            # Generate summary
            self._generate_summary(results_folder, len(validation_frames), total_frames)
            
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
                      bbox_model, hive_model, chamber_model,
                      bee_writer, hive_writer, chamber_writer,
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
            pred_bees = self._predict_bees(frame_rgb, bbox_model)
            gt_bees = self._extract_gt_bees_from_coco(coco_data, frame_idx)
            
            # Always extract GT hive mask for distance calculations (even if no hive model)
            gt_hive_mask = self._extract_gt_hive(video_annotations, frame.shape[:2])
            # Run hive prediction if model available
            pred_hive_mask = self._predict_hive(frame_rgb, hive_model) if hive_model else None
            
            # Always extract GT chamber masks for chamber ID (even if no chamber model)
            gt_chamber_masks = self._extract_gt_chambers(video_annotations, frame.shape[:2])
            # Run chamber prediction if model available
            pred_chamber_masks = self._predict_chambers(frame_rgb, chamber_model) if chamber_model else None
            
            # Match bees and get matching info for visualization
            matched_pairs, unmatched_preds, unmatched_gts = self._match_and_write_bees(
                video_id, frame_idx, pred_bees, gt_bees,
                pred_hive_mask, gt_hive_mask,
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
                    pred_chamber_masks, gt_chamber_masks
                )
            
            # Write hive results
            if hive_writer and pred_hive_mask is not None:
                self._write_hive_results(video_id, frame_idx, pred_hive_mask, gt_hive_mask, 
                                        hive_writer, pred_chamber_masks)
            
            # Write chamber results
            if chamber_writer and pred_chamber_masks is not None:
                self._write_chamber_results(video_id, frame_idx, pred_chamber_masks, gt_chamber_masks, chamber_writer)
            
            # Write frame summary
            self._write_frame_summary(
                video_id, frame_idx,
                pred_bees, gt_bees,
                pred_hive_mask, gt_hive_mask,
                summary_writer,
                pred_chamber_masks, gt_chamber_masks
            )
            
        except Exception as e:
            import traceback
            self.log_message.emit(f"  Error processing {video_id} frame {frame_idx}: {str(e)}")
            traceback.print_exc()
    
    def _predict_bees(self, frame: np.ndarray, model) -> List[Dict]:
        """Run YOLO bbox prediction for bees"""
        results = model.predict(
            frame,
            conf=self.config['conf_threshold'],
            verbose=False
        )
        
        bees = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                # Filter by class ID - only include bee class (class_id=0 for bees)
                # In YOLO models, if trained with multiple classes, we need to filter
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
                    'confidence': conf
                })
        
        return bees
    
    def _extract_gt_bees_from_coco(self, coco_data: Dict, frame_idx: int) -> List[Dict]:
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
                    
                    bees.append({
                        'bbox': [cx, cy, w, h],
                        'instance_id': instance_id
                    })
        
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
                    'instance_id': instance_id
                })
        
        return bees
    
    def _predict_hive(self, frame: np.ndarray, model) -> Optional[np.ndarray]:
        """Run YOLO segmentation prediction for hive"""
        if model is None:
            return None
        
        results = model.predict(
            frame,
            conf=self.config['conf_threshold'],
            verbose=False
        )
        
        # Combine all hive masks
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            h, w = frame.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            for mask in masks:
                # Resize mask to frame size
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8) * 255)
            
            return combined_mask
        
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    
    def _extract_gt_hive(self, video_annotations: List[Dict], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract ground truth hive mask from video-level annotations"""
        h, w = frame_shape
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for ann in video_annotations:
            if ann.get('category', '') == 'hive' or ann.get('category_id', -1) == 2:
                if 'mask' in ann and ann['mask'] is not None:
                    # Resize mask to frame size if necessary
                    mask = ann['mask']
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    combined_mask = np.maximum(combined_mask, mask)
        
        return combined_mask
    
    def _predict_chambers(self, frame: np.ndarray, model) -> Optional[Dict[int, np.ndarray]]:
        """Run YOLO segmentation prediction for chambers"""
        if model is None:
            return None
        
        results = model.predict(
            frame,
            conf=self.config['conf_threshold'],
            verbose=False
        )
        
        chamber_masks = {}
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            h, w = frame.shape[:2]
            
            # Store masks with their x-coordinates for sorting
            chambers_with_coords = []
            for i, mask in enumerate(masks):
                # Resize mask to frame size
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                chamber_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                
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
    
    def _calculate_nearest_bee_distance(self, bee_idx: int, all_bees: List[Dict]) -> Optional[float]:
        """Calculate distance from bee at bee_idx to its nearest neighbor bee"""
        if len(all_bees) <= 1:
            return None  # No other bee to measure distance to
        
        bee_center = (all_bees[bee_idx]['bbox'][0], all_bees[bee_idx]['bbox'][1])
        min_dist = np.inf
        
        for i, other_bee in enumerate(all_bees):
            if i == bee_idx:
                continue  # Skip self
            other_center = (other_bee['bbox'][0], other_bee['bbox'][1])
            dist = np.sqrt((bee_center[0] - other_center[0])**2 + (bee_center[1] - other_center[1])**2)
            min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != np.inf else None
    
    def _calculate_avg_chamber_bee_distance(self, bee_idx: int, all_bees: List[Dict], 
                                           chamber_id: Optional[int], chamber_masks) -> Optional[float]:
        """Calculate average distance from bee to other bees in the same chamber"""
        if chamber_id is None or not chamber_masks:
            return None  # No chamber information
        
        if len(all_bees) <= 1:
            return None  # No other bees
        
        bee_center = (all_bees[bee_idx]['bbox'][0], all_bees[bee_idx]['bbox'][1])
        chamber_distances = []
        
        for i, other_bee in enumerate(all_bees):
            if i == bee_idx:
                continue  # Skip self
            
            # Check if other bee is in the same chamber
            other_center = (other_bee['bbox'][0], other_bee['bbox'][1])
            other_chamber_id = point_in_chamber(other_center, chamber_masks)
            
            if other_chamber_id == chamber_id:
                # Calculate distance to this chamber mate
                dist = np.sqrt((bee_center[0] - other_center[0])**2 + (bee_center[1] - other_center[1])**2)
                chamber_distances.append(dist)
        
        # Return average distance to chamber mates
        if chamber_distances:
            return np.mean(chamber_distances)
        return None  # No other bees in same chamber
    
    def _match_and_write_bees(self, video_id: str, frame_idx: int,
                              pred_bees: List[Dict], gt_bees: List[Dict],
                              pred_hive_mask, gt_hive_mask,
                              pred_chamber_masks, gt_chamber_masks,
                              writer):
        """Match predictions to GT and write to CSV. Returns matching info for visualization."""
        # Build IoU cost matrix
        n_preds = len(pred_bees)
        n_gts = len(gt_bees)
        
        if n_preds == 0 and n_gts == 0:
            return [], [], []  # Nothing to match
        
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
            pred_chamber_id = point_in_chamber((pred['bbox'][0], pred['bbox'][1]), pred_chamber_masks) if pred_chamber_masks else None
            gt_chamber_id = point_in_chamber((gt['bbox'][0], gt['bbox'][1]), gt_chamber_masks) if gt_chamber_masks else None
            
            # Get hive distances
            pred_hive_dist = None
            if pred_hive_mask is not None and np.any(pred_hive_mask > 0):
                pred_hive_dist = distance_to_mask((pred['bbox'][0], pred['bbox'][1]), pred_hive_mask)
            
            gt_hive_dist = None
            if gt_hive_mask is not None and np.any(gt_hive_mask > 0):
                gt_hive_dist = distance_to_mask((gt['bbox'][0], gt['bbox'][1]), gt_hive_mask)
            
            # Get nearest bee distances
            pred_nearest_dist = self._calculate_nearest_bee_distance(pred_idx, pred_bees)
            gt_nearest_dist = self._calculate_nearest_bee_distance(gt_idx, gt_bees)
            
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
                pred_nearest_dist if pred_nearest_dist is not None else '',
                gt_nearest_dist if gt_nearest_dist is not None else '',
                pred_chamber_dist if pred_chamber_dist is not None else '',
                gt_chamber_dist if gt_chamber_dist is not None else ''
            ])
        
        # Write false positives (unmatched predictions)
        for pred_idx in unmatched_preds:
            pred = pred_bees[pred_idx]
            pred_chamber_id = point_in_chamber((pred['bbox'][0], pred['bbox'][1]), pred_chamber_masks) if pred_chamber_masks else None
            
            pred_hive_dist = None
            if pred_hive_mask is not None and np.any(pred_hive_mask > 0):
                pred_hive_dist = distance_to_mask((pred['bbox'][0], pred['bbox'][1]), pred_hive_mask)
            
            # Get nearest bee distance
            pred_nearest_dist = self._calculate_nearest_bee_distance(pred_idx, pred_bees)
            
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
                pred_nearest_dist if pred_nearest_dist is not None else '',
                '',  # No GT bee distance
                pred_chamber_dist if pred_chamber_dist is not None else '',
                ''  # No GT chamber bee distance
            ])
        
        # Write false negatives (unmatched GT)
        for gt_idx in unmatched_gts:
            gt = gt_bees[gt_idx]
            gt_chamber_id = point_in_chamber((gt['bbox'][0], gt['bbox'][1]), gt_chamber_masks) if gt_chamber_masks else None
            
            gt_hive_dist = None
            if gt_hive_mask is not None and np.any(gt_hive_mask > 0):
                gt_hive_dist = distance_to_mask((gt['bbox'][0], gt['bbox'][1]), gt_hive_mask)
            
            # Get nearest bee distance
            gt_nearest_dist = self._calculate_nearest_bee_distance(gt_idx, gt_bees)
            
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
                '',  # No pred bee distance
                gt_nearest_dist if gt_nearest_dist is not None else '',
                '',  # No pred chamber bee distance
                gt_chamber_dist if gt_chamber_dist is not None else ''
            ])
        
        return matched_pairs, unmatched_preds, unmatched_gts
    
    def _save_visualization(self, frame: np.ndarray, video_id: str, frame_idx: int,
                           pred_bees: List[Dict], gt_bees: List[Dict],
                           matched_pairs: List[Tuple[int, int]],
                           unmatched_preds: List[int],
                           unmatched_gts: List[int],
                           results_folder: Path,
                           pred_hive_mask=None, gt_hive_mask=None,
                           pred_chamber_masks=None, gt_chamber_masks=None):
        """Save visualization with color-coded bee boxes and hive/chamber outlines"""
        try:
            # Create visualizations folder
            vis_folder = results_folder / "visualizations" / video_id
            vis_folder.mkdir(parents=True, exist_ok=True)
            
            # Create single image visualization
            vis_img = frame.copy()
            
            # Draw hive masks as yellow outlines
            hive_color = (0, 255, 255)  # Yellow in BGR
            if gt_hive_mask is not None and np.any(gt_hive_mask > 0):
                contours, _ = cv2.findContours(gt_hive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, hive_color, 3)
            
            if pred_hive_mask is not None and np.any(pred_hive_mask > 0):
                contours, _ = cv2.findContours(pred_hive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, hive_color, 3)
            
            # Draw chamber masks as blue outlines
            chamber_color = (255, 0, 0)  # Blue in BGR
            
            if gt_chamber_masks:
                for chamber_id, mask in gt_chamber_masks.items():
                    if np.any(mask > 0):
                        mask_uint8 = mask.astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis_img, contours, -1, chamber_color, 3)
            
            if pred_chamber_masks:
                for chamber_id, mask in pred_chamber_masks.items():
                    if np.any(mask > 0):
                        mask_uint8 = mask.astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis_img, contours, -1, chamber_color, 3)
                        
                        # Draw chamber ID label at bottom-center of chamber
                        cx, cy, w, h = bbox_from_mask(mask_uint8)
                        label_x = int(cx)
                        label_y = int(cy + h/2 - 10)  # Bottom of chamber
                        cv2.putText(vis_img, f"C{chamber_id}", (label_x, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, chamber_color, 3)
            
            # Draw false negatives (GT with no match) in RED
            for gt_idx in unmatched_gts:
                gt = gt_bees[gt_idx]
                cx, cy, bw, bh = gt['bbox']
                x1, y1 = int(cx - bw/2), int(cy - bh/2)
                x2, y2 = int(cx + bw/2), int(cy + bh/2)
                
                # RED for false negatives
                color = (0, 0, 255)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                
                # Draw GT instance ID
                instance_id = gt.get('instance_id', -1)
                label = f"FN:{instance_id}"
                cv2.putText(vis_img, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw false positives (prediction with no GT match) in PINK
            for pred_idx in unmatched_preds:
                pred = pred_bees[pred_idx]
                cx, cy, bw, bh = pred['bbox']
                x1, y1 = int(cx - bw/2), int(cy - bh/2)
                x2, y2 = int(cx + bw/2), int(cy + bh/2)
                
                # PINK for false positives (light magenta in BGR)
                color = (203, 192, 255)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                conf = pred.get('confidence', 0)
                label = f"FP ({conf:.2f})"
                cv2.putText(vis_img, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw matched pairs (correct predictions) in GREEN
            for pred_idx, gt_idx in matched_pairs:
                pred = pred_bees[pred_idx]
                gt = gt_bees[gt_idx]
                cx, cy, bw, bh = pred['bbox']
                x1, y1 = int(cx - bw/2), int(cy - bh/2)
                x2, y2 = int(cx + bw/2), int(cy + bh/2)
                
                # GREEN for correct matches
                color = (0, 255, 0)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                
                # Draw GT instance ID and confidence
                instance_id = gt.get('instance_id', -1)
                conf = pred.get('confidence', 0)
                label = f"ID:{instance_id} ({conf:.2f})"
                cv2.putText(vis_img, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
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
            if (gt_hive_mask is not None and np.any(gt_hive_mask > 0)) or \
               (pred_hive_mask is not None and np.any(pred_hive_mask > 0)):
                cv2.putText(vis_img, "Yellow = Hive", (10, legend_y + 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if (gt_chamber_masks and any(np.any(m > 0) for m in gt_chamber_masks.values())) or \
               (pred_chamber_masks and any(np.any(m > 0) for m in pred_chamber_masks.values())):
                legend_offset = 150 if ((gt_hive_mask is not None and np.any(gt_hive_mask > 0)) or \
                               (pred_hive_mask is not None and np.any(pred_hive_mask > 0))) else 120
                cv2.putText(vis_img, "Blue = Chambers", (10, legend_y + legend_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
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
    
    def _write_hive_results(self, video_id: str, frame_idx: int,
                           pred_mask, gt_mask, writer, pred_chamber_masks=None):
        """Write hive analysis to CSV, split by chamber if available"""
        
        # If we have chamber masks, write one row per chamber
        if pred_chamber_masks and len(pred_chamber_masks) > 0:
            for chamber_id, chamber_mask in enumerate(pred_chamber_masks, start=1):
                # Mask the hive to only this chamber's region
                pred_masked = None
                if pred_mask is not None:
                    pred_masked = pred_mask.copy()
                    pred_masked[chamber_mask == 0] = 0
                
                gt_masked = None
                if gt_mask is not None:
                    gt_masked = gt_mask.copy()
                    gt_masked[chamber_mask == 0] = 0
                
                # Calculate metrics for this chamber
                pred_pixels = int(np.sum(pred_masked > 0)) if pred_masked is not None else 0
                
                # Leave gt_pixels and iou blank if no ground truth
                gt_pixels = ''
                iou = ''
                if gt_masked is not None and np.any(gt_masked > 0):
                    gt_pixels = int(np.sum(gt_masked > 0))
                    if pred_masked is not None:
                        iou = compute_mask_iou(pred_masked, gt_masked)
                
                writer.writerow([
                    video_id, frame_idx, chamber_id,
                    pred_pixels, gt_pixels, iou
                ])
        else:
            # No chamber masks available, write single row for whole frame
            pred_pixels = int(np.sum(pred_mask > 0)) if pred_mask is not None else 0
            
            # Leave gt_pixels and iou blank if no ground truth
            gt_pixels = ''
            iou = ''
            if gt_mask is not None and np.any(gt_mask > 0):
                gt_pixels = int(np.sum(gt_mask > 0))
                if pred_mask is not None:
                    iou = compute_mask_iou(pred_mask, gt_mask)
            
            writer.writerow([
                video_id, frame_idx, '',  # Empty chamber_id for whole-frame
                pred_pixels, gt_pixels, iou
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
                pred_area, gt_area, iou
            ])
        
        # Write unmatched predictions (FP)
        for pred_idx in unmatched_preds:
            pred_id = pred_ids[pred_idx]
            pred_area = int(np.sum(pred_masks[pred_id] > 0))
            
            writer.writerow([
                video_id, frame_idx, 'false_positive',
                pred_id, '',
                pred_area, '', ''
            ])
        
        # Write unmatched GT (FN)
        for gt_idx in unmatched_gts:
            gt_id = gt_ids[gt_idx]
            gt_area = int(np.sum(gt_masks[gt_id] > 0))
            
            writer.writerow([
                video_id, frame_idx, 'false_negative',
                '', gt_id,
                '', gt_area, ''
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
                            writer, pred_chamber_masks=None, gt_chamber_masks=None):
        """Write per-frame summary statistics to CSV, split by chamber if available"""
        
        # Parse video name for bumblebox videos
        bumblebox_number, datetime_str = self._parse_bumblebox_video_name(video_id)
        
        # Helper function to calculate statistics for a list of bees
        def calculate_bee_stats(bees, hive_mask, chamber_masks=None, is_chamber_filtered=False):
            count = len(bees)
            avg_hive_dist = None
            avg_bee_dist = None
            avg_chamber_bee_dist = None
            
            # Calculate average hive distance
            if count > 0 and hive_mask is not None and np.any(hive_mask > 0):
                hive_dists = []
                for bee in bees:
                    dist = distance_to_mask((bee['bbox'][0], bee['bbox'][1]), hive_mask)
                    if dist != np.inf:
                        hive_dists.append(dist)
                if hive_dists:
                    avg_hive_dist = np.mean(hive_dists)
            
            # Calculate average nearest bee distance (to any bee)
            if count > 1:
                bee_dists = []
                for i in range(count):
                    dist = self._calculate_nearest_bee_distance(i, bees)
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
                        bee_center = (bees[i]['bbox'][0], bees[i]['bbox'][1])
                        dists_to_others = []
                        for j in range(count):
                            if i != j:
                                other_center = (bees[j]['bbox'][0], bees[j]['bbox'][1])
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
                        bee_center = (bees[i]['bbox'][0], bees[i]['bbox'][1])
                        chamber_id = point_in_chamber(bee_center, chamber_masks)
                        dist = self._calculate_avg_chamber_bee_distance(i, bees, chamber_id, chamber_masks)
                        if dist is not None:
                            chamber_dists.append(dist)
                    if chamber_dists:
                        avg_chamber_bee_dist = np.mean(chamber_dists)
            
            return count, avg_hive_dist, avg_bee_dist, avg_chamber_bee_dist
        
        # If we have chamber masks, write per-chamber rows
        if pred_chamber_masks and len(pred_chamber_masks) > 0:
            for chamber_id in range(1, len(pred_chamber_masks) + 1):
                # Filter pred bees to this chamber
                pred_chamber_bees = [
                    bee for bee in pred_bees
                    if point_in_chamber((bee['bbox'][0], bee['bbox'][1]), pred_chamber_masks) == chamber_id
                ]
                
                # Filter gt bees to this chamber
                gt_chamber_bees = []
                if gt_chamber_masks:
                    gt_chamber_bees = [
                        bee for bee in gt_bees
                        if point_in_chamber((bee['bbox'][0], bee['bbox'][1]), gt_chamber_masks) == chamber_id
                    ]
                else:
                    # If no GT chambers, use pred chambers for GT bees too
                    gt_chamber_bees = [
                        bee for bee in gt_bees
                        if point_in_chamber((bee['bbox'][0], bee['bbox'][1]), pred_chamber_masks) == chamber_id
                    ]
                
                # Calculate statistics for this chamber
                pred_count, pred_avg_hive, pred_avg_bee, pred_avg_chamber = calculate_bee_stats(
                    pred_chamber_bees, pred_hive_mask, pred_chamber_masks, is_chamber_filtered=True)
                gt_count, gt_avg_hive, gt_avg_bee, gt_avg_chamber = calculate_bee_stats(
                    gt_chamber_bees, gt_hive_mask, gt_chamber_masks, is_chamber_filtered=True)
                
                # Write chamber row
                writer.writerow([
                    video_id, frame_idx, chamber_id,
                    bumblebox_number, datetime_str,
                    pred_count, gt_count,
                    pred_avg_hive if pred_avg_hive is not None else '',
                    gt_avg_hive if gt_avg_hive is not None else '',
                    pred_avg_bee if pred_avg_bee is not None else '',
                    gt_avg_bee if gt_avg_bee is not None else '',
                    pred_avg_chamber if pred_avg_chamber is not None else '',
                    gt_avg_chamber if gt_avg_chamber is not None else ''
                ])
        
        # Always write whole-frame summary row
        pred_count, pred_avg_hive, pred_avg_bee, pred_avg_chamber = calculate_bee_stats(
            pred_bees, pred_hive_mask, pred_chamber_masks, is_chamber_filtered=False)
        gt_count, gt_avg_hive, gt_avg_bee, gt_avg_chamber = calculate_bee_stats(
            gt_bees, gt_hive_mask, gt_chamber_masks, is_chamber_filtered=False)
        
        writer.writerow([
            video_id, frame_idx, '',  # Empty chamber_id for whole-frame
            bumblebox_number, datetime_str,
            pred_count, gt_count,
            pred_avg_hive if pred_avg_hive is not None else '',
            gt_avg_hive if gt_avg_hive is not None else '',
            pred_avg_bee if pred_avg_bee is not None else '',
            gt_avg_bee if gt_avg_bee is not None else '',
            pred_avg_chamber if pred_avg_chamber is not None else '',
            gt_avg_chamber if gt_avg_chamber is not None else ''
        ])
    
    def _generate_summary(self, results_folder: Path, num_videos: int, num_frames: int):
        """Generate human-readable summary file"""
        summary_path = results_folder / "summary.txt"
        
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
            if self.config['hive_model_path']:
                f.write("  - hive_detections.csv: Hive segmentation analysis\n")
            if self.config['chamber_model_path']:
                f.write("  - chamber_detections.csv: Chamber segmentation analysis\n")
            if self.config.get('save_visualizations', False):
                f.write("  - visualizations/: Images with color-coded boxes (Green=Matched, Red=FN, Pink=FP) and hive/chamber outlines (Yellow=Hive, Blue=Chambers)\n")
        
        self.log_message.emit(f"\nSummary:")
        self.log_message.emit(f"  Matched: {self.total_matches}")
        self.log_message.emit(f"  False Positives: {self.total_fps}")
        self.log_message.emit(f"  False Negatives: {self.total_fns}")
