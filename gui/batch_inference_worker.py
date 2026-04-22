"""
Worker thread for batch image inference
"""

import csv
import json
import random
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from PyQt6.QtCore import QThread, pyqtSignal

from utils.validation_metrics import (
    distance_to_mask, point_in_chamber, bbox_from_mask,
    mask_centroid, mask_to_simplified_polygon, polygon_to_string,
    distance_between_masks
)


class BatchInferenceWorker(QThread):
    """Worker thread for batch image inference"""
    
    # Signals
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)  # current, total
    time_remaining_updated = pyqtSignal(str)
    stats_updated = pyqtSignal(int, int, int)  # bees, chambers, hives
    log_message = pyqtSignal(str)
    finished = pyqtSignal(bool)  # success
    
    def __init__(self, config: Dict, project_path: Path):
        super().__init__()
        self.config = config
        self.project_path = project_path
        self.should_stop = False
        self.start_time = None
        
        # Statistics
        self.total_bees = 0
        self.total_chambers = 0
        self.total_hives = 0
        
    def run(self):
        """Main execution"""
        try:
            self.start_time = datetime.now()
            
            # Create output folder with model type subfolder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = self.config.get('bee_model_type', 'bbox')
            results_folder = self.project_path / "batch_inference" / model_type / timestamp
            results_folder.mkdir(parents=True, exist_ok=True)
            
            self.log_message.emit(f"Output folder: {results_folder}")
            
            # Save configuration
            config_path = results_folder / "config.json"
            config_save = {
                'input_folder': str(self.config['input_folder']),
                'bbox_model': str(self.config['bbox_model']),
                'bee_model_type': self.config.get('bee_model_type', 'bbox'),
                'distance_method': self.config.get('distance_method', 'bbox_filter'),
                'hive_model': str(self.config['hive_model']) if self.config['hive_model'] else None,
                'chamber_model': str(self.config['chamber_model']) if self.config['chamber_model'] else None,
                'conf_threshold': self.config['conf_threshold'],
                'save_annotations': self.config['save_annotations'],
                'save_visualizations': self.config['save_visualizations'],
                'debug_mode': self.config['debug_mode'],
                'timestamp': timestamp
            }
            with open(config_path, 'w') as f:
                json.dump(config_save, f, indent=2)
            
            # Discover images
            self.status_updated.emit("Discovering images...")
            self.log_message.emit("Searching for PNG images...")
            
            image_paths = self._discover_images(self.config['input_folder'])
            
            if not image_paths:
                self.log_message.emit("No PNG images found!")
                self.finished.emit(False)
                return
            
            if self.config['debug_mode']:
                image_paths = image_paths[:1]
                self.log_message.emit(f"Debug mode: processing only first image")
            else:
                # Shuffle images for processing variety
                random.shuffle(image_paths)
                self.log_message.emit(f"Shuffled image processing order for variety")
            
            self.log_message.emit(f"Found {len(image_paths)} image(s) to process")
            
            # Load models
            self.status_updated.emit("Loading models...")
            self.log_message.emit("Loading YOLO models...")
            
            from ultralytics import YOLO
            
            bbox_model = YOLO(str(self.config['bbox_model']))
            self.log_message.emit(f"✓ Loaded bee detection model: {self.config['bbox_model'].name}")
            
            hive_model = None
            if self.config['hive_model']:
                hive_model = YOLO(str(self.config['hive_model']))
                self.log_message.emit(f"✓ Loaded hive segmentation model: {self.config['hive_model'].name}")
            
            chamber_model = None
            if self.config['chamber_model']:
                chamber_model = YOLO(str(self.config['chamber_model']))
                self.log_message.emit(f"✓ Loaded chamber segmentation model: {self.config['chamber_model'].name}")
            
            # Open CSV files
            bee_csv_path = results_folder / "bee_detections.csv"
            hive_csv_path = results_folder / "hive_detections.csv"
            chamber_csv_path = results_folder / "chamber_detections.csv"
            summary_csv_path = results_folder / "image_summary.csv"
            
            bee_csv = open(bee_csv_path, 'w', newline='')
            bee_writer = csv.writer(bee_csv)
            bee_writer.writerow([
                'image_path',
                'instance_id',
                'pred_bbox_center_x', 'pred_bbox_center_y', 'pred_bbox_width', 'pred_bbox_height',
                'pred_centroid_x', 'pred_centroid_y',
                'pred_polygon',
                'pred_confidence',
                'pred_chamber_id', 'pred_hive_distance',
                'pred_nearest_bee_distance', 'pred_avg_chamber_bee_distance'
            ])
            bee_csv.flush()  # Ensure header is written immediately
            
            summary_csv = open(summary_csv_path, 'w', newline='')
            summary_writer = csv.writer(summary_csv)
            summary_writer.writerow([
                'image_path', 'chamber_id',
                'bumblebox_number', 'datetime',
                'pred_bee_count', 
                'pred_avg_hive_distance', 
                'pred_avg_nearest_bee_distance',
                'pred_avg_chamber_bee_distance'
            ])
            summary_csv.flush()  # Ensure header is written immediately
            
            hive_csv = None
            hive_writer = None
            if hive_model:
                self.log_message.emit("Creating hive_detections.csv with headers...")
                hive_csv = open(hive_csv_path, 'w', newline='')
                hive_writer = csv.writer(hive_csv)
                hive_writer.writerow([
                    'image_path', 'chamber_id', 'pred_hive_pixels'
                ])
                hive_csv.flush()  # Ensure header is written immediately
                self.log_message.emit("✓ Hive CSV created with headers")
            
            chamber_csv = None
            chamber_writer = None
            if chamber_model:
                self.log_message.emit("Creating chamber_detections.csv with headers...")
                chamber_csv = open(chamber_csv_path, 'w', newline='')
                chamber_writer = csv.writer(chamber_csv)
                chamber_writer.writerow([
                    'image_path', 'chamber_id', 'chamber_pixels'
                ])
                chamber_csv.flush()  # Ensure header is written immediately
                self.log_message.emit("✓ Chamber CSV created with headers")
            
            # Process images
            total_images = len(image_paths)
            images_processed = 0
            
            for img_idx, img_path in enumerate(image_paths):
                if self.should_stop:
                    self.log_message.emit("Processing cancelled by user")
                    break
                
                # Get relative path for CSV
                relative_path = img_path.relative_to(self.config['input_folder'])
                
                self.status_updated.emit(f"Processing {relative_path}...")
                self.log_message.emit(f"\n=== Image {img_idx + 1}/{total_images}: {relative_path} ===")
                
                # Process this image
                self._process_image(
                    img_path, relative_path,
                    bbox_model, hive_model, chamber_model,
                    bee_writer, hive_writer, chamber_writer,
                    summary_writer,
                    bee_csv, hive_csv, chamber_csv, summary_csv,
                    results_folder
                )
                
                images_processed += 1
                self.progress_updated.emit(images_processed, total_images)
                self.stats_updated.emit(self.total_bees, self.total_chambers, self.total_hives)
                
                # Calculate time remaining
                if images_processed > 0 and self.start_time is not None:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    avg_time_per_image = elapsed / images_processed
                    images_remaining = total_images - images_processed
                    seconds_remaining = avg_time_per_image * images_remaining
                    
                    # Format time remaining
                    if seconds_remaining < 60:
                        time_str = f"{int(seconds_remaining)}s"
                    elif seconds_remaining < 3600:
                        time_str = f"{int(seconds_remaining / 60)}m {int(seconds_remaining % 60)}s"
                    else:
                        hours = int(seconds_remaining / 3600)
                        minutes = int((seconds_remaining % 3600) / 60)
                        time_str = f"{hours}h {minutes}m"
                    
                    self.time_remaining_updated.emit(time_str)
            
            # Close CSV files
            bee_csv.flush()
            bee_csv.close()
            summary_csv.flush()
            summary_csv.close()
            if hive_csv:
                hive_csv.flush()
                hive_csv.close()
            if chamber_csv:
                chamber_csv.flush()
                chamber_csv.close()
            
            # Sort CSV files by image path and chamber ID
            self.status_updated.emit("Sorting results...")
            self.log_message.emit("\nSorting CSV files by image path and chamber ID...")
            self._sort_csv_files(results_folder, hive_model is not None, chamber_model is not None)
            self.log_message.emit("✓ CSV files sorted")
            
            # Generate summary
            self._generate_summary(results_folder, total_images, images_processed)
            
            self.log_message.emit("\n" + "="*60)
            self.log_message.emit("ANALYSIS COMPLETE")
            self.log_message.emit(f"Results saved to: {results_folder}")
            self.log_message.emit("="*60)
            
            self.finished.emit(True)
            
        except Exception as e:
            import traceback
            error_msg = f"Error during batch inference: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(error_msg)
            self.finished.emit(False)
    
    def _discover_images(self, root_folder: Path) -> List[Path]:
        """Recursively discover all PNG images in folder"""
        return sorted(root_folder.rglob("*.png"))
    
    def _process_image(self, img_path: Path, relative_path: Path,
                      bbox_model, hive_model, chamber_model,
                      bee_writer, hive_writer, chamber_writer,
                      summary_writer,
                      bee_csv, hive_csv, chamber_csv, summary_csv,
                      results_folder: Path):
        """Process a single image"""
        try:
            # Load image
            if not img_path.exists():
                self.log_message.emit(f"  Warning: Image not found: {img_path}")
                return
            
            image = cv2.imread(str(img_path))
            if image is None:
                self.log_message.emit(f"  Warning: Failed to load image: {img_path}")
                return
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run bee bbox predictions
            pred_bees = self._predict_bees(image_rgb, bbox_model)
            self.log_message.emit(f"  Detected {len(pred_bees)} bee(s)")
            self.total_bees += len(pred_bees)
            
            # Run hive prediction if model available
            pred_hive_mask = self._predict_hive(image_rgb, hive_model) if hive_model else None
            if pred_hive_mask is not None and np.any(pred_hive_mask > 0):
                self.total_hives += 1
                self.log_message.emit(f"  Detected hive")
            
            # Run chamber prediction if model available
            pred_chamber_masks = self._predict_chambers(image_rgb, chamber_model) if chamber_model else None
            if pred_chamber_masks:
                self.log_message.emit(f"  Detected {len(pred_chamber_masks)} chamber(s)")
                self.total_chambers += len(pred_chamber_masks)
            
            # Write bee detections
            self._write_bees(str(relative_path), pred_bees, pred_hive_mask, pred_chamber_masks, bee_writer)
            bee_csv.flush()  # Flush after each image
            
            # Write hive results
            if hive_writer is not None:
                if pred_hive_mask is not None and np.any(pred_hive_mask > 0):
                    self.log_message.emit(f"  Writing hive results to CSV")
                    self._write_hive_results(str(relative_path), pred_hive_mask, pred_chamber_masks, hive_writer)
                    hive_csv.flush()  # Flush after each write
                else:
                    self.log_message.emit(f"  Skipping hive CSV write - mask empty or None")
            
            # Write chamber results
            if chamber_writer is not None:
                if pred_chamber_masks:
                    self.log_message.emit(f"  Writing {len(pred_chamber_masks)} chamber(s) to CSV")
                    self._write_chamber_results(str(relative_path), pred_chamber_masks, chamber_writer)
                    chamber_csv.flush()  # Flush after each write
                else:
                    self.log_message.emit(f"  Skipping chamber CSV write - no chambers detected")
            
            # Write image summary
            self._write_image_summary(
                str(relative_path),
                pred_bees, pred_hive_mask, pred_chamber_masks,
                summary_writer
            )
            summary_csv.flush()  # Flush after each write
            
            # Save annotations if enabled
            if self.config['save_annotations']:
                self._save_annotations(
                    img_path, relative_path, results_folder,
                    pred_bees, pred_hive_mask, pred_chamber_masks
                )
            
            # Save visualization if enabled
            if self.config['save_visualizations']:
                self._save_visualization(
                    image, relative_path, results_folder,
                    pred_bees, pred_hive_mask, pred_chamber_masks
                )
            
        except Exception as e:
            import traceback
            self.log_message.emit(f"  Error processing {relative_path}: {str(e)}")
            traceback.print_exc()
    
    def _predict_bees(self, frame: np.ndarray, model) -> List[Dict]:
        """Run YOLO prediction for bees (bbox or segmentation based on config)"""
        bee_model_type = self.config.get('bee_model_type', 'bbox')
        
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
                    polygon = mask_to_simplified_polygon(mask, epsilon_percent=2.0)
                    
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
    
    def _predict_hive(self, frame: np.ndarray, model) -> Optional[np.ndarray]:
        """Run YOLO segmentation prediction for hive"""
        if model is None:
            return None
        
        results = model.predict(
            frame,
            conf=self.config['conf_threshold'],
            retina_masks=True,  # Return masks at original image size
            verbose=False
        )
        
        # Combine all hive masks
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            h, w = frame.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            for mask in masks:
                # With retina_masks=True, masks are already at original image size
                combined_mask = np.maximum(combined_mask, (mask > 0.5).astype(np.uint8) * 255)
            
            return combined_mask
        
        return None
    
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
            h, w = frame.shape[:2]
            
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
    
    def _calculate_nearest_bee_distance(self, bee_idx: int, all_bees: List[Dict],
                                         chamber_id: Optional[int] = None, 
                                         chamber_masks = None) -> Optional[float]:
        """Calculate distance from bee at bee_idx to its nearest neighbor bee
        
        Uses pixel-to-pixel distance for segmentation masks, otherwise centroid/bbox center distance.
        If chamber_id and chamber_masks are provided, only considers bees in the same chamber.
        """
        if len(all_bees) <= 1:
            return None  # No other bee to measure distance to
        
        current_bee = all_bees[bee_idx]
        has_mask = current_bee.get('mask') is not None and current_bee.get('is_segmentation', False)
        
        min_dist = np.inf
        
        for i, other_bee in enumerate(all_bees):
            if i == bee_idx:
                continue  # Skip self
            
            # If chamber filtering is requested, check if other bee is in same chamber
            if chamber_id is not None and chamber_masks is not None:
                check_point = (other_bee['bbox'][0], other_bee['bbox'][1])
                other_chamber_id = point_in_chamber(check_point, chamber_masks)
                if other_chamber_id != chamber_id:
                    continue  # Skip bees in different chambers
            
            # Calculate distance: pixel-to-pixel for masks, centroid-to-centroid otherwise
            if has_mask and other_bee.get('mask') is not None and other_bee.get('is_segmentation', False):
                # Pixel-to-pixel distance between masks (method from config)
                method = self.config.get('distance_method', 'bbox_filter')
                dist = distance_between_masks(current_bee['mask'], other_bee['mask'], method=method)
            else:
                # Centroid-to-centroid distance (fallback for bbox models)
                bee_center = current_bee['centroid']
                other_center = other_bee['centroid']
                dist = np.sqrt((bee_center[0] - other_center[0])**2 + (bee_center[1] - other_center[1])**2)
            
            min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != np.inf else None
    
    def _calculate_avg_chamber_bee_distance(self, bee_idx: int, all_bees: List[Dict], 
                                           chamber_id: Optional[int], chamber_masks) -> Optional[float]:
        """Calculate average distance from bee to other bees in the same chamber
        
        Uses centroids if available (segmentation), otherwise bbox centers
        """
        if chamber_id is None or not chamber_masks:
            return None  # No chamber information
        
        if len(all_bees) <= 1:
            return None  # No other bees
        
        # Use centroid if available, otherwise bbox center
        bee_center = all_bees[bee_idx]['centroid']
        chamber_distances = []
        
        for i, other_bee in enumerate(all_bees):
            if i == bee_idx:
                continue  # Skip self
            
            # Check if other bee is in the same chamber (using bbox center for chamber check)
            check_point = (other_bee['bbox'][0], other_bee['bbox'][1])
            other_chamber_id = point_in_chamber(check_point, chamber_masks)
            
            if other_chamber_id == chamber_id:
                # Calculate distance to this chamber mate (using centroid)
                other_center = other_bee['centroid']
                dist = np.sqrt((bee_center[0] - other_center[0])**2 + (bee_center[1] - other_center[1])**2)
                chamber_distances.append(dist)
        
        # Return average distance to chamber mates
        if chamber_distances:
            return np.mean(chamber_distances)
        return None  # No other bees in same chamber
    
    def _write_bees(self, image_path: str, pred_bees: List[Dict],
                   pred_hive_mask, pred_chamber_masks, writer):
        """Write bee detections to CSV"""
        for bee_idx, bee in enumerate(pred_bees):
            # Get chamber ID
            chamber_id = point_in_chamber((bee['bbox'][0], bee['bbox'][1]), pred_chamber_masks) if pred_chamber_masks else None
            
            # Get hive distance
            hive_dist = None
            if pred_hive_mask is not None and np.any(pred_hive_mask > 0):
                hive_dist = distance_to_mask((bee['bbox'][0], bee['bbox'][1]), pred_hive_mask)
            
            # Get nearest bee distance (within same chamber if chambers available)
            nearest_dist = self._calculate_nearest_bee_distance(bee_idx, pred_bees, chamber_id, pred_chamber_masks)
            
            # Get average chamber bee distance
            chamber_dist = self._calculate_avg_chamber_bee_distance(bee_idx, pred_bees, chamber_id, pred_chamber_masks)
            
            # Convert polygon to string format for CSV
            polygon_str = polygon_to_string(bee['polygon']) if bee['polygon'] is not None else ''
            
            writer.writerow([
                image_path,
                bee_idx + 1,  # instance_id (1-indexed)
                bee['bbox'][0], bee['bbox'][1], bee['bbox'][2], bee['bbox'][3],
                bee['centroid'][0], bee['centroid'][1],
                polygon_str,
                bee['confidence'],
                chamber_id if chamber_id is not None else '',
                hive_dist if hive_dist is not None and hive_dist != np.inf else '',
                nearest_dist if nearest_dist is not None else '',
                chamber_dist if chamber_dist is not None else ''
            ])
    
    def _write_hive_results(self, image_path: str, pred_mask, pred_chamber_masks, writer):
        """Write hive analysis to CSV, split by chamber if available"""
        # If we have chamber masks, write one row per chamber
        if pred_chamber_masks and len(pred_chamber_masks) > 0:
            for chamber_id, chamber_mask in pred_chamber_masks.items():
                # Mask the hive to only this chamber's region
                pred_masked = pred_mask.copy()
                pred_masked[chamber_mask == 0] = 0
                
                # Calculate metrics for this chamber
                pred_pixels = int(np.sum(pred_masked > 0))
                
                writer.writerow([
                    image_path, chamber_id, pred_pixels
                ])
        else:
            # No chamber masks available, write single row for whole image
            pred_pixels = int(np.sum(pred_mask > 0))
            
            writer.writerow([
                image_path, '', pred_pixels
            ])
    
    def _write_chamber_results(self, image_path: str, pred_masks: Dict[int, np.ndarray], writer):
        """Write chamber analysis to CSV"""
        for chamber_id, mask in pred_masks.items():
            pixels = int(np.sum(mask > 0))
            writer.writerow([
                image_path, chamber_id, pixels
            ])
    
    def _parse_filename(self, image_path: str):
        """Parse filename to extract metadata (e.g., bumblebox numbers)
        
        Format: bumblebox-XX_YYYY-MM-DD_HH_MM_SS
        Returns: (bumblebox_number, datetime_str) or ('', '') if not matching
        """
        import re
        # Get just the filename without extension
        filename = Path(image_path).stem
        
        match = re.match(r'bumblebox-(\d+)_(.+)', filename)
        if match:
            bumblebox_num = match.group(1)
            datetime_raw = match.group(2)
            
            # Convert from YYYY-MM-DD_HH_MM_SS to YYYY-MM-DD HH:MM:SS
            parts = datetime_raw.split('_', 1)
            if len(parts) == 2:
                date_part = parts[0]
                time_part = parts[1].replace('_', ':')
                datetime_formatted = f"{date_part} {time_part}"
                return bumblebox_num, datetime_formatted
            
            return bumblebox_num, datetime_raw
        return '', ''
    
    def _write_image_summary(self, image_path: str, pred_bees: List[Dict],
                            pred_hive_mask, pred_chamber_masks, writer):
        """Write per-image summary statistics to CSV"""
        
        # Parse filename for metadata
        bumblebox_number, datetime_str = self._parse_filename(image_path)
        
        # Log that we're writing summary
        num_chambers = len(pred_chamber_masks) if pred_chamber_masks else 0
        rows_to_write = num_chambers + 1  # per-chamber rows + whole-image row
        self.log_message.emit(f"  Writing image summary ({rows_to_write} row(s)) to CSV")
        
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
            
            # Calculate average nearest bee distance
            if count > 1:
                bee_dists = []
                for i in range(count):
                    # If chamber filtered, bees list is already filtered so no need to pass chamber info
                    # If not chamber filtered but chambers exist, calculate within-chamber distances
                    if is_chamber_filtered:
                        dist = self._calculate_nearest_bee_distance(i, bees)
                    elif chamber_masks:
                        bee_center = (bees[i]['bbox'][0], bees[i]['bbox'][1])
                        bee_chamber_id = point_in_chamber(bee_center, chamber_masks)
                        dist = self._calculate_nearest_bee_distance(i, bees, bee_chamber_id, chamber_masks)
                    else:
                        dist = self._calculate_nearest_bee_distance(i, bees)
                    if dist is not None:
                        bee_dists.append(dist)
                if bee_dists:
                    avg_bee_dist = np.mean(bee_dists)
            
            # Calculate average chamber bee distance
            if count > 1:
                if is_chamber_filtered:
                    # Bees are already filtered to one chamber
                    chamber_dists = []
                    for i in range(count):
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
                    # For whole-image: calculate avg chamber bee distance for each bee
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
                
                # Calculate statistics for this chamber
                count, avg_hive, avg_bee, avg_chamber = calculate_bee_stats(
                    pred_chamber_bees, pred_hive_mask, pred_chamber_masks, is_chamber_filtered=True)
                
                # Write chamber row
                writer.writerow([
                    image_path, chamber_id,
                    bumblebox_number, datetime_str,
                    count,
                    avg_hive if avg_hive is not None else '',
                    avg_bee if avg_bee is not None else '',
                    avg_chamber if avg_chamber is not None else ''
                ])
        
        # Always write whole-image summary row
        count, avg_hive, avg_bee, avg_chamber = calculate_bee_stats(
            pred_bees, pred_hive_mask, pred_chamber_masks, is_chamber_filtered=False)
        
        writer.writerow([
            image_path, '',  # Empty chamber_id for whole-image
            bumblebox_number, datetime_str,
            count,
            avg_hive if avg_hive is not None else '',
            avg_bee if avg_bee is not None else '',
            avg_chamber if avg_chamber is not None else ''
        ])
    
    def _save_annotations(self, img_path: Path, relative_path: Path, results_folder: Path,
                         pred_bees: List[Dict], pred_hive_mask, pred_chamber_masks):
        """Save annotations in PNG+JSON format"""
        # Create annotations folder preserving directory structure
        ann_folder = results_folder / "annotations" / relative_path.parent
        ann_folder.mkdir(parents=True, exist_ok=True)
        
        # Create JSON folder
        json_folder = ann_folder / "json"
        json_folder.mkdir(parents=True, exist_ok=True)
        
        # Save bee annotations as JSON (bbox only)
        if pred_bees:
            bee_annotations = []
            for bee_idx, bee in enumerate(pred_bees, start=1):
                # Convert from center format to x, y, w, h
                cx, cy, w, h = bee['bbox']
                x = cx - w / 2
                y = cy - h / 2
                
                bee_annotations.append({
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'bbox_only': True,
                    'confidence': float(bee['confidence']),
                    'source': 'batch_inference',
                    'category': 'bee',
                    'category_id': 1,
                    'mask_id': bee_idx
                })
            
            frame_json_path = json_folder / f"{img_path.stem}.json"
            with open(frame_json_path, 'w') as f:
                json.dump(bee_annotations, f, indent=2)
        
        # Save hive/chamber masks if available
        if pred_hive_mask is not None or pred_chamber_masks:
            png_folder = ann_folder / "png"
            png_folder.mkdir(parents=True, exist_ok=True)
            
            # Save video-level annotations JSON
            video_annotations = []
            
            if pred_hive_mask is not None and np.any(pred_hive_mask > 0):
                # Save hive mask PNG
                hive_png = png_folder / f"{img_path.stem}_hive.png"
                cv2.imwrite(str(hive_png), pred_hive_mask)
                
                video_annotations.append({
                    'category': 'hive',
                    'category_id': 2,
                    'mask_id': 1,
                    'bbox_only': False
                })
            
            if pred_chamber_masks:
                # Combine chamber masks into single multi-instance PNG
                h, w = next(iter(pred_chamber_masks.values())).shape[:2]
                combined_chamber_mask = np.zeros((h, w), dtype=np.uint8)
                
                for chamber_id, mask in pred_chamber_masks.items():
                    combined_chamber_mask[mask > 0] = chamber_id
                
                chamber_png = png_folder / f"{img_path.stem}_chamber.png"
                cv2.imwrite(str(chamber_png), combined_chamber_mask)
                
                # Add video annotations for each chamber
                for chamber_id in pred_chamber_masks.keys():
                    video_annotations.append({
                        'category': 'chamber',
                        'category_id': 3,
                        'mask_id': chamber_id,
                        'bbox_only': False
                    })
            
            # Save video annotations JSON
            if video_annotations:
                video_json_path = json_folder / f"{img_path.stem}_video_annotations.json"
                with open(video_json_path, 'w') as f:
                    json.dump(video_annotations, f, indent=2)
    
    def _save_visualization(self, image: np.ndarray, relative_path: Path, results_folder: Path,
                           pred_bees: List[Dict], pred_hive_mask, pred_chamber_masks):
        """Save visualization with color-coded bee boxes and hive/chamber outlines"""
        try:
            # Create visualizations folder
            vis_folder = results_folder / "visualizations" / relative_path.parent
            vis_folder.mkdir(parents=True, exist_ok=True)
            
            # Create visualization image
            vis_img = image.copy()
            
            # Draw hive mask as yellow outline
            hive_color = (0, 255, 255)  # Yellow in BGR
            if pred_hive_mask is not None and np.any(pred_hive_mask > 0):
                contours, _ = cv2.findContours(pred_hive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, hive_color, 3)
            
            # Draw chamber masks as blue outlines with chamber IDs
            chamber_color = (255, 0, 0)  # Blue in BGR
            if pred_chamber_masks:
                for chamber_id, mask in pred_chamber_masks.items():
                    if np.any(mask > 0):
                        mask_uint8 = mask.astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis_img, contours, -1, chamber_color, 3)
                        
                        # Draw chamber ID label
                        cx, cy, w, h = bbox_from_mask(mask_uint8)
                        label_x = int(cx)
                        label_y = int(cy + h/2 - 10)
                        cv2.putText(vis_img, f"C{chamber_id}", (label_x, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, chamber_color, 3)
            
            # Draw bee predictions (segmentation masks or bounding boxes)
            bee_color = (0, 255, 0)  # Green in BGR
            for bee_idx, bee in enumerate(pred_bees):
                instance_id = bee_idx + 1  # 1-indexed
                
                if bee.get('is_segmentation', False) and bee.get('mask') is not None:
                    # Draw segmentation mask as filled semi-transparent overlay
                    mask = bee['mask']
                    if np.any(mask > 0):
                        # Create colored overlay
                        overlay = vis_img.copy()
                        overlay[mask > 0] = bee_color
                        
                        # Blend with original image (30% opacity)
                        cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
                        
                        # Draw contour outline
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis_img, contours, -1, bee_color, 2)
                        
                        # Draw centroid marker
                        centroid_x, centroid_y = bee['centroid']
                        cv2.circle(vis_img, (int(centroid_x), int(centroid_y)), 4, bee_color, -1)
                        
                        # Draw instance ID and confidence at centroid
                        conf = bee['confidence']
                        label = f"#{instance_id} {conf:.2f}"
                        cv2.putText(vis_img, label, (int(centroid_x) + 8, int(centroid_y) - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, bee_color, 2)
                else:
                    # Draw bounding box (original behavior)
                    cx, cy, bw, bh = bee['bbox']
                    x1, y1 = int(cx - bw/2), int(cy - bh/2)
                    x2, y2 = int(cx + bw/2), int(cy + bh/2)
                    
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), bee_color, 3)
                    
                    # Draw instance ID and confidence
                    conf = bee['confidence']
                    label = f"#{instance_id} {conf:.2f}"
                    cv2.putText(vis_img, label, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, bee_color, 2)
            
            # Add legend
            legend_y = 30
            cv2.putText(vis_img, "Detections:", (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, f"Bees: {len(pred_bees)}", (10, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bee_color, 2)
            
            if pred_hive_mask is not None and np.any(pred_hive_mask > 0):
                cv2.putText(vis_img, "Yellow = Hive", (10, legend_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, hive_color, 2)
            
            if pred_chamber_masks and len(pred_chamber_masks) > 0:
                offset = 90 if (pred_hive_mask is not None and np.any(pred_hive_mask > 0)) else 60
                cv2.putText(vis_img, f"Blue = Chambers ({len(pred_chamber_masks)})", (10, legend_y + offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, chamber_color, 2)
            
            # Save image
            vis_path = vis_folder / relative_path.name
            cv2.imwrite(str(vis_path), vis_img)
            
        except Exception as e:
            self.log_message.emit(f"  Warning: Failed to save visualization: {str(e)}")
    
    def _sort_csv_files(self, results_folder: Path, has_hive: bool, has_chamber: bool):
        """Sort CSV files by image path and chamber ID"""
        try:
            # Sort bee_detections.csv by image_path
            bee_csv_path = results_folder / "bee_detections.csv"
            if bee_csv_path.exists():
                with open(bee_csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    rows = list(reader)
                
                # Only sort and rewrite if there are data rows
                if rows:
                    # Sort by image_path (column 0)
                    rows.sort(key=lambda x: x[0] if x else '')
                    
                    with open(bee_csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(rows)
                else:
                    self.log_message.emit("  Bee CSV has no data rows, skipping sort")
            
            # Sort hive_detections.csv by image_path, then chamber_id
            if has_hive:
                hive_csv_path = results_folder / "hive_detections.csv"
                if hive_csv_path.exists():
                    with open(hive_csv_path, 'r', newline='') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        rows = list(reader)
                    
                    # Only sort and rewrite if there are data rows
                    if rows:
                        # Sort by image_path (column 0), then chamber_id (column 1)
                        # Empty chamber_id should come last for each image
                        def sort_key(row):
                            if len(row) < 2:
                                return (row[0] if row else '', float('inf'))
                            img_path = row[0]
                            chamber_id = row[1]
                            # Use inf for empty chamber_id to put it last
                            chamber_num = float('inf') if chamber_id == '' else (int(chamber_id) if chamber_id.isdigit() else float('inf'))
                            return (img_path, chamber_num)
                        
                        rows.sort(key=sort_key)
                        
                        with open(hive_csv_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(header)
                            writer.writerows(rows)
                    else:
                        self.log_message.emit("  Hive CSV has no data rows, skipping sort")
            
            # Sort chamber_detections.csv by image_path, then chamber_id
            if has_chamber:
                chamber_csv_path = results_folder / "chamber_detections.csv"
                if chamber_csv_path.exists():
                    with open(chamber_csv_path, 'r', newline='') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        rows = list(reader)
                    
                    # Only sort and rewrite if there are data rows
                    if rows:
                        # Sort by image_path (column 0), then chamber_id (column 1)
                        def sort_key(row):
                            if len(row) < 2:
                                return (row[0] if row else '', 0)
                            img_path = row[0]
                            chamber_id = int(row[1]) if row[1].isdigit() else 0
                            return (img_path, chamber_id)
                        
                        rows.sort(key=sort_key)
                        
                        with open(chamber_csv_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(header)
                            writer.writerows(rows)
                    else:
                        self.log_message.emit("  Chamber CSV has no data rows, skipping sort")
            
            # Sort image_summary.csv by image_path, then chamber_id
            summary_csv_path = results_folder / "image_summary.csv"
            if summary_csv_path.exists():
                with open(summary_csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    rows = list(reader)
                
                # Only sort and rewrite if there are data rows
                if rows:
                    # Sort by image_path (column 0), then chamber_id (column 1)
                    # Empty chamber_id (whole-image summary) should come last for each image
                    def sort_key(row):
                        if len(row) < 2:
                            return (row[0] if row else '', float('inf'))
                        img_path = row[0]
                        chamber_id = row[1]
                        # Use inf for empty chamber_id to put it last
                        chamber_num = float('inf') if chamber_id == '' else (int(chamber_id) if chamber_id.isdigit() else float('inf'))
                        return (img_path, chamber_num)
                    
                    rows.sort(key=sort_key)
                    
                    with open(summary_csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(rows)
                else:
                    self.log_message.emit("  Summary CSV has no data rows, skipping sort")
        
        except Exception as e:
            self.log_message.emit(f"  Warning: Failed to sort CSV files: {str(e)}")
    
    def _generate_summary(self, results_folder: Path, total_images: int, images_processed: int):
        """Generate human-readable summary file"""
        summary_path = results_folder / "summary.txt"
        
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        with open(summary_path, 'w') as f:
            f.write("Batch Image Inference Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Input folder: {self.config['input_folder']}\n")
            f.write(f"Total images found: {total_images}\n")
            f.write(f"Images processed: {images_processed}\n")
            f.write(f"Processing time: {elapsed:.1f} seconds\n\n")
            
            f.write("Models Used:\n")
            f.write(f"  Bee detection: {self.config['bbox_model'].name}\n")
            if self.config['hive_model']:
                f.write(f"  Hive segmentation: {self.config['hive_model'].name}\n")
            if self.config['chamber_model']:
                f.write(f"  Chamber segmentation: {self.config['chamber_model'].name}\n")
            f.write("\n")
            
            f.write("Detection Summary:\n")
            f.write(f"  Total bees detected: {self.total_bees}\n")
            if self.config['chamber_model']:
                f.write(f"  Total chambers detected: {self.total_chambers}\n")
            if self.config['hive_model']:
                f.write(f"  Images with hive: {self.total_hives}\n")
            f.write("\n")
            
            f.write("Output Files:\n")
            f.write(f"  - bee_detections.csv: All bee detections\n")
            if self.config['hive_model']:
                f.write(f"  - hive_detections.csv: Hive detection results\n")
            if self.config['chamber_model']:
                f.write(f"  - chamber_detections.csv: Chamber detection results\n")
            f.write(f"  - image_summary.csv: Per-image statistics\n")
            if self.config['save_annotations']:
                f.write(f"  - annotations/: Exported annotations (PNG+JSON)\n")
            if self.config['save_visualizations']:
                f.write(f"  - visualizations/: Visualization images\n")
