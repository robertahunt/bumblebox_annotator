"""
CSV exporter for batch video inference results
"""

import csv
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from core.batch_video_processor import BeeDetectionData, ChamberFrameData, BeeTrajectory


class VideoInferenceExporter:
    """Export video inference results to CSV files"""
    
    def __init__(self, output_folder: Path):
        """
        Args:
            output_folder: Path to output folder for CSV files
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def export_bee_detections(self, bee_detections: List[BeeDetectionData]):
        """
        Export bee_detections.csv
        
        Columns: video_id, chamber_id, frame_number, bee_id, aruco_code, bbox_x, bbox_y, 
                 bbox_width, bbox_height, confidence, centroid_x, centroid_y, 
                 distance_to_hive_pixels, num_bees_in_chamber, avg_distance_to_other_bees_pixels,
                 distance_to_nearest_bee_pixels, avg_distance_to_nearest_2_bees_pixels,
                 avg_distance_to_nearest_3_bees_pixels
        """
        csv_path = self.output_folder / 'bee_detections.csv'
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'video_id', 'chamber_id', 'frame_number', 'bee_id', 'aruco_code',
                'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'confidence',
                'centroid_x', 'centroid_y', 'distance_to_hive_pixels',
                'num_bees_in_chamber', 'avg_distance_to_other_bees_pixels',
                'distance_to_nearest_bee_pixels', 'avg_distance_to_nearest_2_bees_pixels',
                'avg_distance_to_nearest_3_bees_pixels'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for detection in bee_detections:
                writer.writerow({
                    'video_id': detection.video_id,
                    'chamber_id': detection.chamber_id,
                    'frame_number': detection.frame_number,
                    'bee_id': detection.bee_id,
                    'aruco_code': detection.aruco_code,
                    'bbox_x': f"{detection.bbox_x:.2f}",
                    'bbox_y': f"{detection.bbox_y:.2f}",
                    'bbox_width': f"{detection.bbox_width:.2f}",
                    'bbox_height': f"{detection.bbox_height:.2f}",
                    'confidence': f"{detection.confidence:.4f}",
                    'centroid_x': f"{detection.centroid_x:.2f}",
                    'centroid_y': f"{detection.centroid_y:.2f}",
                    'distance_to_hive_pixels': f"{detection.distance_to_hive_pixels:.2f}",
                    'num_bees_in_chamber': detection.num_bees_in_chamber,
                    'avg_distance_to_other_bees_pixels': f"{detection.avg_distance_to_other_bees_pixels:.2f}",
                    'distance_to_nearest_bee_pixels': f"{detection.distance_to_nearest_bee_pixels:.2f}",
                    'avg_distance_to_nearest_2_bees_pixels': f"{detection.avg_distance_to_nearest_2_bees_pixels:.2f}",
                    'avg_distance_to_nearest_3_bees_pixels': f"{detection.avg_distance_to_nearest_3_bees_pixels:.2f}"
                })
        
        return csv_path
    
    def export_bee_velocity(self, bee_trajectories: Dict):
        """
        Export bee_velocity.csv
        
        Columns: video_id, chamber_id, bee_id, aruco_code, average_velocity_pixels_per_frame, num_frame_transitions
        
        Calculate velocity as: avg displacement between consecutive frames
        
        Args:
            bee_trajectories: Dict with keys as (video_id, bee_id) tuples or just bee_id (for backward compatibility)
        """
        csv_path = self.output_folder / 'bee_velocity.csv'
        
        # Calculate velocities
        velocity_data = []
        
        for key, trajectory in bee_trajectories.items():
            # Handle both composite keys (video_id, bee_id) and simple bee_id keys
            if isinstance(key, tuple) and len(key) == 2:
                video_id, bee_id = key
            else:
                # Legacy format or single video
                bee_id = key
                video_id = ''  # Will be filled later
            
            if len(trajectory.positions) < 2:
                # Need at least 2 positions to calculate velocity
                continue
            
            # Calculate velocities between consecutive frames
            velocities = []
            for i in range(1, len(trajectory.positions)):
                frame1, x1, y1 = trajectory.positions[i-1]
                frame2, x2, y2 = trajectory.positions[i]
                
                # Displacement
                displacement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Frame difference (usually 1, but could be more if frames skipped)
                frame_diff = frame2 - frame1
                
                if frame_diff > 0:
                    velocity = displacement / frame_diff
                    velocities.append(velocity)
            
            if velocities:
                avg_velocity = np.mean(velocities)
                num_transitions = len(velocities)
                
                velocity_data.append({
                    'video_id': video_id,
                    'bee_id': bee_id,
                    'chamber_id': trajectory.chamber_id,
                    'aruco_code': trajectory.aruco_code,
                    'average_velocity': avg_velocity,
                    'num_transitions': num_transitions
                })
        
        # Write CSV
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'video_id', 'chamber_id', 'bee_id', 'aruco_code', 
                'average_velocity_pixels_per_frame', 'num_frame_transitions'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Sort by video_id and bee_id for consistent output
            sorted_data = sorted(velocity_data, key=lambda x: (x['video_id'], x['bee_id']))
            
            for data in sorted_data:
                writer.writerow({
                    'video_id': data['video_id'],
                    'chamber_id': data['chamber_id'],
                    'bee_id': data['bee_id'],
                    'aruco_code': data['aruco_code'],
                    'average_velocity_pixels_per_frame': f"{data['average_velocity']:.2f}",
                    'num_frame_transitions': data['num_transitions']
                })
        
        return csv_path
    
    def export_hive_detections(self, accumulated_hive_masks: Dict):
        """
        Export hive_detections.csv
        
        Columns: video_id, chamber_id, hive_pixels, centroid_x, centroid_y
        
        Average masks across frames, threshold at 0.5, then count pixels
        Calculate centroid of the averaged hive mask
        """
        csv_path = self.output_folder / 'hive_detections.csv'
        
        # Write CSV
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['video_id', 'chamber_id', 'hive_pixels', 'centroid_x', 'centroid_y']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Sort by video_id and chamber_id for consistent output
            sorted_keys = sorted(accumulated_hive_masks.keys(), key=lambda x: (x[0], x[1]))
            
            for (video_id, chamber_id) in sorted_keys:
                data = accumulated_hive_masks[(video_id, chamber_id)]
                accumulated_mask = data['accumulated_mask']
                frame_count = data['frame_count']
                
                if frame_count == 0:
                    hive_pixels = 0
                    centroid_x, centroid_y = 0.0, 0.0
                else:
                    # Average the mask
                    avg_mask = accumulated_mask / frame_count
                    
                    # Threshold at 0.5
                    thresholded_mask = (avg_mask > 0.5).astype(np.uint8)
                    
                    # Count pixels
                    hive_pixels = int(np.sum(thresholded_mask))
                    
                    # Calculate centroid of the hive mask
                    if hive_pixels > 0:
                        y_coords, x_coords = np.where(thresholded_mask > 0)
                        centroid_x = float(np.mean(x_coords))
                        centroid_y = float(np.mean(y_coords))
                    else:
                        centroid_x, centroid_y = 0.0, 0.0
                
                writer.writerow({
                    'video_id': video_id,
                    'chamber_id': chamber_id,
                    'hive_pixels': hive_pixels,
                    'centroid_x': f"{centroid_x:.2f}",
                    'centroid_y': f"{centroid_y:.2f}"
                })
        
        return csv_path
    
    def export_chamber_detections(self, accumulated_chamber_masks: Dict):
        """
        Export chamber_detections.csv
        
        Columns: video_id, chamber_id, chamber_pixels, centroid_x, centroid_y
        
        Average masks across frames, threshold at 0.5, then count pixels
        Average centroids across frames
        """
        csv_path = self.output_folder / 'chamber_detections.csv'
        
        # Write CSV
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['video_id', 'chamber_id', 'chamber_pixels', 'centroid_x', 'centroid_y']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Sort by video_id and chamber_id for consistent output
            sorted_keys = sorted(accumulated_chamber_masks.keys(), key=lambda x: (x[0], x[1]))
            
            for (video_id, chamber_id) in sorted_keys:
                data = accumulated_chamber_masks[(video_id, chamber_id)]
                accumulated_mask = data['accumulated_mask']
                frame_count = data['frame_count']
                accumulated_centroid = data.get('accumulated_centroid', np.array([0.0, 0.0]))
                
                if frame_count == 0:
                    chamber_pixels = 0
                    centroid_x, centroid_y = 0.0, 0.0
                else:
                    # Average the mask
                    avg_mask = accumulated_mask / frame_count
                    
                    # Threshold at 0.5
                    thresholded_mask = (avg_mask > 0.5).astype(np.uint8)
                    
                    # Count pixels
                    chamber_pixels = int(np.sum(thresholded_mask))
                    
                    # Average centroid
                    avg_centroid = accumulated_centroid / frame_count
                    centroid_x, centroid_y = float(avg_centroid[0]), float(avg_centroid[1])
                
                writer.writerow({
                    'video_id': video_id,
                    'chamber_id': chamber_id,
                    'chamber_pixels': chamber_pixels,
                    'centroid_x': f"{centroid_x:.2f}",
                    'centroid_y': f"{centroid_y:.2f}"
                })
        
        return csv_path
    
    def export_all(self, bee_detections: List[BeeDetectionData],
                   bee_trajectories: Dict[int, BeeTrajectory],
                   chamber_frame_data: List[ChamberFrameData],
                   accumulated_hive_masks: Dict,
                   accumulated_chamber_masks: Dict) -> Dict[str, Path]:
        """
        Export all CSV files
        
        Returns:
            Dict mapping csv_name -> path
        """
        results = {}
        
        # Export bee detections
        results['bee_detections'] = self.export_bee_detections(bee_detections)
        
        # Export bee velocity
        results['bee_velocity'] = self.export_bee_velocity(bee_trajectories)
        
        # Export hive detections (averaged masks)
        results['hive_detections'] = self.export_hive_detections(accumulated_hive_masks)
        
        # Export chamber detections (averaged masks)
        results['chamber_detections'] = self.export_chamber_detections(accumulated_chamber_masks)
        
        return results
    
    def update_bee_velocity_video_ids(self, bee_detections: List[BeeDetectionData]):
        """
        Update bee_velocity.csv with correct video_ids
        
        This is a post-processing step since trajectories don't store video_id
        """
        velocity_csv = self.output_folder / 'bee_velocity.csv'
        
        if not velocity_csv.exists():
            return
        
        # Build mapping: bee_id -> video_id
        bee_to_video = {}
        for detection in bee_detections:
            if detection.bee_id not in bee_to_video:
                bee_to_video[detection.bee_id] = detection.video_id
        
        # Read existing velocity CSV
        rows = []
        with open(velocity_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            for row in reader:
                bee_id = int(row['bee_id'])
                row['video_id'] = bee_to_video.get(bee_id, '')
                rows.append(row)
        
        # Write back with updated video_ids
        with open(velocity_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
