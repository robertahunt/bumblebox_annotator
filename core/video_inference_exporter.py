"""
CSV exporter for batch video inference results
"""

import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

from core.batch_video_processor import (
    ArucoObservationData,
    BeeDetectionData,
    BeeIdentityEventData,
    BeeIdentitySegmentData,
    BeeInteractionData,
    ChamberFrameData,
    PollenFrameData,
    BeeTrajectory,
)


class VideoInferenceExporter:
    """Export video inference results to CSV files"""
    
    def __init__(self, output_folder: Path):
        """
        Args:
            output_folder: Path to output folder for CSV files
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_optional_float(value, decimals: int = 2) -> str:
        """Format a numeric value for CSV, leaving missing values blank."""
        if value is None:
            return ""
        return f"{value:.{decimals}f}"

    @staticmethod
    def _format_optional_int(value) -> str:
        """Format an integer value for CSV, leaving missing values blank."""
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _format_optional_bool(value) -> str:
        """Format a bool for CSV, leaving missing values blank."""
        if value is None:
            return ""
        return "true" if bool(value) else "false"

    @staticmethod
    def _should_write_header(csv_path: Path, append: bool) -> bool:
        """Return True when a CSV header should be written."""
        return (not append) or (not csv_path.exists()) or csv_path.stat().st_size == 0
    
    def export_bee_detections(self, bee_detections: List[BeeDetectionData], append: bool = False):
        """
        Export bee_detections.csv
        
        Columns: video_id, chamber_id, frame_number, bee_id, aruco_code, bbox_x, bbox_y, 
                 bbox_width, bbox_height, confidence, centroid_x, centroid_y, 
                 identity_segment_id, identity_support_level,
                 frames_since_last_aruco, frames_until_next_aruco,
                 nearest_aruco_gap_frames,
                 bee_mask_pixels, bbox_area_pixels, bee_mask_area_mm2, bbox_area_mm2,
                 pred_polygon, distance_to_hive_pixels, distance_to_nearest_pollen_pixels,
                 pollen_count_in_chamber, on_pollen_ball, pollen_overlap_pixels,
                 pollen_overlap_fraction, on_temporal_hive,
                 temporal_hive_overlap_fraction, temporal_hive_mean_probability,
                 temporal_hive_known_fraction, num_bees_in_chamber,
                 avg_distance_to_other_bees_pixels, distance_to_nearest_bee_pixels,
                 avg_distance_to_nearest_2_bees_pixels, avg_distance_to_nearest_3_bees_pixels
        """
        csv_path = self.output_folder / 'bee_detections.csv'
        write_header = self._should_write_header(csv_path, append)
        mode = 'a' if append else 'w'
        
        with open(csv_path, mode, newline='') as f:
            fieldnames = [
                'video_id', 'chamber_id', 'frame_number', 'bee_id', 'aruco_code',
                'identity_segment_id', 'identity_support_level',
                'frames_since_last_aruco', 'frames_until_next_aruco',
                'nearest_aruco_gap_frames',
                'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'confidence',
                'centroid_x', 'centroid_y', 'bee_mask_pixels', 'bbox_area_pixels',
                'bee_mask_area_mm2', 'bbox_area_mm2', 'pred_polygon',
                'distance_to_hive_pixels', 'distance_to_hive_mm',
                'distance_to_nearest_pollen_pixels', 'distance_to_nearest_pollen_mm',
                'pollen_count_in_chamber', 'on_pollen_ball',
                'pollen_overlap_pixels', 'pollen_overlap_fraction',
                'on_temporal_hive', 'temporal_hive_overlap_fraction',
                'temporal_hive_mean_probability', 'temporal_hive_known_fraction',
                'temporal_hive_overlap_pixels_norm', 'temporal_hive_known_pixels_norm',
                'temporal_hive_bee_pixels_norm', 'temporal_hive_prior_weight_mean',
                'temporal_hive_prior_weight_sum',
                'num_bees_in_chamber', 'avg_distance_to_other_bees_pixels',
                'distance_to_nearest_bee_pixels', 'avg_distance_to_nearest_2_bees_pixels',
                'avg_distance_to_nearest_3_bees_pixels',
                'avg_distance_to_other_bees_mm', 'distance_to_nearest_bee_mm',
                'avg_distance_to_nearest_2_bees_mm', 'avg_distance_to_nearest_3_bees_mm'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            
            for detection in bee_detections:
                writer.writerow({
                    'video_id': detection.video_id,
                    'chamber_id': detection.chamber_id,
                    'frame_number': detection.frame_number,
                    'bee_id': detection.bee_id,
                    'aruco_code': detection.aruco_code,
                    'identity_segment_id': detection.identity_segment_id,
                    'identity_support_level': detection.identity_support_level,
                    'frames_since_last_aruco': self._format_optional_int(detection.frames_since_last_aruco),
                    'frames_until_next_aruco': self._format_optional_int(detection.frames_until_next_aruco),
                    'nearest_aruco_gap_frames': self._format_optional_int(detection.nearest_aruco_gap_frames),
                    'bbox_x': f"{detection.bbox_x:.2f}",
                    'bbox_y': f"{detection.bbox_y:.2f}",
                    'bbox_width': f"{detection.bbox_width:.2f}",
                    'bbox_height': f"{detection.bbox_height:.2f}",
                    'confidence': f"{detection.confidence:.4f}",
                    'centroid_x': f"{detection.centroid_x:.2f}",
                    'centroid_y': f"{detection.centroid_y:.2f}",
                    'bee_mask_pixels': self._format_optional_int(detection.bee_mask_pixels),
                    'bbox_area_pixels': f"{detection.bbox_area_pixels:.2f}",
                    'bee_mask_area_mm2': self._format_optional_float(detection.bee_mask_area_mm2, decimals=4),
                    'bbox_area_mm2': self._format_optional_float(detection.bbox_area_mm2, decimals=4),
                    'pred_polygon': detection.pred_polygon,
                    'distance_to_hive_pixels': self._format_optional_float(detection.distance_to_hive_pixels),
                    'distance_to_hive_mm': self._format_optional_float(detection.distance_to_hive_mm, decimals=4),
                    'distance_to_nearest_pollen_pixels': self._format_optional_float(detection.distance_to_nearest_pollen_pixels),
                    'distance_to_nearest_pollen_mm': self._format_optional_float(detection.distance_to_nearest_pollen_mm, decimals=4),
                    'pollen_count_in_chamber': self._format_optional_int(detection.pollen_count_in_chamber),
                    'on_pollen_ball': self._format_optional_bool(detection.on_pollen_ball),
                    'pollen_overlap_pixels': self._format_optional_int(detection.pollen_overlap_pixels),
                    'pollen_overlap_fraction': self._format_optional_float(detection.pollen_overlap_fraction, decimals=4),
                    'on_temporal_hive': self._format_optional_bool(detection.on_temporal_hive),
                    'temporal_hive_overlap_fraction': self._format_optional_float(detection.temporal_hive_overlap_fraction, decimals=4),
                    'temporal_hive_mean_probability': self._format_optional_float(detection.temporal_hive_mean_probability, decimals=4),
                    'temporal_hive_known_fraction': self._format_optional_float(detection.temporal_hive_known_fraction, decimals=4),
                    'temporal_hive_overlap_pixels_norm': self._format_optional_int(detection.temporal_hive_overlap_pixels_norm),
                    'temporal_hive_known_pixels_norm': self._format_optional_int(detection.temporal_hive_known_pixels_norm),
                    'temporal_hive_bee_pixels_norm': self._format_optional_int(detection.temporal_hive_bee_pixels_norm),
                    'temporal_hive_prior_weight_mean': self._format_optional_float(detection.temporal_hive_prior_weight_mean, decimals=4),
                    'temporal_hive_prior_weight_sum': self._format_optional_float(detection.temporal_hive_prior_weight_sum, decimals=2),
                    'num_bees_in_chamber': self._format_optional_int(detection.num_bees_in_chamber),
                    'avg_distance_to_other_bees_pixels': self._format_optional_float(detection.avg_distance_to_other_bees_pixels),
                    'distance_to_nearest_bee_pixels': self._format_optional_float(detection.distance_to_nearest_bee_pixels),
                    'avg_distance_to_nearest_2_bees_pixels': self._format_optional_float(detection.avg_distance_to_nearest_2_bees_pixels),
                    'avg_distance_to_nearest_3_bees_pixels': self._format_optional_float(detection.avg_distance_to_nearest_3_bees_pixels),
                    'avg_distance_to_other_bees_mm': self._format_optional_float(detection.avg_distance_to_other_bees_mm, decimals=4),
                    'distance_to_nearest_bee_mm': self._format_optional_float(detection.distance_to_nearest_bee_mm, decimals=4),
                    'avg_distance_to_nearest_2_bees_mm': self._format_optional_float(detection.avg_distance_to_nearest_2_bees_mm, decimals=4),
                    'avg_distance_to_nearest_3_bees_mm': self._format_optional_float(detection.avg_distance_to_nearest_3_bees_mm, decimals=4)
                })
        
        return csv_path

    def export_bee_interactions(self, bee_interactions: List[BeeInteractionData], append: bool = False):
        """
        Export bee_interactions.csv

        One row per same-frame bee pair whose masks overlap or touch within one pixel.
        """
        csv_path = self.output_folder / 'bee_interactions.csv'
        write_header = self._should_write_header(csv_path, append)
        mode = 'a' if append else 'w'

        with open(csv_path, mode, newline='') as f:
            fieldnames = [
                'video_id', 'chamber_id', 'frame_number',
                'bee_id_1', 'bee_id_2', 'aruco_code_1', 'aruco_code_2',
                'mask_overlap_pixels', 'mask_contact_pixels',
                'centroid_distance_pixels', 'centroid_distance_mm'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for interaction in bee_interactions:
                writer.writerow({
                    'video_id': interaction.video_id,
                    'chamber_id': interaction.chamber_id,
                    'frame_number': interaction.frame_number,
                    'bee_id_1': interaction.bee_id_1,
                    'bee_id_2': interaction.bee_id_2,
                    'aruco_code_1': interaction.aruco_code_1,
                    'aruco_code_2': interaction.aruco_code_2,
                    'mask_overlap_pixels': interaction.mask_overlap_pixels,
                    'mask_contact_pixels': interaction.mask_contact_pixels,
                    'centroid_distance_pixels': self._format_optional_float(interaction.centroid_distance_pixels),
                    'centroid_distance_mm': self._format_optional_float(interaction.centroid_distance_mm, decimals=4),
                })

        return csv_path

    def export_aruco_observations(self, aruco_observations: List[ArucoObservationData], append: bool = False):
        """
        Export aruco_observations.csv

        One row per physical marker observation matched to a tracked bee candidate.
        Rejected observations are retained so identity decisions are auditable.
        """
        csv_path = self.output_folder / 'aruco_observations.csv'
        write_header = self._should_write_header(csv_path, append)
        mode = 'a' if append else 'w'

        with open(csv_path, mode, newline='') as f:
            fieldnames = [
                'video_id', 'frame_number', 'tracker_bee_id', 'bee_id',
                'aruco_code', 'accepted', 'decision', 'marker_confidence',
                'marker_center_x', 'marker_center_y', 'dict_type'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for observation in aruco_observations:
                writer.writerow({
                    'video_id': observation.video_id,
                    'frame_number': observation.frame_number,
                    'tracker_bee_id': observation.tracker_bee_id,
                    'bee_id': observation.bee_id,
                    'aruco_code': observation.aruco_code,
                    'accepted': self._format_optional_bool(observation.accepted),
                    'decision': observation.decision,
                    'marker_confidence': self._format_optional_float(observation.marker_confidence, decimals=4),
                    'marker_center_x': self._format_optional_float(observation.marker_center_x, decimals=2),
                    'marker_center_y': self._format_optional_float(observation.marker_center_y, decimals=2),
                    'dict_type': observation.dict_type,
                })

        return csv_path

    def export_bee_identity_events(self, bee_identity_events: List[BeeIdentityEventData], append: bool = False):
        """
        Export bee_identity_events.csv

        One row per ArUco-based identity decision: assignment, confirmation,
        reidentification, or rejection.
        """
        csv_path = self.output_folder / 'bee_identity_events.csv'
        write_header = self._should_write_header(csv_path, append)
        mode = 'a' if append else 'w'

        with open(csv_path, mode, newline='') as f:
            fieldnames = [
                'video_id', 'frame_number', 'event_type', 'aruco_code',
                'source_bee_id', 'target_bee_id', 'accepted', 'reason',
                'marker_confidence'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for event in bee_identity_events:
                writer.writerow({
                    'video_id': event.video_id,
                    'frame_number': event.frame_number,
                    'event_type': event.event_type,
                    'aruco_code': event.aruco_code,
                    'source_bee_id': self._format_optional_int(event.source_bee_id),
                    'target_bee_id': self._format_optional_int(event.target_bee_id),
                    'accepted': self._format_optional_bool(event.accepted),
                    'reason': event.reason,
                    'marker_confidence': self._format_optional_float(event.marker_confidence, decimals=4),
                })

        return csv_path

    def export_bee_identity_segments(self, bee_identity_segments: List[BeeIdentitySegmentData], append: bool = False):
        """
        Export bee_identity_segments.csv

        Contiguous stretches of a track with the same ArUco identity-support level.
        """
        csv_path = self.output_folder / 'bee_identity_segments.csv'
        write_header = self._should_write_header(csv_path, append)
        mode = 'a' if append else 'w'

        with open(csv_path, mode, newline='') as f:
            fieldnames = [
                'video_id', 'identity_segment_id', 'bee_id', 'aruco_code',
                'support_level', 'start_frame', 'end_frame', 'duration_frames',
                'detection_count', 'tag_observation_count', 'first_aruco_frame',
                'last_aruco_frame', 'max_gap_frames', 'median_gap_frames',
                'support_reason'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for segment in bee_identity_segments:
                writer.writerow({
                    'video_id': segment.video_id,
                    'identity_segment_id': segment.identity_segment_id,
                    'bee_id': segment.bee_id,
                    'aruco_code': segment.aruco_code,
                    'support_level': segment.support_level,
                    'start_frame': segment.start_frame,
                    'end_frame': segment.end_frame,
                    'duration_frames': segment.duration_frames,
                    'detection_count': segment.detection_count,
                    'tag_observation_count': segment.tag_observation_count,
                    'first_aruco_frame': self._format_optional_int(segment.first_aruco_frame),
                    'last_aruco_frame': self._format_optional_int(segment.last_aruco_frame),
                    'max_gap_frames': self._format_optional_int(segment.max_gap_frames),
                    'median_gap_frames': self._format_optional_float(segment.median_gap_frames, decimals=2),
                    'support_reason': segment.support_reason,
                })

        return csv_path

    def export_temporal_hive_priors(self, temporal_hive_prior_summaries: Optional[List[Dict]]):
        """
        Export temporal_hive_priors.csv

        Stable hive perimeter/probability summaries in normalized chamber coordinates.
        """
        csv_path = self.output_folder / 'temporal_hive_priors.csv'

        fieldnames = [
            'context_id', 'chamber_id', 'prior_hive_pixels_norm',
            'centroid_x_norm', 'centroid_y_norm', 'mean_prior_weight',
            'max_prior_weight', 'observation_count', 'prior_polygon_norm',
            'resolution_width', 'resolution_height'
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in temporal_hive_prior_summaries or []:
                writer.writerow({
                    'context_id': row.get('context_id', ''),
                    'chamber_id': row.get('chamber_id', ''),
                    'prior_hive_pixels_norm': row.get('prior_hive_pixels_norm', 0),
                    'centroid_x_norm': self._format_optional_float(row.get('centroid_x_norm'), decimals=6),
                    'centroid_y_norm': self._format_optional_float(row.get('centroid_y_norm'), decimals=6),
                    'mean_prior_weight': self._format_optional_float(row.get('mean_prior_weight'), decimals=2),
                    'max_prior_weight': self._format_optional_float(row.get('max_prior_weight'), decimals=2),
                    'observation_count': row.get('observation_count', 0),
                    'prior_polygon_norm': row.get('prior_polygon_norm', ''),
                    'resolution_width': row.get('resolution_width', ''),
                    'resolution_height': row.get('resolution_height', ''),
                })

        return csv_path

    def export_pollen_detections(self, pollen_frame_data: List[PollenFrameData], append: bool = False):
        """
        Export pollen_detections.csv.

        One row per video/chamber/frame with pollen count and total pollen pixels.
        """
        csv_path = self.output_folder / 'pollen_detections.csv'
        write_header = self._should_write_header(csv_path, append)
        mode = 'a' if append else 'w'

        with open(csv_path, mode, newline='') as f:
            fieldnames = [
                'video_id', 'chamber_id', 'frame_number',
                'pollen_count', 'pollen_pixels', 'pollen_area_mm2'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for row in pollen_frame_data:
                writer.writerow({
                    'video_id': row.video_id,
                    'chamber_id': row.chamber_id,
                    'frame_number': row.frame_number,
                    'pollen_count': row.pollen_count,
                    'pollen_pixels': row.pollen_pixels,
                    'pollen_area_mm2': self._format_optional_float(row.pollen_area_mm2, decimals=4),
                })

        return csv_path
    
    def export_bee_velocity(self, bee_trajectories: Dict, append: bool = False):
        """
        Export bee_velocity.csv
        
        Columns: video_id, chamber_id, bee_id, aruco_code, average_velocity_pixels_per_frame, num_frame_transitions
        
        Calculate velocity as: avg displacement between consecutive frames
        
        Args:
            bee_trajectories: Dict with keys as (video_id, bee_id) tuples or just bee_id (for backward compatibility)
        """
        csv_path = self.output_folder / 'bee_velocity.csv'
        write_header = self._should_write_header(csv_path, append)
        mode = 'a' if append else 'w'
        
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
        with open(csv_path, mode, newline='') as f:
            fieldnames = [
                'video_id', 'chamber_id', 'bee_id', 'aruco_code', 
                'average_velocity_pixels_per_frame', 'num_frame_transitions'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
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
    
    def export_hive_detections(self, accumulated_hive_masks: Dict, append: bool = False):
        """
        Export hive_detections.csv
        
        Columns: video_id, chamber_id, hive_pixels, centroid_x, centroid_y
        
        Average masks across frames, threshold at 0.5, then count pixels
        Calculate centroid of the averaged hive mask
        """
        csv_path = self.output_folder / 'hive_detections.csv'
        write_header = self._should_write_header(csv_path, append)
        mode = 'a' if append else 'w'
        
        # Write CSV
        with open(csv_path, mode, newline='') as f:
            fieldnames = ['video_id', 'chamber_id', 'hive_pixels', 'centroid_x', 'centroid_y']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
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
                    # Threshold pixels present in more than half of sampled frames.
                    thresholded_mask = accumulated_mask > (0.5 * frame_count)
                    
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
    
    def export_chamber_detections(self, accumulated_chamber_masks: Dict, append: bool = False):
        """
        Export chamber_detections.csv
        
        Columns: video_id, chamber_id, chamber_pixels, centroid_x, centroid_y
        
        Average masks across frames, threshold at 0.5, then count pixels
        Average centroids across frames
        """
        csv_path = self.output_folder / 'chamber_detections.csv'
        write_header = self._should_write_header(csv_path, append)
        mode = 'a' if append else 'w'
        
        # Write CSV
        with open(csv_path, mode, newline='') as f:
            fieldnames = ['video_id', 'chamber_id', 'chamber_pixels', 'centroid_x', 'centroid_y']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
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
                    # Threshold pixels present in more than half of sampled frames.
                    thresholded_mask = accumulated_mask > (0.5 * frame_count)
                    
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
                   bee_interactions: List[BeeInteractionData],
                   aruco_observations: List[ArucoObservationData],
                   bee_identity_events: List[BeeIdentityEventData],
                   bee_identity_segments: List[BeeIdentitySegmentData],
                   bee_trajectories: Dict[int, BeeTrajectory],
                   chamber_frame_data: List[ChamberFrameData],
                   pollen_frame_data: List[PollenFrameData],
                   accumulated_hive_masks: Dict,
                   accumulated_chamber_masks: Dict,
                   export_hive_detections: bool = True,
                   export_pollen_detections: bool = False,
                   temporal_hive_prior_summaries: Optional[List[Dict]] = None,
                   append: bool = False) -> Dict[str, Path]:
        """
        Export all CSV files
        
        Returns:
            Dict mapping csv_name -> path
        """
        results = {}
        
        # Export bee detections
        results['bee_detections'] = self.export_bee_detections(bee_detections, append=append)

        # Export pairwise bee mask-contact events
        results['bee_interactions'] = self.export_bee_interactions(bee_interactions, append=append)

        # Export ArUco identity evidence and support summaries
        results['aruco_observations'] = self.export_aruco_observations(aruco_observations, append=append)
        results['bee_identity_events'] = self.export_bee_identity_events(bee_identity_events, append=append)
        results['bee_identity_segments'] = self.export_bee_identity_segments(bee_identity_segments, append=append)

        if export_pollen_detections:
            results['pollen_detections'] = self.export_pollen_detections(pollen_frame_data, append=append)
        elif not append:
            stale_pollen_csv = self.output_folder / 'pollen_detections.csv'
            stale_pollen_csv.unlink(missing_ok=True)
        
        # Export bee velocity
        results['bee_velocity'] = self.export_bee_velocity(bee_trajectories, append=append)
        
        # Export hive detections (averaged masks) only when a hive model was provided
        if export_hive_detections:
            results['hive_detections'] = self.export_hive_detections(accumulated_hive_masks, append=append)
        elif not append:
            stale_hive_csv = self.output_folder / 'hive_detections.csv'
            stale_hive_csv.unlink(missing_ok=True)
        
        # Export chamber detections (averaged masks)
        results['chamber_detections'] = self.export_chamber_detections(accumulated_chamber_masks, append=append)

        if temporal_hive_prior_summaries is not None:
            results['temporal_hive_priors'] = self.export_temporal_hive_priors(temporal_hive_prior_summaries)
        
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
