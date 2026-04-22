"""
Visualization generator for batch video inference
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

from core.batch_video_processor import BeeDetectionData, ChamberFrameData


class VisualizationGenerator:
    """Generate annotated videos with tracking, ArUco, and spatial information"""
    
    def __init__(self, video_path: Path, output_path: Path, 
                 bee_detections: List[BeeDetectionData],
                 chamber_frame_data: List[ChamberFrameData],
                 chambers: Dict[int, Dict]):
        """
        Args:
            video_path: Path to input video
            output_path: Path to output video
            bee_detections: List of bee detections
            chamber_frame_data: List of chamber frame data
            chambers: Chamber information with ArUco markers
        """
        self.video_path = video_path
        self.output_path = output_path
        self.bee_detections = bee_detections
        self.chamber_frame_data = chamber_frame_data
        self.chambers = chambers
        
        # Organize detections by frame
        self.detections_by_frame = defaultdict(list)
        for detection in bee_detections:
            self.detections_by_frame[detection.frame_number].append(detection)
        
        # Organize chamber data by frame
        self.chamber_data_by_frame = defaultdict(dict)
        for data in chamber_frame_data:
            self.chamber_data_by_frame[data.frame_number][data.chamber_id] = data
        
        # Track bee positions for trajectory lines
        self.bee_trajectories = defaultdict(list)  # bee_id -> [(x, y), ...]
        self.trajectory_max_length = 30  # frames to keep in trail
    
    def generate(self) -> bool:
        """
        Generate annotated video
        
        Returns:
            True if successful, False otherwise
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Annotate this frame
            annotated = self._annotate_frame(frame, frame_number)
            
            # Write frame
            out.write(annotated)
        
        cap.release()
        out.release()
        
        return True
    
    def _annotate_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Annotate a single frame"""
        annotated = frame.copy()
        
        # 1. Draw chamber markers and boundaries
        self._draw_chambers(annotated)
        
        # 2. Draw bee detections with IDs and ArUco codes
        detections = self.detections_by_frame.get(frame_number, [])
        self._draw_bee_detections(annotated, detections, frame_number)
        
        # 3. Draw hive pixel info
        chamber_data = self.chamber_data_by_frame.get(frame_number, {})
        self._draw_chamber_info(annotated, chamber_data)
        
        # 4. Draw status bar
        self._draw_status_bar(annotated, frame_number, len(detections))
        
        return annotated
    
    def _draw_chambers(self, frame: np.ndarray):
        """Draw chamber boundaries and ArUco markers"""
        for chamber_id, marker_info in self.chambers.items():
            corners = marker_info['corners']
            aruco_id = marker_info['aruco_id']
            centroid = marker_info['centroid']
            
            # Draw marker outline
            corners_int = corners.astype(np.int32)
            cv2.polylines(frame, [corners_int], True, (0, 255, 255), 2)
            
            # Draw chamber ID label
            cx, cy = centroid
            label = f"Chamber {chamber_id}"
            cv2.putText(frame, label, (int(cx) - 40, int(cy) - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw ArUco ID
            aruco_label = f"ArUco {aruco_id}"
            cv2.putText(frame, aruco_label, (int(cx) - 35, int(cy) - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _draw_bee_detections(self, frame: np.ndarray, detections: List[BeeDetectionData],
                            frame_number: int):
        """Draw bee bounding boxes, IDs, tracking trails"""
        for detection in detections:
            # Bounding box
            x1 = int(detection.bbox_x)
            y1 = int(detection.bbox_y)
            x2 = int(detection.bbox_x + detection.bbox_width)
            y2 = int(detection.bbox_y + detection.bbox_height)
            
            # Color based on chamber
            colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), 
                     (255, 255, 100), (255, 100, 255), (100, 255, 255)]
            color = colors[detection.chamber_id % len(colors)]
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Update trajectory
            centroid = (int(detection.centroid_x), int(detection.centroid_y))
            self.bee_trajectories[detection.bee_id].append(centroid)
            
            # Limit trajectory length
            if len(self.bee_trajectories[detection.bee_id]) > self.trajectory_max_length:
                self.bee_trajectories[detection.bee_id].pop(0)
            
            # Draw trajectory trail
            traj = self.bee_trajectories[detection.bee_id]
            if len(traj) > 1:
                for i in range(1, len(traj)):
                    cv2.line(frame, traj[i-1], traj[i], color, 2)
                # Draw current position dot
                cv2.circle(frame, traj[-1], 4, color, -1)
            
            # Draw ID label
            label_y = y1 - 10 if y1 > 30 else y2 + 20
            
            # Bee ID
            label = f"ID:{detection.bee_id}"
            cv2.putText(frame, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # ArUco code (if available)
            if detection.aruco_code:
                aruco_label = f"ArUco:{detection.aruco_code}"
                cv2.putText(frame, aruco_label, (x1, label_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _draw_chamber_info(self, frame: np.ndarray, chamber_data: Dict[int, ChamberFrameData]):
        """Draw chamber information (hive pixels, etc.)"""
        # Draw semi-transparent panel
        panel_height = 30 + len(chamber_data) * 25
        panel_width = 250
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 70
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "Hive Pixels per Chamber", 
                   (panel_x + 10, panel_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Chamber data
        y_offset = 45
        for chamber_id in sorted(chamber_data.keys()):
            data = chamber_data[chamber_id]
            text = f"Chamber {chamber_id}: {data.hive_pixels} px"
            cv2.putText(frame, text, (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_offset += 25
    
    def _draw_status_bar(self, frame: np.ndarray, frame_number: int, num_bees: int):
        """Draw status bar with frame info"""
        # Semi-transparent bar at top
        bar_height = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Frame number
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Bee count
        cv2.putText(frame, f"Bees Detected: {num_bees}", (250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
