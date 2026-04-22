"""
Visualization generator for batch video inference - outputs annotated frames as images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict

from core.batch_video_processor import BeeDetectionData, ChamberFrameData


class VisualizationGenerator:
    """Generate annotated frames as individual images with tracking trails and spatial info"""
    
    def __init__(self, video_path: Path, output_folder: Path, video_id: str,
                 bee_detections: List[BeeDetectionData],
                 chamber_frame_data: List[ChamberFrameData],
                 chambers_by_frame: Dict[int, Dict],
                 hive_masks_by_frame: Dict[int, Dict[int, Optional[np.ndarray]]],
                 bee_masks_by_frame: Dict[int, Dict[int, Optional[np.ndarray]]] = None):
        """
        Args:
            video_path: Path to input video
            output_folder: Base output folder for visualizations
            video_id: Video identifier (used for subfolder name)
            bee_detections: List of bee detections
            chamber_frame_data: List of chamber frame data
            chambers_by_frame: Chamber information per frame {frame_number -> {chamber_id -> chamber_info}}
            hive_masks_by_frame: Hive masks per frame per chamber
            bee_masks_by_frame: Bee masks per frame per bee_id (for segmentation visualization)
        """
        self.video_path = video_path
        self.video_id = video_id
        self.output_folder = output_folder / video_id
        self.bee_detections = bee_detections
        self.chamber_frame_data = chamber_frame_data
        self.chambers_by_frame = chambers_by_frame
        self.hive_masks_by_frame = hive_masks_by_frame
        self.bee_masks_by_frame = bee_masks_by_frame if bee_masks_by_frame is not None else {}
        
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
        
        # Track which bee IDs have been seen to color new vs existing
        self.seen_bee_ids: Set[int] = set()
        self.new_ids_in_frame: Set[int] = set()
    
    def generate(self) -> bool:
        """
        Generate annotated frame images and summary visualizations
        
        Returns:
            True if successful, False otherwise
        """
        # Create output folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        print(f"[VizGen] Created output folder: {self.output_folder}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[VizGen] ERROR: Failed to open video: {self.video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[VizGen] Video opened successfully. Total frames: {total_frames}")
        
        frame_number = 0
        first_frame = None
        frames_written = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Save first frame for summary images
            if frame_number == 1:
                first_frame = frame.copy()
            
            # Annotate this frame
            try:
                annotated = self._annotate_frame(frame, frame_number)
            except Exception as e:
                print(f"[VizGen] ERROR annotating frame {frame_number}: {str(e)}")
                import traceback
                traceback.print_exc()
                annotated = frame.copy()  # Use original frame if annotation fails
            
            # Save frame as image
            frame_filename = self.output_folder / f"frame_{frame_number:06d}.png"
            success = cv2.imwrite(str(frame_filename), annotated)
            if success:
                frames_written += 1
            else:
                print(f"[VizGen] WARNING: Failed to write frame {frame_number}")
            
            # Log progress every 100 frames
            if frame_number % 100 == 0:
                print(f"[VizGen] Progress: {frame_number}/{total_frames} frames processed")
        
        cap.release()
        
        print(f"[VizGen] Finished processing. {frames_written} frames written to {self.output_folder}")
        
        # Generate summary images
        if first_frame is not None:
            try:
                self._generate_summary_images(first_frame)
                print(f"[VizGen] Summary images generated")
            except Exception as e:
                print(f"[VizGen] WARNING: Failed to generate summary images: {str(e)}")
        
        return frames_written > 0  # Return True only if we wrote at least one frame
    
    def _annotate_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Annotate a single frame"""
        annotated = frame.copy()
        
        # Reset new IDs for this frame
        self.new_ids_in_frame = set()
        
        # Get detections for this frame
        detections = self.detections_by_frame.get(frame_number, [])
        
        # Identify new IDs
        for detection in detections:
            if detection.bee_id not in self.seen_bee_ids:
                self.new_ids_in_frame.add(detection.bee_id)
                self.seen_bee_ids.add(detection.bee_id)
        
        # 1. Draw chamber boundaries (if chambers detected)
        chambers = self.chambers_by_frame.get(frame_number, {})
        self._draw_chambers(annotated, chambers)
        
        # 2. Draw hive masks (lightly overlaid)
        hive_masks = self.hive_masks_by_frame.get(frame_number, {})
        self._draw_hive_masks(annotated, hive_masks)
        
        # 3. Draw bee detections with IDs, ArUco codes, and tracking trails
        self._draw_bee_detections(annotated, detections, frame_number)
        
        # 4. Draw chamber info panel
        chamber_data = self.chamber_data_by_frame.get(frame_number, {})
        self._draw_chamber_info(annotated, chamber_data)
        
        # 5. Draw status bar
        self._draw_status_bar(annotated, frame_number, len(detections))
        
        return annotated
    
    def _draw_chambers(self, frame: np.ndarray, chambers: Dict[int, Dict]):
        """Draw chamber boundaries from YOLO detections"""
        for chamber_id, chamber_info in chambers.items():
            bbox = chamber_info.get('bbox')
            centroid = chamber_info.get('centroid')
            
            if bbox is not None:
                # Draw bounding box
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                
                # Draw chamber ID label
                if centroid is not None:
                    cx, cy = int(centroid[0]), int(centroid[1])
                    label = f"Chamber {chamber_id}"
                    
                    # Background for text
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (cx - text_w//2 - 5, cy - text_h - 10), 
                                (cx + text_w//2 + 5, cy + 5), (0, 0, 0), -1)
                    
                    cv2.putText(frame, label, (cx - text_w//2, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def _draw_hive_masks(self, frame: np.ndarray, hive_masks: Dict[int, Optional[np.ndarray]]):
        """Draw hive masks as semi-transparent overlay"""
        for chamber_id, mask in hive_masks.items():
            if mask is not None and mask.shape[:2] == frame.shape[:2]:
                # Create colored overlay (yellow for hive)
                overlay = frame.copy()
                overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([0, 200, 200]) * 0.3
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    def _draw_bee_detections(self, frame: np.ndarray, detections: List[BeeDetectionData],
                            frame_number: int):
        """Draw bee bounding boxes/masks, IDs, tracking trails"""
        # Define colors: new IDs in bright green, existing in other colors by chamber
        new_id_color = (0, 255, 0)  # Bright green for new IDs
        chamber_colors = [(255, 100, 100), (100, 100, 255), (255, 255, 100), 
                         (255, 100, 255), (100, 255, 255), (200, 150, 100)]
        
        # Get bee masks for this frame (if available)
        bee_masks = self.bee_masks_by_frame.get(frame_number, {})
        
        for detection in detections:
            # Bounding box
            x1 = int(detection.bbox_x)
            y1 = int(detection.bbox_y)
            x2 = int(detection.bbox_x + detection.bbox_width)
            y2 = int(detection.bbox_y + detection.bbox_height)
            
            # Choose color: new IDs get bright green, existing get chamber color
            if detection.bee_id in self.new_ids_in_frame:
                color = new_id_color
                thickness = 4  # Thicker for new IDs
            else:
                color = chamber_colors[detection.chamber_id % len(chamber_colors)]
                thickness = 2
            
            # Draw segmentation mask if available
            mask = bee_masks.get(detection.bee_id)
            if mask is not None and mask.shape[:2] == frame.shape[:2]:
                # Create colored overlay for this instance
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask > 0] = color
                # Blend with frame
                alpha = 0.4
                frame[mask > 0] = cv2.addWeighted(frame[mask > 0], 1 - alpha, 
                                                  colored_mask[mask > 0], alpha, 0)
                # Draw mask contour
                contours, _ = cv2.findContours((mask > 0).astype(np.uint8), 
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, thickness)
            else:
                # Draw bbox if no mask available
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Update trajectory
            centroid = (int(detection.centroid_x), int(detection.centroid_y))
            self.bee_trajectories[detection.bee_id].append(centroid)
            
            # Limit trajectory length
            if len(self.bee_trajectories[detection.bee_id]) > self.trajectory_max_length:
                self.bee_trajectories[detection.bee_id].pop(0)
            
            # Draw trajectory trail (line from previous position)
            traj = self.bee_trajectories[detection.bee_id]
            if len(traj) > 1:
                # Draw line from previous position
                prev_pos = traj[-2]
                cv2.arrowedLine(frame, prev_pos, centroid, color, 2, tipLength=0.3)
                
                # Draw full trail with fading
                for i in range(1, len(traj) - 1):
                    cv2.circle(frame, traj[i], 2, color, -1)
            
            # Draw current position dot
            cv2.circle(frame, centroid, 5, color, -1)
            
            # Draw ID label with background
            label_y = y1 - 10 if y1 > 30 else y2 + 20
            
            # Bee ID
            if detection.bee_id in self.new_ids_in_frame:
                label = f"ID:{detection.bee_id} (NEW)"
            else:
                label = f"ID:{detection.bee_id}"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1 - 2, label_y - text_h - 2), 
                         (x1 + text_w + 2, label_y + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # ArUco code (if available)
            if detection.aruco_code:
                aruco_label = f"ArUco:{detection.aruco_code}"
                aruco_y = label_y + 20 if y1 > 30 else label_y + 20
                (text_w2, text_h2), _ = cv2.getTextSize(aruco_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1 - 2, aruco_y - text_h2 - 2), 
                             (x1 + text_w2 + 2, aruco_y + 2), (0, 0, 0), -1)
                cv2.putText(frame, aruco_label, (x1, aruco_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _draw_chamber_info(self, frame: np.ndarray, chamber_data: Dict[int, ChamberFrameData]):
        """Draw chamber information panel (hive pixels, etc.)"""
        if not chamber_data:
            return
        
        # Calculate panel size
        panel_height = 40 + len(chamber_data) * 25
        panel_width = 280
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 70
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "Hive Pixels per Chamber", 
                   (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        # Chamber data
        y_offset = 50
        for chamber_id in sorted(chamber_data.keys()):
            data = chamber_data[chamber_id]
            text = f"Chamber {chamber_id}: {data.hive_pixels} px"
            cv2.putText(frame, text, (panel_x + 15, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
    
    def _draw_status_bar(self, frame: np.ndarray, frame_number: int, num_bees: int):
        """Draw status bar with frame info"""
        # Semi-transparent bar at top
        bar_height = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Frame number
        cv2.putText(frame, f"Frame: {frame_number}", (10, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Bee count
        cv2.putText(frame, f"Bees: {num_bees}", (220, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Unique bee count
        cv2.putText(frame, f"Unique IDs: {len(self.seen_bee_ids)}", (400, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # New IDs indicator
        if self.new_ids_in_frame:
            cv2.putText(frame, f"NEW: {len(self.new_ids_in_frame)}", (650, 32), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _generate_summary_images(self, first_frame: np.ndarray):
        """Generate summary visualization images"""
        # 1. Chamber detection summary (first frame)
        if 1 in self.chambers_by_frame:
            chamber_summary = first_frame.copy()
            chambers = self.chambers_by_frame[1]
            
            # Draw all chambers with labels
            for chamber_id, chamber_info in chambers.items():
                bbox = chamber_info.get('bbox')
                mask = chamber_info.get('mask')
                centroid = chamber_info.get('centroid')
                
                # Draw mask overlay if available
                if mask is not None and mask.shape[:2] == chamber_summary.shape[:2]:
                    overlay = chamber_summary.copy()
                    color_overlay = np.zeros_like(chamber_summary)
                    # Use different colors for different chambers
                    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), 
                             (255, 255, 100), (255, 100, 255), (100, 255, 255)]
                    color = colors[chamber_id % len(colors)]
                    color_overlay[mask > 0] = color
                    cv2.addWeighted(chamber_summary, 0.7, color_overlay, 0.3, 0, chamber_summary)
                
                # Draw bbox
                if bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(chamber_summary, (x1, y1), (x2, y2), (0, 255, 255), 4)
                
                # Draw label
                if centroid is not None:
                    cx, cy = int(centroid[0]), int(centroid[1])
                    label = f"Chamber {chamber_id}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    cv2.rectangle(chamber_summary, (cx - text_w//2 - 10, cy - text_h - 15), 
                                (cx + text_w//2 + 10, cy + 10), (0, 0, 0), -1)
                    cv2.putText(chamber_summary, label, (cx - text_w//2, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Add title
            title = f"Chamber Detection Summary - {self.video_id}"
            cv2.putText(chamber_summary, title, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Save chamber summary
            chamber_path = self.output_folder / "chamber_detection_summary.png"
            cv2.imwrite(str(chamber_path), chamber_summary)
        
        # 2. Hive detection summary (first frame)
        if 1 in self.hive_masks_by_frame:
            hive_summary = first_frame.copy()
            hive_masks = self.hive_masks_by_frame[1]
            
            # Draw all hive masks
            for chamber_id, mask in hive_masks.items():
                if mask is not None and mask.shape[:2] == hive_summary.shape[:2]:
                    overlay = hive_summary.copy()
                    # Yellow overlay for hive
                    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 200, 200]) * 0.5
                    cv2.addWeighted(overlay, 0.6, hive_summary, 0.4, 0, hive_summary)
                    
                    # Draw hive contour
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(hive_summary, contours, -1, (0, 255, 255), 3)
            
            # Add title
            title = f"Hive Detection Summary - {self.video_id}"
            cv2.putText(hive_summary, title, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Save hive summary
            hive_path = self.output_folder / "hive_detection_summary.png"
            cv2.imwrite(str(hive_path), hive_summary)
