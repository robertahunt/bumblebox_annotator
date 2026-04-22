"""
Worker thread for batch video inference with tracking and ArUco detection
"""

import cv2
import gc
import json
import numpy as np
import random
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

from core.batch_video_processor import BatchVideoProcessor
from core.video_inference_exporter import VideoInferenceExporter
from core.visualization_generator import VisualizationGenerator


class BatchVideoInferenceWorker(QThread):
    """Worker thread for batch video inference with tracking"""
    
    # Signals
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)  # current, total
    log_message = pyqtSignal(str)
    inference_complete = pyqtSignal()
    inference_failed = pyqtSignal(str)
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.should_stop = False
        
        # Data storage across all videos
        self.all_bee_detections = []
        self.all_chamber_frame_data = []
        # Use composite keys (video_id, bee_id) to avoid collisions across videos
        self.all_bee_trajectories = {}  # {(video_id, bee_id): BeeTrajectory}
        
        # Accumulated masks for averaging
        # Format: {(video_id, chamber_id): {'accumulated_mask': np.ndarray, 'frame_count': int, 'shape': tuple}}
        self.accumulated_hive_masks = {}
        # Format: {(video_id, chamber_id): {'accumulated_mask': np.ndarray, 'frame_count': int, 'shape': tuple}}
        self.accumulated_chamber_masks = {}
    
    def stop(self):
        """Request worker to stop"""
        self.should_stop = True
    
    def run(self):
        """Main execution"""
        try:
            self.log_message.emit("=== Batch Video Inference with Tracking ===")
            self.log_message.emit(f"Model type: {self.config.get('bee_model_type', 'bbox')}")
            self.log_message.emit(f"Distance method: {self.config.get('distance_method', 'contour')}")
            self.log_message.emit(f"Tracking algorithm: {self.config['tracking_config']['algorithm']}")
            self.log_message.emit(f"ArUco detection: {'Enabled' if self.config['enable_aruco'] else 'Disabled'}")
            self.log_message.emit(f"Output folder: {self.config['output_folder']}")
            self.log_message.emit("")
            
            # Discover video files
            self.status_updated.emit("Discovering videos...")
            video_files = self._discover_videos()
            
            if not video_files:
                self.log_message.emit("❌ No video files found!")
                self.log_message.emit(f"  Looking in: {self.config['video_source']}")
                self.log_message.emit(f"  Folder mode: {self.config['folder_mode']}")
                self.inference_failed.emit("No video files found")
                return
            
            self.log_message.emit(f"Found {len(video_files)} video(s) to process (randomized order)")
            for i, vf in enumerate(video_files[:5], 1):  # Log first 5 videos
                self.log_message.emit(f"  {i}. {vf.name}")
            if len(video_files) > 5:
                self.log_message.emit(f"  ... and {len(video_files) - 5} more")
            self.log_message.emit(f"  CSV exports will occur after video 1 and every 10 videos thereafter")
            self.log_message.emit("")
            
            # Load models
            self.status_updated.emit("Loading models...")
            self.log_message.emit("Loading detection models...")
            
            bee_model = YOLO(self.config['bee_model_path'])
            self.log_message.emit(f"✓ Loaded bee model: {Path(self.config['bee_model_path']).name}")
            
            hive_model = YOLO(self.config['hive_model_path'])
            self.log_message.emit(f"✓ Loaded hive model: {Path(self.config['hive_model_path']).name}")
            
            # Chamber model is optional
            chamber_model = None
            if self.config.get('chamber_model_path'):
                chamber_model = YOLO(self.config['chamber_model_path'])
                self.log_message.emit(f"✓ Loaded chamber model: {Path(self.config['chamber_model_path']).name}")
            else:
                self.log_message.emit("  (No chamber model - treating video as single chamber)")
            
            # Set models to eval mode to disable gradient tracking (memory optimization)
            if hasattr(bee_model, 'model') and hasattr(bee_model.model, 'eval'):
                bee_model.model.eval()
            if hasattr(hive_model, 'model') and hasattr(hive_model.model, 'eval'):
                hive_model.model.eval()
            if chamber_model and hasattr(chamber_model, 'model') and hasattr(chamber_model.model, 'eval'):
                chamber_model.model.eval()
            
            self.log_message.emit("")
            
            # Initialize tracking algorithm
            tracker = self._initialize_tracker()
            
            # Log ArUco status
            if self.config['enable_aruco']:
                self.log_message.emit("✓ ArUco detection enabled (for bee ID tracking)")
            
            # Process each video
            total_videos = len(video_files)
            for video_idx, video_path in enumerate(video_files, 1):
                if self.should_stop:
                    self.log_message.emit("\n⚠️ Processing stopped by user")
                    return
                
                self.status_updated.emit(f"Processing video {video_idx}/{total_videos}: {video_path.name}")
                self.log_message.emit(f"=== Video {video_idx}/{total_videos}: {video_path.name} ===")
                
                # Process this video
                self._process_video(
                    video_path, 
                    bee_model, 
                    hive_model,
                    chamber_model,
                    tracker,
                    video_idx
                )
                
                # Update progress after processing is complete
                self.progress_updated.emit(video_idx, total_videos)
                
                # Periodic CSV export: after 1st video and every 10th video
                if video_idx == 1 or video_idx % 10 == 0:
                    self.status_updated.emit(f"Exporting intermediate results ({video_idx}/{total_videos})...")
                    self.log_message.emit(f"\n--- Intermediate Export ({video_idx}/{total_videos} videos processed) ---")
                    self._export_csvs(intermediate=True)
                    
                    # Aggressive GPU memory cleanup at checkpoint
                    self.log_message.emit(f"  Running aggressive GPU memory cleanup at checkpoint...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for all operations to complete
                        mem_allocated = torch.cuda.memory_allocated() / 1e9
                        mem_reserved = torch.cuda.memory_reserved() / 1e9
                        self.log_message.emit(f"    GPU memory after checkpoint cleanup: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
                    
                    self.log_message.emit("")
                
                if self.should_stop:
                    break
            
            if self.should_stop:
                return
            
            # Final export of all data to CSVs
            self.status_updated.emit("Exporting final results...")
            self.log_message.emit("\n=== Final Export ===")
            self._export_csvs(intermediate=False)
            
            self.log_message.emit("\n✓ Batch inference complete!")
            self.log_message.emit(f"Results saved to: {self.config['output_folder']}")
            self.inference_complete.emit()
            
        except Exception as e:
            import traceback
            error_msg = f"Inference failed: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(f"\n❌ {error_msg}")
            self.inference_failed.emit(error_msg)
    
    def _discover_videos(self) -> List[Path]:
        """Discover video files from folder or file list (randomized order)"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mjpeg', '.mjpg'}
        
        if self.config['folder_mode']:
            # Folder mode - discover recursively
            folder = Path(self.config['video_source'])
            video_files = []
            for ext in video_extensions:
                video_files.extend(folder.rglob(f"*{ext}"))
            # Randomize order so we get diverse samples as processing progresses
            random.shuffle(video_files)
            return video_files
        else:
            # File mode - use provided list (randomize as well)
            video_files = [Path(f) for f in self.config['video_source']]
            random.shuffle(video_files)
            return video_files
    
    def _initialize_tracker(self):
        """Initialize tracking algorithm based on config"""
        tracking_config = self.config['tracking_config']
        algo = tracking_config['algorithm']
        
        if algo == 'bytetrack':
            from core.instance_tracker import InstanceTracker
            
            tracker_config = {
                'high_conf_threshold': tracking_config['high_conf_threshold'],
                'high_iou_threshold': tracking_config['high_iou_threshold'],
                'low_iou_threshold': tracking_config['low_iou_threshold'],
                'max_frames_lost': tracking_config['max_frames_lost'],
                'use_mask_iou': tracking_config['use_mask_iou']
            }
            tracker = InstanceTracker(config=tracker_config)
            self.log_message.emit(f"✓ Initialized ByteTrack tracker")
            
        elif algo == 'simple_iou':
            # Import from tracking_validation_worker where these are defined
            from gui.tracking_validation_worker import SimpleIoUTracker
            
            tracker = SimpleIoUTracker(
                iou_threshold=tracking_config['iou_threshold'],
                use_mask_iou=tracking_config['use_mask_iou']
            )
            self.log_message.emit(f"✓ Initialized SimpleIoU tracker")
            
        elif algo == 'centroid':
            from gui.tracking_validation_worker import CentroidTracker
            
            tracker = CentroidTracker(
                max_distance=tracking_config['max_distance'],
                max_frames_missing=tracking_config['max_frames_missing']
            )
            self.log_message.emit(f"✓ Initialized Centroid tracker")
        
        else:
            raise ValueError(f"Unknown tracking algorithm: {algo}")
        
        return tracker
    
    def _process_video(self, video_path, bee_model, hive_model, chamber_model, tracker, video_idx):
        """Process a single video file"""
        video_id = video_path.stem  # Use filename without extension as video_id
        self.log_message.emit(f"  Processing: {video_path.name}")
        
        # Check if visualization is enabled to decide whether to store masks
        visualization_enabled = self.config.get('save_visualizations', False)
        
        # Log memory optimization setting
        if not visualization_enabled:
            self.log_message.emit(f"  Memory optimization: Masks will NOT be stored (visualization disabled)")
        else:
            self.log_message.emit(f"  Memory usage: Storing masks for visualization")
        
        # Create video processor
        processor = BatchVideoProcessor(
            video_path=video_path,
            video_id=video_id,
            bee_model=bee_model,
            hive_model=hive_model,
            chamber_model=chamber_model,
            tracker=tracker,
            confidence_threshold=self.config['confidence_threshold'],
            nms_iou_threshold=self.config['nms_iou_threshold'],
            enable_aruco=self.config['enable_aruco'],
            output_folder=Path(self.config['output_folder']),
            distance_method=self.config.get('distance_method', 'contour'),
            bee_model_type=self.config.get('bee_model_type', 'bbox'),
            store_masks=visualization_enabled  # Only store masks if visualization is enabled
        )
        
        # Process video
        success = processor.process()
        
        if not success:
            self.log_message.emit(f"  ❌ Failed to process video: {video_path}")
            self.log_message.emit(f"     Check that video file is valid and models are compatible")
            return
        
        # Log processing statistics
        num_detections = len(processor.get_bee_detections())
        num_trajectories = len(processor.get_bee_trajectories())
        num_chamber_data = len(processor.get_chamber_frame_data())
        
        self.log_message.emit(f"  ✓ Processed successfully:")
        self.log_message.emit(f"    - {num_detections} bee detections")
        self.log_message.emit(f"    - {num_trajectories} unique tracked bees")
        self.log_message.emit(f"    - {num_chamber_data} chamber frame records")
        
        # Collect data
        self.all_bee_detections.extend(processor.get_bee_detections())
        self.all_chamber_frame_data.extend(processor.get_chamber_frame_data())
        
        # Collect trajectories with composite keys (video_id, bee_id) to avoid collisions
        for bee_id, trajectory in processor.get_bee_trajectories().items():
            composite_key = (video_id, bee_id)
            self.all_bee_trajectories[composite_key] = trajectory
        
        # Accumulate masks for averaging
        self._accumulate_masks(video_id, processor.get_hive_masks_by_frame(), processor.get_chambers_by_frame())
        
        # Reset tracker state to prevent memory buildup across videos
        if hasattr(tracker, 'reset'):
            tracker.reset()
        
        # Clear GPU cache and run garbage collection to free memory
        self.log_message.emit(f"  Cleaning up GPU memory...")
        if torch.cuda.is_available():
            # Log GPU memory before cleanup
            mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            mem_reserved = torch.cuda.memory_reserved() / 1e9  # GB
            self.log_message.emit(f"    Before cleanup: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
            
            torch.cuda.empty_cache()
            
            # Log GPU memory after cleanup
            mem_allocated_after = torch.cuda.memory_allocated() / 1e9  # GB
            mem_reserved_after = torch.cuda.memory_reserved() / 1e9  # GB
            freed = mem_reserved - mem_reserved_after
            self.log_message.emit(f"    After cleanup: {mem_allocated_after:.2f} GB allocated, {mem_reserved_after:.2f} GB reserved (freed {freed:.2f} GB)")
        
        gc.collect()
        
        # Explicitly delete processor to free memory (especially mask storage)
        del processor
        
        # Force another GPU cache clear after deletion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate visualization if requested
        visualization_enabled = self.config.get('save_visualizations', False)
        self.log_message.emit(f"  Visualization setting: {visualization_enabled}")
        
        if visualization_enabled:
            self.log_message.emit(f"  Generating visualizations...")
            
            # Create visualizations subfolder
            viz_base_folder = Path(self.config['output_folder']) / 'visualizations'
            
            try:
                viz_gen = VisualizationGenerator(
                    video_path=video_path,
                    output_folder=viz_base_folder,
                    video_id=video_id,
                    bee_detections=processor.get_bee_detections(),
                    chamber_frame_data=processor.get_chamber_frame_data(),
                    chambers_by_frame=processor.get_chambers_by_frame(),
                    hive_masks_by_frame=processor.get_hive_masks_by_frame(),
                    bee_masks_by_frame=processor.get_bee_masks_by_frame()
                )
                
                success = viz_gen.generate()
                if success:
                    num_frames = len(processor.get_bee_detections())
                    self.log_message.emit(f"  ✓ Saved annotated frames to: visualizations/{video_id}/")
                    self.log_message.emit(f"    Output folder: {viz_base_folder / video_id}")
                else:
                    self.log_message.emit(f"  ⚠️ Visualization generation returned False")
            except Exception as e:
                self.log_message.emit(f"  ⚠️ Visualization error: {str(e)}")
                import traceback
                self.log_message.emit(traceback.format_exc())
        else:
            self.log_message.emit(f"  Skipping visualizations (checkbox not enabled)")
    
    def _accumulate_masks(self, video_id: str, hive_masks_by_frame: Dict, chambers_by_frame: Dict):
        """
        Accumulate hive and chamber masks across frames for averaging
        
        Args:
            video_id: Video identifier
            hive_masks_by_frame: Dict[frame_number -> Dict[chamber_id -> mask]]
            chambers_by_frame: Dict[frame_number -> Dict[chamber_id -> chamber_info]]
        
        Note: If visualization is disabled, these dictionaries will be empty (store_masks=False)
        and this function will do nothing, which is the intended behavior for memory efficiency.
        """
        # Skip if no data (visualization disabled)
        if not hive_masks_by_frame and not chambers_by_frame:
            return
        
        # Accumulate hive masks
        for frame_number, hive_masks in hive_masks_by_frame.items():
            for chamber_id, mask in hive_masks.items():
                if mask is None:
                    continue
                
                key = (video_id, chamber_id)
                
                if key not in self.accumulated_hive_masks:
                    # Initialize with zeros
                    self.accumulated_hive_masks[key] = {
                        'accumulated_mask': np.zeros_like(mask, dtype=np.float32),
                        'frame_count': 0,
                        'shape': mask.shape
                    }
                
                # Add this frame's mask
                self.accumulated_hive_masks[key]['accumulated_mask'] += mask.astype(np.float32)
                self.accumulated_hive_masks[key]['frame_count'] += 1
        
        # Accumulate chamber masks
        for frame_number, chambers in chambers_by_frame.items():
            for chamber_id, chamber_info in chambers.items():
                mask = chamber_info.get('mask')
                centroid = chamber_info.get('centroid')
                
                if mask is None:
                    continue
                
                key = (video_id, chamber_id)
                
                if key not in self.accumulated_chamber_masks:
                    # Initialize with zeros
                    self.accumulated_chamber_masks[key] = {
                        'accumulated_mask': np.zeros_like(mask, dtype=np.float32),
                        'frame_count': 0,
                        'shape': mask.shape,
                        'accumulated_centroid': np.array([0.0, 0.0], dtype=np.float32)
                    }
                
                # Add this frame's mask
                self.accumulated_chamber_masks[key]['accumulated_mask'] += mask.astype(np.float32)
                self.accumulated_chamber_masks[key]['frame_count'] += 1
                
                # Accumulate centroid
                if centroid is not None:
                    self.accumulated_chamber_masks[key]['accumulated_centroid'] += np.array(centroid, dtype=np.float32)
    
    def _export_csvs(self, intermediate: bool = False):
        """Export all CSV files
        
        Args:
            intermediate: If True, this is an intermediate export during processing
        """
        output_folder = Path(self.config['output_folder'])
        
        if intermediate:
            self.log_message.emit(f"  Exporting intermediate CSVs to: {output_folder}")
        else:
            self.log_message.emit(f"  Exporting final CSVs to: {output_folder}")
        
        # Check if we have any data to export
        if not self.all_bee_detections:
            self.log_message.emit(f"  ⚠️ No bee detections to export!")
            return
        
        # Create exporter
        exporter = VideoInferenceExporter(output_folder)
        
        # Export all CSVs
        try:
            csv_paths = exporter.export_all(
                self.all_bee_detections,
                self.all_bee_trajectories,
                self.all_chamber_frame_data,
                self.accumulated_hive_masks,
                self.accumulated_chamber_masks
            )
            
            # Log results
            if not intermediate:
                for csv_name, csv_path in csv_paths.items():
                    file_size = csv_path.stat().st_size if csv_path.exists() else 0
                    self.log_message.emit(f"  ✓ {csv_name}.csv created ({file_size:,} bytes)")
        except Exception as e:
            self.log_message.emit(f"  ❌ CSV export failed: {str(e)}")
            import traceback
            self.log_message.emit(traceback.format_exc())
        
        # Summary statistics
        if not intermediate:
            self.log_message.emit(f"\n=== Summary ===")
        self.log_message.emit(f"Total bee detections: {len(self.all_bee_detections)}")
        self.log_message.emit(f"Unique bees tracked: {len(self.all_bee_trajectories)}")
        self.log_message.emit(f"Chamber frame records: {len(self.all_chamber_frame_data)}")
